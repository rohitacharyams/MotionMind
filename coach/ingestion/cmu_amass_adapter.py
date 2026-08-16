"""cmu_amass_adapter.py — stream CMU AMASS (SMPL+H 52-joint poses) → AIST-format .pkl.

AMASS `*_poses.npz` files contain:
    trans            (T, 3)    body translation in meters
    poses            (T, 156)  axis-angle for 52 joints (22 body + 15 LH + 15 RH)
    betas            (16,)     body shape (we ignore — VRM mesh is fixed)
    mocap_framerate  float     usually 120 Hz for CMU

Our existing retarget pipeline (scripts/export_motion_json.py via
coach/batch_retarget.py) expects the AIST++ convention:
    smpl_poses   (T, 72)  axis-angle for 24 SMPL joints
    smpl_trans   (T, 3)   translation
    smpl_scaling [1.0]
    fps          float

This adapter:
  • streams the tar.bz2 without exploding to disk
  • drops finger joints (keeps 22 body joints; SMPL joints 22, 23 = 0)
  • downsamples to 30 fps (every 4th frame on 120 Hz CMU)
  • filters out very short or very long clips
  • writes one .pkl per AMASS sequence to data/motion_db/amass_cmu/

Usage:
    py -3.12 -m coach.ingestion.cmu_amass_adapter \\
        --tar C:\\Users\\<you>\\Downloads\\CMU.tar.bz2 \\
        --out data/motion_db/amass_cmu \\
        [--limit 500]
"""
from __future__ import annotations

import argparse
import io
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np

from coach.physics_validator import fix_motion_arrays

# Tuning knobs. CMU has some VERY long clips (>30s) and some 1-second
# snippets. Keep a useful range for dance-coach drilling.
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 30.0
TARGET_FPS       = 30.0


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z = (aa / theta).tolist()
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = float(np.arccos(c))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-5:
        # Near pi: robust axis from diagonal.
        xx = max(0.0, (R[0, 0] + 1.0) * 0.5)
        yy = max(0.0, (R[1, 1] + 1.0) * 0.5)
        zz = max(0.0, (R[2, 2] + 1.0) * 0.5)
        axis = np.array([np.sqrt(xx), np.sqrt(yy), np.sqrt(zz)], dtype=np.float64)
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= n
        return axis * theta
    v = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)
    axis = v / (2.0 * np.sin(theta))
    return axis * theta


def _cmu_needs_upright_fix(smpl24: np.ndarray) -> bool:
    # CMU root often arrives near +/-90deg on X, while AIST-like upright
    # roots are near identity-ish around X. We use first-frame root only.
    aa = np.asarray(smpl24[0, 0], dtype=np.float64)
    return abs(float(aa[0])) > 0.9


def _upright_align_cmu(smpl24: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_pose = np.asarray(smpl24, dtype=np.float64).copy()
    out_trans = np.asarray(trans, dtype=np.float64).copy()

    # Rotate global frame +90deg around X to map CMU lying frame to
    # Dance.AI/AIST upright frame.
    a = np.pi * 0.5
    R_fix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(a), -np.sin(a)],
        [0.0, np.sin(a), np.cos(a)],
    ], dtype=np.float64)

    roots = out_pose[:, 0, :]
    for i in range(roots.shape[0]):
        Rr = _axis_angle_to_matrix(roots[i])
        roots[i] = _matrix_to_axis_angle(R_fix @ Rr)
    out_pose[:, 0, :] = roots
    out_trans = (R_fix @ out_trans.T).T
    return out_pose, out_trans


def amass_to_aist_dict(npz_bytes: bytes) -> dict | None:
    """Convert one AMASS npz blob to an AIST-style dict. Returns None
    if the clip is too short / too long / unparseable."""
    try:
        d = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
        poses = np.asarray(d['poses'])     # (T, 156)
        trans = np.asarray(d['trans'])     # (T, 3)
        fps   = float(d['mocap_framerate'])
    except Exception:
        return None
    T = poses.shape[0]
    if T == 0:
        return None
    # Drop finger joints → first 22 body joints, pad joints 22,23 = 0.
    body = poses[:, :22 * 3].reshape(T, 22, 3)
    smpl24 = np.zeros((T, 24, 3), dtype=np.float64)
    smpl24[:, :22, :] = body
    # Downsample to TARGET_FPS by stride sampling. AMASS CMU is 120 Hz,
    # so stride = 4 gives clean 30 fps.
    if fps > TARGET_FPS + 1:
        stride = max(1, int(round(fps / TARGET_FPS)))
        smpl24 = smpl24[::stride]
        trans  = trans[::stride]
        fps    = fps / stride
    dur = smpl24.shape[0] / max(fps, 1.0)
    if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
        return None
    if _cmu_needs_upright_fix(smpl24):
        smpl24, trans = _upright_align_cmu(smpl24, trans)
    fixed_poses, fixed_trans, _ = fix_motion_arrays(
        smpl24,
        trans,
        fps=int(round(fps)),
        floor_percentile=2.0,
        target_floor_z=0.0,
        max_xy_radius=2.5,
        center_xy=True,
    )
    return {
        'smpl_poses':   fixed_poses.reshape(-1, 72).astype(np.float32),
        'smpl_trans':   fixed_trans.astype(np.float32),
        'smpl_scaling': np.array([1.0], dtype=np.float32),
        'fps':          float(fps),
        # provenance — survives pickling, helps debug later
        '_source':      'AMASS-CMU',
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--tar', required=True,
                   help='Path to CMU.tar.bz2')
    p.add_argument('--out', default='data/motion_db/amass_cmu',
                   help='Output directory for .pkl files')
    p.add_argument('--limit', type=int, default=None,
                   help='Stop after N sequences (smoke test).')
    args = p.parse_args()

    tar_path = Path(args.tar).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        print(f'[err] tar not found: {tar_path}')
        return 2

    t0 = time.time()
    n_ok = n_skip = n_short = n_long = n_fail = 0
    print(f'[adapter] streaming {tar_path.name} → {out_dir}')
    with tarfile.open(tar_path, 'r:bz2') as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith('_poses.npz'):
                continue
            # Flatten subject/sequence → single filename so the existing
            # retarget pipeline (which globs *.pkl) picks them all up.
            # e.g. "CMU/38/38_03_poses.npz" → "cmu_38_03.pkl"
            parts = Path(member.name).parts
            try:
                subj, seq = parts[-2], parts[-1].replace('_poses.npz', '')
                out_name = f'cmu_{subj}_{seq}.pkl'.replace('/', '_')
            except Exception:
                continue
            out_path = out_dir / out_name
            if out_path.exists():
                n_skip += 1
                continue
            try:
                ef = tf.extractfile(member)
                if ef is None:
                    n_fail += 1
                    continue
                blob = ef.read()
            except Exception:
                n_fail += 1
                continue
            d = amass_to_aist_dict(blob)
            if d is None:
                # Either too short, too long, or unparseable — count
                # roughly by inspecting blob length so the user gets
                # a visible distribution at the end.
                if len(blob) < 4096:
                    n_short += 1
                else:
                    n_long += 1
                continue
            with open(out_path, 'wb') as fp:
                pickle.dump(d, fp, protocol=4)
            n_ok += 1
            if n_ok % 50 == 0:
                elapsed = time.time() - t0
                print(f'  [{n_ok}] {out_name}  '
                      f'({elapsed:.1f}s, {n_ok/elapsed:.1f}/s)')
            if args.limit and n_ok >= args.limit:
                break

    elapsed = time.time() - t0
    print(f'[adapter] done in {elapsed:.1f}s — '
          f'ok={n_ok} skip={n_skip} short={n_short} '
          f'long_or_bad={n_long} fail={n_fail}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
