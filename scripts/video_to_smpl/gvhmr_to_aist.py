"""gvhmr_to_aist.py — convert a video->SMPL model's output into the AIST++-style
.pkl that this repo's motion pipeline already understands.

Supported sources
-----------------
  --source gvhmr : GVHMR `hmr4d_results.pt` (SMPL-X params, axis-angle,
                   gravity-aligned Y-up world frame).
                   pred["smpl_params_global"] = {
                       global_orient (T,3), body_pose (T,63),
                       betas (T,10), transl (T,3) }
  --source wham  : WHAM output .pkl. Per-track dict with SMPL params.
                   Uses pose_world (T,72) + trans_world (T,3) when present,
                   else pose (T,72) + trans (T,3).

Output (drop-in for scripts/export_motion_json.py -> load_aist_pkl)
-------------------------------------------------------------------
  { 'smpl_poses'   : (T, 72) float64  axis-angle, SMPL 24-joint order
    'smpl_trans'   : (T, 3)  float64  meters (root translation)
    'smpl_scaling' : 1.0     -> keeps trans in meters downstream
    'fps'          : float }

Why this is enough: the repo already ingests exactly this schema for AIST++
clips, retargets to the VRM rig, fixes quaternion signs, and gates on
orientation. So a correct SMPL extraction is the ONLY new thing needed.

SMPL-X -> SMPL note: SMPL-X body_pose is 63 = joints 1..21 (hips..wrists),
which are identical to SMPL's first 21 body joints. SMPL adds joints 22,23
(L_Hand, R_Hand) which SMPL-X replaces with articulated fingers. We set those
2 joints to identity (zeros) because the target VRM rig uses single Hand bones.

Run this ON THE POD (where torch is available to load the .pt); the resulting
.pkl is tiny (a few hundred KB) and is what you download to run build_clip.py
locally.

Usage
-----
  # GVHMR (on the RunPod pod, torch present)
  python gvhmr_to_aist.py --source gvhmr \
      --in outputs/demo/<video_name>/hmr4d_results.pt \
      --out my_clip.pkl --fps 30

  # WHAM
  python gvhmr_to_aist.py --source wham \
      --in output/demo/<video>/wham_output.pkl \
      --out my_clip.pkl --fps 30
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np


# ────────────────────────────────────────────────────────────────────────
#  Root-frame fixes
# ────────────────────────────────────────────────────────────────────────
# GVHMR global frame is gravity-aligned Y-up. AIST++ (what the retarget was
# tuned on) expects the same Y-up convention, so IDENTITY is the default and
# usually correct. If your verified clip comes out lying-down / rotated, pick
# one of these presets with --root-fix and re-verify (cheap: just re-run
# build_clip). Each is applied to the pelvis (global_orient) only.
ROOT_FIX = {
    'none':   np.eye(3),
    'x-90':   np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),   # Y-up -> Z-up
    'x+90':   np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
    'y180':   np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float),  # face flip
}


def _aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (...,3) -> rotation matrix (...,3,3). Rodrigues, vectorized."""
    aa = np.asarray(aa, float)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)          # (...,1)
    small = theta < 1e-8
    axis = np.where(small, 0.0, aa / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta)[..., 0]
    s = np.sin(theta)[..., 0]
    C = 1 - c
    R = np.empty(aa.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def _matrix_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (...,3,3) -> axis-angle (...,3)."""
    R = np.asarray(R, float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)                                       # (...)
    vx = R[..., 2, 1] - R[..., 1, 2]
    vy = R[..., 0, 2] - R[..., 2, 0]
    vz = R[..., 1, 0] - R[..., 0, 1]
    v = np.stack([vx, vy, vz], axis=-1)                          # (...,3)
    sin = np.linalg.norm(v, axis=-1, keepdims=True)
    small = sin[..., 0] < 1e-8
    axis = np.where(small[..., None], np.array([1.0, 0.0, 0.0]), v / np.where(sin < 1e-8, 1.0, sin))
    return axis * theta[..., None]


def _apply_root_fix(global_orient_aa: np.ndarray, transl: np.ndarray, fix: str):
    """Left-multiply pelvis rotation (and rotate translation) by a fixed basis."""
    Rfix = ROOT_FIX[fix]
    if fix == 'none':
        return global_orient_aa, transl
    Rg = _aa_to_matrix(global_orient_aa)             # (T,3,3)
    Rg = np.einsum('ij,tjk->tik', Rfix, Rg)
    transl = np.einsum('ij,tj->ti', Rfix, transl)
    return _matrix_to_aa(Rg), transl


# ────────────────────────────────────────────────────────────────────────
#  Source loaders
# ────────────────────────────────────────────────────────────────────────
def _to_numpy(x):
    """torch.Tensor / list / ndarray -> ndarray, without importing torch here."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_gvhmr(path: str):
    """Load GVHMR hmr4d_results.pt -> (global_orient(T,3), body_pose(T,63), transl(T,3))."""
    import torch  # only needed to unpickle the .pt
    pred = torch.load(path, map_location='cpu')
    if 'smpl_params_global' not in pred:
        raise KeyError(
            "GVHMR result has no 'smpl_params_global'. Keys: "
            f"{list(pred.keys())}")
    p = pred['smpl_params_global']
    go = _to_numpy(p['global_orient']).reshape(-1, 3)
    bp = _to_numpy(p['body_pose']).reshape(len(go), -1)
    if bp.shape[1] != 63:
        raise ValueError(f"expected SMPL-X body_pose (T,63), got {bp.shape}")
    tr = _to_numpy(p['transl']).reshape(-1, 3)
    return go, bp, tr


def load_wham(path: str):
    """Load WHAM output .pkl -> (global_orient(T,3), body_pose(T,69), transl(T,3))."""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    # WHAM saves a dict keyed by track id; take the longest track.
    if isinstance(d, dict) and 'pose' not in d and 'pose_world' not in d:
        track = max(d.values(), key=lambda t: len(_to_numpy(
            t.get('pose_world', t.get('pose')))))
    else:
        track = d
    pose = track.get('pose_world', track.get('pose'))
    trans = track.get('trans_world', track.get('trans'))
    pose = _to_numpy(pose).reshape(-1, 72)          # SMPL full pose (24*3)
    tr = _to_numpy(trans).reshape(-1, 3)
    go = pose[:, :3]
    bp = pose[:, 3:72]                               # 69 = SMPL 23 body joints
    return go, bp, tr


# ────────────────────────────────────────────────────────────────────────
#  Assemble SMPL 72
# ────────────────────────────────────────────────────────────────────────
def build_smpl_poses(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """global_orient(T,3) + body_pose(T,63 or 69) -> smpl_poses (T,72)."""
    T = len(global_orient)
    if body_pose.shape[1] == 63:            # SMPL-X: pad hands (joints 22,23)
        body = np.concatenate([body_pose, np.zeros((T, 6), float)], axis=1)
    elif body_pose.shape[1] == 69:          # already SMPL
        body = body_pose
    else:
        raise ValueError(f"body_pose must be 63 or 69 wide, got {body_pose.shape}")
    return np.concatenate([global_orient, body], axis=1).astype(np.float64)  # (T,72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=['gvhmr', 'wham'])
    ap.add_argument('--in', dest='inp', required=True, help='model output file')
    ap.add_argument('--out', required=True, help='AIST-style .pkl to write')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--root-fix', default='none', choices=list(ROOT_FIX))
    ap.add_argument('--trim-start', type=int, default=0,
                    help='drop N leading frames (settling)')
    ap.add_argument('--trim-end', type=int, default=0)
    args = ap.parse_args()

    if args.source == 'gvhmr':
        go, bp, tr = load_gvhmr(args.inp)
    else:
        go, bp, tr = load_wham(args.inp)

    go, tr = _apply_root_fix(go, tr, args.root_fix)
    smpl_poses = build_smpl_poses(go, bp)               # (T,72)
    smpl_trans = tr.astype(np.float64)                  # meters

    # center XZ so the avatar starts near origin (Y/height preserved)
    smpl_trans = smpl_trans - smpl_trans[0:1] * np.array([1.0, 0.0, 1.0])

    a, b = args.trim_start, (len(smpl_poses) - args.trim_end)
    smpl_poses, smpl_trans = smpl_poses[a:b], smpl_trans[a:b]

    out = {
        'smpl_poses':   smpl_poses,          # (T,72) axis-angle
        'smpl_trans':   smpl_trans,          # (T,3) meters
        'smpl_scaling': 1.0,                 # trans already in meters
        'fps':          float(args.fps),
    }
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)

    finite = np.isfinite(smpl_poses).all() and np.isfinite(smpl_trans).all()
    print(f"[gvhmr_to_aist] wrote {args.out}")
    print(f"  frames={len(smpl_poses)}  fps={args.fps}  finite={finite}")
    print(f"  pose range [{smpl_poses.min():.3f}, {smpl_poses.max():.3f}] rad")
    print(f"  trans Y (height) range [{smpl_trans[:,1].min():.3f}, "
          f"{smpl_trans[:,1].max():.3f}] m")
    if not finite:
        raise SystemExit("NON-FINITE values in output — extraction failed.")


if __name__ == '__main__':
    main()
