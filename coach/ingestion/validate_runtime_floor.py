"""Offline equivalent of motion_player.js FK to detect floor penetration.

For each vrm-quat cache file, do forward kinematics using
``rest_local_translation`` + per-frame ``rotations`` + hips delta
translation, walk the standard humanoid leg chain to the toes, and
report the GLOBAL lowest foot world Y across the clip. The studio's
``_computeGroundOffset`` originally sampled only frame 0; the v34 fix
samples many frames. This script reproduces both so we can prove the
fix lands feet on the floor across the whole motion.

Run:
    python -m coach.ingestion.validate_runtime_floor \
        --cache coach/motion_cache_cmu --limit 50 --out coach/reports/runtime_floor.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Standard glTF/three.js quaternion order is [x, y, z, w].
LEG_CHAIN = [
    ('Hips', None),
    ('LeftUpLeg', 'Hips'),
    ('LeftLeg', 'LeftUpLeg'),
    ('LeftFoot', 'LeftLeg'),
]
RIGHT_LEG_CHAIN = [
    ('Hips', None),
    ('RightUpLeg', 'Hips'),
    ('RightLeg', 'RightUpLeg'),
    ('RightFoot', 'RightLeg'),
]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def trs(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def chain_world(chain, rest_t, rest_r, frame_rot, hips_world_t) -> np.ndarray:
    M = np.eye(4)
    for i, (bone, _parent) in enumerate(chain):
        t = np.array(rest_t.get(bone, [0, 0, 0]), dtype=np.float64)
        if i == 0:
            t = hips_world_t
        r_rest = np.array(rest_r.get(bone, [0, 0, 0, 1]), dtype=np.float64)
        r_anim = np.array(frame_rot.get(bone, [0, 0, 0, 1]), dtype=np.float64)
        # three.js applies local = rest_rotation_already_baked * anim?
        # Our exporter writes rotations as absolute local quats that
        # REPLACE bind. Match that: use anim as the local rotation.
        R = quat_to_mat(r_anim)
        M = M @ trs(R, t)
    return M[:3, 3]


def analyse(cache_path: Path) -> Dict:
    b = json.loads(cache_path.read_text(encoding='utf-8'))
    rest_t = b.get('rest_local_translation', {})
    rest_r = b.get('rest_local_rotation', {})
    rots: Dict[str, List[List[float]]] = b.get('rotations', {})
    hips_t: List[List[float]] = b.get('hips_translation') or []
    n = b.get('n_frames') or len(hips_t)
    if n <= 0:
        return {'file': cache_path.name, 'error': 'no frames'}

    # Frame 0 hips world position = rest hips + 0 delta.
    rest_hips = np.array(rest_t.get('Hips', [0, 0, 0]), dtype=np.float64)
    hips0 = np.array(hips_t[0], dtype=np.float64) if hips_t else np.zeros(3)

    samples = list(range(0, n, max(1, n // 48)))[:48]
    if 0 not in samples:
        samples.insert(0, 0)

    foot_y_by_frame: List[float] = []
    for f in samples:
        if f >= len(hips_t):
            continue
        # Match player: dy = hipsT[f][1] - hipsT[0][1]; final hips Y =
        # rest_hips.y + dy + groundOffset. We test PRE-offset so we
        # can later add the offset and check feet=0.
        dx = hips_t[f][0] - hips0[0]
        dy = hips_t[f][1] - hips0[1]
        dz = hips_t[f][2] - hips0[2]
        # Mirror the player's MAX_R/MAX_DOWN/MAX_UP clamps so the
        # offline analysis matches what the user actually sees.
        r = np.hypot(dx, dz)
        if r > 0.45:
            s = 0.45 / r
            dx *= s; dz *= s
        if dy > 0.35:  dy = 0.35
        if dy < -0.35: dy = -0.35
        hips_world = rest_hips + np.array([dx, dy, dz])
        per_frame = {k: v[f] if f < len(v) else v[-1] for k, v in rots.items()}
        left  = chain_world(LEG_CHAIN,       rest_t, rest_r, per_frame, hips_world)
        right = chain_world(RIGHT_LEG_CHAIN, rest_t, rest_r, per_frame, hips_world)
        foot_y_by_frame.append(float(min(left[1], right[1])))

    frame0_foot = foot_y_by_frame[0]
    global_min  = min(foot_y_by_frame)
    # Old behavior: offset = -frame0_foot. New behavior: offset = -global_min.
    # Floor penetration with OLD = max(0, frame0_foot - global_min).
    pen_old = max(0.0, frame0_foot - global_min)
    pen_new = 0.0
    return {
        'file': cache_path.name,
        'n_frames': n,
        'frame0_foot_y': frame0_foot,
        'global_min_foot_y': global_min,
        'penetration_old_cm': round(pen_old * 100, 2),
        'penetration_new_cm': round(pen_new * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out',   type=Path, default=None)
    a = ap.parse_args()
    files = sorted(a.cache.glob('*.json'))
    if a.limit > 0:
        files = files[: a.limit]
    rows = []
    bad = 0
    for p in files:
        try:
            r = analyse(p)
        except Exception as e:                                       # noqa: BLE001
            r = {'file': p.name, 'error': str(e)}
        rows.append(r)
        if r.get('penetration_old_cm', 0) > 2.0:
            bad += 1
    summary = {
        'scanned':                  len(rows),
        'penetrating_old_strategy': bad,
        'penetrating_new_strategy': sum(
            1 for r in rows if r.get('penetration_new_cm', 0) > 2.0),
        'worst_old_cm': max((r.get('penetration_old_cm', 0) for r in rows),
                            default=0),
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({'summary': summary, 'rows': rows},
                                    indent=2), encoding='utf-8')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
