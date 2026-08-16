"""Fix orientation for already-ingested CMU .pkl clips.

Applies the same CMU->upright transform used in cmu_amass_adapter, then
re-runs deterministic motion fixes.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from coach.ingestion.cmu_amass_adapter import _cmu_needs_upright_fix, _upright_align_cmu
from coach.physics_validator import fix_motion_arrays


def process_file(path: Path, dry_run: bool = False) -> tuple[bool, bool]:
    d = pickle.load(open(path, 'rb'))
    poses = np.asarray(d['smpl_poses'], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(d['smpl_trans'], dtype=np.float64).reshape(-1, 3)
    fps = float(d.get('fps', 30.0))

    needs = _cmu_needs_upright_fix(poses)
    if needs:
        poses, trans = _upright_align_cmu(poses, trans)

    poses, trans, _ = fix_motion_arrays(
        poses,
        trans,
        fps=int(round(fps)),
        floor_percentile=2.0,
        target_floor_z=0.0,
        max_xy_radius=2.5,
        center_xy=True,
    )

    if not dry_run:
        d['smpl_poses'] = poses.reshape(-1, 72).astype(np.float32)
        d['smpl_trans'] = trans.astype(np.float32)
        with open(path, 'wb') as f:
            pickle.dump(d, f, protocol=4)
    return True, needs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=r'C:\dan\data\motion_db\amass_cmu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    p = Path(args.dir)
    files = sorted(p.glob('*.pkl'))
    if args.limit > 0:
        files = files[: args.limit]

    n = 0
    n_rot = 0
    for i, f in enumerate(files, start=1):
        ok, rotated = process_file(f, dry_run=bool(args.dry_run))
        if ok:
            n += 1
        if rotated:
            n_rot += 1
        if i % 100 == 0:
            print(f'[cmu-fix] {i}/{len(files)} done (rotated={n_rot})')

    print(f'[cmu-fix] complete files={n} rotated={n_rot} dry_run={bool(args.dry_run)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
