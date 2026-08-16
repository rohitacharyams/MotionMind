"""batch_retarget.py — convert AIST .pkl → VRM-bone-local quat JSON.

Uses the existing scripts/export_motion_json.py retargeter under the hood.
Writes to coach/motion_cache/<id>.json. Skips files that already exist.

Usage:
    py -3.12 -m coach.batch_retarget --limit 20             # smoke
    py -3.12 -m coach.batch_retarget --per-genre 12         # 120 clips
    py -3.12 -m coach.batch_retarget                        # all 423
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIST_DIR = ROOT / 'data' / 'motion_db' / 'aistpp_full' / 'motions'
CACHE_DIR = Path(__file__).resolve().parent / 'motion_cache'
EXPORT_SCRIPT = ROOT / 'scripts' / 'export_motion_json.py'
VRM = ROOT / 'data' / 'models' / 'extra' / 'AvatarSample_A.vrm'
SMPL_PKL = ROOT / 'data' / 'smplx_models' / 'smpl' / 'SMPL_FEMALE.pkl'

PY = r'C:\Users\rohitacharya\AppData\Local\Programs\Python\Python312\python.exe'


def pick_files(per_genre: int | None, limit: int | None,
               src_dir: Path) -> list[Path]:
    files = sorted(src_dir.glob('*.pkl'))
    if per_genre is None and limit is None:
        return files
    if per_genre is not None:
        by_genre: dict[str, list[Path]] = {}
        for f in files:
            g = f.name[:3]
            by_genre.setdefault(g, []).append(f)
        out: list[Path] = []
        for g, fs in by_genre.items():
            out.extend(fs[:per_genre])
        return out
    return files[:limit]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=str(AIST_DIR),
                   help='Source dir of .pkl files (default: AIST++).')
    p.add_argument('--dst', default=str(CACHE_DIR),
                   help='Output dir for retargeted JSONs.')
    p.add_argument('--per-genre', type=int, default=None,
                   help='Cap per genre (3-char prefix).')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--shuffle', action='store_true',
                   help='Shuffle before applying --limit so the sample '
                        'is not biased to early subjects.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    dst_dir.mkdir(exist_ok=True, parents=True)
    targets = pick_files(args.per_genre, args.limit, src_dir)
    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(targets)
        if args.limit:
            targets = targets[:args.limit]
    print(f'[batch] {len(targets)} clips  src={src_dir}  dst={dst_dir}')
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    for i, src in enumerate(targets):
        out = dst_dir / f'{src.stem}.json'
        if out.exists() and not args.force:
            n_skip += 1
            continue
        t1 = time.time()
        try:
            r = subprocess.run(
                [PY, str(EXPORT_SCRIPT),
                 '--aist', str(src),
                 '--vrm',  str(VRM),
                 '--smpl_pkl', str(SMPL_PKL),
                 '--out',  str(out)],
                capture_output=True, text=True, timeout=180,
                cwd=str(ROOT))
            if r.returncode == 0 and out.exists():
                n_ok += 1
                elapsed = time.time() - t1
                print(f'  [{i+1}/{len(targets)}] OK {src.name}  '
                      f'{elapsed:.1f}s')
            else:
                n_fail += 1
                tail = (r.stderr or r.stdout).strip().splitlines()[-2:]
                print(f'  [{i+1}/{len(targets)}] FAIL {src.name}: {tail}')
        except subprocess.TimeoutExpired:
            n_fail += 1
            print(f'  [{i+1}/{len(targets)}] TIMEOUT {src.name}')
        # show progress
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(targets) - i - 1) / max(rate, 0.001)
            print(f'  ··· {i+1}/{len(targets)}  '
                  f'rate={rate:.2f}/s  eta={remaining/60:.1f}min')
    print(f'[batch] done  ok={n_ok}  skip={n_skip}  fail={n_fail}  '
          f'elapsed={(time.time()-t0)/60:.1f}min')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
