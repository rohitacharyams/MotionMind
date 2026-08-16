"""motion_qa_pipeline.py — batch QA + auto-fix for motion datasets.

This pipeline validates and fixes source `.pkl` motions before they are
exposed to teaching/runtime systems.

Usage:
    py -3.11 -m coach.ingestion.motion_qa_pipeline
    py -3.11 -m coach.ingestion.motion_qa_pipeline --fix
    py -3.11 -m coach.ingestion.motion_qa_pipeline --only-cmu --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
COACH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AIST = ROOT / 'data' / 'motion_db' / 'aistpp_full' / 'motions'
DEFAULT_CMU = ROOT / 'data' / 'motion_db' / 'amass_cmu'
OUT_DIR = COACH_DIR / 'reports'


def _run_scan(directory: Path,
              out_path: Path,
              broken_path: Path,
              fix: bool,
              dry_run: bool,
              floor_percentile: float,
              target_floor_z: float,
              max_xy_radius: float,
              center_xy: bool) -> int:
    cmd = [
        sys.executable,
        '-m',
        'coach.physics_validator',
        'scan',
        '--dir', str(directory),
        '--glob', '*.pkl',
        '--out', str(out_path),
        '--broken-out', str(broken_path),
        '--floor-percentile', str(floor_percentile),
        '--target-floor-z', str(target_floor_z),
        '--max-xy-radius', str(max_xy_radius),
    ]
    if not center_xy:
        cmd.append('--no-center-xy')
    if fix:
        cmd.append('--fix')
    if dry_run:
        cmd.append('--dry-run-fix')

    print(f'[qa] scanning {directory} ...')
    r = subprocess.run(cmd, cwd=str(ROOT))
    return int(r.returncode)


def _summary(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0}
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
        return d.get('summary', {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0})
    except Exception:
        return {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--aist-dir', default=str(DEFAULT_AIST))
    p.add_argument('--cmu-dir', default=str(DEFAULT_CMU))
    p.add_argument('--only-aist', action='store_true')
    p.add_argument('--only-cmu', action='store_true')
    p.add_argument('--fix', action='store_true',
                   help='Write fixes into source motion files.')
    p.add_argument('--dry-run', action='store_true',
                   help='Simulate fixes and output post-fix reports without writing files.')
    p.add_argument('--floor-percentile', type=float, default=2.0)
    p.add_argument('--target-floor-z', type=float, default=0.0)
    p.add_argument('--max-xy-radius', type=float, default=2.5)
    p.add_argument('--no-center-xy', action='store_true')
    args = p.parse_args()

    if args.only_aist and args.only_cmu:
        print('[qa] choose only one of --only-aist / --only-cmu')
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, str]] = []

    if not args.only_cmu:
        runs.append({
            'name': 'aist',
            'dir': str(Path(args.aist_dir).resolve()),
            'report': str((OUT_DIR / 'motion_qa_aist.json').resolve()),
            'broken': str((OUT_DIR / 'motion_qa_aist_broken.json').resolve()),
        })
    if not args.only_aist:
        runs.append({
            'name': 'cmu',
            'dir': str(Path(args.cmu_dir).resolve()),
            'report': str((OUT_DIR / 'motion_qa_cmu.json').resolve()),
            'broken': str((OUT_DIR / 'motion_qa_cmu_broken.json').resolve()),
        })

    rc = 0
    totals = {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0}
    for run in runs:
        ds = Path(run['dir'])
        if not ds.exists():
            print(f"[qa] skip missing dir: {ds}")
            continue
        scan_rc = _run_scan(
            directory=ds,
            out_path=Path(run['report']),
            broken_path=Path(run['broken']),
            fix=bool(args.fix),
            dry_run=bool(args.dry_run),
            floor_percentile=float(args.floor_percentile),
            target_floor_z=float(args.target_floor_z),
            max_xy_radius=float(args.max_xy_radius),
            center_xy=not bool(args.no_center_xy),
        )
        rc = max(rc, scan_rc)
        sm = _summary(Path(run['report']))
        totals['total'] += int(sm.get('total', 0))
        totals['pass'] += int(sm.get('pass', 0))
        totals['warn'] += int(sm.get('warn', 0))
        totals['fail'] += int(sm.get('fail', 0))
        print(f"[qa] {run['name']} summary: total={sm.get('total', 0)} pass={sm.get('pass', 0)} warn={sm.get('warn', 0)} fail={sm.get('fail', 0)}")

    combined = {
        'mode': 'fix' if args.fix else ('dry-run' if args.dry_run else 'scan'),
        'reports': runs,
        'summary': totals,
    }
    combined_path = OUT_DIR / 'motion_qa_summary.json'
    combined_path.write_text(json.dumps(combined, indent=2), encoding='utf-8')
    print(f"[qa] combined summary written -> {combined_path}")
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
