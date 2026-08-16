"""Generate Blender review files and floor-audit reports for motion clips.

Pipeline per clip:
1) export .pkl motion to .glb
2) run Blender floor audit (saves .blend + audit .json)
3) aggregate summary JSON for gating

Default input set is the currently non-ok clips from motion QA reports.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
COACH = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'

DEFAULT_AIST_BROKEN = COACH / 'reports' / 'motion_qa_aist_broken.json'
DEFAULT_CMU_BROKEN = COACH / 'reports' / 'motion_qa_cmu_broken.json'
OUT_ROOT = COACH / 'reports' / 'floor_audit'

BLENDER = Path(r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe')
PYTHON312 = Path(r'C:\Users\rohitacharya\AppData\Local\Programs\Python\Python312\python.exe')
PYTHON = PYTHON312 if PYTHON312.exists() else Path(sys.executable)
EXPORT_GLB = SCRIPTS / 'export_glb.py'
BLENDER_AUDIT = SCRIPTS / 'blender_floor_audit.py'

# Default to Kira avatar for consistent cross-clip visual checks.
VRM_DEFAULT = ROOT / 'data' / 'models' / 'extra' / 'AvatarSample_K.vrm'
SMPL_PKL = ROOT / 'data' / 'models' / 'smpl_raw' / 'smpl' / 'models' / 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'


def _load_broken(path: Path) -> List[str]:
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding='utf-8'))
    out: List[str] = []
    for c in d.get('clips', []):
        p = c.get('path', '')
        if p:
            out.append(str(Path(p).resolve()))
    return out


def _clip_id_from_path(p: Path) -> str:
    return p.stem


def _run(cmd: List[str], cwd: Path | None = None) -> int:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return int(r.returncode)


def _export_glb(pkl_path: Path, out_glb: Path, vrm_path: Path) -> int:
    cmd = [
        str(PYTHON),
        str(EXPORT_GLB),
        '--aist', str(pkl_path),
        '--vrm', str(vrm_path),
        '--smpl_pkl', str(SMPL_PKL),
        '--out', str(out_glb),
    ]
    return _run(cmd, cwd=ROOT)


def _run_blender_audit(glb: Path, blend_out: Path, audit_out: Path) -> int:
    cmd = [
        str(BLENDER),
        '--background',
        '--python', str(BLENDER_AUDIT),
        '--',
        '--glb', str(glb),
        '--blend_out', str(blend_out),
        '--audit_out', str(audit_out),
        '--frame_stride', '5',
        '--max_samples', '450',
    ]
    return _run(cmd, cwd=ROOT)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--aist-broken', default=str(DEFAULT_AIST_BROKEN))
    p.add_argument('--cmu-broken', default=str(DEFAULT_CMU_BROKEN))
    p.add_argument('--out', default=str(OUT_ROOT))
    p.add_argument('--vrm', default=str(VRM_DEFAULT))
    p.add_argument('--limit', type=int, default=0,
                   help='optional cap for smoke testing')
    p.add_argument('--force', action='store_true',
                   help='recompute even when blend+audit already exist')
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    glb_dir = out_root / 'glb'
    blend_dir = out_root / 'blend'
    audit_dir = out_root / 'audit_json'
    for d in (out_root, glb_dir, blend_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    clips = []
    clips.extend(_load_broken(Path(args.aist_broken).resolve()))
    clips.extend(_load_broken(Path(args.cmu_broken).resolve()))

    # Stable order + dedupe
    clips = sorted(set(clips))
    if args.limit and args.limit > 0:
        clips = clips[: args.limit]

    if not clips:
        print('[floor-audit] no clips to process')
        return 0

    vrm = Path(args.vrm).resolve()
    if not vrm.exists():
        print(f'[floor-audit] vrm missing: {vrm}')
        return 2
    if not BLENDER.exists():
        print(f'[floor-audit] blender missing: {BLENDER}')
        return 2

    rows: List[Dict[str, object]] = []
    fail_count = 0

    for i, clip_path in enumerate(clips, start=1):
        pkl = Path(clip_path)
        clip_id = _clip_id_from_path(pkl)
        print(f'[floor-audit] [{i}/{len(clips)}] {clip_id}')
        if not pkl.exists():
            rows.append({'clip_id': clip_id, 'pkl': str(pkl), 'status': 'missing_pkl'})
            fail_count += 1
            continue

        glb = glb_dir / f'{clip_id}.glb'
        blend = blend_dir / f'{clip_id}.blend'
        audit = audit_dir / f'{clip_id}.json'

        if (not args.force) and blend.exists() and audit.exists():
            try:
                ad = json.loads(audit.read_text(encoding='utf-8'))
                rows.append({
                    'clip_id': clip_id,
                    'pkl': str(pkl),
                    'glb': str(glb) if glb.exists() else '',
                    'blend': str(blend),
                    'audit_json': str(audit),
                    'severity': ad.get('severity', 'unknown'),
                    'stats': ad.get('stats', {}),
                    'status': 'ok',
                    'reused': True,
                })
                continue
            except Exception:
                pass

        rc1 = _export_glb(pkl, glb, vrm)
        if rc1 != 0:
            rows.append({'clip_id': clip_id, 'pkl': str(pkl), 'status': 'export_failed'})
            fail_count += 1
            continue

        rc2 = _run_blender_audit(glb, blend, audit)
        if rc2 != 0:
            rows.append({'clip_id': clip_id, 'pkl': str(pkl), 'glb': str(glb), 'status': 'blender_audit_failed'})
            fail_count += 1
            continue

        try:
            ad = json.loads(audit.read_text(encoding='utf-8'))
            rows.append({
                'clip_id': clip_id,
                'pkl': str(pkl),
                'glb': str(glb),
                'blend': str(blend),
                'audit_json': str(audit),
                'severity': ad.get('severity', 'unknown'),
                'stats': ad.get('stats', {}),
                'status': 'ok',
            })
        except Exception:
            rows.append({
                'clip_id': clip_id,
                'pkl': str(pkl),
                'glb': str(glb),
                'blend': str(blend),
                'audit_json': str(audit),
                'status': 'ok_no_parse',
            })

    summary = {
        'total': len(rows),
        'ok': sum(1 for r in rows if r.get('status') == 'ok'),
        'failed': fail_count,
        'rows': rows,
    }
    out_summary = out_root / 'floor_audit_summary.json'
    out_summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[floor-audit] summary -> {out_summary}')
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
