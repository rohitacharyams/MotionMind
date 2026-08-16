"""build_clip.py — turn an AIST-style SMPL .pkl (from gvhmr_to_aist.py) into a
playable VRM-quat clip in coach/motion_cache/, reusing the repo's EXISTING
retarget + safety pipeline. Run this LOCALLY.

Chain
-----
  1. (rename)  <pkl> -> <name>.pkl        so the clip's internal name is <name>
  2. export    scripts/export_motion_json.py  (SMPL -> VRM-quat JSON)
  3. sign-fix  fix_quaternion_signs.fix_clip  (quaternion hemisphere continuity)
  4. validate  coach.physics_validator.validate_motion  (absurd-pose gate)
  5. install   copy JSON -> coach/motion_cache/<name>.json  (served by the coach)

Nothing here is new logic — it's the same steps AIST clips already go through.
The point of the prototype is step 2+4 telling you whether the SMPL you pulled
from a random video is actually clean.

Usage
-----
  python scripts/video_to_smpl/build_clip.py \
      --aist my_clip.pkl --name my_first_video_clip \
      --vrm  data/models/extra/AliciaSolid.vrm \
      --preview            # optional: also render an mp4 to eyeball it
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

MOTION_CACHE = os.path.join(REPO, 'coach', 'motion_cache')
DEFAULT_VRM = os.path.join('data', 'models', 'extra', 'AliciaSolid.vrm')
DEFAULT_SMPL = os.path.join('data', 'models', 'smpl_raw', 'smpl', 'SMPL_NEUTRAL.pkl')


def _validate(pkl_path: str, fps: float):
    """Run the same physics gate the coach uses. Returns SafetyReport."""
    from coach.physics_validator import validate_motion
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    poses = np.asarray(d['smpl_poses']).reshape(-1, 24, 3)
    trans = np.asarray(d['smpl_trans']).reshape(-1, 3)
    return validate_motion(poses, trans, fps=int(round(fps)), path=pkl_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aist', required=True, help='AIST-style SMPL .pkl')
    ap.add_argument('--name', required=True, help='clip id (no extension)')
    ap.add_argument('--vrm', default=DEFAULT_VRM)
    ap.add_argument('--smpl_pkl', default=DEFAULT_SMPL)
    ap.add_argument('--out-cache', default=MOTION_CACHE)
    ap.add_argument('--preview', action='store_true',
                    help='render an mp4 with play_smpl_motion.py to eyeball it')
    ap.add_argument('--allow-fail', action='store_true',
                    help='install the clip even if the safety gate fails')
    args = ap.parse_args()

    with open(args.aist, 'rb') as f:
        fps = float(pickle.load(f).get('fps', 30.0))

    # ── 1. rename pkl so export_motion_json names the clip <name> ──────────
    tmpdir = tempfile.mkdtemp(prefix='v2smpl_')
    named_pkl = os.path.join(tmpdir, f'{args.name}.pkl')
    shutil.copy2(args.aist, named_pkl)

    out_json = os.path.join(args.out_cache, f'{args.name}.json')
    os.makedirs(args.out_cache, exist_ok=True)

    # ── 2. retarget SMPL -> VRM-quat JSON (existing script) ────────────────
    print('── [2/5] retargeting SMPL -> VRM-quat ...')
    cmd = [sys.executable, os.path.join('scripts', 'export_motion_json.py'),
           '--aist', named_pkl, '--vrm', args.vrm,
           '--smpl_pkl', args.smpl_pkl, '--out', out_json]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit('export_motion_json.py failed — see output above.')

    # ── 3. quaternion sign continuity (existing function) ──────────────────
    print('── [3/5] fixing quaternion signs ...')
    from fix_quaternion_signs import fix_clip
    from pathlib import Path
    res = fix_clip(Path(out_json))
    print(f'   flipped {res["flipped"]} frames across {res["bones"]} bones')

    # ── 4. physics / orientation safety gate (existing validator) ──────────
    print('── [4/5] validating motion ...')
    rep = _validate(named_pkl, fps)
    print(f'   passed={rep.passed}  severity={rep.severity}  '
          f'max_joint_speed={rep.max_joint_speed_rad_s:.1f} rad/s  '
          f'max_pelvis_speed={rep.max_pelvis_speed_m_s:.2f} m/s')
    for v in rep.violations[:8]:
        print(f'     ! {v}')
    if not rep.passed and not args.allow_fail:
        os.remove(out_json)
        raise SystemExit(
            'SAFETY GATE FAILED — clip not installed. '
            'Try --root-fix in gvhmr_to_aist.py, trim bad frames, or re-shoot '
            'the video. Re-run with --allow-fail to install anyway.')

    # ── 5. installed (already written into motion_cache) ───────────────────
    print(f'── [5/5] installed: {out_json}')
    print(f'   Coach will serve it at /api/motion/data/{args.name}.json')

    # ── optional visual preview ────────────────────────────────────────────
    if args.preview:
        mp4 = os.path.join('data', 'output_videos', f'v2smpl_{args.name}.mp4')
        print(f'── rendering preview -> {mp4} ...')
        subprocess.run(
            [sys.executable, os.path.join('scripts', 'play_smpl_motion.py'),
             '--aist', named_pkl, '--vrm', args.vrm,
             '--smpl_pkl', args.smpl_pkl, '--out', mp4],
            cwd=REPO)

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
