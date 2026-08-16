"""Per-frame motion sanity check across the whole catalogue.

For every clip in both caches, walks every frame and records:
  - frame-to-frame angular jump per bone (deg). >120° = jitter/glitch.
  - lateral drift (sqrt(dx²+dz²) of hips over the clip).
  - vertical bob (hip_y_max - hip_y_min).
  - lowest foot world Y (using FK from validate_runtime_floor.py logic).
  - duplicate-frame runs (frozen output).

Outputs coach/reports/per_frame_qa.json with:
  - per-clip stats
  - top-N worst offenders by category
  - a 'verdict' (clean / warn / bad) per clip.
"""
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).parent
CACHES = [
    ROOT / 'coach' / 'motion_cache',
    ROOT / 'coach' / 'motion_cache_cmu',
]
OUT = ROOT / 'coach' / 'reports' / 'per_frame_qa.json'

# glTF quat order [x,y,z,w]
def quat_angle_deg(q1, q2) -> float:
    """Geodesic angle between two unit quaternions in degrees."""
    d = abs(sum(a*b for a, b in zip(q1, q2)))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def lowest_foot_y(rest_t, rotations, hips_t, frames_to_sample=30) -> float:
    """Conservative lowest foot Y across sampled frames (world coords)."""
    leg_chains = [
        ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'],
        ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot'],
    ]
    n = len(hips_t) if hips_t else 0
    if n == 0:
        return 0.0
    sample = max(1, n // frames_to_sample)
    lowest = float('inf')
    for f in range(0, n, sample):
        for chain in leg_chains:
            pos = np.zeros(3)
            R = np.eye(3)
            for i, bone in enumerate(chain):
                if i == 0:
                    pos = np.array(hips_t[f], dtype=float)
                    q = rotations.get(bone, [[0,0,0,1]])
                    if f < len(q):
                        R = quat_to_mat(q[f])
                else:
                    offset = np.array(rest_t.get(bone, [0,0,0]), dtype=float)
                    pos = pos + R @ offset
                    q = rotations.get(bone, [[0,0,0,1]])
                    if f < len(q):
                        R = R @ quat_to_mat(q[f])
            if pos[1] < lowest:
                lowest = pos[1]
    return float(lowest) if lowest != float('inf') else 0.0


def analyse(path: Path) -> Dict:
    d = json.loads(path.read_text(encoding='utf-8'))
    rotations = d.get('rotations', {})
    hips_t = d.get('hips_translation') or []
    rest_t = d.get('rest_local_translation', {})
    n = len(hips_t) if hips_t else max((len(v) for v in rotations.values()), default=0)

    # 1. frame-to-frame jumps per bone
    max_jump = 0.0
    max_jump_bone = ''
    max_jump_frame = 0
    over_threshold = 0  # count frame×bone with >120° jump
    for bone, frames in rotations.items():
        for f in range(1, len(frames)):
            a = quat_angle_deg(frames[f-1], frames[f])
            if a > 120.0:
                over_threshold += 1
            if a > max_jump:
                max_jump = a
                max_jump_bone = bone
                max_jump_frame = f

    # 2. lateral drift + vertical bob
    if hips_t:
        xs = [p[0] for p in hips_t]
        ys = [p[1] for p in hips_t]
        zs = [p[2] for p in hips_t]
        lateral = math.hypot(max(xs)-min(xs), max(zs)-min(zs))
        bob = max(ys) - min(ys)
        end_drift = math.hypot(xs[-1]-xs[0], zs[-1]-zs[0])
    else:
        lateral = bob = end_drift = 0.0

    # 3. duplicate-frame runs (any bone frozen >1.5s)
    frozen_runs = 0
    longest_frozen = 0
    fps = d.get('fps') or 30.0
    threshold_frames = int(fps * 1.5)
    for bone, frames in rotations.items():
        run = 1
        for f in range(1, len(frames)):
            if quat_angle_deg(frames[f-1], frames[f]) < 0.1:
                run += 1
            else:
                if run > threshold_frames:
                    frozen_runs += 1
                if run > longest_frozen:
                    longest_frozen = run
                run = 1
        if run > threshold_frames:
            frozen_runs += 1
        if run > longest_frozen:
            longest_frozen = run

    # 4. lowest foot y
    lowest = lowest_foot_y(rest_t, rotations, hips_t)

    # 5. verdict
    issues: List[str] = []
    if max_jump > 120: issues.append(f'angular_jitter_{max_jump:.0f}deg_at_{max_jump_bone}_f{max_jump_frame}')
    if bob > 0.20:    issues.append(f'hip_bob_{bob*100:.0f}cm')
    if lateral > 5.0: issues.append(f'lateral_drift_{lateral:.1f}m')
    if longest_frozen > threshold_frames * 2:
        issues.append(f'frozen_run_{longest_frozen}frames')
    if lowest < -0.05: issues.append(f'foot_below_floor_{lowest*100:.1f}cm')
    if lowest > 0.30:  issues.append(f'foot_high_{lowest*100:.0f}cm_above_floor')

    verdict = 'clean'
    if issues:
        verdict = 'bad' if any('jitter' in i or 'frozen' in i for i in issues) else 'warn'

    return {
        'clip': path.stem,
        'n_frames': n,
        'fps': fps,
        'max_angular_jump_deg': round(max_jump, 1),
        'max_jump_bone': max_jump_bone,
        'max_jump_frame': max_jump_frame,
        'jump_events_over_120deg': over_threshold,
        'lateral_drift_m': round(lateral, 3),
        'end_drift_m': round(end_drift, 3),
        'hip_bob_m': round(bob, 3),
        'lowest_foot_y_m': round(lowest, 3),
        'longest_frozen_run_frames': longest_frozen,
        'frozen_runs': frozen_runs,
        'issues': issues,
        'verdict': verdict,
    }


def main():
    rows = []
    for cache in CACHES:
        if not cache.exists():
            continue
        for f in sorted(cache.glob('*.json')):
            try:
                rows.append(analyse(f))
            except Exception as e:
                rows.append({'clip': f.stem, 'error': str(e), 'verdict': 'error'})

    by_verdict: Dict[str, int] = {}
    for r in rows:
        by_verdict[r.get('verdict', '?')] = by_verdict.get(r.get('verdict', '?'), 0) + 1

    worst_jitter = sorted([r for r in rows if r.get('jump_events_over_120deg', 0) > 0],
                          key=lambda r: -r.get('jump_events_over_120deg', 0))[:10]
    worst_bob = sorted([r for r in rows if r.get('hip_bob_m', 0) > 0.20],
                       key=lambda r: -r.get('hip_bob_m', 0))[:10]
    worst_drift = sorted([r for r in rows if r.get('lateral_drift_m', 0) > 3.0],
                         key=lambda r: -r.get('lateral_drift_m', 0))[:10]
    worst_floor = sorted(rows, key=lambda r: r.get('lowest_foot_y_m', 0))[:10]

    out = {
        'total': len(rows),
        'by_verdict': by_verdict,
        'top_jitter': [(r['clip'], r['jump_events_over_120deg'], r['max_angular_jump_deg']) for r in worst_jitter],
        'top_hip_bob_m': [(r['clip'], r['hip_bob_m']) for r in worst_bob],
        'top_lateral_drift_m': [(r['clip'], r['lateral_drift_m']) for r in worst_drift],
        'top_foot_below_floor_m': [(r['clip'], r['lowest_foot_y_m']) for r in worst_floor[:10]],
        'rows': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f'wrote {OUT}')
    print(f'verdicts: {by_verdict}')
    print('top jitter:', out['top_jitter'][:5])
    print('top hip_bob:', out['top_hip_bob_m'][:5])
    print('top drift:', out['top_lateral_drift_m'][:5])


if __name__ == '__main__':
    main()
