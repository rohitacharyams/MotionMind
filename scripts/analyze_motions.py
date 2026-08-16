"""Offline motion analyzer.

For every clip in coach/motion_cache and coach/motion_cache_cmu, computes:
  - yaw_rad        radians to add to vrm.scene.rotation.y so the dominant
                   XZ travel direction points away from the camera (-Z)
  - in_place       True if total XZ travel < 0.30 m
  - travel_m       total straight-line XZ travel in first 2 s
  - ground_offset_y     ADD this to hips.y so global lowest foot lands on
                        floor (y = 0). Computed via FK on up to 90 sample
                        frames.
  - hip_y_min/max/range
  - foot_min       global lowest foot world Y observed
  - is_jump        True if a significant share of sampled frames have both
                   feet > 0.25 m above the clip's foot baseline

Output: coach/motion_meta/corrections.json   { name: {meta...} }

Run:  python c:\\dan\\analyze_motions.py
"""
from __future__ import annotations
import json, os, math, glob, sys
import numpy as np

ROOT = r'c:\dan\coach'
DIRS = [os.path.join(ROOT, 'motion_cache'), os.path.join(ROOT, 'motion_cache_cmu')]
OUT  = os.path.join(ROOT, 'motion_meta', 'corrections.json')

PARENT = {
    'Hips': None,
    'Spine': 'Hips', 'Spine2': 'Spine', 'Neck': 'Spine2', 'Head': 'Neck',
    'LeftShoulder':  'Spine2', 'LeftArm':  'LeftShoulder',
    'LeftForeArm':   'LeftArm', 'LeftHand': 'LeftForeArm',
    'RightShoulder': 'Spine2', 'RightArm': 'RightShoulder',
    'RightForeArm':  'RightArm', 'RightHand':'RightForeArm',
    'LeftUpLeg':  'Hips', 'LeftLeg':  'LeftUpLeg',  'LeftFoot':  'LeftLeg',
    'RightUpLeg': 'Hips', 'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg',
}

def q_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])

def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]

CHAINS = {
    'LeftFoot':  ['Hips', 'LeftUpLeg',  'LeftLeg',  'LeftFoot'],
    'RightFoot': ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot'],
}

def fk_foot_y(j, i):
    """Return min world Y of the two feet at frame i."""
    rest_t = j['rest_local_translation']
    rest_q = j.get('rest_local_rotation') or {}
    rot    = j['rotations']
    htr    = j['hips_translation'][i]
    min_y = float('inf')
    for chain in CHAINS.values():
        # hips_translation already encodes the absolute world position
        # of the Hips bone — do NOT add rest_t['Hips'] on top of it.
        pos = np.array([htr[0], htr[1], htr[2]], dtype=float)
        R = np.eye(3)
        for k, bone in enumerate(chain):
            if k > 0:  # skip the root translation; only children move within parent frame
                lt  = np.array(rest_t.get(bone, [0, 0, 0]), dtype=float)
                pos = pos + R @ lt
            br  = rest_q.get(bone, [0, 0, 0, 1])
            arr = rot.get(bone)
            if arr is None or i >= len(arr):
                q = br
            else:
                q = q_mul(br, arr[i])
            R = R @ q_to_mat(q)
        if pos[1] < min_y:
            min_y = pos[1]
    return min_y

def analyze(path):
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    n   = int(j.get('n_frames') or 0)
    fps = float(j.get('fps') or 30.0)
    hT  = j.get('hips_translation') or []
    if not hT or len(hT) < 2 or n < 2:
        return name, None

    # ---- travel-direction yaw (first 2 s) -----------------------------
    end = min(len(hT), max(2, int(round(fps * 2.0))))
    a, b = hT[0], hT[end - 1]
    dx, dz = b[0] - a[0], b[2] - a[2]
    travel = math.hypot(dx, dz)
    in_place = travel < 0.30
    if travel > 0.30:
        yaw_rad = math.pi - math.atan2(dx, dz)
        yaw_source = 'travel'
    else:
        # Pelvis-quat based yaw is unreliable for CMU rigs because the
        # Hips quat bakes in a rig-conversion rotation that three-vrm's
        # normalized humanoid handles separately. For in-place clips we
        # leave yaw at 0 here and let motion_player measure the actual
        # rendered shoulder-line at runtime.
        yaw_rad = 0.0
        yaw_source = 'none'

    while yaw_rad > math.pi:  yaw_rad -= 2 * math.pi
    while yaw_rad < -math.pi: yaw_rad += 2 * math.pi

    # ---- hip Y stats --------------------------------------------------
    ys = np.array([t[1] for t in hT if t])
    hip_y_min, hip_y_max = float(ys.min()), float(ys.max())
    hip_y_med = float(np.median(ys))
    hip_y_range = hip_y_max - hip_y_min

    # ---- FK sample to find lowest foot --------------------------------
    n_samp = min(90, n)
    step   = max(1, n // n_samp)
    foot_ys = []
    for i in range(0, n, step):
        try:
            foot_ys.append(fk_foot_y(j, i))
        except Exception:
            pass
    if not foot_ys:
        return name, None
    foot_arr = np.array(foot_ys)
    foot_min_global = float(foot_arr.min())
    foot_p10 = float(np.percentile(foot_arr, 10))  # robust baseline
    # ground_offset_y: ADD to hips.y so the 10th-percentile lowest foot
    # sits on the floor (y=0). Using p10 instead of absolute min lets
    # genuine jump apex stay above floor without yanking the whole clip
    # underground.
    ground_offset_y = -foot_p10

    # Jump detection: fraction of sample frames where the min foot is
    # > 25 cm above the p10 baseline.
    n_airborne = int(((foot_arr - foot_p10) > 0.25).sum())
    is_jump = n_airborne >= max(2, len(foot_arr) // 8)

    # v15: SQUAT-REHAB DETECTOR. CMU rehab subjects (105, 137, 140, ...)
    # were captured doing PT exercises — leg lifts, balance drills —
    # in which the hip drops 25-40 cm without leaving the floor. Played
    # back as a "warmup" the avatar appears to crouch + sway awkwardly
    # while the upper body does the intended arm move. Detect: in_place
    # + not-jump + significant hip vertical range. Frontend uses this
    # flag to clamp dy tightly so only upper-body motion survives.
    suppress_hip_bob = bool(in_place and (not is_jump) and hip_y_range > 0.20)

    # v15: QUALITY SCORE. Lower = better. Used by qa_motions.py to
    # quarantine the worst clips before they reach users.
    quality_issues = []
    if suppress_hip_bob:
        quality_issues.append(f'hip_bob_{hip_y_range:.2f}m')
    if travel > 3.0 and not in_place:
        quality_issues.append(f'huge_travel_{travel:.1f}m')
    if abs(ground_offset_y) > 0.50:
        quality_issues.append(f'bad_ground_offset_{ground_offset_y:.2f}')

    return name, {
        'yaw_rad': round(yaw_rad, 4),
        'yaw_source': yaw_source,
        'in_place': bool(in_place),
        'travel_m': round(travel, 3),
        'ground_offset_y': round(ground_offset_y, 4),
        'hip_y_min': round(hip_y_min, 3),
        'hip_y_max': round(hip_y_max, 3),
        'hip_y_range': round(hip_y_range, 3),
        'hip_y_med': round(hip_y_med, 3),
        'foot_min': round(foot_min_global, 3),
        'foot_p10': round(foot_p10, 3),
        'is_jump': bool(is_jump),
        'suppress_hip_bob': suppress_hip_bob,
        'quality_issues': quality_issues,
        'n_frames': n,
        'fps': fps,
    }

def main():
    files = []
    for d in DIRS:
        if os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, '*.json')))
    print(f'analyzing {len(files)} files...', flush=True)
    out, fails = {}, []
    for i, p in enumerate(files):
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(files)}', flush=True)
        try:
            name, meta = analyze(p)
            if meta:
                out[name] = meta
            else:
                fails.append(p)
        except Exception as e:
            fails.append((p, str(e)))
            print(f'  FAIL {os.path.basename(p)}: {e}', flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {len(out)} corrections -> {OUT}', flush=True)
    if fails:
        print(f'  {len(fails)} failures', flush=True)
    # quick stats
    if out:
        n_inplace = sum(1 for v in out.values() if v['in_place'])
        n_jump    = sum(1 for v in out.values() if v['is_jump'])
        offs = np.array([v['ground_offset_y'] for v in out.values()])
        travel = np.array([v['travel_m'] for v in out.values()])
        print(f'in_place: {n_inplace}/{len(out)}  jumps: {n_jump}/{len(out)}')
        print(f'ground_offset_y: min={offs.min():.3f} max={offs.max():.3f} med={np.median(offs):.3f}')
        print(f'travel_m (first 2s): min={travel.min():.2f} max={travel.max():.2f} med={np.median(travel):.2f}')

if __name__ == '__main__':
    sys.exit(main())
