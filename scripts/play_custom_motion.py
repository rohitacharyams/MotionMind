"""
Play Custom Motion — Drive the VRM avatar with hand-authored keyframes.

Demonstrates that once we have a rigged character + working FK/skinning,
ANY motion can be authored as a sequence of joint-position dictionaries
and played back. Here we hand-author a "right-leg front kick" cycle:

    rest pose  →  knee lift  →  full extension  →  retract  →  rest

You can also load motion from public datasets:
  • AIST++   (COCO 17 3D, .pkl)            — provided as a follow-up
  • AMASS    (SMPL pose params, .npz)
  • Mixamo   (FBX rigged anim)

Usage:
    python scripts/play_custom_motion.py
    python scripts/play_custom_motion.py --motion wave
    python scripts/play_custom_motion.py --json my_keyframes.json

Outputs:
    data/output_videos/custom_motion_<name>.mp4
"""
import os, sys, time, json, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_transfer_vrm import _TPOSE
from motion_transfer_v6 import VRMCharacterRendererV6


VRM_MODEL = 'data/models/fem_vroid.vrm'
OUT_DIR   = 'data/output_videos'
WIDTH, HEIGHT = 720, 720
FPS = 30.0

os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Pose builders — small library of named keyframes
# ═══════════════════════════════════════════════════════════════
#
# All positions are in CANONICAL space (Y up, +Z forward toward viewer,
# subject-right is +X, subject-left is -X). Hips ≈ (0, 0.95, 0).
# A bone-length unit ≈ 0.1-0.4 (canonical T-pose dimensions).
#
# Convention: each function returns a complete posed_joints dict.
# ───────────────────────────────────────────────────────────────

def rest_pose():
    """Plain T-pose copy."""
    return {k: np.array(v, dtype=np.float64) for k, v in _TPOSE.items()}


def kick_keyframes():
    """5 keyframes: stance → knee-lift → kick → retract → stance.

    Right leg performs a front kick. Arms swing slightly for balance.
    NOTE: in the renderer's convention, **negative Z = toward viewer**
    (the R180 mesh flip negates Z), so a forward kick uses Z < 0.
    """
    p0 = rest_pose()  # stance

    # Frame 1: right knee lifts (90° hip flexion, 90° knee flexion)
    p1 = rest_pose()
    p1['RightUpLeg'] = np.array([0.10, 0.92,  0.00])
    p1['RightLeg']   = np.array([0.10, 0.92, -0.30])         # knee out front (toward viewer)
    p1['RightFoot']  = np.array([0.10, 0.55, -0.32])         # foot tucked down/forward
    # Counter-balance: left arm forward, right arm back
    p1['LeftArm']    = np.array([-0.22, 1.38,  0.00])
    p1['LeftForeArm']= np.array([-0.30, 1.32, -0.20])
    p1['LeftHand']   = np.array([-0.32, 1.30, -0.40])
    p1['RightArm']   = np.array([0.22, 1.38, 0.00])
    p1['RightForeArm']=np.array([0.32, 1.32, 0.20])
    p1['RightHand']  = np.array([0.36, 1.30, 0.40])

    # Frame 2: full kick — leg extends straight forward
    p2 = rest_pose()
    p2['RightUpLeg'] = np.array([0.10, 0.92,  0.00])
    p2['RightLeg']   = np.array([0.10, 0.92, -0.42])
    p2['RightFoot']  = np.array([0.10, 0.92, -0.84])
    p2['LeftArm']    = np.array([-0.22, 1.38,  0.00])
    p2['LeftForeArm']= np.array([-0.30, 1.30, -0.30])
    p2['LeftHand']   = np.array([-0.34, 1.28, -0.55])
    p2['RightArm']   = np.array([0.22, 1.38, 0.00])
    p2['RightForeArm']=np.array([0.34, 1.30, 0.30])
    p2['RightHand']  = np.array([0.40, 1.28, 0.55])
    # Slight forward lean (counter-balance)
    p2['Spine']  = np.array([0.0, 1.05,  0.05])
    p2['Spine2'] = np.array([0.0, 1.20,  0.08])
    p2['Neck']   = np.array([0.0, 1.35,  0.10])
    p2['Head']   = np.array([0.0, 1.55,  0.10])

    # Frame 3: retract (mirror of frame 1)
    p3 = rest_pose()
    p3['RightUpLeg'] = np.array([0.10, 0.92,  0.00])
    p3['RightLeg']   = np.array([0.10, 0.85, -0.20])
    p3['RightFoot']  = np.array([0.10, 0.45, -0.22])
    p3['LeftArm']    = np.array([-0.22, 1.38,  0.00])
    p3['LeftForeArm']= np.array([-0.32, 1.32, -0.10])
    p3['LeftHand']   = np.array([-0.36, 1.30, -0.20])
    p3['RightArm']   = np.array([0.22, 1.38, 0.00])
    p3['RightForeArm']=np.array([0.32, 1.32, 0.10])
    p3['RightHand']  = np.array([0.36, 1.30, 0.20])

    p4 = rest_pose()  # back to stance

    return [p0, p1, p2, p3, p4]


def wave_keyframes():
    """Right hand wave: arm raised, hand swings left-right."""
    arm_up_neutral = rest_pose()
    arm_up_neutral['RightArm']     = np.array([0.22, 1.38, 0.00])
    arm_up_neutral['RightForeArm'] = np.array([0.45, 1.65, 0.00])
    arm_up_neutral['RightHand']    = np.array([0.50, 1.95, 0.00])

    arm_left = rest_pose()
    arm_left['RightArm']     = np.array([0.22, 1.38, 0.00])
    arm_left['RightForeArm'] = np.array([0.40, 1.65, 0.00])
    arm_left['RightHand']    = np.array([0.20, 1.95, 0.10])

    arm_right = rest_pose()
    arm_right['RightArm']     = np.array([0.22, 1.38, 0.00])
    arm_right['RightForeArm'] = np.array([0.50, 1.65, 0.00])
    arm_right['RightHand']    = np.array([0.75, 1.95, 0.10])

    return [rest_pose(), arm_up_neutral, arm_left, arm_right, arm_left,
            arm_right, arm_up_neutral, rest_pose()]


def squat_keyframes():
    """Simple squat-down + back-up cycle."""
    deep = rest_pose()
    # Lower hips, bend knees forward (negative Z = toward viewer)
    deep['Hips']        = np.array([0.0, 0.65, 0.0])
    deep['Spine']       = np.array([0.0, 0.78,  0.05])
    deep['Spine2']      = np.array([0.0, 0.95,  0.08])
    deep['Neck']        = np.array([0.0, 1.10,  0.08])
    deep['Head']        = np.array([0.0, 1.30,  0.08])
    deep['LeftUpLeg']   = np.array([-0.10, 0.62,  0.00])
    deep['LeftLeg']     = np.array([-0.13, 0.40, -0.20])
    deep['LeftFoot']    = np.array([-0.10, 0.08, -0.05])
    deep['RightUpLeg']  = np.array([0.10, 0.62,  0.00])
    deep['RightLeg']    = np.array([0.13, 0.40, -0.20])
    deep['RightFoot']   = np.array([0.10, 0.08, -0.05])
    # Arms forward for balance
    deep['LeftArm']     = np.array([-0.22, 1.05,  0.00])
    deep['LeftForeArm'] = np.array([-0.28, 1.00, -0.25])
    deep['LeftHand']    = np.array([-0.30, 0.98, -0.50])
    deep['RightArm']    = np.array([0.22, 1.05,  0.00])
    deep['RightForeArm']= np.array([0.28, 1.00, -0.25])
    deep['RightHand']   = np.array([0.30, 0.98, -0.50])

    return [rest_pose(), deep, rest_pose()]


MOTION_LIBRARY = {
    'kick':  (kick_keyframes,  60),   # 60 frames between keyframes ≈ 2 s/seg
    'wave':  (wave_keyframes,  20),
    'squat': (squat_keyframes, 30),
}


# ═══════════════════════════════════════════════════════════════
#  Interpolation
# ═══════════════════════════════════════════════════════════════

def smoothstep(t):
    """Cubic smoothstep for ease in/out."""
    return t * t * (3 - 2 * t)


def interpolate_keyframes(keyframes, frames_per_seg):
    """Linearly interp (with smoothstep ease) between successive keyframes.

    keyframes: list of posed-joint dicts.
    frames_per_seg: int — frames to spend transitioning A→B for each pair.

    Returns: flat list of posed-joint dicts.
    """
    out = []
    for k in range(len(keyframes) - 1):
        A, B = keyframes[k], keyframes[k + 1]
        for i in range(frames_per_seg):
            t = smoothstep(i / max(frames_per_seg - 1, 1))
            frame = {}
            for name in A:
                if name in B:
                    frame[name] = (1 - t) * A[name] + t * B[name]
                else:
                    frame[name] = A[name].copy()
            out.append(frame)
    out.append(keyframes[-1])
    return out


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--motion', default='kick',
                    choices=list(MOTION_LIBRARY.keys()),
                    help='Named motion from the library')
    ap.add_argument('--json', default=None,
                    help='Path to JSON file with keyframes (overrides --motion).')
    ap.add_argument('--frames-per-seg', type=int, default=None,
                    help='Override interpolation length per keyframe pair')
    ap.add_argument('--loops', type=int, default=2,
                    help='Repeat the motion N times (default 2)')
    ap.add_argument('--out', default=None,
                    help='Output mp4 path')
    args = ap.parse_args()

    print("=" * 70)
    print(f"  PLAY CUSTOM MOTION — '{args.motion}' x {args.loops}")
    print("=" * 70)

    # Build keyframes
    if args.json:
        with open(args.json) as f:
            raw = json.load(f)
        keyframes = []
        for kf in raw:
            keyframes.append({n: np.array(v, dtype=np.float64)
                              for n, v in kf.items()})
        fps_seg = args.frames_per_seg or 30
        name = os.path.splitext(os.path.basename(args.json))[0]
    else:
        kf_fn, default_seg = MOTION_LIBRARY[args.motion]
        keyframes = kf_fn()
        fps_seg = args.frames_per_seg or default_seg
        name = args.motion

    print(f"  {len(keyframes)} keyframes, {fps_seg} interp frames/seg")

    posed_list = interpolate_keyframes(keyframes, fps_seg)
    if args.loops > 1:
        # Loop seamlessly: drop the last frame on internal repeats
        loop_unit = posed_list[:-1] if len(posed_list) > 1 else posed_list
        posed_list = []
        for _ in range(args.loops):
            posed_list.extend(loop_unit)
        posed_list.append(keyframes[-1])

    n = len(posed_list)
    print(f"  Total frames: {n}  ({n / FPS:.1f}s @ {FPS:.0f} fps)")

    # Render
    print(f"\n  Loading VRM: {VRM_MODEL}")
    vrm = VRMCharacterRendererV6(VRM_MODEL, WIDTH, HEIGHT)

    out_path = args.out or os.path.join(OUT_DIR, f'custom_motion_{name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (WIDTH, HEIGHT))

    sample_idx = {0, n // 4, n // 2, 3 * n // 4, n - 1}
    t0 = time.time()
    for i in range(n):
        try:
            fr = vrm.render_frame_v2(posed_list[i],
                                     world_offset=np.zeros(3))
        except Exception as e:
            print(f"  render fail frame {i}: {e}")
            fr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.putText(fr, f"motion: {name}  frame {i+1}/{n}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        vw.write(fr)
        if i in sample_idx:
            cv2.imwrite(os.path.join(OUT_DIR,
                        f'custom_{name}_{i:04d}.png'), fr)
        if (i + 1) % 30 == 0:
            print(f"  rendered {i+1}/{n}  ({time.time()-t0:.1f}s)")

    vw.release()
    vrm.cleanup()
    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"  Output: {out_path}")
    print(f"  Sample previews: {OUT_DIR}/custom_{name}_*.png")


if __name__ == '__main__':
    main()
