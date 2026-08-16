"""
Demo script — generates a synthetic dance animation to test the
rendering pipeline without needing MMPose or real video input.

This creates fake keypoint data (a simple dancing stick figure)
and renders it in all available styles.

Usage:
    python scripts/demo.py
    python scripts/demo.py --style neon --frames 120 --trail
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.avatar.renderer import AvatarRenderer
from src.video.exporter import VideoExporter
from src.video.effects import VideoEffects


def generate_dance_keypoints(n_frames: int = 120, fps: float = 30.0) -> np.ndarray:
    """Generate synthetic dancing keypoints (133 wholebody joints).
    
    Creates a full COCO-WholeBody dance animation:
    - 17 body joints with dance motion
    - 6 foot joints tracking ankles
    - 68 face landmarks (jaw, eyes, brows, nose, mouth)
    - 21 left hand joints (fingers)
    - 21 right hand joints (fingers)
    """
    t = np.linspace(0, n_frames / fps * 2 * np.pi, n_frames)
    keypoints = np.zeros((n_frames, 133, 2), dtype=np.float32)

    # Base body proportions (hip-centered, normalized)
    base_body = np.array([
        [0.0, -0.70],    # 0: nose
        [-0.04, -0.74],  # 1: left_eye
        [0.04, -0.74],   # 2: right_eye
        [-0.08, -0.70],  # 3: left_ear
        [0.08, -0.70],   # 4: right_ear
        [-0.18, -0.45],  # 5: left_shoulder
        [0.18, -0.45],   # 6: right_shoulder
        [-0.32, -0.15],  # 7: left_elbow
        [0.32, -0.15],   # 8: right_elbow
        [-0.38, 0.10],   # 9: left_wrist
        [0.38, 0.10],    # 10: right_wrist
        [-0.10, 0.00],   # 11: left_hip
        [0.10, 0.00],    # 12: right_hip
        [-0.13, 0.35],   # 13: left_knee
        [0.13, 0.35],    # 14: right_knee
        [-0.13, 0.70],   # 15: left_ankle
        [0.13, 0.70],    # 16: right_ankle
    ], dtype=np.float32)

    # Base foot offsets from ankles
    base_feet = np.array([
        [-0.05, 0.74],  # 17: left_big_toe
        [0.03, 0.74],   # 18: left_small_toe
        [-0.01, 0.72],  # 19: left_heel
        [0.05, 0.74],   # 20: right_big_toe
        [0.17, 0.74],   # 21: right_small_toe
        [0.11, 0.72],   # 22: right_heel
    ], dtype=np.float32)

    # Hand finger offsets (21 keypoints per hand, relative to wrist)
    # Layout: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)
    def _make_hand_offsets(side: float):
        """Create hand keypoints. side=-1 for left, +1 for right."""
        s = 0.015  # finger spacing
        l = 0.025  # finger segment length
        offsets = np.zeros((21, 2), dtype=np.float32)
        offsets[0] = [0, 0]  # wrist
        # Thumb
        for j in range(4):
            offsets[1 + j] = [side * (s * 2 + l * j * 0.7), -l * (j + 1) * 0.8]
        # Index
        for j in range(4):
            offsets[5 + j] = [side * s, -l * (j + 1) * 1.1]
        # Middle
        for j in range(4):
            offsets[9 + j] = [0, -l * (j + 1) * 1.2]
        # Ring
        for j in range(4):
            offsets[13 + j] = [-side * s, -l * (j + 1) * 1.0]
        # Pinky
        for j in range(4):
            offsets[17 + j] = [-side * s * 2, -l * (j + 1) * 0.85]
        return offsets

    left_hand_base = _make_hand_offsets(-1.0)
    right_hand_base = _make_hand_offsets(1.0)

    # Face landmarks (68 points, relative to nose)
    def _make_face_offsets():
        """Create 68 face landmark offsets relative to nose."""
        offsets = np.zeros((68, 2), dtype=np.float32)
        r = 0.06  # face radius
        # Jaw contour (0-16): arc from left ear to right ear
        for i in range(17):
            angle = np.pi * 0.15 + (np.pi * 0.7) * i / 16
            offsets[i] = [r * 1.3 * np.cos(angle), r * 1.4 * np.sin(angle) + r * 0.3]
        # Left eyebrow (17-21)
        for i in range(5):
            offsets[17 + i] = [-r * 0.6 + r * 0.3 * i / 4, -r * 0.55 - r * 0.08 * np.sin(np.pi * i / 4)]
        # Right eyebrow (22-26)
        for i in range(5):
            offsets[22 + i] = [r * 0.2 + r * 0.3 * i / 4, -r * 0.55 - r * 0.08 * np.sin(np.pi * i / 4)]
        # Nose bridge (27-30)
        for i in range(4):
            offsets[27 + i] = [0, -r * 0.35 + r * 0.2 * i]
        # Nose bottom (31-35)
        for i in range(5):
            offsets[31 + i] = [-r * 0.15 + r * 0.075 * i, r * 0.1]
        # Left eye (36-41)
        for i in range(6):
            a = np.pi * 2 * i / 6
            offsets[36 + i] = [-r * 0.3 + r * 0.12 * np.cos(a), -r * 0.3 + r * 0.06 * np.sin(a)]
        # Right eye (42-47)
        for i in range(6):
            a = np.pi * 2 * i / 6
            offsets[42 + i] = [r * 0.3 + r * 0.12 * np.cos(a), -r * 0.3 + r * 0.06 * np.sin(a)]
        # Outer lip (48-59)
        for i in range(12):
            a = np.pi * 2 * i / 12
            offsets[48 + i] = [r * 0.2 * np.cos(a), r * 0.4 + r * 0.08 * np.sin(a)]
        # Inner lip (60-67)
        for i in range(8):
            a = np.pi * 2 * i / 8
            offsets[60 + i] = [r * 0.12 * np.cos(a), r * 0.4 + r * 0.04 * np.sin(a)]
        return offsets

    face_offsets = _make_face_offsets()

    for i in range(n_frames):
        kps = np.zeros((133, 2), dtype=np.float32)
        body = base_body.copy()

        # ── Body animation — dramatic dance ──
        # Hip sway (large lateral + vertical bounce)
        hip_sway = 0.12 * np.sin(t[i])
        hip_bounce = 0.06 * abs(np.sin(2 * t[i]))
        body[[11, 12], 0] += hip_sway
        body[[11, 12], 1] += hip_bounce
        body[:11, 0] += hip_sway * 0.6
        body[:11, 1] += hip_bounce

        # Torso twist — shoulders counter-rotate to hips
        torso_twist = 0.06 * np.sin(t[i] + 0.3)
        body[5, 0] += torso_twist
        body[6, 0] -= torso_twist

        # Left arm — big swing (amplitude ~1.4 rad ≈ 80°)
        arm_angle_l = 1.4 * np.sin(t[i] + 0.5)
        body[7, 0] = body[5, 0] + 0.28 * np.sin(arm_angle_l - 0.3)
        body[7, 1] = body[5, 1] + 0.30 * np.cos(arm_angle_l)
        body[9, 0] = body[7, 0] + 0.22 * np.sin(arm_angle_l + 0.7)
        body[9, 1] = body[7, 1] + 0.24 * np.cos(arm_angle_l + 0.5)

        # Right arm — opposite phase, big swing
        arm_angle_r = 1.4 * np.sin(t[i] + np.pi + 0.5)
        body[8, 0] = body[6, 0] + 0.28 * np.sin(arm_angle_r + 0.3)
        body[8, 1] = body[6, 1] + 0.30 * np.cos(arm_angle_r)
        body[10, 0] = body[8, 0] + 0.22 * np.sin(arm_angle_r - 0.7)
        body[10, 1] = body[8, 1] + 0.24 * np.cos(arm_angle_r - 0.5)

        # Knee bob — stronger vertical bounce
        bob = 0.05 * np.sin(2 * t[i])
        body[:13, 1] += bob

        # Legs — wider stepping motion
        leg_l = np.sin(t[i])
        body[13, 0] = body[11, 0] + 0.10 * np.sin(leg_l)
        body[13, 1] = body[11, 1] + 0.35 + 0.04 * np.cos(t[i])
        body[15, 0] = body[13, 0] + 0.05 * np.sin(leg_l)
        body[15, 1] = body[13, 1] + 0.35 + 0.03 * np.sin(t[i])
        leg_r = np.sin(t[i] + np.pi)
        body[14, 0] = body[12, 0] + 0.10 * np.sin(leg_r)
        body[14, 1] = body[12, 1] + 0.35 + 0.04 * np.cos(t[i] + np.pi)
        body[16, 0] = body[14, 0] + 0.05 * np.sin(leg_r)
        body[16, 1] = body[14, 1] + 0.35 + 0.03 * np.sin(t[i] + np.pi)

        # Head bob — bigger
        body[0, 1] += 0.03 * np.sin(2 * t[i])
        body[0, 0] += 0.04 * np.sin(t[i] * 0.5)

        kps[:17] = body

        # ── Feet (follow ankles) ──
        feet = base_feet.copy()
        feet[0:3, 0] += body[15, 0] - base_body[15, 0]
        feet[0:3, 1] += body[15, 1] - base_body[15, 1]
        feet[3:6, 0] += body[16, 0] - base_body[16, 0]
        feet[3:6, 1] += body[16, 1] - base_body[16, 1]
        kps[17:23] = feet

        # ── Face (follow nose, with micro-expressions) ──
        nose = body[0]
        blink = 0.3 * (np.sin(t[i] * 3) > 0.95)  # occasional blink
        mouth_open = 0.01 * max(0, np.sin(t[i] * 2.5))
        face = face_offsets.copy()
        # Blink: compress eye height
        for ei in range(36, 42):
            face[ei][1] *= (1 - blink * 0.8)
        for ei in range(42, 48):
            face[ei][1] *= (1 - blink * 0.8)
        # Mouth open
        for mi in range(54, 60):
            face[mi][1] += mouth_open
        for mi in range(64, 68):
            face[mi][1] += mouth_open * 0.5
        kps[23:91] = nose + face

        # ── Hands (follow wrists, with finger curl animation) ──
        curl_l = 0.3 + 0.2 * np.sin(t[i] * 1.5)  # finger curl
        curl_r = 0.3 + 0.2 * np.sin(t[i] * 1.5 + np.pi)
        left_hand = left_hand_base.copy()
        right_hand = right_hand_base.copy()
        # Curl fingers by reducing y-extent
        for f_start in [1, 5, 9, 13, 17]:
            for j in range(4):
                left_hand[f_start + j, 1] *= (1 - curl_l * (j + 1) / 5)
                right_hand[f_start + j, 1] *= (1 - curl_r * (j + 1) / 5)
        kps[91:112] = body[9] + left_hand   # left wrist
        kps[112:133] = body[10] + right_hand  # right wrist

        keypoints[i] = kps

    return keypoints


def main():
    parser = argparse.ArgumentParser(description="Demo: synthetic dance animation")
    parser.add_argument("--style", default="all",
                       help="Style to render (stick_figure, silhouette, neon, cartoon, or 'all')")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS")
    parser.add_argument("--trail", action="store_true", help="Enable motion trail")
    parser.add_argument("--output-dir", default="data/output_videos")
    args = parser.parse_args()

    # Generate synthetic keypoints
    print("Generating synthetic dance keypoints (133 wholebody)...")
    keypoints = generate_dance_keypoints(args.frames, args.fps)
    scores = np.ones((args.frames, 133), dtype=np.float32)
    # Slightly lower confidence for face/hands (realistic)
    scores[:, 23:91] *= 0.85  # face
    scores[:, 91:133] *= 0.75  # hands

    # Default config
    config = {
        "avatar": {
            "skeleton_type": "wholebody_133",
            "canvas_width": 1920,
            "canvas_height": 1080,
            "background_color": [10, 10, 15],
            "fps": int(args.fps),
            "styles": {
                "stick_figure": {
                    "joint_color": [255, 255, 255],
                    "bone_color": [0, 200, 255],
                    "joint_radius": 8,
                    "bone_width": 4,
                },
                "silhouette": {
                    "fill_color": [255, 255, 255],
                    "outline_color": [0, 200, 255],
                    "outline_width": 3,
                },
                "neon": {
                    "glow_color": [0, 255, 200],
                    "core_color": [255, 255, 255],
                    "glow_radius": 18,
                    "bone_width": 5,
                },
                "cartoon": {
                    "skin_color": [255, 220, 185],
                    "outline_color": [40, 40, 40],
                    "outfit_color": [100, 149, 237],
                    "outline_width": 3,
                },
            },
        },
        "video_output": {
            "codec": "libx264",
            "quality": 18,
            "output_dir": args.output_dir,
            "trail_length": 5,
            "trail_opacity_decay": 0.6,
        },
    }

    styles = ["stick_figure", "silhouette", "neon", "cartoon"] if args.style == "all" else [args.style]
    exporter = VideoExporter(config)

    for style in styles:
        print(f"\nRendering style: {style}")
        renderer = AvatarRenderer(config)
        renderer.set_style(style)

        frames = renderer.render_sequence(
            keypoints, scores,
            center_character=True,
            motion_trail=args.trail,
            trail_length=5,
            trail_opacity_decay=0.6,
        )

        # Add effects
        frames = VideoEffects.fade_in_out(frames, fade_in_frames=15, fade_out_frames=15)
        frames = VideoEffects.vignette(frames, strength=0.4)

        output_name = f"demo_{style}.mp4"
        output_path = exporter.export_opencv(frames, output_name, fps=args.fps)
        print(f"  Saved: {output_path}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
