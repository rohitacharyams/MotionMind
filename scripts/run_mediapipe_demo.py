"""
Demo: MediaPipe pose extraction → motion transfer to 2D animated character.

Usage:
    python scripts/run_mediapipe_demo.py --video data/input_videos/dance.mp4
    python scripts/run_mediapipe_demo.py --video data/input_videos/dance.mp4 --style cartoon
    python scripts/run_mediapipe_demo.py --video data/input_videos/dance.mp4 --style deform2d --holistic
    python scripts/run_mediapipe_demo.py --synthetic  # no video needed

Output saved to data/output_videos/mediapipe_demo.mp4
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="MediaPipe → 2D Character Motion Transfer")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic motion (no video)")
    parser.add_argument("--style", type=str, default="cartoon",
                        choices=["stick_figure", "cartoon", "puppet", "deform2d",
                                 "silhouette", "neon", "ghost", "hqchar"],
                        help="Character rendering style")
    parser.add_argument("--holistic", action="store_true",
                        help="Use MediaPipe Holistic (includes hands + face)")
    parser.add_argument("--output", type=str, default="data/output_videos/mediapipe_demo.mp4")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--side-by-side", action="store_true",
                        help="Show original video side-by-side with character")
    args = parser.parse_args()

    if not args.video and not args.synthetic:
        parser.error("Provide --video or --synthetic")

    from src.engine import DanceStudioEngine

    engine = DanceStudioEngine()

    # ── Step 1: Extract poses ──
    print("\n=== Step 1: Pose Extraction ===")
    if args.synthetic:
        info = engine.load_synthetic(n_frames=180, fps=30)
        print(f"Generated {info['n_frames']} synthetic frames at {info['fps']} FPS")
    else:
        print(f"Using MediaPipe (holistic={args.holistic})...")
        t0 = time.time()
        info = engine.load_video_mediapipe(args.video, use_holistic=args.holistic)
        elapsed = time.time() - t0
        print(f"Extracted {info['n_frames']} frames at {info['fps']:.1f} FPS "
              f"(detection rate: {info['detection_rate']:.1%}) in {elapsed:.1f}s")

    # ── Step 2: Process motion (physics + smoothing) ──
    print("\n=== Step 2: Motion Processing ===")
    engine.smoothing_method = "multi"
    engine.smoothing_window = 9
    engine.smoothing_passes = 1
    engine.butterworth_cutoff = 8.0
    engine.physics_enabled = True
    engine.velocity_clamp = True
    engine.bone_constraints = True
    engine.gravity = True
    engine.ground_plane = True

    stats = engine.process_motion()
    print(f"Jitter reduction: {stats['jitter_before']:.2f} → {stats['jitter_after']:.2f} "
          f"({stats['reduction_pct']:.1f}% reduction)")

    # ── Step 3: Render ──
    print(f"\n=== Step 3: Rendering ({args.style}) ===")
    engine.style = args.style
    engine.output_format = "landscape_720p"
    engine.motion_trail = False

    # Render full video
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kps = engine._get_kps()
    scores = engine._scores
    fps = engine._fps

    from src.avatar.renderer import AvatarRenderer
    from src.scene.backgrounds import SceneComposer

    config = engine._build_render_config(args.width, args.height)
    renderer = AvatarRenderer(config)
    renderer.set_style(args.style)

    # Pre-compute transform for stable framing
    scale, offset = renderer._compute_transform(kps)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = args.width * 2 if args.side_by_side else args.width
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, args.height))

    # Open source video for side-by-side
    src_cap = None
    if args.side_by_side and args.video:
        src_cap = cv2.VideoCapture(args.video)

    scene = SceneComposer(args.width, args.height, "studio_dark")

    T = len(kps)
    for t in tqdm(range(T), desc="Rendering", unit="frame"):
        frame = renderer.render_frame(
            kps[t], scores[t] if scores is not None else None,
            position_offset=offset, scale=scale,
        )

        # Compose with background
        frame = scene.compose_frame(frame, t, T)

        if args.side_by_side and src_cap is not None:
            ret, src_frame = src_cap.read()
            if ret:
                src_resized = cv2.resize(src_frame, (args.width, args.height))
                frame = np.hstack([src_resized, frame])
            else:
                pad = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                frame = np.hstack([pad, frame])

        writer.write(frame)

    writer.release()
    if src_cap is not None:
        src_cap.release()

    print(f"\n✓ Output saved to: {args.output}")
    print(f"  {T} frames, {T/fps:.1f}s, {args.width}x{args.height}")


if __name__ == "__main__":
    main()
