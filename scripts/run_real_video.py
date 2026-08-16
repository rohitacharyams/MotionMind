"""
Process a real dance video through the full pipeline:
  1. Extract poses (RTMPose + DirectML)
  2. Apply physics constraints (gravity, ground plane, bone limits)
  3. Multi-pass smoothing (butterworth + savgol + adaptive)
  4. Render all 4 character styles
  5. Create overlay video (character on top of source)
  6. Export all outputs + comparison grid
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2


def main():
    video_path = "data/input_videos/dance_video.mp4"
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return

    os.makedirs("data/output_videos/previews", exist_ok=True)

    # ── Get video info ──
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 70)
    print("  REAL DANCE VIDEO PIPELINE — Physics + Smooth Rendering")
    print("=" * 70)
    print(f"  Source: {video_path}")
    print(f"  Resolution: {src_w}x{src_h}, FPS: {src_fps:.1f}, Frames: {total_frames}")
    print()

    # ── Step 1: Extract poses ──
    print("[1/6] Extracting poses (RTMPose + DirectML on AMD GPU)...")
    from src.pose_extraction import PoseExtractor

    config = {
        "pipeline": {"device": "auto"},
        "pose_extraction": {"mode": "performance"},
    }
    extractor = PoseExtractor(config)

    t0 = time.time()
    extraction = extractor.extract_from_video(video_path, progress=True)
    extract_time = time.time() - t0

    T = extraction["n_frames"]
    fps = extraction["fps"]
    frame_size = extraction["frame_size"]  # (W, H)
    kps_all = extraction["keypoints"]  # (T, N_persons, 133, 2)
    scores_all = extraction["scores"]  # (T, N_persons, 133)

    # Take primary person — ALL 133 keypoints (body + feet + face + hands)
    kps = kps_all[:, 0]       # (T, 133, 2)
    scores = scores_all[:, 0] # (T, 133)

    n_detected = (kps[:, :17].sum(axis=(1, 2)) != 0).sum()
    print(f"  Extracted {T} frames in {extract_time:.1f}s ({T/extract_time:.1f} fps)")
    print(f"  Person detected: {n_detected}/{T} frames")

    # Report keypoint group quality
    groups = {
        'Body (0-16)': range(0, 17), 'Feet (17-22)': range(17, 23),
        'Face (23-90)': range(23, 91), 'L.Hand (91-111)': range(91, 112),
        'R.Hand (112-132)': range(112, 133),
    }
    for name, idx_range in groups.items():
        avg_score = scores[:, list(idx_range)].mean()
        valid_pct = (scores[:, list(idx_range)] > 0.3).mean() * 100
        print(f"    {name:20s}: {valid_pct:.0f}% valid (avg score: {avg_score:.2f})")

    # ── Step 2: Apply physics constraints (body joints only) ──
    print("\n[2/6] Applying physics constraints (gravity, ground plane, bone limits)...")
    from src.motion_processing.physics import PhysicsConstraints

    physics_config = {
        "physics": {
            "gravity": True,
            "ground_plane": True,
            "bone_constraints": True,
            "velocity_clamp": True,
            "max_velocity_factor": 0.12,
            "ground_margin": 0.03,
        }
    }
    physics = PhysicsConstraints(physics_config)
    # Apply physics to body joints (first 17) — the rest follow naturally
    kps_phys = kps.copy()
    kps_phys[:, :17] = physics.apply(
        kps[:, :17], scores[:, :17], fps=fps, frame_size=frame_size
    )
    # Propagate body corrections to connected joints (hands follow wrists, feet follow ankles)
    # Compute correction deltas from body joints
    body_delta = kps_phys[:, :17] - kps[:, :17]
    # Hands follow wrist corrections
    kps_phys[:, 91:112] = kps[:, 91:112] + body_delta[:, 9:10]   # left wrist (9)
    kps_phys[:, 112:133] = kps[:, 112:133] + body_delta[:, 10:11] # right wrist (10)
    # Feet follow ankle corrections
    kps_phys[:, 17:20] = kps[:, 17:20] + body_delta[:, 15:16]    # left ankle (15)
    kps_phys[:, 20:23] = kps[:, 20:23] + body_delta[:, 16:17]    # right ankle (16)
    # Face follows nose correction
    kps_phys[:, 23:91] = kps[:, 23:91] + body_delta[:, 0:1]      # nose (0)

    print(f"  Physics applied: velocity clamped, bones constrained, ground enforced")
    print(f"  Hand/foot/face positions propagated from body corrections")

    # ── Step 3: Multi-pass smoothing (all 133 keypoints) ──
    print("\n[3/6] Smoothing motion (butterworth + savgol + adaptive)...")
    from src.motion_processing import MotionSmoother

    smooth_config = {
        "motion_processing": {
            "smoothing": {
                "enabled": True,
                "method": "multi",
                "window_size": 11,
                "poly_order": 3,
                "passes": 2,
                "adaptive": True,
                "butterworth_cutoff": 5.0,
                "fps": fps,
            }
        }
    }
    smoother = MotionSmoother(smooth_config)
    kps_smooth = smoother.smooth(kps_phys, scores)
    print(f"  Smoothed: {kps_smooth.shape} (all 133 keypoints)")

    # Compute jitter reduction (body joints)
    raw_vel = np.linalg.norm(np.diff(kps[:, :17], axis=0), axis=-1).mean()
    smooth_vel = np.linalg.norm(np.diff(kps_smooth[:, :17], axis=0), axis=-1).mean()
    print(f"  Avg frame-to-frame displacement: {raw_vel:.2f}px (raw) -> {smooth_vel:.2f}px (smooth)")
    print(f"  Jitter reduction: {(1 - smooth_vel/raw_vel)*100:.1f}%")

    # ── Step 4: Render all character styles (full 133 keypoints) ──
    print("\n[4/6] Rendering character animations (4 styles, full wholebody)...")
    from src.avatar.renderer import AvatarRenderer

    canvas_w, canvas_h = 640, 480

    render_config = {
        "avatar": {
            "skeleton_type": "wholebody_133",
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "background_color": [10, 10, 18],
            "fps": fps,
            "styles": {
                "stick_figure": {
                    "joint_color": [255, 255, 255], "bone_color": [0, 200, 255],
                    "hand_color": [180, 255, 180], "face_color": [255, 200, 200],
                    "foot_color": [200, 200, 255],
                    "joint_radius": 5, "bone_width": 3,
                },
                "silhouette": {
                    "fill_color": [240, 240, 240], "outline_color": [0, 180, 255],
                    "outline_width": 2,
                },
                "neon": {
                    "glow_color": [0, 255, 200], "core_color": [255, 255, 255],
                    "hand_glow": [255, 180, 0], "face_glow": [200, 100, 255],
                    "glow_radius": 12, "bone_width": 3,
                },
                "cartoon": {
                    "skin_color": [220, 200, 175], "outline_color": [40, 40, 40],
                    "outfit_color": [100, 149, 237], "shoe_color": [50, 50, 60],
                    "outline_width": 2,
                },
            },
        },
    }

    styles = ["stick_figure", "silhouette", "neon", "cartoon"]
    style_frames = {}

    for style in styles:
        t0 = time.time()
        renderer = AvatarRenderer(render_config)
        renderer.set_style(style)
        frames = renderer.render_sequence(
            kps_smooth, scores,
            center_character=True,
            motion_trail=(style == "neon"),
            trail_length=4,
            trail_opacity_decay=0.5,
        )
        # Light effects
        style_frames[style] = frames
        render_time = time.time() - t0
        print(f"  {style:15s}: {len(frames)} frames in {render_time:.1f}s")

    # ── Step 5: Export everything ──
    print("\n[5/6] Exporting videos...")
    from src.video.exporter import VideoExporter

    export_config = {"video_output": {"codec": "libx264", "quality": 18, "output_dir": "data/output_videos"}}
    exporter = VideoExporter(export_config)

    output_paths = {}
    for style, frames in style_frames.items():
        out_name = f"real_{style}.mp4"
        path = exporter.export_opencv(frames, out_name, fps=fps)
        size_kb = os.path.getsize(path) / 1024
        output_paths[style] = path
        print(f"  {style:15s}: {path} ({size_kb:.0f} KB)")

    # ── Comparison grid ──
    print("\n[6/6] Creating comparison grid...")
    grid_frames = []
    n = min(len(f) for f in style_frames.values())
    for i in range(n):
        top = np.hstack([
            cv2.resize(style_frames["stick_figure"][i], (canvas_w, canvas_h)),
            cv2.resize(style_frames["silhouette"][i], (canvas_w, canvas_h)),
        ])
        bottom = np.hstack([
            cv2.resize(style_frames["neon"][i], (canvas_w, canvas_h)),
            cv2.resize(style_frames["cartoon"][i], (canvas_w, canvas_h)),
        ])
        grid = np.vstack([top, bottom])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            ("Stick Figure", (10, 25)), ("Silhouette", (canvas_w + 10, 25)),
            ("Neon Glow", (10, canvas_h + 25)), ("Cartoon", (canvas_w + 10, canvas_h + 25)),
        ]
        for text, pos in labels:
            cv2.putText(grid, text, pos, font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(grid, text, pos, font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        grid_frames.append(grid)

    grid_path = exporter.export_opencv(grid_frames, "real_comparison_grid.mp4", fps=fps)
    grid_size = os.path.getsize(grid_path) / 1024
    print(f"  {'comparison':15s}: {grid_path} ({grid_size:.0f} KB)")

    # ── Save preview frames ──
    print("\n  Saving preview frames...")
    preview_dir = "data/output_videos/previews"
    mid_frame = min(T // 3, n - 1)  # Use 1/3 of video for preview (usually an active pose)

    # Source frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, src_frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(f"{preview_dir}/real_source.png", src_frame)

    # Style previews
    for style, frames in style_frames.items():
        if mid_frame < len(frames):
            cv2.imwrite(f"{preview_dir}/real_{style}.png", frames[mid_frame])

    # Grid preview
    if mid_frame < len(grid_frames):
        cv2.imwrite(f"{preview_dir}/real_grid.png", grid_frames[mid_frame])

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    for name, path in output_paths.items():
        print(f"  {name:15s} -> {path}")
    print(f"  {'comparison':15s} -> {grid_path}")
    print(f"\n  Pose extraction: {extract_time:.1f}s ({T/extract_time:.1f} fps)")
    print(f"  Jitter reduction: {(1 - smooth_vel/raw_vel)*100:.1f}%")
    print(f"  Physics: gravity + ground plane + bone constraints + velocity clamp")
    print("=" * 70)

    # ── Step 7: Generate social media reels ──
    print("\n[7/7] Generating social media reels (9:16 vertical)...")
    from src.scene.reel_composer import ReelComposer
    from src.avatar.character_rigs import CharacterRig

    reel_dir = "data/output_videos/reels"
    os.makedirs(reel_dir, exist_ok=True)

    reel_configs = [
        {"style": "neon",         "rig": "dancer_female",  "bg": "neon_club",    "name": "neon_club"},
        {"style": "cartoon",      "rig": "dancer_female",  "bg": "dance_studio", "name": "cartoon_studio"},
        {"style": "silhouette",   "rig": "shadow_dancer",  "bg": "studio_dark",  "name": "shadow_dark"},
        {"style": "stick_figure", "rig": "robot",          "bg": "sunset_orange", "name": "robot_sunset"},
        {"style": "neon",         "rig": "chibi",          "bg": "pink_pop",     "name": "chibi_pop"},
    ]

    composer = ReelComposer()
    for rc in reel_configs:
        t0 = time.time()
        out_path = f"{reel_dir}/reel_{rc['name']}.mp4"
        try:
            composer.compose_reel(
                kps_smooth, scores,
                style=rc["style"],
                rig_name=rc["rig"],
                bg_preset=rc["bg"],
                output_path=out_path,
                fps=fps,
                watermark="@studioOs",
                motion_trail=(rc["style"] == "neon"),
            )
            reel_time = time.time() - t0
            reel_size = os.path.getsize(out_path) / 1024
            print(f"  {rc['name']:20s}: {out_path} ({reel_size:.0f} KB, {reel_time:.1f}s)")
        except Exception as e:
            print(f"  {rc['name']:20s}: FAILED - {e}")

    # Multi-style reel (cycles through all styles)
    print("  Creating multi-style reel...")
    t0 = time.time()
    multi_path = f"{reel_dir}/reel_multi_style.mp4"
    composer.compose_multi_style_reel(
        kps_smooth, scores,
        styles=["stick_figure", "neon", "cartoon", "silhouette"],
        rig_name="dancer_female",
        bg_preset="studio_dark",
        output_path=multi_path,
        fps=fps,
        segment_frames=int(fps * 3),  # switch every 3 seconds
    )
    multi_time = time.time() - t0
    multi_size = os.path.getsize(multi_path) / 1024
    print(f"  {'multi_style':20s}: {multi_path} ({multi_size:.0f} KB, {multi_time:.1f}s)")

    # Side-by-side comparison reel
    print("  Creating comparison reel...")
    t0 = time.time()
    sbs_path = f"{reel_dir}/reel_comparison.mp4"
    composer.compose_side_by_side_reel(
        kps_smooth, scores,
        left_style="stick_figure",
        right_style="neon",
        bg_preset="studio_dark",
        output_path=sbs_path,
        fps=fps,
    )
    sbs_time = time.time() - t0
    sbs_size = os.path.getsize(sbs_path) / 1024
    print(f"  {'comparison':20s}: {sbs_path} ({sbs_size:.0f} KB, {sbs_time:.1f}s)")

    # Final summary
    print("\n" + "=" * 70)
    print("  ALL OUTPUTS")
    print("=" * 70)
    print("  Rendered videos (landscape):")
    for name, path in output_paths.items():
        print(f"    {name:15s} -> {path}")
    print(f"    {'comparison':15s} -> {grid_path}")
    print(f"\n  Social media reels (9:16 vertical, 1080x1920):")
    for rc in reel_configs:
        print(f"    {rc['name']:20s} -> {reel_dir}/reel_{rc['name']}.mp4")
    print(f"    {'multi_style':20s} -> {multi_path}")
    print(f"    {'comparison':20s} -> {sbs_path}")
    print(f"\n  Character rigs available: {[c['id'] for c in CharacterRig.list_characters()]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
