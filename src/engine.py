"""
Dance Studio Engine — central orchestrator for the creative playground.

Exposes every customization knob as a clean API that the Gradio UI
(or any other frontend) can drive.
"""

import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .pose_extraction import PoseExtractor
from .pose_extraction.extractor_mediapipe import MediaPipePoseExtractor
from .motion_processing.physics import PhysicsConstraints
from .motion_processing import MotionSmoother
from .avatar.renderer import AvatarRenderer
from .avatar.character_rigs import CharacterRig, CHARACTER_RIGS
from .avatar.characters import STYLE_REGISTRY, CHARACTER_PRESETS, list_available_models
from .scene.backgrounds import BackgroundGenerator, SceneComposer
from .scene.reel_composer import ReelComposer
from .video.effects import VideoEffects
from .video.exporter import VideoExporter
from .choreography import MotionMixer, TransitionEngine, MotionStyleTransfer


class DanceStudioEngine:
    """All-in-one engine for the Dance Motion Studio.

    Typical workflow:
        engine = DanceStudioEngine()
        engine.load_video("dance.mp4")          # or .load_synthetic(...)
        preview = engine.render_preview_frame(0) # quick preview
        video   = engine.render_video()          # full render
    """

    # ── Available options (class-level) ──────────────────────────

    STYLES = list(STYLE_REGISTRY.keys())
    RIGS = list(CHARACTER_RIGS.keys())
    BACKGROUNDS = [p["id"] for p in BackgroundGenerator.list_presets()]
    PUPPET_PRESETS = list(CHARACTER_PRESETS.keys())
    FORMATS = {
        "landscape_1080p": (1920, 1080),
        "landscape_720p": (1280, 720),
        "portrait_reel": (720, 1280),
        "portrait_hd": (1080, 1920),
        "square_1080": (1080, 1080),
        "square_720": (720, 720),
    }
    SMOOTHING_METHODS = ["savgol", "gaussian", "moving_average", "butterworth", "multi"]
    EFFECTS_LIST = ["vignette", "fade", "color_grade", "motion_blur"]

    def __init__(self):
        # Motion data
        self._keypoints: np.ndarray | None = None  # (T, K, 2)
        self._scores: np.ndarray | None = None     # (T, K)
        self._fps: float = 30.0
        self._frame_size: tuple[int, int] = (0, 0)
        self._source_path: str | None = None

        # Processed data (after physics + smoothing)
        self._kps_processed: np.ndarray | None = None

        # --- Rendering parameters (all tuneable) ---
        self.style: str = "ghost"
        self.rig_name: str = "dancer_female"
        self.bg_preset: str = "studio_dark"
        self.output_format: str = "landscape_720p"

        # Style colours (overrides)
        self.style_overrides: dict = {}

        # Puppet character preset
        self.puppet_preset: str = "casual_male"
        # 3D mesh model path
        self.mesh3d_model: str = ""

        # Background custom params
        self.bg_custom: dict = {}

        # Physics
        self.physics_enabled: bool = True
        self.gravity: bool = True
        self.ground_plane: bool = True
        self.bone_constraints: bool = True
        self.velocity_clamp: bool = True

        # Smoothing
        self.smoothing_enabled: bool = True
        self.smoothing_method: str = "multi"
        self.smoothing_window: int = 11
        self.smoothing_passes: int = 2
        self.butterworth_cutoff: float = 5.0

        # Effects
        self.motion_trail: bool = False
        self.trail_length: int = 5
        self.trail_opacity: float = 0.6
        self.vignette_strength: float = 0.0
        self.fade_frames: int = 0
        self.color_hue_shift: float = 0
        self.color_saturation: float = 1.0
        self.color_brightness: float = 1.0
        self.color_contrast: float = 1.0
        self.motion_blur_strength: int = 0

        # Watermark
        self.watermark: str = ""

        # Reel
        self.retarget: bool = False

    # ── Data Loading ─────────────────────────────────────────────

    def load_video(self, video_path: str, progress_cb=None) -> dict:
        """Extract poses from a video file.

        Returns:
            dict with keys: n_frames, fps, frame_size, detection_rate
        """
        config = {"pipeline": {"device": "auto"},
                  "pose_extraction": {"mode": "performance"}}
        extractor = PoseExtractor(config)
        result = extractor.extract_from_video(video_path, progress=True)

        self._keypoints = result["keypoints"][:, 0]   # first person
        self._scores = result["scores"][:, 0]
        self._fps = result["fps"]
        self._frame_size = result["frame_size"]
        self._source_path = video_path
        self._kps_processed = None  # invalidate

        return {
            "n_frames": len(self._keypoints),
            "fps": self._fps,
            "frame_size": self._frame_size,
            "detection_rate": float((self._scores[:, :17].max(axis=-1) > 0.3).mean()),
        }

    def load_video_mediapipe(self, video_path: str, use_holistic: bool = False) -> dict:
        """Extract poses using MediaPipe (more stable tracking)."""
        config = {"pose_extraction": {"mediapipe_complexity": 2}}
        extractor = MediaPipePoseExtractor(config, use_holistic=use_holistic)
        result = extractor.extract_from_video(video_path, progress=True)

        self._keypoints = result["keypoints"][:, 0]
        self._scores = result["scores"][:, 0]
        self._fps = result["fps"]
        self._frame_size = result["frame_size"]
        self._source_path = video_path
        self._kps_processed = None

        return {
            "n_frames": len(self._keypoints),
            "fps": self._fps,
            "frame_size": self._frame_size,
            "detection_rate": float((self._scores[:, :17].max(axis=-1) > 0.3).mean()),
        }

    def load_synthetic(self, n_frames: int = 120, fps: float = 30.0) -> dict:
        """Generate synthetic dance motion (no video needed)."""
        # Import the demo generator
        import importlib.util
        demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "scripts", "demo.py")
        spec = importlib.util.spec_from_file_location("demo", demo_path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)

        self._keypoints = demo.generate_dance_keypoints(n_frames, fps)
        self._scores = np.ones((n_frames, 133), dtype=np.float32)
        self._scores[:, 23:91] *= 0.85
        self._scores[:, 91:133] *= 0.75
        self._fps = fps
        self._frame_size = (640, 480)
        self._source_path = None
        self._kps_processed = None

        return {"n_frames": n_frames, "fps": fps, "frame_size": (640, 480)}

    def load_keypoints(self, keypoints: np.ndarray,
                       scores: np.ndarray | None = None,
                       fps: float = 30.0) -> dict:
        """Load pre-computed keypoints directly."""
        self._keypoints = keypoints
        self._scores = scores if scores is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
        self._fps = fps
        self._frame_size = (640, 480)
        self._kps_processed = None
        return {"n_frames": len(keypoints), "fps": fps}

    @property
    def has_data(self) -> bool:
        return self._keypoints is not None

    @property
    def n_frames(self) -> int:
        return len(self._keypoints) if self._keypoints is not None else 0

    # ── Processing ───────────────────────────────────────────────

    def process_motion(self) -> dict:
        """Apply physics + smoothing to current keypoints.

        Returns dict with jitter reduction stats.
        """
        if self._keypoints is None:
            raise ValueError("No motion data loaded")

        kps = self._keypoints.copy()
        scores = self._scores

        # Physics
        if self.physics_enabled:
            physics = PhysicsConstraints({
                "physics": {
                    "gravity": self.gravity,
                    "ground_plane": self.ground_plane,
                    "bone_constraints": self.bone_constraints,
                    "velocity_clamp": self.velocity_clamp,
                }
            })
            kps_phys = kps.copy()
            kps_phys[:, :17] = physics.apply(
                kps[:, :17], scores[:, :17],
                fps=self._fps, frame_size=self._frame_size
            )
            # Propagate body corrections to extremities
            body_delta = kps_phys[:, :17] - kps[:, :17]
            kps_phys[:, 91:112] += body_delta[:, 9:10]   # left hand follows wrist
            kps_phys[:, 112:133] += body_delta[:, 10:11]  # right hand follows wrist
            kps_phys[:, 17:20] += body_delta[:, 15:16]    # left foot
            kps_phys[:, 20:23] += body_delta[:, 16:17]    # right foot
            kps_phys[:, 23:91] += body_delta[:, 0:1]      # face follows nose
            kps = kps_phys

        # Smoothing — apply to ALL keypoints, not just body
        jitter_before = self._compute_jitter(kps)
        if self.smoothing_enabled:
            smoother = MotionSmoother({
                "motion_processing": {"smoothing": {
                    "enabled": True,
                    "method": self.smoothing_method,
                    "window_size": self.smoothing_window,
                    "poly_order": 3,
                    "passes": self.smoothing_passes,
                    "adaptive": True,
                    "butterworth_cutoff": self.butterworth_cutoff,
                    "fps": self._fps,
                }}
            })
            # Smooth body keypoints
            kps[:, :17] = smoother.smooth(kps[:, :17], scores[:, :17])
            # Smooth hands with extra smoothing (more jittery)
            hand_smoother = MotionSmoother({
                "motion_processing": {"smoothing": {
                    "enabled": True,
                    "method": "butterworth",
                    "window_size": 11,
                    "poly_order": 3,
                    "passes": 1,
                    "adaptive": False,
                    "butterworth_cutoff": 4.0,
                    "fps": self._fps,
                }}
            })
            if kps.shape[1] > 91:
                kps[:, 91:112] = hand_smoother.smooth(kps[:, 91:112], scores[:, 91:112] if scores is not None else None)
                kps[:, 112:133] = hand_smoother.smooth(kps[:, 112:133], scores[:, 112:133] if scores is not None else None)
            # Smooth face
            if kps.shape[1] > 23:
                kps[:, 23:91] = hand_smoother.smooth(kps[:, 23:91], scores[:, 23:91] if scores is not None else None)
            # Smooth feet
            if kps.shape[1] > 17:
                kps[:, 17:23] = smoother.smooth(kps[:, 17:23], scores[:, 17:23] if scores is not None else None)

        jitter_after = self._compute_jitter(kps)
        self._kps_processed = kps

        return {
            "jitter_before": float(jitter_before),
            "jitter_after": float(jitter_after),
            "reduction_pct": float((1 - jitter_after / max(jitter_before, 1e-8)) * 100),
        }

    def _compute_jitter(self, kps: np.ndarray) -> float:
        if len(kps) < 2:
            return 0.0
        diff = np.diff(kps[:, :17], axis=0)
        return float(np.mean(np.linalg.norm(diff, axis=-1)))

    def _get_kps(self) -> np.ndarray:
        """Get processed keypoints, running processing if needed."""
        if self._kps_processed is None:
            self.process_motion()
        return self._kps_processed

    # ── Render Config Builder ────────────────────────────────────

    def _build_render_config(self, width: int, height: int) -> dict:
        """Build the full renderer config from current settings."""
        config = {
            "avatar": {
                "canvas_width": width,
                "canvas_height": height,
                "background_color": [0, 0, 0],
                "fps": int(self._fps),
                "skeleton_type": "wholebody_133",
                "default_style": self.style,
                "styles": {},
            },
        }
        # Merge style overrides
        if self.style_overrides:
            config["avatar"]["styles"][self.style] = self.style_overrides.copy()

        # Puppet preset
        if self.style == "puppet":
            puppet_cfg = config["avatar"]["styles"].setdefault("puppet", {})
            puppet_cfg["preset"] = self.puppet_preset

        # Mesh3D model path
        if self.style == "mesh3d" and self.mesh3d_model:
            mesh_cfg = config["avatar"]["styles"].setdefault("mesh3d", {})
            mesh_cfg["model_path"] = self.mesh3d_model

        return config

    def _get_canvas_size(self) -> tuple[int, int]:
        """Get (width, height) from current output format."""
        return self.FORMATS.get(self.output_format, (1280, 720))

    # ── Preview ──────────────────────────────────────────────────

    def render_preview_frame(self, frame_idx: int = 0) -> np.ndarray:
        """Render a single preview frame (fast).

        Returns BGR numpy array.
        """
        if not self.has_data:
            raise ValueError("No motion data loaded")

        kps = self._get_kps()
        frame_idx = max(0, min(frame_idx, len(kps) - 1))

        w, h = self._get_canvas_size()
        config = self._build_render_config(w, h)

        renderer = AvatarRenderer(config)
        renderer.set_style(self.style)

        # Render single frame with auto-centering
        # We need to compute transform from whole sequence for consistency
        char_frame = renderer.render_sequence(
            kps[frame_idx:frame_idx+1],
            self._scores[frame_idx:frame_idx+1] if self._scores is not None else None,
            center_character=True,
            motion_trail=False,
        )[0]

        # Compose over background
        scene = SceneComposer(w, h, self.bg_preset)
        composed = scene.compose_frame(
            char_frame, frame_idx, len(kps),
            watermark=self.watermark or None,
        )

        return composed

    def render_preview_sequence(self, max_frames: int = 30) -> list[np.ndarray]:
        """Render a short preview clip (subsampled if needed)."""
        kps = self._get_kps()
        T = len(kps)

        # Subsample for speed
        step = max(1, T // max_frames)
        indices = list(range(0, T, step))[:max_frames]

        w, h = self._get_canvas_size()
        config = self._build_render_config(w, h)
        renderer = AvatarRenderer(config)
        renderer.set_style(self.style)

        sub_kps = kps[indices]
        sub_scores = self._scores[indices] if self._scores is not None else None

        char_frames = renderer.render_sequence(
            sub_kps, sub_scores,
            center_character=True,
            motion_trail=self.motion_trail,
            trail_length=self.trail_length,
            trail_opacity_decay=self.trail_opacity,
        )

        scene = SceneComposer(w, h, self.bg_preset)
        composed = []
        for i, cf in enumerate(char_frames):
            frame = scene.compose_frame(cf, indices[i], T,
                                        watermark=self.watermark or None)
            composed.append(frame)

        return composed

    # ── Full Render ──────────────────────────────────────────────

    def render_video(self, output_path: str | None = None) -> str:
        """Render the full video with all current settings.

        Returns path to output video file.
        """
        if not self.has_data:
            raise ValueError("No motion data loaded")

        kps = self._get_kps()
        T = len(kps)
        w, h = self._get_canvas_size()

        config = self._build_render_config(w, h)
        renderer = AvatarRenderer(config)
        renderer.set_style(self.style)

        # Render character frames
        char_frames = renderer.render_sequence(
            kps, self._scores,
            center_character=True,
            motion_trail=self.motion_trail,
            trail_length=self.trail_length,
            trail_opacity_decay=self.trail_opacity,
        )

        # Compose over background
        scene = SceneComposer(w, h, self.bg_preset)
        composed = []
        for i, cf in enumerate(char_frames):
            frame = scene.compose_frame(cf, i, T,
                                        watermark=self.watermark or None)
            composed.append(frame)

        # Apply post-processing effects
        composed = self._apply_effects(composed)

        # Export
        if output_path is None:
            os.makedirs("data/output_videos/studio", exist_ok=True)
            ts = int(time.time())
            output_path = f"data/output_videos/studio/render_{self.style}_{ts}.mp4"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (w, h))
        for f in composed:
            writer.write(f)
        writer.release()

        return output_path

    def render_reel(self, output_path: str | None = None) -> str:
        """Render a social-media reel with current settings."""
        if not self.has_data:
            raise ValueError("No motion data loaded")

        kps = self._get_kps()
        if output_path is None:
            os.makedirs("data/output_videos/studio", exist_ok=True)
            ts = int(time.time())
            output_path = f"data/output_videos/studio/reel_{self.style}_{ts}.mp4"

        composer = ReelComposer()
        w, h = self._get_canvas_size()
        composer.REEL_WIDTH = w
        composer.REEL_HEIGHT = h

        composer.compose_reel(
            kps, self._scores,
            style=self.style,
            rig_name=self.rig_name,
            bg_preset=self.bg_preset,
            output_path=output_path,
            fps=self._fps,
            watermark=self.watermark or None,
            motion_trail=self.motion_trail,
            retarget=self.retarget,
        )
        return output_path

    def render_comparison(self, styles: list[str],
                          output_path: str | None = None) -> str:
        """Render a multi-style comparison grid."""
        if not self.has_data:
            raise ValueError("No motion data loaded")

        kps = self._get_kps()
        T = len(kps)
        w, h = self._get_canvas_size()

        n_styles = len(styles)
        if n_styles <= 2:
            cols, rows = n_styles, 1
        elif n_styles <= 4:
            cols, rows = 2, 2
        else:
            cols = 3
            rows = (n_styles + cols - 1) // cols

        cell_w = w // cols
        cell_h = h // rows

        # Render each style
        style_frames_list = []
        for s in styles:
            cfg = self._build_render_config(cell_w, cell_h)
            renderer = AvatarRenderer(cfg)
            renderer.set_style(s)
            sf = renderer.render_sequence(kps, self._scores, center_character=True)
            style_frames_list.append(sf)

        scene = SceneComposer(cell_w, cell_h, self.bg_preset)

        if output_path is None:
            os.makedirs("data/output_videos/studio", exist_ok=True)
            ts = int(time.time())
            output_path = f"data/output_videos/studio/comparison_{ts}.mp4"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (w, h))

        for fi in range(T):
            grid = np.zeros((h, w, 3), dtype=np.uint8)
            for si, s in enumerate(styles):
                r, c = divmod(si, cols)
                cf = style_frames_list[si][fi]
                composed = scene.compose_frame(cf, fi, T)
                y0 = r * cell_h
                x0 = c * cell_w
                grid[y0:y0+cell_h, x0:x0+cell_w] = composed

                # Label
                cv2.putText(grid, s, (x0 + 10, y0 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
            writer.write(grid)

        writer.release()
        return output_path

    # ── Effects Pipeline ─────────────────────────────────────────

    def _apply_effects(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply post-processing effects to rendered frames."""
        if self.motion_blur_strength > 0:
            frames = VideoEffects.motion_blur(frames, self.motion_blur_strength)

        if (self.color_hue_shift != 0 or self.color_saturation != 1.0
                or self.color_brightness != 1.0 or self.color_contrast != 1.0):
            frames = VideoEffects.color_grade(
                frames,
                hue_shift=self.color_hue_shift,
                saturation=self.color_saturation,
                brightness=self.color_brightness,
                contrast=self.color_contrast,
            )

        if self.vignette_strength > 0:
            frames = VideoEffects.vignette(frames, self.vignette_strength)

        if self.fade_frames > 0:
            frames = VideoEffects.fade_in_out(frames, self.fade_frames, self.fade_frames)

        return frames

    # ── Utility ──────────────────────────────────────────────────

    def get_state_summary(self) -> dict:
        """Return a summary of all current engine settings."""
        return {
            "has_data": self.has_data,
            "n_frames": self.n_frames,
            "fps": self._fps,
            "source": self._source_path,
            "style": self.style,
            "rig": self.rig_name,
            "background": self.bg_preset,
            "format": self.output_format,
            "physics": self.physics_enabled,
            "smoothing": self.smoothing_enabled,
            "motion_trail": self.motion_trail,
            "watermark": self.watermark,
        }
