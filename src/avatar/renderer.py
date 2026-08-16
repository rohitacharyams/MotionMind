"""
Avatar renderer — orchestrates skeleton + character style rendering.

Takes normalized motion data and renders animated frames using the
selected character style, with optional motion trails and effects.
"""

import cv2
import numpy as np
from .skeleton import Skeleton2D
from .characters import STYLE_REGISTRY


class AvatarRenderer:
    """Render animated character frames from motion data."""

    def __init__(self, config: dict):
        self.config = config
        avatar_cfg = config.get("avatar", {})

        self.canvas_w = avatar_cfg.get("canvas_width", 1920)
        self.canvas_h = avatar_cfg.get("canvas_height", 1080)
        self.bg_color = tuple(avatar_cfg.get("background_color", [0, 0, 0]))
        self.fps = avatar_cfg.get("fps", 30)
        self.antialiasing = avatar_cfg.get("antialiasing", True)

        skeleton_type = avatar_cfg.get("skeleton_type", "body_17")
        self.skeleton = Skeleton2D(skeleton_type)

        default_style = avatar_cfg.get("default_style", "stick_figure")
        self.set_style(default_style)

        # Motion trail state
        self._trail_buffer = []

    def set_style(self, style_name: str):
        """Switch character rendering style."""
        if style_name not in STYLE_REGISTRY:
            raise ValueError(
                f"Unknown style '{style_name}'. Available: {list(STYLE_REGISTRY.keys())}"
            )
        self.style = STYLE_REGISTRY[style_name](self.config)
        self.style_name = style_name

    def render_frame(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        canvas: np.ndarray | None = None,
        position_offset: np.ndarray | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Render a single frame with the character.
        
        Args:
            keypoints: (K, 2) keypoints (normalized or pixel coords).
            scores: (K,) confidence scores.
            canvas: Optional pre-existing canvas. Created if None.
            position_offset: (2,) offset to apply to all keypoints.
            scale: Scale factor for the character.
            
        Returns:
            (H, W, 3) rendered frame.
        """
        if canvas is None:
            canvas = np.full(
                (self.canvas_h, self.canvas_w, 3),
                self.bg_color, dtype=np.uint8
            )

        # Apply transforms
        kps = keypoints.copy()
        if scale != 1.0:
            kps *= scale
        if position_offset is not None:
            kps += position_offset

        # Replace NaN/inf with 0 and suppress those joints
        nan_mask = ~np.isfinite(kps)
        if nan_mask.any():
            kps[nan_mask] = 0.0
            if scores is not None:
                scores = scores.copy()
                bad_joints = nan_mask.any(axis=-1)
                scores[bad_joints] = 0.0

        # Update skeleton
        self.skeleton.set_pose(kps)

        # Render character
        canvas = self.style.render(canvas, kps, scores)

        return canvas

    def render_sequence(
        self,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        center_character: bool = True,
        scale: float | None = None,
        motion_trail: bool = False,
        trail_length: int = 5,
        trail_opacity_decay: float = 0.7,
    ) -> list[np.ndarray]:
        """Render a full motion sequence.
        
        Args:
            keypoints_seq: (T, K, 2) motion sequence.
            scores_seq: (T, K) confidence scores.
            center_character: Auto-center the character on canvas.
            scale: Scale factor. Auto-computed if None.
            motion_trail: Enable motion trail effect.
            trail_length: Number of trail frames.
            trail_opacity_decay: Opacity decay per trail frame.
            
        Returns:
            List of (H, W, 3) rendered frames.
        """
        T, K, _ = keypoints_seq.shape

        # Compute auto-scale and offset
        if center_character or scale is None:
            scale, offset = self._compute_transform(keypoints_seq)
        else:
            offset = np.array([self.canvas_w / 2, self.canvas_h / 2])

        frames = []
        self._trail_buffer = []

        for t in range(T):
            kps = keypoints_seq[t]
            scores_t = scores_seq[t] if scores_seq is not None else None

            canvas = np.full(
                (self.canvas_h, self.canvas_w, 3),
                self.bg_color, dtype=np.uint8
            )

            # Draw motion trail
            if motion_trail and len(self._trail_buffer) > 0:
                for i, (trail_kps, trail_scores) in enumerate(self._trail_buffer):
                    opacity = trail_opacity_decay ** (len(self._trail_buffer) - i)
                    trail_canvas = np.zeros_like(canvas)
                    trail_canvas = self.style.render(
                        trail_canvas,
                        trail_kps * scale + offset,
                        trail_scores,
                    )
                    canvas = cv2.addWeighted(canvas, 1.0, trail_canvas, opacity, 0)

            # Render current frame
            canvas = self.render_frame(
                kps, scores_t, canvas,
                position_offset=offset, scale=scale,
            )

            frames.append(canvas)

            # Update trail buffer
            if motion_trail:
                self._trail_buffer.append((kps.copy(), scores_t))
                if len(self._trail_buffer) > trail_length:
                    self._trail_buffer.pop(0)

        return frames

    def _compute_transform(
        self, keypoints_seq: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Compute scale and offset to center character on canvas.
        
        Returns:
            (scale, offset) tuple.
        """
        # Find bounding box across all frames (exclude NaN and zero)
        flat = keypoints_seq.reshape(-1, 2)
        finite_mask = np.isfinite(flat).all(axis=-1) & (flat.sum(axis=-1) != 0)
        valid = flat[finite_mask]
        if len(valid) == 0:
            return 1.0, np.array([self.canvas_w / 2, self.canvas_h / 2])

        min_xy = valid.min(axis=0)
        max_xy = valid.max(axis=0)
        bbox_size = max_xy - min_xy
        bbox_center = (min_xy + max_xy) / 2

        # Scale to fit canvas with margin
        margin = 0.15
        available = np.array([
            self.canvas_w * (1 - 2 * margin),
            self.canvas_h * (1 - 2 * margin),
        ])
        
        if bbox_size[0] > 0 and bbox_size[1] > 0:
            scale = min(available[0] / bbox_size[0], available[1] / bbox_size[1])
        else:
            scale = 1.0

        canvas_center = np.array([self.canvas_w / 2, self.canvas_h / 2])
        offset = canvas_center - bbox_center * scale

        return scale, offset
