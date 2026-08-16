"""
Video composer — arrange multiple animated characters and elements
into a final video composition.

Supports:
- Single character on background
- Multi-character compositions (side by side, overlapping)
- Split screen layouts
- Picture-in-picture with source video
"""

import cv2
import numpy as np
from ..avatar.renderer import AvatarRenderer


class VideoComposer:
    """Compose animated character video from rendered frames."""

    def __init__(self, config: dict):
        self.config = config
        self.canvas_w = config.get("avatar", {}).get("canvas_width", 1920)
        self.canvas_h = config.get("avatar", {}).get("canvas_height", 1080)
        self.bg_color = tuple(config.get("avatar", {}).get("background_color", [0, 0, 0]))
        self.fps = config.get("avatar", {}).get("fps", 30)

    def compose_single(
        self,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        style: str = "stick_figure",
        background: np.ndarray | None = None,
        motion_trail: bool = False,
    ) -> list[np.ndarray]:
        """Compose video with a single character.
        
        Args:
            keypoints_seq: (T, K, 2) normalized motion.
            scores_seq: (T, K) confidence scores.
            style: Character style name.
            background: Optional (H, W, 3) background image.
            motion_trail: Enable motion trail effect.
            
        Returns:
            List of rendered frames.
        """
        renderer = AvatarRenderer(self.config)
        renderer.set_style(style)

        trail_cfg = self.config.get("video_output", {})

        return renderer.render_sequence(
            keypoints_seq,
            scores_seq,
            center_character=True,
            motion_trail=motion_trail,
            trail_length=trail_cfg.get("trail_length", 5),
            trail_opacity_decay=trail_cfg.get("trail_opacity_decay", 0.7),
        )

    def compose_multi_character(
        self,
        characters: list[dict],
        layout: str = "side_by_side",
    ) -> list[np.ndarray]:
        """Compose video with multiple characters.
        
        Args:
            characters: List of dicts with keys:
                'keypoints': (T, K, 2)
                'scores': (T, K) optional
                'style': str
                'label': str optional
            layout: 'side_by_side', 'overlapping', 'grid'
            
        Returns:
            List of rendered frames.
        """
        if not characters:
            raise ValueError("No characters to compose")

        # Find max frame count
        max_T = max(c["keypoints"].shape[0] for c in characters)
        n_chars = len(characters)

        # Compute positions per layout
        positions = self._compute_layout(n_chars, layout)

        frames = []
        renderers = []
        for char in characters:
            renderer = AvatarRenderer(self.config)
            renderer.set_style(char.get("style", "stick_figure"))
            renderers.append(renderer)

        for t in range(max_T):
            canvas = np.full(
                (self.canvas_h, self.canvas_w, 3),
                self.bg_color, dtype=np.uint8
            )

            for i, (char, renderer) in enumerate(zip(characters, renderers)):
                kps = char["keypoints"]
                scores = char.get("scores")

                # Loop if clip is shorter
                frame_idx = t % kps.shape[0]

                offset, scale = positions[i]

                kps_frame = kps[frame_idx]
                scores_frame = scores[frame_idx] if scores is not None else None

                canvas = renderer.render_frame(
                    kps_frame, scores_frame, canvas,
                    position_offset=offset, scale=scale,
                )

            frames.append(canvas)

        return frames

    def compose_with_source(
        self,
        source_video_path: str,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        style: str = "neon",
        overlay_opacity: float = 0.6,
        pip_mode: bool = False,
    ) -> list[np.ndarray]:
        """Compose animated overlay on the original source video.
        
        Args:
            source_video_path: Path to original video.
            keypoints_seq: (T, K, 2) keypoints in original pixel coords.
            scores_seq: (T, K) scores.
            style: Character style for overlay.
            overlay_opacity: Opacity of the character overlay.
            pip_mode: If True, show character in corner instead of overlay.
            
        Returns:
            List of composed frames.
        """
        from ..avatar.characters import STYLE_REGISTRY

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source_video_path}")

        style_renderer = STYLE_REGISTRY[style](self.config)
        frames = []

        T = keypoints_seq.shape[0]
        for t in range(T):
            ret, frame = cap.read()
            if not ret:
                break

            kps = keypoints_seq[t]
            scores_t = scores_seq[t] if scores_seq is not None else None

            if pip_mode:
                # Render character in small corner window
                pip_size = (self.canvas_w // 4, self.canvas_h // 4)
                pip_canvas = np.full(
                    (pip_size[1], pip_size[0], 3), self.bg_color, dtype=np.uint8
                )
                # Scale keypoints to PIP size
                h, w = frame.shape[:2]
                scale_x = pip_size[0] / w
                scale_y = pip_size[1] / h
                kps_pip = kps.copy()
                kps_pip[:, 0] *= scale_x
                kps_pip[:, 1] *= scale_y
                pip_canvas = style_renderer.render(pip_canvas, kps_pip, scores_t)

                # Place PIP in corner
                frame_resized = cv2.resize(frame, (self.canvas_w, self.canvas_h))
                y_off = self.canvas_h - pip_size[1] - 20
                x_off = self.canvas_w - pip_size[0] - 20
                frame_resized[y_off:y_off + pip_size[1], x_off:x_off + pip_size[0]] = pip_canvas
                frames.append(frame_resized)
            else:
                # Overlay on frame
                overlay = np.zeros_like(frame)
                overlay = style_renderer.render(overlay, kps, scores_t)

                composed = cv2.addWeighted(
                    frame, 1.0, overlay, overlay_opacity, 0
                )
                frames.append(cv2.resize(composed, (self.canvas_w, self.canvas_h)))

        cap.release()
        return frames

    def _compute_layout(
        self, n_chars: int, layout: str
    ) -> list[tuple[np.ndarray, float]]:
        """Compute positions and scales for each character.
        
        Returns:
            List of (offset, scale) tuples.
        """
        positions = []

        if layout == "side_by_side":
            segment_w = self.canvas_w / n_chars
            for i in range(n_chars):
                offset = np.array([segment_w * (i + 0.5), self.canvas_h * 0.5])
                scale = 0.8 / n_chars
                positions.append((offset, scale))

        elif layout == "overlapping":
            center = np.array([self.canvas_w / 2, self.canvas_h / 2])
            spread = self.canvas_w * 0.2
            for i in range(n_chars):
                x_off = (i - n_chars / 2) * spread / n_chars
                offset = center + np.array([x_off, 0])
                scale = 0.6
                positions.append((offset, scale))

        elif layout == "grid":
            cols = int(np.ceil(np.sqrt(n_chars)))
            rows = int(np.ceil(n_chars / cols))
            cell_w = self.canvas_w / cols
            cell_h = self.canvas_h / rows
            for i in range(n_chars):
                r, c = divmod(i, cols)
                offset = np.array([cell_w * (c + 0.5), cell_h * (r + 0.5)])
                scale = 0.7 / max(cols, rows)
                positions.append((offset, scale))

        else:
            # Default: center
            center = np.array([self.canvas_w / 2, self.canvas_h / 2])
            for _ in range(n_chars):
                positions.append((center, 0.5))

        return positions
