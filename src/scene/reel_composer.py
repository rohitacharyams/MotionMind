"""
Reel composer — end-to-end pipeline for creating social media reels.

Composes dance motion + character rig + background + effects + audio
into vertical 9:16 reels ready for Instagram/TikTok/YouTube Shorts.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from ..avatar.renderer import AvatarRenderer
from ..avatar.character_rigs import CharacterRig, CHARACTER_RIGS
from ..scene.backgrounds import BackgroundGenerator, SceneComposer

logger = logging.getLogger(__name__)


class ReelComposer:
    """Compose dance reels from motion data."""

    # Standard reel format
    REEL_WIDTH = 1080
    REEL_HEIGHT = 1920

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def compose_reel(
        self,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        style: str = "neon",
        rig_name: str = "dancer_female",
        bg_preset: str = "studio_dark",
        output_path: str = "reel_output.mp4",
        fps: float = 30.0,
        audio_path: str | None = None,
        watermark: str | None = None,
        motion_trail: bool = False,
        retarget: bool = True,
    ) -> str:
        """Create a complete reel from motion data.

        Args:
            keypoints_seq: (T, K, 2) motion keypoints.
            scores_seq: (T, K) confidence scores.
            style: Character render style (stick_figure/silhouette/neon/cartoon).
            rig_name: Character rig to use.
            bg_preset: Background preset name.
            output_path: Output video file path.
            fps: Frame rate.
            audio_path: Optional audio file to mux.
            watermark: Optional watermark text.
            motion_trail: Enable motion trail.
            retarget: Apply character rig proportions.

        Returns:
            Path to the output video file.
        """
        T, K, _ = keypoints_seq.shape
        logger.info(
            "Composing reel: %d frames, style=%s, rig=%s, bg=%s",
            T, style, rig_name, bg_preset,
        )

        # Prepare rig
        rig = CharacterRig(rig_name)

        # Retarget to character proportions
        if retarget:
            keypoints_seq = rig.retarget_motion(keypoints_seq)

        # Build reel-format renderer config
        reel_config = self._reel_render_config(style)
        renderer = AvatarRenderer(reel_config)
        renderer.set_style(style)

        # Scene composer for background
        scene = SceneComposer(
            self.REEL_WIDTH, self.REEL_HEIGHT, bg_preset
        )

        # Render character frames
        char_frames = renderer.render_sequence(
            keypoints_seq,
            scores_seq,
            center_character=True,
            motion_trail=motion_trail,
        )

        # Compose character over background
        composed = []
        for i, cf in enumerate(char_frames):
            frame = scene.compose_frame(
                cf,
                frame_idx=i,
                total_frames=T,
                watermark=watermark,
            )
            composed.append(frame)

        # Encode video
        self._write_video(composed, output_path, fps)

        # Mux audio if provided
        if audio_path:
            self._mux_audio(output_path, audio_path)

        logger.info("Reel saved: %s (%d frames @ %.1f fps)", output_path, T, fps)
        return output_path

    def compose_multi_style_reel(
        self,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        styles: list[str] | None = None,
        rig_name: str = "dancer_female",
        bg_preset: str = "studio_dark",
        output_path: str = "multi_style_reel.mp4",
        fps: float = 30.0,
        segment_frames: int = 90,
    ) -> str:
        """Create a reel that switches between character styles.

        Splits the motion into segments, rendering each segment
        in a different style with smooth transitions.
        """
        if styles is None:
            styles = ["stick_figure", "neon", "cartoon", "silhouette"]

        T = keypoints_seq.shape[0]
        kps = keypoints_seq

        scene = SceneComposer(self.REEL_WIDTH, self.REEL_HEIGHT, bg_preset)
        transition_frames = min(10, segment_frames // 4)

        all_frames = []
        seg_start = 0
        style_idx = 0

        while seg_start < T:
            seg_end = min(seg_start + segment_frames, T)
            s = styles[style_idx % len(styles)]

            reel_config = self._reel_render_config(s)
            renderer = AvatarRenderer(reel_config)
            renderer.set_style(s)

            seg_kps = kps[seg_start:seg_end]
            seg_scores = scores_seq[seg_start:seg_end] if scores_seq is not None else None

            char_frames = renderer.render_sequence(
                seg_kps, seg_scores, center_character=True
            )

            for i, cf in enumerate(char_frames):
                frame = scene.compose_frame(cf, seg_start + i, T)
                all_frames.append(frame)

            seg_start = seg_end
            style_idx += 1

        # Add cross-fade transitions between segments
        final = self._add_transitions(all_frames, segment_frames, transition_frames)

        self._write_video(final, output_path, fps)
        logger.info("Multi-style reel: %s (%d frames)", output_path, len(final))
        return output_path

    def compose_side_by_side_reel(
        self,
        keypoints_seq: np.ndarray,
        scores_seq: np.ndarray | None = None,
        left_style: str = "stick_figure",
        right_style: str = "neon",
        bg_preset: str = "studio_dark",
        output_path: str = "comparison_reel.mp4",
        fps: float = 30.0,
    ) -> str:
        """Create a split-screen reel comparing two styles."""
        T = keypoints_seq.shape[0]
        half_w = self.REEL_WIDTH // 2

        frames = []
        for s, style in enumerate([left_style, right_style]):
            cfg = self._reel_render_config(style)
            cfg["avatar"]["canvas_width"] = half_w
            renderer = AvatarRenderer(cfg)
            renderer.set_style(style)
            style_frames = renderer.render_sequence(
                keypoints_seq, scores_seq, center_character=True
            )
            frames.append(style_frames)

        scene = SceneComposer(self.REEL_WIDTH, self.REEL_HEIGHT, bg_preset)

        composed = []
        for i in range(T):
            bg = scene.bg.generate(i, T)
            left_f = frames[0][i] if i < len(frames[0]) else frames[0][-1]
            right_f = frames[1][i] if i < len(frames[1]) else frames[1][-1]

            # Place side by side
            bg[0:self.REEL_HEIGHT, 0:half_w] = cv2.resize(
                left_f, (half_w, self.REEL_HEIGHT)
            )
            bg[0:self.REEL_HEIGHT, half_w:self.REEL_WIDTH] = cv2.resize(
                right_f, (half_w, self.REEL_HEIGHT)
            )

            # Divider line
            cv2.line(bg, (half_w, 0), (half_w, self.REEL_HEIGHT), (255, 255, 255), 2)
            composed.append(bg)

        self._write_video(composed, output_path, fps)
        return output_path

    # ── Internal ──

    def _reel_render_config(self, style: str) -> dict:
        """Build a renderer config for reel dimensions."""
        return {
            "avatar": {
                "canvas_width": self.REEL_WIDTH,
                "canvas_height": self.REEL_HEIGHT,
                "background_color": [0, 0, 0],
                "fps": 30,
                "skeleton_type": "wholebody_133",
                "default_style": style,
            },
        }

    def _add_transitions(
        self,
        frames: list[np.ndarray],
        segment_len: int,
        trans_len: int,
    ) -> list[np.ndarray]:
        """Add cross-fade transitions between segments."""
        if trans_len <= 0 or len(frames) < segment_len + trans_len:
            return frames

        result = list(frames)
        seg_boundaries = list(range(segment_len, len(frames), segment_len))

        for boundary in seg_boundaries:
            start = max(0, boundary - trans_len // 2)
            end = min(len(frames) - 1, boundary + trans_len // 2)
            for i in range(start, end + 1):
                if i <= 0 or i >= len(frames) - 1:
                    continue
                alpha = (i - start) / max(end - start, 1)
                result[i] = cv2.addWeighted(
                    frames[max(0, i - 1)], 1 - alpha * 0.3,
                    frames[i], 1 - (1 - alpha) * 0.3,
                    0,
                )

        return result

    def _write_video(
        self, frames: list[np.ndarray], output_path: str, fps: float
    ):
        """Write frames to MP4 using OpenCV."""
        if not frames:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

    def _mux_audio(self, video_path: str, audio_path: str):
        """Mux audio into video using ffmpeg (if available)."""
        import subprocess
        import shutil

        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found - skipping audio mux")
            return

        tmp = video_path + ".tmp.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    tmp,
                ],
                capture_output=True, check=True,
            )
            os.replace(tmp, video_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Audio mux failed: %s", e)
            if os.path.exists(tmp):
                os.remove(tmp)
