"""
Video exporter — write rendered frames to video files.

Uses OpenCV VideoWriter or imageio-ffmpeg for high-quality output
with configurable codec, resolution, and quality settings.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoExporter:
    """Export rendered frames to video files."""

    def __init__(self, config: dict):
        cfg = config.get("video_output", {})
        self.codec = cfg.get("codec", "libx264")
        self.quality = cfg.get("quality", 18)
        self.output_dir = Path(cfg.get("output_dir", "data/output_videos"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_opencv(
        self,
        frames: list[np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> str:
        """Export frames using OpenCV VideoWriter.
        
        Args:
            frames: List of (H, W, 3) BGR frames.
            output_path: Output file path (relative to output_dir or absolute).
            fps: Output frames per second.
            
        Returns:
            Absolute path to output file.
        """
        if not frames:
            raise ValueError("No frames to export")

        output_path = self._resolve_path(output_path)
        H, W = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        for frame in tqdm(frames, desc="Writing video", unit="frame"):
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            writer.write(frame)

        writer.release()
        logger.info("Exported video: %s (%d frames, %.1f fps)", output_path, len(frames), fps)
        return str(output_path)

    def export_ffmpeg(
        self,
        frames: list[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        audio_path: str | None = None,
    ) -> str:
        """Export frames using FFmpeg (higher quality, more codecs).
        
        Args:
            frames: List of (H, W, 3) BGR frames.
            output_path: Output file path.
            fps: Output FPS.
            audio_path: Optional audio file to mux in.
            
        Returns:
            Absolute path to output file.
        """
        try:
            import imageio
            import imageio_ffmpeg
        except ImportError:
            logger.warning("imageio-ffmpeg not available, falling back to OpenCV")
            return self.export_opencv(frames, output_path, fps)

        output_path = self._resolve_path(output_path)
        H, W = frames[0].shape[:2]

        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=self.codec,
            quality=None,
            output_params=["-crf", str(self.quality)],
            macro_block_size=1,
        )

        for frame in tqdm(frames, desc="Writing video (ffmpeg)", unit="frame"):
            # imageio expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

        writer.close()

        # Mux audio if provided
        if audio_path and os.path.exists(audio_path):
            self._mux_audio(str(output_path), audio_path)

        logger.info("Exported video: %s (%d frames, %.1f fps)", output_path, len(frames), fps)
        return str(output_path)

    def _mux_audio(self, video_path: str, audio_path: str):
        """Mux audio track into video using FFmpeg."""
        import subprocess
        
        temp_path = video_path + ".temp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            temp_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(temp_path, video_path)
            logger.info("Muxed audio: %s", audio_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Audio muxing failed: %s", e)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _resolve_path(self, path: str) -> Path:
        """Resolve output path (relative to output_dir if not absolute)."""
        p = Path(path)
        if not p.is_absolute():
            p = self.output_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def export_frame_sequence(
        self,
        frames: list[np.ndarray],
        output_dir: str,
        prefix: str = "frame",
        fmt: str = "png",
    ) -> str:
        """Export frames as individual images.
        
        Args:
            frames: List of frames.
            output_dir: Directory for frame images.
            prefix: Filename prefix.
            fmt: Image format (png, jpg).
            
        Returns:
            Path to output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            path = out / f"{prefix}_{i:06d}.{fmt}"
            cv2.imwrite(str(path), frame)

        logger.info("Exported %d frames to %s", len(frames), out)
        return str(out)
