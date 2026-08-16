"""
Video effects — post-processing effects for rendered frames.

Supports motion blur, color grading, vignette, and particle effects.
"""

import cv2
import numpy as np


class VideoEffects:
    """Apply visual effects to rendered frames."""

    @staticmethod
    def motion_blur(
        frames: list[np.ndarray],
        strength: int = 3,
    ) -> list[np.ndarray]:
        """Apply temporal motion blur by blending consecutive frames.
        
        Args:
            frames: List of (H, W, 3) frames.
            strength: Number of frames to blend.
            
        Returns:
            List of blurred frames.
        """
        result = []
        for i in range(len(frames)):
            start = max(0, i - strength + 1)
            window = frames[start:i + 1]

            # Weighted blend (recent frames have more weight)
            weights = np.array([0.5 ** (i - j) for j in range(start, i + 1)])
            weights /= weights.sum()

            blended = np.zeros_like(frames[i], dtype=np.float64)
            for w, f in zip(weights, window):
                blended += f.astype(np.float64) * w
            result.append(np.clip(blended, 0, 255).astype(np.uint8))

        return result

    @staticmethod
    def color_grade(
        frames: list[np.ndarray],
        hue_shift: float = 0,
        saturation: float = 1.0,
        brightness: float = 1.0,
        contrast: float = 1.0,
    ) -> list[np.ndarray]:
        """Apply color grading to frames.
        
        Args:
            frames: List of (H, W, 3) BGR frames.
            hue_shift: Hue rotation in degrees (0-180).
            saturation: Saturation multiplier.
            brightness: Brightness multiplier.
            contrast: Contrast multiplier.
        """
        result = []
        for frame in frames:
            f = frame.copy()

            # Contrast and brightness
            if contrast != 1.0 or brightness != 1.0:
                f = cv2.convertScaleAbs(f, alpha=contrast, beta=(brightness - 1) * 128)

            # HSV adjustments
            if hue_shift != 0 or saturation != 1.0:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            result.append(f)
        return result

    @staticmethod
    def vignette(
        frames: list[np.ndarray],
        strength: float = 0.5,
    ) -> list[np.ndarray]:
        """Apply vignette (darkened edges) effect.
        
        Args:
            frames: List of (H, W, 3) frames.
            strength: Vignette intensity (0-1).
        """
        if not frames:
            return frames

        H, W = frames[0].shape[:2]

        # Create vignette mask
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X ** 2 + Y ** 2)
        vignette_mask = 1 - np.clip(dist * strength, 0, 1)
        vignette_mask = vignette_mask[:, :, np.newaxis]

        result = []
        for frame in frames:
            f = (frame.astype(np.float64) * vignette_mask).astype(np.uint8)
            result.append(f)
        return result

    @staticmethod
    def add_text_overlay(
        frames: list[np.ndarray],
        text: str,
        position: tuple[int, int] = (50, 50),
        font_scale: float = 1.0,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        duration_frames: int = -1,
    ) -> list[np.ndarray]:
        """Add text overlay to frames.
        
        Args:
            frames: List of frames.
            text: Text to display.
            position: (x, y) position.
            font_scale: Font size.
            color: BGR color.
            thickness: Text thickness.
            duration_frames: How many frames to show (-1 for all).
        """
        result = []
        for i, frame in enumerate(frames):
            f = frame.copy()
            if duration_frames == -1 or i < duration_frames:
                cv2.putText(
                    f, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness, cv2.LINE_AA
                )
            result.append(f)
        return result

    @staticmethod
    def fade_in_out(
        frames: list[np.ndarray],
        fade_in_frames: int = 30,
        fade_out_frames: int = 30,
    ) -> list[np.ndarray]:
        """Add fade-in and fade-out to frame sequence."""
        result = []
        T = len(frames)

        for i, frame in enumerate(frames):
            alpha = 1.0
            if i < fade_in_frames:
                alpha = i / fade_in_frames
            elif i >= T - fade_out_frames:
                alpha = (T - 1 - i) / fade_out_frames

            f = (frame.astype(np.float64) * alpha).astype(np.uint8)
            result.append(f)

        return result
