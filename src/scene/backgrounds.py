"""
Background and scene composition system.

Provides various background generators and scene layouts
for rendering dance reels — studio, gradient, stage, outdoor, etc.
"""

import cv2
import numpy as np


class BackgroundGenerator:
    """Generates background frames for reel rendering."""

    PRESETS = {
        "studio_dark": {
            "type": "gradient",
            "top_color": [15, 15, 25],
            "bottom_color": [35, 30, 45],
            "spotlight": True,
            "spotlight_color": [60, 55, 70],
        },
        "studio_white": {
            "type": "gradient",
            "top_color": [245, 245, 250],
            "bottom_color": [210, 210, 220],
            "spotlight": True,
            "spotlight_color": [255, 255, 255],
        },
        "neon_club": {
            "type": "gradient",
            "top_color": [5, 0, 20],
            "bottom_color": [0, 0, 10],
            "spotlight": False,
            "particles": True,
            "particle_color": [100, 0, 200],
        },
        "sunset_orange": {
            "type": "gradient",
            "top_color": [20, 30, 80],
            "bottom_color": [80, 100, 230],
            "spotlight": False,
        },
        "pink_pop": {
            "type": "gradient",
            "top_color": [120, 50, 180],
            "bottom_color": [200, 80, 150],
            "spotlight": True,
            "spotlight_color": [255, 150, 200],
        },
        "stage_floor": {
            "type": "stage",
            "floor_color": [50, 45, 40],
            "wall_color": [20, 18, 25],
            "floor_ratio": 0.35,
            "reflection": True,
        },
        "dance_studio": {
            "type": "stage",
            "floor_color": [180, 150, 110],
            "wall_color": [230, 225, 220],
            "floor_ratio": 0.40,
            "reflection": False,
            "mirror_wall": True,
        },
        "black": {
            "type": "solid",
            "color": [0, 0, 0],
        },
        "white": {
            "type": "solid",
            "color": [255, 255, 255],
        },
        "custom_image": {
            "type": "image",
            "path": None,
        },
    }

    def __init__(self, width: int, height: int, preset: str = "studio_dark"):
        self.width = width
        self.height = height
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        self.config = self.PRESETS[preset].copy()
        self._cached_bg = None

    def generate(self, frame_idx: int = 0, total_frames: int = 1) -> np.ndarray:
        """Generate a background frame.
        
        Args:
            frame_idx: Current frame number (for animated backgrounds).
            total_frames: Total frames in sequence.
            
        Returns:
            (H, W, 3) uint8 BGR image.
        """
        bg_type = self.config["type"]
        if bg_type == "gradient":
            bg = self._gradient(frame_idx, total_frames)
        elif bg_type == "stage":
            bg = self._stage(frame_idx, total_frames)
        elif bg_type == "solid":
            bg = self._solid()
        elif bg_type == "image":
            bg = self._from_image()
        else:
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return bg

    def _solid(self) -> np.ndarray:
        if self._cached_bg is not None:
            return self._cached_bg.copy()
        c = self.config["color"]
        bg = np.full((self.height, self.width, 3), c, dtype=np.uint8)
        self._cached_bg = bg
        return bg.copy()

    def _gradient(self, frame_idx: int, total_frames: int) -> np.ndarray:
        # Cache static gradient + spotlight (only particles are animated)
        if self._cached_bg is None:
            top = np.array(self.config["top_color"], dtype=np.float32)
            bot = np.array(self.config["bottom_color"], dtype=np.float32)

            # Vectorized vertical gradient
            t = np.linspace(0, 1, self.height, dtype=np.float32)[:, np.newaxis]
            grad = (top[np.newaxis, :] * (1 - t) + bot[np.newaxis, :] * t).astype(np.uint8)
            grad = np.broadcast_to(grad[:, np.newaxis, :], (self.height, self.width, 3)).copy()

            # Spotlight
            if self.config.get("spotlight"):
                sc = self.config.get("spotlight_color", [100, 100, 120])
                center_x = self.width // 2
                center_y = int(self.height * 0.35)
                radius = int(min(self.width, self.height) * 0.45)
                mask = np.zeros((self.height, self.width), dtype=np.float32)
                cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.4)
                for c_idx in range(3):
                    grad[:, :, c_idx] = np.clip(
                        grad[:, :, c_idx].astype(np.float32) + mask * sc[c_idx] * 0.3,
                        0, 255,
                    ).astype(np.uint8)

            self._cached_bg = grad

        # If no particles, return cached copy directly
        if not self.config.get("particles"):
            return self._cached_bg.copy()

        # Animated particles on a copy
        grad = self._cached_bg.copy()
        pc = self.config.get("particle_color", [100, 100, 255])
        rng = np.random.RandomState(42)
        n_particles = 25
        for _ in range(n_particles):
            px = rng.randint(0, self.width)
            base_py = rng.randint(0, self.height)
            py = (base_py + frame_idx * rng.randint(1, 4)) % self.height
            size = rng.randint(1, 4)
            alpha = rng.uniform(0.2, 0.7)
            cv2.circle(grad, (px, py), size, [int(c * alpha) for c in pc], -1)

        return grad

    def _stage(self, frame_idx: int, total_frames: int) -> np.ndarray:
        if self._cached_bg is not None:
            return self._cached_bg.copy()

        fc = np.array(self.config["floor_color"], dtype=np.uint8)
        wc = np.array(self.config["wall_color"], dtype=np.uint8)
        floor_y = int(self.height * (1 - self.config.get("floor_ratio", 0.35)))

        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Wall - vectorized vertical gradient
        if floor_y > 0:
            t = np.linspace(0, 1, floor_y, dtype=np.float32)
            wall_colors = (wc.astype(np.float32) * (0.85 + 0.15 * t[:, np.newaxis])).astype(np.uint8)
            bg[:floor_y, :] = wall_colors[:, np.newaxis, :]

        # Floor
        bg[floor_y:, :] = fc

        # Horizon line
        cv2.line(bg, (0, floor_y), (self.width, floor_y),
                 ((fc.astype(np.int32) + wc.astype(np.int32)) // 2).tolist(), 2)

        # Reflection zone
        if self.config.get("reflection"):
            reflect_h = int((self.height - floor_y) * 0.3)
            if reflect_h > 0:
                t = np.linspace(0, 1, reflect_h, dtype=np.float32)
                alphas = 0.15 * (1 - t)
                for y_off in range(reflect_h):
                    bg[floor_y + y_off, :] = np.clip(
                        bg[floor_y + y_off, :].astype(np.float32) + alphas[y_off] * 30,
                        0, 255,
                    ).astype(np.uint8)

        # Mirror wall effect
        if self.config.get("mirror_wall"):
            # Thin bright strip at top
            strip_h = max(3, self.height // 80)
            bg[floor_y - strip_h : floor_y, :] = [200, 200, 210]

        self._cached_bg = bg
        return bg.copy()

    def _from_image(self) -> np.ndarray:
        if self._cached_bg is not None:
            return self._cached_bg.copy()
        path = self.config.get("path")
        if path is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._cached_bg = cv2.resize(img, (self.width, self.height))
        return self._cached_bg.copy()

    @staticmethod
    def list_presets() -> list[dict]:
        """List all available background presets."""
        return [
            {"id": k, "type": v["type"]}
            for k, v in BackgroundGenerator.PRESETS.items()
        ]


class SceneComposer:
    """Composes character animation over a background with effects."""

    def __init__(
        self,
        width: int,
        height: int,
        bg_preset: str = "studio_dark",
        layout: str = "center",
    ):
        self.width = width
        self.height = height
        self.bg = BackgroundGenerator(width, height, bg_preset)
        self.layout = layout

    def compose_frame(
        self,
        character_frame: np.ndarray,
        frame_idx: int = 0,
        total_frames: int = 1,
        watermark: str | None = None,
    ) -> np.ndarray:
        """Overlay character onto background.
        
        Args:
            character_frame: (H, W, 3) or (H, W, 4) character rendering.
                If 3-channel, black pixels are treated as transparent.
                If 4-channel, alpha channel is used.
            frame_idx: For animated background effects.
            total_frames: Total frame count.
            watermark: Optional watermark text at bottom.
            
        Returns:
            (H, W, 3) composed frame.
        """
        bg = self.bg.generate(frame_idx, total_frames)

        # Resize character to match bg if needed
        ch, cw = character_frame.shape[:2]
        if ch != self.height or cw != self.width:
            character_frame = cv2.resize(character_frame, (self.width, self.height))

        # Create mask
        if character_frame.shape[2] == 4:
            alpha = character_frame[:, :, 3:4].astype(np.float32) / 255.0
            char_rgb = character_frame[:, :, :3]
            # Float blend for alpha channel
            result = (char_rgb.astype(np.float32) * alpha +
                      bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        else:
            # Fast bitwise blend for 3-channel (black = transparent)
            gray = cv2.cvtColor(character_frame, cv2.COLOR_BGR2GRAY)
            mask = gray > 5
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            result = np.where(mask_3ch, character_frame, bg)

        # Floor reflection for stage backgrounds
        if self.bg.config["type"] == "stage" and self.bg.config.get("reflection"):
            floor_y = int(self.height * (1 - self.bg.config.get("floor_ratio", 0.35)))
            if character_frame.shape[2] == 4:
                char_rgb = character_frame[:, :, :3]
                alpha = character_frame[:, :, 3:4].astype(np.float32) / 255.0
            else:
                char_rgb = character_frame
                gray = cv2.cvtColor(character_frame, cv2.COLOR_BGR2GRAY)
                alpha = (gray > 5).astype(np.float32)[:, :, np.newaxis]
            self._add_floor_reflection(result, char_rgb, alpha, floor_y)

        # Watermark
        if watermark:
            self._add_watermark(result, watermark)

        return result

    def _add_floor_reflection(
        self,
        result: np.ndarray,
        char_rgb: np.ndarray,
        alpha: np.ndarray,
        floor_y: int,
    ):
        """Add a subtle floor reflection below the character."""
        reflect_zone = self.height - floor_y
        if reflect_zone < 20:
            return

        # Take bottom portion of character, flip, fade
        char_bottom = char_rgb[max(0, floor_y - reflect_zone // 2) : floor_y]
        if char_bottom.shape[0] < 5:
            return

        reflected = cv2.flip(char_bottom, 0)
        rh = min(reflected.shape[0], reflect_zone)
        reflected = reflected[:rh]

        # Apply fade
        for y in range(rh):
            fade = 0.08 * (1 - y / rh)
            row = floor_y + y
            if row >= self.height:
                break
            result[row] = np.clip(
                result[row].astype(np.float32) + reflected[y].astype(np.float32) * fade,
                0, 255,
            ).astype(np.uint8)

    def _add_watermark(self, frame: np.ndarray, text: str):
        """Add a semi-transparent watermark."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, self.width / 2500)
        thickness = max(1, int(self.width / 900))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = self.width - tw - 15
        y = self.height - 15
        # Dark shadow
        cv2.putText(frame, text, (x + 1, y + 1), font, font_scale,
                     (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale,
                     (200, 200, 200), thickness, cv2.LINE_AA)

    @staticmethod
    def reel_dimensions() -> dict:
        """Standard social media reel dimensions."""
        return {
            "instagram_reel": (1080, 1920),  # 9:16
            "tiktok": (1080, 1920),
            "youtube_short": (1080, 1920),
            "instagram_square": (1080, 1080),
            "landscape_1080p": (1920, 1080),
        }
