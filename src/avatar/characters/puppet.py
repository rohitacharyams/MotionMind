"""
Puppet character style — 2D body-part sprites with rotation & layering.

Each body part is a pre-rendered sprite (BGRA) that gets rotated and
placed based on joint keypoint positions.  Gives a real character look
without needing a 3D engine.

Supports:
  - Built-in character presets (casual, sporty, ninja, mech, etc.)
  - Customizable skin tone, hair, clothing, shoe colours
  - Proper z-ordering based on body facing direction
  - Gradient shading on limbs for 3D-like depth
  - Face rendering from 68-point face landmarks
"""

import cv2
import numpy as np
from ...pose_extraction.utils import HAND_EDGES


# ── Helpers ──────────────────────────────────────────────────────

def _angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle from p1 → p2 in degrees (0 = right, 90 = down)."""
    d = p2 - p1
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _limb_length(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p2 - p1))


def _make_sprite(width: int, height: int,
                 colour: tuple, shade_colour: tuple,
                 shape: str = "round_rect",
                 outline: tuple | None = None,
                 outline_w: int = 1) -> np.ndarray:
    """Generate a single body-part sprite (BGRA).

    shape: round_rect, ellipse, circle, trapezoid
    """
    img = np.zeros((height, width, 4), dtype=np.uint8)
    b, g, r = colour[:3]

    if shape == "ellipse":
        cx, cy = width // 2, height // 2
        cv2.ellipse(img, (cx, cy), (cx - 1, cy - 1), 0, 0, 360, (*colour, 255), -1, cv2.LINE_AA)
        # Gradient shading (left side darker)
        for x in range(width):
            shade = 0.75 + 0.25 * (x / max(width - 1, 1))
            img[:, x, :3] = np.clip(img[:, x, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
    elif shape == "circle":
        cx, cy = width // 2, height // 2
        rad = min(cx, cy) - 1
        cv2.circle(img, (cx, cy), rad, (*colour, 255), -1, cv2.LINE_AA)
        # Radial gradient for 3D look (vectorized)
        Y, X = np.mgrid[:height, :width]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32) / max(rad, 1)
        shade = np.clip(1.0 - 0.3 * dist, 0.7, 1.0)
        for c_idx in range(3):
            img[:, :, c_idx] = np.clip(img[:, :, c_idx].astype(np.float32) * shade, 0, 255).astype(np.uint8)
    elif shape == "trapezoid":
        # Wider at top, narrower at bottom (for torso)
        top_w = width
        bot_w = int(width * 0.75)
        pts = np.array([
            [width // 2 - top_w // 2, 0],
            [width // 2 + top_w // 2, 0],
            [width // 2 + bot_w // 2, height - 1],
            [width // 2 - bot_w // 2, height - 1],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (*colour, 255), cv2.LINE_AA)
        # Vertical shading
        for y in range(height):
            shade = 0.85 + 0.15 * (1 - y / max(height - 1, 1))
            img[y, :, :3] = np.clip(img[y, :, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
    else:
        # round_rect
        rad = min(width, height) // 4
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), (*colour, 255), -1, cv2.LINE_AA)
        # Round corners via masking
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (rad, 0), (width - rad, height), 255, -1)
        cv2.rectangle(mask, (0, rad), (width, height - rad), 255, -1)
        cv2.circle(mask, (rad, rad), rad, 255, -1)
        cv2.circle(mask, (width - rad, rad), rad, 255, -1)
        cv2.circle(mask, (rad, height - rad), rad, 255, -1)
        cv2.circle(mask, (width - rad, height - rad), rad, 255, -1)
        img[:, :, 3] = mask
        # Horizontal shading
        for x in range(width):
            shade = 0.80 + 0.20 * (x / max(width - 1, 1))
            img[:, x, :3] = np.clip(img[:, x, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)

    # Apply shade colour to bottom half for depth
    half = height // 2
    for y in range(half, height):
        t = (y - half) / max(height - half, 1)
        img[y, :, 0] = np.clip(img[y, :, 0].astype(np.float32) * (1 - t * 0.15) + shade_colour[0] * t * 0.15, 0, 255).astype(np.uint8)
        img[y, :, 1] = np.clip(img[y, :, 1].astype(np.float32) * (1 - t * 0.15) + shade_colour[1] * t * 0.15, 0, 255).astype(np.uint8)
        img[y, :, 2] = np.clip(img[y, :, 2].astype(np.float32) * (1 - t * 0.15) + shade_colour[2] * t * 0.15, 0, 255).astype(np.uint8)

    if outline:
        if shape == "ellipse":
            cx, cy = width // 2, height // 2
            cv2.ellipse(img, (cx, cy), (cx - 1, cy - 1), 0, 0, 360, (*outline, 255), outline_w, cv2.LINE_AA)
        elif shape == "circle":
            cx, cy = width // 2, height // 2
            rad = min(cx, cy) - 1
            cv2.circle(img, (cx, cy), rad, (*outline, 255), outline_w, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (*outline, 255), outline_w, cv2.LINE_AA)

    return img


def _place_sprite(canvas: np.ndarray, sprite: np.ndarray,
                  center: np.ndarray, angle: float, scale: float = 1.0):
    """Place a rotated BGRA sprite onto a BGR canvas at center position."""
    h, w = sprite.shape[:2]
    sw = max(int(w * scale), 1)
    sh = max(int(h * scale), 1)
    scaled = cv2.resize(sprite, (sw, sh), interpolation=cv2.INTER_AREA)

    # Rotation matrix around sprite center
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), -angle, 1.0)

    # Compute new bounding box after rotation
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    nw = int(sh * sin_a + sw * cos_a)
    nh = int(sh * cos_a + sw * sin_a)
    M[0, 2] += (nw - sw) / 2
    M[1, 2] += (nh - sh) / 2

    rotated = cv2.warpAffine(scaled, M, (nw, nh),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    # Place on canvas
    cx, cy = int(center[0]), int(center[1])
    x0 = cx - nw // 2
    y0 = cy - nh // 2

    ch, cw = canvas.shape[:2]
    # Clip to canvas bounds
    sx0 = max(-x0, 0)
    sy0 = max(-y0, 0)
    sx1 = min(nw, cw - x0)
    sy1 = min(nh, ch - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return

    dx0 = max(x0, 0)
    dy0 = max(y0, 0)

    region = rotated[sy0:sy1, sx0:sx1]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    rgb = region[:, :, :3]

    dst = canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)]
    if dst.shape[:2] != rgb.shape[:2]:
        return
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = (
        rgb.astype(np.float32) * alpha +
        dst.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)


# ── Presets ──────────────────────────────────────────────────────

CHARACTER_PRESETS = {
    "casual_male": {
        "skin": (200, 185, 160),
        "skin_shade": (160, 140, 120),
        "hair": (50, 40, 30),
        "hair_style": "short",
        "top": (180, 120, 80),           # jacket brown
        "top_shade": (130, 85, 55),
        "bottom": (100, 80, 60),         # dark pants
        "bottom_shade": (70, 55, 40),
        "shoe": (45, 40, 35),
        "outline": (35, 30, 25),
    },
    "casual_female": {
        "skin": (210, 195, 180),
        "skin_shade": (175, 155, 140),
        "hair": (40, 30, 80),
        "hair_style": "long",
        "top": (230, 180, 200),          # pink top
        "top_shade": (190, 140, 160),
        "bottom": (80, 60, 50),          # dark jeans
        "bottom_shade": (55, 40, 30),
        "shoe": (180, 160, 200),
        "outline": (35, 30, 25),
    },
    "sporty": {
        "skin": (190, 175, 155),
        "skin_shade": (150, 135, 115),
        "hair": (35, 30, 25),
        "hair_style": "short",
        "top": (50, 50, 230),            # red jersey
        "top_shade": (30, 30, 180),
        "bottom": (60, 60, 60),          # dark shorts
        "bottom_shade": (35, 35, 35),
        "shoe": (240, 240, 240),
        "outline": (30, 30, 30),
    },
    "ninja": {
        "skin": (180, 170, 155),
        "skin_shade": (140, 130, 115),
        "hair": (25, 20, 15),
        "hair_style": "short",
        "top": (40, 35, 30),             # dark outfit
        "top_shade": (25, 20, 15),
        "bottom": (40, 35, 30),
        "bottom_shade": (25, 20, 15),
        "shoe": (30, 25, 20),
        "outline": (15, 10, 8),
    },
    "mech": {
        "skin": (190, 190, 200),         # metallic
        "skin_shade": (140, 140, 160),
        "hair": (80, 80, 90),
        "hair_style": "none",
        "top": (160, 160, 170),          # silver armor
        "top_shade": (110, 110, 130),
        "bottom": (140, 140, 150),
        "bottom_shade": (100, 100, 120),
        "shoe": (120, 120, 130),
        "outline": (60, 60, 70),
    },
    "anime_heroine": {
        "skin": (220, 210, 235),
        "skin_shade": (190, 175, 210),
        "hair": (255, 120, 180),         # pink hair
        "hair_style": "long",
        "top": (240, 220, 100),          # yellow top
        "top_shade": (200, 180, 70),
        "bottom": (200, 80, 80),         # red skirt
        "bottom_shade": (160, 50, 50),
        "shoe": (80, 60, 60),
        "outline": (40, 30, 35),
    },
}


class PuppetStyle:
    """Render a 2D puppet character with rotatable body part sprites."""

    PRESETS = list(CHARACTER_PRESETS.keys())

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("puppet", {})
        preset_name = style_cfg.get("preset", "casual_male")
        preset = CHARACTER_PRESETS.get(preset_name, CHARACTER_PRESETS["casual_male"])

        # Allow per-property overrides
        self.skin = tuple(style_cfg.get("skin_color", preset["skin"]))
        self.skin_shade = tuple(style_cfg.get("skin_shade", preset["skin_shade"]))
        self.hair_color = tuple(style_cfg.get("hair_color", preset["hair"]))
        self.hair_style = style_cfg.get("hair_style", preset["hair_style"])
        self.top_color = tuple(style_cfg.get("top_color", preset["top"]))
        self.top_shade = tuple(style_cfg.get("top_shade", preset["top_shade"]))
        self.bottom_color = tuple(style_cfg.get("bottom_color", preset["bottom"]))
        self.bottom_shade = tuple(style_cfg.get("bottom_shade", preset["bottom_shade"]))
        self.shoe_color = tuple(style_cfg.get("shoe_color", preset["shoe"]))
        self.outline = tuple(style_cfg.get("outline_color", preset["outline"]))
        self.outline_w = style_cfg.get("outline_width", 2)

        # Custom sprite directory (optional)
        self.sprite_dir = style_cfg.get("sprite_dir", None)
        self._custom_sprites: dict | None = None

        # Cached generated sprites (keyed by reference size)
        self._sprite_cache: dict = {}
        self._last_ref_size: int = 0

    # ── Public API ───────────────────────────────────────────

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        K = len(keypoints)
        if K < 17:
            return canvas

        kps = keypoints.astype(np.float64)
        torso_w = _limb_length(kps[5], kps[6])
        if torso_w < 3:
            return canvas

        # Reference size for sprite scaling
        ref = int(torso_w)
        if ref != self._last_ref_size:
            self._sprite_cache.clear()
            self._last_ref_size = ref

        # Determine facing direction (left shoulder vs right shoulder x)
        facing_right = kps[6][0] > kps[5][0]

        # Build draw list (back-to-front)
        draw_list = self._build_draw_list(kps, scores, K, torso_w,
                                          facing_right, min_score)

        for item in draw_list:
            self._draw_part(canvas, item, kps, torso_w)

        # Face (always on top)
        self._draw_face(canvas, kps, scores, K, torso_w, min_score)

        return canvas

    # ── Part Drawing ─────────────────────────────────────────

    def _build_draw_list(self, kps, scores, K, tw, facing_right, min_score):
        """Return draw order list based on body facing direction.

        Each item: (part_name, sprite_key, center, angle, scale)
        """
        # far/near arm & leg based on facing direction
        if facing_right:
            back_arm = ("arm", 5, 7, 9)    # left arm is back
            front_arm = ("arm", 6, 8, 10)
            back_leg = ("leg", 11, 13, 15)  # left leg is back
            front_leg = ("leg", 12, 14, 16)
        else:
            back_arm = ("arm", 6, 8, 10)
            front_arm = ("arm", 5, 7, 9)
            back_leg = ("leg", 12, 14, 16)
            front_leg = ("leg", 11, 13, 15)

        items = []

        # 1. Back upper arm
        items.append(self._limb_item("upper_arm", back_arm[1], back_arm[2],
                                     kps, scores, tw, min_score, is_skin=True))
        # 2. Back forearm
        items.append(self._limb_item("forearm", back_arm[2], back_arm[3],
                                     kps, scores, tw, min_score, is_skin=True))
        # 3. Back hand
        items.append(self._hand_item(back_arm[3], kps, scores, tw, K, min_score))
        # 4. Back thigh
        items.append(self._limb_item("thigh", back_leg[1], back_leg[2],
                                     kps, scores, tw, min_score, is_skin=False))
        # 5. Back shin
        items.append(self._limb_item("shin", back_leg[2], back_leg[3],
                                     kps, scores, tw, min_score, is_skin=False))
        # 6. Back foot
        items.append(self._foot_item(back_leg[3], back_leg[2], kps, tw))
        # 7. Torso
        items.append(("torso", kps, tw))
        # 8. Head + hair
        items.append(("head", kps, tw))
        # 9. Front thigh
        items.append(self._limb_item("thigh", front_leg[1], front_leg[2],
                                     kps, scores, tw, min_score, is_skin=False))
        # 10. Front shin
        items.append(self._limb_item("shin", front_leg[2], front_leg[3],
                                     kps, scores, tw, min_score, is_skin=False))
        # 11. Front foot
        items.append(self._foot_item(front_leg[3], front_leg[2], kps, tw))
        # 12. Front upper arm
        items.append(self._limb_item("upper_arm", front_arm[1], front_arm[2],
                                     kps, scores, tw, min_score, is_skin=True))
        # 13. Front forearm
        items.append(self._limb_item("forearm", front_arm[2], front_arm[3],
                                     kps, scores, tw, min_score, is_skin=True))
        # 14. Front hand
        items.append(self._hand_item(front_arm[3], kps, scores, tw, K, min_score))

        return items

    def _limb_item(self, part, j1, j2, kps, scores, tw, min_score, is_skin):
        if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
            return None
        return ("limb", part, kps[j1], kps[j2], tw, is_skin)

    def _hand_item(self, wrist_idx, kps, scores, tw, K, min_score):
        if scores is not None and scores[wrist_idx] < min_score:
            return None
        return ("hand", kps[wrist_idx], tw, wrist_idx, kps, scores, K, min_score)

    def _foot_item(self, ankle_idx, knee_idx, kps, tw):
        return ("foot", kps[ankle_idx], kps[knee_idx], tw)

    def _draw_part(self, canvas, item, kps, tw):
        if item is None:
            return

        kind = item[0]
        if kind == "limb":
            self._draw_limb(canvas, *item[1:])
        elif kind == "torso":
            self._draw_torso(canvas, item[1], item[2])
        elif kind == "head":
            self._draw_head(canvas, item[1], item[2])
        elif kind == "hand":
            self._draw_hand(canvas, *item[1:])
        elif kind == "foot":
            self._draw_foot(canvas, *item[1:])

    # ── Limbs ────────────────────────────────────────────────

    def _draw_limb(self, canvas, part, p1, p2, tw, is_skin):
        length = _limb_length(p1, p2)
        if length < 2:
            return
        angle = _angle_deg(p1, p2)
        mid = (p1 + p2) / 2

        width_map = {
            "upper_arm": 0.22,
            "forearm": 0.17,
            "thigh": 0.28,
            "shin": 0.20,
        }
        w_ratio = width_map.get(part, 0.18)
        w = max(int(tw * w_ratio), 6)
        h = max(int(length), 4)

        colour = self.skin if is_skin else self.bottom_color
        shade = self.skin_shade if is_skin else self.bottom_shade

        sprite = self._get_sprite(f"limb_{part}_{is_skin}_{w}_{h}",
                                  w, h, colour, shade, "ellipse")
        _place_sprite(canvas, sprite, mid, angle - 90, 1.0)

    # ── Torso ────────────────────────────────────────────────

    def _draw_torso(self, canvas, kps, tw):
        shoulder_mid = (kps[5] + kps[6]) / 2
        hip_mid = (kps[11] + kps[12]) / 2
        torso_h = _limb_length(shoulder_mid, hip_mid)
        if torso_h < 3:
            return

        w = max(int(tw * 1.1), 10)
        h = max(int(torso_h), 8)

        sprite = self._get_sprite(f"torso_{w}_{h}",
                                  w, h, self.top_color, self.top_shade,
                                  "trapezoid")
        center = (shoulder_mid + hip_mid) / 2
        # Torso angle from shoulder-mid to hip-mid
        angle = _angle_deg(shoulder_mid, hip_mid)
        _place_sprite(canvas, sprite, center, angle - 90, 1.0)

        # Collar / neckline detail
        neck_pt = (shoulder_mid * 0.7 + kps[0] * 0.3).astype(int)
        neck_w = max(int(tw * 0.15), 3)
        cv2.line(canvas, tuple(shoulder_mid.astype(int)),
                 tuple(neck_pt), self.skin, neck_w, cv2.LINE_AA)

    # ── Head ─────────────────────────────────────────────────

    def _draw_head(self, canvas, kps, tw):
        head_center = kps[0].astype(np.float64)
        K = len(kps)

        if K > 4:
            eye_dist = _limb_length(kps[1], kps[2])
            head_r = max(int(eye_dist * 2.0), int(tw * 0.25))
        else:
            head_r = max(int(tw * 0.28), 12)

        cx, cy = int(head_center[0]), int(head_center[1])

        # Neck
        shoulder_mid = ((kps[5] + kps[6]) / 2).astype(int)
        neck_w = max(int(tw * 0.12), 4)
        cv2.line(canvas, tuple(shoulder_mid), (cx, cy + head_r // 2),
                 self.skin, neck_w, cv2.LINE_AA)

        # Head circle with skin
        cv2.circle(canvas, (cx, cy), head_r, self.skin, -1, cv2.LINE_AA)

        # 3D shading — radial gradient overlay using cv2
        if head_r > 5:
            overlay = canvas.copy()
            # Highlight circle offset upper-right
            hx = cx + head_r // 4
            hy = cy - head_r // 4
            cv2.circle(overlay, (hx, hy), head_r // 2,
                       tuple(min(c + 40, 255) for c in self.skin), -1, cv2.LINE_AA)
            # Blend highlight
            mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), head_r, 255, -1)
            alpha = cv2.GaussianBlur(mask, (head_r | 1, head_r | 1), head_r * 0.3)
            a = (alpha.astype(np.float32) / 255.0 * 0.3)[:, :, np.newaxis]
            roi_y0 = max(0, cy - head_r - 2)
            roi_y1 = min(canvas.shape[0], cy + head_r + 2)
            roi_x0 = max(0, cx - head_r - 2)
            roi_x1 = min(canvas.shape[1], cx + head_r + 2)
            a_roi = a[roi_y0:roi_y1, roi_x0:roi_x1]
            canvas[roi_y0:roi_y1, roi_x0:roi_x1] = (
                overlay[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float32) * a_roi +
                canvas[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float32) * (1 - a_roi)
            ).astype(np.uint8)

        # Hair
        self._draw_hair(canvas, cx, cy, head_r)

        # Outline
        cv2.circle(canvas, (cx, cy), head_r, self.outline, self.outline_w, cv2.LINE_AA)

    def _draw_hair(self, canvas, cx, cy, r):
        if self.hair_style == "none":
            return

        if self.hair_style == "short":
            # Hair on top half of head
            cv2.ellipse(canvas, (cx, cy - r // 6), (r + 2, r - r // 4),
                        0, 180, 360, self.hair_color, -1, cv2.LINE_AA)
            # Side burns
            cv2.ellipse(canvas, (cx - r + 2, cy), (4, r // 3),
                        0, 0, 360, self.hair_color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + r - 2, cy), (4, r // 3),
                        0, 0, 360, self.hair_color, -1, cv2.LINE_AA)
        elif self.hair_style == "long":
            # Full hair on top
            cv2.ellipse(canvas, (cx, cy - r // 4), (r + 4, r),
                        0, 180, 360, self.hair_color, -1, cv2.LINE_AA)
            # Side hair falling down
            cv2.ellipse(canvas, (cx - r, cy + r // 2), (6, r),
                        0, 0, 360, self.hair_color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + r, cy + r // 2), (6, r),
                        0, 0, 360, self.hair_color, -1, cv2.LINE_AA)

    # ── Face ─────────────────────────────────────────────────

    def _draw_face(self, canvas, kps, scores, K, tw, min_score):
        head_center = kps[0].astype(int)
        if K > 4:
            eye_dist = _limb_length(kps[1], kps[2])
            head_r = max(int(eye_dist * 2.0), int(tw * 0.25))
        else:
            head_r = max(int(tw * 0.28), 12)

        cx, cy = int(head_center[0]), int(head_center[1])

        if K >= 91 and scores is not None:
            # Detailed face from landmarks
            self._draw_face_landmarks(canvas, kps, scores, cx, cy, head_r, min_score)
        elif K > 4:
            # Simple face from body keypoints
            self._draw_simple_face(canvas, kps, cx, cy, head_r)

    def _draw_simple_face(self, canvas, kps, cx, cy, r):
        eye_r = max(r // 7, 2)
        le = tuple(kps[1].astype(int))
        re = tuple(kps[2].astype(int))
        # Whites
        cv2.circle(canvas, le, eye_r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, re, eye_r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        # Iris
        cv2.circle(canvas, le, eye_r, (60, 40, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, re, eye_r, (60, 40, 30), -1, cv2.LINE_AA)
        # Highlight
        cv2.circle(canvas, (le[0] + 1, le[1] - 1), max(eye_r // 2, 1), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (re[0] + 1, re[1] - 1), max(eye_r // 2, 1), (255, 255, 255), -1, cv2.LINE_AA)
        # Mouth
        mouth_y = cy + int(r * 0.35)
        mouth_w = max(int(r * 0.25), 3)
        cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, mouth_w // 2),
                    0, 0, 180, (120, 80, 100), 2, cv2.LINE_AA)
        # Nose hint
        cv2.circle(canvas, (cx, cy + int(r * 0.1)), max(r // 12, 1),
                   self.skin_shade, -1, cv2.LINE_AA)

    def _draw_face_landmarks(self, canvas, kps, scores, cx, cy, r, min_score):
        """Draw face using 68 landmarks (indices 23-90)."""
        # Eyes from landmarks
        for eye_start, eye_end in [(36, 42), (42, 48)]:
            pts = []
            valid = True
            for i in range(eye_start, eye_end):
                idx = 23 + i
                if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                    valid = False
                    break
                pts.append(kps[idx].astype(int))
            if valid and len(pts) >= 4:
                arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(canvas, [arr], (255, 255, 255), cv2.LINE_AA)
                cv2.polylines(canvas, [arr], True, self.outline, 1, cv2.LINE_AA)
                ec = arr.mean(axis=0).astype(int)
                pupil_r = max(int(np.linalg.norm(arr[0] - arr[3]) * 0.25), 1)
                cv2.circle(canvas, tuple(ec), pupil_r, (60, 40, 30), -1, cv2.LINE_AA)
                cv2.circle(canvas, (ec[0] + 1, ec[1] - 1), max(pupil_r // 2, 1),
                           (255, 255, 255), -1, cv2.LINE_AA)

        # Eyebrows
        for brow_start, brow_end in [(17, 22), (22, 27)]:
            pts = []
            for i in range(brow_start, brow_end):
                idx = 23 + i
                if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                    continue
                pts.append(kps[idx].astype(int))
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.int32)
                cv2.polylines(canvas, [arr], False, self.outline, 2, cv2.LINE_AA)

        # Mouth
        outer_pts = []
        for i in range(48, 60):
            idx = 23 + i
            if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                continue
            outer_pts.append(kps[idx].astype(int))
        if len(outer_pts) >= 8:
            arr = np.array(outer_pts, dtype=np.int32)
            cv2.fillPoly(canvas, [arr], (120, 90, 120), cv2.LINE_AA)
            cv2.polylines(canvas, [arr], True, self.outline, 1, cv2.LINE_AA)

        # Nose hint
        nose_tip_idx = 23 + 30
        if nose_tip_idx < len(kps) and (scores is None or scores[nose_tip_idx] >= min_score):
            pt = tuple(kps[nose_tip_idx].astype(int))
            cv2.circle(canvas, pt, max(r // 15, 1), self.skin_shade, -1, cv2.LINE_AA)

    # ── Hands ────────────────────────────────────────────────

    def _draw_hand(self, canvas, pos, tw, wrist_idx, kps, scores, K, min_score):
        r = max(int(tw * 0.07), 3)
        pt = tuple(pos.astype(int))
        cv2.circle(canvas, pt, r, self.skin, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, self.outline, 1, cv2.LINE_AA)

        # Finger details from hand landmarks
        if K >= 133:
            hand_start = 91 if wrist_idx == 9 else 112
            finger_w = max(int(tw * 0.025), 1)
            for local_a, local_b in HAND_EDGES:
                a = hand_start + local_a
                b = hand_start + local_b
                if a >= K or b >= K:
                    continue
                if scores is not None and (scores[a] < min_score or scores[b] < min_score):
                    continue
                pa = tuple(kps[a].astype(int))
                pb = tuple(kps[b].astype(int))
                cv2.line(canvas, pa, pb, self.skin, finger_w + 1, cv2.LINE_AA)
            # Fingertip circles
            for tip in [4, 8, 12, 16, 20]:
                idx = hand_start + tip
                if idx >= K or (scores is not None and scores[idx] < min_score):
                    continue
                pt = tuple(kps[idx].astype(int))
                cv2.circle(canvas, pt, finger_w + 1, self.skin, -1, cv2.LINE_AA)

    # ── Feet ─────────────────────────────────────────────────

    def _draw_foot(self, canvas, ankle_pos, knee_pos, tw):
        foot_w = max(int(tw * 0.15), 5)
        foot_h = max(int(tw * 0.08), 3)
        pt = tuple(ankle_pos.astype(int))
        foot_dir = ankle_pos - knee_pos
        angle = float(np.degrees(np.arctan2(foot_dir[1], foot_dir[0])))
        cv2.ellipse(canvas, pt, (foot_w, foot_h), angle + 90, 0, 360,
                    self.shoe_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, pt, (foot_w, foot_h), angle + 90, 0, 360,
                    self.outline, self.outline_w, cv2.LINE_AA)

    # ── Sprite Cache ─────────────────────────────────────────

    def _get_sprite(self, key, w, h, colour, shade, shape):
        if key not in self._sprite_cache:
            self._sprite_cache[key] = _make_sprite(
                w, h, colour, shade, shape,
                outline=self.outline, outline_w=self.outline_w
            )
        return self._sprite_cache[key]
