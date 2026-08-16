"""
High-quality 2D character style — renders a detailed, smooth character
directly from COCO-17 keypoints each frame.

Uses anti-aliased drawing with gradients, proper layering, detailed
face/hair rendering, and smooth limb shapes for a polished look.
No mesh deformation needed — draws fresh each frame.
"""

import cv2
import numpy as np


# ── Appearance presets ─────────────────────────────────────────────

HQCHAR_PRESETS = {
    "realistic_male": {
        "skin": (195, 175, 155),
        "skin_shade": (155, 135, 115),
        "hair": (45, 35, 25),
        "hair_style": "short",
        "top": (140, 90, 60),
        "top_shade": (100, 60, 35),
        "sleeve": (140, 90, 60),
        "bottom": (85, 70, 55),
        "bottom_shade": (55, 40, 28),
        "shoe": (40, 35, 30),
        "shoe_shade": (25, 20, 15),
        "eye": (80, 50, 30),
        "lip": (130, 110, 140),
        "brow": (60, 45, 30),
    },
    "realistic_female": {
        "skin": (215, 200, 185),
        "skin_shade": (180, 160, 145),
        "hair": (35, 25, 65),
        "hair_style": "long",
        "top": (230, 180, 200),
        "top_shade": (185, 140, 160),
        "sleeve": (230, 180, 200),
        "bottom": (75, 55, 45),
        "bottom_shade": (50, 35, 25),
        "shoe": (170, 150, 190),
        "shoe_shade": (120, 100, 140),
        "eye": (140, 80, 50),
        "lip": (120, 100, 180),
        "brow": (50, 35, 60),
    },
    "stylized_anime": {
        "skin": (225, 215, 240),
        "skin_shade": (190, 175, 210),
        "hair": (255, 120, 180),
        "hair_style": "long",
        "top": (240, 220, 100),
        "top_shade": (200, 180, 70),
        "sleeve": (240, 220, 100),
        "bottom": (200, 80, 80),
        "bottom_shade": (160, 50, 50),
        "shoe": (80, 60, 60),
        "shoe_shade": (50, 35, 35),
        "eye": (200, 100, 60),
        "lip": (140, 100, 200),
        "brow": (180, 80, 130),
    },
    "dark_suit": {
        "skin": (200, 185, 165),
        "skin_shade": (160, 145, 125),
        "hair": (30, 25, 20),
        "hair_style": "short",
        "top": (55, 50, 45),
        "top_shade": (30, 28, 25),
        "sleeve": (55, 50, 45),
        "bottom": (50, 45, 40),
        "bottom_shade": (30, 28, 25),
        "shoe": (25, 22, 18),
        "shoe_shade": (15, 12, 10),
        "eye": (90, 60, 35),
        "lip": (140, 120, 150),
        "brow": (40, 30, 22),
    },
}


def _angle_deg(p1, p2):
    d = p2 - p1
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _lerp(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


class HQCharacterStyle:
    """High-quality 2D character rendered fresh each frame from keypoints."""

    PRESETS = list(HQCHAR_PRESETS.keys())

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("hqchar", {})
        preset_name = style_cfg.get("preset", "realistic_male")
        p = HQCHAR_PRESETS.get(preset_name, HQCHAR_PRESETS["realistic_male"])

        self.skin = tuple(style_cfg.get("skin_color", p["skin"]))
        self.skin_shade = tuple(style_cfg.get("skin_shade", p["skin_shade"]))
        self.hair_color = tuple(style_cfg.get("hair_color", p["hair"]))
        self.hair_style = style_cfg.get("hair_style", p["hair_style"])
        self.top = tuple(style_cfg.get("top_color", p["top"]))
        self.top_shade = tuple(style_cfg.get("top_shade", p["top_shade"]))
        self.sleeve = tuple(style_cfg.get("sleeve_color", p.get("sleeve", p["top"])))
        self.bottom = tuple(style_cfg.get("bottom_color", p["bottom"]))
        self.bottom_shade = tuple(style_cfg.get("bottom_shade", p["bottom_shade"]))
        self.shoe = tuple(style_cfg.get("shoe_color", p["shoe"]))
        self.shoe_shade = tuple(style_cfg.get("shoe_shade", p.get("shoe_shade", p["shoe"])))
        self.eye_color = tuple(style_cfg.get("eye_color", p.get("eye", (80, 50, 30))))
        self.lip_color = tuple(style_cfg.get("lip_color", p.get("lip", (130, 110, 140))))
        self.brow_color = tuple(style_cfg.get("brow_color", p.get("brow", (50, 40, 30))))
        self.outline = style_cfg.get("outline", True)

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

        kps = keypoints[:17].astype(np.float64)
        torso_w = np.linalg.norm(kps[5] - kps[6])
        if torso_w < 3:
            return canvas

        H, W = canvas.shape[:2]

        # Layer for alpha compositing
        layer = np.zeros((H, W, 4), dtype=np.uint8)

        # Determine facing
        facing_right = kps[6][0] > kps[5][0]

        # Draw order: back limbs → torso → front limbs → head
        if facing_right:
            back_arm = (5, 7, 9)
            front_arm = (6, 8, 10)
            back_leg = (11, 13, 15)
            front_leg = (12, 14, 16)
        else:
            back_arm = (6, 8, 10)
            front_arm = (5, 7, 9)
            back_leg = (12, 14, 16)
            front_leg = (11, 13, 15)

        tw = torso_w
        lw = max(int(tw * 0.11), 3)  # limb width

        # 1. Back arm
        self._draw_arm(layer, kps, *back_arm, tw, lw, shade=0.85)
        # 2. Back leg
        self._draw_leg(layer, kps, *back_leg, tw, lw, shade=0.9)
        # 3. Torso
        self._draw_torso(layer, kps, tw)
        # 4. Front leg
        self._draw_leg(layer, kps, *front_leg, tw, lw, shade=1.0)
        # 5. Front arm
        self._draw_arm(layer, kps, *front_arm, tw, lw, shade=1.0)
        # 6. Head (always on top)
        self._draw_head(layer, kps, tw)

        # Composite layer onto canvas
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        mask = layer[:, :, 3] > 0
        if mask.any():
            canvas[mask] = (
                layer[:, :, :3][mask].astype(np.float32) * alpha[mask] +
                canvas[mask].astype(np.float32) * (1 - alpha[mask])
            ).astype(np.uint8)

        return canvas

    def _draw_smooth_limb(self, layer, p1, p2, width, color, shade_color,
                          shade=1.0, caps=True):
        """Draw an anti-aliased limb with gradient and rounded caps."""
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1:
            return

        # Apply shade multiplier
        if shade < 1.0:
            color = tuple(int(c * shade) for c in color)
            shade_color = tuple(int(c * shade) for c in shade_color)

        w = max(width, 2)
        perp = np.array([-d[1], d[0]]) / length

        # Tapered width: slightly narrower at p2
        w1 = w / 2
        w2 = w / 2 * 0.85

        # Build polygon with slight taper
        pts = np.array([
            p1 + perp * w1,
            p2 + perp * w2,
            p2 - perp * w2,
            p1 - perp * w1,
        ], dtype=np.int32)

        # Fill with base color
        cv2.fillConvexPoly(layer, pts, (*color, 255), cv2.LINE_AA)

        # Gradient shading along length
        # Draw a darker strip on one side
        shade_pts = np.array([
            p1 - perp * w1,
            p2 - perp * w2,
            p2 - perp * w2 * 0.3,
            p1 - perp * w1 * 0.3,
        ], dtype=np.int32)
        overlay = np.zeros_like(layer)
        cv2.fillConvexPoly(overlay, shade_pts, (*shade_color, 80), cv2.LINE_AA)
        mask = overlay[:, :, 3] > 0
        if mask.any():
            a = overlay[:, :, 3:4][mask].astype(np.float32) / 255.0
            layer[:, :, :3][mask] = (
                overlay[:, :, :3][mask].astype(np.float32) * a +
                layer[:, :, :3][mask].astype(np.float32) * (1 - a)
            ).astype(np.uint8)
            layer[:, :, 3][mask] = np.maximum(layer[:, :, 3][mask], overlay[:, :, 3][mask])

        if caps:
            cv2.circle(layer, tuple(p1.astype(int)), int(w1),
                       (*color, 255), -1, cv2.LINE_AA)
            cv2.circle(layer, tuple(p2.astype(int)), int(w2),
                       (*color, 255), -1, cv2.LINE_AA)

        if self.outline:
            cv2.polylines(layer, [pts], True, (*shade_color, 200), 1, cv2.LINE_AA)

    def _draw_arm(self, layer, kps, shoulder, elbow, wrist, tw, lw, shade=1.0):
        """Draw a full arm: upper arm (sleeved) + forearm (skin) + hand."""
        # Upper arm with sleeve
        self._draw_smooth_limb(layer, kps[shoulder], kps[elbow],
                                int(lw * 1.15), self.sleeve, self.top_shade,
                                shade=shade)
        # Forearm — skin
        self._draw_smooth_limb(layer, kps[elbow], kps[wrist],
                                lw, self.skin, self.skin_shade,
                                shade=shade)
        # Hand
        hand_r = max(int(lw * 0.5), 2)
        sk = self.skin if shade >= 1.0 else tuple(int(c * shade) for c in self.skin)
        cv2.circle(layer, tuple(kps[wrist].astype(int)), hand_r,
                   (*sk, 255), -1, cv2.LINE_AA)
        # Subtle finger lines
        d = kps[wrist] - kps[elbow]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        for fi in range(3):
            offset = (fi - 1) * hand_r * 0.5
            perp = np.array([-d_norm[1], d_norm[0]])
            fp = kps[wrist] + d_norm * hand_r * 0.8 + perp * offset
            cv2.line(layer, tuple(kps[wrist].astype(int)), tuple(fp.astype(int)),
                     (*self.skin_shade, 200), 1, cv2.LINE_AA)

    def _draw_leg(self, layer, kps, hip, knee, ankle, tw, lw, shade=1.0):
        """Draw a full leg: thigh + shin + shoe."""
        # Thigh
        self._draw_smooth_limb(layer, kps[hip], kps[knee],
                                int(lw * 1.3), self.bottom, self.bottom_shade,
                                shade=shade)
        # Shin
        self._draw_smooth_limb(layer, kps[knee], kps[ankle],
                                int(lw * 1.1), self.bottom, self.bottom_shade,
                                shade=shade)
        # Shoe
        shoe_w = int(lw * 1.0)
        shoe_h = int(lw * 0.5)
        ankle_pt = kps[ankle].astype(int)
        # Shoe is an ellipse angled slightly from shin direction
        shin_angle = _angle_deg(kps[knee], kps[ankle])
        sh = self.shoe if shade >= 1.0 else tuple(int(c * shade) for c in self.shoe)
        cv2.ellipse(layer, tuple(ankle_pt), (shoe_w, shoe_h),
                    shin_angle * 0.3, 0, 360, (*sh, 255), -1, cv2.LINE_AA)
        if self.outline:
            sh_s = self.shoe_shade if shade >= 1.0 else tuple(int(c*shade) for c in self.shoe_shade)
            cv2.ellipse(layer, tuple(ankle_pt), (shoe_w, shoe_h),
                        shin_angle * 0.3, 0, 360, (*sh_s, 200), 1, cv2.LINE_AA)

    def _draw_torso(self, layer, kps, tw):
        """Draw torso as a filled trapezoid with gradient."""
        ls, rs = kps[5].astype(int), kps[6].astype(int)
        lh, rh = kps[11].astype(int), kps[12].astype(int)

        # Slightly wider shoulders
        shoulder_expand = int(tw * 0.04)
        ls_out = ls.copy()
        rs_out = rs.copy()
        ls_out[0] -= shoulder_expand
        rs_out[0] += shoulder_expand

        # Slightly narrower hips
        hip_shrink = int(tw * 0.01)
        lh_in = lh.copy()
        rh_in = rh.copy()
        lh_in[0] += hip_shrink
        rh_in[0] -= hip_shrink

        pts = np.array([ls_out, rs_out, rh_in, lh_in], dtype=np.int32)
        cv2.fillConvexPoly(layer, pts, (*self.top, 255), cv2.LINE_AA)

        # Shading: darker on left side (vectorized)
        H, W = layer.shape[:2]
        x_min = max(pts[:, 0].min(), 0)
        x_max = min(pts[:, 0].max(), W - 1)
        if x_max > x_min:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            shade_end = min(x_min + (x_max - x_min) // 3, x_max)
            if shade_end > x_min:
                x_range = np.arange(x_min, shade_end)
                t_vals = 1.0 - (x_range - x_min) / max(shade_end - x_min, 1)
                for xi, x in enumerate(x_range):
                    col_mask = mask[:, x] > 0
                    for c in range(3):
                        layer[col_mask, x, c] = np.clip(
                            layer[col_mask, x, c].astype(float) * (1 - t_vals[xi] * 0.2), 0, 255
                        ).astype(np.uint8)

        if self.outline:
            cv2.polylines(layer, [pts], True, (*self.top_shade, 200), 1, cv2.LINE_AA)

        # Collar / neckline
        neck = ((kps[5] + kps[6]) / 2).astype(int)
        collar_w = int(tw * 0.12)
        cv2.ellipse(layer, tuple(neck), (collar_w, collar_w // 2),
                    0, 0, 180, (*self.skin, 255), -1, cv2.LINE_AA)

    def _draw_head(self, layer, kps, tw):
        """Draw detailed head with face, hair, eyes, mouth."""
        nose = kps[0].astype(np.float64)
        neck = ((kps[5] + kps[6]) / 2)

        head_r = int(tw * 0.22)
        head_center = nose.astype(int)

        # Neck
        neck_w = max(int(tw * 0.1), 2)
        neck_top = nose.copy()
        neck_top[1] += head_r * 0.7
        self._draw_smooth_limb(layer, neck_top, neck, neck_w,
                                self.skin, self.skin_shade, caps=False)

        # Hair (behind face for long hair)
        if self.hair_style == "long":
            hair_pts = np.array([
                [head_center[0] - int(head_r * 1.15), head_center[1]],
                [head_center[0] - int(head_r * 0.5), head_center[1] - int(head_r * 1.15)],
                [head_center[0] + int(head_r * 0.5), head_center[1] - int(head_r * 1.15)],
                [head_center[0] + int(head_r * 1.15), head_center[1]],
                [head_center[0] + int(head_r * 0.85), head_center[1] + int(head_r * 1.4)],
                [head_center[0] - int(head_r * 0.85), head_center[1] + int(head_r * 1.4)],
            ], dtype=np.int32)
            cv2.fillConvexPoly(layer, hair_pts, (*self.hair_color, 255), cv2.LINE_AA)

        # Face circle
        cv2.circle(layer, tuple(head_center), head_r,
                   (*self.skin, 255), -1, cv2.LINE_AA)
        # Cheek blush (subtle)
        blush_r = int(head_r * 0.25)
        blush_color = _lerp(self.skin, (180, 160, 200), 0.15)
        cv2.circle(layer, (head_center[0] - int(head_r * 0.35),
                           head_center[1] + int(head_r * 0.15)),
                   blush_r, (*blush_color, 40), -1, cv2.LINE_AA)
        cv2.circle(layer, (head_center[0] + int(head_r * 0.35),
                           head_center[1] + int(head_r * 0.15)),
                   blush_r, (*blush_color, 40), -1, cv2.LINE_AA)

        # Short hair (on top of face)
        if self.hair_style == "short":
            cv2.ellipse(layer, (head_center[0], head_center[1] - int(head_r * 0.1)),
                        (int(head_r * 1.05), int(head_r * 0.9)),
                        0, 180, 360, (*self.hair_color, 255), -1, cv2.LINE_AA)

        # Eyes
        eye_y = head_center[1] - int(head_r * 0.1)
        eye_dx = int(head_r * 0.28)
        eye_w = max(int(head_r * 0.16), 3)
        eye_h = max(int(head_r * 0.1), 2)

        for side in [-1, 1]:
            ex = head_center[0] + side * eye_dx
            # Eye white
            cv2.ellipse(layer, (ex, eye_y), (eye_w, eye_h),
                        0, 0, 360, (255, 255, 255, 255), -1, cv2.LINE_AA)
            # Iris
            iris_r = max(eye_h - 1, 1)
            cv2.circle(layer, (ex, eye_y), iris_r,
                       (*self.eye_color, 255), -1, cv2.LINE_AA)
            # Pupil
            pupil_r = max(iris_r // 2, 1)
            cv2.circle(layer, (ex, eye_y), pupil_r,
                       (15, 10, 8, 255), -1, cv2.LINE_AA)
            # Eye highlight
            cv2.circle(layer, (ex - pupil_r, eye_y - pupil_r),
                       max(pupil_r // 2, 1),
                       (255, 255, 255, 200), -1, cv2.LINE_AA)
            # Eyebrow
            brow_y = eye_y - int(head_r * 0.18)
            brow_pts = np.array([
                [ex - eye_w, brow_y],
                [ex + eye_w, brow_y - 2],
            ], dtype=np.int32)
            cv2.line(layer, tuple(brow_pts[0]), tuple(brow_pts[1]),
                     (*self.brow_color, 255), max(eye_h // 3, 1), cv2.LINE_AA)
            # Eyelash line
            cv2.ellipse(layer, (ex, eye_y), (eye_w, eye_h),
                        0, 0, 360, (*self.brow_color, 200), 1, cv2.LINE_AA)

        # Nose (subtle)
        nose_tip_y = head_center[1] + int(head_r * 0.12)
        cv2.line(layer, (head_center[0], head_center[1]),
                 (head_center[0], nose_tip_y),
                 (*self.skin_shade, 180), 1, cv2.LINE_AA)
        cv2.circle(layer, (head_center[0], nose_tip_y), max(int(head_r * 0.04), 1),
                   (*self.skin_shade, 150), -1, cv2.LINE_AA)

        # Mouth
        mouth_y = head_center[1] + int(head_r * 0.32)
        mouth_w = int(head_r * 0.2)
        mouth_h = max(int(head_r * 0.06), 2)
        cv2.ellipse(layer, (head_center[0], mouth_y), (mouth_w, mouth_h),
                    0, 10, 170, (*self.lip_color, 255), -1, cv2.LINE_AA)
        # Upper lip line
        cv2.ellipse(layer, (head_center[0], mouth_y - 1), (mouth_w, max(mouth_h // 2, 1)),
                    0, 190, 350, (*self.lip_color, 200), 1, cv2.LINE_AA)

        # Face outline (subtle)
        if self.outline:
            cv2.circle(layer, tuple(head_center), head_r,
                       (*self.skin_shade, 120), 1, cv2.LINE_AA)
