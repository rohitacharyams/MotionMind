"""
Anime / cel-shaded 2D character renderer.

Inspired by VRM / Live2D character styles (like Kalidokit demos).
Clean black outlines, flat cel-shading, expressive features,
visible hands with fingers, proper feet/shoes.
"""

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# ── Appearance presets ─────────────────────────────────────────────

PRESETS = {
    "anime_girl": {
        "skin": (200, 210, 240),       # warm peach (BGR)
        "skin_shadow": (170, 180, 210),
        "hair": (100, 80, 180),        # reddish-brown
        "hair_highlight": (130, 110, 210),
        "hair_style": "long",
        "eye_color": (180, 100, 60),   # blue eyes
        "top": (180, 140, 200),        # pink top
        "top_shadow": (150, 110, 170),
        "skirt": (130, 100, 160),
        "bottom": (130, 100, 160),
        "bottom_shadow": (100, 70, 130),
        "shoe": (80, 60, 50),
        "shoe_accent": (120, 100, 90),
        "outline": (40, 30, 25),
        "lip": (160, 150, 210),
        "blush": (175, 180, 230),
        "gender": "female",
    },
    "anime_boy": {
        "skin": (195, 205, 235),
        "skin_shadow": (165, 175, 200),
        "hair": (50, 40, 30),          # dark hair
        "hair_highlight": (80, 65, 50),
        "hair_style": "short_messy",
        "eye_color": (130, 90, 50),
        "top": (160, 120, 80),         # blue jacket
        "top_shadow": (130, 90, 55),
        "skirt": None,
        "bottom": (65, 55, 45),        # dark pants
        "bottom_shadow": (40, 35, 28),
        "shoe": (35, 30, 25),
        "shoe_accent": (60, 55, 45),
        "outline": (35, 28, 22),
        "lip": (170, 175, 215),
        "blush": (180, 185, 225),
        "gender": "male",
    },
    "casual_girl": {
        "skin": (190, 200, 230),
        "skin_shadow": (160, 170, 195),
        "hair": (40, 30, 20),
        "hair_highlight": (70, 55, 40),
        "hair_style": "ponytail",
        "eye_color": (60, 120, 80),    # green eyes
        "top": (80, 200, 230),         # yellow top
        "top_shadow": (55, 170, 200),
        "skirt": None,
        "bottom": (90, 80, 70),
        "bottom_shadow": (60, 52, 42),
        "shoe": (230, 230, 240),
        "shoe_accent": (200, 200, 210),
        "outline": (40, 30, 25),
        "lip": (155, 160, 210),
        "blush": (170, 175, 220),
        "gender": "female",
    },
    "cool_guy": {
        "skin": (180, 195, 225),
        "skin_shadow": (150, 162, 190),
        "hair": (25, 20, 15),
        "hair_highlight": (55, 45, 35),
        "hair_style": "short_messy",
        "eye_color": (90, 70, 50),
        "top": (45, 42, 38),           # black jacket
        "top_shadow": (28, 25, 22),
        "skirt": None,
        "bottom": (55, 48, 42),
        "bottom_shadow": (32, 28, 22),
        "shoe": (25, 22, 18),
        "shoe_accent": (45, 40, 32),
        "outline": (30, 22, 18),
        "lip": (165, 170, 210),
        "blush": (175, 180, 218),
        "gender": "male",
    },
}


class HQCharacterStyle:
    """Anime-style 2D character from COCO-17 keypoints."""

    PRESETS = list(PRESETS.keys())

    def __init__(self, config: dict):
        s = config.get("avatar", {}).get("styles", {}).get("hqchar", {})
        pname = s.get("preset", "anime_girl")
        p = PRESETS.get(pname, PRESETS["anime_girl"])

        for key in p:
            setattr(self, key, p[key])
        # Override from config
        for key in ("skin", "hair", "top", "bottom", "shoe", "outline"):
            if key + "_color" in s:
                setattr(self, key, tuple(s[key + "_color"]))

        self._outline_w = 2  # will scale with body size

    def render(self, canvas, keypoints, scores=None, min_score=0.3):
        K = len(keypoints)
        if K < 17:
            return canvas
        kps = keypoints[:17].astype(np.float64)
        tw = np.linalg.norm(kps[5] - kps[6])
        if tw < 5:
            return canvas

        H, W = canvas.shape[:2]

        # Body scale from 8-head canon
        mid_sh = (kps[5] + kps[6]) / 2
        mid_hp = (kps[11] + kps[12]) / 2
        body_h = (np.linalg.norm(kps[0] - mid_sh) +
                  np.linalg.norm(mid_sh - mid_hp) +
                  np.linalg.norm(mid_hp - (kps[13] + kps[14]) / 2) +
                  np.linalg.norm((kps[13] + kps[14]) / 2 - (kps[15] + kps[16]) / 2))
        hu = max(body_h / 7.5, 4)  # head-unit (slightly bigger heads for anime)

        self._outline_w = max(int(hu * 0.04), 2)
        self._hu = hu

        facing_right = kps[6][0] > kps[5][0]

        # Layer order: back arm → back leg → hair-back → torso → front leg → front arm → head → hair-front
        ba = (5, 7, 9) if facing_right else (6, 8, 10)
        fa = (6, 8, 10) if facing_right else (5, 7, 9)
        bl = (11, 13, 15) if facing_right else (12, 14, 16)
        fl = (12, 14, 16) if facing_right else (11, 13, 15)

        # 1. Back arm
        self._draw_arm(canvas, kps, *ba, hu, H, W, is_back=True)
        # 2. Back leg + foot
        self._draw_leg(canvas, kps, *bl, hu, H, W, is_back=True)
        # 3. Hair behind head (long styles)
        if self.hair_style in ("long", "ponytail"):
            self._draw_hair_back(canvas, kps, hu, H, W)
        # 4. Torso + neck
        self._draw_torso(canvas, kps, hu, H, W)
        # 5. Front leg + foot
        self._draw_leg(canvas, kps, *fl, hu, H, W, is_back=False)
        # 6. Front arm + hand
        self._draw_arm(canvas, kps, *fa, hu, H, W, is_back=False)
        # 7. Head + face
        self._draw_head(canvas, kps, hu, H, W)
        # 8. Hair front
        self._draw_hair_front(canvas, kps, hu, H, W)

        return canvas

    # ── Smooth limb polygon ────────────────────────────────────────

    def _limb_poly(self, joints, widths, n=30):
        """Smooth closed polygon for a limb chain using CubicSpline."""
        nj = len(joints)
        if nj < 2:
            return None
        t = np.linspace(0, 1, nj)
        tf = np.linspace(0, 1, n)

        bc = 'clamped' if nj < 4 else 'natural'
        cx = CubicSpline(t, joints[:, 0], bc_type=bc)(tf)
        cy = CubicSpline(t, joints[:, 1], bc_type=bc)(tf)
        wf = np.interp(tf, t, widths)

        dx = np.gradient(cx)
        dy = np.gradient(cy)
        ln = np.sqrt(dx**2 + dy**2) + 1e-8
        px, py = -dy / ln, dx / ln

        left = np.column_stack([cx + px * wf, cy + py * wf])
        right = np.column_stack([cx - px * wf, cy - py * wf])
        # Close with rounded caps
        tip_pts = self._semicircle(left[-1], right[-1], cx[-1], cy[-1], 8)
        base_pts = self._semicircle(right[0], left[0], cx[0], cy[0], 8)
        return np.vstack([left, tip_pts, right[::-1], base_pts]).astype(np.int32)

    def _semicircle(self, p1, p2, cx, cy, n=8):
        """Generate semicircle points between p1 and p2 around center."""
        pts = []
        for i in range(1, n):
            t = i / n
            angle = np.pi * t
            mid = (p1 + p2) / 2
            radius = np.linalg.norm(p1 - p2) / 2
            dx = p2 - p1
            ux = dx / (np.linalg.norm(dx) + 1e-8)
            uy = np.array([-ux[1], ux[0]])
            # Determine which direction "out" is
            to_center = np.array([cx, cy]) - mid
            if np.dot(to_center, uy) > 0:
                uy = -uy
            pt = mid + ux * radius * np.cos(angle) + uy * radius * np.sin(angle)
            pts.append(pt)
        return np.array(pts) if pts else np.empty((0, 2))

    # ── Cel-shade fill ─────────────────────────────────────────────

    def _cel_fill(self, canvas, contour, base, shadow, H, W,
                  outline_color=None, shadow_dir=(0.4, 0.7)):
        """Fill contour with 2-tone cel shading + outline."""
        if contour is None or len(contour) < 3:
            return
        oc = outline_color or self.outline

        # Fill base color
        cv2.fillPoly(canvas, [contour], base, cv2.LINE_AA)

        # Shadow: bottom-right half gets shadow color
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Shadow region: distance from edge + directional
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = mask[y0:y1, x0:x1]

        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
        dmax = dist.max()
        if dmax < 2:
            # Too small for shadow
            cv2.polylines(canvas, [contour], True, oc, self._outline_w, cv2.LINE_AA)
            return

        # Normalized coordinates within bounding box
        ch, cw = crop.shape
        yy, xx = np.mgrid[0:ch, 0:cw]
        # Shadow line: diagonal from top-left to bottom-right
        sdx, sdy = shadow_dir
        shadow_t = (xx / max(cw, 1)) * sdx + (yy / max(ch, 1)) * sdy
        # Shadow where: deeper into shape AND in shadow direction
        depth_ratio = dist / dmax
        # Shadow threshold: shadow region is where shadow_t > 0.45 and we're deep enough
        shadow_mask_crop = (shadow_t > 0.4) & (depth_ratio > 0.15) & (crop > 0)

        # Smooth the shadow edge
        shadow_u8 = shadow_mask_crop.astype(np.uint8) * 255
        shadow_u8 = cv2.GaussianBlur(shadow_u8, (5, 5), 2)
        shadow_blend = shadow_u8.astype(np.float32) / 255.0

        for c in range(3):
            region = canvas[y0:y1, x0:x1, c]
            blended = region.astype(np.float32) * (1 - shadow_blend * 0.5) + shadow[c] * shadow_blend * 0.5
            region[crop > 0] = np.clip(blended, 0, 255).astype(np.uint8)[crop > 0]

        # Outline
        cv2.polylines(canvas, [contour], True, oc, self._outline_w, cv2.LINE_AA)

    # ── Arms + Hands ───────────────────────────────────────────────

    def _draw_arm(self, canvas, kps, sh_i, el_i, wr_i, hu, H, W, is_back=False):
        shade_mult = 0.85 if is_back else 1.0

        def sc(color):
            return tuple(int(c * shade_mult) for c in color)

        shoulder = kps[sh_i]
        elbow = kps[el_i]
        wrist = kps[wr_i]

        # Upper arm (clothing)
        mid_up = (shoulder + elbow) / 2
        upper_j = np.array([shoulder, mid_up, elbow])
        w_sh = hu * 0.22
        w_el = hu * 0.18
        upper_w = np.array([w_sh, (w_sh + w_el) / 2, w_el])
        upper_c = self._limb_poly(upper_j, upper_w, 20)
        self._cel_fill(canvas, upper_c, sc(self.top), sc(self.top_shadow), H, W)

        # Forearm (skin)
        mid_fore = (elbow + wrist) / 2
        fore_j = np.array([elbow, mid_fore, wrist])
        w_wr = hu * 0.14
        fore_w = np.array([w_el, (w_el + w_wr) / 2, w_wr])
        fore_c = self._limb_poly(fore_j, fore_w, 20)
        self._cel_fill(canvas, fore_c, sc(self.skin), sc(self.skin_shadow), H, W)

        # Hand
        self._draw_hand(canvas, kps, el_i, wr_i, hu, H, W, shade_mult)

    def _draw_hand(self, canvas, kps, el_i, wr_i, hu, H, W, shade_mult=1.0):
        """Draw a proper hand with visible fingers."""
        wrist = kps[wr_i]
        elbow = kps[el_i]

        hand_dir = wrist - elbow
        hand_dir = hand_dir / (np.linalg.norm(hand_dir) + 1e-8)
        perp = np.array([-hand_dir[1], hand_dir[0]])

        def sc(color):
            return tuple(int(c * shade_mult) for c in color)

        palm_center = wrist + hand_dir * hu * 0.15
        palm_w = hu * 0.16
        palm_h = hu * 0.18

        # Palm as rounded rectangle
        p1 = palm_center - perp * palm_w - hand_dir * palm_h * 0.3
        p2 = palm_center + perp * palm_w - hand_dir * palm_h * 0.3
        p3 = palm_center + perp * palm_w + hand_dir * palm_h * 0.7
        p4 = palm_center - perp * palm_w + hand_dir * palm_h * 0.7
        palm_pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(canvas, [palm_pts], sc(self.skin), cv2.LINE_AA)

        # 4 fingers
        finger_len = hu * 0.18
        finger_w = hu * 0.035
        finger_offsets = [-0.6, -0.2, 0.2, 0.6]
        for i, off in enumerate(finger_offsets):
            # Each finger slightly different length
            fl = finger_len * (1.0 - abs(off) * 0.15)
            base = palm_center + hand_dir * palm_h * 0.6 + perp * palm_w * off
            tip = base + hand_dir * fl
            # Slight splay outward
            tip = tip + perp * off * hu * 0.04

            # Draw finger as thin rounded shape
            f_j = np.array([base, (base + tip) / 2, tip])
            f_w = np.array([finger_w, finger_w * 0.9, finger_w * 0.5])
            f_c = self._limb_poly(f_j, f_w, 10)
            if f_c is not None:
                cv2.fillPoly(canvas, [f_c], sc(self.skin), cv2.LINE_AA)
                cv2.polylines(canvas, [f_c], True, sc(self.outline), 1, cv2.LINE_AA)

        # Thumb (thicker, off to the side)
        thumb_base = palm_center - perp * palm_w * 0.8 + hand_dir * palm_h * 0.1
        thumb_dir = hand_dir * 0.5 - perp * 0.8
        thumb_dir = thumb_dir / (np.linalg.norm(thumb_dir) + 1e-8)
        thumb_tip = thumb_base + thumb_dir * finger_len * 0.8
        thumb_mid = (thumb_base + thumb_tip) / 2
        t_j = np.array([thumb_base, thumb_mid, thumb_tip])
        t_w = np.array([finger_w * 1.3, finger_w * 1.1, finger_w * 0.6])
        t_c = self._limb_poly(t_j, t_w, 10)
        if t_c is not None:
            cv2.fillPoly(canvas, [t_c], sc(self.skin), cv2.LINE_AA)
            cv2.polylines(canvas, [t_c], True, sc(self.outline), 1, cv2.LINE_AA)

        # Palm outline
        cv2.polylines(canvas, [palm_pts], True, sc(self.outline), self._outline_w, cv2.LINE_AA)

    # ── Legs + Feet ────────────────────────────────────────────────

    def _draw_leg(self, canvas, kps, hp_i, kn_i, an_i, hu, H, W, is_back=False):
        shade_mult = 0.88 if is_back else 1.0

        def sc(color):
            return tuple(int(c * shade_mult) for c in color)

        hip = kps[hp_i]
        knee = kps[kn_i]
        ankle = kps[an_i]

        # Full leg contour
        mid_thigh = (hip + knee) / 2
        mid_calf = (knee + ankle) / 2
        joints = np.array([hip, mid_thigh, knee, mid_calf, ankle])
        w_hip = hu * 0.25
        w_knee = hu * 0.18
        w_ankle = hu * 0.14
        widths = np.array([w_hip, (w_hip + w_knee) / 2, w_knee,
                           (w_knee + w_ankle) / 2, w_ankle])

        leg_c = self._limb_poly(joints, widths, 30)
        self._cel_fill(canvas, leg_c, sc(self.bottom), sc(self.bottom_shadow), H, W)

        # Foot / shoe
        self._draw_foot(canvas, kps, kn_i, an_i, hu, H, W, shade_mult)

    def _draw_foot(self, canvas, kps, kn_i, an_i, hu, H, W, shade_mult=1.0):
        """Draw a proper shoe/foot with volume."""
        ankle = kps[an_i]
        knee = kps[kn_i]

        def sc(color):
            return tuple(int(c * shade_mult) for c in color)

        shin_dir = ankle - knee
        shin_dir = shin_dir / (np.linalg.norm(shin_dir) + 1e-8)
        perp = np.array([-shin_dir[1], shin_dir[0]])

        # Shoe shape: extends forward from ankle
        shoe_len = hu * 0.35
        shoe_h = hu * 0.14
        ankle_pt = ankle

        # Determine forward direction (slightly forward + down)
        # Foot points mostly horizontal, slightly in the direction the leg leans
        foot_dir = np.array([perp[0], abs(shin_dir[1]) * 0.3 + 0.7])
        foot_dir = foot_dir / (np.linalg.norm(foot_dir) + 1e-8)
        foot_perp = np.array([-foot_dir[1], foot_dir[0]])

        # Shoe contour points
        heel = ankle_pt - foot_dir * shoe_len * 0.3
        toe = ankle_pt + foot_dir * shoe_len * 0.7
        # Top of shoe
        top_heel = heel - foot_perp * shoe_h * 0.6
        top_mid = ankle_pt - foot_perp * shoe_h * 0.8
        top_toe = toe - foot_perp * shoe_h * 0.3
        # Bottom of shoe
        bot_heel = heel + foot_perp * shoe_h * 0.4
        bot_mid = ankle_pt + foot_perp * shoe_h * 0.3
        bot_toe = toe + foot_perp * shoe_h * 0.2

        shoe_pts = np.array([
            top_heel, top_mid, top_toe,
            bot_toe, bot_mid, bot_heel
        ], dtype=np.int32)

        cv2.fillPoly(canvas, [shoe_pts], sc(self.shoe), cv2.LINE_AA)
        # Sole line
        sole_pts = np.array([bot_heel, bot_mid, bot_toe], dtype=np.int32)
        cv2.polylines(canvas, [sole_pts], False, sc(self.shoe_accent),
                      max(int(hu * 0.03), 2), cv2.LINE_AA)
        # Outline
        cv2.polylines(canvas, [shoe_pts], True, sc(self.outline),
                      self._outline_w, cv2.LINE_AA)

    # ── Torso ──────────────────────────────────────────────────────

    def _draw_torso(self, canvas, kps, hu, H, W):
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        mid_sh = (ls + rs) / 2
        mid_hp = (lh + rh) / 2

        sh_dir = (rs - ls)
        sh_dir_n = sh_dir / (np.linalg.norm(sh_dir) + 1e-8)
        hp_dir = (rh - lh)
        hp_dir_n = hp_dir / (np.linalg.norm(hp_dir) + 1e-8)

        sh_half = np.linalg.norm(rs - ls) / 2 + hu * 0.08
        hp_half = np.linalg.norm(rh - lh) / 2 + hu * 0.04
        waist_pinch = hu * 0.10 if self.gender == "female" else hu * 0.02
        waist_half = min(sh_half, hp_half) - waist_pinch

        waist_center = mid_sh * 0.38 + mid_hp * 0.62
        waist_dir_n = (sh_dir_n + hp_dir_n)
        waist_dir_n = waist_dir_n / (np.linalg.norm(waist_dir_n) + 1e-8)

        # 8-point torso for smoother shape
        mid_upper = mid_sh * 0.75 + mid_hp * 0.25
        mid_lower = mid_sh * 0.25 + mid_hp * 0.75
        mid_u_half = (sh_half * 0.7 + waist_half * 0.3)
        mid_l_half = (hp_half * 0.7 + waist_half * 0.3)
        mid_u_dir = (sh_dir_n * 0.7 + waist_dir_n * 0.3)
        mid_u_dir = mid_u_dir / (np.linalg.norm(mid_u_dir) + 1e-8)
        mid_l_dir = (hp_dir_n * 0.7 + waist_dir_n * 0.3)
        mid_l_dir = mid_l_dir / (np.linalg.norm(mid_l_dir) + 1e-8)

        pts = np.array([
            mid_sh - sh_dir_n * sh_half,           # left shoulder
            mid_sh + sh_dir_n * sh_half,           # right shoulder
            mid_upper + mid_u_dir * mid_u_half,    # right upper
            waist_center + waist_dir_n * waist_half,  # right waist
            mid_lower + mid_l_dir * mid_l_half,    # right lower
            mid_hp + hp_dir_n * hp_half,           # right hip
            mid_hp - hp_dir_n * hp_half,           # left hip
            mid_lower - mid_l_dir * mid_l_half,    # left lower
            waist_center - waist_dir_n * waist_half,  # left waist
            mid_upper - mid_u_dir * mid_u_half,    # left upper
        ], dtype=np.int32)

        self._cel_fill(canvas, pts, self.top, self.top_shadow, H, W)

        # Collar area (skin)
        neck_base = mid_sh
        collar_w = int(hu * 0.28)
        collar_h = int(hu * 0.12)
        cv2.ellipse(canvas, tuple(neck_base.astype(int)), (collar_w, collar_h),
                    0, 10, 170, self.skin, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, tuple(neck_base.astype(int)), (collar_w, collar_h),
                    0, 10, 170, self.outline, self._outline_w, cv2.LINE_AA)

        # Neck
        head_pos = kps[0]
        neck_top = head_pos + np.array([0, hu * 0.45])
        neck_w = hu * 0.15
        n_j = np.array([neck_top, mid_sh])
        n_w = np.array([neck_w, neck_w * 1.2])
        neck_c = self._limb_poly(n_j, n_w, 12)
        self._cel_fill(canvas, neck_c, self.skin, self.skin_shadow, H, W)

    # ── Head ───────────────────────────────────────────────────────

    def _draw_head(self, canvas, kps, hu, H, W):
        hc = kps[0].astype(int)
        hr = int(hu * 0.85)

        # Face shape: slightly elongated oval
        face_pts = []
        for i in range(48):
            angle = 2 * np.pi * i / 48
            rx = hr * 0.90
            ry = hr * 1.0
            # Slightly narrower chin
            if angle > np.pi * 0.3 and angle < np.pi * 0.7:
                rx *= 0.88
            x = hc[0] + rx * np.cos(angle)
            y = hc[1] + ry * np.sin(angle)
            face_pts.append([x, y])
        face_pts = np.array(face_pts, dtype=np.int32)

        # Fill face with cel-shading
        self._cel_fill(canvas, face_pts, self.skin, self.skin_shadow, H, W,
                       shadow_dir=(0.3, 0.6))

        # ── Eyes (anime style: large, expressive) ──
        eye_y = hc[1] - int(hr * 0.05)
        eye_dx = int(hr * 0.30)
        eye_w = max(int(hr * 0.22), 4)
        eye_h = max(int(hr * 0.17), 3)

        for side in (-1, 1):
            ex = hc[0] + side * eye_dx

            # Eye white
            cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h),
                        0, 0, 360, (250, 248, 245), -1, cv2.LINE_AA)

            # Iris (large, anime-style)
            ir_r = max(int(eye_h * 0.85), 3)
            cv2.circle(canvas, (ex, eye_y + 1), ir_r, self.eye_color, -1, cv2.LINE_AA)

            # Pupil
            pr = max(ir_r // 2, 2)
            cv2.circle(canvas, (ex, eye_y + 1), pr, (15, 10, 5), -1, cv2.LINE_AA)

            # Large highlight (anime sparkle)
            hl_r = max(pr, 2)
            cv2.circle(canvas, (ex - pr + 1, eye_y - pr), hl_r,
                       (255, 255, 255), -1, cv2.LINE_AA)
            # Small secondary highlight
            cv2.circle(canvas, (ex + pr - 1, eye_y + pr - 1),
                       max(hl_r // 2, 1), (255, 255, 255), -1, cv2.LINE_AA)

            # Upper eyelid (thick line, anime style)
            cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h),
                        0, 180, 360, self.outline, max(self._outline_w, 2), cv2.LINE_AA)

            # Lower eyelid (thin)
            cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h),
                        0, 10, 170, self.outline, 1, cv2.LINE_AA)

            # Eyebrow (expressive arc)
            brow_y = eye_y - int(hr * 0.22)
            brow_pts = []
            for bi in range(10):
                bx = ex - eye_w + bi * eye_w * 2 // 9
                by = brow_y - int(3.0 * np.sin(np.pi * bi / 9))
                brow_pts.append([bx, by])
            brow_arr = np.array(brow_pts, dtype=np.int32)
            cv2.polylines(canvas, [brow_arr], False, self.outline,
                          max(int(hr * 0.05), 2), cv2.LINE_AA)

        # ── Blush (anime cheek blush) ──
        blush_y = eye_y + int(hr * 0.18)
        blush_r = max(int(hr * 0.12), 3)
        blush_layer = canvas.copy()
        for side in (-1, 1):
            bx = hc[0] + side * int(hr * 0.38)
            cv2.ellipse(blush_layer, (bx, blush_y), (blush_r, blush_r // 2),
                        0, 0, 360, self.blush, -1, cv2.LINE_AA)
        cv2.addWeighted(blush_layer, 0.3, canvas, 0.7, 0, canvas)

        # ── Nose (subtle, anime-style) ──
        nose_y = hc[1] + int(hr * 0.12)
        cv2.line(canvas, (hc[0], nose_y), (hc[0] - 1, nose_y + max(int(hr * 0.06), 2)),
                 self.skin_shadow, max(int(hr * 0.03), 1), cv2.LINE_AA)

        # ── Mouth ──
        mouth_y = hc[1] + int(hr * 0.32)
        mouth_w = max(int(hr * 0.16), 3)
        # Simple curved line for closed mouth
        cv2.ellipse(canvas, (hc[0], mouth_y), (mouth_w, max(int(hr * 0.04), 2)),
                    0, 10, 170, self.lip, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (hc[0], mouth_y), (mouth_w, max(int(hr * 0.04), 2)),
                    0, 0, 180, self.outline, 1, cv2.LINE_AA)

        # ── Ears (behind hair for long styles) ──
        if self.hair_style not in ("long",):
            ear_y = hc[1] + int(hr * 0.02)
            ear_w = max(int(hr * 0.10), 2)
            ear_h = max(int(hr * 0.14), 3)
            for side in (-1, 1):
                ear_x = hc[0] + side * int(hr * 0.88)
                cv2.ellipse(canvas, (ear_x, ear_y), (ear_w, ear_h),
                            0, 0, 360, self.skin, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, (ear_x, ear_y), (ear_w, ear_h),
                            0, 0, 360, self.outline, 1, cv2.LINE_AA)

    # ── Hair ───────────────────────────────────────────────────────

    def _draw_hair_back(self, canvas, kps, hu, H, W):
        """Long hair behind the head."""
        hc = kps[0].astype(int)
        hr = int(hu * 0.85)

        # Hair flows behind and below
        hair_pts = []
        # Top arc
        for i in range(20):
            angle = np.pi + np.pi * i / 19  # top half
            r = hr * 1.15
            x = hc[0] + r * np.cos(angle)
            y = hc[1] + r * np.sin(angle) - hr * 0.05
            hair_pts.append([x, y])

        # Side strands flowing down
        right_x = hc[0] + int(hr * 1.1)
        left_x = hc[0] - int(hr * 1.1)
        flow_len = hr * 1.8

        hair_pts.append([right_x, hc[1]])
        hair_pts.append([right_x + int(hr * 0.1), hc[1] + flow_len * 0.5])
        hair_pts.append([right_x + int(hr * 0.05), hc[1] + flow_len])
        # Bottom (wavy)
        for i in range(6):
            x = right_x - i * int(hr * 0.4)
            y = hc[1] + flow_len + int(hr * 0.08 * np.sin(i * 1.5))
            hair_pts.append([x, y])
        hair_pts.append([left_x + int(hr * 0.05), hc[1] + flow_len])
        hair_pts.append([left_x - int(hr * 0.1), hc[1] + flow_len * 0.5])
        hair_pts.append([left_x, hc[1]])

        hair_pts = np.array(hair_pts, dtype=np.int32)
        cv2.fillPoly(canvas, [hair_pts], self.hair, cv2.LINE_AA)

        # Strand lines
        for i in range(8):
            sx = hc[0] + (i - 4) * int(hr * 0.22)
            cv2.line(canvas, (sx, hc[1] - int(hr * 0.3)),
                     (sx + int((i - 4) * hr * 0.06), hc[1] + int(flow_len * 0.8)),
                     self.hair_highlight, 1, cv2.LINE_AA)

        cv2.polylines(canvas, [hair_pts], True, self.outline, self._outline_w, cv2.LINE_AA)

    def _draw_hair_front(self, canvas, kps, hu, H, W):
        """Hair on top / bangs."""
        hc = kps[0].astype(int)
        hr = int(hu * 0.85)

        if self.hair_style == "short_messy":
            self._draw_short_messy_hair(canvas, hc, hr, hu, H, W)
        elif self.hair_style in ("long", "ponytail"):
            self._draw_bangs(canvas, hc, hr, hu, H, W)
            if self.hair_style == "ponytail":
                self._draw_ponytail(canvas, kps, hc, hr, hu, H, W)

    def _draw_short_messy_hair(self, canvas, hc, hr, hu, H, W):
        """Short messy anime hair with spiky chunks."""
        hair_pts = []
        # Base dome
        n_spikes = 7
        for i in range(n_spikes * 3 + 1):
            angle = np.pi + np.pi * i / (n_spikes * 3)
            # Add spikiness
            spike = 1.0 + 0.12 * np.sin(i * np.pi / 1.5)
            r = hr * 1.15 * spike
            x = hc[0] + r * np.cos(angle)
            y = hc[1] + r * np.sin(angle) - hr * 0.08
            hair_pts.append([x, y])

        # Side burns
        hair_pts.append([hc[0] + int(hr * 1.05), hc[1] + int(hr * 0.2)])
        hair_pts.append([hc[0] + int(hr * 0.85), hc[1] + int(hr * 0.35)])
        # Bottom back
        hair_pts.append([hc[0] - int(hr * 0.85), hc[1] + int(hr * 0.35)])
        hair_pts.append([hc[0] - int(hr * 1.05), hc[1] + int(hr * 0.2)])

        hair_pts = np.array(hair_pts, dtype=np.int32)
        cv2.fillPoly(canvas, [hair_pts], self.hair, cv2.LINE_AA)
        cv2.polylines(canvas, [hair_pts], True, self.outline, self._outline_w, cv2.LINE_AA)

        # Highlight streak
        hl_x = hc[0] - int(hr * 0.2)
        hl_y = hc[1] - int(hr * 0.6)
        cv2.line(canvas, (hl_x, hl_y), (hl_x + int(hr * 0.15), hl_y + int(hr * 0.3)),
                 self.hair_highlight, max(int(hr * 0.06), 2), cv2.LINE_AA)

    def _draw_bangs(self, canvas, hc, hr, hu, H, W):
        """Anime bangs / fringe."""
        # Multiple bang strands
        n_bangs = 5
        bang_width = hr * 1.6 / n_bangs

        for i in range(n_bangs):
            bx = hc[0] - int(hr * 0.8) + int(i * bang_width)
            # Each bang is a triangle-ish shape
            top_y = hc[1] - int(hr * 0.85)
            # Vary length
            bot_y = hc[1] - int(hr * 0.05) + int(hr * 0.12 * np.sin(i * 1.2))

            pts = np.array([
                [bx - int(bang_width * 0.3), top_y],
                [bx + int(bang_width * 0.7), top_y],
                [bx + int(bang_width * 0.4), bot_y],
                [bx + int(bang_width * 0.1), bot_y - int(hr * 0.05)],
            ], dtype=np.int32)

            cv2.fillPoly(canvas, [pts], self.hair, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, self.outline, max(self._outline_w - 1, 1), cv2.LINE_AA)

        # Hair top volume
        top_pts = []
        for i in range(20):
            angle = np.pi + np.pi * i / 19
            r = hr * 1.18
            x = hc[0] + r * np.cos(angle)
            y = hc[1] + r * np.sin(angle) - hr * 0.08
            top_pts.append([x, y])
        top_pts = np.array(top_pts, dtype=np.int32)
        cv2.fillPoly(canvas, [top_pts], self.hair, cv2.LINE_AA)
        cv2.polylines(canvas, [top_pts], True, self.outline, self._outline_w, cv2.LINE_AA)

    def _draw_ponytail(self, canvas, kps, hc, hr, hu, H, W):
        """Ponytail flowing from back of head."""
        # Ponytail attachment point
        pt_start = np.array([hc[0] + int(hr * 0.5), hc[1] - int(hr * 0.3)], dtype=np.float64)
        pt_mid = pt_start + np.array([hr * 0.8, hr * 0.3])
        pt_end = pt_start + np.array([hr * 1.2, hr * 1.0])

        pt_j = np.array([pt_start, pt_mid, pt_end])
        pt_w = np.array([hr * 0.25, hr * 0.20, hr * 0.08])
        pt_c = self._limb_poly(pt_j, pt_w, 20)
        if pt_c is not None:
            cv2.fillPoly(canvas, [pt_c], self.hair, cv2.LINE_AA)
            cv2.polylines(canvas, [pt_c], True, self.outline, self._outline_w, cv2.LINE_AA)

        # Hair tie
        tie_pos = pt_start.astype(int)
        cv2.circle(canvas, tuple(tie_pos), max(int(hr * 0.08), 3),
                   self.outline, -1, cv2.LINE_AA)
