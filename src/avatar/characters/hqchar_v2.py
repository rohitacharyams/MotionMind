"""
Realistic 2D human character renderer.

Renders a smooth, anatomically-proportioned human silhouette from
COCO-17 keypoints each frame.  Uses:
  - Cubic B-spline body contour (scipy splprep/splev)
  - Catmull-Rom limb chains for seamless joints
  - Distance-transform shading for volume
  - Directional lighting via Sobel normals
  - Detailed face/hair with Gaussian-blur softness
  - Clothing regions with fold lines at bent joints
"""

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

# ── Appearance presets ─────────────────────────────────────────────

HQCHAR_PRESETS = {
    "realistic_male": {
        "skin": (215, 200, 185),
        "skin_shade": (190, 175, 158),
        "hair": (40, 32, 22),
        "hair_style": "short",
        "top": (145, 85, 55),
        "top_shade": (100, 55, 30),
        "bottom": (75, 65, 50),
        "bottom_shade": (45, 38, 25),
        "shoe": (35, 30, 25),
        "shoe_sole": (20, 18, 15),
        "eye": (85, 55, 30),
        "lip": (145, 120, 140),
        "brow": (55, 42, 28),
        "belt": (50, 40, 30),
        "gender": "male",
    },
    "realistic_female": {
        "skin": (210, 195, 180),
        "skin_shade": (175, 155, 140),
        "hair": (30, 22, 55),
        "hair_style": "long",
        "top": (220, 170, 190),
        "top_shade": (175, 130, 150),
        "bottom": (65, 50, 40),
        "bottom_shade": (40, 30, 22),
        "shoe": (160, 140, 180),
        "shoe_sole": (110, 90, 130),
        "eye": (130, 75, 45),
        "lip": (115, 95, 175),
        "brow": (45, 30, 50),
        "belt": (90, 70, 55),
        "gender": "female",
    },
    "dark_suit": {
        "skin": (195, 178, 158),
        "skin_shade": (155, 138, 118),
        "hair": (28, 23, 18),
        "hair_style": "short",
        "top": (48, 44, 40),
        "top_shade": (28, 25, 22),
        "bottom": (42, 38, 34),
        "bottom_shade": (25, 22, 18),
        "shoe": (22, 20, 16),
        "shoe_sole": (12, 10, 8),
        "eye": (85, 55, 30),
        "lip": (140, 118, 145),
        "brow": (38, 28, 20),
        "belt": (55, 45, 35),
        "gender": "male",
    },
    "athletic_female": {
        "skin": (200, 180, 155),
        "skin_shade": (165, 145, 120),
        "hair": (25, 18, 12),
        "hair_style": "ponytail",
        "top": (55, 190, 210),
        "top_shade": (35, 150, 170),
        "bottom": (50, 50, 55),
        "bottom_shade": (30, 30, 35),
        "shoe": (230, 230, 240),
        "shoe_sole": (180, 180, 195),
        "eye": (100, 70, 40),
        "lip": (130, 105, 160),
        "brow": (35, 25, 15),
        "belt": (40, 40, 45),
        "gender": "female",
    },
}


class HQCharacterStyle:
    """Realistic 2D human character from COCO-17 keypoints."""

    PRESETS = list(HQCHAR_PRESETS.keys())

    def __init__(self, config: dict):
        s = config.get("avatar", {}).get("styles", {}).get("hqchar", {})
        pname = s.get("preset", "realistic_male")
        p = HQCHAR_PRESETS.get(pname, HQCHAR_PRESETS["realistic_male"])

        self.skin = tuple(s.get("skin_color", p["skin"]))
        self.skin_shade = tuple(s.get("skin_shade", p["skin_shade"]))
        self.hair = tuple(s.get("hair_color", p["hair"]))
        self.hair_style = s.get("hair_style", p["hair_style"])
        self.top = tuple(s.get("top_color", p["top"]))
        self.top_shade = tuple(s.get("top_shade", p["top_shade"]))
        self.bottom = tuple(s.get("bottom_color", p["bottom"]))
        self.bottom_shade = tuple(s.get("bottom_shade", p["bottom_shade"]))
        self.shoe = tuple(s.get("shoe_color", p["shoe"]))
        self.shoe_sole = tuple(s.get("shoe_sole", p.get("shoe_sole", p["shoe"])))
        self.eye = tuple(s.get("eye_color", p.get("eye", (85, 55, 30))))
        self.lip = tuple(s.get("lip_color", p.get("lip", (140, 120, 140))))
        self.brow = tuple(s.get("brow_color", p.get("brow", (50, 40, 30))))
        self.belt_color = tuple(s.get("belt_color", p.get("belt", (50, 40, 30))))
        self.gender = s.get("gender", p.get("gender", "male"))

    # ── Public API ─────────────────────────────────────────────────

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
        tw = np.linalg.norm(kps[5] - kps[6])
        if tw < 5:
            return canvas

        H, W = canvas.shape[:2]

        # Estimate body scale from 8-head canon
        mid_sh = (kps[5] + kps[6]) / 2
        mid_hp = (kps[11] + kps[12]) / 2
        body_h = (np.linalg.norm(kps[0] - mid_sh) +
                  np.linalg.norm(mid_sh - mid_hp) +
                  np.linalg.norm(mid_hp - (kps[13] + kps[14]) / 2) +
                  np.linalg.norm((kps[13] + kps[14]) / 2 - (kps[15] + kps[16]) / 2))
        hu = max(body_h / 8.0, 3)  # head-unit

        # Width lookup (half-widths in head-units)
        def hw(ratio):
            return ratio * hu

        # ── Dynamic z-ordering per arm ──
        # Determine which arm is "in front" by checking which arm's
        # centroid crosses further toward the opposite side of the torso.
        torso_cx = (kps[5][0] + kps[6][0]) / 2
        left_arm_cx = (kps[5][0] + kps[7][0] + kps[9][0]) / 3
        right_arm_cx = (kps[6][0] + kps[8][0] + kps[10][0]) / 3
        # left_cross > 0 means left arm crosses rightward past center
        left_cross = left_arm_cx - torso_cx
        right_cross = torso_cx - right_arm_cx
        left_arm_in_front = left_cross > right_cross

        # Same logic for legs
        left_leg_cx = (kps[11][0] + kps[13][0] + kps[15][0]) / 3
        right_leg_cx = (kps[12][0] + kps[14][0] + kps[16][0]) / 3
        left_leg_cross = left_leg_cx - torso_cx
        right_leg_cross = torso_cx - right_leg_cx
        left_leg_in_front = left_leg_cross > right_leg_cross

        # Back arm / front arm (draw back first, front last)
        if left_arm_in_front:
            ba = (6, 8, 10)   # right arm in back
            fa = (5, 7, 9)    # left arm in front
        else:
            ba = (5, 7, 9)    # left arm in back
            fa = (6, 8, 10)   # right arm in front

        if left_leg_in_front:
            bl = (12, 14, 16)  # right leg in back
            fl = (11, 13, 15)  # left leg in front
        else:
            bl = (11, 13, 15)  # left leg in back
            fl = (12, 14, 16)  # right leg in front

        # 1) Back arm
        self._draw_arm(canvas, kps, *ba, hu, H, W, shade=0.82)
        # 2) Back leg
        self._draw_leg(canvas, kps, *bl, hu, H, W, shade=0.88)
        # 3) Torso
        self._draw_torso(canvas, kps, hu, H, W)
        # 4) Front leg
        self._draw_leg(canvas, kps, *fl, hu, H, W, shade=1.0)
        # 5) Front arm
        self._draw_arm(canvas, kps, *fa, hu, H, W, shade=1.0)
        # 6) Head + hair
        self._draw_head(canvas, kps, hu, H, W)

        return canvas

    # ── Smooth limb chain ──────────────────────────────────────────

    def _limb_contour(self, joints, widths, n_samples=40):
        """Build a smooth closed polygon for a limb chain.

        joints:  (N, 2) array of joint positions along the chain.
        widths:  (N,)   half-width at each joint.
        Returns: (M, 2) int32 contour polygon.
        """
        n = len(joints)
        if n < 2:
            return None

        t = np.linspace(0, 1, n)
        t_fine = np.linspace(0, 1, n_samples)

        # Interpolate centers
        if n >= 4:
            cs_x = CubicSpline(t, joints[:, 0], bc_type='natural')
            cs_y = CubicSpline(t, joints[:, 1], bc_type='natural')
        else:
            # For 2–3 points use linear + gentle smoothing
            cs_x = CubicSpline(t, joints[:, 0], bc_type='clamped')
            cs_y = CubicSpline(t, joints[:, 1], bc_type='clamped')

        cx = cs_x(t_fine)
        cy = cs_y(t_fine)

        # Interpolate widths
        w_fine = np.interp(t_fine, t, widths)

        # Perpendiculars from tangent
        dx = np.gradient(cx)
        dy = np.gradient(cy)
        lengths = np.sqrt(dx**2 + dy**2) + 1e-8
        px = -dy / lengths
        py = dx / lengths

        # Left and right contour
        left = np.column_stack([cx + px * w_fine, cy + py * w_fine])
        right = np.column_stack([cx - px * w_fine, cy - py * w_fine])

        contour = np.vstack([left, right[::-1]]).astype(np.int32)
        return contour

    def _shade_region(self, canvas, mask, base_color, shade_color,
                      light_dir=(-0.4, -0.7), ambient=0.62, diffuse=0.38):
        """Apply distance-transform + directional lighting to a masked region."""
        # Crop to bounding box for performance
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        # Pad by 2 for Sobel
        y0p, x0p = max(y0 - 2, 0), max(x0 - 2, 0)
        y1p, x1p = min(y1 + 2, mask.shape[0]), min(x1 + 2, mask.shape[1])

        crop_mask = mask[y0p:y1p, x0p:x1p]

        dist = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
        dmax = dist.max()
        if dmax < 1:
            # Flat fill fallback
            m = mask > 0
            canvas[m] = base_color[:3] if len(base_color) == 3 else base_color
            return

        dist_n = dist / dmax

        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)
        gnorm = np.sqrt(gx**2 + gy**2) + 1e-8
        nx = gx / gnorm
        ny = gy / gnorm

        lx, ly = light_dir
        light_dot = np.clip(nx * lx + ny * ly, 0, 1)

        shade_factor = (ambient + diffuse * light_dot) * (0.90 + 0.10 * dist_n)

        # Smooth shading to avoid artifacts on thin shapes
        shade_factor = cv2.GaussianBlur(shade_factor.astype(np.float32), (7, 7), 2.0)

        cm = crop_mask > 0
        edge_blend = 1.0 - dist_n
        for c in range(3):
            color_val = base_color[c] * (1 - edge_blend * 0.15) + shade_color[c] * (edge_blend * 0.15)
            vals = np.clip(color_val * shade_factor, 0, 255).astype(np.uint8)
            region = canvas[y0p:y1p, x0p:x1p, c]
            region[cm] = vals[cm]

    def _fill_and_shade(self, canvas, contour, base_color, shade_color,
                        H, W, shade_mult=1.0):
        """Fill a contour with shaded lighting."""
        if contour is None or len(contour) < 3:
            return
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255, cv2.LINE_AA)

        if shade_mult < 1.0:
            base_color = tuple(int(c * shade_mult) for c in base_color)
            shade_color = tuple(int(c * shade_mult) for c in shade_color)

        self._shade_region(canvas, mask, base_color, shade_color)

    # ── Arms ───────────────────────────────────────────────────────

    def _draw_arm(self, canvas, kps, sh, el, wr, hu, H, W, shade=1.0):
        """Draw arm as a single smooth contour: shoulder→elbow→wrist+hand."""
        # Joint chain: shoulder, mid-upper, elbow, mid-fore, wrist, hand-tip
        mid_up = (kps[sh] + kps[el]) / 2
        mid_fore = (kps[el] + kps[wr]) / 2
        hand_dir = kps[wr] - kps[el]
        hand_dir = hand_dir / (np.linalg.norm(hand_dir) + 1e-8)
        hand_tip = kps[wr] + hand_dir * hu * 0.35

        joints = np.array([kps[sh], mid_up, kps[el], mid_fore, kps[wr], hand_tip])

        # Widths: upper arm thicker, forearm tapers, hand rounds
        w_shoulder = hu * 0.30
        w_upper_mid = hu * 0.26
        w_elbow = hu * 0.22
        w_fore_mid = hu * 0.19
        w_wrist = hu * 0.17
        w_hand = hu * 0.12
        widths = np.array([w_shoulder, w_upper_mid, w_elbow, w_fore_mid, w_wrist, w_hand])

        contour = self._limb_contour(joints, widths)
        if contour is None:
            return

        # Upper arm = clothing, forearm = skin
        # Draw as two separate filled regions for different colors
        # Upper arm (shoulder → elbow)
        upper_joints = np.array([kps[sh], mid_up, kps[el]])
        upper_w = np.array([w_shoulder, w_upper_mid, w_elbow])
        upper_c = self._limb_contour(upper_joints, upper_w, n_samples=25)
        self._fill_and_shade(canvas, upper_c, self.top, self.top_shade, H, W, shade)

        # Forearm (elbow → wrist → hand) = skin
        fore_joints = np.array([kps[el], mid_fore, kps[wr], hand_tip])
        fore_w = np.array([w_elbow, w_fore_mid, w_wrist, w_hand])
        fore_c = self._limb_contour(fore_joints, fore_w, n_samples=25)
        self._fill_and_shade(canvas, fore_c, self.skin, self.skin_shade, H, W, shade)

        # Clothing fold at elbow if bent
        upper_dir = kps[el] - kps[sh]
        fore_dir = kps[wr] - kps[el]
        cos_angle = np.dot(upper_dir, fore_dir) / (
            np.linalg.norm(upper_dir) * np.linalg.norm(fore_dir) + 1e-8)
        bend_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if bend_angle > 25:
            self._draw_fold_lines(canvas, kps[el], fore_dir, w_elbow,
                                  self.top_shade, shade, n=min(int(bend_angle / 40), 3))

        # Fingertip details — subtle short lines
        perp = np.array([-hand_dir[1], hand_dir[0]])
        sh_col = tuple(int(c * shade) for c in self.skin_shade)
        for i in range(4):
            offset = (i - 1.5) * hu * 0.04
            base = kps[wr] + hand_dir * hu * 0.12 + perp * offset
            tip = base + hand_dir * hu * 0.15
            cv2.line(canvas, tuple(base.astype(int)),
                     tuple(tip.astype(int)), sh_col, 1, cv2.LINE_AA)

    # ── Legs ───────────────────────────────────────────────────────

    def _draw_leg(self, canvas, kps, hip, knee, ankle, hu, H, W, shade=1.0):
        """Draw leg as smooth contour: hip→knee→ankle+shoe."""
        mid_thigh = (kps[hip] + kps[knee]) / 2
        mid_calf = (kps[knee] + kps[ankle]) / 2

        joints = np.array([kps[hip], mid_thigh, kps[knee], mid_calf, kps[ankle]])

        # Anatomical widths
        w_hip = hu * 0.38
        w_mid_thigh = hu * 0.32
        w_knee = hu * 0.24
        w_mid_calf = hu * 0.20
        w_ankle = hu * 0.16
        widths = np.array([w_hip, w_mid_thigh, w_knee, w_mid_calf, w_ankle])

        contour = self._limb_contour(joints, widths, n_samples=35)
        self._fill_and_shade(canvas, contour, self.bottom, self.bottom_shade, H, W, shade)

        # Fold at knee if bent
        thigh_dir = kps[knee] - kps[hip]
        calf_dir = kps[ankle] - kps[knee]
        cos_a = np.dot(thigh_dir, calf_dir) / (
            np.linalg.norm(thigh_dir) * np.linalg.norm(calf_dir) + 1e-8)
        bend = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if bend > 20:
            self._draw_fold_lines(canvas, kps[knee], calf_dir, w_knee,
                                  self.bottom_shade, shade, n=min(int(bend / 45), 3))

        # Shoe
        shin_dir = kps[ankle] - kps[knee]
        shin_angle = np.degrees(np.arctan2(shin_dir[1], shin_dir[0]))
        shoe_w = int(hu * 0.32)
        shoe_h = int(hu * 0.16)
        ankle_pt = kps[ankle].astype(int)
        # Shoe angle: mostly horizontal with slight lean from shin
        shoe_angle = shin_angle * 0.15
        sh_col = tuple(int(c * shade) for c in self.shoe)
        cv2.ellipse(canvas, tuple(ankle_pt), (shoe_w, shoe_h),
                    shoe_angle, 0, 360, sh_col, -1, cv2.LINE_AA)
        # Sole line
        sole_col = tuple(int(c * shade) for c in self.shoe_sole)
        cv2.ellipse(canvas, (ankle_pt[0], ankle_pt[1] + shoe_h // 2),
                    (shoe_w, max(shoe_h // 4, 2)),
                    shoe_angle, 0, 360, sole_col, -1, cv2.LINE_AA)

    # ── Torso ──────────────────────────────────────────────────────

    def _draw_torso(self, canvas, kps, hu, H, W):
        """Draw torso with smooth contour and clothing detail."""
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        mid_sh = (ls + rs) / 2
        mid_hp = (lh + rh) / 2
        waist = mid_sh * 0.4 + mid_hp * 0.6  # waist is lower than midpoint

        # Shoulder expansion / hip contour
        sh_dir = rs - ls
        sh_perp = np.array([-sh_dir[1], sh_dir[0]])
        sh_perp = sh_perp / (np.linalg.norm(sh_perp) + 1e-8)

        hp_dir = rh - lh
        hp_perp = np.array([-hp_dir[1], hp_dir[0]])
        hp_perp = hp_perp / (np.linalg.norm(hp_perp) + 1e-8)

        # Contour points (left side down, right side up)
        expand_sh = hu * 0.06
        waist_pinch = hu * 0.12 if self.gender == "female" else hu * 0.03
        expand_hp = hu * 0.04

        # Key contour points clockwise — natural torso shape
        # Use shoulder width for top, narrow at waist, wider at hips
        sh_half = np.linalg.norm(rs - ls) / 2 + expand_sh
        hp_half = np.linalg.norm(rh - lh) / 2 + expand_hp
        waist_half = min(sh_half, hp_half) - waist_pinch

        top_center = mid_sh
        bot_center = mid_hp
        waist_center = top_center * 0.35 + bot_center * 0.65

        sh_dir_n = (rs - ls) / (np.linalg.norm(rs - ls) + 1e-8)
        hp_dir_n = (rh - lh) / (np.linalg.norm(rh - lh) + 1e-8)
        waist_dir_n = (sh_dir_n + hp_dir_n) / 2
        waist_dir_n = waist_dir_n / (np.linalg.norm(waist_dir_n) + 1e-8)

        pts = np.array([
            top_center - sh_dir_n * sh_half,     # left shoulder
            top_center + sh_dir_n * sh_half,     # right shoulder
            waist_center + waist_dir_n * waist_half,  # right waist
            bot_center + hp_dir_n * hp_half,     # right hip
            bot_center - hp_dir_n * hp_half,     # left hip
            waist_center - waist_dir_n * waist_half,  # left waist
        ], dtype=np.int32)

        pts = np.array(pts, dtype=np.int32)
        self._fill_and_shade(canvas, pts, self.top, self.top_shade, H, W)

        # Belt line at waist
        belt_y = int(waist[1])
        belt_x1 = int(waist[0] - hu * 0.5)
        belt_x2 = int(waist[0] + hu * 0.5)
        cv2.line(canvas, (belt_x1, belt_y), (belt_x2, belt_y),
                 self.belt_color, max(int(hu * 0.06), 2), cv2.LINE_AA)
        cv2.line(canvas, (belt_x1, belt_y + 2), (belt_x2, belt_y + 2),
                 self.belt_color, 1, cv2.LINE_AA)

        # Collar / neckline  (skin-colored V or round)
        neck = mid_sh.astype(int)
        collar_w = int(hu * 0.3)
        collar_h = int(hu * 0.15)
        cv2.ellipse(canvas, tuple(neck), (collar_w, collar_h),
                    0, 10, 170, self.skin, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, tuple(neck), (collar_w, collar_h),
                    0, 10, 170, self.top_shade, 1, cv2.LINE_AA)

        # Neck (skin between collar and head)
        neck_w = hu * 0.24
        head_bottom = kps[0].copy()
        head_bottom[1] += hu * 0.5
        neck_joints = np.array([head_bottom, mid_sh])
        neck_widths = np.array([neck_w, neck_w * 1.1])
        neck_c = self._limb_contour(neck_joints, neck_widths, n_samples=15)
        self._fill_and_shade(canvas, neck_c, self.skin, self.skin_shade, H, W)

    # ── Head ───────────────────────────────────────────────────────

    def _draw_head(self, canvas, kps, hu, H, W):
        """Draw detailed head: face, eyes, brows, nose, mouth, hair."""
        nose = kps[0]
        hc = nose.astype(int)  # head center
        hr = int(hu * 0.85)    # head radius

        # ── Hair (drawn first, behind face for long hair) ──
        if self.hair_style in ("long", "ponytail"):
            self._draw_long_hair(canvas, hc, hr, hu, H, W)

        # ── Face ──
        face_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(face_mask, tuple(hc), hr, 255, -1, cv2.LINE_AA)
        # Slight oval: taller than wide
        cv2.ellipse(face_mask, tuple(hc), (int(hr * 0.92), hr),
                    0, 0, 360, 255, -1, cv2.LINE_AA)
        self._shade_region(canvas, face_mask, self.skin, self.skin_shade,
                           light_dir=(-0.2, -0.5), ambient=0.82, diffuse=0.18)

        # ── Short hair (on top of face) ──
        if self.hair_style == "short":
            self._draw_short_hair(canvas, hc, hr, hu, H, W)

        # ── Eyes ──
        eye_y = hc[1] - int(hr * 0.08)
        eye_dx = int(hr * 0.30)
        eye_w = max(int(hr * 0.18), 3)
        eye_h = max(int(hr * 0.11), 2)

        for side in (-1, 1):
            ex = hc[0] + side * eye_dx
            # Eye white
            cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h),
                        0, 0, 360, (245, 242, 238), -1, cv2.LINE_AA)
            # Iris
            ir = max(eye_h, 2)
            cv2.circle(canvas, (ex, eye_y), ir, self.eye, -1, cv2.LINE_AA)
            # Pupil
            pr = max(ir // 2, 1)
            cv2.circle(canvas, (ex, eye_y), pr, (12, 8, 5), -1, cv2.LINE_AA)
            # Highlight
            cv2.circle(canvas, (ex - pr + 1, eye_y - pr + 1),
                       max(pr // 2, 1), (255, 255, 255), -1, cv2.LINE_AA)
            # Upper eyelid line
            cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h),
                        0, 180, 360, self.brow, 1, cv2.LINE_AA)
            # Lashes (tiny lines)
            for li in range(3):
                la = 200 + li * 25
                lx = int(ex + eye_w * np.cos(np.radians(la)))
                ly = int(eye_y + eye_h * np.sin(np.radians(la)))
                cv2.line(canvas, (lx, ly), (lx, ly - 2), self.brow, 1, cv2.LINE_AA)
            # Eyebrow
            brow_y = eye_y - int(hr * 0.20)
            brow_pts = []
            for bi in range(8):
                bx = ex - eye_w + bi * eye_w // 4
                by = brow_y - int(2.0 * np.sin(np.pi * bi / 7))
                brow_pts.append([bx, by])
            brow_pts = np.array(brow_pts, dtype=np.int32)
            cv2.polylines(canvas, [brow_pts], False, self.brow,
                          max(int(hr * 0.04), 1), cv2.LINE_AA)

        # ── Nose ──
        nose_tip_y = hc[1] + int(hr * 0.15)
        nose_w = max(int(hr * 0.05), 1)
        # Bridge
        cv2.line(canvas, (hc[0], hc[1] - int(hr * 0.05)),
                 (hc[0], nose_tip_y), self.skin_shade, 1, cv2.LINE_AA)
        # Nostrils
        cv2.circle(canvas, (hc[0] - nose_w, nose_tip_y), max(nose_w, 1),
                   self.skin_shade, -1, cv2.LINE_AA)
        cv2.circle(canvas, (hc[0] + nose_w, nose_tip_y), max(nose_w, 1),
                   self.skin_shade, -1, cv2.LINE_AA)

        # ── Mouth ──
        mouth_y = hc[1] + int(hr * 0.35)
        mouth_w = int(hr * 0.22)
        mouth_h = max(int(hr * 0.06), 2)
        # Lower lip
        cv2.ellipse(canvas, (hc[0], mouth_y), (mouth_w, mouth_h),
                    0, 10, 170, self.lip, -1, cv2.LINE_AA)
        # Upper lip (cupid's bow)
        cv2.ellipse(canvas, (hc[0], mouth_y - 1), (mouth_w, max(mouth_h // 2, 1)),
                    0, 190, 350, self.lip, 1, cv2.LINE_AA)
        # Lip line
        cv2.line(canvas, (hc[0] - mouth_w, mouth_y),
                 (hc[0] + mouth_w, mouth_y), self.lip, 1, cv2.LINE_AA)

        # ── Chin shadow ──
        chin_y = hc[1] + int(hr * 0.65)
        cv2.ellipse(canvas, (hc[0], chin_y), (int(hr * 0.4), int(hr * 0.1)),
                    0, 0, 180, self.skin_shade, 1, cv2.LINE_AA)

        # ── Ears ──
        ear_y = hc[1] + int(hr * 0.05)
        ear_w = max(int(hr * 0.12), 2)
        ear_h = max(int(hr * 0.18), 3)
        for side in (-1, 1):
            ear_x = hc[0] + side * int(hr * 0.88)
            cv2.ellipse(canvas, (ear_x, ear_y), (ear_w, ear_h),
                        0, 0, 360, self.skin, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (ear_x, ear_y), (ear_w, ear_h),
                        0, 0, 360, self.skin_shade, 1, cv2.LINE_AA)

    # ── Hair ───────────────────────────────────────────────────────

    def _draw_short_hair(self, canvas, hc, hr, hu, H, W):
        """Short hair as a dome on top of head."""
        hair_mask = np.zeros((H, W), dtype=np.uint8)
        # Main volume
        cv2.ellipse(hair_mask, (hc[0], hc[1] - int(hr * 0.1)),
                    (int(hr * 1.08), int(hr * 0.92)),
                    0, 180, 360, 255, -1, cv2.LINE_AA)
        # Side volume
        cv2.ellipse(hair_mask, (hc[0], hc[1] - int(hr * 0.05)),
                    (int(hr * 1.05), int(hr * 0.5)),
                    0, 200, 340, 255, -1, cv2.LINE_AA)

        hair_shade = tuple(max(c - 15, 0) for c in self.hair)
        self._shade_region(canvas, hair_mask, self.hair, hair_shade,
                           light_dir=(-0.3, -0.8))

        # Hairline texture — a few curved strokes
        for i in range(6):
            angle = 200 + i * 20
            x1 = hc[0] + int(hr * 0.3 * np.cos(np.radians(angle)))
            y1 = hc[1] - int(hr * 0.5) + int(hr * 0.3 * np.sin(np.radians(angle)))
            x2 = x1 + int(hr * 0.15 * np.cos(np.radians(angle + 30)))
            y2 = y1 + int(hr * 0.15 * np.sin(np.radians(angle + 30)))
            cv2.line(canvas, (x1, y1), (x2, y2), hair_shade, 1, cv2.LINE_AA)

    def _draw_long_hair(self, canvas, hc, hr, hu, H, W):
        """Long/ponytail hair with volume and strands."""
        # Build hair boundary with Bézier-like contour
        hair_pts = []
        n_pts = 24
        for i in range(n_pts):
            angle = -np.pi + 2 * np.pi * i / n_pts
            # Larger at top and sides, extends below chin
            if angle < -np.pi / 2 or angle > np.pi / 2:
                # Back / top
                r = hr * 1.30
            else:
                # Sides
                r = hr * 1.20
            # Add waviness
            r += hr * 0.06 * np.sin(angle * 5)

            px = hc[0] + r * np.cos(angle)
            # Extend downward for length
            if abs(angle) > np.pi * 0.6:
                py = hc[1] + r * np.sin(angle) + hr * 0.5
            else:
                py = hc[1] + r * np.sin(angle)
            hair_pts.append([px, py])

        # Extend bottom strands downward
        for i in range(5):
            x = hc[0] + (i - 2) * int(hr * 0.35)
            y = hc[1] + int(hr * 1.6) + (abs(i - 2)) * int(hr * 0.1)
            hair_pts.append([x, y])

        hair_pts = np.array(hair_pts, dtype=np.float64)
        try:
            tck, u = splprep([hair_pts[:, 0], hair_pts[:, 1]],
                             s=len(hair_pts) * 1.5, per=True, k=3)
            u_fine = np.linspace(0, 1, 200)
            sx, sy = splev(u_fine, tck)
            smooth = np.column_stack([sx, sy]).astype(np.int32)
        except Exception:
            smooth = hair_pts.astype(np.int32)

        hair_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(hair_mask, [smooth], 255, cv2.LINE_AA)

        hair_shade = tuple(max(c - 20, 0) for c in self.hair)
        self._shade_region(canvas, hair_mask, self.hair, hair_shade,
                           light_dir=(-0.3, -0.6))

        # Strand lines for texture
        for i in range(10):
            t = (i - 5) * hr * 0.15
            x0 = hc[0] + int(t * 0.3)
            y0 = hc[1] - int(hr * 0.7)
            x1 = hc[0] + int(t * 0.9)
            y1 = hc[1] + int(hr * 1.4)
            # Slight curve
            mid_x = (x0 + x1) // 2 + int(t * 0.15)
            mid_y = (y0 + y1) // 2
            pts = np.array([[x0, y0], [mid_x, mid_y], [x1, y1]], dtype=np.int32)
            highlight = tuple(min(c + 25, 255) for c in self.hair)
            cv2.polylines(canvas, [pts], False, highlight, 1, cv2.LINE_AA)

        # Hair highlight (glossy shine)
        hl_x = hc[0] - int(hr * 0.15)
        hl_y = hc[1] - int(hr * 0.5)
        highlight_c = tuple(min(c + 40, 255) for c in self.hair)
        hl_layer = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.ellipse(hl_layer, (hl_x, hl_y), (int(hr * 0.35), int(hr * 0.2)),
                    -25, 0, 360, highlight_c, -1, cv2.LINE_AA)
        hl_layer = cv2.GaussianBlur(hl_layer, (15, 15), 5)
        hl_mask = hl_layer.sum(axis=2) > 0
        hair_region = hair_mask > 0
        blend_mask = hl_mask & hair_region
        if blend_mask.any():
            alpha = 0.3
            canvas[blend_mask] = np.clip(
                canvas[blend_mask].astype(float) * (1 - alpha) +
                hl_layer[blend_mask].astype(float) * alpha, 0, 255
            ).astype(np.uint8)

        # Ponytail
        if self.hair_style == "ponytail":
            pt_start = np.array([hc[0] + int(hr * 0.3), hc[1] - int(hr * 0.2)])
            pt_end = np.array([hc[0] + int(hr * 1.5), hc[1] + int(hr * 0.8)])
            pt_mid = (pt_start + pt_end) / 2 + np.array([hr * 0.3, -hr * 0.2])
            pt_joints = np.array([pt_start, pt_mid, pt_end])
            pt_widths = np.array([hr * 0.3, hr * 0.25, hr * 0.1])
            pt_c = self._limb_contour(pt_joints, pt_widths, n_samples=20)
            if pt_c is not None:
                pt_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(pt_mask, [pt_c], 255, cv2.LINE_AA)
                self._shade_region(canvas, pt_mask, self.hair, hair_shade)

    # ── Helpers ────────────────────────────────────────────────────

    def _draw_fold_lines(self, canvas, joint_pos, bone_dir, width,
                         shade_color, shade_mult, n=2):
        """Draw clothing fold lines near a bent joint."""
        bone_len = np.linalg.norm(bone_dir)
        if bone_len < 1:
            return
        d = bone_dir / bone_len
        perp = np.array([-d[1], d[0]])
        col = tuple(int(c * shade_mult) for c in shade_color)
        for i in range(n):
            t = 0.05 + i * 0.06
            center = joint_pos + d * bone_len * t
            p1 = center + perp * width * 0.6
            p2 = center - perp * width * 0.6
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     col, 1, cv2.LINE_AA)
