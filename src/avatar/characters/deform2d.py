"""
Mesh-deformation 2D character style — high-quality character rendered via
triangulated mesh skinning with smooth per-vertex bone weights.

A detailed T-pose character image is rendered once, triangulated, and
bone weights assigned.  Each frame, COCO-17 keypoints drive bone
transforms; vertices are LBS-deformed and triangles are affine-warped
to produce the final image.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay


# ── Skeleton definition ────────────────────────────────────────────
# 10 bones connecting COCO-17 joints
_BONES = [
    (0, 5), (0, 6),      # head → shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso → hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (11, 12),            # hip bar
    (5, 6),              # shoulder bar
]

# T-pose reference body (normalised coords, origin at hip center)
# Matching demo.py base_body but with arms horizontally out for T-pose
_TPOSE = np.array([
    [0.0, -0.70],     # 0  nose
    [-0.04, -0.74],   # 1  left_eye
    [0.04, -0.74],    # 2  right_eye
    [-0.08, -0.70],   # 3  left_ear
    [0.08, -0.70],    # 4  right_ear
    [-0.18, -0.45],   # 5  left_shoulder
    [0.18, -0.45],    # 6  right_shoulder
    [-0.40, -0.45],   # 7  left_elbow  (arms out)
    [0.40, -0.45],    # 8  right_elbow
    [-0.58, -0.45],   # 9  left_wrist
    [0.58, -0.45],    # 10 right_wrist
    [-0.10, 0.00],    # 11 left_hip
    [0.10, 0.00],     # 12 right_hip
    [-0.12, 0.35],    # 13 left_knee
    [0.12, 0.35],     # 14 right_knee
    [-0.12, 0.70],    # 15 left_ankle
    [0.12, 0.70],     # 16 right_ankle
], dtype=np.float32)


# ── Character appearance presets ───────────────────────────────────

DEFORM_PRESETS = {
    "realistic_male": {
        "skin": (195, 175, 155),
        "skin_shade": (155, 135, 115),
        "hair": (45, 35, 25),
        "hair_style": "short",
        "top": (140, 90, 60),
        "top_shade": (100, 60, 35),
        "bottom": (85, 70, 55),
        "bottom_shade": (55, 40, 28),
        "shoe": (40, 35, 30),
        "eye": (80, 50, 30),
        "lip": (130, 110, 140),
    },
    "realistic_female": {
        "skin": (215, 200, 185),
        "skin_shade": (180, 160, 145),
        "hair": (35, 25, 65),
        "hair_style": "long",
        "top": (230, 180, 200),
        "top_shade": (185, 140, 160),
        "bottom": (75, 55, 45),
        "bottom_shade": (50, 35, 25),
        "shoe": (170, 150, 190),
        "eye": (120, 70, 40),
        "lip": (120, 100, 160),
    },
    "stylized": {
        "skin": (225, 215, 240),
        "skin_shade": (190, 175, 210),
        "hair": (255, 120, 180),
        "hair_style": "long",
        "top": (240, 220, 100),
        "top_shade": (200, 180, 70),
        "bottom": (200, 80, 80),
        "bottom_shade": (160, 50, 50),
        "shoe": (80, 60, 60),
        "eye": (180, 80, 50),
        "lip": (140, 100, 180),
    },
}


class Deform2DStyle:
    """High-quality 2D mesh-deformation character renderer."""

    PRESETS = list(DEFORM_PRESETS.keys())

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("deform2d", {})
        preset_name = style_cfg.get("preset", "realistic_male")
        preset = DEFORM_PRESETS.get(preset_name, DEFORM_PRESETS["realistic_male"])

        self.skin = tuple(style_cfg.get("skin_color", preset["skin"]))
        self.skin_shade = tuple(style_cfg.get("skin_shade", preset["skin_shade"]))
        self.hair_color = tuple(style_cfg.get("hair_color", preset["hair"]))
        self.hair_style = style_cfg.get("hair_style", preset["hair_style"])
        self.top_color = tuple(style_cfg.get("top_color", preset["top"]))
        self.top_shade = tuple(style_cfg.get("top_shade", preset["top_shade"]))
        self.bottom_color = tuple(style_cfg.get("bottom_color", preset["bottom"]))
        self.bottom_shade = tuple(style_cfg.get("bottom_shade", preset["bottom_shade"]))
        self.shoe_color = tuple(style_cfg.get("shoe_color", preset["shoe"]))
        self.eye_color = tuple(style_cfg.get("eye_color", preset.get("eye", (80, 50, 30))))
        self.lip_color = tuple(style_cfg.get("lip_color", preset.get("lip", (130, 110, 140))))

        # Caches - will be built on first render at the target resolution
        self._ref_img: np.ndarray | None = None  # BGRA T-pose image
        self._ref_verts: np.ndarray | None = None  # (V, 2) pixel coords
        self._tri: Delaunay | None = None
        self._weights: np.ndarray | None = None  # (V, 17) bone weights
        self._ref_joints_px: np.ndarray | None = None  # (17, 2) T-pose joints in px
        self._ref_size: int = 0

    # ── Public API ─────────────────────────────────────────────────

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        """Render character directly from keypoints each frame.

        Uses the detailed drawing functions (clothing, shading, hair, etc.)
        with direct per-frame rendering for reliable results at any pose.
        """
        K = len(keypoints)
        if K < 17:
            return canvas

        kps = keypoints[:17].astype(np.float64)
        H, W = canvas.shape[:2]

        # Compute character size from torso
        torso_w = np.linalg.norm(kps[5] - kps[6])
        if torso_w < 3:
            return canvas

        shoulder_w = torso_w
        j = kps.astype(np.int32)

        # Body measurements
        torso_h = np.linalg.norm((kps[5] + kps[6]) / 2 - (kps[11] + kps[12]) / 2)
        head_r = int(shoulder_w * 0.35)
        limb_w = max(int(shoulder_w * 0.12), 3)
        torso_w_px = int(shoulder_w * 0.55)

        # Dynamic z-ordering
        torso_cx = (kps[5][0] + kps[6][0]) / 2
        left_arm_cx = (kps[5][0] + kps[7][0] + kps[9][0]) / 3
        right_arm_cx = (kps[6][0] + kps[8][0] + kps[10][0]) / 3
        left_arm_in_front = (left_arm_cx - torso_cx) > (torso_cx - right_arm_cx)

        left_leg_cx = (kps[11][0] + kps[13][0] + kps[15][0]) / 3
        right_leg_cx = (kps[12][0] + kps[14][0] + kps[16][0]) / 3
        left_leg_in_front = (left_leg_cx - torso_cx) > (torso_cx - right_leg_cx)

        back_arm = (6, 8, 10) if left_arm_in_front else (5, 7, 9)
        front_arm = (5, 7, 9) if left_arm_in_front else (6, 8, 10)
        back_leg = (12, 14, 16) if left_leg_in_front else (11, 13, 15)
        front_leg = (11, 13, 15) if left_leg_in_front else (12, 14, 16)

        shade_back = 0.80

        def _shaded(color, factor):
            return tuple(max(0, min(255, int(c * factor))) for c in color)

        # ── 1) Back arm ──
        sh, el, wr = back_arm
        self._draw_thick_line(canvas, kps[sh], kps[el],
                              _shaded(self.top_color, shade_back),
                              _shaded(self.top_shade, shade_back),
                              int(limb_w * 1.1))
        self._draw_thick_line(canvas, kps[el], kps[wr],
                              _shaded(self.skin, shade_back),
                              _shaded(self.skin_shade, shade_back), limb_w)
        cv2.circle(canvas, tuple(j[wr]), int(limb_w * 0.8),
                   _shaded(self.skin, shade_back), -1, cv2.LINE_AA)

        # ── 2) Back leg ──
        hip_i, knee_i, ankle_i = back_leg
        self._draw_thick_line(canvas, kps[hip_i], kps[knee_i],
                              _shaded(self.bottom_color, shade_back),
                              _shaded(self.bottom_shade, shade_back),
                              int(limb_w * 1.2))
        self._draw_thick_line(canvas, kps[knee_i], kps[ankle_i],
                              _shaded(self.bottom_color, shade_back),
                              _shaded(self.bottom_shade, shade_back),
                              int(limb_w * 1.0))
        shoe_c = j[ankle_i].copy()
        shoe_w_px = int(limb_w * 1.4)
        shoe_h_px = int(limb_w * 0.7)
        cv2.ellipse(canvas, tuple(shoe_c), (shoe_w_px, shoe_h_px), 0, 0, 360,
                    _shaded(self.shoe_color, shade_back), -1, cv2.LINE_AA)

        # ── 3) Torso ──
        shoulder_mid = ((kps[5] + kps[6]) / 2).astype(int)
        hip_mid = ((kps[11] + kps[12]) / 2).astype(int)
        top_w = torso_w_px
        bot_w = int(torso_w_px * 0.85)
        pts = np.array([
            [shoulder_mid[0] - top_w, shoulder_mid[1]],
            [shoulder_mid[0] + top_w, shoulder_mid[1]],
            [hip_mid[0] + bot_w, hip_mid[1]],
            [hip_mid[0] - bot_w, hip_mid[1]],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], self.top_color, cv2.LINE_AA)

        # Torso gradient shading
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        ys_idx = np.where(mask > 0)
        if len(ys_idx[0]) > 0:
            y_min, y_max = ys_idx[0].min(), ys_idx[0].max()
            t_vals = (ys_idx[0] - y_min) / max(y_max - y_min, 1)
            for c_idx in range(3):
                blended = (self.top_color[c_idx] * (1 - t_vals * 0.2) +
                           self.top_shade[c_idx] * t_vals * 0.2)
                canvas[ys_idx[0], ys_idx[1], c_idx] = np.clip(blended, 0, 255).astype(np.uint8)
        cv2.polylines(canvas, [pts], True, self.top_shade, 1, cv2.LINE_AA)

        # ── 4) Front leg ──
        hip_i, knee_i, ankle_i = front_leg
        self._draw_thick_line(canvas, kps[hip_i], kps[knee_i],
                              self.bottom_color, self.bottom_shade,
                              int(limb_w * 1.2))
        self._draw_thick_line(canvas, kps[knee_i], kps[ankle_i],
                              self.bottom_color, self.bottom_shade,
                              int(limb_w * 1.0))
        shoe_c = j[ankle_i].copy()
        cv2.ellipse(canvas, tuple(shoe_c), (shoe_w_px, shoe_h_px), 0, 0, 360,
                    self.shoe_color, -1, cv2.LINE_AA)

        # ── 5) Front arm ──
        sh, el, wr = front_arm
        self._draw_thick_line(canvas, kps[sh], kps[el],
                              self.top_color, self.top_shade,
                              int(limb_w * 1.1))
        self._draw_thick_line(canvas, kps[el], kps[wr],
                              self.skin, self.skin_shade, limb_w)
        cv2.circle(canvas, tuple(j[wr]), int(limb_w * 0.8),
                   self.skin, -1, cv2.LINE_AA)

        # ── 6) Neck ──
        neck = ((kps[5] + kps[6]) / 2).astype(int)
        head_bottom = j[0].copy()
        head_bottom[1] += head_r
        self._draw_thick_line(canvas, neck.astype(float), head_bottom.astype(float),
                              self.skin, self.skin_shade, int(limb_w * 0.8))

        # ── 7) Head ──
        head_center = j[0].copy()
        cv2.circle(canvas, tuple(head_center), head_r,
                   self.skin, -1, cv2.LINE_AA)

        # Head shading
        y0h = max(head_center[1] - head_r, 0)
        y1h = min(head_center[1] + head_r + 1, H)
        x0h = max(head_center[0] - head_r, 0)
        x1h = min(head_center[0] + head_r + 1, W)
        if y1h > y0h and x1h > x0h:
            Y, X = np.mgrid[y0h:y1h, x0h:x1h]
            dist_sq = (X - head_center[0])**2 + (Y - head_center[1])**2
            in_circle = dist_sq <= head_r**2
            shade_vals = 0.85 + 0.15 * ((X - head_center[0] + head_r) / max(2 * head_r, 1))
            for c_idx in range(3):
                region = canvas[Y[in_circle], X[in_circle], c_idx].astype(np.float32)
                canvas[Y[in_circle], X[in_circle], c_idx] = np.clip(
                    region * shade_vals[in_circle], 0, 255
                ).astype(np.uint8)

        # Eyes
        eye_y = head_center[1] - int(head_r * 0.15)
        eye_dx = int(head_r * 0.3)
        eye_r = max(int(head_r * 0.12), 2)
        for dx in [-eye_dx, eye_dx]:
            cv2.circle(canvas, (head_center[0] + dx, eye_y), eye_r + 1,
                       (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (head_center[0] + dx, eye_y), eye_r,
                       self.eye_color, -1, cv2.LINE_AA)
            pupil_r = max(eye_r // 2, 1)
            cv2.circle(canvas, (head_center[0] + dx, eye_y), pupil_r,
                       (20, 15, 10), -1, cv2.LINE_AA)

        # Mouth
        mouth_y = head_center[1] + int(head_r * 0.35)
        mouth_w = int(head_r * 0.25)
        cv2.ellipse(canvas, (head_center[0], mouth_y), (mouth_w, max(mouth_w // 3, 2)),
                    0, 0, 180, self.lip_color, -1, cv2.LINE_AA)

        # Hair
        if self.hair_style == "long":
            hair_pts = np.array([
                [head_center[0] - int(head_r * 1.1), head_center[1] - int(head_r * 0.3)],
                [head_center[0], head_center[1] - int(head_r * 1.2)],
                [head_center[0] + int(head_r * 1.1), head_center[1] - int(head_r * 0.3)],
                [head_center[0] + int(head_r * 0.9), head_center[1] + int(head_r * 1.2)],
                [head_center[0] - int(head_r * 0.9), head_center[1] + int(head_r * 1.2)],
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [hair_pts], self.hair_color, cv2.LINE_AA)
            # Redraw face on top
            cv2.circle(canvas, tuple(head_center), int(head_r * 0.85),
                       self.skin, -1, cv2.LINE_AA)
            for dx in [-eye_dx, eye_dx]:
                cv2.circle(canvas, (head_center[0] + dx, eye_y), eye_r + 1,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (head_center[0] + dx, eye_y), eye_r,
                           self.eye_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (head_center[0] + dx, eye_y), pupil_r,
                           (20, 15, 10), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (head_center[0], mouth_y), (mouth_w, max(mouth_w // 3, 2)),
                        0, 0, 180, self.lip_color, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (head_center[0], head_center[1] - int(head_r * 0.15)),
                        (int(head_r * 1.05), int(head_r * 0.85)),
                        0, 180, 360, self.hair_color, -1, cv2.LINE_AA)

        return canvas

    # ── Reference character construction ───────────────────────────

    def _build_reference(self, W: int, H: int, target_kps: np.ndarray):
        """Build the T-pose reference image, mesh, and weights."""
        # Scale T-pose to match target character size
        # Use distance from nose to mid-hip as reference height
        target_hip = (target_kps[11] + target_kps[12]) / 2
        target_shoulder = (target_kps[5] + target_kps[6]) / 2
        target_height = np.linalg.norm(target_kps[0] - target_hip) + \
                        np.linalg.norm(target_hip - (target_kps[15] + target_kps[16]) / 2)

        ref_hip = (_TPOSE[11] + _TPOSE[12]) / 2
        ref_height = abs(_TPOSE[0, 1] - ref_hip[1]) + \
                     abs(ref_hip[1] - _TPOSE[15, 1])

        scale = target_height / max(ref_height, 0.01)
        center = (target_kps[11] + target_kps[12]) / 2  # center on hips

        # T-pose joints in pixel coords
        ref_joints = _TPOSE.copy() * scale
        ref_joints[:, 0] += center[0]
        ref_joints[:, 1] += center[1]
        self._ref_joints_px = ref_joints
        self._ref_size = int(np.linalg.norm(target_kps[5] - target_kps[6]))

        # Render T-pose character image (BGRA)
        img_h = H + 100  # extra margin
        img_w = W + 100
        self._ref_img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        self._render_tpose_character(self._ref_img, ref_joints)

        # Build mesh vertices: joints + contour points around each body part
        verts = self._generate_mesh_vertices(ref_joints, scale)
        self._ref_verts = verts

        # Triangulate
        self._tri = Delaunay(verts)

        # Compute bone weights for each vertex
        self._weights = self._compute_weights(verts, ref_joints)

    def _render_tpose_character(self, img: np.ndarray, joints: np.ndarray):
        """Draw a detailed T-pose character onto BGRA image."""
        j = joints.astype(np.int32)
        H, W = img.shape[:2]

        # Measurements
        shoulder_w = np.linalg.norm(joints[5] - joints[6])
        torso_h = np.linalg.norm((joints[5] + joints[6]) / 2 - (joints[11] + joints[12]) / 2)
        head_r = int(shoulder_w * 0.35)
        limb_w = max(int(shoulder_w * 0.12), 3)
        torso_w_px = int(shoulder_w * 0.55)

        def _grad_color(c1, c2, t):
            return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

        # ── Draw back limbs slightly darker ──
        shade = 0.75

        # ── Legs (bottom → pants) ──
        for (hip, knee, ankle) in [(11, 13, 15), (12, 14, 16)]:
            # Thigh
            self._draw_thick_line(img, joints[hip], joints[knee],
                                  self.bottom_color, self.bottom_shade,
                                  int(limb_w * 1.2))
            # Shin
            self._draw_thick_line(img, joints[knee], joints[ankle],
                                  self.bottom_color, self.bottom_shade,
                                  int(limb_w * 1.0))
            # Shoe
            shoe_c = j[ankle].copy()
            shoe_w = int(limb_w * 1.4)
            shoe_h = int(limb_w * 0.7)
            cv2.ellipse(img, tuple(shoe_c), (shoe_w, shoe_h), 0, 0, 360,
                        (*self.shoe_color, 255), -1, cv2.LINE_AA)

        # ── Torso ──
        shoulder_mid = ((joints[5] + joints[6]) / 2).astype(int)
        hip_mid = ((joints[11] + joints[12]) / 2).astype(int)
        # Trapezoid torso
        top_w = torso_w_px
        bot_w = int(torso_w_px * 0.85)
        pts = np.array([
            [shoulder_mid[0] - top_w, shoulder_mid[1]],
            [shoulder_mid[0] + top_w, shoulder_mid[1]],
            [hip_mid[0] + bot_w, hip_mid[1]],
            [hip_mid[0] - bot_w, hip_mid[1]],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (*self.top_color, 255), cv2.LINE_AA)
        # Vertical gradient shading on torso (vectorized)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        ys_idx = np.where(mask > 0)
        if len(ys_idx[0]) > 0:
            y_min, y_max = ys_idx[0].min(), ys_idx[0].max()
            t_vals = (ys_idx[0] - y_min) / max(y_max - y_min, 1)
            for c_idx in range(3):
                blended = (self.top_color[c_idx] * (1 - t_vals * 0.2) +
                           self.top_shade[c_idx] * t_vals * 0.2)
                img[ys_idx[0], ys_idx[1], c_idx] = np.clip(blended, 0, 255).astype(np.uint8)
        # Torso outline
        cv2.polylines(img, [pts], True, (*self.top_shade, 255), 1, cv2.LINE_AA)

        # ── Arms ──
        for (shoulder, elbow, wrist) in [(5, 7, 9), (6, 8, 10)]:
            # Upper arm (skin or sleeve)
            self._draw_thick_line(img, joints[shoulder], joints[elbow],
                                  self.top_color, self.top_shade,
                                  int(limb_w * 1.1))
            # Forearm (skin)
            self._draw_thick_line(img, joints[elbow], joints[wrist],
                                  self.skin, self.skin_shade, limb_w)
            # Hand circle
            cv2.circle(img, tuple(j[wrist]), int(limb_w * 0.8),
                       (*self.skin, 255), -1, cv2.LINE_AA)

        # ── Neck ──
        neck = ((joints[5] + joints[6]) / 2).astype(int)
        head_bottom = j[0].copy()
        head_bottom[1] += head_r
        self._draw_thick_line(img, neck.astype(float), head_bottom.astype(float),
                              self.skin, self.skin_shade, int(limb_w * 0.8))

        # ── Head ──
        head_center = j[0].copy()
        # Face circle
        cv2.circle(img, tuple(head_center), head_r,
                   (*self.skin, 255), -1, cv2.LINE_AA)
        # Shading gradient on head (vectorized)
        y0h = max(head_center[1] - head_r, 0)
        y1h = min(head_center[1] + head_r + 1, H)
        x0h = max(head_center[0] - head_r, 0)
        x1h = min(head_center[0] + head_r + 1, W)
        if y1h > y0h and x1h > x0h:
            Y, X = np.mgrid[y0h:y1h, x0h:x1h]
            dist_sq = (X - head_center[0])**2 + (Y - head_center[1])**2
            in_circle = dist_sq <= head_r**2
            shade_vals = 0.85 + 0.15 * ((X - head_center[0] + head_r) / max(2 * head_r, 1))
            for c_idx in range(3):
                region = img[Y[in_circle], X[in_circle], c_idx].astype(np.float32)
                img[Y[in_circle], X[in_circle], c_idx] = np.clip(
                    region * shade_vals[in_circle], 0, 255
                ).astype(np.uint8)

        # Eyes
        eye_y = head_center[1] - int(head_r * 0.15)
        eye_dx = int(head_r * 0.3)
        eye_r = max(int(head_r * 0.12), 2)
        cv2.circle(img, (head_center[0] - eye_dx, eye_y), eye_r + 1,
                   (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (head_center[0] + eye_dx, eye_y), eye_r + 1,
                   (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (head_center[0] - eye_dx, eye_y), eye_r,
                   (*self.eye_color, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (head_center[0] + eye_dx, eye_y), eye_r,
                   (*self.eye_color, 255), -1, cv2.LINE_AA)
        # Pupils
        pupil_r = max(eye_r // 2, 1)
        cv2.circle(img, (head_center[0] - eye_dx, eye_y), pupil_r,
                   (20, 15, 10, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (head_center[0] + eye_dx, eye_y), pupil_r,
                   (20, 15, 10, 255), -1, cv2.LINE_AA)

        # Mouth
        mouth_y = head_center[1] + int(head_r * 0.35)
        mouth_w = int(head_r * 0.25)
        cv2.ellipse(img, (head_center[0], mouth_y), (mouth_w, max(mouth_w // 3, 2)),
                    0, 0, 180, (*self.lip_color, 255), -1, cv2.LINE_AA)

        # Hair
        if self.hair_style == "long":
            hair_pts = np.array([
                [head_center[0] - int(head_r * 1.1), head_center[1] - int(head_r * 0.3)],
                [head_center[0], head_center[1] - int(head_r * 1.2)],
                [head_center[0] + int(head_r * 1.1), head_center[1] - int(head_r * 0.3)],
                [head_center[0] + int(head_r * 0.9), head_center[1] + int(head_r * 1.2)],
                [head_center[0] - int(head_r * 0.9), head_center[1] + int(head_r * 1.2)],
            ], dtype=np.int32)
            cv2.fillPoly(img, [hair_pts], (*self.hair_color, 255), cv2.LINE_AA)
            # Redraw face on top of hair
            cv2.circle(img, tuple(head_center), int(head_r * 0.85),
                       (*self.skin, 255), -1, cv2.LINE_AA)
            # Re-draw eyes and mouth over face
            cv2.circle(img, (head_center[0] - eye_dx, eye_y), eye_r + 1,
                       (255, 255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (head_center[0] + eye_dx, eye_y), eye_r + 1,
                       (255, 255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (head_center[0] - eye_dx, eye_y), eye_r,
                       (*self.eye_color, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (head_center[0] + eye_dx, eye_y), eye_r,
                       (*self.eye_color, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (head_center[0] - eye_dx, eye_y), pupil_r,
                       (20, 15, 10, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (head_center[0] + eye_dx, eye_y), pupil_r,
                       (20, 15, 10, 255), -1, cv2.LINE_AA)
            cv2.ellipse(img, (head_center[0], mouth_y), (mouth_w, max(mouth_w // 3, 2)),
                        0, 0, 180, (*self.lip_color, 255), -1, cv2.LINE_AA)
        else:
            # Short hair — cap on top
            cv2.ellipse(img, (head_center[0], head_center[1] - int(head_r * 0.15)),
                        (int(head_r * 1.05), int(head_r * 0.85)),
                        0, 180, 360, (*self.hair_color, 255), -1, cv2.LINE_AA)

    def _draw_thick_line(self, img, p1, p2, color, shade_color, width):
        """Draw an anti-aliased thick line with gradient shading."""
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1:
            return

        # Draw filled polygon (rectangle along the line)
        perp = np.array([-d[1], d[0]]) / length * width / 2
        pts = np.array([
            p1 + perp, p2 + perp, p2 - perp, p1 - perp
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (*color, 255), cv2.LINE_AA)

        # Rounded caps
        cv2.circle(img, tuple(p1.astype(int)), width // 2,
                   (*color, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p2.astype(int)), width // 2,
                   (*color, 255), -1, cv2.LINE_AA)

    # ── Mesh generation ────────────────────────────────────────────

    def _generate_mesh_vertices(self, joints: np.ndarray, scale: float) -> np.ndarray:
        """Generate mesh vertices: joint positions + contour samples."""
        verts = list(joints[:17].copy())  # Start with 17 joint positions

        # Add contour points around each body segment for better deformation
        limb_w = scale * 0.08  # half-width of limb contour

        segments = [
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
            (5, 11), (6, 12),  # torso sides
        ]

        for j1, j2 in segments:
            p1, p2 = joints[j1], joints[j2]
            d = p2 - p1
            length = np.linalg.norm(d)
            if length < 1:
                continue
            perp = np.array([-d[1], d[0]]) / length * limb_w

            # Sample along segment with contour offset
            for t_val in [0.25, 0.5, 0.75]:
                mid = p1 + d * t_val
                verts.append(mid + perp)
                verts.append(mid - perp)

        # Add head contour points
        head_r = scale * 0.15
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            verts.append(joints[0] + head_r * np.array([np.cos(angle), np.sin(angle)]))

        # Bounding box corners (to ensure full coverage)
        all_v = np.array(verts)
        margin = scale * 0.2
        x_min, y_min = all_v.min(axis=0) - margin
        x_max, y_max = all_v.max(axis=0) + margin
        for x in [x_min, x_max]:
            for y in [y_min, y_max]:
                verts.append(np.array([x, y]))

        return np.array(verts, dtype=np.float64)

    def _compute_weights(self, verts: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """Compute smooth multi-bone weights for each vertex.

        Returns (V, N_bones) array of per-bone influence weights that
        sum to 1.0 for each vertex. Uses inverse-distance-squared
        weighting with top-K bones to prevent distant bone influence.
        """
        V = len(verts)

        # Bone segments defined by (start_joint, end_joint)
        bone_segs = [
            (5, 6),    # 0: shoulder bar
            (11, 12),  # 1: hip bar
            (5, 11),   # 2: left torso
            (6, 12),   # 3: right torso
            (5, 7),    # 4: left upper arm
            (7, 9),    # 5: left forearm
            (6, 8),    # 6: right upper arm
            (8, 10),   # 7: right forearm
            (11, 13),  # 8: left thigh
            (13, 15),  # 9: left shin
            (12, 14),  # 10: right thigh
            (14, 16),  # 11: right shin
            (0, 5),    # 12: head-neck left
            (0, 6),    # 13: head-neck right
        ]
        self._bone_segs = bone_segs
        N_bones = len(bone_segs)

        weights = np.zeros((V, N_bones), dtype=np.float64)

        for vi in range(V):
            v = verts[vi]
            dists = np.full(N_bones, 1e12)

            for bi, (j1, j2) in enumerate(bone_segs):
                a = joints[j1]
                b = joints[j2]
                ab = b - a
                ab_len_sq = np.dot(ab, ab)
                if ab_len_sq < 1e-8:
                    proj = a
                else:
                    t = np.clip(np.dot(v - a, ab) / ab_len_sq, 0, 1)
                    proj = a + t * ab
                dists[bi] = np.linalg.norm(v - proj)

            # Keep top-4 closest bones, zero out the rest
            topk = min(4, N_bones)
            topk_indices = np.argpartition(dists, topk)[:topk]
            inv_dist = np.zeros(N_bones)
            for idx in topk_indices:
                inv_dist[idx] = 1.0 / (dists[idx] ** 2 + 1e-6)

            total = inv_dist.sum()
            if total > 0:
                weights[vi] = inv_dist / total
            else:
                # Fallback: assign to closest bone
                weights[vi, np.argmin(dists)] = 1.0

        return weights

    # ── Per-frame deformation ──────────────────────────────────────

    def _deform_vertices(self, target_kps: np.ndarray) -> np.ndarray:
        """Deform vertices using multi-bone Linear Blend Skinning.

        Each vertex blends transforms from its top-K nearest bones
        weighted by inverse distance, preventing tearing at body
        part boundaries.
        """
        ref_joints = self._ref_joints_px
        target = target_kps[:17].astype(np.float64)

        V = len(self._ref_verts)
        N_bones = len(self._bone_segs)
        deformed = np.zeros((V, 2), dtype=np.float64)

        # Pre-compute per-bone transforms
        bone_transforms = []
        for bi, (j1, j2) in enumerate(self._bone_segs):
            ref_a = ref_joints[j1]
            ref_b = ref_joints[j2]
            ref_ab = ref_b - ref_a
            ref_len = np.linalg.norm(ref_ab)

            tgt_a = target[j1]
            tgt_b = target[j2]
            tgt_ab = tgt_b - tgt_a
            tgt_len = np.linalg.norm(tgt_ab)

            bone_transforms.append({
                'ref_a': ref_a, 'ref_ab': ref_ab, 'ref_len': ref_len,
                'tgt_a': tgt_a, 'tgt_ab': tgt_ab, 'tgt_len': tgt_len,
            })

        for vi in range(V):
            pos = np.zeros(2, dtype=np.float64)
            v = self._ref_verts[vi]

            for bi in range(N_bones):
                w = self._weights[vi, bi]
                if w < 0.005:
                    continue

                bt = bone_transforms[bi]
                ref_len = bt['ref_len']
                tgt_len = bt['tgt_len']

                if ref_len < 1e-6:
                    # Degenerate bone: just translate
                    pos += w * (v + (bt['tgt_a'] - bt['ref_a']))
                    continue

                if tgt_len < 1e-6:
                    pos += w * bt['tgt_a']
                    continue

                # Decompose vertex position in reference bone local frame
                ref_dir = bt['ref_ab'] / ref_len
                ref_perp = np.array([-ref_dir[1], ref_dir[0]])

                v_rel = v - bt['ref_a']
                along = np.dot(v_rel, ref_dir)
                across = np.dot(v_rel, ref_perp)

                # Reconstruct in target bone local frame
                scale = tgt_len / ref_len
                tgt_dir = bt['tgt_ab'] / tgt_len
                tgt_perp = np.array([-tgt_dir[1], tgt_dir[0]])

                bone_pos = bt['tgt_a'] + along * scale * tgt_dir + across * scale * tgt_perp
                pos += w * bone_pos

            deformed[vi] = pos

        return deformed

        return deformed

    # ── Triangle warping ───────────────────────────────────────────

    def _warp_triangles(self, canvas: np.ndarray, deformed: np.ndarray):
        """Warp reference image to canvas via per-triangle affine warps."""
        if self._tri is None or self._ref_img is None:
            return

        H, W = canvas.shape[:2]
        ref_img = self._ref_img
        result = np.zeros((H, W, 4), dtype=np.uint8)

        for simplex in self._tri.simplices:
            src_pts = self._ref_verts[simplex].astype(np.float32)
            dst_pts = deformed[simplex].astype(np.float32)

            # Skip degenerate or off-screen
            area = abs((dst_pts[1, 0] - dst_pts[0, 0]) * (dst_pts[2, 1] - dst_pts[0, 1]) -
                       (dst_pts[2, 0] - dst_pts[0, 0]) * (dst_pts[1, 1] - dst_pts[0, 1]))
            if area < 1.0:
                continue

            x, y, w, h = cv2.boundingRect(dst_pts.astype(np.int32))
            x, y = max(x, 0), max(y, 0)
            w = min(w, W - x)
            h = min(h, H - y)
            if w <= 0 or h <= 0:
                continue

            # Inverse affine: dst → src
            M_inv = cv2.getAffineTransform(dst_pts, src_pts)

            # Adjust for crop
            M_crop = M_inv.copy()
            M_crop[0, 2] += M_inv[0, 0] * x + M_inv[0, 1] * y
            M_crop[1, 2] += M_inv[1, 0] * x + M_inv[1, 1] * y
            # Actually we need to offset by (x,y) in dst space
            # For pixel (px, py) in crop, world coord = (px+x, py+y)
            # src = M_inv @ [px+x, py+y, 1]
            # We want M_crop such that src = M_crop @ [px, py, 1]
            # M_crop[:, :2] = M_inv[:, :2], M_crop[:, 2] = M_inv @ [x, y, 1]
            M_crop[0, 2] = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
            M_crop[1, 2] = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]

            warped = cv2.warpAffine(ref_img, M_crop, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

            # Triangle mask
            mask_pts = dst_pts.copy()
            mask_pts[:, 0] -= x
            mask_pts[:, 1] -= y
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, mask_pts.astype(np.int32), 255)

            pm = (mask > 0) & (warped[:, :, 3] > 0)
            y_end = min(y + h, H)
            x_end = min(x + w, W)
            ah, aw = y_end - y, x_end - x
            pm = pm[:ah, :aw]

            result[y:y_end, x:x_end][pm] = warped[:ah, :aw][pm]

        # Composite
        alpha = result[:, :, 3].astype(np.float32) / 255.0
        has_content = alpha > 0
        if has_content.any():
            a = alpha[has_content, None]
            canvas[has_content] = (
                result[:, :, :3][has_content].astype(np.float32) * a +
                canvas[has_content].astype(np.float32) * (1 - a)
            ).astype(np.uint8)
