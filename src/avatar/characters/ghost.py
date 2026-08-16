"""
Ghost / Semantic character style — ethereal translucent body shape.

Renders the full human body as a smooth, glowing translucent figure
with soft edges and inner energy effects. Think motion-capture ghost,
dance game silhouette, or spectral body visualization.
"""

import cv2
import numpy as np
from ...pose_extraction.utils import (
    BODY_SKELETON, JOINT_GROUPS, HAND_EDGES,
    LEFT_HAND_SKELETON, RIGHT_HAND_SKELETON,
)

# Same limb segments as silhouette
LIMB_SEGMENTS = {
    "left_upper_arm": (5, 7),
    "left_forearm": (7, 9),
    "right_upper_arm": (6, 8),
    "right_forearm": (8, 10),
    "left_thigh": (11, 13),
    "left_shin": (13, 15),
    "right_thigh": (12, 14),
    "right_shin": (14, 16),
}

LIMB_WIDTH_RATIOS = {
    "left_upper_arm": (0.18, 0.15),
    "left_forearm": (0.15, 0.11),
    "right_upper_arm": (0.18, 0.15),
    "right_forearm": (0.15, 0.11),
    "left_thigh": (0.24, 0.18),
    "left_shin": (0.18, 0.13),
    "right_thigh": (0.24, 0.18),
    "right_shin": (0.18, 0.13),
}


class GhostStyle:
    """Render an ethereal ghost/semantic body visualization."""

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("ghost", {})
        self.body_color = list(style_cfg.get("body_color", [180, 220, 255]))
        self.edge_color = list(style_cfg.get("edge_color", [100, 180, 255]))
        self.core_color = list(style_cfg.get("core_color", [255, 255, 255]))
        self.body_opacity = style_cfg.get("body_opacity", 0.45)
        self.glow_size = style_cfg.get("glow_size", 25)
        self.inner_glow = style_cfg.get("inner_glow", True)
        self.edge_pulse = style_cfg.get("edge_pulse", True)
        self.skeleton_visible = style_cfg.get("skeleton_visible", False)

        # Animated pulse state
        self._frame_counter = 0

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
        torso_width = np.linalg.norm(kps[5] - kps[6])
        if torso_width < 3:
            return canvas

        h, w = canvas.shape[:2]
        self._frame_counter += 1

        # --- Build body mask ---
        mask = np.zeros((h, w), dtype=np.uint8)
        self._draw_body_shape(mask, kps, scores, K, torso_width, min_score)

        # --- Morphological smoothing ---
        k_size = max(int(torso_width * 0.05), 3) | 1  # ensure odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 1.5)

        # --- Outer glow layer ---
        glow_k = max(self.glow_size, 3) | 1
        glow_mask = cv2.GaussianBlur(mask, (glow_k, glow_k), glow_k * 0.4)

        # Edge pulse animation
        pulse = 1.0
        if self.edge_pulse:
            pulse = 0.7 + 0.3 * np.sin(self._frame_counter * 0.15)

        # --- Compose layers onto canvas ---
        # 1. Outer glow (subtle, wide)
        glow_alpha = (glow_mask.astype(np.float32) / 255.0) * 0.2 * pulse
        for c in range(3):
            canvas[:, :, c] = np.clip(
                canvas[:, :, c].astype(np.float32)
                + glow_alpha * self.edge_color[c],
                0, 255
            ).astype(np.uint8)

        # 2. Main body fill (translucent)
        body_alpha = (mask.astype(np.float32) / 255.0) * self.body_opacity
        for c in range(3):
            canvas[:, :, c] = np.clip(
                canvas[:, :, c].astype(np.float32) * (1 - body_alpha)
                + self.body_color[c] * body_alpha,
                0, 255
            ).astype(np.uint8)

        # 3. Inner core glow (brighter center skeleton line)
        if self.inner_glow:
            core_mask = np.zeros((h, w), dtype=np.uint8)
            self._draw_skeleton_core(core_mask, kps, scores, K, torso_width, min_score)
            core_blur = cv2.GaussianBlur(core_mask, (7, 7), 2.0)
            core_alpha = (core_blur.astype(np.float32) / 255.0) * 0.6
            for c in range(3):
                canvas[:, :, c] = np.clip(
                    canvas[:, :, c].astype(np.float32)
                    + core_alpha * self.core_color[c] * 0.5,
                    0, 255
                ).astype(np.uint8)

        # 4. Edge contour highlight
        edge_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_w = max(int(torso_width * 0.02), 1)
        edge_c = [int(c * pulse) for c in self.edge_color]
        cv2.drawContours(canvas, contours, -1, edge_c, edge_w, cv2.LINE_AA)

        # 5. Optional skeleton wireframe inside
        if self.skeleton_visible:
            skel_alpha = 0.25
            overlay = canvas.copy()
            for j1, j2 in BODY_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(kps[j1].astype(int))
                pt2 = tuple(kps[j2].astype(int))
                cv2.line(overlay, pt1, pt2, self.core_color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, skel_alpha, canvas, 1 - skel_alpha, 0, canvas)

        return canvas

    def _draw_body_shape(self, mask, kps, scores, K, torso_width, min_score):
        """Draw the filled body shape on the mask."""
        # Torso polygon
        shoulder_dir = kps[6] - kps[5]
        shoulder_norm = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-8)
        perp = np.array([-shoulder_norm[1], shoulder_norm[0]])

        torso_pts = []
        for t_val in np.linspace(0, 1, 12):
            l_pt = (1 - t_val) * kps[5] + t_val * kps[11]
            r_pt = (1 - t_val) * kps[6] + t_val * kps[12]
            bulge = torso_width * 0.1 * np.sin(t_val * np.pi)
            l_pt = l_pt - perp * bulge
            r_pt = r_pt + perp * bulge
            torso_pts.append(l_pt)
            torso_pts.insert(0, r_pt)

        torso_arr = np.array(torso_pts, dtype=np.int32)
        cv2.fillPoly(mask, [torso_arr], 255, cv2.LINE_AA)

        # Head
        head_center = kps[0].astype(int)
        if K > 4:
            eye_dist = np.linalg.norm(kps[1] - kps[2])
            head_radius = max(int(eye_dist * 2.0), 16)
        else:
            head_radius = int(torso_width * 0.28)
        cv2.circle(mask, tuple(head_center), head_radius, 255, -1, cv2.LINE_AA)

        # Neck
        shoulder_mid = ((kps[5] + kps[6]) / 2).astype(int)
        neck_width = max(int(torso_width * 0.11), 4)
        neck_top = (kps[0] + (shoulder_mid - kps[0].astype(int)) * 0.3).astype(int)
        cv2.line(mask, tuple(neck_top), tuple(shoulder_mid), 255, neck_width * 2, cv2.LINE_AA)

        # Limbs
        for limb_name, (j1, j2) in LIMB_SEGMENTS.items():
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            w1_ratio, w2_ratio = LIMB_WIDTH_RATIOS.get(limb_name, (0.16, 0.13))
            w1 = max(int(torso_width * w1_ratio), 5)
            w2 = max(int(torso_width * w2_ratio), 4)
            self._draw_tapered_limb(mask, kps[j1], kps[j2], w1, w2)

        # Feet
        foot_w = max(int(torso_width * 0.13), 5)
        foot_h = max(int(torso_width * 0.07), 3)
        for idx in [15, 16]:
            pt = tuple(kps[idx].astype(int))
            cv2.ellipse(mask, pt, (foot_w, foot_h), 0, 0, 360, 255, -1, cv2.LINE_AA)

        if K >= 23:
            toe_w = max(int(torso_width * 0.04), 2)
            for j1, j2 in [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(kps[j1].astype(int))
                pt2 = tuple(kps[j2].astype(int))
                cv2.line(mask, pt1, pt2, 255, toe_w + 2, cv2.LINE_AA)

        # Hands
        if K >= 133:
            finger_width = max(int(torso_width * 0.04), 2)
            for hand_start in [91, 112]:
                for local_a, local_b in HAND_EDGES:
                    a = hand_start + local_a
                    b = hand_start + local_b
                    if a >= K or b >= K:
                        continue
                    if scores is not None and (scores[a] < min_score or scores[b] < min_score):
                        continue
                    pt1 = tuple(kps[a].astype(int))
                    pt2 = tuple(kps[b].astype(int))
                    cv2.line(mask, pt1, pt2, 255, finger_width, cv2.LINE_AA)
        else:
            hand_r = max(int(torso_width * 0.07), 3)
            for idx in [9, 10]:
                pt = tuple(kps[idx].astype(int))
                cv2.circle(mask, pt, hand_r, 255, -1, cv2.LINE_AA)

        # Face contour
        if K >= 91:
            jaw_pts = []
            for i in range(17):
                idx = 23 + i
                if idx >= K or (scores is not None and scores[idx] < min_score):
                    continue
                jaw_pts.append(kps[idx].astype(int))
            if len(jaw_pts) >= 10:
                pts = np.array(jaw_pts, dtype=np.int32)
                cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)

    def _draw_skeleton_core(self, mask, kps, scores, K, torso_width, min_score):
        """Draw thin skeleton lines for inner glow effect."""
        core_w = max(int(torso_width * 0.04), 2)
        for j1, j2 in BODY_SKELETON:
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            pt1 = tuple(kps[j1].astype(int))
            pt2 = tuple(kps[j2].astype(int))
            cv2.line(mask, pt1, pt2, 255, core_w, cv2.LINE_8)

        # Joint highlights
        joint_r = max(int(torso_width * 0.04), 2)
        for i in range(min(K, 17)):
            if scores is not None and scores[i] < min_score:
                continue
            pt = tuple(kps[i].astype(int))
            cv2.circle(mask, pt, joint_r, 255, -1, cv2.LINE_8)

    def _draw_tapered_limb(self, mask, pt1, pt2, w1, w2):
        """Draw a tapered limb polygon."""
        d = pt2 - pt1
        length = np.linalg.norm(d)
        if length < 1:
            return
        perp = np.array([-d[1], d[0]]) / length
        pts = np.array([
            pt1 + perp * w1 / 2,
            pt2 + perp * w2 / 2,
            pt2 - perp * w2 / 2,
            pt1 - perp * w1 / 2,
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)
