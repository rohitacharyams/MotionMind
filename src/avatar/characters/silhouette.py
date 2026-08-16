"""
Silhouette character style — full 133-keypoint wholebody rendering.

Renders a smooth filled body silhouette with natural body proportions,
including hand shapes, face contour, and feet detail.
"""

import cv2
import numpy as np
from ...pose_extraction.utils import (
    BODY_SKELETON, JOINT_GROUPS, HAND_EDGES,
    LEFT_HAND_SKELETON, RIGHT_HAND_SKELETON,
)


# Body segment definitions (pairs of keypoint indices)
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

# Limb widths relative to torso width (top, bottom)
LIMB_WIDTH_RATIOS = {
    "left_upper_arm": (0.17, 0.14),
    "left_forearm": (0.14, 0.10),
    "right_upper_arm": (0.17, 0.14),
    "right_forearm": (0.14, 0.10),
    "left_thigh": (0.22, 0.17),
    "left_shin": (0.17, 0.12),
    "right_thigh": (0.22, 0.17),
    "right_shin": (0.17, 0.12),
}


class SilhouetteStyle:
    """Render a smooth filled silhouette character from keypoints."""

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("silhouette", {})
        self.fill_color = tuple(style_cfg.get("fill_color", [255, 255, 255]))
        self.outline_color = tuple(style_cfg.get("outline_color", [0, 200, 255]))
        self.outline_width = style_cfg.get("outline_width", 2)

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        """Render smooth silhouette on canvas."""
        K = len(keypoints)
        if K < 17:
            return canvas

        kps = keypoints.astype(np.float64)
        torso_width = np.linalg.norm(kps[5] - kps[6])
        if torso_width < 3:
            return canvas

        # Draw on a separate mask for smooth morphological processing
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)

        # Draw torso as a smooth polygon
        shoulder_dir = kps[6] - kps[5]
        shoulder_norm = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-8)
        perp = np.array([-shoulder_norm[1], shoulder_norm[0]])

        # Create smooth torso shape
        torso_pts = []
        for t in np.linspace(0, 1, 10):
            l_pt = (1 - t) * kps[5] + t * kps[11]
            r_pt = (1 - t) * kps[6] + t * kps[12]
            # Add some width
            bulge = torso_width * 0.08 * np.sin(t * np.pi)
            l_pt = l_pt - perp * bulge
            r_pt = r_pt + perp * bulge
            torso_pts.append(l_pt)
            torso_pts.insert(0, r_pt)  # Build both sides

        torso_arr = np.array(torso_pts, dtype=np.int32)
        cv2.fillPoly(mask, [torso_arr], 255, cv2.LINE_AA)

        # Draw head
        head_center = kps[0].astype(int)
        if K > 4:
            eye_dist = np.linalg.norm(kps[1] - kps[2])
            head_radius = max(int(eye_dist * 1.8), 15)
        else:
            head_radius = int(torso_width * 0.25)
        cv2.circle(mask, tuple(head_center), head_radius, 255, -1, cv2.LINE_AA)

        # Draw neck
        shoulder_mid = ((kps[5] + kps[6]) / 2).astype(int)
        neck_width = max(int(torso_width * 0.10), 4)
        neck_top = (kps[0] + (shoulder_mid - kps[0].astype(int)) * 0.3).astype(int)
        cv2.line(mask, tuple(neck_top), tuple(shoulder_mid), 255, neck_width * 2, cv2.LINE_AA)

        # Draw limbs as tapered polygons
        for limb_name, (j1, j2) in LIMB_SEGMENTS.items():
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue

            w1_ratio, w2_ratio = LIMB_WIDTH_RATIOS.get(limb_name, (0.15, 0.12))
            w1 = max(int(torso_width * w1_ratio), 5)
            w2 = max(int(torso_width * w2_ratio), 4)

            self._draw_tapered_limb(mask, kps[j1], kps[j2], w1, w2)

        # Draw feet
        foot_w = max(int(torso_width * 0.12), 5)
        foot_h = max(int(torso_width * 0.06), 3)
        for idx in [15, 16]:
            pt = tuple(kps[idx].astype(int))
            cv2.ellipse(mask, pt, (foot_w, foot_h), 0, 0, 360, 255, -1, cv2.LINE_AA)

        # Detailed feet (toes) if available
        if K >= 23:
            toe_w = max(int(torso_width * 0.04), 2)
            foot_connections = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
            for j1, j2 in foot_connections:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(kps[j1].astype(int))
                pt2 = tuple(kps[j2].astype(int))
                cv2.line(mask, pt1, pt2, 255, toe_w + 2, cv2.LINE_AA)
                cv2.circle(mask, pt2, toe_w + 1, 255, -1, cv2.LINE_AA)

        # Detailed hands with fingers if available
        if K >= 133:
            finger_width = max(int(torso_width * 0.035), 2)
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
                # Fingertip circles
                for tip_offset in [4, 8, 12, 16, 20]:
                    idx = hand_start + tip_offset
                    if idx >= K or (scores is not None and scores[idx] < min_score):
                        continue
                    pt = tuple(kps[idx].astype(int))
                    cv2.circle(mask, pt, finger_width, 255, -1, cv2.LINE_AA)
        else:
            # Fallback: simple hand circles
            hand_r = max(int(torso_width * 0.06), 3)
            for idx in [9, 10]:
                pt = tuple(kps[idx].astype(int))
                cv2.circle(mask, pt, hand_r, 255, -1, cv2.LINE_AA)

        # Face contour (jaw line) for better head shape
        if K >= 91:
            # Jaw contour: face landmarks 0-16 (indices 23-39)
            jaw_pts = []
            for i in range(0, 17):
                idx = 23 + i
                if idx >= K or (scores is not None and scores[idx] < min_score):
                    continue
                jaw_pts.append(kps[idx].astype(int))
            if len(jaw_pts) >= 10:
                pts = np.array(jaw_pts, dtype=np.int32)
                cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)

        # Morphological smoothing — close small gaps, smooth jagged edges
        kernel_size = max(int(torso_width * 0.04), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Light blur for anti-aliased edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0.8)

        # Apply fill using mask (use cv2 for speed, avoid float32 arrays)
        fill = np.full_like(canvas, self.fill_color, dtype=np.uint8)
        inv_mask = cv2.bitwise_not(mask)
        canvas = cv2.add(
            cv2.bitwise_and(canvas, canvas, mask=inv_mask),
            cv2.bitwise_and(fill, fill, mask=mask),
        )

        # Draw smooth outline
        if self.outline_width > 0:
            edge_mask = (mask > 128).astype(np.uint8)
            contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Smooth contours
            smooth_contours = []
            for cnt in contours:
                if len(cnt) > 5:
                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)
                smooth_contours.append(cnt)
            cv2.drawContours(canvas, smooth_contours, -1, self.outline_color,
                           self.outline_width, cv2.LINE_AA)

        return canvas

    def _draw_tapered_limb(
        self,
        mask: np.ndarray,
        pt1: np.ndarray,
        pt2: np.ndarray,
        width1: int,
        width2: int,
    ):
        """Draw a tapered limb shape on the mask."""
        p1 = pt1.astype(np.float64)
        p2 = pt2.astype(np.float64)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1:
            return

        perp = np.array([-direction[1], direction[0]]) / length

        # Create tapered polygon
        n_pts = 6
        pts = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            pt = p1 + direction * t
            w = width1 * (1 - t) + width2 * t
            pts.append(pt + perp * w / 2)
        for i in range(n_pts - 1, -1, -1):
            t = i / (n_pts - 1)
            pt = p1 + direction * t
            w = width1 * (1 - t) + width2 * t
            pts.append(pt - perp * w / 2)

        polygon = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
        cv2.circle(mask, tuple(p1.astype(int)), width1 // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, tuple(p2.astype(int)), width2 // 2, 255, -1, cv2.LINE_AA)
