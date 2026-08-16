"""
Cartoon character style — full 133-keypoint wholebody rendering.

Renders a cartoon character with smooth curved limbs, detailed
finger rendering, face features from landmarks, and feet.
"""

import cv2
import numpy as np
from ...pose_extraction.utils import (
    BODY_SKELETON, FOOT_SKELETON, JOINT_GROUPS,
    LEFT_HAND_SKELETON, RIGHT_HAND_SKELETON, HAND_EDGES,
)


class CartoonStyle:
    """Render a cartoon character from 133 wholebody keypoints."""

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("cartoon", {})
        self.skin_color = tuple(style_cfg.get("skin_color", [255, 220, 185]))
        self.outline_color = tuple(style_cfg.get("outline_color", [40, 40, 40]))
        self.outfit_color = tuple(style_cfg.get("outfit_color", [100, 149, 237]))
        self.shoe_color = tuple(style_cfg.get("shoe_color", [50, 50, 60]))
        self.outline_width = style_cfg.get("outline_width", 2)

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        """Render cartoon character on canvas."""
        K = len(keypoints)
        if K < 17:
            return canvas

        kps = keypoints.astype(np.float64)

        # Compute body metrics
        torso_width = np.linalg.norm(kps[5] - kps[6])
        torso_height = np.linalg.norm(
            (kps[5] + kps[6]) / 2 - (kps[11] + kps[12]) / 2
        )

        if torso_width < 3 or torso_height < 3:
            return canvas

        # --- Dynamic z-ordering: determine which arm/leg is in front ---
        torso_cx = (kps[5][0] + kps[6][0]) / 2
        left_arm_cx = (kps[5][0] + kps[7][0] + kps[9][0]) / 3
        right_arm_cx = (kps[6][0] + kps[8][0] + kps[10][0]) / 3
        left_arm_in_front = (left_arm_cx - torso_cx) > (torso_cx - right_arm_cx)

        left_leg_cx = (kps[11][0] + kps[13][0] + kps[15][0]) / 3
        right_leg_cx = (kps[12][0] + kps[14][0] + kps[16][0]) / 3
        left_leg_in_front = (left_leg_cx - torso_cx) > (torso_cx - right_leg_cx)

        # Back limb indices
        back_arm = (6, 8, 10) if left_arm_in_front else (5, 7, 9)
        front_arm = (5, 7, 9) if left_arm_in_front else (6, 8, 10)
        back_leg = (12, 14, 16) if left_leg_in_front else (11, 13, 15)
        front_leg = (11, 13, 15) if left_leg_in_front else (12, 14, 16)

        # --- Draw order: back limbs, torso, front limbs, head ---

        # Draw back leg
        leg_width_upper = max(int(torso_width * 0.20), 8)
        leg_width_lower = max(int(torso_width * 0.15), 6)
        bl_h, bl_k, bl_a = back_leg
        self._draw_smooth_limb(canvas, kps[bl_h], kps[bl_k], leg_width_upper, leg_width_upper - 2, self.outfit_color)
        self._draw_smooth_limb(canvas, kps[bl_k], kps[bl_a], leg_width_lower, leg_width_lower - 3, self.outfit_color)

        # Draw back arm
        arm_width_upper = max(int(torso_width * 0.14), 5)
        arm_width_lower = max(int(torso_width * 0.11), 4)
        ba_s, ba_e, ba_w = back_arm
        self._draw_smooth_limb(canvas, kps[ba_s], kps[ba_e], arm_width_upper, arm_width_upper - 1, self.skin_color)
        self._draw_smooth_limb(canvas, kps[ba_e], kps[ba_w], arm_width_lower, arm_width_lower - 1, self.skin_color)

        # Draw smooth torso polygon with rounded edges
        shoulder_mid = (kps[5] + kps[6]) / 2
        hip_mid = (kps[11] + kps[12]) / 2
        shoulder_dir = kps[6] - kps[5]
        shoulder_dir_n = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-8)
        perp = np.array([-shoulder_dir_n[1], shoulder_dir_n[0]])

        # Create curved torso shape
        torso_pts = []
        # Left side (shoulder to hip)
        for t in np.linspace(0, 1, 8):
            pt = (1 - t) * kps[5] + t * kps[11]
            width_at_t = torso_width * (0.12 + 0.04 * np.sin(t * np.pi))
            pt += perp * width_at_t * (0.5 - t * 0.3)
            torso_pts.append(pt)
        # Right side (hip to shoulder)
        for t in np.linspace(0, 1, 8):
            pt = (1 - t) * kps[12] + t * kps[6]
            width_at_t = torso_width * (0.12 + 0.04 * np.sin(t * np.pi))
            pt -= perp * width_at_t * (0.5 - t * 0.3)
            torso_pts.append(pt)

        torso_arr = np.array(torso_pts, dtype=np.int32)
        cv2.fillPoly(canvas, [torso_arr], self.outfit_color, cv2.LINE_AA)
        cv2.polylines(canvas, [torso_arr], True, self.outline_color,
                      self.outline_width, cv2.LINE_AA)

        # Draw front leg
        fl_h, fl_k, fl_a = front_leg
        self._draw_smooth_limb(canvas, kps[fl_h], kps[fl_k], leg_width_upper, leg_width_upper - 2, self.outfit_color)
        self._draw_smooth_limb(canvas, kps[fl_k], kps[fl_a], leg_width_lower, leg_width_lower - 3, self.outfit_color)

        # Draw front arm
        fa_s, fa_e, fa_w = front_arm
        self._draw_smooth_limb(canvas, kps[fa_s], kps[fa_e], arm_width_upper, arm_width_upper - 1, self.skin_color)
        self._draw_smooth_limb(canvas, kps[fa_e], kps[fa_w], arm_width_lower, arm_width_lower - 1, self.skin_color)

        # Draw hands (circles with outline)
        hand_radius = max(int(torso_width * 0.06), 4)
        for idx in [9, 10]:
            pt = tuple(kps[idx].astype(int))
            cv2.circle(canvas, pt, hand_radius, self.skin_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, hand_radius, self.outline_color,
                       self.outline_width, cv2.LINE_AA)

        # Draw feet/shoes (rounded rectangles)
        foot_w = max(int(torso_width * 0.14), 6)
        foot_h = max(int(torso_width * 0.07), 4)
        for idx in [15, 16]:
            if idx >= K:
                continue
            pt = kps[idx].astype(int)
            # Determine foot direction from knee
            knee_idx = 13 if idx == 15 else 14
            foot_dir = kps[idx] - kps[knee_idx]
            angle = np.degrees(np.arctan2(foot_dir[1], foot_dir[0]))
            cv2.ellipse(canvas, tuple(pt), (foot_w, foot_h), angle + 90, 0, 360,
                       self.shoe_color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, tuple(pt), (foot_w, foot_h), angle + 90, 0, 360,
                       self.outline_color, self.outline_width, cv2.LINE_AA)

        # Draw neck
        neck_start = shoulder_mid.astype(int)
        neck_width = max(int(torso_width * 0.08), 3)
        head_center = kps[0].astype(int)
        neck_bottom = (shoulder_mid + (kps[0] - shoulder_mid) * 0.3).astype(int)
        cv2.line(canvas, tuple(neck_start), tuple(neck_bottom),
                 self.skin_color, neck_width * 2, cv2.LINE_AA)

        # Draw head
        if K > 4:
            eye_dist = np.linalg.norm(kps[1] - kps[2])
            head_radius = max(int(eye_dist * 2.0), 18)
        else:
            head_radius = max(int(torso_width * 0.3), 18)

        cv2.circle(canvas, tuple(head_center), head_radius,
                   self.skin_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(head_center), head_radius,
                   self.outline_color, self.outline_width, cv2.LINE_AA)

        # Hair (arc on top of head)
        hair_color = (50, 40, 35)
        cv2.ellipse(canvas, tuple(head_center), (head_radius, head_radius),
                    0, 180, 360, hair_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, tuple(head_center), (head_radius, head_radius),
                    0, 180, 360, self.outline_color, self.outline_width, cv2.LINE_AA)

        # Face features — use 68 face landmarks if available (indices 23-90)
        if K >= 91:
            self._draw_face_detailed(canvas, kps, scores, head_radius)
        elif K > 4:
            # Fallback: simple face from body keypoints
            eye_radius = max(int(head_radius * 0.13), 2)
            left_eye = tuple(kps[1].astype(int))
            right_eye = tuple(kps[2].astype(int))
            cv2.circle(canvas, left_eye, eye_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, right_eye, eye_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, left_eye, eye_radius, self.outline_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, right_eye, eye_radius, self.outline_color, -1, cv2.LINE_AA)
            mouth_y = int(head_center[1] + head_radius * 0.35)
            mouth_x = int(head_center[0])
            mouth_w = max(int(head_radius * 0.3), 3)
            cv2.ellipse(canvas, (mouth_x, mouth_y), (mouth_w, mouth_w // 2),
                       0, 0, 180, self.outline_color, 2, cv2.LINE_AA)

        # --- Draw detailed hands with fingers ---
        if K >= 133:
            self._draw_hand_detailed(canvas, kps, scores, torso_width, hand_start=91)
            self._draw_hand_detailed(canvas, kps, scores, torso_width, hand_start=112)

        # --- Draw feet with toes ---
        if K >= 23:
            self._draw_feet_detailed(canvas, kps, scores, torso_width)

        return canvas

    def _draw_face_detailed(self, canvas, kps, scores, head_radius):
        """Draw face features using 68 face landmarks (indices 23-90)."""
        min_score = 0.3

        # Face landmark groups (relative to offset 23):
        # Left eye: 36-41, Right eye: 42-47
        # Outer lip: 48-59, Inner lip: 60-67
        # Left eyebrow: 17-21, Right eyebrow: 22-26

        # Eyes — draw as filled shapes from landmarks
        for eye_start, eye_end in [(36, 42), (42, 48)]:
            eye_pts = []
            valid = True
            for i in range(eye_start, eye_end):
                idx = 23 + i
                if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                    valid = False
                    break
                eye_pts.append(kps[idx].astype(int))
            if valid and len(eye_pts) >= 4:
                pts = np.array(eye_pts, dtype=np.int32)
                # White of eye
                cv2.fillPoly(canvas, [pts], (255, 255, 255), cv2.LINE_AA)
                cv2.polylines(canvas, [pts], True, self.outline_color, 1, cv2.LINE_AA)
                # Pupil at center
                center = pts.mean(axis=0).astype(int)
                pupil_r = max(int(np.linalg.norm(pts[0] - pts[3]) * 0.2), 2)
                cv2.circle(canvas, tuple(center), pupil_r, self.outline_color, -1, cv2.LINE_AA)

        # Eyebrows
        for brow_start, brow_end in [(17, 22), (22, 27)]:
            brow_pts = []
            for i in range(brow_start, brow_end):
                idx = 23 + i
                if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                    continue
                brow_pts.append(kps[idx].astype(int))
            if len(brow_pts) >= 3:
                pts = np.array(brow_pts, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, self.outline_color, 2, cv2.LINE_AA)

        # Mouth — outer lip shape
        lip_pts = []
        for i in range(48, 60):
            idx = 23 + i
            if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                continue
            lip_pts.append(kps[idx].astype(int))
        if len(lip_pts) >= 8:
            pts = np.array(lip_pts, dtype=np.int32)
            # Lip fill
            lip_color = (130, 130, 200)
            cv2.fillPoly(canvas, [pts], lip_color, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, self.outline_color, 1, cv2.LINE_AA)

        # Nose tip
        nose_idx = 23 + 30  # nose tip landmark
        if nose_idx < len(kps) and (scores is None or scores[nose_idx] >= min_score):
            pt = tuple(kps[nose_idx].astype(int))
            nose_r = max(int(head_radius * 0.06), 2)
            cv2.circle(canvas, pt, nose_r, self.outline_color, -1, cv2.LINE_AA)

    def _draw_hand_detailed(self, canvas, kps, scores, torso_width, hand_start):
        """Draw a hand with individual fingers."""
        min_score = 0.3
        finger_width = max(int(torso_width * 0.03), 2)

        # Draw finger bones using HAND_EDGES
        for local_a, local_b in HAND_EDGES:
            a = hand_start + local_a
            b = hand_start + local_b
            if a >= len(kps) or b >= len(kps):
                continue
            if scores is not None and (scores[a] < min_score or scores[b] < min_score):
                continue
            pt1 = tuple(kps[a].astype(int))
            pt2 = tuple(kps[b].astype(int))
            cv2.line(canvas, pt1, pt2, self.skin_color, finger_width, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, self.outline_color, finger_width + self.outline_width, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, self.skin_color, finger_width, cv2.LINE_AA)

        # Fingertip dots
        for tip_offset in [4, 8, 12, 16, 20]:
            idx = hand_start + tip_offset
            if idx >= len(kps) or (scores is not None and scores[idx] < min_score):
                continue
            pt = tuple(kps[idx].astype(int))
            r = max(finger_width, 2)
            cv2.circle(canvas, pt, r, self.skin_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r, self.outline_color, 1, cv2.LINE_AA)

        # Wrist joint
        wrist_idx = hand_start
        if wrist_idx < len(kps) and (scores is None or scores[wrist_idx] >= min_score):
            pt = tuple(kps[wrist_idx].astype(int))
            r = max(int(torso_width * 0.04), 3)
            cv2.circle(canvas, pt, r, self.skin_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r, self.outline_color, 1, cv2.LINE_AA)

    def _draw_feet_detailed(self, canvas, kps, scores, torso_width):
        """Draw feet with toe details."""
        min_score = 0.3
        toe_width = max(int(torso_width * 0.04), 2)

        # Foot connections from FOOT_SKELETON  
        foot_bones = [
            (15, 17), (15, 18), (15, 19),  # left foot
            (16, 20), (16, 21), (16, 22),  # right foot
        ]
        for j1, j2 in foot_bones:
            if j1 >= len(kps) or j2 >= len(kps):
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            pt1 = tuple(kps[j1].astype(int))
            pt2 = tuple(kps[j2].astype(int))
            cv2.line(canvas, pt1, pt2, self.shoe_color, toe_width + 2, cv2.LINE_AA)
            cv2.circle(canvas, pt2, toe_width, self.shoe_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt2, toe_width, self.outline_color, 1, cv2.LINE_AA)

    def _draw_smooth_limb(
        self,
        canvas: np.ndarray,
        pt1: np.ndarray,
        pt2: np.ndarray,
        width1: int,
        width2: int,
        color: tuple,
    ):
        """Draw a tapered limb with smooth outline using a filled polygon."""
        p1 = pt1.astype(np.float64)
        p2 = pt2.astype(np.float64)

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1:
            return

        # Perpendicular direction
        perp = np.array([-direction[1], direction[0]]) / length

        # Create tapered capsule polygon
        n_pts = 6
        pts = []
        # One side
        for i in range(n_pts):
            t = i / (n_pts - 1)
            pt = p1 + direction * t
            w = width1 * (1 - t) + width2 * t
            pts.append(pt + perp * w / 2)
        # Other side (reversed)
        for i in range(n_pts - 1, -1, -1):
            t = i / (n_pts - 1)
            pt = p1 + direction * t
            w = width1 * (1 - t) + width2 * t
            pts.append(pt - perp * w / 2)

        polygon = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [polygon], color, cv2.LINE_AA)
        cv2.polylines(canvas, [polygon], True, self.outline_color,
                      self.outline_width, cv2.LINE_AA)

        # Round joints
        cv2.circle(canvas, tuple(p1.astype(int)), width1 // 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p2.astype(int)), width2 // 2, color, -1, cv2.LINE_AA)
