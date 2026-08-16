"""
Stick figure character style — full 133-keypoint wholebody rendering.

Shows body skeleton, feet, finger details, and face landmarks.
"""

import cv2
import numpy as np
from ...pose_extraction.utils import (
    BODY_SKELETON, FOOT_SKELETON, JOINT_GROUPS,
    LEFT_HAND_SKELETON, RIGHT_HAND_SKELETON, FACE_SKELETON,
)


class StickFigureStyle:
    """Render a full wholebody stick figure from 133 keypoints."""

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("stick_figure", {})
        self.joint_color = tuple(style_cfg.get("joint_color", [255, 255, 255]))
        self.bone_color = tuple(style_cfg.get("bone_color", [0, 200, 255]))
        self.hand_color = tuple(style_cfg.get("hand_color", [180, 255, 180]))
        self.face_color = tuple(style_cfg.get("face_color", [255, 200, 200]))
        self.foot_color = tuple(style_cfg.get("foot_color", [200, 200, 255]))
        self.joint_radius = style_cfg.get("joint_radius", 6)
        self.bone_width = style_cfg.get("bone_width", 3)

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        K = len(keypoints)

        # --- Body bones ---
        bones = BODY_SKELETON
        for j1, j2 in bones:
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            pt1 = tuple(keypoints[j1].astype(int))
            pt2 = tuple(keypoints[j2].astype(int))
            alpha = min(scores[j1], scores[j2]) if scores is not None else 1.0
            color = tuple(int(c * min(alpha, 1.0)) for c in self.bone_color)
            cv2.line(canvas, pt1, pt2, color, self.bone_width, cv2.LINE_AA)

        # --- Foot bones (different color) ---
        if K >= 23:
            for j1, j2 in FOOT_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(canvas, pt1, pt2, self.foot_color, max(self.bone_width - 1, 1), cv2.LINE_AA)

        # --- Body + foot joints ---
        for i in range(min(K, 23)):
            if scores is not None and scores[i] < min_score:
                continue
            pt = tuple(keypoints[i].astype(int))
            r = self.joint_radius if i < 17 else self.joint_radius - 2
            cv2.circle(canvas, pt, r, self.joint_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r + 1, self.bone_color, 1, cv2.LINE_AA)

        # --- Hand skeleton ---
        if K >= 133:
            hand_bones = LEFT_HAND_SKELETON + RIGHT_HAND_SKELETON
            for j1, j2 in hand_bones:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(canvas, pt1, pt2, self.hand_color, max(self.bone_width - 1, 1), cv2.LINE_AA)

            # Hand joints
            for i in list(range(91, 112)) + list(range(112, 133)):
                if i >= K:
                    continue
                if scores is not None and scores[i] < min_score:
                    continue
                pt = tuple(keypoints[i].astype(int))
                # Fingertips are slightly larger
                is_tip = (i - 91) % 21 in (4, 8, 12, 16, 20) if i < 112 else (i - 112) % 21 in (4, 8, 12, 16, 20)
                r = 3 if is_tip else 2
                cv2.circle(canvas, pt, r, self.hand_color, -1, cv2.LINE_AA)

        # --- Face landmarks ---
        if K >= 91:
            for j1, j2 in FACE_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(canvas, pt1, pt2, self.face_color, 1, cv2.LINE_AA)

            # Face keypoint dots (LINE_8 for speed — dots too small for AA)
            for i in range(23, 91):
                if i >= K:
                    continue
                if scores is not None and scores[i] < min_score:
                    continue
                pt = tuple(keypoints[i].astype(int))
                cv2.circle(canvas, pt, 1, self.face_color, -1, cv2.LINE_8)

        return canvas
