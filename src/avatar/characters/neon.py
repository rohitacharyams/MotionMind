"""
Neon glow character style — full 133-keypoint wholebody rendering.

Renders a glowing skeleton with bloom effects, including detailed
hand fingers, face contour, and foot keypoints.
"""

import cv2
import numpy as np
from ...pose_extraction.utils import (
    BODY_SKELETON, FOOT_SKELETON, JOINT_GROUPS,
    LEFT_HAND_SKELETON, RIGHT_HAND_SKELETON, FACE_SKELETON,
)


class NeonStyle:
    """Render a neon glow wholebody character from 133 keypoints."""

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("neon", {})
        self.glow_color = tuple(style_cfg.get("glow_color", [0, 255, 200]))
        self.core_color = tuple(style_cfg.get("core_color", [255, 255, 255]))
        self.hand_glow = tuple(style_cfg.get("hand_glow", [255, 180, 0]))
        self.face_glow = tuple(style_cfg.get("face_glow", [200, 100, 255]))
        self.glow_radius = style_cfg.get("glow_radius", 15)
        self.bone_width = style_cfg.get("bone_width", 4)

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        K = len(keypoints)
        bones = BODY_SKELETON + (FOOT_SKELETON if K >= 23 else [])

        # ── Glow layer (body) ──
        glow_layer = np.zeros_like(canvas)

        for j1, j2 in bones:
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            pt1 = tuple(keypoints[j1].astype(int))
            pt2 = tuple(keypoints[j2].astype(int))
            cv2.line(glow_layer, pt1, pt2, self.glow_color,
                     self.bone_width + self.glow_radius, cv2.LINE_AA)

        for i in range(min(K, 23)):
            if scores is not None and scores[i] < min_score:
                continue
            pt = tuple(keypoints[i].astype(int))
            cv2.circle(glow_layer, pt, self.glow_radius,
                       self.glow_color, -1, cv2.LINE_AA)

        # ── Hand glow (smaller, different color) ──
        if K >= 133:
            hand_bones = LEFT_HAND_SKELETON + RIGHT_HAND_SKELETON
            hand_glow_r = max(self.glow_radius // 3, 3)
            for j1, j2 in hand_bones:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(glow_layer, pt1, pt2, self.hand_glow,
                         self.bone_width // 2 + hand_glow_r, cv2.LINE_AA)

            # Fingertip accents
            for hand_start in [91, 112]:
                for tip_offset in [4, 8, 12, 16, 20]:
                    idx = hand_start + tip_offset
                    if idx >= K:
                        continue
                    if scores is not None and scores[idx] < min_score:
                        continue
                    pt = tuple(keypoints[idx].astype(int))
                    cv2.circle(glow_layer, pt, hand_glow_r + 2,
                               self.hand_glow, -1, cv2.LINE_AA)

        # ── Face glow ──
        if K >= 91:
            face_glow_r = max(self.glow_radius // 4, 2)
            for j1, j2 in FACE_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(glow_layer, pt1, pt2, self.face_glow,
                         face_glow_r, cv2.LINE_AA)

        # Blur for bloom
        blur_size = self.glow_radius * 2 + 1
        blur_size = min(blur_size, 31)
        glow_layer = cv2.GaussianBlur(glow_layer, (blur_size, blur_size), 0)
        canvas = cv2.add(canvas, glow_layer)

        # ── Core lines (body) ──
        for j1, j2 in bones:
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue
            pt1 = tuple(keypoints[j1].astype(int))
            pt2 = tuple(keypoints[j2].astype(int))
            cv2.line(canvas, pt1, pt2, self.core_color, self.bone_width, cv2.LINE_AA)

        for i in range(min(K, 23)):
            if scores is not None and scores[i] < min_score:
                continue
            pt = tuple(keypoints[i].astype(int))
            cv2.circle(canvas, pt, self.bone_width + 1, self.core_color, -1, cv2.LINE_AA)

        # ── Core lines (hands) ──
        if K >= 133:
            hand_w = max(self.bone_width // 2, 1)
            for j1, j2 in LEFT_HAND_SKELETON + RIGHT_HAND_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(canvas, pt1, pt2, self.core_color, hand_w, cv2.LINE_AA)

            # Fingertip dots
            for hand_start in [91, 112]:
                for tip_offset in [4, 8, 12, 16, 20]:
                    idx = hand_start + tip_offset
                    if idx >= K or (scores is not None and scores[idx] < min_score):
                        continue
                    pt = tuple(keypoints[idx].astype(int))
                    cv2.circle(canvas, pt, hand_w + 1, self.hand_glow, -1, cv2.LINE_AA)

        # ── Core lines (face) ──
        if K >= 91:
            for j1, j2 in FACE_SKELETON:
                if j1 >= K or j2 >= K:
                    continue
                if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                    continue
                pt1 = tuple(keypoints[j1].astype(int))
                pt2 = tuple(keypoints[j2].astype(int))
                cv2.line(canvas, pt1, pt2, self.face_glow, 1, cv2.LINE_AA)

        return canvas
