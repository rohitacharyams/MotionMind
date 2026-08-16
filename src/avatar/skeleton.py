"""
2D Skeleton definition and manipulation.

Defines a hierarchical skeleton with joints and bones,
supporting both the full 133-keypoint whole-body layout
and a simplified body-only layout.
"""

import numpy as np
from dataclasses import dataclass, field
from ..pose_extraction.utils import (
    BODY_SKELETON, FOOT_SKELETON, LEFT_HAND_SKELETON,
    RIGHT_HAND_SKELETON, WHOLEBODY_SKELETON, JOINT_GROUPS,
    BODY_KEYPOINTS,
)


@dataclass
class Joint:
    """A single joint in the skeleton."""
    index: int
    name: str
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rotation: float = 0.0
    length: float = 0.0  # bone length to parent


class Skeleton2D:
    """Hierarchical 2D skeleton for avatar animation.
    
    Supports forward kinematics (FK) from joint angles and
    provides the structure needed for inverse kinematics (IK).
    """

    def __init__(self, skeleton_type: str = "body_17"):
        self.skeleton_type = skeleton_type
        self.joints: list[Joint] = []
        self.bones: list[tuple[int, int]] = []
        self._build_skeleton()

    def _build_skeleton(self):
        """Build skeleton hierarchy based on type."""
        if self.skeleton_type == "wholebody_133":
            self._build_wholebody()
        else:
            self._build_body17()

    def _build_body17(self):
        """Build 17-joint body skeleton (COCO format)."""
        for i, name in enumerate(BODY_KEYPOINTS):
            self.joints.append(Joint(index=i, name=name))

        # Parent hierarchy — hip center is virtual root
        # Torso chain: hips → shoulders → head
        # Limbs branch from shoulders/hips
        parent_map = {
            11: None,   # left_hip (root)
            12: 11,     # right_hip → left_hip
            5: 11,      # left_shoulder → left_hip (torso)
            6: 12,      # right_shoulder → right_hip (torso)
            0: 5,       # nose → left_shoulder (neck approximation)
            1: 0, 2: 0, # eyes → nose
            3: 1, 4: 2, # ears → eyes
            7: 5, 9: 7, # left arm: shoulder → elbow → wrist
            8: 6, 10: 8, # right arm
            13: 11, 15: 13, # left leg: hip → knee → ankle
            14: 12, 16: 14, # right leg
        }

        for idx, parent_idx in parent_map.items():
            self.joints[idx].parent = parent_idx
            if parent_idx is not None:
                self.joints[parent_idx].children.append(idx)

        self.bones = BODY_SKELETON.copy()

    def _build_wholebody(self):
        """Build 133-joint whole-body skeleton with full parent hierarchy.
        
        Layout: Body(0-16), Feet(17-22), Face(23-90),
                Left hand(91-111), Right hand(112-132)
        """
        # Joint names
        names = list(BODY_KEYPOINTS)
        names += ["left_big_toe", "left_small_toe", "left_heel",
                   "right_big_toe", "right_small_toe", "right_heel"]
        for i in range(68):
            names.append(f"face_{i}")
        finger_names = ["wrist", "thumb_1", "thumb_2", "thumb_3", "thumb_tip",
                        "index_1", "index_2", "index_3", "index_tip",
                        "middle_1", "middle_2", "middle_3", "middle_tip",
                        "ring_1", "ring_2", "ring_3", "ring_tip",
                        "pinky_1", "pinky_2", "pinky_3", "pinky_tip"]
        for fn in finger_names:
            names.append(f"left_{fn}")
        for fn in finger_names:
            names.append(f"right_{fn}")

        for i in range(133):
            name = names[i] if i < len(names) else f"joint_{i}"
            self.joints.append(Joint(index=i, name=name))

        # ── Parent hierarchy ──
        # Body (same as body17)
        body_parents = {
            11: None, 12: 11, 5: 11, 6: 12,
            0: 5, 1: 0, 2: 0, 3: 1, 4: 2,
            7: 5, 9: 7, 8: 6, 10: 8,
            13: 11, 15: 13, 14: 12, 16: 14,
        }
        # Feet → ankles
        foot_parents = {17: 15, 18: 15, 19: 15, 20: 16, 21: 16, 22: 16}
        # Face → nose (all face landmarks parent to nose)
        face_parents = {i: 0 for i in range(23, 91)}
        # Hands → wrists (21 joints per hand)
        # Left hand (91-111): wrist=91 parents to body wrist(9)
        hand_l_parents = {91: 9}
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4),      # thumb
                     (0, 5), (5, 6), (6, 7), (7, 8),      # index
                     (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                     (0, 13), (13, 14), (14, 15), (15, 16),# ring
                     (0, 17), (17, 18), (18, 19), (19, 20)]:# pinky
            hand_l_parents[91 + b] = 91 + a
        # Right hand (112-132)
        hand_r_parents = {112: 10}
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4),
                     (0, 5), (5, 6), (6, 7), (7, 8),
                     (0, 9), (9, 10), (10, 11), (11, 12),
                     (0, 13), (13, 14), (14, 15), (15, 16),
                     (0, 17), (17, 18), (18, 19), (19, 20)]:
            hand_r_parents[112 + b] = 112 + a

        all_parents = {**body_parents, **foot_parents, **face_parents,
                       **hand_l_parents, **hand_r_parents}

        for idx, parent_idx in all_parents.items():
            self.joints[idx].parent = parent_idx
            if parent_idx is not None:
                self.joints[parent_idx].children.append(idx)

        self.bones = WHOLEBODY_SKELETON.copy()

    def set_pose(self, keypoints: np.ndarray):
        """Set joint positions from keypoint array.
        
        Args:
            keypoints: (K, 2) keypoint coordinates.
        """
        K = min(len(keypoints), len(self.joints))
        for i in range(K):
            self.joints[i].position = keypoints[i].copy()

        # Compute bone lengths
        for j1, j2 in self.bones:
            if j1 < K and j2 < K:
                dist = np.linalg.norm(
                    self.joints[j1].position - self.joints[j2].position
                )
                self.joints[j2].length = dist

    def get_pose(self) -> np.ndarray:
        """Get current joint positions as array.
        
        Returns:
            (K, 2) array.
        """
        return np.array([j.position for j in self.joints])

    def get_bone_vectors(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get start/end positions for each bone.
        
        Returns:
            List of (start_pos, end_pos) tuples.
        """
        vectors = []
        for j1, j2 in self.bones:
            vectors.append((
                self.joints[j1].position.copy(),
                self.joints[j2].position.copy(),
            ))
        return vectors

    def get_limb_chain(self, group: str) -> list[int]:
        """Get ordered joint indices for a limb group.
        
        Args:
            group: One of 'left_arm', 'right_arm', 'left_leg', 'right_leg'.
        """
        return JOINT_GROUPS.get(group, [])

    def compute_angles(self) -> dict[str, float]:
        """Compute joint angles (radians) for major joints."""
        angles = {}
        angle_joints = {
            "left_elbow": (5, 7, 9),
            "right_elbow": (6, 8, 10),
            "left_knee": (11, 13, 15),
            "right_knee": (12, 14, 16),
            "left_shoulder": (0, 5, 7),
            "right_shoulder": (0, 6, 8),
            "left_hip": (5, 11, 13),
            "right_hip": (6, 12, 14),
        }

        for name, (a, b, c) in angle_joints.items():
            if a < len(self.joints) and b < len(self.joints) and c < len(self.joints):
                v1 = self.joints[a].position - self.joints[b].position
                v2 = self.joints[c].position - self.joints[b].position
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angles[name] = float(np.arccos(np.clip(cos_angle, -1, 1)))

        return angles
