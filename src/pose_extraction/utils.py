"""
COCO-WholeBody 133-keypoint layout and skeleton connectivity.

Keypoints:
  0-16   : Body (17 keypoints)
  17-22  : Feet (6 keypoints)
  23-90  : Face (68 keypoints)
  91-111 : Left hand (21 keypoints)
  112-132: Right hand (21 keypoints)
"""

import numpy as np

# Body keypoint names (COCO 17)
BODY_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Foot keypoints
FOOT_KEYPOINTS = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# Body skeleton connections (pairs of keypoint indices)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9),                              # left arm
    (6, 8), (8, 10),                             # right arm
    (5, 11), (6, 12),                            # torso
    (11, 12),                                    # hips
    (11, 13), (13, 15),                          # left leg
    (12, 14), (14, 16),                          # right leg
]

# Foot connections (offset by 17)
FOOT_SKELETON = [
    (15, 17), (15, 18), (15, 19),  # left foot
    (16, 20), (16, 21), (16, 22),  # right foot
]

# Hand connections (21 keypoints per hand)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),            # palm
]

LEFT_HAND_SKELETON = [(a + 91, b + 91) for a, b in HAND_EDGES]
RIGHT_HAND_SKELETON = [(a + 112, b + 112) for a, b in HAND_EDGES]

# Face contour connections (68 keypoints, indices 23-90)
# Standard 68-point face landmark layout:
#   0-16: jaw contour, 17-21: left eyebrow, 22-26: right eyebrow,
#   27-30: nose bridge, 31-35: nose bottom, 36-41: left eye,
#   42-47: right eye, 48-59: outer lip, 60-67: inner lip
FACE_EDGES = (
    # Jaw contour (0-16)
    [(i, i + 1) for i in range(0, 16)] +
    # Left eyebrow (17-21)
    [(i, i + 1) for i in range(17, 21)] +
    # Right eyebrow (22-26)
    [(i, i + 1) for i in range(22, 26)] +
    # Nose bridge (27-30)
    [(i, i + 1) for i in range(27, 30)] +
    # Nose bottom (31-35)
    [(i, i + 1) for i in range(31, 35)] +
    # Left eye (36-41, closed loop)
    [(i, i + 1) for i in range(36, 41)] + [(41, 36)] +
    # Right eye (42-47, closed loop)
    [(i, i + 1) for i in range(42, 47)] + [(47, 42)] +
    # Outer lip (48-59, closed loop)
    [(i, i + 1) for i in range(48, 59)] + [(59, 48)] +
    # Inner lip (60-67, closed loop)
    [(i, i + 1) for i in range(60, 67)] + [(67, 60)]
)

FACE_SKELETON = [(a + 23, b + 23) for a, b in FACE_EDGES]

# Complete skeleton (with face)
WHOLEBODY_SKELETON = BODY_SKELETON + FOOT_SKELETON + LEFT_HAND_SKELETON + RIGHT_HAND_SKELETON + FACE_SKELETON

# Semantic joint groups for IK and rendering
JOINT_GROUPS = {
    "head": list(range(0, 5)),
    "torso": [5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "left_foot": [17, 18, 19],
    "right_foot": [20, 21, 22],
    "face": list(range(23, 91)),
    "left_hand": list(range(91, 112)),
    "right_hand": list(range(112, 133)),
}


def compute_hip_center(keypoints: np.ndarray) -> np.ndarray:
    """Compute hip center as midpoint of left_hip (11) and right_hip (12).
    
    Args:
        keypoints: (..., K, 2|3) array of keypoints.
    Returns:
        (..., 2|3) hip center coordinates.
    """
    return (keypoints[..., 11, :] + keypoints[..., 12, :]) / 2.0


def compute_torso_size(keypoints: np.ndarray) -> np.ndarray:
    """Compute torso size as distance from hip center to shoulder center.
    
    Args:
        keypoints: (..., K, 2|3) array of keypoints.
    Returns:
        (...,) torso sizes.
    """
    hip = compute_hip_center(keypoints)
    shoulder = (keypoints[..., 5, :] + keypoints[..., 6, :]) / 2.0
    return np.linalg.norm(shoulder - hip, axis=-1)
