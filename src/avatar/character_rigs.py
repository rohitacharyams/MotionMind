"""
Rigged 2D character system with IK-driven animation.

Provides pre-built character rigs with proper proportions,
IK chain definitions, and motion retargeting. Characters
have defined body segments that can be rendered by any style.
"""

import numpy as np
from ..pose_extraction.utils import JOINT_GROUPS


# ── Character Rig Definitions ──
# Each rig defines body proportions relative to torso length (hip-to-shoulder = 1.0)

CHARACTER_RIGS = {
    "dancer_female": {
        "name": "Female Dancer",
        "description": "Slim dancer proportions — long legs, narrow shoulders",
        "proportions": {
            "head_radius": 0.28,
            "neck_length": 0.15,
            "shoulder_width": 0.75,
            "upper_arm": 0.55,
            "forearm": 0.48,
            "hip_width": 0.45,
            "thigh": 0.80,
            "shin": 0.78,
            "torso_width": 0.50,
            "hand_scale": 0.18,
            "foot_length": 0.25,
        },
        "style_overrides": {
            "cartoon": {
                "skin_color": [230, 210, 190],
                "outfit_color": [180, 80, 120],
                "shoe_color": [200, 60, 80],
            },
            "silhouette": {"fill_color": [255, 220, 240]},
            "neon": {"glow_color": [255, 50, 200]},
        },
    },
    "dancer_male": {
        "name": "Male Dancer",
        "description": "Athletic male dancer — broader shoulders, balanced limbs",
        "proportions": {
            "head_radius": 0.25,
            "neck_length": 0.13,
            "shoulder_width": 0.90,
            "upper_arm": 0.58,
            "forearm": 0.50,
            "hip_width": 0.50,
            "thigh": 0.75,
            "shin": 0.72,
            "torso_width": 0.55,
            "hand_scale": 0.20,
            "foot_length": 0.28,
        },
        "style_overrides": {
            "cartoon": {
                "skin_color": [210, 180, 150],
                "outfit_color": [60, 80, 140],
                "shoe_color": [40, 40, 50],
            },
        },
    },
    "chibi": {
        "name": "Chibi Character",
        "description": "Cute big-head small-body anime proportions",
        "proportions": {
            "head_radius": 0.50,
            "neck_length": 0.08,
            "shoulder_width": 0.55,
            "upper_arm": 0.35,
            "forearm": 0.30,
            "hip_width": 0.40,
            "thigh": 0.45,
            "shin": 0.40,
            "torso_width": 0.45,
            "hand_scale": 0.15,
            "foot_length": 0.20,
        },
        "style_overrides": {
            "cartoon": {
                "skin_color": [255, 230, 210],
                "outfit_color": [255, 150, 50],
                "shoe_color": [150, 80, 40],
                "outline_width": 3,
            },
        },
    },
    "robot": {
        "name": "Robot",
        "description": "Angular robotic character with mechanical joints",
        "proportions": {
            "head_radius": 0.22,
            "neck_length": 0.10,
            "shoulder_width": 0.95,
            "upper_arm": 0.55,
            "forearm": 0.55,
            "hip_width": 0.60,
            "thigh": 0.65,
            "shin": 0.65,
            "torso_width": 0.60,
            "hand_scale": 0.22,
            "foot_length": 0.30,
        },
        "style_overrides": {
            "cartoon": {
                "skin_color": [180, 190, 200],
                "outfit_color": [80, 90, 110],
                "shoe_color": [60, 65, 75],
                "outline_color": [30, 35, 40],
            },
            "neon": {"glow_color": [0, 200, 255], "core_color": [200, 220, 255]},
        },
    },
    "shadow_dancer": {
        "name": "Shadow Dancer",
        "description": "Elegant silhouette-optimized proportions",
        "proportions": {
            "head_radius": 0.24,
            "neck_length": 0.18,
            "shoulder_width": 0.70,
            "upper_arm": 0.60,
            "forearm": 0.55,
            "hip_width": 0.42,
            "thigh": 0.85,
            "shin": 0.82,
            "torso_width": 0.45,
            "hand_scale": 0.16,
            "foot_length": 0.22,
        },
        "style_overrides": {
            "silhouette": {"fill_color": [20, 20, 30], "outline_color": [255, 200, 0]},
            "neon": {"glow_color": [255, 255, 0], "face_glow": [255, 200, 0]},
        },
    },
}

# IK chain definitions for motion retargeting
IK_CHAINS = {
    "left_arm": [5, 7, 9],      # shoulder → elbow → wrist
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],   # hip → knee → ankle
    "right_leg": [12, 14, 16],
    "spine": [11, 5, 0],        # hip → shoulder → nose (approximation)
    "left_hand_thumb": [91, 92, 93, 94, 95],
    "left_hand_index": [91, 96, 97, 98, 99],
    "left_hand_middle": [91, 100, 101, 102, 103],
    "left_hand_ring": [91, 104, 105, 106, 107],
    "left_hand_pinky": [91, 108, 109, 110, 111],
    "right_hand_thumb": [112, 113, 114, 115, 116],
    "right_hand_index": [112, 117, 118, 119, 120],
}


class CharacterRig:
    """A rigged character with IK chains and proportions."""

    def __init__(self, rig_name: str = "dancer_female"):
        if rig_name not in CHARACTER_RIGS:
            raise ValueError(f"Unknown rig: {rig_name}. Available: {list(CHARACTER_RIGS.keys())}")
        self.rig_data = CHARACTER_RIGS[rig_name]
        self.name = self.rig_data["name"]
        self.proportions = self.rig_data["proportions"]
        self.ik_chains = IK_CHAINS

    def retarget_motion(
        self,
        keypoints: np.ndarray,
        source_proportions: dict | None = None,
    ) -> np.ndarray:
        """Retarget a motion sequence to this character's proportions.
        
        Uses IK to adjust limb lengths while preserving joint angles
        and motion dynamics.
        
        Args:
            keypoints: (T, K, 2) source motion.
            source_proportions: Source character proportions. Auto-detected if None.
            
        Returns:
            (T, K, 2) retargeted motion.
        """
        from ..avatar.ik_solver import IKSolver2D
        ik = IKSolver2D(max_iterations=15, tolerance=0.3)

        T, K, _ = keypoints.shape
        result = keypoints.copy()

        # Compute source reference torso length (median)
        shoulder_mid = (keypoints[:, 5] + keypoints[:, 6]) / 2
        hip_mid = (keypoints[:, 11] + keypoints[:, 12]) / 2
        torso_lengths = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
        valid_torsos = torso_lengths[torso_lengths > 1e-6]
        if len(valid_torsos) == 0:
            return result
        ref_torso = float(np.median(valid_torsos))

        if not np.isfinite(ref_torso) or ref_torso < 1e-6:
            return result

        # Build target proportions (absolute lengths based on torso)
        target_limbs = {
            "left_arm": ref_torso * (self.proportions["upper_arm"] + self.proportions["forearm"]),
            "right_arm": ref_torso * (self.proportions["upper_arm"] + self.proportions["forearm"]),
            "left_leg": ref_torso * (self.proportions["thigh"] + self.proportions["shin"]),
            "right_leg": ref_torso * (self.proportions["thigh"] + self.proportions["shin"]),
        }

        # Retarget each frame
        limb_chains = {
            "left_arm": [5, 7, 9],
            "right_arm": [6, 8, 10],
            "left_leg": [11, 13, 15],
            "right_leg": [12, 14, 16],
        }

        for t_idx in range(T):
            result[t_idx] = ik.retarget_pose(
                result[t_idx], target_limbs, limb_chains
            )

        return result

    def get_style_config(self, base_config: dict) -> dict:
        """Merge character style overrides into a render config."""
        config = base_config.copy()
        overrides = self.rig_data.get("style_overrides", {})
        styles = config.get("avatar", {}).get("styles", {})
        for style_name, style_overrides in overrides.items():
            if style_name in styles:
                styles[style_name].update(style_overrides)
        return config

    @staticmethod
    def list_characters() -> list[dict]:
        """List all available character rigs."""
        return [
            {"id": k, "name": v["name"], "description": v["description"]}
            for k, v in CHARACTER_RIGS.items()
        ]
