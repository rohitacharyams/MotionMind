"""
Motion style transfer — apply stylistic qualities of one motion to another.

Transfers motion statistics (velocity distributions, joint angle ranges,
movement amplitude) from a style reference to a content motion.
"""

import numpy as np
from scipy.signal import savgol_filter


class MotionStyleTransfer:
    """Transfer motion style characteristics between sequences."""

    def __init__(self, config: dict):
        cfg = config.get("choreography", {})
        self.style_weight = cfg.get("style_weight", 0.5)

    def transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        weight: float | None = None,
    ) -> np.ndarray:
        """Apply style motion characteristics to content motion.
        
        Uses Adaptive Instance Normalization (AdaIN) on motion statistics.
        
        Args:
            content: (T_c, K, 2) content motion (structure to preserve).
            style: (T_s, K, 2) style motion (characteristics to transfer).
            weight: Blend weight (0=no style, 1=full style). Uses config default if None.
            
        Returns:
            (T_c, K, 2) stylized motion.
        """
        weight = weight if weight is not None else self.style_weight

        # Compute per-joint statistics
        T_c, K, D = content.shape

        # 1. Normalize content (remove its statistics)
        c_mean = content.mean(axis=0, keepdims=True)
        c_std = content.std(axis=0, keepdims=True) + 1e-8
        content_norm = (content - c_mean) / c_std

        # 2. Compute style statistics
        s_mean = style.mean(axis=0, keepdims=True)
        s_std = style.std(axis=0, keepdims=True) + 1e-8

        # 3. Apply style statistics (AdaIN)
        styled = content_norm * s_std + s_mean

        # 4. Blend with original (weighted)
        result = content * (1 - weight) + styled * weight

        return result.astype(np.float32)

    def transfer_velocity_profile(
        self,
        content: np.ndarray,
        style: np.ndarray,
        weight: float | None = None,
    ) -> np.ndarray:
        """Transfer velocity/dynamics from style to content while preserving poses.
        
        This modifies the timing/energy of movement without changing
        the spatial path.
        
        Args:
            content: (T_c, K, 2) content motion.
            style: (T_s, K, 2) style motion.
            weight: Blend weight.
            
        Returns:
            (T_c, K, 2) motion with style dynamics.
        """
        weight = weight if weight is not None else self.style_weight

        # Compute velocity profiles
        c_vel = np.diff(content, axis=0)
        s_vel = np.diff(style, axis=0)

        # Compute velocity magnitudes per frame
        c_speed = np.linalg.norm(c_vel, axis=-1).mean(axis=-1)  # (T_c-1,)
        s_speed = np.linalg.norm(s_vel, axis=-1).mean(axis=-1)  # (T_s-1,)

        # Normalize style speed profile to match content length
        s_speed_resampled = np.interp(
            np.linspace(0, 1, len(c_speed)),
            np.linspace(0, 1, len(s_speed)),
            s_speed
        )

        # Create speed ratio
        speed_ratio = np.ones_like(c_speed)
        mask = c_speed > 1e-6
        speed_ratio[mask] = s_speed_resampled[mask] / c_speed[mask]

        # Blend ratio with neutral (1.0)
        speed_ratio = 1.0 * (1 - weight) + speed_ratio * weight

        # Clamp extreme ratios
        speed_ratio = np.clip(speed_ratio, 0.3, 3.0)

        # Apply speed modulation
        result = content.copy()
        for t in range(len(c_speed)):
            delta = content[t + 1] - content[t]
            result[t + 1] = result[t] + delta * speed_ratio[t]

        return result.astype(np.float32)

    def compute_style_descriptor(self, motion: np.ndarray) -> dict:
        """Compute a style descriptor summarizing motion characteristics.
        
        Useful for comparing dance styles or clustering motions.
        
        Returns:
            dict with style features.
        """
        T, K, D = motion.shape
        vel = np.diff(motion, axis=0)
        accel = np.diff(vel, axis=0)

        speed = np.linalg.norm(vel, axis=-1)
        accel_mag = np.linalg.norm(accel, axis=-1)

        return {
            "mean_speed": float(speed.mean()),
            "max_speed": float(speed.max()),
            "speed_variance": float(speed.var()),
            "mean_acceleration": float(accel_mag.mean()),
            "max_acceleration": float(accel_mag.max()),
            "movement_range": float(motion.max() - motion.min()),
            "symmetry": float(self._compute_symmetry(motion)),
            "smoothness": float(self._compute_smoothness(vel)),
        }

    @staticmethod
    def _compute_symmetry(motion: np.ndarray) -> float:
        """Compute left-right symmetry of motion."""
        K = motion.shape[1]
        if K < 17:
            return 0.0

        # Compare left vs right body parts
        left_indices = [5, 7, 9, 11, 13, 15]
        right_indices = [6, 8, 10, 12, 14, 16]

        left_motion = motion[:, left_indices]
        right_motion = motion[:, right_indices]

        # Reflect right side
        right_reflected = right_motion.copy()
        right_reflected[..., 0] *= -1

        diff = np.linalg.norm(left_motion - right_reflected, axis=-1)
        return 1.0 / (1.0 + diff.mean())

    @staticmethod
    def _compute_smoothness(velocity: np.ndarray) -> float:
        """Compute motion smoothness (inverse of jerk)."""
        jerk = np.diff(velocity, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=-1)
        return 1.0 / (1.0 + jerk_mag.mean())
