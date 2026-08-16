"""
Motion normalization — hip-centering, scale normalization, and coordinate transforms.

All operations work on arrays of shape (T, K, 2) where T=frames, K=keypoints.
"""

import numpy as np
from ..pose_extraction.utils import compute_hip_center, compute_torso_size


class MotionNormalizer:
    """Normalize motion sequences for consistency and storage."""

    def __init__(self, config: dict):
        cfg = config.get("motion_processing", {})
        self.normalize_to_hip = cfg.get("normalize_to_hip", True)
        self.scale_to_unit = cfg.get("scale_to_unit", True)
        self.min_confidence = cfg.get("min_confidence", 0.3)
        self.interpolate_low = cfg.get("interpolate_low_confidence", True)

    def normalize(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> dict:
        """Normalize a motion sequence.
        
        Args:
            keypoints: (T, K, 2) array of keypoint coordinates.
            scores: (T, K) confidence scores, optional.
            
        Returns:
            dict with:
                'keypoints': normalized (T, K, 2)
                'hip_positions': original hip centers (T, 2)
                'scale_factors': per-frame scale factors (T,)
                'scores': cleaned scores (T, K)
        """
        kps = keypoints.copy().astype(np.float64)
        T, K, _ = kps.shape

        # Filter low-confidence keypoints
        if scores is not None and self.interpolate_low:
            kps = self._interpolate_low_confidence(kps, scores)

        # Compute hip centers
        hip_positions = compute_hip_center(kps)  # (T, 2)

        # Center on hip
        if self.normalize_to_hip:
            kps = kps - hip_positions[:, np.newaxis, :]

        # Scale normalization
        scale_factors = np.ones(T, dtype=np.float64)
        if self.scale_to_unit:
            torso_sizes = compute_torso_size(keypoints)  # use original for stable reference
            # Use median torso size to avoid outlier scaling
            median_torso = np.median(torso_sizes[torso_sizes > 0]) if np.any(torso_sizes > 0) else 1.0
            scale_factors = np.where(torso_sizes > 0, torso_sizes, median_torso)
            kps = kps / scale_factors[:, np.newaxis, np.newaxis]

        return {
            "keypoints": kps.astype(np.float32),
            "hip_positions": hip_positions.astype(np.float32),
            "scale_factors": scale_factors.astype(np.float32),
            "scores": scores,
        }

    def denormalize(
        self,
        keypoints: np.ndarray,
        hip_positions: np.ndarray,
        scale_factors: np.ndarray,
    ) -> np.ndarray:
        """Reverse normalization to get pixel coordinates back.
        
        Args:
            keypoints: (T, K, 2) normalized keypoints.
            hip_positions: (T, 2) original hip centers.
            scale_factors: (T,) scale factors used.
            
        Returns:
            (T, K, 2) denormalized keypoints.
        """
        kps = keypoints.copy().astype(np.float64)
        kps = kps * scale_factors[:, np.newaxis, np.newaxis]
        kps = kps + hip_positions[:, np.newaxis, :]
        return kps.astype(np.float32)

    def _interpolate_low_confidence(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate keypoints with low confidence scores."""
        T, K, D = keypoints.shape
        result = keypoints.copy()

        for k in range(K):
            mask = scores[:, k] >= self.min_confidence
            if mask.sum() < 2:
                continue
            good_indices = np.where(mask)[0]
            bad_indices = np.where(~mask)[0]
            if len(bad_indices) == 0:
                continue
            for d in range(D):
                result[bad_indices, k, d] = np.interp(
                    bad_indices, good_indices, keypoints[good_indices, k, d]
                )
        return result
