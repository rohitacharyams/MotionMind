"""
Temporal smoothing for motion sequences.

Applies configurable filters (Savitzky-Golay, Gaussian, moving average,
Butterworth low-pass) with multi-pass and adaptive options to reduce
jitter while preserving natural motion dynamics.
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d


class MotionSmoother:
    """Smooth motion sequences to reduce pose estimation jitter."""

    def __init__(self, config: dict):
        cfg = config.get("motion_processing", {}).get("smoothing", {})
        self.enabled = cfg.get("enabled", True)
        self.method = cfg.get("method", "savgol")
        self.window_size = cfg.get("window_size", 7)
        self.poly_order = cfg.get("poly_order", 3)
        self.passes = cfg.get("passes", 1)
        self.adaptive = cfg.get("adaptive", False)
        self.butterworth_cutoff = cfg.get("butterworth_cutoff", 6.0)
        self.fps = cfg.get("fps", 30.0)

    def smooth(self, keypoints: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
        """Smooth a motion sequence temporally.
        
        Args:
            keypoints: (T, K, 2) array.
            scores: optional (T, K) confidence weights.
            
        Returns:
            (T, K, 2) smoothed keypoints.
        """
        if not self.enabled:
            return keypoints

        T, K, D = keypoints.shape
        if T < max(self.window_size, 5):
            return keypoints

        result = keypoints.copy()

        for _ in range(self.passes):
            if self.method == "savgol":
                result = self._savgol(result)
            elif self.method == "gaussian":
                result = self._gaussian(result)
            elif self.method == "moving_average":
                result = self._moving_average(result)
            elif self.method == "butterworth":
                result = self._butterworth(result)
            elif self.method == "multi":
                # Multi-pass: butterworth for global smoothness, then savgol for edges
                result = self._butterworth(result)
                result = self._savgol(result)
            else:
                raise ValueError(f"Unknown smoothing method: {self.method}")

        # Adaptive: smooth low-confidence joints MORE aggressively
        if self.adaptive and scores is not None:
            extra_smooth = self._gaussian(result)
            # Low confidence → blend toward extra-smoothed version
            confidence = np.clip(scores, 0.1, 1.0)[..., np.newaxis]
            # High confidence → keep the already-smoothed result
            # Low confidence → use even more smoothed version
            result = confidence * result + (1 - confidence) * extra_smooth

        # NOTE: We do NOT blend back toward the raw original based on confidence.
        # The old approach (weight * original + (1-weight) * smoothed) was backwards:
        # it kept high-confidence joints UN-smoothed (jittery) and only smoothed
        # low-confidence ones. ALL joints need temporal smoothing regardless of
        # per-frame confidence.

        return result.astype(np.float32)

    def _savgol(self, kps: np.ndarray) -> np.ndarray:
        """Savitzky-Golay filter — best balance of smoothing + edge preservation."""
        T, K, D = kps.shape
        window = min(self.window_size, T)
        if window % 2 == 0:
            window -= 1
        if window < self.poly_order + 1:
            return kps
        
        result = np.empty_like(kps)
        for k in range(K):
            for d in range(D):
                result[:, k, d] = savgol_filter(
                    kps[:, k, d], window, self.poly_order
                )
        return result

    def _gaussian(self, kps: np.ndarray) -> np.ndarray:
        """Gaussian smoothing."""
        sigma = self.window_size / 4.0
        T, K, D = kps.shape
        result = np.empty_like(kps)
        for k in range(K):
            for d in range(D):
                result[:, k, d] = gaussian_filter1d(kps[:, k, d], sigma)
        return result

    def _moving_average(self, kps: np.ndarray) -> np.ndarray:
        """Simple moving average."""
        T, K, D = kps.shape
        kernel = np.ones(self.window_size) / self.window_size
        result = np.empty_like(kps)
        pad = self.window_size // 2
        for k in range(K):
            for d in range(D):
                padded = np.pad(kps[:, k, d], pad, mode='edge')
                result[:, k, d] = np.convolve(padded, kernel, mode='valid')[:T]
        return result

    def _butterworth(self, kps: np.ndarray) -> np.ndarray:
        """Butterworth low-pass filter — very smooth motion, removes high-freq jitter."""
        T, K, D = kps.shape
        nyquist = self.fps / 2.0
        cutoff = min(self.butterworth_cutoff, nyquist * 0.9)
        normal_cutoff = cutoff / nyquist
        normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)

        b, a = butter(2, normal_cutoff, btype='low')

        result = np.empty_like(kps)
        # Need at least 3*max(len(a), len(b)) samples for filtfilt
        min_len = 3 * max(len(a), len(b))
        if T < min_len:
            return kps

        for k in range(K):
            for d in range(D):
                result[:, k, d] = filtfilt(b, a, kps[:, k, d])
        return result
