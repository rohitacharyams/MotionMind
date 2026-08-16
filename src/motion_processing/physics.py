"""
Physics-aware motion post-processing.

Applies physical constraints (gravity, ground plane, joint limits,
velocity clamping) to prevent impossible poses and aerial drift.
"""

import numpy as np
from scipy.signal import savgol_filter


class PhysicsConstraints:
    """Apply physics-based constraints to motion sequences."""

    # Anatomical bone-length ratios (relative to torso length = hip-to-shoulder)
    # These are approximate human proportions
    BONE_PAIRS = {
        "left_upper_arm": (5, 7),
        "left_forearm": (7, 9),
        "right_upper_arm": (6, 8),
        "right_forearm": (8, 10),
        "left_thigh": (11, 13),
        "left_shin": (13, 15),
        "right_thigh": (12, 14),
        "right_shin": (14, 16),
    }

    BONE_LENGTH_RATIOS = {
        "left_upper_arm": 0.55,
        "left_forearm": 0.50,
        "right_upper_arm": 0.55,
        "right_forearm": 0.50,
        "left_thigh": 0.75,
        "left_shin": 0.72,
        "right_thigh": 0.75,
        "right_shin": 0.72,
    }

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("physics", {})
        self.gravity_enabled = cfg.get("gravity", True)
        self.ground_plane = cfg.get("ground_plane", True)
        self.bone_constraints = cfg.get("bone_constraints", True)
        self.velocity_clamp = cfg.get("velocity_clamp", True)
        self.max_velocity_factor = cfg.get("max_velocity_factor", 0.25)
        self.ground_margin = cfg.get("ground_margin", 0.02)

    def apply(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray = None,
        fps: float = 30.0,
        frame_size: tuple = None,
    ) -> np.ndarray:
        """Apply all physics constraints to a motion sequence.

        Args:
            keypoints: (T, K, 2) keypoints in pixel coordinates.
            scores: (T, K) optional confidence scores.
            fps: Video frame rate.
            frame_size: (W, H) frame dimensions for ground plane.

        Returns:
            (T, K, 2) physics-corrected keypoints.
        """
        kps = keypoints.copy().astype(np.float64)
        T, K, _ = kps.shape

        if T < 3:
            return kps.astype(np.float32)

        # 1. Clamp velocity to prevent teleportation/flying
        if self.velocity_clamp:
            kps = self._clamp_velocity(kps, scores, fps, frame_size)

        # 2. Enforce bone length consistency
        if self.bone_constraints and K >= 17:
            kps = self._enforce_bone_lengths(kps, scores)

        # 3. Apply gravity — pull floating poses down
        if self.gravity_enabled and K >= 17:
            kps = self._apply_gravity(kps, scores, fps, frame_size)

        # 4. Enforce ground plane — feet can't go below ground
        if self.ground_plane and frame_size is not None and K >= 17:
            kps = self._enforce_ground_plane(kps, frame_size)

        return kps.astype(np.float32)

    def _clamp_velocity(
        self, kps: np.ndarray, scores: np.ndarray, fps: float, frame_size: tuple
    ) -> np.ndarray:
        """Clamp per-joint velocity to prevent sudden jumps/teleportation."""
        T, K, _ = kps.shape

        # Max displacement per frame: fraction of frame diagonal
        if frame_size is not None:
            diag = np.sqrt(frame_size[0] ** 2 + frame_size[1] ** 2)
        else:
            # Estimate from data range
            valid = kps[kps.sum(axis=-1) != 0]
            diag = np.sqrt((valid.max(axis=0) - valid.min(axis=0)).sum() ** 2) if len(valid) > 0 else 1000

        max_disp = diag * self.max_velocity_factor

        for t in range(1, T):
            for k in range(K):
                # Skip zero/undetected joints
                if kps[t, k].sum() == 0 or kps[t - 1, k].sum() == 0:
                    continue
                # Skip low-confidence joints
                if scores is not None and scores[t, k] < 0.2:
                    continue

                delta = kps[t, k] - kps[t - 1, k]
                dist = np.linalg.norm(delta)

                if dist > max_disp:
                    # Blend toward previous position (soft clamp, not hard)
                    # Use exponential decay so fast motion is still allowed
                    ratio = max_disp / dist
                    blend = 0.3 + 0.7 * ratio  # never fully snap back
                    kps[t, k] = kps[t - 1, k] + delta * blend

        return kps

    def _enforce_bone_lengths(
        self, kps: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Enforce consistent bone lengths across frames to prevent stretching."""
        T, K, _ = kps.shape

        # Compute reference torso length (median across valid frames)
        shoulder_mid = (kps[:, 5] + kps[:, 6]) / 2
        hip_mid = (kps[:, 11] + kps[:, 12]) / 2
        torso_lengths = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)

        valid_torso = torso_lengths[torso_lengths > 1]
        if len(valid_torso) == 0:
            return kps

        ref_torso = np.median(valid_torso)

        # For each bone, compute temporally-smoothed expected length
        # This prevents frame-to-frame bone length jitter
        for bone_name, (j1, j2) in self.BONE_PAIRS.items():
            if j1 >= K or j2 >= K:
                continue

            ratio = self.BONE_LENGTH_RATIOS[bone_name]
            expected_len = ref_torso * ratio

            # Compute actual bone lengths across all frames
            bone_vecs = kps[:, j2] - kps[:, j1]
            bone_lens = np.linalg.norm(bone_vecs, axis=-1)

            # Use a running median of actual lengths (temporal coherence)
            valid_lens = bone_lens[bone_lens > 1e-3]
            if len(valid_lens) > 0:
                median_actual = np.median(valid_lens)
                # Blend expected (anatomical) with actual (observed)
                expected_len = 0.4 * expected_len + 0.6 * median_actual

            # Wider tolerance band to avoid over-correction
            min_len = expected_len * 0.5
            max_len = expected_len * 1.5

            for t in range(T):
                if kps[t, j1].sum() == 0 or kps[t, j2].sum() == 0:
                    continue

                actual = kps[t, j2] - kps[t, j1]
                actual_len = bone_lens[t]

                if actual_len < 1e-3:
                    continue

                if actual_len < min_len or actual_len > max_len:
                    # Soft correction — only move 60% of the way
                    target = np.clip(actual_len, min_len, max_len)
                    direction = actual / actual_len

                    if scores is not None:
                        s1, s2 = scores[t, j1], scores[t, j2]
                        w = s1 / (s1 + s2 + 1e-8)
                    else:
                        w = 0.5

                    correction = direction * (target - actual_len) * 0.6
                    kps[t, j1] -= correction * w
                    kps[t, j2] += correction * (1 - w)

        return kps

    def _apply_gravity(
        self, kps: np.ndarray, scores: np.ndarray, fps: float, frame_size: tuple
    ) -> np.ndarray:
        """Apply gravity constraint — detect and correct floating poses.

        When the hip center suddenly moves up without feet moving down
        (indicating a tracking glitch, not a real jump), pull the pose
        back down smoothly.
        """
        T, K, _ = kps.shape

        # Compute hip center trajectory
        hip_y = (kps[:, 11, 1] + kps[:, 12, 1]) / 2  # y-coordinate

        # Compute feet positions (lowest points)
        foot_indices = [15, 16]  # left_ankle, right_ankle
        feet_y = np.max(kps[:, foot_indices, 1], axis=1)  # max y = lowest on screen

        # Find frames where hip moves up significantly but feet don't go down
        # (i.e., the whole body is "floating")
        if frame_size is not None:
            body_height = np.abs(hip_y - feet_y)
            median_height = np.median(body_height[body_height > 1])
            if median_height < 1:
                return kps

            # Compute the "expected" feet position based on smooth trajectory
            smooth_feet = savgol_filter(feet_y, min(15, T if T % 2 == 1 else T - 1), 3)

            for t in range(T):
                if feet_y[t] < 1:
                    continue
                # How much is this frame floating above expected ground?
                ground_level = frame_size[1] * (1 - self.ground_margin)
                lowest_foot = feet_y[t]

                # If the lowest foot is above 75% of frame and way above smooth trajectory
                # it's likely a tracking error
                deviation = smooth_feet[t] - feet_y[t]
                if deviation > median_height * 0.3:
                    # Pull entire pose down
                    correction = deviation * 0.6  # partial correction
                    kps[t, :, 1] += correction

        return kps

    def _enforce_ground_plane(
        self, kps: np.ndarray, frame_size: tuple
    ) -> np.ndarray:
        """Ensure feet stay on a consistent ground plane, not floating or sinking."""
        T, K, _ = kps.shape
        H = frame_size[1]

        # Find the typical ground level from foot positions
        foot_indices = [15, 16]  # ankles
        all_foot_y = []
        for t in range(T):
            for fi in foot_indices:
                if kps[t, fi].sum() != 0:
                    all_foot_y.append(kps[t, fi, 1])

        if len(all_foot_y) < T // 2:
            return kps

        # Ground level = high percentile of foot y-positions (close to bottom of frame)
        ground_y = np.percentile(all_foot_y, 90)

        # Smooth the per-frame lowest foot to prevent jitter on ground contact
        lowest_foot_y = np.zeros(T)
        for t in range(T):
            foot_ys = [kps[t, fi, 1] for fi in foot_indices if kps[t, fi].sum() != 0]
            lowest_foot_y[t] = max(foot_ys) if foot_ys else ground_y

        # Don't let feet go significantly below ground
        for t in range(T):
            max_foot_y = lowest_foot_y[t]
            if max_foot_y > ground_y + H * 0.03:
                # Push the entire pose up to keep feet at ground level
                excess = max_foot_y - ground_y
                kps[t, :, 1] -= excess * 0.8

        return kps
