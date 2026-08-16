"""
Transition engine — smooth transitions between motion clips.

Supports linear interpolation, SLERP (for rotations), and
Bezier curve interpolation for natural-looking transitions.
"""

import numpy as np


class TransitionEngine:
    """Create smooth transitions between motion segments."""

    def __init__(self, config: dict):
        cfg = config.get("choreography", {})
        self.default_method = cfg.get("transition_method", "slerp")
        self.default_frames = cfg.get("transition_frames", 15)

    def create_transition(
        self,
        clip_a: np.ndarray,
        clip_b: np.ndarray,
        n_frames: int | None = None,
        method: str | None = None,
    ) -> np.ndarray:
        """Create a transition between two motion segments.
        
        Args:
            clip_a: (T_a, K, 2) end of first clip.
            clip_b: (T_b, K, 2) start of second clip.
            n_frames: Number of transition frames.
            method: Interpolation method ('linear', 'slerp', 'bezier').
            
        Returns:
            (n_frames, K, 2) transition frames.
        """
        n_frames = n_frames or self.default_frames
        method = method or self.default_method

        # Take the last frames of A and first frames of B
        start_pose = clip_a[-1]  # (K, 2)
        end_pose = clip_b[0]     # (K, 2)

        # Also use velocity for smoother transitions
        if len(clip_a) >= 2:
            start_vel = clip_a[-1] - clip_a[-2]
        else:
            start_vel = np.zeros_like(start_pose)

        if len(clip_b) >= 2:
            end_vel = clip_b[1] - clip_b[0]
        else:
            end_vel = np.zeros_like(end_pose)

        if method == "linear":
            return self._linear_interp(start_pose, end_pose, n_frames)
        elif method == "slerp":
            return self._slerp_interp(start_pose, end_pose, n_frames)
        elif method == "bezier":
            return self._bezier_interp(
                start_pose, end_pose, start_vel, end_vel, n_frames
            )
        else:
            return self._linear_interp(start_pose, end_pose, n_frames)

    def _linear_interp(
        self, start: np.ndarray, end: np.ndarray, n: int
    ) -> np.ndarray:
        """Simple linear interpolation."""
        t = np.linspace(0, 1, n)[:, None, None]
        return (start[None] * (1 - t) + end[None] * t).astype(np.float32)

    def _slerp_interp(
        self, start: np.ndarray, end: np.ndarray, n: int
    ) -> np.ndarray:
        """Spherical linear interpolation per joint.
        
        Treats each joint position relative to hip center as a vector
        and interpolates direction and magnitude separately, giving
        more natural rotational transitions.
        """
        K, D = start.shape
        result = np.zeros((n, K, D), dtype=np.float64)

        # Hip center interpolation (linear)
        hip_start = (start[11] + start[12]) / 2 if K > 12 else start[0]
        hip_end = (end[11] + end[12]) / 2 if K > 12 else end[0]

        for frame_idx in range(n):
            t = frame_idx / max(n - 1, 1)
            # Smooth ease-in-out
            t = self._smoothstep(t)

            for k in range(K):
                v_start = start[k] - hip_start
                v_end = end[k] - hip_end
                hip_interp = hip_start * (1 - t) + hip_end * t

                len_start = np.linalg.norm(v_start)
                len_end = np.linalg.norm(v_end)

                if len_start < 1e-8 or len_end < 1e-8:
                    # Fallback to linear
                    result[frame_idx, k] = start[k] * (1 - t) + end[k] * t
                    continue

                # Interpolate magnitude
                mag = len_start * (1 - t) + len_end * t

                # Interpolate direction (slerp)
                d_start = v_start / len_start
                d_end = v_end / len_end

                cos_omega = np.clip(np.dot(d_start, d_end), -1, 1)
                omega = np.arccos(cos_omega)

                if abs(omega) < 1e-6:
                    direction = d_start
                else:
                    direction = (
                        np.sin((1 - t) * omega) * d_start +
                        np.sin(t * omega) * d_end
                    ) / np.sin(omega)

                result[frame_idx, k] = hip_interp + direction * mag

        return result.astype(np.float32)

    def _bezier_interp(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_vel: np.ndarray,
        end_vel: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Cubic Bezier interpolation using velocities as control points.
        
        Gives the smoothest transitions with momentum continuity.
        """
        # Control points for cubic Bezier
        p0 = start
        p1 = start + start_vel * (n / 3)
        p2 = end - end_vel * (n / 3)
        p3 = end

        result = np.zeros((n, *start.shape), dtype=np.float64)
        for i in range(n):
            t = i / max(n - 1, 1)
            t = self._smoothstep(t)

            # De Casteljau's algorithm
            result[i] = (
                (1 - t) ** 3 * p0 +
                3 * (1 - t) ** 2 * t * p1 +
                3 * (1 - t) * t ** 2 * p2 +
                t ** 3 * p3
            )

        return result.astype(np.float32)

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Hermite smoothstep for ease-in-out."""
        return t * t * (3 - 2 * t)
