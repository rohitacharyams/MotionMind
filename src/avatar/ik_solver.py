"""
2D Inverse Kinematics Solver.

Implements CCD (Cyclic Coordinate Descent) IK for adjusting limb poses,
and FABRIK for more natural-looking results. Used to refine pose data
and retarget motions to characters with different proportions.
"""

import numpy as np


class IKSolver2D:
    """2D Inverse Kinematics solver using CCD and FABRIK algorithms."""

    def __init__(self, max_iterations: int = 20, tolerance: float = 0.5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve_ccd(
        self,
        chain: np.ndarray,
        target: np.ndarray,
        fixed_root: bool = True,
    ) -> np.ndarray:
        """Solve IK using Cyclic Coordinate Descent.
        
        Args:
            chain: (N, 2) joint positions from root to end effector.
            target: (2,) target position for end effector.
            fixed_root: Whether to keep the root joint fixed.
            
        Returns:
            (N, 2) updated joint positions.
        """
        chain = chain.copy().astype(np.float64)
        N = len(chain)
        if N < 2:
            return chain

        # Compute bone lengths (preserve structure)
        bone_lengths = np.array([
            np.linalg.norm(chain[i + 1] - chain[i]) for i in range(N - 1)
        ])

        for _ in range(self.max_iterations):
            # Check convergence
            if np.linalg.norm(chain[-1] - target) < self.tolerance:
                break

            # Iterate from end effector back to root
            start_idx = 1 if fixed_root else 0
            for i in range(N - 2, start_idx - 1, -1):
                end_effector = chain[-1]
                joint = chain[i]

                # Vectors from current joint to end effector and target
                to_end = end_effector - joint
                to_target = target - joint

                len_end = np.linalg.norm(to_end)
                len_target = np.linalg.norm(to_target)

                if len_end < 1e-8 or len_target < 1e-8:
                    continue

                # Compute rotation angle
                cos_angle = np.dot(to_end, to_target) / (len_end * len_target)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Determine rotation direction (cross product sign)
                cross = to_end[0] * to_target[1] - to_end[1] * to_target[0]
                if cross < 0:
                    angle = -angle

                # Rotate all joints from i+1 onward
                rotation_matrix = self._rotation_matrix(angle)
                for j in range(i + 1, N):
                    chain[j] = joint + rotation_matrix @ (chain[j] - joint)

        return chain.astype(np.float32)

    def solve_fabrik(
        self,
        chain: np.ndarray,
        target: np.ndarray,
        fixed_root: bool = True,
    ) -> np.ndarray:
        """Solve IK using FABRIK (Forward And Backward Reaching Inverse Kinematics).
        
        More natural results than CCD, better for limb chains.
        
        Args:
            chain: (N, 2) joint positions from root to end effector.
            target: (2,) target position for end effector.
            fixed_root: Whether to keep the root joint fixed.
            
        Returns:
            (N, 2) updated joint positions.
        """
        chain = chain.copy().astype(np.float64)
        N = len(chain)
        if N < 2:
            return chain

        # Compute bone lengths
        bone_lengths = np.array([
            np.linalg.norm(chain[i + 1] - chain[i]) for i in range(N - 1)
        ])

        total_length = bone_lengths.sum()
        root = chain[0].copy()

        # Check reachability
        dist_to_target = np.linalg.norm(target - root)
        if dist_to_target > total_length:
            # Target unreachable — stretch towards it
            direction = (target - root) / dist_to_target
            for i in range(N - 1):
                chain[i + 1] = chain[i] + direction * bone_lengths[i]
            return chain.astype(np.float32)

        for _ in range(self.max_iterations):
            if np.linalg.norm(chain[-1] - target) < self.tolerance:
                break

            # Forward reaching (end -> root)
            chain[-1] = target.copy()
            for i in range(N - 2, -1, -1):
                direction = chain[i] - chain[i + 1]
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    direction /= dist
                chain[i] = chain[i + 1] + direction * bone_lengths[i]

            # Backward reaching (root -> end)
            if fixed_root:
                chain[0] = root
            for i in range(N - 1):
                direction = chain[i + 1] - chain[i]
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    direction /= dist
                chain[i + 1] = chain[i] + direction * bone_lengths[i]

        return chain.astype(np.float32)

    def retarget_pose(
        self,
        source_pose: np.ndarray,
        target_proportions: dict[str, float],
        limb_chains: dict[str, list[int]],
    ) -> np.ndarray:
        """Retarget a pose to different body proportions using IK.
        
        Args:
            source_pose: (K, 2) source keypoints.
            target_proportions: dict mapping limb names to desired lengths.
            limb_chains: dict mapping limb names to joint index chains.
            
        Returns:
            (K, 2) retargeted keypoints.
        """
        result = source_pose.copy()

        for limb_name, joint_indices in limb_chains.items():
            if limb_name not in target_proportions:
                continue

            chain = result[joint_indices]
            target_length = target_proportions[limb_name]

            # Compute current length
            current_length = sum(
                np.linalg.norm(chain[i + 1] - chain[i])
                for i in range(len(chain) - 1)
            )

            if current_length < 1e-8:
                continue

            # Scale bone lengths proportionally
            scale = target_length / current_length
            for i in range(len(chain) - 1):
                direction = chain[i + 1] - chain[i]
                chain[i + 1] = chain[i] + direction * scale

            # Apply back
            result[joint_indices] = chain

        return result

    def apply_constraints(
        self,
        chain: np.ndarray,
        min_angles: np.ndarray | None = None,
        max_angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply angular constraints to a joint chain.
        
        Args:
            chain: (N, 2) joint positions.
            min_angles: (N-2,) minimum angles at each internal joint.
            max_angles: (N-2,) maximum angles at each internal joint.
        """
        if min_angles is None or max_angles is None:
            return chain

        chain = chain.copy()
        for i in range(1, len(chain) - 1):
            v1 = chain[i - 1] - chain[i]
            v2 = chain[i + 1] - chain[i]

            angle = np.arctan2(
                v1[0] * v2[1] - v1[1] * v2[0],
                np.dot(v1, v2)
            )

            idx = i - 1
            if idx < len(min_angles):
                clamped = np.clip(angle, min_angles[idx], max_angles[idx])
                if abs(clamped - angle) > 1e-6:
                    # Rotate v2 to satisfy constraint
                    delta = clamped - angle
                    rot = self._rotation_matrix(delta)
                    chain[i + 1] = chain[i] + rot @ v2

        return chain

    @staticmethod
    def _rotation_matrix(angle: float) -> np.ndarray:
        """2D rotation matrix."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
