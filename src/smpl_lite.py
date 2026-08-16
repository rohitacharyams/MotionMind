"""
Standalone SMPL skeleton + LBS forward kinematics WITHOUT chumpy/smplx.

Loads the SMPL .pkl by replacing chumpy `Ch` objects with plain numpy arrays.
Provides:
    SMPL_PARENTS                  : (24,) parent index per joint (-1 for root)
    SMPL_NAMES                    : standard 24 joint names
    load_smpl_skeleton(path)      : returns dict { 'rest_joints': (24,3),
                                                    'parents'    : (24,),
                                                    'names'      : list[str] }
    forward_kinematics(rest_joints, parents, pose_aa, trans=None)
        pose_aa: (24, 3) axis-angle local rotations
        trans  : (3,) global translation (or None)
        Returns: (joint_world_pos (24,3), joint_world_R (24,3,3))
"""
from __future__ import annotations
import pickle
import io
import sys
import numpy as np

# ── Standard SMPL constants ──
SMPL_NAMES = [
    'pelvis', 'L_Hip', 'R_Hip', 'Spine1',
    'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3',
    'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head',
    'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist',
    'L_Hand', 'R_Hand',
]
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
     13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────
# Robust SMPL .pkl loader (no chumpy)
# ─────────────────────────────────────────────────────────────────
class _ChumpyShim:
    """Stand-in for chumpy's `Ch` class. Just stores `.r` (the value array)."""
    def __init__(self, *args, **kwargs):
        self.r = np.array([])
    def __setstate__(self, state):
        # chumpy `Ch` stores its array under '.x' or as bare ndarray; capture
        # the numpy contents however they appear.
        if isinstance(state, dict):
            for key in ('x', 'r', '_data'):
                if key in state and isinstance(state[key], np.ndarray):
                    self.r = state[key]
                    return
            # Fall back: first ndarray we find
            for v in state.values():
                if isinstance(v, np.ndarray):
                    self.r = v
                    return
        elif isinstance(state, np.ndarray):
            self.r = state

    def __array__(self):
        return np.asarray(self.r)


class _ChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return _ChumpyShim
        return super().find_class(module, name)


def _to_array(x):
    """Coerce SMPL .pkl values (chumpy/scipy.sparse/ndarray) → ndarray."""
    if isinstance(x, _ChumpyShim):
        return np.asarray(x.r)
    if hasattr(x, 'todense'):                # scipy.sparse
        return np.asarray(x.todense())
    return np.asarray(x)


def load_smpl_skeleton(pkl_path: str) -> dict:
    """Load SMPL .pkl and return a small dict of skeleton-only data.

    Returns:
        {
          'rest_joints' : (24, 3) numpy array of world-space rest positions,
          'parents'     : (24,) int parent indices,
          'names'       : SMPL_NAMES,
          'v_template'  : (V, 3) optional mean-shape vertices,
        }
    """
    with open(pkl_path, 'rb') as f:
        data = _ChumpyUnpickler(f, encoding='latin1').load()

    v_template  = _to_array(data['v_template'])         # (V, 3)
    J_regressor = _to_array(data['J_regressor'])        # (24, V) sparse → dense
    kintree     = _to_array(data['kintree_table'])      # (2, 24)

    rest_joints = J_regressor @ v_template              # (24, 3)
    # kintree_table[0] is parents, [1] is children-id (we use 0)
    parents = kintree[0].astype(np.int64).copy()
    parents[parents == 4294967295] = -1                 # uint32 -1 → -1
    parents[0] = -1

    return {
        'rest_joints': np.asarray(rest_joints, dtype=np.float64),
        'parents'    : parents,
        'names'      : SMPL_NAMES,
        'v_template' : v_template,
    }


# ─────────────────────────────────────────────────────────────────
# Rotation utilities
# ─────────────────────────────────────────────────────────────────
def axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle → (..., 3, 3) rotation matrix (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = aa / safe                                           # (..., 3) unit axis
    cos = np.cos(theta)[..., 0]                             # (...,)
    sin = np.sin(theta)[..., 0]
    one_minus_cos = 1.0 - cos
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = np.zeros_like(kx)
    K = np.stack([
        np.stack([zeros, -kz, ky], axis=-1),
        np.stack([kz, zeros, -kx], axis=-1),
        np.stack([-ky, kx, zeros], axis=-1),
    ], axis=-2)                                             # (..., 3, 3)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    R = I + sin[..., None, None] * K + one_minus_cos[..., None, None] * (K @ K)
    # When θ → 0, K ≈ 0 so R ≈ I, which is correct.
    return R


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation → (..., 4) [x,y,z,w] quaternion."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    out = np.empty(R.shape[:-2] + (4,), dtype=np.float64)
    case0 = tr > 0
    # case 0
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2.0
    out[..., 3] = 0.25 * s
    out[..., 0] = (m21 - m12) / s
    out[..., 1] = (m02 - m20) / s
    out[..., 2] = (m10 - m01) / s
    # other cases (no broadcasting fancy — handle in scalar fallback if needed)
    if not np.all(case0):
        flat_R = R.reshape(-1, 3, 3)
        flat_q = out.reshape(-1, 4)
        for i in range(flat_R.shape[0]):
            r = flat_R[i]
            tri = r[0, 0] + r[1, 1] + r[2, 2]
            if tri > 0:
                continue                                  # already handled
            if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                w = (r[2, 1] - r[1, 2]) / s
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
            elif r[1, 1] > r[2, 2]:
                s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                w = (r[0, 2] - r[2, 0]) / s
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                w = (r[1, 0] - r[0, 1]) / s
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
            flat_q[i] = (x, y, z, w)
        out = flat_q.reshape(R.shape[:-2] + (4,))
    # Normalize
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return out / n


# ─────────────────────────────────────────────────────────────────
# SMPL forward kinematics
# ─────────────────────────────────────────────────────────────────
def forward_kinematics(rest_joints: np.ndarray,
                       parents:     np.ndarray,
                       pose_aa:     np.ndarray,
                       trans:       np.ndarray | None = None):
    """Compute world transforms for an SMPL skeleton given LOCAL axis-angle.

    Args:
        rest_joints : (J, 3) world-space joint positions in T-pose.
        parents     : (J,) parent indices (-1 for root).
        pose_aa     : (J, 3) per-joint axis-angle (LOCAL rotation).
                      Convention matches SMPL: pose_aa[0] is global root rot.
        trans       : (3,) optional global translation added to root.

    Returns:
        joint_pos : (J, 3) world position per joint
        joint_R   : (J, 3, 3) world rotation per joint
        local_R   : (J, 3, 3) the input local rotations as matrices (for export)
    """
    J = rest_joints.shape[0]
    local_R = axis_angle_to_R(pose_aa)                  # (J, 3, 3)
    # Bone offsets relative to parent (rest)
    bone_offset = rest_joints.copy()
    for i in range(1, J):
        bone_offset[i] = rest_joints[i] - rest_joints[parents[i]]

    joint_R   = np.zeros((J, 3, 3))
    joint_pos = np.zeros((J, 3))
    for i in range(J):
        if parents[i] < 0:
            joint_R[i]   = local_R[i]
            joint_pos[i] = bone_offset[i]
        else:
            p = parents[i]
            joint_R[i]   = joint_R[p] @ local_R[i]
            joint_pos[i] = joint_pos[p] + joint_R[p] @ bone_offset[i]
    if trans is not None:
        joint_pos = joint_pos + np.asarray(trans).reshape(1, 3)
    return joint_pos, joint_R, local_R


if __name__ == '__main__':
    # Smoke test
    sk = load_smpl_skeleton('data/models/smpl_raw/smpl/models/'
                             'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    print('parents :', sk['parents'].tolist())
    print('rest joints (first 24):')
    for i, n in enumerate(sk['names']):
        x, y, z = sk['rest_joints'][i]
        print(f'  {i:2d}  {n:12s}  ({x:+.3f}, {y:+.3f}, {z:+.3f})')

    # Identity FK should reproduce rest positions
    pose = np.zeros((24, 3))
    pos, R, _ = forward_kinematics(sk['rest_joints'], sk['parents'], pose)
    err = np.max(np.abs(pos - sk['rest_joints']))
    print(f'identity FK max error: {err:.2e}')

    # Rotate the pelvis 90° around Y: head should swing
    pose[0] = np.array([0, np.pi / 2, 0])
    pos, R, _ = forward_kinematics(sk['rest_joints'], sk['parents'], pose)
    print(f'after pelvis Y+90°, head pos: {pos[15].round(3)}')
