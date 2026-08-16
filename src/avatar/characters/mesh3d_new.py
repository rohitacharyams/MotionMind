"""
3D Mesh character style — render GLTF/GLB models driven by 2D keypoints.

Supports two modes:
  A) Skinned meshes (Mixamo, ReadyPlayerMe, etc.):
     Single-mesh + skeleton with vertex weights.
     Parses GLTF skin, inverse-bind-matrices, and joint weights.
     Computes Linear Blend Skinning (LBS) per frame.
  B) Segmented body-part meshes (rigged_human.glb):
     Multiple named geometry parts, each rigidly transformed.

Both modes use analytical IK to lift 2D COCO keypoints → 3D,
then compute bone rotations to drive the character.

Requires: pip install trimesh pyrender pyglet PyOpenGL>=3.1.7
"""

import os
import json
import struct
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_HAS_TRIMESH = False
_HAS_PYRENDER = False

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    pass

try:
    import pyrender
    _HAS_PYRENDER = True
except ImportError:
    pass

# ── Character search paths ──
DEFAULT_CHAR_DIRS = ["data/characters", "data/models", "assets/characters"]

# ── T-pose reference skeleton (Y-up) ──
_TPOSE = {
    'Hips':        np.array([0, 0.95, 0]),
    'Spine':       np.array([0, 1.05, 0]),
    'Spine2':      np.array([0, 1.25, 0]),
    'Neck':        np.array([0, 1.4, 0]),
    'Head':        np.array([0, 1.55, 0]),
    'LeftArm':     np.array([-0.22, 1.38, 0]),
    'LeftForeArm': np.array([-0.48, 1.38, 0]),
    'LeftHand':    np.array([-0.72, 1.38, 0]),
    'RightArm':    np.array([0.22, 1.38, 0]),
    'RightForeArm':np.array([0.48, 1.38, 0]),
    'RightHand':   np.array([0.72, 1.38, 0]),
    'LeftUpLeg':   np.array([-0.1, 0.92, 0]),
    'LeftLeg':     np.array([-0.1, 0.5, 0]),
    'LeftFoot':    np.array([-0.1, 0.08, 0]),
    'RightUpLeg':  np.array([0.1, 0.92, 0]),
    'RightLeg':    np.array([0.1, 0.5, 0]),
    'RightFoot':   np.array([0.1, 0.08, 0]),
}

_COCO_TO_JOINT = {
    0: 'Head', 5: 'LeftArm', 6: 'RightArm',
    7: 'LeftForeArm', 8: 'RightForeArm',
    9: 'LeftHand', 10: 'RightHand',
    11: 'LeftUpLeg', 12: 'RightUpLeg',
    13: 'LeftLeg', 14: 'RightLeg',
    15: 'LeftFoot', 16: 'RightFoot',
}

# Segmented model bone defs: bone → (parent_joint, child_joint, [geom_names])
_BONE_DEFS = {
    'spine':           ('Hips', 'Neck', ['torso_upper', 'torso_mid', 'torso_lower']),
    'neck':            ('Neck', 'Head', ['neck']),
    'head':            ('Head', 'Head', ['head', 'hair']),
    'left_upper_arm':  ('LeftArm', 'LeftForeArm', ['left_upper_arm']),
    'left_forearm':    ('LeftForeArm', 'LeftHand', ['left_forearm']),
    'left_hand':       ('LeftHand', 'LeftHand', ['left_hand']),
    'right_upper_arm': ('RightArm', 'RightForeArm', ['right_upper_arm']),
    'right_forearm':   ('RightForeArm', 'RightHand', ['right_forearm']),
    'right_hand':      ('RightHand', 'RightHand', ['right_hand']),
    'left_upper_leg':  ('LeftUpLeg', 'LeftLeg', ['left_upper_leg']),
    'left_lower_leg':  ('LeftLeg', 'LeftFoot', ['left_lower_leg']),
    'left_foot':       ('LeftFoot', 'LeftFoot', ['left_foot']),
    'right_upper_leg': ('RightUpLeg', 'RightLeg', ['right_upper_leg']),
    'right_lower_leg': ('RightLeg', 'RightFoot', ['right_lower_leg']),
    'right_foot':      ('RightFoot', 'RightFoot', ['right_foot']),
}

# Fuzzy keyword mapping: GLTF joint name keywords → our standard joint names
_JOINT_KEYWORDS = {
    'Head':        ['head', 'neck_2', 'neck_joint_2'],
    'Neck':        ['neck', 'neck_1', 'neck_joint_1', 'torso_joint_3'],
    'Spine2':      ['spine2', 'spine02', 'spine_02', 'torso_joint_2', 'chest'],
    'Spine':       ['spine1', 'spine01', 'spine_01', 'torso_joint_1', 'spine'],
    'Hips':        ['hip', 'hips', 'pelvis', 'root'],
    'LeftArm':     ['leftarm', 'leftshoulder', 'left_arm', 'leftupperarm', 'l_arm',
                    'arm_joint_l__4', 'arm_joint_l_4', 'lshoulder'],
    'LeftForeArm': ['leftforearm', 'left_forearm', 'leftlowerarm', 'l_forearm',
                    'arm_joint_l__3', 'arm_joint_l_3', 'lforearm', 'lelbow'],
    'LeftHand':    ['lefthand', 'left_hand', 'l_hand', 'arm_joint_l__2',
                    'arm_joint_l_2', 'lhand', 'lwrist'],
    'RightArm':    ['rightarm', 'rightshoulder', 'right_arm', 'rightupperarm',
                    'r_arm', 'arm_joint_r', 'rshoulder',
                    'arm_joint_r__1'],
    'RightForeArm':['rightforearm', 'right_forearm', 'rightlowerarm', 'r_forearm',
                    'arm_joint_r__2', 'arm_joint_r_2', 'rforearm', 'relbow'],
    'RightHand':   ['righthand', 'right_hand', 'r_hand', 'arm_joint_r__3',
                    'arm_joint_r_3', 'rhand', 'rwrist'],
    'LeftUpLeg':   ['leftupleg', 'left_upper_leg', 'leftthigh', 'l_upleg',
                    'leg_joint_l_1', 'leg_l_1', 'lthigh'],
    'LeftLeg':     ['leftleg', 'left_leg', 'leftlowerleg', 'l_leg',
                    'leg_joint_l_2', 'leg_l_2', 'lknee', 'leftknee', 'leftshin'],
    'LeftFoot':    ['leftfoot', 'left_foot', 'l_foot',
                    'leg_joint_l_3', 'leg_l_3', 'lfoot', 'leftankle'],
    'RightUpLeg':  ['rightupleg', 'right_upper_leg', 'rightthigh', 'r_upleg',
                    'leg_joint_r_1', 'leg_r_1', 'rthigh'],
    'RightLeg':    ['rightleg', 'right_leg', 'rightlowerleg', 'r_leg',
                    'leg_joint_r_2', 'leg_r_2', 'rknee', 'rightknee', 'rightshin'],
    'RightFoot':   ['rightfoot', 'right_foot', 'r_foot',
                    'leg_joint_r_3', 'leg_r_3', 'rfoot', 'rightankle'],
}


def _match_joint_name(gltf_name):
    """Match a GLTF node name to our standard joint name."""
    if not gltf_name:
        return None
    clean = gltf_name.lower().replace(' ', '_').replace('-', '_')
    # Remove common prefixes
    for prefix in ['skeleton_', 'mixamorig:', 'mixamorig_', 'bip01_', 'bip_']:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # Try exact match first, then substring
    for std_name, keywords in _JOINT_KEYWORDS.items():
        for kw in keywords:
            if clean == kw:
                return std_name

    for std_name, keywords in _JOINT_KEYWORDS.items():
        for kw in keywords:
            if clean.endswith('_' + kw) or clean.startswith(kw + '_'):
                return std_name
            if len(kw) > 5 and kw in clean:
                return std_name
    return None


# ── IK utilities ──

def _rotation_between(v1, v2):
    """3x3 rotation matrix rotating unit vector v1 to v2 (Rodrigues)."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    cross = np.cross(v1, v2)
    dot = float(np.dot(v1, v2))
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    skew = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + dot)


def _lift_2d_to_3d(kps_2d, canvas_w, canvas_h):
    """Lift 2D COCO keypoints to 3D using bone-length constraints."""
    kps = kps_2d[:17].copy()
    hip_center = (kps[11] + kps[12]) / 2
    shoulder_center = (kps[5] + kps[6]) / 2
    torso_px = np.linalg.norm(shoulder_center - hip_center)
    if torso_px < 5:
        return None
    scale = 0.46 / torso_px
    pts = (kps - hip_center) * scale
    pts[:, 1] = -pts[:, 1]
    pts[:, 0] += _TPOSE['Hips'][0]
    pts[:, 1] += _TPOSE['Hips'][1]

    pts3d = np.zeros((17, 3))
    pts3d[:, :2] = pts

    ref_lengths = {}
    for pair, jpair in [
        ((5,7),('LeftArm','LeftForeArm')),((7,9),('LeftForeArm','LeftHand')),
        ((6,8),('RightArm','RightForeArm')),((8,10),('RightForeArm','RightHand')),
        ((11,13),('LeftUpLeg','LeftLeg')),((13,15),('LeftLeg','LeftFoot')),
        ((12,14),('RightUpLeg','RightLeg')),((14,16),('RightLeg','RightFoot')),
    ]:
        ref_lengths[pair] = np.linalg.norm(_TPOSE[jpair[1]] - _TPOSE[jpair[0]])

    def _solve_z(pi, ci, ref_len, pz=0.0):
        d2 = (pts3d[ci,0]-pts3d[pi,0])**2 + (pts3d[ci,1]-pts3d[pi,1])**2
        l2 = ref_len**2
        if d2 >= l2:
            return pz
        return pz + np.sqrt(l2 - d2) * 0.3

    for i in [0,5,6,11,12]:
        pts3d[i, 2] = 0
    pts3d[7,2] = _solve_z(5,7,ref_lengths[(5,7)])
    pts3d[9,2] = _solve_z(7,9,ref_lengths[(7,9)], pts3d[7,2])
    pts3d[8,2] = _solve_z(6,8,ref_lengths[(6,8)])
    pts3d[10,2]= _solve_z(8,10,ref_lengths[(8,10)], pts3d[8,2])
    pts3d[13,2]= _solve_z(11,13,ref_lengths[(11,13)])
    pts3d[15,2]= _solve_z(13,15,ref_lengths[(13,15)], pts3d[13,2])
    pts3d[14,2]= _solve_z(12,14,ref_lengths[(12,14)])
    pts3d[16,2]= _solve_z(14,16,ref_lengths[(14,16)], pts3d[14,2])
    return pts3d


def _compute_posed_joints(pts3d):
    """Map 3D keypoints → standard joint positions."""
    posed = {jn: pts3d[ci].copy() for ci, jn in _COCO_TO_JOINT.items()}
    hip = (pts3d[11] + pts3d[12]) / 2
    shoulder = (pts3d[5] + pts3d[6]) / 2
    posed['Hips'] = hip
    posed['Spine'] = hip + (shoulder - hip) * 0.22
    posed['Spine2'] = hip + (shoulder - hip) * 0.67
    posed['Neck'] = shoulder
    return posed


def _bone_transform(tp_parent, tp_child, p_parent, p_child):
    """4x4 rigid transform: T-pose bone → posed bone."""
    td = tp_child - tp_parent
    tl = np.linalg.norm(td)
    td = td / tl if tl > 1e-6 else np.array([0, 1, 0])
    pd = p_child - p_parent
    pl = np.linalg.norm(pd)
    pd = pd / pl if pl > 1e-6 else td
    R = _rotation_between(td, pd)
    tm = (tp_parent + tp_child) / 2
    pm = (p_parent + p_child) / 2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pm - R @ tm
    return T


# ── GLTF binary parser for skinning data ──

class _GLTFSkin:
    """Parse skinning data from a GLB file: joints, weights, inverse-bind-matrices."""

    def __init__(self, glb_path):
        self.joints = []          # list of joint node indices
        self.joint_names = []     # list of joint node names
        self.inv_bind_mats = None # (N_joints, 4, 4) inverse bind matrices
        self.joint_children = {}  # node_idx → [child_indices]
        self.joint_local_transforms = {}  # node_idx → 4x4 local transform
        self.root_joint = None
        self.joint_world_transforms = {}  # node_idx → 4x4 world transform (T-pose)
        self._joint_to_std = {}   # GLTF joint index → standard joint name
        self._std_to_gltf_idx = {}  # standard name → index in joints list

        self._parse(glb_path)

    def _parse(self, path):
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                return
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            # JSON chunk
            chunk_len = struct.unpack('<I', f.read(4))[0]
            chunk_type = f.read(4)
            self._gltf = json.loads(f.read(chunk_len))
            # Binary chunk
            if f.tell() < length:
                bin_len = struct.unpack('<I', f.read(4))[0]
                bin_type = f.read(4)
                self._bin = f.read(bin_len)
            else:
                self._bin = b''

        gltf = self._gltf
        if 'skins' not in gltf or len(gltf['skins']) == 0:
            return

        skin = gltf['skins'][0]
        self.joints = skin.get('joints', [])
        self.root_joint = self.joints[0] if self.joints else None

        # Joint names
        nodes = gltf['nodes']
        for ji in self.joints:
            self.joint_names.append(nodes[ji].get('name', f'joint_{ji}'))

        # Parse joint hierarchy
        for ni, node in enumerate(nodes):
            self.joint_children[ni] = node.get('children', [])
            T = np.eye(4)
            if 'matrix' in node:
                T = np.array(node['matrix']).reshape(4, 4).T  # column-major → row-major
            else:
                if 'translation' in node:
                    T[:3, 3] = node['translation']
                if 'rotation' in node:
                    q = node['rotation']  # [x,y,z,w]
                    T[:3, :3] = self._quat_to_mat(q)
                if 'scale' in node:
                    s = node['scale']
                    T[:3, 0] *= s[0]; T[:3, 1] *= s[1]; T[:3, 2] *= s[2]
            self.joint_local_transforms[ni] = T

        # Compute world transforms via DFS
        self._compute_world_transforms()

        # Inverse bind matrices
        ibm_accessor = skin.get('inverseBindMatrices')
        if ibm_accessor is not None:
            self.inv_bind_mats = self._read_accessor(ibm_accessor)
            n = len(self.joints)
            self.inv_bind_mats = self.inv_bind_mats.reshape(n, 4, 4)
            # GLTF stores column-major, transpose each
            for i in range(n):
                self.inv_bind_mats[i] = self.inv_bind_mats[i].T

        # Map joint names to standard skeleton
        for idx, name in enumerate(self.joint_names):
            std = _match_joint_name(name)
            if std:
                self._joint_to_std[idx] = std
                self._std_to_gltf_idx[std] = idx

        logger.info("Parsed skin: %d joints, mapped %d to standard skeleton",
                     len(self.joints), len(self._std_to_gltf_idx))

    def _compute_world_transforms(self):
        """DFS to compute world-space transforms for all nodes."""
        visited = set()
        def dfs(node_idx, parent_world):
            if node_idx in visited:
                return
            visited.add(node_idx)
            local = self.joint_local_transforms.get(node_idx, np.eye(4))
            world = parent_world @ local
            self.joint_world_transforms[node_idx] = world
            for child in self.joint_children.get(node_idx, []):
                dfs(child, world)

        # Find scene root(s) and traverse
        nodes = self._gltf.get('nodes', [])
        all_children = set()
        for n in nodes:
            for c in n.get('children', []):
                all_children.add(c)
        roots = [i for i in range(len(nodes)) if i not in all_children]
        for r in roots:
            dfs(r, np.eye(4))

    @staticmethod
    def _quat_to_mat(q):
        """Convert [x,y,z,w] quaternion to 3x3 rotation matrix."""
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])

    def _read_accessor(self, idx):
        """Read a GLTF accessor from the binary buffer."""
        gltf = self._gltf
        accessor = gltf['accessors'][idx]
        bv = gltf['bufferViews'][accessor['bufferView']]
        offset = bv.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        count = accessor['count']
        comp_type = accessor['componentType']
        acc_type = accessor['type']

        type_sizes = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
        n_components = type_sizes.get(acc_type, 1)
        total = count * n_components

        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = dtype_map.get(comp_type, np.float32)
        elem_size = np.dtype(dtype).itemsize
        data = np.frombuffer(self._bin, dtype=dtype, count=total, offset=offset)
        return data.astype(np.float32) if dtype != np.float32 else data.copy()

    def get_tpose_joint_positions(self):
        """Extract T-pose joint world positions from the skeleton."""
        positions = {}
        for idx, ji in enumerate(self.joints):
            if ji in self.joint_world_transforms:
                pos = self.joint_world_transforms[ji][:3, 3]
                std_name = self._joint_to_std.get(idx)
                if std_name:
                    positions[std_name] = pos.copy()
        return positions

    def compute_skinning_matrices(self, posed_joints):
        """Compute per-joint skinning matrices for LBS.

        For each joint, computes: M_j = posed_world_j @ inverse_bind_j

        Returns array of shape (n_joints, 4, 4).
        """
        n = len(self.joints)
        skinning_mats = np.zeros((n, 4, 4))

        # Get T-pose world positions for each joint
        tpose_positions = self.get_tpose_joint_positions()

        for idx in range(n):
            ji = self.joints[idx]
            std_name = self._joint_to_std.get(idx)

            # T-pose world transform
            tpose_world = self.joint_world_transforms.get(ji, np.eye(4))

            if std_name and std_name in posed_joints:
                posed_pos = posed_joints[std_name]
                tpose_pos = tpose_positions.get(std_name, tpose_world[:3, 3])

                # Compute rotation for this bone
                # Find child joint to determine bone direction
                child_std = self._find_child_std(idx)
                if child_std and child_std in posed_joints and child_std in tpose_positions:
                    tp = tpose_positions[std_name]
                    tc = tpose_positions[child_std]
                    pp = posed_joints[std_name]
                    pc = posed_joints[child_std]

                    td = tc - tp
                    tl = np.linalg.norm(td)
                    pd = pc - pp
                    pl = np.linalg.norm(pd)

                    if tl > 1e-6 and pl > 1e-6:
                        R = _rotation_between(td / tl, pd / pl)
                    else:
                        R = np.eye(3)
                else:
                    R = np.eye(3)

                # posed_world = translate(posed_pos) @ R @ translate(-tpose_pos)
                posed_world = np.eye(4)
                posed_world[:3, :3] = R
                posed_world[:3, 3] = posed_pos - R @ tpose_pos

                # But we need full world, so: apply to T-pose world
                # final = posed_world @ tpose_world... but adjusted
                # Actually: M = posed_world_transform @ inv_bind
                # Where posed_world_transform puts the joint at the new position
                new_world = np.eye(4)
                new_world[:3, :3] = R @ tpose_world[:3, :3]
                new_world[:3, 3] = posed_pos

                if self.inv_bind_mats is not None:
                    skinning_mats[idx] = new_world @ self.inv_bind_mats[idx]
                else:
                    skinning_mats[idx] = new_world @ np.linalg.inv(tpose_world)
            else:
                # Unmapped joint — identity (keep T-pose)
                if self.inv_bind_mats is not None:
                    skinning_mats[idx] = tpose_world @ self.inv_bind_mats[idx]
                else:
                    skinning_mats[idx] = np.eye(4)

        return skinning_mats

    def _find_child_std(self, joint_list_idx):
        """Find the standard name of the child joint for bone direction."""
        ji = self.joints[joint_list_idx]
        children = self.joint_children.get(ji, [])
        for child_ni in children:
            if child_ni in self.joints:
                child_idx = self.joints.index(child_ni)
                child_std = self._joint_to_std.get(child_idx)
                if child_std:
                    return child_std
        return None


def _apply_lbs(vertices, joints_4, weights_4, skinning_mats):
    """Apply Linear Blend Skinning to vertices.

    vertices: (N, 3) positions
    joints_4: (N, 4) joint indices per vertex
    weights_4: (N, 4) weights per vertex
    skinning_mats: (J, 4, 4) per-joint transform matrices

    Returns: (N, 3) transformed positions
    """
    N = len(vertices)
    # Homogeneous coordinates
    v_homo = np.ones((N, 4), dtype=np.float64)
    v_homo[:, :3] = vertices

    result = np.zeros((N, 3), dtype=np.float64)

    j_indices = joints_4.astype(int)
    n_joints = len(skinning_mats)

    for influence in range(4):
        ji = j_indices[:, influence]
        wi = weights_4[:, influence]

        mask = wi > 0.001
        if not mask.any():
            continue

        valid_ji = np.clip(ji[mask], 0, n_joints - 1)
        valid_w = wi[mask]
        valid_v = v_homo[mask]

        # Batch matrix multiply: for each vertex, M[ji] @ v
        # Group by joint index for efficiency
        for j in np.unique(valid_ji):
            jmask = valid_ji == j
            verts = valid_v[jmask]  # (K, 4)
            w = valid_w[jmask]      # (K,)
            M = skinning_mats[j]    # (4, 4)
            transformed = (M @ verts.T).T[:, :3]  # (K, 3)
            # Write back: need original indices
            orig_indices = np.where(mask)[0][jmask]
            result[orig_indices] += transformed * w[:, np.newaxis]

    return result


# ── GLTF mesh data extraction ──

def _extract_skin_weights(glb_path):
    """Extract vertex joint indices and weights from GLB mesh.

    Returns (joints_4, weights_4) arrays or (None, None) if not skinned.
    """
    with open(glb_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'glTF':
            return None, None
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        chunk_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # chunk type
        gltf = json.loads(f.read(chunk_len))
        if f.tell() < length:
            bin_len = struct.unpack('<I', f.read(4))[0]
            f.read(4)
            bin_data = f.read(bin_len)
        else:
            return None, None

    if 'meshes' not in gltf:
        return None, None

    mesh = gltf['meshes'][0]
    prim = mesh['primitives'][0]
    attrs = prim.get('attributes', {})

    if 'JOINTS_0' not in attrs or 'WEIGHTS_0' not in attrs:
        return None, None

    def read_acc(idx):
        acc = gltf['accessors'][idx]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        comp_type = acc['componentType']
        acc_type = acc['type']
        sizes = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
        n = sizes.get(acc_type, 1)
        total = count * n
        dtypes = {5120:np.int8, 5121:np.uint8, 5122:np.int16,
                  5123:np.uint16, 5125:np.uint32, 5126:np.float32}
        dt = dtypes.get(comp_type, np.float32)
        data = np.frombuffer(bin_data, dtype=dt, count=total, offset=offset)
        return data.reshape(count, n).astype(np.float32)

    joints_4 = read_acc(attrs['JOINTS_0'])
    weights_4 = read_acc(attrs['WEIGHTS_0'])
    return joints_4, weights_4


def list_available_models():
    models = []
    for d in DEFAULT_CHAR_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            ext = Path(f).suffix.lower()
            if ext in ('.glb', '.gltf', '.obj'):
                models.append({'name': Path(f).stem, 'path': os.path.join(d, f), 'format': ext[1:]})
    return models


class Mesh3DStyle:
    """Render a 3D character model driven by 2D keypoints.

    Supports skinned meshes (LBS) and segmented body-part meshes.
    """

    def __init__(self, config: dict):
        style_cfg = config.get('avatar', {}).get('styles', {}).get('mesh3d', {})
        self.model_path = style_cfg.get('model_path', '')
        self.bg_color = tuple(style_cfg.get('bg_color', [0, 0, 0]))
        self.light_intensity = style_cfg.get('light_intensity', 3.0)
        self.camera_distance = style_cfg.get('camera_distance', 2.5)

        self._mesh = None
        self._tpose_geoms = {}
        self._loaded = False
        self._is_skinned = False
        self._skin = None          # _GLTFSkin
        self._joints_4 = None      # vertex joint indices
        self._weights_4 = None     # vertex weights
        self._tpose_verts = None   # original vertex positions

        if self.model_path and os.path.isfile(self.model_path):
            self._try_load_model()

    def _try_load_model(self):
        if not _HAS_TRIMESH:
            logger.warning("trimesh not installed")
            return
        try:
            self._mesh = trimesh.load(self.model_path)
            self._loaded = True

            # Check for skinning data
            if self.model_path.lower().endswith('.glb'):
                j4, w4 = _extract_skin_weights(self.model_path)
                if j4 is not None:
                    self._skin = _GLTFSkin(self.model_path)
                    if self._skin.joints and len(self._skin._std_to_gltf_idx) >= 4:
                        self._is_skinned = True
                        self._joints_4 = j4
                        self._weights_4 = w4

                        # Cache T-pose vertices
                        if isinstance(self._mesh, trimesh.Scene):
                            for name, geom in self._mesh.geometry.items():
                                self._tpose_verts = geom.vertices.copy()
                                break
                        else:
                            self._tpose_verts = self._mesh.vertices.copy()

                        mapped = list(self._skin._std_to_gltf_idx.keys())
                        logger.info("Skinned mesh: %d joints mapped: %s",
                                   len(mapped), mapped)

            # Cache geometry for segmented mode
            if not self._is_skinned:
                if isinstance(self._mesh, trimesh.Scene):
                    for name, geom in self._mesh.geometry.items():
                        self._tpose_geoms[name] = geom.copy()
                else:
                    self._tpose_geoms['mesh'] = self._mesh.copy()

            logger.info("Loaded: %s [skinned=%s]", self.model_path, self._is_skinned)
        except Exception as e:
            logger.error("Failed to load %s: %s", self.model_path, e)

    def render(self, canvas, keypoints, scores=None, min_score=0.3):
        if self._loaded and _HAS_PYRENDER:
            return self._render_3d(canvas, keypoints, scores, min_score)
        return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

    def _render_3d(self, canvas, keypoints, scores, min_score):
        K = len(keypoints)
        if K < 17:
            return canvas
        h, w = canvas.shape[:2]

        try:
            pts3d = _lift_2d_to_3d(keypoints, w, h)
            if pts3d is None:
                return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

            posed = _compute_posed_joints(pts3d)

            if self._is_skinned:
                posed_geoms = self._apply_skinned_pose(posed)
            else:
                posed_geoms = self._apply_segmented_pose(posed)

            # Build scene
            scene = pyrender.Scene(
                bg_color=[self.bg_color[2]/255, self.bg_color[1]/255,
                          self.bg_color[0]/255, 0.0],
                ambient_light=[0.3, 0.3, 0.3],
            )

            for geom in posed_geoms:
                try:
                    mesh = pyrender.Mesh.from_trimesh(geom)
                    scene.add(mesh)
                except Exception:
                    # Fallback: strip textures if they cause issues
                    geom.visual = trimesh.visual.ColorVisual(
                        vertex_colors=np.full((len(geom.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
                    )
                    mesh = pyrender.Mesh.from_trimesh(geom)
                    scene.add(mesh)

            # Camera
            if self._is_skinned:
                tpose_pos = self._skin.get_tpose_joint_positions()
                hip = tpose_pos.get('Hips', np.array([0, 0.95, 0]))
                head = tpose_pos.get('Head', np.array([0, 1.55, 0]))
                char_height = np.linalg.norm(head - hip) * 2.5
                cam_dist = max(char_height * 1.2, self.camera_distance)
                look_y = (hip[1] + head[1]) / 2
            else:
                hip_pos = posed.get('Hips', np.array([0, 0.95, 0]))
                head_pos = posed.get('Head', np.array([0, 1.55, 0]))
                look_y = (hip_pos[1] + head_pos[1]) / 2
                cam_dist = self.camera_distance

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            cam_pose = np.eye(4)
            cam_pose[1, 3] = look_y
            cam_pose[2, 3] = cam_dist
            scene.add(camera, pose=cam_pose)

            # Key light
            light = pyrender.DirectionalLight(color=[1,1,1], intensity=self.light_intensity)
            lp = np.eye(4)
            lp[:3,:3] = _rotation_between(np.array([0,0,-1]), np.array([-0.2,-0.3,-1.0]))
            lp[:3,3] = cam_pose[:3,3]
            scene.add(light, pose=lp)

            # Fill light
            fill = pyrender.DirectionalLight(color=[0.7,0.8,1.0], intensity=self.light_intensity*0.4)
            fp = np.eye(4)
            fp[:3,:3] = _rotation_between(np.array([0,0,-1]), np.array([0.5,-0.1,-1.0]))
            fp[:3,3] = cam_pose[:3,3]
            scene.add(fill, pose=fp)

            r = pyrender.OffscreenRenderer(w, h)
            color, _ = r.render(scene)
            r.delete()

            rendered = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
            mask = gray > 5
            canvas[mask] = rendered[mask]
            return canvas

        except Exception as e:
            logger.error("3D render failed: %s", e)
            import traceback; traceback.print_exc()
            return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

    def _apply_skinned_pose(self, posed_joints):
        """Apply LBS skinning to the mesh using IK-derived joint positions."""
        skinning_mats = self._skin.compute_skinning_matrices(posed_joints)

        # Apply LBS to vertices
        new_verts = _apply_lbs(
            self._tpose_verts, self._joints_4, self._weights_4, skinning_mats
        )

        # Create posed mesh copy
        if isinstance(self._mesh, trimesh.Scene):
            for name, geom in self._mesh.geometry.items():
                g = geom.copy()
                if len(new_verts) == len(g.vertices):
                    g.vertices = new_verts.astype(np.float32)
                return [g]
        else:
            g = self._mesh.copy()
            g.vertices = new_verts.astype(np.float32)
            return [g]

    def _apply_segmented_pose(self, posed_joints):
        """Apply rigid per-bone transforms to segmented geometry."""
        result = []
        for bone_name, (pj, cj, geom_names) in _BONE_DEFS.items():
            tp = _TPOSE.get(pj)
            tc = _TPOSE.get(cj)
            pp = posed_joints.get(pj)
            pc = posed_joints.get(cj)
            if tp is None or tc is None or pp is None or pc is None:
                continue
            if pj == cj:
                T = np.eye(4)
                T[:3, 3] = pp - tp
            else:
                T = _bone_transform(tp, tc, pp, pc)
            for gn in geom_names:
                if gn in self._tpose_geoms:
                    g = self._tpose_geoms[gn].copy()
                    g.apply_transform(T)
                    result.append(g)
        return result

    def _render_wireframe_3d(self, canvas, keypoints, scores, min_score):
        """Fallback wireframe renderer."""
        K = len(keypoints)
        if K < 17:
            return canvas
        kps = keypoints.astype(np.float64)
        torso_w = np.linalg.norm(kps[5] - kps[6])
        if torso_w < 3:
            return canvas

        base_color = np.array([180, 200, 220], dtype=np.float32)
        segments = [
            (5,7,0.18),(7,9,0.14),(6,8,0.18),(8,10,0.14),
            (11,13,0.24),(13,15,0.18),(12,14,0.24),(14,16,0.18),
        ]

        # Torso
        pts = np.array([kps[5],kps[6],kps[12],kps[11]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], tuple(base_color.astype(int).tolist()), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True,
                      tuple((base_color*1.3).clip(0,255).astype(int).tolist()), 2, cv2.LINE_AA)

        mid_y_top = int((kps[5][1]+kps[6][1])/2)
        mid_y_bot = int((kps[11][1]+kps[12][1])/2)
        step = max(int(torso_w*0.12), 4)
        hc = tuple((base_color*0.7).astype(int).tolist())
        for y in range(mid_y_top, mid_y_bot, step):
            cv2.line(canvas,(int(kps[5][0]),y),(int(kps[6][0]),y),hc,1,cv2.LINE_AA)
        for x in range(int(kps[5][0]),int(kps[6][0]),step):
            cv2.line(canvas,(x,mid_y_top),(x,mid_y_bot),hc,1,cv2.LINE_AA)

        for j1,j2,wr in segments:
            if j1>=K or j2>=K: continue
            if scores is not None and (scores[j1]<min_score or scores[j2]<min_score): continue
            p1,p2 = kps[j1],kps[j2]
            l = np.linalg.norm(p2-p1)
            if l<2: continue
            d = p2-p1; perp = np.array([-d[1],d[0]])/l
            w = max(int(torso_w*wr),4)
            poly = np.array([p1+perp*w/2,p2+perp*w/2,p2-perp*w/2,p1-perp*w/2],dtype=np.int32)
            cv2.fillConvexPoly(canvas,poly,tuple(base_color.astype(int).tolist()),cv2.LINE_AA)
            cv2.line(canvas,tuple(p1.astype(int)),tuple(p2.astype(int)),
                     tuple((base_color*1.4).clip(0,255).astype(int).tolist()),max(w//4,1),cv2.LINE_AA)

        jr = max(int(torso_w*0.06),3)
        for idx in range(min(K,17)):
            if scores is not None and scores[idx]<min_score: continue
            pt = tuple(kps[idx].astype(int))
            cv2.circle(canvas,pt,jr,tuple((base_color*0.8).astype(int).tolist()),-1,cv2.LINE_AA)

        hr = max(int(torso_w*0.22),10)
        hc2 = tuple(kps[0].astype(int))
        cv2.circle(canvas,hc2,hr,tuple(base_color.astype(int).tolist()),-1,cv2.LINE_AA)
        cv2.circle(canvas,hc2,hr,tuple((base_color*1.3).clip(0,255).astype(int).tolist()),2,cv2.LINE_AA)

        for idx in [9,10]:
            if idx>=K: continue
            pt = tuple(kps[idx].astype(int))
            cv2.circle(canvas,pt,max(int(torso_w*0.05),2),tuple(base_color.astype(int).tolist()),-1,cv2.LINE_AA)

        for idx in [15,16]:
            if idx>=K: continue
            knee = 13 if idx==15 else 14
            fd = kps[idx]-kps[knee]
            angle = float(np.degrees(np.arctan2(fd[1],fd[0])))
            fw,fh = max(int(torso_w*0.12),4), max(int(torso_w*0.06),3)
            cv2.ellipse(canvas,tuple(kps[idx].astype(int)),(fw,fh),angle+90,0,360,
                        tuple(base_color.astype(int).tolist()),-1,cv2.LINE_AA)

        return canvas
