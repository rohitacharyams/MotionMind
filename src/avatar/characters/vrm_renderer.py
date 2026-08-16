"""
VRM character renderer: loads VRM model, applies textures properly,
renders with skinning driven by COCO-17 keypoints.

VRM files are GLB (glTF binary) with VRM extension for humanoid bone mapping.
We use trimesh for mesh loading, custom GLB parser for skinning,
and pyrender for rendering.
"""
import os
import struct
import json
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
import io

# ── VRM humanoid bone name → our standard skeleton ──
_VRM_TO_STD = {
    'hips': 'Hips',
    'spine': 'Spine',
    'chest': 'Spine2',
    'upperChest': 'UpperChest',
    'neck': 'Neck',
    'head': 'Head',
    'leftShoulder': 'LeftShoulder',
    'leftUpperArm': 'LeftArm',
    'leftLowerArm': 'LeftForeArm',
    'leftHand': 'LeftHand',
    'rightShoulder': 'RightShoulder',
    'rightUpperArm': 'RightArm',
    'rightLowerArm': 'RightForeArm',
    'rightHand': 'RightHand',
    'leftUpperLeg': 'LeftUpLeg',
    'leftLowerLeg': 'LeftLeg',
    'leftFoot': 'LeftFoot',
    'leftToes': 'LeftToes',
    'rightUpperLeg': 'RightUpLeg',
    'rightLowerLeg': 'RightLeg',
    'rightFoot': 'RightFoot',
    'rightToes': 'RightToes',
}

# COCO-17 keypoint index → standard bone name
_COCO_TO_JOINT = {
    0: 'Head', 5: 'LeftArm', 6: 'RightArm',
    7: 'LeftForeArm', 8: 'RightForeArm',
    9: 'LeftHand', 10: 'RightHand',
    11: 'LeftUpLeg', 12: 'RightUpLeg',
    13: 'LeftLeg', 14: 'RightLeg',
    15: 'LeftFoot', 16: 'RightFoot',
}


class VRMRenderer:
    """Load and render a VRM character model driven by 2D keypoints."""

    def __init__(self, vrm_path, width=1280, height=720):
        self.width = width
        self.height = height
        self.vrm_path = vrm_path

        # Parse GLB
        self._parse_glb(vrm_path)

        # Build skin data
        self._build_skin()

        # Load scene with trimesh (for geometry + textures)
        self._load_trimesh_scene()

        # Setup pyrender
        self._setup_renderer()

    def _parse_glb(self, path):
        """Parse GLB binary to extract GLTF JSON + binary buffer."""
        with open(path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'glTF', f"Not a GLB file: {magic}"
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

    def _read_accessor(self, idx):
        """Read a GLTF accessor from binary buffer."""
        gltf = self._gltf
        accessor = gltf['accessors'][idx]
        bv = gltf['bufferViews'][accessor['bufferView']]
        offset = bv.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        count = accessor['count']
        comp_type = accessor['componentType']
        acc_type = accessor['type']
        type_sizes = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
        n_comp = type_sizes.get(acc_type, 1)
        total = count * n_comp
        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = dtype_map.get(comp_type, np.float32)
        data = np.frombuffer(self._bin, dtype=dtype, count=total, offset=offset)
        return data.astype(np.float32) if dtype != np.float32 else data.copy()

    @staticmethod
    def _quat_to_mat(q):
        """[x,y,z,w] quaternion to 3x3 rotation."""
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def _build_skin(self):
        """Build skinning data: joint hierarchy, inverse bind matrices, bone map."""
        gltf = self._gltf
        if not gltf.get('skins'):
            raise ValueError("No skin found in VRM model")

        skin = gltf['skins'][0]
        self.joint_indices = skin.get('joints', [])
        nodes = gltf['nodes']

        # Joint names
        self.joint_names = [nodes[j].get('name', f'j{j}') for j in self.joint_indices]

        # Parent map
        self._parent_map = {}
        self._children_map = {}
        for ni, node in enumerate(nodes):
            self._children_map[ni] = node.get('children', [])
            for c in node.get('children', []):
                self._parent_map[c] = ni

        # Local transforms
        self._local_transforms = {}
        for ni, node in enumerate(nodes):
            T = np.eye(4)
            if 'matrix' in node:
                T = np.array(node['matrix']).reshape(4, 4).T
            else:
                if 'rotation' in node:
                    T[:3, :3] = self._quat_to_mat(node['rotation'])
                if 'scale' in node:
                    s = node['scale']
                    T[:3, 0] *= s[0]; T[:3, 1] *= s[1]; T[:3, 2] *= s[2]
                if 'translation' in node:
                    T[:3, 3] = node['translation']
            self._local_transforms[ni] = T

        # Compute world transforms
        self._world_transforms = {}
        visited = set()
        def dfs(ni, parent_world):
            if ni in visited:
                return
            visited.add(ni)
            local = self._local_transforms.get(ni, np.eye(4))
            world = parent_world @ local
            self._world_transforms[ni] = world
            for c in self._children_map.get(ni, []):
                dfs(c, world)

        all_children = set()
        for n in nodes:
            for c in n.get('children', []):
                all_children.add(c)
        roots = [i for i in range(len(nodes)) if i not in all_children]
        for r in roots:
            dfs(r, np.eye(4))

        # Inverse bind matrices
        ibm_idx = skin.get('inverseBindMatrices')
        if ibm_idx is not None:
            ibm_data = self._read_accessor(ibm_idx)
            n = len(self.joint_indices)
            self._ibm = ibm_data.reshape(n, 4, 4)
            for i in range(n):
                self._ibm[i] = self._ibm[i].T  # column-major to row-major
        else:
            n = len(self.joint_indices)
            self._ibm = np.array([np.eye(4)] * n)

        # Strip scene root transform
        root_ji = self.joint_indices[0]
        ancestors = []
        cur = self._parent_map.get(root_ji)
        while cur is not None:
            ancestors.append(cur)
            cur = self._parent_map.get(cur)
        ancestors.reverse()
        scene_root = np.eye(4)
        for a in ancestors:
            scene_root = scene_root @ self._local_transforms.get(a, np.eye(4))
        self._inv_scene_root = np.linalg.inv(scene_root)

        # Skin-space world transforms
        self._skin_world = {}
        for ni, wt in self._world_transforms.items():
            self._skin_world[ni] = self._inv_scene_root @ wt

        # VRM bone mapping
        self._vrm_bone_map = {}  # std_name -> joint list index
        ext = self._gltf.get('extensions', {})
        if 'VRM' in ext:
            for bone in ext['VRM'].get('humanoid', {}).get('humanBones', []):
                vrm_name = bone['bone']
                node_idx = bone['node']
                std_name = _VRM_TO_STD.get(vrm_name)
                if std_name and node_idx in self.joint_indices:
                    ji = self.joint_indices.index(node_idx)
                    self._vrm_bone_map[std_name] = ji

        # T-pose joint positions in skin space
        self._tpose_positions = {}
        for std_name, ji in self._vrm_bone_map.items():
            ni = self.joint_indices[ji]
            swt = self._skin_world.get(ni)
            if swt is not None:
                self._tpose_positions[std_name] = swt[:3, 3].copy()

        # Read vertex skinning data (joints + weights)
        self._mesh_skin_data = []
        for mesh_def in gltf.get('meshes', []):
            for prim in mesh_def.get('primitives', []):
                attrs = prim.get('attributes', {})
                if 'JOINTS_0' in attrs and 'WEIGHTS_0' in attrs:
                    joints_data = self._read_accessor(attrs['JOINTS_0'])
                    weights_data = self._read_accessor(attrs['WEIGHTS_0'])
                    nv = len(joints_data) // 4
                    self._mesh_skin_data.append({
                        'joints': joints_data.reshape(nv, 4).astype(int),
                        'weights': weights_data.reshape(nv, 4),
                    })
                else:
                    self._mesh_skin_data.append(None)

        print(f"VRM skin: {len(self.joint_indices)} joints, "
              f"{len(self._vrm_bone_map)} mapped bones")
        for name, pos in sorted(self._tpose_positions.items()):
            print(f"  {name:20s}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def _load_trimesh_scene(self):
        """Load geometries with trimesh for proper texture extraction."""
        self._trimesh_scene = trimesh.load(self.vrm_path, file_type='glb', process=False)
        print(f"Loaded {len(self._trimesh_scene.geometry)} geometries")

    def _setup_renderer(self):
        """Setup pyrender scene with lighting and camera."""
        self._pr_scene = pyrender.Scene(
            bg_color=[0.08, 0.08, 0.12, 1.0],
            ambient_light=[0.4, 0.4, 0.4]
        )

        # Add meshes
        self._mesh_nodes = []
        for name, geom in self._trimesh_scene.geometry.items():
            try:
                # Get the transform from the scene graph
                transform = self._trimesh_scene.graph.get(name)[0] if hasattr(
                    self._trimesh_scene.graph, 'get') else np.eye(4)
            except Exception:
                transform = np.eye(4)

            try:
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
            except Exception:
                # Fallback: create simple mesh
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.7, 0.65, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.7
                )
                pr_mesh = pyrender.Mesh.from_trimesh(geom, material=mat, smooth=True)

            node = self._pr_scene.add(pr_mesh, pose=transform)
            self._mesh_nodes.append((name, node, geom))

        # Camera — face the character from front (VRM characters face -Z in T-pose)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 7.0,
                                          aspectRatio=self.width / self.height)
        cam_pose = np.eye(4)
        cam_pose[1, 3] = 1.0  # look at chest height
        cam_pose[2, 3] = 4.0  # distance
        self._cam_node = self._pr_scene.add(cam, pose=cam_pose)

        # Lights
        light = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.0)
        lp = np.eye(4)
        lp[:3, :3] = self._rotation_x(-0.5) @ self._rotation_y(0.3)
        self._pr_scene.add(light, pose=lp)

        # Fill light
        fill = pyrender.DirectionalLight(color=[0.7, 0.75, 0.9], intensity=1.5)
        fp = np.eye(4)
        fp[:3, :3] = self._rotation_x(-0.3) @ self._rotation_y(-0.5)
        self._pr_scene.add(fill, pose=fp)

        self._renderer = pyrender.OffscreenRenderer(self.width, self.height)

    @staticmethod
    def _rotation_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _rotation_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def _rotation_between(v1, v2):
        """Rotation matrix from v1 to v2."""
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

    def _lift_2d_to_3d(self, kps_2d):
        """Lift 2D COCO-17 keypoints into 3D model space."""
        kps = kps_2d[:17].copy()
        hip_center = (kps[11] + kps[12]) / 2
        shoulder_center = (kps[5] + kps[6]) / 2
        torso_px = np.linalg.norm(shoulder_center - hip_center)
        if torso_px < 5:
            return None

        # Scale: model hip-to-head distance
        hip_pos = self._tpose_positions.get('Hips', np.array([0, 0.95, 0]))
        head_pos = self._tpose_positions.get('Head', np.array([0, 1.55, 0]))
        model_torso = np.linalg.norm(head_pos - hip_pos)
        px_torso = np.linalg.norm(kps[0] - hip_center)
        if px_torso < 5:
            px_torso = torso_px * 1.5
        scale = model_torso / px_torso

        # Center and scale
        pts = (kps - hip_center) * scale
        pts[:, 1] = -pts[:, 1]  # flip Y (screen coords → 3D Y-up)
        pts[:, 0] += hip_pos[0]
        pts[:, 1] += hip_pos[1]

        pts3d = np.zeros((17, 3))
        pts3d[:, :2] = pts

        # Solve Z from bone length constraints
        bone_pairs = [
            (5, 7, 'LeftArm', 'LeftForeArm'),
            (7, 9, 'LeftForeArm', 'LeftHand'),
            (6, 8, 'RightArm', 'RightForeArm'),
            (8, 10, 'RightForeArm', 'RightHand'),
            (11, 13, 'LeftUpLeg', 'LeftLeg'),
            (13, 15, 'LeftLeg', 'LeftFoot'),
            (12, 14, 'RightUpLeg', 'RightLeg'),
            (14, 16, 'RightLeg', 'RightFoot'),
        ]
        for pi, ci, pn, cn in bone_pairs:
            p_pos = self._tpose_positions.get(pn)
            c_pos = self._tpose_positions.get(cn)
            if p_pos is None or c_pos is None:
                continue
            ref_len = np.linalg.norm(c_pos - p_pos)
            d2 = (pts3d[ci, 0] - pts3d[pi, 0])**2 + (pts3d[ci, 1] - pts3d[pi, 1])**2
            l2 = ref_len ** 2
            if d2 < l2:
                pts3d[ci, 2] = pts3d[pi, 2] + np.sqrt(l2 - d2) * 0.7

        return pts3d

    def _compute_bone_rotations(self, pts3d):
        """Compute bone rotation matrices from posed 3D joints."""
        posed = {jn: pts3d[ci].copy() for ci, jn in _COCO_TO_JOINT.items()}
        hip = (pts3d[11] + pts3d[12]) / 2
        shoulder = (pts3d[5] + pts3d[6]) / 2
        posed['Hips'] = hip
        posed['Spine'] = hip + (shoulder - hip) * 0.22
        posed['Spine2'] = hip + (shoulder - hip) * 0.67
        posed['UpperChest'] = hip + (shoulder - hip) * 0.85
        posed['Neck'] = shoulder
        posed['LeftShoulder'] = shoulder + (pts3d[5] - shoulder) * 0.3
        posed['RightShoulder'] = shoulder + (pts3d[6] - shoulder) * 0.3

        # Compute rotation for each mapped bone
        rotations = {}  # std_name -> 4x4 transform

        bone_chains = [
            ('Hips', 'Spine'), ('Spine', 'Spine2'), ('Spine2', 'Neck'),
            ('Neck', 'Head'),
            ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
            ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
            ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'),
            ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'),
        ]

        for parent_name, child_name in bone_chains:
            if parent_name not in self._tpose_positions or child_name not in self._tpose_positions:
                continue
            if parent_name not in posed or child_name not in posed:
                continue

            tp_parent = self._tpose_positions[parent_name]
            tp_child = self._tpose_positions[child_name]
            p_parent = posed[parent_name]
            p_child = posed[child_name]

            td = tp_child - tp_parent
            tl = np.linalg.norm(td)
            if tl < 1e-6:
                continue
            td = td / tl

            pd = p_child - p_parent
            pl = np.linalg.norm(pd)
            if pl < 1e-6:
                continue
            pd = pd / pl

            R = self._rotation_between(td, pd)
            tm = (tp_parent + tp_child) / 2
            pm = (p_parent + p_child) / 2

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pm - R @ tm
            rotations[parent_name] = T

        return rotations

    def _apply_skinning(self, bone_transforms):
        """Apply LBS to all meshes using bone transforms."""
        # Compute per-joint final transforms: posed_world @ IBM
        n_joints = len(self.joint_indices)
        joint_mats = np.zeros((n_joints, 4, 4))

        # Build reverse maps for efficient lookup
        ji_to_std = {}
        for sn, idx in self._vrm_bone_map.items():
            ji_to_std[idx] = sn
        node_to_ji = {}
        for ji, ni in enumerate(self.joint_indices):
            node_to_ji[ni] = ji

        for ji in range(n_joints):
            ni = self.joint_indices[ji]
            std_name = ji_to_std.get(ji)

            if std_name and std_name in bone_transforms:
                # Use our computed transform
                joint_mats[ji] = bone_transforms[std_name] @ self._ibm[ji]
            else:
                # Walk up parent chain to find nearest ancestor with a transform
                ancestor_transform = None
                cur = self._parent_map.get(ni)
                while cur is not None:
                    cur_ji = node_to_ji.get(cur)
                    if cur_ji is not None:
                        cur_std = ji_to_std.get(cur_ji)
                        if cur_std and cur_std in bone_transforms:
                            ancestor_transform = bone_transforms[cur_std]
                            break
                    cur = self._parent_map.get(cur)

                if ancestor_transform is not None:
                    joint_mats[ji] = ancestor_transform @ self._ibm[ji]
                else:
                    # No ancestor has a transform — stay in T-pose
                    joint_mats[ji] = self._skin_world.get(ni, np.eye(4)) @ self._ibm[ji]

        # Apply to each mesh geometry
        skin_idx = 0
        for name, node, geom in self._mesh_nodes:
            if skin_idx >= len(self._mesh_skin_data) or self._mesh_skin_data[skin_idx] is None:
                skin_idx += 1
                continue

            sd = self._mesh_skin_data[skin_idx]
            joints = sd['joints']
            weights = sd['weights']
            nv = len(geom.vertices)

            if len(joints) != nv:
                skin_idx += 1
                continue

            # LBS
            new_verts = np.zeros((nv, 3))
            orig_verts = np.hstack([geom.vertices, np.ones((nv, 1))])

            for j in range(4):
                w = weights[:, j]
                ji = joints[:, j].astype(int)
                valid = w > 0.001
                if not valid.any():
                    continue
                for vi in np.where(valid)[0]:
                    M = joint_mats[ji[vi]]
                    new_verts[vi] += w[vi] * (M @ orig_verts[vi])[:3]

            # Update mesh in pyrender
            geom.vertices = new_verts
            # Remove old node and add new
            self._pr_scene.remove_node(node)
            try:
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
            except Exception:
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.7, 0.65, 1.0])
                pr_mesh = pyrender.Mesh.from_trimesh(geom, material=mat, smooth=True)
            new_node = self._pr_scene.add(pr_mesh)
            self._mesh_nodes[skin_idx] = (name, new_node, geom)

            skin_idx += 1

    def render_frame(self, keypoints_2d=None):
        """Render a single frame, optionally with pose from 2D keypoints."""
        if keypoints_2d is not None and len(keypoints_2d) >= 17:
            pts3d = self._lift_2d_to_3d(keypoints_2d)
            if pts3d is not None:
                bone_transforms = self._compute_bone_rotations(pts3d)
                self._apply_skinning(bone_transforms)

        color, depth = self._renderer.render(self._pr_scene)
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def render_tpose(self):
        """Render the T-pose (no keypoints)."""
        color, depth = self._renderer.render(self._pr_scene)
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def cleanup(self):
        """Release renderer resources."""
        self._renderer.delete()


# ── Quick test ──
if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else 'data/models/fem_vroid.vrm'
    out_dir = 'data/output_videos'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {model}...")
    renderer = VRMRenderer(model, 1280, 720)

    print("Rendering T-pose...")
    frame = renderer.render_tpose()
    cv2.imwrite(f'{out_dir}/vrm_tpose.png', frame)
    print(f"Saved {out_dir}/vrm_tpose.png")

    renderer.cleanup()
