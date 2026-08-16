"""
Motion Transfer VRM — AI-Choreographer-quality 3D character dance rendering.

Pipeline:
  1. Extract 3D pose from video (MediaPipe world landmarks)
  2. Smooth landmarks temporally
  3. Map to VRM skeleton joint positions (Y-up)
  4. Drive VRM skinned mesh via BFS FK + LBS
  5. Render with pyrender (textured, lit, high quality)
  6. Export side-by-side + avatar-only videos

Uses the proven VRMDanceV4 renderer with direct 3D landmark input
(skips the lossy 2D→3D lifting).
"""
import os, sys, time, struct, json
import numpy as np
import cv2
import mediapipe as mp
import trimesh
import pyrender
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──
INPUT_VIDEO = 'data/input_videos/wab.mp4'
VRM_MODEL   = 'data/models/fem_vroid.vrm'
OUT_DIR     = 'data/output_videos'
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 720, 720        # render resolution
START_FRAME   = 250
MAX_FRAMES    = 600
FPS           = 30.0

# ── VRM bone map ──
_VRM_BONE_MAP = {
    'hips': 'Hips', 'spine': 'Spine', 'chest': 'Spine2',
    'upperChest': 'Spine2',
    'neck': 'Neck', 'head': 'Head',
    'leftShoulder': 'LeftShoulder',
    'leftUpperArm': 'LeftArm', 'leftLowerArm': 'LeftForeArm', 'leftHand': 'LeftHand',
    'rightShoulder': 'RightShoulder',
    'rightUpperArm': 'RightArm', 'rightLowerArm': 'RightForeArm', 'rightHand': 'RightHand',
    'leftUpperLeg': 'LeftUpLeg', 'leftLowerLeg': 'LeftLeg', 'leftFoot': 'LeftFoot',
    'rightUpperLeg': 'RightUpLeg', 'rightLowerLeg': 'RightLeg', 'rightFoot': 'RightFoot',
}

# Canonical T-pose (Y-up, meters)
_TPOSE = {
    'Hips':          np.array([0, 0.95, 0]),
    'Spine':         np.array([0, 1.05, 0]),
    'Spine2':        np.array([0, 1.25, 0]),
    'Neck':          np.array([0, 1.4, 0]),
    'Head':          np.array([0, 1.55, 0]),
    'LeftShoulder':  np.array([-0.12, 1.38, 0]),
    'LeftArm':       np.array([-0.22, 1.38, 0]),
    'LeftForeArm':   np.array([-0.48, 1.38, 0]),
    'LeftHand':      np.array([-0.72, 1.38, 0]),
    'RightShoulder': np.array([0.12, 1.38, 0]),
    'RightArm':      np.array([0.22, 1.38, 0]),
    'RightForeArm':  np.array([0.48, 1.38, 0]),
    'RightHand':     np.array([0.72, 1.38, 0]),
    'LeftUpLeg':     np.array([-0.1, 0.92, 0]),
    'LeftLeg':       np.array([-0.1, 0.5, 0]),
    'LeftFoot':      np.array([-0.1, 0.08, 0]),
    'RightUpLeg':    np.array([0.1, 0.92, 0]),
    'RightLeg':      np.array([0.1, 0.5, 0]),
    'RightFoot':     np.array([0.1, 0.08, 0]),
}


# ═══════════════════════════════════════════════════════════════
#  MediaPipe → Canonical 3D joints
# ═══════════════════════════════════════════════════════════════

# MediaPipe landmark indices
MP_NOSE = 0
MP_L_EAR, MP_R_EAR = 7, 8
MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_ELBOW, MP_R_ELBOW = 13, 14
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_INDEX, MP_R_INDEX = 19, 20
MP_L_HIP, MP_R_HIP = 23, 24
MP_L_KNEE, MP_R_KNEE = 25, 26
MP_L_ANKLE, MP_R_ANKLE = 27, 28
MP_L_FOOT, MP_R_FOOT = 31, 32


def mp_to_posed_joints(mp_lms):
    """Convert MediaPipe 3D world landmarks to canonical joint positions.

    MediaPipe uses Y-down, Z-toward-camera.
    Our canonical skeleton uses Y-up.

    Returns dict: joint_name -> (3,) position
    """
    lm = mp_lms.copy()
    lm[:, 1] = -lm[:, 1]   # Y: down -> up
    lm[:, 2] = -lm[:, 2]   # Z flip for front-facing

    pelvis   = (lm[MP_L_HIP] + lm[MP_R_HIP]) / 2
    neck     = (lm[MP_L_SHOULDER] + lm[MP_R_SHOULDER]) / 2
    head_mid = (lm[MP_L_EAR] + lm[MP_R_EAR]) / 2

    posed = {}
    posed['Hips']         = pelvis
    posed['Spine']        = pelvis + (neck - pelvis) * 0.22
    posed['Spine2']       = pelvis + (neck - pelvis) * 0.67
    posed['Neck']         = neck
    posed['Head']         = head_mid
    posed['LeftShoulder'] = neck + (lm[MP_L_SHOULDER] - neck) * 0.4
    posed['LeftArm']      = lm[MP_L_SHOULDER]
    posed['LeftForeArm']  = lm[MP_L_ELBOW]
    posed['LeftHand']     = lm[MP_L_WRIST]
    posed['RightShoulder']= neck + (lm[MP_R_SHOULDER] - neck) * 0.4
    posed['RightArm']     = lm[MP_R_SHOULDER]
    posed['RightForeArm'] = lm[MP_R_ELBOW]
    posed['RightHand']    = lm[MP_R_WRIST]
    posed['LeftUpLeg']    = lm[MP_L_HIP]
    posed['LeftLeg']      = lm[MP_L_KNEE]
    posed['LeftFoot']     = lm[MP_L_ANKLE]
    posed['RightUpLeg']   = lm[MP_R_HIP]
    posed['RightLeg']     = lm[MP_R_KNEE]
    posed['RightFoot']    = lm[MP_R_ANKLE]

    # Normalize: center at canonical hips, scale to canonical skeleton height
    canon_height = np.linalg.norm(_TPOSE['Head'] - _TPOSE['Hips'])  # ~0.6
    curr_height  = np.linalg.norm(posed['Head'] - posed['Hips'])
    if curr_height < 0.01:
        return None

    scale = canon_height / curr_height
    offset = _TPOSE['Hips'] - posed['Hips'] * scale

    for name in posed:
        posed[name] = posed[name] * scale + offset

    return posed


# ═══════════════════════════════════════════════════════════════
#  VRM Renderer (adapted from VRMDanceV4)
# ═══════════════════════════════════════════════════════════════

def _rotation_between(v1, v2):
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


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def _read_accessor(gltf, bdata, idx):
    acc = gltf['accessors'][idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    ts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    n = acc['count'] * ts.get(acc['type'], 1)
    dm = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
          5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    dt = dm.get(acc['componentType'], np.float32)
    d = np.frombuffer(bdata, dtype=dt, count=n, offset=off)
    return d.astype(np.float32) if dt != np.float32 else d.copy()


class VRMCharacterRenderer:
    """High-quality VRM character renderer with BFS FK + LBS skinning."""

    def __init__(self, vrm_path, w=WIDTH, h=HEIGHT):
        self.W, self.H = w, h

        # ── Parse GLB ──
        with open(vrm_path, 'rb') as f:
            f.read(4)  # magic
            f.read(4)  # version
            ln = struct.unpack('<I', f.read(4))[0]
            cl = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # chunk type
            self._gltf = json.loads(f.read(cl))
            if f.tell() < ln:
                bl = struct.unpack('<I', f.read(4))[0]
                f.read(4)
                self._bin = f.read(bl)
            else:
                self._bin = b''

        nodes = self._gltf['nodes']
        all_skins = self._gltf.get('skins', [])
        # ── Multi-skin support: build UNION of all skins' joints ──
        # Some VRMs (e.g. AliciaSolid) split the rig across many sub-skins,
        # where skin[0] may not contain the leg/foot bones. We union all skins
        # so every animated bone is present.
        seen = set()
        union_joints = []
        for sk in all_skins:
            for ji in sk['joints']:
                if ji not in seen:
                    seen.add(ji)
                    union_joints.append(ji)
        self.joints = union_joints
        nj = len(self.joints)
        # First skin (kept for legacy single-skin code paths)
        skin = all_skins[0] if all_skins else {'joints': [], 'inverseBindMatrices': None}

        # ── Parse node hierarchy ──
        self._children = {}
        self._local_transforms = {}
        for ni, nd in enumerate(nodes):
            self._children[ni] = nd.get('children', [])
            T = np.eye(4)
            if 'matrix' in nd:
                T = np.array(nd['matrix']).reshape(4, 4).T
            else:
                if 'rotation' in nd:
                    T[:3, :3] = _quat_to_mat(nd['rotation'])
                if 'scale' in nd:
                    s = nd['scale']
                    T[:3, 0] *= s[0]; T[:3, 1] *= s[1]; T[:3, 2] *= s[2]
                if 'translation' in nd:
                    T[:3, 3] = nd['translation']
            self._local_transforms[ni] = T

        # ── World transforms via DFS ──
        self._world_transforms = {}
        visited = set()
        def dfs(ni, pw):
            if ni in visited:
                return
            visited.add(ni)
            self._world_transforms[ni] = pw @ self._local_transforms.get(ni, np.eye(4))
            for c in self._children.get(ni, []):
                dfs(c, self._world_transforms[ni])
        all_children = {c for n in nodes for c in n.get('children', [])}
        for r in [i for i in range(len(nodes)) if i not in all_children]:
            dfs(r, np.eye(4))

        # ── Parent map ──
        self._parent = {}
        for ni, children in self._children.items():
            for c in children:
                self._parent[c] = ni

        # ── Inverse bind matrices (union over all skins) ──
        # For each global joint index, take the inv-bind from the first skin
        # that includes that joint. (Multi-skin VRMs all use the same rest pose,
        # so duplicates agree to numerical precision.)
        joint_to_global = {ji: gi for gi, ji in enumerate(self.joints)}
        self.inv_bind_mats = np.tile(np.eye(4), (nj, 1, 1))
        filled = [False] * nj
        for sk in all_skins:
            ibm_idx = sk.get('inverseBindMatrices')
            sk_joints = sk['joints']
            n_sk = len(sk_joints)
            if ibm_idx is None:
                continue
            d = _read_accessor(self._gltf, self._bin, ibm_idx).reshape(n_sk, 4, 4)
            for k, ji in enumerate(sk_joints):
                gi = joint_to_global[ji]
                if not filled[gi]:
                    self.inv_bind_mats[gi] = d[k].T
                    filled[gi] = True
        # Per-skin local→global remap (used when loading per-primitive joint indices)
        self._skin_remap = []  # list[np.ndarray]: skin-local-idx -> global-idx
        for sk in all_skins:
            self._skin_remap.append(
                np.array([joint_to_global[ji] for ji in sk['joints']], dtype=np.int32)
            )
        # Mesh → skin mapping: walk nodes, record which skin each mesh is bound to.
        self._mesh_to_skin = {}
        for nd in nodes:
            if 'mesh' in nd and 'skin' in nd:
                self._mesh_to_skin.setdefault(nd['mesh'], nd['skin'])

        # ── Strip scene root transform ──
        root_ji = self.joints[0]
        ancestors = []
        c = self._parent.get(root_ji)
        while c is not None:
            ancestors.append(c)
            c = self._parent.get(c)
        ancestors.reverse()
        scene_root = np.eye(4)
        for anc in ancestors:
            scene_root = scene_root @ self._local_transforms.get(anc, np.eye(4))
        inv_scene = np.linalg.inv(scene_root)

        test = (inv_scene @ self._world_transforms.get(root_ji, np.eye(4))) @ self.inv_bind_mats[0]
        if np.abs(test - np.eye(4)).max() < 0.01:
            self._skin_wt = {ni: inv_scene @ wt for ni, wt in self._world_transforms.items()}
        else:
            self._skin_wt = dict(self._world_transforms)

        # ── VRM bone mapping ──
        self._joint_to_std = {}
        self._std_to_idx = {}
        ext = self._gltf.get('extensions', {})
        if 'VRM' in ext:
            for b in ext['VRM']['humanoid']['humanBones']:
                std = _VRM_BONE_MAP.get(b['bone'])
                ni = b['node']
                if std and ni in self.joints:
                    idx = self.joints.index(ni)
                    prefer = ('leftUpperArm', 'rightUpperArm', 'chest', 'upperChest')
                    if std not in self._std_to_idx or b['bone'] in prefer:
                        self._joint_to_std[idx] = std
                        self._std_to_idx[std] = idx

        print(f"  Mapped {len(self._std_to_idx)} joints: {sorted(self._std_to_idx.keys())}")

        # ── Canon-to-model transform ──
        model_pos = {}
        for idx in range(nj):
            std = self._joint_to_std.get(idx)
            if std:
                ni = self.joints[idx]
                swt = self._skin_wt.get(ni)
                if swt is not None:
                    model_pos[std] = swt[:3, 3].copy()

        if 'Hips' in model_pos and 'Head' in model_pos:
            up = model_pos['Head'] - model_pos['Hips']
        else:
            up = np.array([0, 1, 0])
        up = up / np.linalg.norm(up)

        if 'LeftArm' in model_pos and 'RightArm' in model_pos:
            left = model_pos['LeftArm'] - model_pos['RightArm']
        else:
            left = np.array([-1, 0, 0])
        left = left - np.dot(left, up) * up
        left = left / np.linalg.norm(left)
        fwd = np.cross(up, left)
        fwd = fwd / np.linalg.norm(fwd)
        self._canon_to_model = np.column_stack([-left, up, fwd])

        # ── Load skin weights ──
        # Each primitive's JOINTS_0 indexes into ITS mesh's skin's joints array
        # (a per-skin local index). We remap to the union joints array so that
        # one global skinning_mats array works for every primitive.
        self._skin_data = []
        for mi, md in enumerate(self._gltf.get('meshes', [])):
            sk_idx = self._mesh_to_skin.get(mi, 0)
            remap = self._skin_remap[sk_idx] if sk_idx < len(self._skin_remap) else None
            for prim in md.get('primitives', []):
                a = prim.get('attributes', {})
                if 'JOINTS_0' in a and 'WEIGHTS_0' in a:
                    jd = _read_accessor(self._gltf, self._bin, a['JOINTS_0'])
                    wd = _read_accessor(self._gltf, self._bin, a['WEIGHTS_0'])
                    nv = len(jd) // 4
                    j_local = jd.reshape(nv, 4).astype(np.int32)
                    if remap is not None and len(remap) > 0:
                        j_local = np.clip(j_local, 0, len(remap) - 1)
                        j_global = remap[j_local]
                    else:
                        j_global = j_local
                    self._skin_data.append((j_global,
                                           wd.reshape(nv, 4)))
                else:
                    self._skin_data.append(None)

        # ── Load trimesh scene ──
        self._tmesh = trimesh.load(vrm_path, file_type='glb', process=False)
        self._gnames = list(self._tmesh.geometry.keys())
        self._orig_verts = {n: self._tmesh.geometry[n].vertices.copy()
                           for n in self._gnames}

        # VRM faces -Z → flip to face camera
        self._R180 = np.eye(4)
        self._R180[0, 0] = -1
        self._R180[2, 2] = -1

        # ── Pre-build pyrender primitives ──
        self._prim_cache = []
        for name in self._gnames:
            g = self._tmesh.geometry[name].copy()
            g.apply_transform(self._R180)
            try:
                pm = pyrender.Mesh.from_trimesh(g, smooth=True)
                pds = []
                for p in pm.primitives:
                    pds.append({
                        'idx': p.indices.copy() if p.indices is not None else None,
                        'nrm': p.normals.copy() if p.normals is not None else None,
                        'uv': (p.texcoord_0.copy()
                               if hasattr(p, 'texcoord_0') and p.texcoord_0 is not None
                               else None),
                        'col': (p.color_0.copy()
                                if hasattr(p, 'color_0') and p.color_0 is not None
                                else None),
                        'mat': p.material,
                    })
                self._prim_cache.append(pds)
            except Exception as e:
                print(f"  Warn: {name}: {e}")
                self._prim_cache.append(None)

        self._renderer = pyrender.OffscreenRenderer(self.W, self.H)
        print("  VRM renderer ready")

    # ── BFS FK Skinning (identical algorithm to mesh3d.py / VRMDanceV4) ──

    def _find_child_std(self, idx):
        ni = self.joints[idx]
        for child_ni in self._children.get(ni, []):
            if child_ni in self.joints:
                child_idx = self.joints.index(child_ni)
                child_std = self._joint_to_std.get(child_idx)
                if child_std:
                    return child_std
        return None

    def compute_skinning_matrices(self, posed_joints):
        nj = len(self.joints)
        T = self._canon_to_model

        # Model T-pose positions
        model_tpose = {}
        for idx in range(nj):
            std = self._joint_to_std.get(idx)
            if std:
                ni = self.joints[idx]
                swt = self._skin_wt.get(ni)
                if swt is not None:
                    model_tpose[std] = swt[:3, 3].copy()

        # Scale: canonical → model
        canon_hips = _TPOSE.get('Hips', np.zeros(3))
        canon_head = _TPOSE.get('Head', np.array([0, 1.55, 0]))
        model_hips = model_tpose.get('Hips', np.zeros(3))
        model_head = model_tpose.get('Head', np.array([0, 1.37, 0]))
        canon_h = np.linalg.norm(canon_head - canon_hips)
        model_h = np.linalg.norm(model_head - model_hips)
        h_scale = model_h / canon_h if canon_h > 1e-6 else 1.0

        # Transform posed joints into model skin space
        model_posed = {}
        for name, pos in posed_joints.items():
            rel = pos - canon_hips
            model_posed[name] = model_hips + T @ rel * h_scale

        # Initialize posed world transforms from T-pose
        posed_world = {}
        for idx in range(nj):
            ni = self.joints[idx]
            posed_world[idx] = self._skin_wt.get(ni, np.eye(4)).copy()

        def _get_descendants(idx):
            desc = []
            q = []
            ni = self.joints[idx]
            for c in self._children.get(ni, []):
                if c in self.joints:
                    q.append(self.joints.index(c))
            while q:
                ci = q.pop(0)
                desc.append(ci)
                cni = self.joints[ci]
                for c in self._children.get(cni, []):
                    if c in self.joints:
                        q.append(self.joints.index(c))
            return desc

        # BFS FK swing rotations
        processed = set()
        queue = []
        for idx in range(nj):
            ni = self.joints[idx]
            pni = self._parent.get(ni)
            if pni is None or pni not in self.joints:
                queue.append(idx)

        while queue:
            idx = queue.pop(0)
            if idx in processed:
                continue
            processed.add(idx)
            ni = self.joints[idx]

            std_name = self._joint_to_std.get(idx)
            child_std = self._find_child_std(idx) if std_name else None

            if std_name and child_std:
                child_idx = self._std_to_idx.get(child_std)
                if (child_idx is not None and
                        std_name in model_posed and child_std in model_posed):
                    cur_dir = (posed_world[child_idx][:3, 3] -
                               posed_world[idx][:3, 3])
                    des_dir = model_posed[child_std] - model_posed[std_name]
                    cur_len = np.linalg.norm(cur_dir)
                    des_len = np.linalg.norm(des_dir)

                    if cur_len > 1e-6 and des_len > 1e-6:
                        R = _rotation_between(cur_dir / cur_len, des_dir / des_len)
                        pivot = posed_world[idx][:3, 3].copy()
                        descendants = _get_descendants(idx)
                        for di in [idx] + descendants:
                            posed_world[di][:3, :3] = R @ posed_world[di][:3, :3]
                            rel = posed_world[di][:3, 3] - pivot
                            posed_world[di][:3, 3] = pivot + R @ rel

            for child in self._children.get(ni, []):
                if child in self.joints:
                    cidx = self.joints.index(child)
                    if cidx not in processed:
                        queue.append(cidx)

        skinning_mats = np.zeros((nj, 4, 4))
        for idx in range(nj):
            skinning_mats[idx] = posed_world[idx] @ self.inv_bind_mats[idx]
        return skinning_mats

    def _apply_lbs(self, verts, j4, w4, skinning_mats):
        N = len(verts)
        v_homo = np.ones((N, 4), dtype=np.float64)
        v_homo[:, :3] = verts
        result = np.zeros((N, 3), dtype=np.float64)
        n_joints = len(skinning_mats)

        for influence in range(4):
            ji = j4[:, influence]
            wi = w4[:, influence]
            mask = wi > 0.001
            if not mask.any():
                continue
            valid_ji = np.clip(ji[mask], 0, n_joints - 1)
            valid_w = wi[mask]
            valid_v = v_homo[mask]
            for j in np.unique(valid_ji):
                jmask = valid_ji == j
                verts_j = valid_v[jmask]
                w_j = valid_w[jmask]
                M = skinning_mats[j]
                transformed = (M @ verts_j.T).T[:, :3]
                orig_indices = np.where(mask)[0][jmask]
                result[orig_indices] += transformed * w_j[:, np.newaxis]
        return result

    def render_frame(self, posed_joints):
        """Render one frame given canonical posed joint positions.

        Args:
            posed_joints: dict {joint_name: np.array([x,y,z])} or None for T-pose
        """
        new_positions = None

        if posed_joints is not None:
            skinning_mats = self.compute_skinning_matrices(posed_joints)
            R = self._R180[:3, :3]
            new_positions = {}
            for gi, name in enumerate(self._gnames):
                if gi >= len(self._skin_data) or self._skin_data[gi] is None:
                    continue
                j4, w4 = self._skin_data[gi]
                orig = self._orig_verts[name]
                if len(j4) != len(orig):
                    continue
                deformed = self._apply_lbs(orig, j4, w4, skinning_mats)
                new_positions[gi] = (deformed @ R.T).astype(np.float32)

        # Build scene
        scene = pyrender.Scene(bg_color=[0.06, 0.06, 0.10, 1.0],
                               ambient_light=[0.4, 0.4, 0.4])
        for gi, name in enumerate(self._gnames):
            pd = self._prim_cache[gi]
            if pd is None:
                continue
            prims = []
            for pdata in pd:
                pos = (new_positions[gi]
                       if new_positions and gi in new_positions
                       else None)
                if pos is None:
                    g = self._tmesh.geometry[name].copy()
                    g.apply_transform(self._R180)
                    pos = g.vertices.astype(np.float32)
                prims.append(pyrender.Primitive(
                    positions=pos, indices=pdata['idx'], normals=pdata.get('nrm'),
                    texcoord_0=pdata.get('uv'), color_0=pdata.get('col'),
                    material=pdata['mat'], mode=4))
            scene.add(pyrender.Mesh(primitives=prims))

        # Camera — auto-frame character
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 7.5,
                                         aspectRatio=self.W / self.H)
        cp = np.eye(4)
        cp[1, 3] = 1.0    # look at chest height
        cp[2, 3] = 4.0    # distance
        scene.add(cam, pose=cp)

        # Key light (warm, from upper right)
        kl = pyrender.DirectionalLight(color=[1.0, 0.97, 0.92], intensity=4.5)
        kp = np.eye(4)
        c1, s1 = np.cos(-0.5), np.sin(-0.5)
        c2, s2 = np.cos(0.3), np.sin(0.3)
        kp[:3, :3] = (np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]]) @
                       np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]]))
        scene.add(kl, pose=kp)

        # Fill light (cool, from left)
        fl = pyrender.DirectionalLight(color=[0.7, 0.8, 0.95], intensity=2.0)
        fp = np.eye(4)
        c3, s3 = np.cos(-0.3), np.sin(-0.3)
        c4, s4 = np.cos(-0.4), np.sin(-0.4)
        fp[:3, :3] = (np.array([[c4,0,s4],[0,1,0],[-s4,0,c4]]) @
                       np.array([[1,0,0],[0,c3,-s3],[0,s3,c3]]))
        scene.add(fl, pose=fp)

        # Rim light (from behind)
        rl = pyrender.DirectionalLight(color=[0.5, 0.5, 0.6], intensity=1.5)
        rp = np.eye(4)
        rp[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        scene.add(rl, pose=rp)

        color, _ = self._renderer.render(scene)
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def cleanup(self):
        self._renderer.delete()


# ═══════════════════════════════════════════════════════════════
#  MediaPipe 3D Extraction
# ═══════════════════════════════════════════════════════════════

def extract_3d_landmarks(video_path, start_frame=0, max_frames=None):
    """Extract 3D world landmarks from video using MediaPipe Pose."""
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total} frames, {fps:.1f} fps, {vw}x{vh}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = min(total, start_frame + max_frames) if max_frames else total

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'models', 'pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    all_kps = []
    input_frames = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((frame_idx / fps) * 1000)
        results = landmarker.detect_for_video(mp_image, ts)

        kps = np.zeros((33, 3), dtype=np.float32)
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            for i, lm in enumerate(results.pose_world_landmarks[0]):
                kps[i] = [lm.x, lm.y, lm.z]
        all_kps.append(kps)
        input_frames.append(cv2.resize(frame, (WIDTH, HEIGHT)))

        if (frame_idx - start_frame) % 100 == 0:
            det = "YES" if results.pose_world_landmarks else "NO"
            print(f"  Frame {frame_idx - start_frame}/{end_frame - start_frame} - pose: {det}")
        frame_idx += 1

    landmarker.close()
    cap.release()
    return np.array(all_kps), input_frames


def smooth_landmarks(kps, window=5):
    s = kps.copy()
    for j in range(kps.shape[1]):
        for c in range(kps.shape[2]):
            s[:, j, c] = uniform_filter1d(kps[:, j, c], size=window, mode='nearest')
    return s


# ═══════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Step 1: Extract 3D landmarks ──
    print("=" * 60)
    print("STEP 1: Extracting 3D pose landmarks...")
    print("=" * 60)
    all_kps, input_frames = extract_3d_landmarks(INPUT_VIDEO, START_FRAME, MAX_FRAMES)
    all_kps = smooth_landmarks(all_kps, window=5)
    n_frames = len(all_kps)
    print(f"  {n_frames} frames extracted in {time.time()-t0:.1f}s")

    # ── Step 2: Initialize VRM renderer ──
    print("\n" + "=" * 60)
    print("STEP 2: Loading VRM character model...")
    print("=" * 60)
    renderer = VRMCharacterRenderer(VRM_MODEL, WIDTH, HEIGHT)

    # ── Step 3: Convert landmarks to posed joints ──
    print("\n" + "=" * 60)
    print("STEP 3: Converting poses and rendering frames...")
    print("=" * 60)

    avatar_frames = []
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4]

    for i in range(n_frames):
        mp_lms = all_kps[i]

        # Check valid pose
        hip_c = (mp_lms[MP_L_HIP] + mp_lms[MP_R_HIP]) / 2
        sh_c  = (mp_lms[MP_L_SHOULDER] + mp_lms[MP_R_SHOULDER]) / 2
        valid = np.linalg.norm(sh_c - hip_c) > 0.05

        if valid:
            posed = mp_to_posed_joints(mp_lms)
            if posed is not None:
                frame = renderer.render_frame(posed)
            else:
                frame = np.full((HEIGHT, WIDTH, 3), [15, 15, 25], dtype=np.uint8)
        else:
            frame = np.full((HEIGHT, WIDTH, 3), [15, 15, 25], dtype=np.uint8)

        avatar_frames.append(frame)

        if i % 20 == 0:
            print(f"  Frame {i}/{n_frames} ({time.time()-t0:.1f}s)")

        if i in sample_indices:
            sbs = np.hstack([input_frames[i], frame])
            cv2.putText(sbs, "Input", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(sbs, "VRM Character", (WIDTH + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(OUT_DIR, f'vrm_transfer_{i:04d}.png'), sbs)

    renderer.cleanup()

    # ── Step 4: Write videos ──
    print("\n" + "=" * 60)
    print("STEP 4: Writing output videos...")
    print("=" * 60)

    sbs_path = os.path.join(OUT_DIR, 'motion_transfer_vrm_wab.mp4')
    avatar_path = os.path.join(OUT_DIR, 'avatar_dance_vrm_wab.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    sbs_w = cv2.VideoWriter(sbs_path, fourcc, FPS, (WIDTH * 2, HEIGHT))
    av_w  = cv2.VideoWriter(avatar_path, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(n_frames):
        sbs = np.hstack([input_frames[i], avatar_frames[i]])
        cv2.putText(sbs, "Input", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(sbs, "VRM Character", (WIDTH + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        sbs_w.write(sbs)
        av_w.write(avatar_frames[i])

    sbs_w.release()
    av_w.release()

    elapsed = time.time() - t0
    print(f"\nDone! ({elapsed:.1f}s total)")
    print(f"  Side-by-side: {sbs_path}")
    print(f"  Avatar only:  {avatar_path}")
    print(f"  Previews:     {OUT_DIR}/vrm_transfer_*.png")


if __name__ == '__main__':
    main()
