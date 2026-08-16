"""
3D Mesh character style — render GLTF/GLB/OBJ models driven by keypoints.

Load characters from Mixamo, Ready Player Me, VRoid, Sketchfab, etc.
Requires: pip install trimesh pyrender pyglet

Falls back gracefully if trimesh/pyrender are not installed.

Workflow:
  1. Download character from Mixamo → Export as FBX
  2. Convert FBX → GLTF/GLB (use https://github.com/facebookincubator/FBX2glTF
     or online at https://products.aspose.app/3d/conversion/fbx-to-glb)
  3. Place the .glb file in data/characters/
  4. Select mesh3d style in the UI and pick the character

The renderer maps COCO-WholeBody keypoints to joint rotations and
renders the 3D model to a 2D image using offscreen rendering.
Uses analytical IK to lift 2D keypoints → 3D joint positions → bone
rotations, then transforms individual body-part meshes accordingly.
"""

import os
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependency check ──
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


# ── COCO → Mixamo skeleton mapping ──

COCO_TO_BONE_MAP = {
    0:  "Head",
    5:  "LeftArm",
    6:  "RightArm",
    7:  "LeftForeArm",
    8:  "RightForeArm",
    9:  "LeftHand",
    10: "RightHand",
    11: "LeftUpLeg",
    12: "RightUpLeg",
    13: "LeftLeg",
    14: "RightLeg",
    15: "LeftFoot",
    16: "RightFoot",
}

# Bone chains for computing angles
BONE_CHAINS = {
    "left_arm":  [(5, 7), (7, 9)],
    "right_arm": [(6, 8), (8, 10)],
    "left_leg":  [(11, 13), (13, 15)],
    "right_leg": [(12, 14), (14, 16)],
    "spine":     [(11, 5)],
}

# Default character search paths
DEFAULT_CHAR_DIRS = [
    "data/characters",
    "data/models",
    "assets/characters",
]

# T-pose skeleton definition (Y-up, matches rigged_human.glb)
_TPOSE_JOINTS = {
    'Hips':          np.array([0, 0.95, 0]),
    'Spine':         np.array([0, 1.05, 0]),
    'Spine2':        np.array([0, 1.25, 0]),
    'Neck':          np.array([0, 1.4, 0]),
    'Head':          np.array([0, 1.55, 0]),
    'LeftArm':       np.array([-0.22, 1.38, 0]),
    'LeftForeArm':   np.array([-0.48, 1.38, 0]),
    'LeftHand':      np.array([-0.72, 1.38, 0]),
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

# Map COCO keypoint indices to T-pose joint names
_COCO_TO_JOINT = {
    0: 'Head', 5: 'LeftArm', 6: 'RightArm',
    7: 'LeftForeArm', 8: 'RightForeArm',
    9: 'LeftHand', 10: 'RightHand',
    11: 'LeftUpLeg', 12: 'RightUpLeg',
    13: 'LeftLeg', 14: 'RightLeg',
    15: 'LeftFoot', 16: 'RightFoot',
}

# Skeleton hierarchy: bone_name → (parent_joint, child_joint, geometry_names)
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


# ── IK utilities ──

def _rotation_between(v1, v2):
    """Compute 3x3 rotation matrix that rotates unit vector v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180 degree rotation — pick arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    R = np.eye(3) + skew + skew @ skew / (1 + dot)
    return R


def _lift_2d_to_3d(kps_2d, canvas_w, canvas_h):
    """Lift 2D COCO keypoints to 3D using bone-length constraints.

    Maps pixel coordinates to normalized 3D space where the character
    is ~1.8 units tall, centered at hips. Uses heuristic depth estimation
    based on limb foreshortening.
    """
    kps = kps_2d[:17].copy()  # only body keypoints

    # Normalize to [-1, 1] range centered on hips
    hip_center = (kps[11] + kps[12]) / 2
    shoulder_center = (kps[5] + kps[6]) / 2

    # Scale factor: shoulder-to-hip distance should be ~0.46 in our skeleton
    torso_px = np.linalg.norm(shoulder_center - hip_center)
    if torso_px < 5:
        return None
    scale = 0.46 / torso_px

    # Center and scale
    pts = (kps - hip_center) * scale
    # Flip Y (screen Y is down, 3D Y is up)
    pts[:, 1] = -pts[:, 1]

    # Offset so hips are at skeleton's hip height
    pts[:, 0] += _TPOSE_JOINTS['Hips'][0]
    pts[:, 1] += _TPOSE_JOINTS['Hips'][1]

    # Create 3D points with depth estimation
    pts3d = np.zeros((17, 3))
    pts3d[:, 0] = pts[:, 0]  # X from 2D
    pts3d[:, 1] = pts[:, 1]  # Y from 2D (flipped)

    # Estimate depth (Z) from bone foreshortening
    # Reference bone lengths from T-pose
    ref_lengths = {}
    for coco_pair, joint_pair in [
        ((5, 7), ('LeftArm', 'LeftForeArm')),
        ((7, 9), ('LeftForeArm', 'LeftHand')),
        ((6, 8), ('RightArm', 'RightForeArm')),
        ((8, 10), ('RightForeArm', 'RightHand')),
        ((11, 13), ('LeftUpLeg', 'LeftLeg')),
        ((13, 15), ('LeftLeg', 'LeftFoot')),
        ((12, 14), ('RightUpLeg', 'RightLeg')),
        ((14, 16), ('RightLeg', 'RightFoot')),
    ]:
        ref_lengths[coco_pair] = np.linalg.norm(
            _TPOSE_JOINTS[joint_pair[1]] - _TPOSE_JOINTS[joint_pair[0]]
        )

    # Solve Z for each limb joint using bone length constraint
    # |p_child - p_parent|_3d = ref_length
    # We know x,y; solve for z: z = sqrt(L^2 - dx^2 - dy^2)
    def _solve_depth(parent_idx, child_idx, ref_len, parent_z=0.0):
        dx = pts3d[child_idx, 0] - pts3d[parent_idx, 0]
        dy = pts3d[child_idx, 1] - pts3d[parent_idx, 1]
        d2 = dx * dx + dy * dy
        l2 = ref_len * ref_len
        if d2 >= l2:
            # Bone appears longer than reference → fully in-plane
            return parent_z
        dz = np.sqrt(l2 - d2)
        # Heuristic: arms come forward, legs go slightly back
        return parent_z + dz * 0.3  # bias forward slightly

    # Hips/spine roughly at z=0
    pts3d[11, 2] = 0
    pts3d[12, 2] = 0
    pts3d[5, 2] = 0
    pts3d[6, 2] = 0
    pts3d[0, 2] = 0  # head

    # Left arm chain
    pts3d[7, 2] = _solve_depth(5, 7, ref_lengths[(5, 7)], pts3d[5, 2])
    pts3d[9, 2] = _solve_depth(7, 9, ref_lengths[(7, 9)], pts3d[7, 2])

    # Right arm chain
    pts3d[8, 2] = _solve_depth(6, 8, ref_lengths[(6, 8)], pts3d[6, 2])
    pts3d[10, 2] = _solve_depth(8, 10, ref_lengths[(8, 10)], pts3d[8, 2])

    # Left leg chain
    pts3d[13, 2] = _solve_depth(11, 13, ref_lengths[(11, 13)], pts3d[11, 2])
    pts3d[15, 2] = _solve_depth(13, 15, ref_lengths[(13, 15)], pts3d[13, 2])

    # Right leg chain
    pts3d[14, 2] = _solve_depth(12, 14, ref_lengths[(12, 14)], pts3d[12, 2])
    pts3d[16, 2] = _solve_depth(14, 16, ref_lengths[(14, 16)], pts3d[14, 2])

    return pts3d


def _compute_posed_joints(pts3d):
    """Compute posed joint positions from lifted 3D keypoints.

    Returns dict mapping joint name → 3D position.
    """
    posed = {}
    # Map COCO indices to our joint names
    for coco_idx, joint_name in _COCO_TO_JOINT.items():
        posed[joint_name] = pts3d[coco_idx].copy()

    # Derive spine joints from hip and shoulder positions
    hip_center = (pts3d[11] + pts3d[12]) / 2
    shoulder_center = (pts3d[5] + pts3d[6]) / 2
    posed['Hips'] = hip_center
    posed['Spine'] = hip_center + (shoulder_center - hip_center) * 0.22
    posed['Spine2'] = hip_center + (shoulder_center - hip_center) * 0.67
    posed['Neck'] = shoulder_center

    return posed


def _compute_bone_transform(tpose_parent, tpose_child, posed_parent, posed_child):
    """Compute 4x4 transform for a bone from T-pose to posed position.

    Returns a transformation that when applied to geometry vertices
    (which are in T-pose world space), moves them to the posed position.
    """
    # T-pose bone direction
    tpose_dir = tpose_child - tpose_parent
    tpose_len = np.linalg.norm(tpose_dir)
    if tpose_len < 1e-6:
        tpose_dir = np.array([0, 1, 0])
    else:
        tpose_dir = tpose_dir / tpose_len

    # Posed bone direction
    posed_dir = posed_child - posed_parent
    posed_len = np.linalg.norm(posed_dir)
    if posed_len < 1e-6:
        posed_dir = tpose_dir
    else:
        posed_dir = posed_dir / posed_len

    # Rotation from T-pose direction to posed direction
    R = _rotation_between(tpose_dir, posed_dir)

    # Build 4x4: translate to origin at tpose_parent, rotate, translate to posed_parent
    tpose_mid = (tpose_parent + tpose_child) / 2
    posed_mid = (posed_parent + posed_child) / 2

    T = np.eye(4)
    T[:3, :3] = R
    # Transform: v' = R @ (v - tpose_mid) + posed_mid
    T[:3, 3] = posed_mid - R @ tpose_mid

    return T


def list_available_models() -> list[dict]:
    """Scan character directories for loadable 3D models."""
    models = []
    for d in DEFAULT_CHAR_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            ext = Path(f).suffix.lower()
            if ext in (".glb", ".gltf", ".obj"):
                models.append({
                    "name": Path(f).stem,
                    "path": os.path.join(d, f),
                    "format": ext[1:],
                })
    return models


class Mesh3DStyle:
    """Render a 3D character model from GLTF/GLB/OBJ driven by 2D keypoints.

    Uses analytical IK to lift 2D keypoints to 3D, compute per-bone
    rotations, transform mesh geometry, and render with pyrender.

    If trimesh/pyrender are not installed, falls back to a simple
    wireframe 3D-like rendering.
    """

    def __init__(self, config: dict):
        style_cfg = config.get("avatar", {}).get("styles", {}).get("mesh3d", {})
        self.model_path = style_cfg.get("model_path", "")
        self.bg_color = tuple(style_cfg.get("bg_color", [0, 0, 0]))
        self.light_intensity = style_cfg.get("light_intensity", 3.0)
        self.camera_distance = style_cfg.get("camera_distance", 2.5)

        self._mesh = None
        self._tpose_geoms = {}  # name → trimesh geometry in T-pose
        self._loaded = False

        # Try loading skeleton definition alongside the model
        self._skeleton_path = ""
        if self.model_path:
            skel_path = os.path.splitext(self.model_path)[0] + "_skeleton.json"
            if os.path.isfile(skel_path):
                self._skeleton_path = skel_path

        if self.model_path and os.path.isfile(self.model_path):
            self._try_load_model()

    def _try_load_model(self):
        """Attempt to load the 3D model and cache T-pose geometry."""
        if not _HAS_TRIMESH:
            logger.warning("trimesh not installed — pip install trimesh pyrender pyglet")
            return

        try:
            self._mesh = trimesh.load(self.model_path)
            self._loaded = True
            logger.info("Loaded 3D model: %s", self.model_path)

            # Cache T-pose geometry copies for per-frame posing
            if isinstance(self._mesh, trimesh.Scene):
                for name, geom in self._mesh.geometry.items():
                    self._tpose_geoms[name] = geom.copy()
            else:
                self._tpose_geoms['mesh'] = self._mesh.copy()

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_path, e)

    def render(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        min_score: float = 0.3,
    ) -> np.ndarray:
        """Render the 3D model onto canvas, posed by 2D keypoints."""
        if self._loaded and _HAS_PYRENDER:
            return self._render_3d(canvas, keypoints, scores, min_score)
        else:
            return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

    def _render_3d(self, canvas, keypoints, scores, min_score):
        """Full 3D rendering with IK-driven posing and pyrender."""
        K = len(keypoints)
        if K < 17:
            return canvas

        h, w = canvas.shape[:2]

        try:
            # Step 1: Lift 2D keypoints to 3D
            pts3d = _lift_2d_to_3d(keypoints, w, h)
            if pts3d is None:
                return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

            # Step 2: Compute posed joint positions
            posed = _compute_posed_joints(pts3d)

            # Step 3: Compute per-bone transforms and apply to geometry
            posed_geoms = self._apply_pose(posed)

            # Step 4: Build pyrender scene and render
            scene = pyrender.Scene(
                bg_color=[self.bg_color[2] / 255, self.bg_color[1] / 255,
                          self.bg_color[0] / 255, 0.0],
                ambient_light=[0.3, 0.3, 0.3],
            )

            for geom in posed_geoms:
                mesh = pyrender.Mesh.from_trimesh(geom)
                scene.add(mesh)

            # Camera looking at character center
            hip_pos = posed.get('Hips', np.array([0, 0.95, 0]))
            head_pos = posed.get('Head', np.array([0, 1.55, 0]))
            look_at = (hip_pos + head_pos) / 2

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            cam_pose = np.eye(4)
            cam_pose[0, 3] = look_at[0]
            cam_pose[1, 3] = look_at[1]
            cam_pose[2, 3] = look_at[2] + self.camera_distance
            scene.add(camera, pose=cam_pose)

            # Key light (front-above)
            light = pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0],
                intensity=self.light_intensity
            )
            light_pose = np.eye(4)
            light_pose[:3, :3] = _rotation_between(
                np.array([0, 0, -1]),
                np.array([-0.2, -0.3, -1.0])
            )
            light_pose[:3, 3] = cam_pose[:3, 3]
            scene.add(light, pose=light_pose)

            # Fill light (side)
            fill = pyrender.DirectionalLight(
                color=[0.7, 0.8, 1.0],
                intensity=self.light_intensity * 0.4
            )
            fill_pose = np.eye(4)
            fill_pose[:3, :3] = _rotation_between(
                np.array([0, 0, -1]),
                np.array([0.5, -0.1, -1.0])
            )
            fill_pose[:3, 3] = cam_pose[:3, 3]
            scene.add(fill, pose=fill_pose)

            # Render offscreen
            r = pyrender.OffscreenRenderer(w, h)
            color, _ = r.render(scene)
            r.delete()

            rendered = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Composite onto canvas (non-black pixels)
            gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
            mask = gray > 5
            canvas[mask] = rendered[mask]

            return canvas

        except Exception as e:
            logger.error("3D rendering failed: %s — falling back to wireframe", e)
            import traceback
            traceback.print_exc()
            return self._render_wireframe_3d(canvas, keypoints, scores, min_score)

    def _apply_pose(self, posed_joints):
        """Apply IK-derived pose to T-pose geometry, return list of posed trimesh objects."""
        result = []

        for bone_name, (parent_joint, child_joint, geom_names) in _BONE_DEFS.items():
            tpose_parent = _TPOSE_JOINTS.get(parent_joint)
            tpose_child = _TPOSE_JOINTS.get(child_joint)
            posed_parent = posed_joints.get(parent_joint)
            posed_child = posed_joints.get(child_joint)

            if tpose_parent is None or tpose_child is None:
                continue
            if posed_parent is None or posed_child is None:
                continue

            # For single-joint bones (head, hands, feet), use parent position only
            if parent_joint == child_joint:
                T = np.eye(4)
                T[:3, 3] = posed_parent - tpose_parent
            else:
                T = _compute_bone_transform(
                    tpose_parent, tpose_child,
                    posed_parent, posed_child
                )

            for geom_name in geom_names:
                if geom_name not in self._tpose_geoms:
                    continue
                g = self._tpose_geoms[geom_name].copy()
                g.apply_transform(T)
                result.append(g)

        return result

    def _render_wireframe_3d(self, canvas, keypoints, scores, min_score):
        """Fallback: 3D-like wireframe with depth shading.

        Simulates a 3D mesh look by drawing body segments with
        depth-based shading and cross-hatching to suggest volume.
        """
        K = len(keypoints)
        if K < 17:
            return canvas

        kps = keypoints.astype(np.float64)
        torso_w = np.linalg.norm(kps[5] - kps[6])
        if torso_w < 3:
            return canvas

        # Compute depth from body proportions (shoulder width as proxy)
        base_color = np.array([180, 200, 220], dtype=np.float32)  # blueish steel

        # Body segments with simulated depth
        segments = [
            (5, 7, 0.18), (7, 9, 0.14),     # left arm
            (6, 8, 0.18), (8, 10, 0.14),     # right arm
            (11, 13, 0.24), (13, 15, 0.18),  # left leg
            (12, 14, 0.24), (14, 16, 0.18),  # right leg
        ]

        # Draw torso as 3D box-like shape
        pts_3d = [kps[5], kps[6], kps[12], kps[11]]
        pts = np.array(pts_3d, dtype=np.int32)
        # Front face
        cv2.fillPoly(canvas, [pts], tuple(base_color.astype(int).tolist()), cv2.LINE_AA)
        # Edge highlights
        cv2.polylines(canvas, [pts], True,
                      tuple((base_color * 1.3).clip(0, 255).astype(int).tolist()),
                      2, cv2.LINE_AA)

        # Cross-hatch on torso for "mesh" look
        mid_x = int((kps[5][0] + kps[6][0]) / 2)
        mid_y_top = int((kps[5][1] + kps[6][1]) / 2)
        mid_y_bot = int((kps[11][1] + kps[12][1]) / 2)
        step = max(int(torso_w * 0.12), 4)
        hatch_color = tuple((base_color * 0.7).astype(int).tolist())
        for y in range(mid_y_top, mid_y_bot, step):
            cv2.line(canvas, (int(kps[5][0]), y), (int(kps[6][0]), y),
                     hatch_color, 1, cv2.LINE_AA)
        for x in range(int(kps[5][0]), int(kps[6][0]), step):
            cv2.line(canvas, (x, mid_y_top), (x, mid_y_bot),
                     hatch_color, 1, cv2.LINE_AA)

        # Draw limbs as 3D cylinders (tapered with highlight)
        for j1, j2, w_ratio in segments:
            if j1 >= K or j2 >= K:
                continue
            if scores is not None and (scores[j1] < min_score or scores[j2] < min_score):
                continue

            p1, p2 = kps[j1], kps[j2]
            length = np.linalg.norm(p2 - p1)
            if length < 2:
                continue

            d = p2 - p1
            perp = np.array([-d[1], d[0]]) / length
            w = max(int(torso_w * w_ratio), 4)

            # Main body
            poly = np.array([
                p1 + perp * w / 2,
                p2 + perp * w / 2,
                p2 - perp * w / 2,
                p1 - perp * w / 2,
            ], dtype=np.int32)
            cv2.fillConvexPoly(canvas, poly, tuple(base_color.astype(int).tolist()), cv2.LINE_AA)

            # Highlight stripe down center
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     tuple((base_color * 1.4).clip(0, 255).astype(int).tolist()),
                     max(w // 4, 1), cv2.LINE_AA)

            # Edge lines
            cv2.line(canvas, tuple((p1 + perp * w / 2).astype(int)),
                     tuple((p2 + perp * w / 2).astype(int)),
                     tuple((base_color * 0.6).astype(int).tolist()), 1, cv2.LINE_AA)
            cv2.line(canvas, tuple((p1 - perp * w / 2).astype(int)),
                     tuple((p2 - perp * w / 2).astype(int)),
                     tuple((base_color * 0.6).astype(int).tolist()), 1, cv2.LINE_AA)

        # Joints as spheres (circles with gradient)
        joint_r = max(int(torso_w * 0.06), 3)
        for idx in range(min(K, 17)):
            if scores is not None and scores[idx] < min_score:
                continue
            pt = tuple(kps[idx].astype(int))
            # Outer ring
            cv2.circle(canvas, pt, joint_r,
                       tuple((base_color * 0.8).astype(int).tolist()), -1, cv2.LINE_AA)
            # Highlight
            cv2.circle(canvas, (pt[0] - 1, pt[1] - 1), max(joint_r // 2, 1),
                       tuple((base_color * 1.5).clip(0, 255).astype(int).tolist()),
                       -1, cv2.LINE_AA)

        # Head
        head_r = max(int(torso_w * 0.22), 10)
        hc = tuple(kps[0].astype(int))
        cv2.circle(canvas, hc, head_r,
                   tuple(base_color.astype(int).tolist()), -1, cv2.LINE_AA)
        # Wireframe overlay on head
        for dy in range(-head_r, head_r + 1, max(head_r // 3, 3)):
            half_w = int(np.sqrt(max(head_r*head_r - dy*dy, 0)))
            cv2.line(canvas, (hc[0] - half_w, hc[1] + dy),
                     (hc[0] + half_w, hc[1] + dy),
                     hatch_color, 1, cv2.LINE_AA)
        # Edge
        cv2.circle(canvas, hc, head_r,
                   tuple((base_color * 1.3).clip(0, 255).astype(int).tolist()),
                   2, cv2.LINE_AA)

        # Hands
        hand_r = max(int(torso_w * 0.05), 2)
        for idx in [9, 10]:
            if idx >= K:
                continue
            pt = tuple(kps[idx].astype(int))
            cv2.circle(canvas, pt, hand_r,
                       tuple(base_color.astype(int).tolist()), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, hand_r,
                       tuple((base_color * 1.3).clip(0, 255).astype(int).tolist()),
                       1, cv2.LINE_AA)

        # Feet
        for idx in [15, 16]:
            if idx >= K:
                continue
            knee = 13 if idx == 15 else 14
            foot_dir = kps[idx] - kps[knee]
            angle = float(np.degrees(np.arctan2(foot_dir[1], foot_dir[0])))
            fw = max(int(torso_w * 0.12), 4)
            fh = max(int(torso_w * 0.06), 3)
            pt = tuple(kps[idx].astype(int))
            cv2.ellipse(canvas, pt, (fw, fh), angle + 90, 0, 360,
                        tuple(base_color.astype(int).tolist()), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, pt, (fw, fh), angle + 90, 0, 360,
                        tuple((base_color * 1.3).clip(0, 255).astype(int).tolist()),
                        1, cv2.LINE_AA)

        return canvas
