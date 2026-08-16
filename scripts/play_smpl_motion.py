"""
play_smpl_motion.py — render an SMPL-pose sequence on a VRM avatar via a
ROTATION-BASED retargeter (not the position/swing-IK path of v6).

This is the foundation of Path A (AIST++ playback) and Path B (EDGE music→
dance). Both produce SMPL pose params; this script consumes them.

Usage
-----
Smoke test (synthetic motion, no external data):
    python scripts/play_smpl_motion.py --test --vrm data/models/AvatarSample_B.vrm

Play an AIST++ .pkl on a VRM:
    python scripts/play_smpl_motion.py --aist <motion.pkl> --vrm <avatar.vrm>

Output:
    data/output_videos/smpl_<tag>.mp4
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402

# Heavy renderer imports (pyrender/trimesh)
from scripts.motion_transfer_v6 import VRMCharacterRendererV6  # noqa: E402

from src.smpl_lite import (  # noqa: E402
    SMPL_NAMES, SMPL_PARENTS, axis_angle_to_R, forward_kinematics,
    load_smpl_skeleton,
)


# ═══════════════════════════════════════════════════════════════
#  SMPL joint-index → VRM std bone-name mapping
# ═══════════════════════════════════════════════════════════════
SMPL_TO_VRM_STD: dict[int, str] = {
    0:  'Hips',
    1:  'LeftUpLeg',
    2:  'RightUpLeg',
    3:  'Spine',
    4:  'LeftLeg',
    5:  'RightLeg',
    6:  'Spine2',
    7:  'LeftFoot',
    8:  'RightFoot',
    # 9  Spine3 — most VRMs lack a 4th spine; fold it into Spine2 chain via FK.
    # 10 L_Foot toe — skip (no toe rotation in AIST++ usually anyway).
    # 11 R_Foot toe — skip.
    12: 'Neck',
    13: 'LeftShoulder',
    14: 'RightShoulder',
    15: 'Head',
    16: 'LeftArm',
    17: 'RightArm',
    18: 'LeftForeArm',
    19: 'RightForeArm',
    20: 'LeftHand',
    21: 'RightHand',
    # 22, 23 are SMPL "hand" tip joints — already covered by LeftHand / RightHand.
}


# ═══════════════════════════════════════════════════════════════
#  Coordinate-frame transform: SMPL world → VRM model world
# ═══════════════════════════════════════════════════════════════
# SMPL convention (verified empirically from rest_joints):
#   • +X = subject's anatomical LEFT  (L_Hip at +0.056, R_Hip at -0.062)
#   • +Y = up
#   • +Z = subject faces +Z direction
#
# Our VRM model frame (BEFORE the renderer's _R180 cosmetic flip):
#   • -X = subject's anatomical LEFT  (J_Bip_L_UpperArm at X=-0.081 in
#         AvatarSample_B; the L-prefixed bones live at -X side)
#   • +Y = up
#   • -Z = subject faces -Z direction
#
# These two frames differ by a 180° rotation around Y (a chirality-preserving
# flip). The transform matrix is M = diag(-1, 1, -1).
SMPL_TO_VRM_R = np.diag([-1.0, 1.0, -1.0]).astype(np.float64)


# ═══════════════════════════════════════════════════════════════
#  Renderer subclass with rotation-driven skinning
# ═══════════════════════════════════════════════════════════════
class VRMRendererSMPL(VRMCharacterRendererV6):
    """Adds `compute_skinning_matrices_from_world_R()` and a render method
    that consumes per-bone world rotations rather than joint positions."""

    # Bones in `world_R_by_std` will have their world rotation OVERRIDDEN.
    # Bones not in the dict inherit FK transforms from their parent
    # (i.e., they ride along with whatever ancestor was overridden).
    def compute_skinning_matrices_from_world_R(
            self,
            world_R_by_std: dict[str, np.ndarray],
            hip_translation_model: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build skinning matrices via rotation-driven FK on the VRM
        skeleton.

        Args:
            world_R_by_std: {VRM-std-bone-name: 3x3 world rotation matrix}.
                The rotation is treated as the WORLD-SPACE DELTA-FROM-REST
                of that bone (i.e., R_smpl_world from SMPL FK, which equals
                identity when SMPL pose is zero).
            hip_translation_model: (3,) translation added to the Hips joint
                in MODEL units (post _canon_to_model scale). None = no shift.

        Returns:
            (n_joints, 4, 4) skinning matrices.
        """
        nj = len(self.joints)
        # Topological order — root nodes first
        order = []
        seen = set()
        stack = []
        for idx in range(nj):
            ni = self.joints[idx]
            pni = self._parent.get(ni)
            if pni is None or pni not in self.joints:
                stack.append(idx)
        while stack:
            idx = stack.pop(0)
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
            ni = self.joints[idx]
            for c in self._children.get(ni, []):
                if c in self.joints and c not in [self.joints[i] for i in seen]:
                    ci = self.joints.index(c)
                    if ci not in seen:
                        stack.append(ci)

        posed_world: dict[int, np.ndarray] = {}

        for idx in order:
            ni = self.joints[idx]
            std = self._joint_to_std.get(idx)
            rest_w = self._skin_wt.get(ni, np.eye(4))
            rest_R = rest_w[:3, :3]
            rest_t = rest_w[:3, 3]

            pni = self._parent.get(ni)
            if pni in self.joints:
                p_idx = self.joints.index(pni)
                parent_posed = posed_world.get(p_idx, np.eye(4))
                parent_rest  = self._skin_wt.get(pni, np.eye(4))
            else:
                parent_posed = np.eye(4)
                parent_rest  = np.eye(4)

            # Rest local transform of THIS bone w.r.t. its parent:
            #   T_rest_local = parent_rest^-1 @ rest_w
            try:
                pr_inv = np.linalg.inv(parent_rest)
            except np.linalg.LinAlgError:
                pr_inv = np.eye(4)
            T_rest_local = pr_inv @ rest_w
            rest_local_R = T_rest_local[:3, :3]
            rest_local_t = T_rest_local[:3, 3]

            # Position: ALWAYS from FK chain (we never override positions —
            # bone lengths are the VRM's own; only rotations come from SMPL).
            new_t = parent_posed[:3, 3] + parent_posed[:3, :3] @ rest_local_t

            # Rotation: override if mapped, else inherit FK
            if std and std in world_R_by_std:
                R_smpl_world = world_R_by_std[std]
                # SMPL's rest world rotation per joint = identity. So
                # R_smpl_world is the world-space delta to apply on top of
                # the bone's REST world rotation:
                #     posed_R_world = R_smpl_world @ rest_R
                # This preserves the VRM's intrinsic bone-axis orientation
                # and applies the SMPL rotation in world coords.
                new_R = R_smpl_world @ rest_R
            else:
                new_R = parent_posed[:3, :3] @ rest_local_R

            T = np.eye(4)
            T[:3, :3] = new_R
            T[:3, 3]  = new_t

            # Hips translation (shift the root only)
            if std == 'Hips' and hip_translation_model is not None:
                T[:3, 3] = T[:3, 3] + np.asarray(hip_translation_model).reshape(3)

            posed_world[idx] = T

        # ── Skirt/coat reparent to Hips (inherited from v6) ──
        hips_idx = self._std_to_idx.get('Hips')
        if hips_idx is not None:
            T_hips_new  = posed_world[hips_idx]
            T_hips_rest = self._skin_wt.get(self.joints[hips_idx], np.eye(4))
            try:
                T_hips_rest_inv = np.linalg.inv(T_hips_rest)
            except np.linalg.LinAlgError:
                T_hips_rest_inv = None
            if T_hips_rest_inv is not None:
                hips_delta = T_hips_new @ T_hips_rest_inv
                for idx in range(nj):
                    ni = self.joints[idx]
                    name = self._gltf['nodes'][ni].get('name', '')
                    if not (name.startswith('J_Sec_') and
                            ('Skirt' in name or 'Coat' in name)):
                        continue
                    T_sec_rest = self._skin_wt.get(ni)
                    if T_sec_rest is None:
                        continue
                    posed_world[idx] = hips_delta @ T_sec_rest

        # Build skinning matrices
        skinning_mats = np.zeros((nj, 4, 4))
        for idx in range(nj):
            skinning_mats[idx] = posed_world.get(idx, np.eye(4)) @ self.inv_bind_mats[idx]
        self._last_posed_world = posed_world
        return skinning_mats


# ═══════════════════════════════════════════════════════════════
#  SMPL pose → per-VRM-bone world rotation
# ═══════════════════════════════════════════════════════════════
def smpl_pose_to_world_R(
        pose_aa: np.ndarray,
        rest_joints: np.ndarray,
        parents: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert one SMPL pose frame → (world_R_by_std_dict, joint_world_pos).

    Args:
        pose_aa : (24, 3) axis-angle local rotations.
        rest_joints : (24, 3) SMPL rest joint positions.
        parents : (24,) parent indices.

    Returns:
        world_R_by_std : {VRM-std-name: 3x3 world rotation matrix}
        joint_world_pos : (24, 3) world joint positions (for hip translation).
    """
    pose_aa = np.asarray(pose_aa, dtype=np.float64).reshape(24, 3)
    pos, joint_R, _ = forward_kinematics(rest_joints, parents, pose_aa)

    out: dict[str, np.ndarray] = {}
    for smpl_idx, vrm_std in SMPL_TO_VRM_STD.items():
        # Conjugate by SMPL_TO_VRM_R if a coordinate flip is needed.
        Rw = SMPL_TO_VRM_R @ joint_R[smpl_idx] @ SMPL_TO_VRM_R.T
        out[vrm_std] = Rw
    return out, pos


# ═══════════════════════════════════════════════════════════════
#  Render an SMPL motion
# ═══════════════════════════════════════════════════════════════
def render_smpl_clip(
        pose_aa_seq: np.ndarray,    # (T, 24, 3)
        trans_seq:   np.ndarray,    # (T, 3)
        vrm_path:    str,
        out_mp4:     str,
        fps:         float = 30.0,
        smpl_pkl:    str = 'data/models/smpl_raw/smpl/models/'
                            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        camera_yaw:  float = 0.0,
        log_every:   int = 30,
):
    """Render an SMPL pose sequence on a VRM avatar."""
    print(f"  Loading SMPL skeleton: {smpl_pkl}")
    sk = load_smpl_skeleton(smpl_pkl)
    rest_joints = sk['rest_joints']
    parents     = sk['parents']

    print(f"  Loading VRM: {vrm_path}")
    R = VRMRendererSMPL(vrm_path, w=720, h=1280)

    # Compute MODEL-units / SMPL-units height scale.
    # SMPL height = head Y - foot Y (rest); VRM height from _skin_wt.
    smpl_h = float(rest_joints[15, 1] - rest_joints[10, 1])  # head − L_Foot toe
    if smpl_h < 1e-3:
        smpl_h = float(rest_joints[15, 1] - rest_joints[7, 1])  # head − ankle
    m_head = m_hips = None
    for idx in range(len(R.joints)):
        std = R._joint_to_std.get(idx)
        if std == 'Head':
            m_head = R._skin_wt.get(R.joints[idx], np.eye(4))[:3, 3]
        if std == 'Hips':
            m_hips = R._skin_wt.get(R.joints[idx], np.eye(4))[:3, 3]
    canon_h = (np.linalg.norm(m_head - m_hips) * 1.55) if (m_head is not None) else 1.0
    h_scale = canon_h / smpl_h if smpl_h > 1e-3 else 1.0
    print(f"  SMPL height: {smpl_h:.3f}, target VRM height: {canon_h:.3f}, scale: {h_scale:.3f}")

    # Reference pelvis position from SMPL rest (for trans normalization)
    smpl_rest_pelvis = rest_joints[0]

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (720, 1280))

    n_frames = len(pose_aa_seq)
    print(f"  Rendering {n_frames} frames -> {out_mp4}")
    t0 = time.time()
    for fi in range(n_frames):
        pose_aa = pose_aa_seq[fi]
        trans   = trans_seq[fi] if trans_seq is not None else None

        world_R_by_std, joint_pos = smpl_pose_to_world_R(
            pose_aa, rest_joints, parents)

        # Hip translation in MODEL units. SMPL trans is in METERS shifting
        # the pelvis. Subtract the rest pelvis position to make it relative,
        # rotate by SMPL_TO_VRM_R, scale to model.
        if trans is not None:
            t_rel = (trans - smpl_rest_pelvis * 0)  # AIST++ trans is absolute pelvis world pos
            hip_translation_model = SMPL_TO_VRM_R @ (t_rel * h_scale)
            # Subtract first-frame trans so character starts at origin
            if fi == 0:
                render_smpl_clip._t0 = hip_translation_model.copy()
            hip_translation_model = hip_translation_model - render_smpl_clip._t0
        else:
            hip_translation_model = None

        skinning_mats = R.compute_skinning_matrices_from_world_R(
            world_R_by_std, hip_translation_model)

        # Render via the v6 path. We bypass posed_joints by using the stashed
        # _last_posed_world directly: render_frame_v2 calls
        # compute_skinning_matrices_v2(posed_joints) — we need a render path
        # that uses our pre-computed skinning_mats. Inline a minimal render:
        frame = _render_with_skinning(R, skinning_mats,
                                      world_offset=np.zeros(3),
                                      cam_target_xy=None,
                                      cam_yaw=camera_yaw)
        writer.write(frame)

        if (fi + 1) % log_every == 0 or fi == n_frames - 1:
            dt = time.time() - t0
            print(f"    frame {fi+1}/{n_frames}  ({(fi+1)/dt:.1f} fps render)")

    writer.release()
    print(f"  [OK] Wrote {out_mp4}")


def _render_with_skinning(R: VRMRendererSMPL,
                          skinning_mats: np.ndarray,
                          world_offset: np.ndarray,
                          cam_target_xy,
                          cam_yaw: float):
    """Mirror of render_frame_v2 but using PRE-COMPUTED skinning_mats."""
    import pyrender
    import trimesh
    R180 = R._R180[:3, :3]

    # Apply LBS per mesh
    new_positions = {}
    for gi, name in enumerate(R._gnames):
        if gi >= len(R._skin_data) or R._skin_data[gi] is None:
            continue
        j4, w4 = R._skin_data[gi]
        orig = R._orig_verts[name]
        if len(j4) != len(orig):
            continue
        deformed = R._apply_lbs(orig, j4, w4, skinning_mats)
        deformed_render = (deformed @ R180.T) + (R180 @ world_offset)
        new_positions[gi] = deformed_render.astype(np.float32)

    scene = pyrender.Scene(bg_color=[0.06, 0.06, 0.10, 1.0],
                           ambient_light=[0.4, 0.4, 0.4])
    for gi, name in enumerate(R._gnames):
        pd = R._prim_cache[gi]
        if pd is None:
            continue
        prims = []
        for pdata in pd:
            pos = new_positions.get(gi)
            if pos is None:
                g = R._tmesh.geometry[name].copy()
                g.apply_transform(R._R180)
                pos = g.vertices.astype(np.float32)
            prims.append(pyrender.Primitive(
                positions=pos, indices=pdata['idx'], normals=pdata.get('nrm'),
                texcoord_0=pdata.get('uv'), color_0=pdata.get('col'),
                material=pdata['mat'], mode=4))
        scene.add(pyrender.Mesh(primitives=prims))

    # Ground
    try:
        ground = trimesh.creation.box(extents=(10.0, 0.04, 8.0))
        ground.apply_translation([0, -0.02, -4.3])
        ground.visual.face_colors = [45, 48, 60, 255]
        scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))
    except Exception:
        pass

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 5.5,
                                     aspectRatio=R.W / R.H)
    cp = np.eye(4)
    cx = float(world_offset[0])
    cy = 0.95
    dist = 4.2
    if abs(cam_yaw) > 1e-6:
        c, s = np.cos(cam_yaw), np.sin(cam_yaw)
        cp[0, 3] = cx + s * dist
        cp[1, 3] = cy
        cp[2, 3] = c * dist
        cp[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        cp[0, 3] = cx; cp[1, 3] = cy; cp[2, 3] = dist
    scene.add(cam, pose=cp)

    kl = pyrender.DirectionalLight(color=[1.0, 0.97, 0.92], intensity=4.5)
    kp = np.eye(4)
    c1, s1 = np.cos(-0.5), np.sin(-0.5)
    c2, s2 = np.cos(0.3),  np.sin(0.3)
    kp[:3, :3] = (np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]]) @
                  np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]]))
    scene.add(kl, pose=kp)

    fl = pyrender.DirectionalLight(color=[0.7, 0.8, 0.95], intensity=2.0)
    fp = np.eye(4)
    c3, s3 = np.cos(-0.3), np.sin(-0.3)
    c4, s4 = np.cos(-0.4), np.sin(-0.4)
    fp[:3, :3] = (np.array([[c4, 0, s4], [0, 1, 0], [-s4, 0, c4]]) @
                  np.array([[1, 0, 0], [0, c3, -s3], [0, s3, c3]]))
    scene.add(fl, pose=fp)

    rl = pyrender.DirectionalLight(color=[0.5, 0.5, 0.6], intensity=1.5)
    rp = np.eye(4)
    rp[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    scene.add(rl, pose=rp)

    color, _ = R._renderer.render(scene)
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════
#  Synthetic motion generator (for smoke-testing without AIST++)
# ═══════════════════════════════════════════════════════════════
def make_test_motion(n_frames: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a simple wave-both-arms + gentle hip bob motion.

    Returns: (pose_aa (T, 24, 3), trans (T, 3))
    """
    T = n_frames
    pose = np.zeros((T, 24, 3), dtype=np.float64)
    trans = np.zeros((T, 3), dtype=np.float64)

    t = np.linspace(0, 4 * np.pi, T)        # 2 full cycles

    # Arms: rotate at the SHOULDER (joints 16=L_Shoulder, 17=R_Shoulder).
    # SMPL shoulder rest axis is along ±X. To wave the arm UP/DOWN we
    # rotate around Z (axis-angle (0, 0, θ)).
    #
    # For LEFT arm at +X: a +Z rotation lifts the arm UP toward +Y.
    # For RIGHT arm at -X: a +Z rotation lifts the arm DOWN. So negate.
    amp = np.deg2rad(70)
    pose[:, 16, 2] = +amp * (0.5 + 0.5 * np.sin(t))     # L: 0..+amp
    pose[:, 17, 2] = -amp * (0.5 + 0.5 * np.sin(t))     # R: 0..-amp
    # Bend elbows slightly so they don't stay rigid (rotate around Y here).
    pose[:, 18, 1] = -np.deg2rad(40) * (0.5 + 0.5 * np.sin(t))   # L_Elbow
    pose[:, 19, 1] = +np.deg2rad(40) * (0.5 + 0.5 * np.sin(t))   # R_Elbow

    # Hip bob: pelvis Y goes up and down a few cm.
    trans[:, 1] = 0.04 * np.sin(2 * t)

    return pose, trans


# ═══════════════════════════════════════════════════════════════
#  AIST++ loader
# ═══════════════════════════════════════════════════════════════
def load_aist_pkl(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load an AIST++-style .pkl. Returns (pose_aa (T,24,3), trans (T,3), fps)."""
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    smpl_poses   = np.asarray(d['smpl_poses'])           # (T, 72) or (T, 24, 3)
    smpl_trans   = np.asarray(d['smpl_trans'])           # (T, 3)
    smpl_scaling = float(np.asarray(d.get('smpl_scaling', [1.0])).reshape(-1)[0])
    if smpl_poses.ndim == 2 and smpl_poses.shape[1] == 72:
        smpl_poses = smpl_poses.reshape(-1, 24, 3)
    elif smpl_poses.ndim == 2 and smpl_poses.shape[1] >= 72:
        smpl_poses = smpl_poses[:, :72].reshape(-1, 24, 3)
    # AIST++ trans is in mm * scaling — divide.
    smpl_trans = smpl_trans / smpl_scaling
    fps = float(d.get('fps', 60.0))
    return smpl_poses.astype(np.float64), smpl_trans.astype(np.float64), fps


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vrm', default='data/models/AvatarSample_B.vrm')
    p.add_argument('--out', default=None)
    p.add_argument('--test', action='store_true', help='synthetic motion')
    p.add_argument('--aist', default=None, help='path to AIST++ .pkl')
    p.add_argument('--frames', type=int, default=90)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--smpl_pkl',
                   default='data/models/smpl_raw/smpl/models/'
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    p.add_argument('--cam_yaw', type=float, default=0.0)
    args = p.parse_args()

    if args.test:
        tag = 'test_wave'
        pose_aa, trans = make_test_motion(args.frames)
        fps = args.fps
    elif args.aist:
        tag = os.path.splitext(os.path.basename(args.aist))[0]
        pose_aa, trans, fps = load_aist_pkl(args.aist)
        if args.frames > 0 and args.frames < len(pose_aa):
            pose_aa = pose_aa[:args.frames]
            trans   = trans[:args.frames]
    else:
        raise SystemExit('Specify --test or --aist <pkl>')

    out = args.out or f'data/output_videos/smpl_{tag}.mp4'
    render_smpl_clip(pose_aa, trans, args.vrm, out,
                     fps=fps, smpl_pkl=args.smpl_pkl,
                     camera_yaw=args.cam_yaw)


if __name__ == '__main__':
    main()
