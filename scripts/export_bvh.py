"""
Export an AIST++ SMPL motion to BVH (Biovision Hierarchy) format.

BVH is the universal motion-capture format. Drop the resulting .bvh into:
- Blender (File -> Import -> Motion Capture (.bvh))
- Unity (drag into Assets, set rig type Humanoid)
- Unreal (Import as Animation Sequence)
- MotionBuilder, Maya, Cascadeur, Mixamo, etc.

The output skeleton is the SMPL 24-joint hierarchy. Bone lengths are derived
from SMPL rest joint positions (real captured proportions).

Usage:
    python scripts/export_bvh.py \
        --aist data/motion_db/aist/gHO_sBM_cAll_d21_mHO5_ch08.pkl \
        --out data/output_videos/gHO.bvh
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smpl_lite import SMPL_NAMES as SMPL_JOINT_NAMES, SMPL_PARENTS
from scripts.play_smpl_motion import load_aist_pkl, load_smpl_skeleton


# Channel order for BVH — XYZ position for root, ZYX rotation (Euler) per joint.
# We use Z-Y-X intrinsic Euler (a.k.a. ZYX) which is the standard for BVH.

def axis_angle_to_matrix(aa):
    """(3,) axis-angle -> (3,3) rotation matrix via Rodrigues."""
    theta = np.linalg.norm(aa)
    if theta < 1e-9:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_zyx_euler_deg(R):
    """Decompose R into intrinsic ZYX Euler angles, in degrees.

    R = Rz(z) @ Ry(y) @ Rx(x), so applied as Rx first then Ry then Rz.
    Returns (z, y, x) in degrees — BVH channel order Zrot Yrot Xrot.
    """
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = np.arcsin(sy)
    if abs(sy) < 0.9999:
        x = np.arctan2(R[2, 1], R[2, 2])
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        z = 0.0
    return np.rad2deg(z), np.rad2deg(y), np.rad2deg(x)


def build_hierarchy_text(rest_joints, parents, scale=100.0):
    """BVH HIERARCHY block. Offsets are bone lengths in cm."""
    children = {i: [] for i in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    lines = []

    def emit(j, depth, is_root):
        ind = '  ' * depth
        name = SMPL_JOINT_NAMES[j]
        keyword = 'ROOT' if is_root else 'JOINT'
        lines.append(f'{ind}{keyword} {name}')
        lines.append(f'{ind}{{')
        if is_root:
            ofs = (0.0, 0.0, 0.0)
        else:
            d = (rest_joints[j] - rest_joints[parents[j]]) * scale
            ofs = (d[0], d[1], d[2])
        lines.append(f'{ind}  OFFSET {ofs[0]:.4f} {ofs[1]:.4f} {ofs[2]:.4f}')
        if is_root:
            lines.append(f'{ind}  CHANNELS 6 Xposition Yposition Zposition '
                         f'Zrotation Yrotation Xrotation')
        else:
            lines.append(f'{ind}  CHANNELS 3 Zrotation Yrotation Xrotation')
        if children[j]:
            for c in children[j]:
                emit(c, depth + 1, False)
        else:
            # End site — give it a small offset along the bone direction guess
            lines.append(f'{ind}  End Site')
            lines.append(f'{ind}  {{')
            # Use parent->this direction extended by 5 cm.
            if parents[j] >= 0:
                d = rest_joints[j] - rest_joints[parents[j]]
                if np.linalg.norm(d) > 1e-6:
                    d = d / np.linalg.norm(d) * 0.10  # 10cm
                else:
                    d = np.array([0.0, 0.10, 0.0])
            else:
                d = np.array([0.0, 0.10, 0.0])
            d = d * scale
            lines.append(f'{ind}    OFFSET {d[0]:.4f} {d[1]:.4f} {d[2]:.4f}')
            lines.append(f'{ind}  }}')
        lines.append(f'{ind}}}')

    emit(0, 0, True)
    return '\n'.join(lines)


def export_bvh(aist_pkl, out_bvh, smpl_pkl):
    pose_aa, trans, fps = load_aist_pkl(aist_pkl)  # (N,24,3), (N,3), fps
    sk = load_smpl_skeleton(smpl_pkl)
    rest_joints = sk['rest_joints']
    parents = sk['parents']

    n_frames = len(pose_aa)
    # SMPL trans is in METERS (after smpl_scaling already applied).  BVH typically
    # uses cm, so multiply by 100 to match the offset scale.
    SCALE = 100.0
    trans_cm = trans * SCALE

    print(f'  Frames: {n_frames}, fps: {fps}')
    hier = build_hierarchy_text(rest_joints, parents, scale=SCALE)

    motion_lines = ['MOTION', f'Frames: {n_frames}',
                    f'Frame Time: {1.0/fps:.6f}']

    # Per-frame channel string. Root: XYZ pos + ZYX rot.  Others: ZYX rot.
    for fi in range(n_frames):
        vals = []
        # Root translation (X Y Z in cm). SMPL has Y up just like BVH.
        vals.append(trans_cm[fi, 0])
        vals.append(trans_cm[fi, 1])
        vals.append(trans_cm[fi, 2])
        # Root rotation (joint 0)
        Rj = axis_angle_to_matrix(pose_aa[fi, 0])
        z, y, x = matrix_to_zyx_euler_deg(Rj)
        vals.extend([z, y, x])
        # Other 23 joints
        for j in range(1, 24):
            Rj = axis_angle_to_matrix(pose_aa[fi, j])
            z, y, x = matrix_to_zyx_euler_deg(Rj)
            vals.extend([z, y, x])
        motion_lines.append(' '.join(f'{v:.4f}' for v in vals))

    text = 'HIERARCHY\n' + hier + '\n' + '\n'.join(motion_lines) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(out_bvh)) or '.', exist_ok=True)
    with open(out_bvh, 'w') as f:
        f.write(text)
    sz = os.path.getsize(out_bvh) / 1024
    print(f'  Wrote {out_bvh} ({sz:.0f} KB, {n_frames} frames @ {fps} fps)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aist', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--smpl_pkl',
                   default='data/models/smpl_raw/smpl/models/'
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    args = p.parse_args()
    out = args.out or os.path.splitext(args.aist)[0] + '.bvh'
    export_bvh(args.aist, out, args.smpl_pkl)


if __name__ == '__main__':
    main()
