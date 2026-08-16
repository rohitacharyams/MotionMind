"""
Export AIST++ SMPL motion as a portable JSON file.

Output JSON schema (per file):
{
  "name":         "gKR_sBM_cAll_d29_mKR1_ch01",
  "fps":          60,
  "n_frames":     640,
  "duration_s":   10.6667,
  "skeleton":     "VRM-humanoid (subset)",
  "bones":        ["Hips", "Spine", "Spine2", "Neck", "Head",
                   "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
                   "RightShoulder","RightArm","RightForeArm","RightHand",
                   "LeftUpLeg","LeftLeg","LeftFoot",
                   "RightUpLeg","RightLeg","RightFoot"],
  "rotations":  { bone_name: [[x,y,z,w], ...n_frames] },   # quaternion, glTF order
  "hips_translation": [[x,y,z], ...n_frames],              # meters, model space
  "rest_local_rotation": { bone_name: [x,y,z,w] },         # rest local (apply
                                                           # rotations relative
                                                           # to this if you want
                                                           # additive deltas)
  "rest_local_translation": { bone_name: [x,y,z] }
}

This is the universal "what each joint did at each frame" format. Pair it with
ANY humanoid skeleton in three.js / Unity / Blender via bone-name retargeting.

Usage:
    python scripts/export_motion_json.py \
        --aist data/motion_db/aist/gHO_sBM_cAll_d21_mHO5_ch08.pkl \
        --vrm  data/models/extra/AliciaSolid.vrm \
        --out  data/output_videos/motion_house.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_glb import compute_motion_tracks, matrix_to_quat_xyzw
from scripts.play_smpl_motion import VRMRendererSMPL, load_aist_pkl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aist', required=True)
    p.add_argument('--vrm', required=True)
    p.add_argument('--smpl_pkl', default='data/models/smpl_raw/smpl/SMPL_NEUTRAL.pkl')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    times, rotations_by_node, hips_t, hips_node = compute_motion_tracks(
        args.vrm, args.smpl_pkl, args.aist)

    # Re-run renderer briefly to recover node->bone-name mapping
    R = VRMRendererSMPL(args.vrm, w=64, h=64)
    node_to_bone = {}
    for idx in range(len(R.joints)):
        std = R._joint_to_std.get(idx)
        if std is not None:
            node_to_bone[int(R.joints[idx])] = std

    _, _, fps = load_aist_pkl(args.aist)
    n_frames = len(times)
    name = os.path.splitext(os.path.basename(args.aist))[0]

    rotations = {}
    rest_local_R = {}
    rest_local_t = {}

    for ni, qs in rotations_by_node.items():
        bone = node_to_bone.get(ni, f'node_{ni}')
        rotations[bone] = qs.tolist()

        # Recover rest local TRS for this node from the renderer
        # parent_w^-1 @ self_w
        children_map = R._children
        parent_idx = None
        for cidx in range(len(R.joints)):
            if ni in children_map.get(R.joints[cidx], []):
                parent_idx = cidx
                break
        if parent_idx is None:
            parent_w = np.eye(4)
        else:
            parent_w = R._skin_wt.get(R.joints[parent_idx], np.eye(4))
        self_w = R._skin_wt.get(ni, np.eye(4))
        rest_local = np.linalg.inv(parent_w) @ self_w
        rest_local_R[bone] = matrix_to_quat_xyzw(rest_local[:3, :3]).tolist()
        rest_local_t[bone] = rest_local[:3, 3].tolist()

    out = {
        'name': name,
        'fps': float(fps),
        'n_frames': int(n_frames),
        'duration_s': float(n_frames) / float(fps),
        'skeleton': 'VRM-humanoid (subset)',
        'bones': sorted(rotations.keys()),
        'rotations': rotations,
        'hips_translation': hips_t.tolist(),
        'rest_local_rotation': rest_local_R,
        'rest_local_translation': rest_local_t,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f'  Wrote {args.out}  ({n_frames} frames, {len(rotations)} bones, '
          f'{size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
