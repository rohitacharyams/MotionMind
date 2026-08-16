"""
Export AIST++ SMPL motion BAKED into a VRM as a glTF/GLB file.

Output: a single .glb with the avatar mesh + skeleton + animation track.
This file works in:
  - Blender (File -> Import -> glTF 2.0)
  - Unity (drag .glb into Assets, set rig Humanoid)
  - Unreal Engine 5 (Import as Skeletal Mesh + Animation)
  - three.js / babylon.js / online glTF viewers

Usage:
    python scripts/export_glb.py \
        --aist data/motion_db/aist/gHO_sBM_cAll_d21_mHO5_ch08.pkl \
        --vrm  data/models/extra/AliciaSolid.vrm \
        --out  data/output_videos/alicia_house.glb
"""
from __future__ import annotations

import argparse
import os
import sys
import struct
from io import BytesIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.play_smpl_motion import (
    load_aist_pkl, VRMRendererSMPL, smpl_pose_to_world_R, SMPL_TO_VRM_R,
    load_smpl_skeleton)


# ---------- math helpers -----------------------------------------------------

def matrix_to_quat_xyzw(R):
    """3x3 rotation matrix -> (x, y, z, w) quaternion (glTF order)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0, 0, 0, 1], np.float32)


# ---------- compute per-frame node TRS from motion ----------------------------

def compute_motion_tracks(vrm_path, smpl_pkl, aist_pkl):
    """
    Returns
      time:        (N,) float32 seconds
      rotations:   dict[node_idx] = (N, 4) float32 quaternion (x,y,z,w) LOCAL
      hips_trans:  (N, 3) float32 LOCAL translation for the Hips node
      hips_node:   int — node index of the Hips bone
    """
    pose_aa, trans, fps = load_aist_pkl(aist_pkl)
    sk = load_smpl_skeleton(smpl_pkl)
    rest_joints = sk['rest_joints']
    parents     = sk['parents']

    # Borrow renderer just for its skeleton metadata
    R = VRMRendererSMPL(vrm_path, w=320, h=320)

    # height scaling, mirrors render_smpl_clip
    smpl_h = float(rest_joints[15, 1] - rest_joints[10, 1])
    if smpl_h < 1e-3:
        smpl_h = float(rest_joints[15, 1] - rest_joints[7, 1])
    m_head = m_hips = None
    for idx in range(len(R.joints)):
        std = R._joint_to_std.get(idx)
        if std == 'Head':
            m_head = R._skin_wt.get(R.joints[idx], np.eye(4))[:3, 3]
        if std == 'Hips':
            m_hips = R._skin_wt.get(R.joints[idx], np.eye(4))[:3, 3]
    canon_h = (np.linalg.norm(m_head - m_hips) * 1.55) if (m_head is not None) else 1.0
    h_scale = canon_h / smpl_h if smpl_h > 1e-3 else 1.0

    # Identify VRM joints we will animate
    animated_idx = sorted([idx for idx in range(len(R.joints))
                           if R._joint_to_std.get(idx) is not None])
    hips_idx = next(idx for idx in animated_idx
                    if R._joint_to_std.get(idx) == 'Hips')
    hips_node = int(R.joints[hips_idx])

    # Original local TRS for each VRM node we touch (so non-animated frames
    # would keep the rest pose).
    rest_local_R = {}
    rest_local_t = {}
    for idx in animated_idx:
        ni = R.joints[idx]
        # parent's WORLD rest = parent's _skin_wt
        # this node's WORLD rest = self._skin_wt
        Rw = R._skin_wt.get(ni, np.eye(4))
        # Find parent in the joint list
        # Use children map from renderer
        parent_idx = None
        for cidx in range(len(R.joints)):
            children_n = R._children.get(R.joints[cidx], [])
            if ni in children_n:
                parent_idx = cidx
                break
        if parent_idx is None:
            parent_w = np.eye(4)
        else:
            parent_w = R._skin_wt.get(R.joints[parent_idx], np.eye(4))
        T_rest_local = np.linalg.inv(parent_w) @ Rw
        rest_local_R[idx] = T_rest_local[:3, :3]
        rest_local_t[idx] = T_rest_local[:3, 3]

    # Per-frame rotations
    n_frames = len(pose_aa)
    times = np.arange(n_frames, dtype=np.float32) / float(fps)

    rotations = {int(R.joints[idx]): np.zeros((n_frames, 4), np.float32)
                 for idx in animated_idx}
    hips_translations = np.zeros((n_frames, 3), np.float32)

    t0 = trans[0]
    for fi in range(n_frames):
        world_R_by_std, _ = smpl_pose_to_world_R(
            pose_aa[fi], rest_joints, parents)
        hip_translation_model = SMPL_TO_VRM_R @ ((trans[fi] - t0) * h_scale)

        # Run the renderer's skinning to populate self._last_posed_world
        R.compute_skinning_matrices_from_world_R(
            world_R_by_std, hip_translation_model)
        posed_world = R._last_posed_world

        # Compute local rotation for each animated node:
        #   posed_world[j] = posed_world[parent] @ local_T[j]
        #   => local_T[j]  = parent_posed_world^-1 @ posed_world[j]
        for idx in animated_idx:
            ni = R.joints[idx]
            Pj = posed_world.get(idx)
            if Pj is None:
                Pj = R._skin_wt.get(ni, np.eye(4))
            # Find parent's posed_world
            parent_idx = None
            for cidx in range(len(R.joints)):
                if ni in R._children.get(R.joints[cidx], []):
                    parent_idx = cidx
                    break
            if parent_idx is None:
                parent_pw = np.eye(4)
            else:
                parent_pw = posed_world.get(parent_idx,
                                            R._skin_wt.get(R.joints[parent_idx],
                                                            np.eye(4)))
            local_T = np.linalg.inv(parent_pw) @ Pj
            local_R = local_T[:3, :3]
            rotations[ni][fi] = matrix_to_quat_xyzw(local_R)

            if idx == hips_idx:
                # local translation = parent_pw^-1 @ Pj (already computed)
                # but for the Hips, the translation IS the hip motion
                hips_translations[fi] = local_T[:3, 3].astype(np.float32)

    return times, rotations, hips_translations, hips_node


# ---------- glTF binary patcher ----------------------------------------------

def _u32(b, o): return struct.unpack_from('<I', b, o)[0]
def _u16(b, o): return struct.unpack_from('<H', b, o)[0]


def load_glb(path):
    """Load GLB and return (json_dict, binary_blob)."""
    import json
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'glTF', f'Not a GLB: {path}'
    version = _u32(data, 4)
    total = _u32(data, 8)
    o = 12
    j_obj = b_obj = None
    while o < total:
        clen = _u32(data, o); ctyp = data[o+4:o+8]; o += 8
        chunk = data[o:o+clen]; o += clen
        if ctyp == b'JSON':
            j_obj = json.loads(chunk.decode('utf-8'))
        elif ctyp == b'BIN\x00':
            b_obj = bytearray(chunk)
    assert j_obj is not None and b_obj is not None
    return j_obj, b_obj


def save_glb(out_path, gltf_json, bin_blob):
    import json
    js = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
    # pad to 4 bytes
    while len(js) % 4: js += b' '
    while len(bin_blob) % 4: bin_blob.append(0)
    total = 12 + 8 + len(js) + 8 + len(bin_blob)
    with open(out_path, 'wb') as f:
        f.write(b'glTF')
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', total))
        f.write(struct.pack('<I', len(js)))
        f.write(b'JSON')
        f.write(js)
        f.write(struct.pack('<I', len(bin_blob)))
        f.write(b'BIN\x00')
        f.write(bytes(bin_blob))


def add_buffer_view(gltf, bin_blob, data_bytes):
    """Append data to bin_blob; create a bufferView. Returns index."""
    # 4-byte align
    while len(bin_blob) % 4: bin_blob.append(0)
    offset = len(bin_blob)
    bin_blob.extend(data_bytes)
    bv = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(data_bytes)}
    gltf.setdefault('bufferViews', []).append(bv)
    return len(gltf['bufferViews']) - 1


def add_accessor(gltf, bv_idx, count, comp_type, atype, vmin=None, vmax=None):
    a = {'bufferView': bv_idx, 'componentType': comp_type,
         'count': count, 'type': atype}
    if vmin is not None: a['min'] = list(map(float, vmin))
    if vmax is not None: a['max'] = list(map(float, vmax))
    gltf.setdefault('accessors', []).append(a)
    return len(gltf['accessors']) - 1


def add_animation(gltf, bin_blob, name, times, rotations, hips_node,
                  hips_translations):
    # buffer 0 must be embedded BIN. Make sure buffer length is updated at end.
    # Times accessor (shared)
    times_bytes = times.astype(np.float32).tobytes()
    bv_t = add_buffer_view(gltf, bin_blob, times_bytes)
    acc_t = add_accessor(gltf, bv_t, len(times), 5126, 'SCALAR',
                         vmin=[float(times.min())],
                         vmax=[float(times.max())])

    samplers = []
    channels = []

    for node, quats in rotations.items():
        qbytes = quats.astype(np.float32).tobytes()
        bv_r = add_buffer_view(gltf, bin_blob, qbytes)
        acc_r = add_accessor(gltf, bv_r, len(quats), 5126, 'VEC4')
        samplers.append({'input': acc_t, 'output': acc_r,
                         'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1,
                         'target': {'node': int(node), 'path': 'rotation'}})

    # Hips translation
    tbytes = hips_translations.astype(np.float32).tobytes()
    bv_ht = add_buffer_view(gltf, bin_blob, tbytes)
    acc_ht = add_accessor(gltf, bv_ht, len(hips_translations), 5126, 'VEC3')
    samplers.append({'input': acc_t, 'output': acc_ht,
                     'interpolation': 'LINEAR'})
    channels.append({'sampler': len(samplers) - 1,
                     'target': {'node': int(hips_node), 'path': 'translation'}})

    anim = {'name': name, 'samplers': samplers, 'channels': channels}
    gltf.setdefault('animations', []).append(anim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aist', required=True)
    p.add_argument('--vrm', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smpl_pkl',
                   default='data/models/smpl_raw/smpl/models/'
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    args = p.parse_args()

    print('  Computing motion tracks...')
    times, rotations, hips_t, hips_node = compute_motion_tracks(
        args.vrm, args.smpl_pkl, args.aist)
    print(f'  Tracks: {len(rotations)} bones, {len(times)} frames, '
          f'{times[-1]:.2f}s')

    print('  Loading source VRM as GLB...')
    gltf, bin_blob = load_glb(args.vrm)

    # Update buffer 0 length later. Currently glTF "buffers"[0]["byteLength"]
    # equals current bin_blob length. We will append; update when done.

    name = os.path.splitext(os.path.basename(args.aist))[0]
    print('  Adding animation track...')
    add_animation(gltf, bin_blob, name, times, rotations, hips_node, hips_t)

    # Update buffer length
    gltf['buffers'][0]['byteLength'] = len(bin_blob)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    save_glb(args.out, gltf, bin_blob)
    sz = os.path.getsize(args.out) / 1024 / 1024
    print(f'  ✓ Wrote {args.out} ({sz:.1f} MB) — has {len(gltf.get("animations",[]))} animation(s)')


if __name__ == '__main__':
    main()
