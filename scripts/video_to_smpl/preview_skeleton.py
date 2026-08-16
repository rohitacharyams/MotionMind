"""preview_skeleton.py — render a GVHMR/AIST SMPL .pkl as a stick-figure animation.

This is a LOCAL visual sanity check that needs NO VRM avatar and NO SMPL body
.pkl — it uses the hardcoded SMPL parent tree + a standard SMPL neutral rest
skeleton, runs forward kinematics (src/smpl_lite.py), and draws the 24 joints
as an animated stick figure. Use it to confirm a video->SMPL extraction is a
real, upright, dancing human BEFORE shipping the clip to the VRM avatar.

Usage:
    python scripts/video_to_smpl/preview_skeleton.py \
        --aist "C:/Users/me/Downloads/dance_video_i_small.pkl" \
        --out  data/output_videos/gvhmr_preview.gif
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.smpl_lite import SMPL_PARENTS, SMPL_NAMES, forward_kinematics

# Standard SMPL neutral rest-pose joint template (meters, Y-up, pelvis at
# origin, +X = subject's left, +Z = forward). Lets us do FK without the SMPL
# body .pkl. Proportions match the SMPL 24-joint skeleton.
SMPL_REST_JOINTS = np.array([
    [ 0.0000,  0.0000,  0.0000],  # 0  pelvis
    [ 0.0586, -0.0820,  0.0148],  # 1  L_Hip
    [-0.0589, -0.0820,  0.0135],  # 2  R_Hip
    [ 0.0044,  0.1112, -0.0266],  # 3  Spine1
    [ 0.1046, -0.4956,  0.0106],  # 4  L_Knee
    [-0.1073, -0.4948,  0.0130],  # 5  R_Knee
    [ 0.0038,  0.2565, -0.0304],  # 6  Spine2
    [ 0.0894, -0.8892, -0.0388],  # 7  L_Ankle
    [-0.0900, -0.8887, -0.0402],  # 8  R_Ankle
    [ 0.0029,  0.3060, -0.0216],  # 9  Spine3
    [ 0.1173, -0.9463,  0.1067],  # 10 L_Foot
    [-0.1178, -0.9459,  0.1054],  # 11 R_Foot
    [ 0.0034,  0.4737, -0.0293],  # 12 Neck
    [ 0.0801,  0.3947, -0.0272],  # 13 L_Collar
    [-0.0771,  0.3894, -0.0276],  # 14 R_Collar
    [ 0.0057,  0.5765,  0.0247],  # 15 Head
    [ 0.1818,  0.4297, -0.0409],  # 16 L_Shoulder
    [-0.1776,  0.4271, -0.0430],  # 17 R_Shoulder
    [ 0.4600,  0.4181, -0.0547],  # 18 L_Elbow
    [-0.4574,  0.4145, -0.0526],  # 19 R_Elbow
    [ 0.7139,  0.4109, -0.0521],  # 20 L_Wrist
    [-0.7135,  0.4067, -0.0526],  # 21 R_Wrist
    [ 0.7906,  0.4058, -0.0498],  # 22 L_Hand
    [-0.7911,  0.4029, -0.0503],  # 23 R_Hand
], dtype=np.float64)


def load_pkl(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    poses = np.asarray(d['smpl_poses'], dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[:, :72].reshape(-1, 24, 3)
    trans = np.asarray(d['smpl_trans'], dtype=np.float64).reshape(-1, 3)
    scaling = float(np.asarray(d.get('smpl_scaling', [1.0])).reshape(-1)[0])
    trans = trans / max(scaling, 1e-6)
    fps = float(d.get('fps', 30.0))
    return poses, trans, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aist', required=True)
    ap.add_argument('--out', default='data/output_videos/gvhmr_preview.gif')
    ap.add_argument('--stride', type=int, default=2, help='use every Nth frame')
    ap.add_argument('--elev', type=float, default=8.0)
    ap.add_argument('--azim', type=float, default=-70.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    poses, trans, fps = load_pkl(args.aist)
    poses, trans = poses[::args.stride], trans[::args.stride]
    T = len(poses)

    # FK every frame -> world joint positions (T, 24, 3)
    J = np.stack([
        forward_kinematics(SMPL_REST_JOINTS, SMPL_PARENTS, poses[i], trans[i])[0]
        for i in range(T)
    ])

    # bones (child -> parent) for drawing
    bones = [(i, int(SMPL_PARENTS[i])) for i in range(1, 24)]

    # fixed axis limits from the whole clip so the figure doesn't jump
    lo, hi = J.min(axis=(0, 1)), J.max(axis=(0, 1))
    ctr = (lo + hi) / 2
    span = float((hi - lo).max()) * 0.6 + 1e-3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111, projection='3d')

    def draw(f):
        ax.clear()
        P = J[f]
        # SMPL: X=left, Y=up, Z=fwd. Plot X, Z(depth), Y(up).
        for c, p in bones:
            ax.plot([P[c, 0], P[p, 0]], [P[c, 2], P[p, 2]], [P[c, 1], P[p, 1]],
                    '-', color='#6589bf', lw=2)
        ax.scatter(P[:, 0], P[:, 2], P[:, 1], c='#78aa58', s=12)
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[2] - span, ctr[2] + span)
        ax.set_zlim(ctr[1] - span, ctr[1] + span)
        ax.set_box_aspect((1, 1, 1.6))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f'GVHMR SMPL  frame {f*args.stride}/{T*args.stride}', fontsize=9)

    out_fps = max(1, int(round(fps / args.stride)))
    anim = FuncAnimation(fig, draw, frames=T, interval=1000 / out_fps)
    anim.save(args.out, writer=PillowWriter(fps=out_fps))
    plt.close(fig)
    print(f'wrote {args.out}  ({T} frames @ {out_fps} fps)')
    # quick numeric confirmation the figure is upright & human-sized
    height = float(J[:, 15, 1].mean() - J[:, [10, 11], 1].mean())  # head - feet (Y)
    print(f'  mean head-above-feet height = {height:.2f} m  (upright if > 0)')


if __name__ == '__main__':
    main()
