"""wild_video_to_smpl_local.py — MediaPipe Tasks Pose -> SMPL .pkl (CPU, no GPU).

Reuses the analytical IK from motion_transfer_v3.py. Outputs a .pkl in the SAME
schema as AIST++ so it drops directly into the existing pipeline
(export_bvh.py, build_glbs.py, play_smpl_motion.py).
"""
from __future__ import annotations
import argparse, os, sys, pickle, time
import numpy as np
import cv2

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from motion_transfer_v3 import (  # noqa: E402
    load_smpl_model,
    compute_pose_analytical,
)

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

DEFAULT_SMPL = r"c:\dan\data\models\smpl_raw\smpl\models\basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
DEFAULT_TASK_HEAVY = r"c:\dan\data\models\pose_landmarker_heavy.task"
DEFAULT_TASK_LITE  = r"c:\dan\data\models\pose_landmarker_lite.task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--smpl", default=DEFAULT_SMPL)
    ap.add_argument("--task", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.isfile(args.smpl):
        sys.exit(f"SMPL model not found: {args.smpl}")

    task_path = args.task or (DEFAULT_TASK_HEAVY if os.path.isfile(DEFAULT_TASK_HEAVY)
                              else DEFAULT_TASK_LITE)
    if not os.path.isfile(task_path):
        sys.exit(f"pose_landmarker .task model not found: {task_path}")
    print(f"[load] pose model: {task_path}")
    print(f"[load] SMPL: {args.smpl}")
    smpl = load_smpl_model(args.smpl)

    print(f"[open] {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("opencv failed to open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  fps={fps:.2f} | total_frames={total} | {W}x{H}")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    n_to_read = (args.max_frames if args.max_frames > 0
                 else total - args.start_frame)

    base_opts = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    poses_aa = np.zeros((n_to_read, 24, 3), dtype=np.float64)
    trans = np.zeros((n_to_read, 3), dtype=np.float64)
    valid = np.zeros(n_to_read, dtype=bool)

    overlay_writer = None
    if args.overlay:
        os.makedirs(os.path.dirname(args.overlay) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(args.overlay, fourcc, fps, (W, H))

    POSE_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),
                  (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(27,31),
                  (24,26),(26,28),(28,30),(28,32)]

    t0 = time.time()
    actual = 0
    for i in range(n_to_read):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        actual = i + 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(round((args.start_frame + i) * 1000.0 / fps))
        try:
            res = landmarker.detect_for_video(mp_img, ts_ms)
        except Exception as e:
            print(f"  frame {i}: detect failed ({e})")
            if overlay_writer is not None:
                overlay_writer.write(frame_bgr)
            continue

        if not res.pose_world_landmarks:
            if overlay_writer is not None:
                cv2.putText(frame_bgr, "NO POSE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                overlay_writer.write(frame_bgr)
            continue

        wl_list = res.pose_world_landmarks[0]   # 33 landmarks (meters)
        wl = np.array([[lm.x, lm.y, lm.z] for lm in wl_list], dtype=np.float64)

        try:
            pose_72 = compute_pose_analytical(smpl, wl)
        except Exception as e:
            print(f"  frame {i}: IK failed ({e})")
            if overlay_writer is not None:
                overlay_writer.write(frame_bgr)
            continue

        poses_aa[i] = pose_72.reshape(24, 3)

        if res.pose_landmarks:
            sl = res.pose_landmarks[0]
            pelvis_x = (sl[23].x + sl[24].x) / 2
            pelvis_y = (sl[23].y + sl[24].y) / 2
            trans[i] = [(pelvis_x - 0.5) * 2.0,
                        -(pelvis_y - 0.5) * 2.0,
                        0.0]
        valid[i] = True

        if overlay_writer is not None and res.pose_landmarks:
            sl = res.pose_landmarks[0]
            for a, b in POSE_EDGES:
                pa = (int(sl[a].x * W), int(sl[a].y * H))
                pb = (int(sl[b].x * W), int(sl[b].y * H))
                cv2.line(frame_bgr, pa, pb, (40, 220, 80), 2)
            for k in range(33):
                cv2.circle(frame_bgr, (int(sl[k].x * W), int(sl[k].y * H)),
                           3, (0, 200, 255), -1)
            cv2.putText(frame_bgr, f"f{i:04d}  IK ok",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (40, 220, 80), 2)
            overlay_writer.write(frame_bgr)

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_to_read} frames | {rate:.1f} fps | "
                  f"valid={valid[:i+1].sum()}")

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    landmarker.close()

    poses_aa = poses_aa[:actual]
    trans = trans[:actual]
    valid = valid[:actual]
    pct = (100 * valid.mean()) if actual else 0.0
    print(f"[done] {valid.sum()}/{actual} valid frames ({pct:.1f}%)")

    if args.smooth > 1 and valid.sum() > args.smooth:
        from scipy.ndimage import uniform_filter1d
        for j in range(24):
            for d in range(3):
                poses_aa[:, j, d] = uniform_filter1d(
                    poses_aa[:, j, d], size=args.smooth, mode='nearest')
        for d in range(3):
            trans[:, d] = uniform_filter1d(trans[:, d], size=args.smooth, mode='nearest')

    if (~valid).any() and valid.any():
        good_idx = np.where(valid)[0]
        for i in range(len(valid)):
            if not valid[i]:
                nearest = good_idx[np.argmin(np.abs(good_idx - i))]
                poses_aa[i] = poses_aa[nearest]
                trans[i] = trans[nearest]

    out_dict = {
        'smpl_poses':   poses_aa.astype(np.float32),
        'smpl_trans':   trans.astype(np.float32),
        'smpl_scaling': np.array([1.0], dtype=np.float32),
        'betas':        np.zeros(10, dtype=np.float32),
        'fps':          float(fps),
        '_source':      'mediapipe-tasks-pose-heavy-v3',
        '_valid_frames': int(valid.sum()),
        '_total_frames': int(actual),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out_dict, f, protocol=4)
    print(f"[save] {args.out} ({os.path.getsize(args.out)/1024:.1f} KB)")
    if args.overlay:
        print(f"[save] {args.overlay}")


if __name__ == "__main__":
    main()
