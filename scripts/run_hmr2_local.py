"""
Local 4D-Humans / HMR2.0 runner — Windows + CPU friendly.

Mirrors the Colab notebook (notebooks/video_to_smpl_v2.ipynb) but reads a
local video and writes an AIST++-format pkl.

Usage (from c:\\dan with the .venv311 active):

    .\\.venv311\\Scripts\\python.exe scripts\\run_hmr2_local.py ^
        --video path\\to\\clip.mp4 ^
        --out   data\\motion_db\\hmr2_clip.pkl ^
        --fps   25 ^
        --max-seconds 5

After running, render against any VRM 0.x avatar:

    python scripts\\play_smpl_motion.py --aist data\\motion_db\\hmr2_clip.pkl ^
        --vrm data\\models\\extra\\AliciaSolid.vrm --frames 0 --fps 25 ^
        --out data\\output_videos\\hmr2_alicia.mp4
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

# Project layout
ROOT = Path(__file__).resolve().parents[1]
SMPL_ZIP = ROOT / "data" / "models" / "smpl_for_colab.zip"
HMR2_CACHE = Path(os.environ.get("HOME") or os.path.expanduser("~")) / ".cache" / "4DHumans"


def ensure_smpl_files() -> None:
    """Place a SMPL_NEUTRAL.pkl in every spot hmr2 may look for it."""
    target_paths = [
        HMR2_CACHE / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
        HMR2_CACHE / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    ]
    if all(p.is_file() for p in target_paths):
        return

    if not SMPL_ZIP.is_file():
        sys.exit(
            f"[!] {SMPL_ZIP} not found. Cannot place SMPL files. "
            "Get smpl_for_colab.zip (134 MB, contains the 4 SMPL pkls)."
        )

    extract_dir = HMR2_CACHE / "_smpl_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {SMPL_ZIP.name} to {extract_dir}")
    with zipfile.ZipFile(SMPL_ZIP) as zf:
        zf.extractall(extract_dir)

    # The neutral model isn't in the zip — male is the closest substitute.
    candidates = [
        "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        "SMPL_MALE.pkl",
        "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
        "SMPL_FEMALE.pkl",
    ]
    src: Path | None = None
    for name in candidates:
        for found in extract_dir.rglob(name):
            src = found
            break
        if src is not None:
            break
    if src is None:
        sys.exit(f"[!] Could not find any SMPL pkl in {extract_dir}")
    print(f"[*] Using {src.name} as SMPL_NEUTRAL")

    for tgt in target_paths:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, tgt)
        print(f"    -> {tgt}")


def trim_video(src: Path, dst: Path, max_seconds: int, fps_out: int) -> Path:
    if dst.is_file():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] ffmpeg trim {src.name} -> {dst.name}  (<= {max_seconds}s @ {fps_out} fps)")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(max_seconds),
        "-r", str(fps_out),
        "-c:v", "libx264", "-an", str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-500:])
        sys.exit("[!] ffmpeg failed (is it on PATH?)")
    return dst


def rotmat_to_aa(R: torch.Tensor) -> torch.Tensor:
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5).clamp(-1, 1)
    angle = torch.acos(cos)
    sin = torch.sin(angle)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1) / (2.0 * sin.unsqueeze(-1) + 1e-6)
    return axis * angle.unsqueeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-seconds", type=int, default=5,
                    help="Trim input to this length before inference (CPU saver).")
    ap.add_argument("--no-trim", action="store_true",
                    help="Skip ffmpeg trim; use the video as-is.")
    ap.add_argument("--smooth-pose", type=float, default=1.5)
    ap.add_argument("--smooth-trans", type=float, default=2.0)
    args = ap.parse_args()

    if not args.video.is_file():
        sys.exit(f"[!] video not found: {args.video}")

    ensure_smpl_files()

    if args.no_trim:
        clip_path = args.video
    else:
        clip_path = trim_video(
            args.video,
            ROOT / "data" / "input_videos" / f"hmr2_clip_{args.video.stem}.mp4",
            args.max_seconds, args.fps,
        )

    # Import after SMPL is staged so the model can find its files.
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
    from hmr2.utils import recursive_to
    from ultralytics import YOLO

    print("[*] download_models() — first run will fetch ~700 MB checkpoint")
    download_models(CACHE_DIR_4DHUMANS)

    print(f"[*] load_hmr2({DEFAULT_CHECKPOINT})")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"[*] HMR2 ready on {device}")

    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(clip_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"[*] {len(frames)} frames loaded")
    if not frames:
        sys.exit("[!] no frames decoded")

    all_poses: list[np.ndarray] = []
    all_trans: list[np.ndarray] = []
    t0 = time.time()
    for fi, img in enumerate(frames):
        det = yolo(img, classes=[0], verbose=False)[0]
        if len(det.boxes) == 0:
            if all_poses:
                all_poses.append(all_poses[-1].copy())
                all_trans.append(all_trans[-1].copy())
            else:
                all_poses.append(np.zeros(72, np.float32))
                all_trans.append(np.zeros(3, np.float32))
        else:
            boxes = det.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = boxes[areas.argmax():areas.argmax() + 1]
            ds = ViTDetDataset(model_cfg, img, best)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for batch in dl:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)
                go_R = out["pred_smpl_params"]["global_orient"].reshape(-1, 3, 3)
                bp_R = out["pred_smpl_params"]["body_pose"].reshape(-1, 3, 3)
                pose72 = torch.cat([
                    rotmat_to_aa(go_R).reshape(-1)[:3],
                    rotmat_to_aa(bp_R).reshape(-1)[:69],
                ]).cpu().numpy().astype(np.float32)
                cam_t = out["pred_cam_t"][0].cpu().numpy().astype(np.float32)
                all_poses.append(pose72)
                all_trans.append(cam_t)
                break

        if (fi + 1) % 5 == 0 or fi == len(frames) - 1:
            el = time.time() - t0
            eta = el / (fi + 1) * (len(frames) - fi - 1)
            print(f"  {fi+1:>4}/{len(frames)}  elapsed {el:6.1f}s  ETA {eta:6.1f}s")

    smpl_poses = np.stack(all_poses, 0)
    smpl_trans = np.stack(all_trans, 0)

    if args.smooth_pose > 0 or args.smooth_trans > 0:
        from scipy.ndimage import gaussian_filter1d
        if args.smooth_pose > 0:
            smpl_poses = gaussian_filter1d(smpl_poses, sigma=args.smooth_pose, axis=0).astype(np.float32)
        if args.smooth_trans > 0:
            smpl_trans = gaussian_filter1d(smpl_trans, sigma=args.smooth_trans, axis=0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "smpl_poses": smpl_poses,
            "smpl_trans": smpl_trans,
            "smpl_scaling": np.array([1.0], dtype=np.float32),
        }, f)
    print(f"[OK] wrote {args.out}  poses={smpl_poses.shape} trans={smpl_trans.shape}")


if __name__ == "__main__":
    main()
