#!/usr/bin/env bash
# runpod_setup.sh — one-time bootstrap for GVHMR on a RunPod pod.
#
# Pod recommendation (all "money-free" except the hourly GPU rental):
#   GPU      : RTX 4090 (24 GB) or A5000 — GVHMR needs ~8-12 GB. NOT an A100.
#   Template : "RunPod PyTorch 2.1" (CUDA 12.1) or any PyTorch 2.x image.
#   Volume   : attach a Network Volume mounted at /workspace so the ~4 GB of
#              checkpoints + SMPL/SMPL-X bodies persist across pod restarts and
#              you pay the download cost only ONCE.
#
# GVHMR + SMPL/SMPL-X model files require a FREE academic registration:
#   - https://smpl.is.tue.mpg.de/     (SMPL)
#   - https://smplify.is.tue.mpg.de/  (SMPLify)
#   - https://smpl-x.is.tue.mpg.de/   (SMPL-X, used by GVHMR internally)
# No payment — just an account. Same registration you already use in this repo.
set -e
cd /workspace

# ── 1. system deps ────────────────────────────────────────────────────────
apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0

# ── 2. clone GVHMR ────────────────────────────────────────────────────────
if [ ! -d GVHMR ]; then
  git clone https://github.com/zju3dv/GVHMR.git --recursive
fi
cd GVHMR

# ── 3. python env ─────────────────────────────────────────────────────────
# Follow docs/INSTALL.md. In short:
pip install -e .
pip install numpy==1.26.4                      # avoid numpy 2.x ABI breaks
# pytorch3d wheel matching the pod's torch/cuda (INSTALL.md lists the exact one)
# e.g. for torch2.1+cu121:
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ── 4. checkpoints + body models ──────────────────────────────────────────
# GVHMR ships a helper / Google-Drive links in docs/INSTALL.md. Place them as:
#   inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
#   inputs/checkpoints/{yolo,vitpose,hmr2,dpvo}/...
#   third-party body models under hmr4d/utils/body_model/
# (Download once; the /workspace network volume keeps them.)
echo "Place GVHMR checkpoints + SMPL(-X) bodies per docs/INSTALL.md, then verify:"
echo "  python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s"

# ── 5. drop the adapter next to the demo so it can read torch .pt ─────────
# Upload scripts/video_to_smpl/gvhmr_to_aist.py to /workspace/GVHMR/ (scp / runpodctl / the web file browser).
echo "Setup done. See README.md for the per-video run + verify loop."
