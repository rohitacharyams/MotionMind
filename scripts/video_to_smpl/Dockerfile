# Dockerfile — RunPod Serverless worker for video → SMPL (GVHMR).
#
# Build ONCE; every request reuses this image with all libs + checkpoints already
# baked in (no per-request install). Push to a registry, then create a RunPod
# Serverless endpoint from it.
#
#   docker build -t <registry>/dance-gvhmr:latest -f Dockerfile .
#   docker push  <registry>/dance-gvhmr:latest
#
# RunPod → Serverless → New Endpoint → Container image = <registry>/dance-gvhmr:latest
#   GPU: 24GB (RTX 4090/A5000).  Container disk: 20GB+.
#   Env: (none required by the handler; GVHMR_DIR defaults to /app/GVHMR)
#
# NOTE on checkpoints: GVHMR needs its checkpoints + the SMPL/SMPL-X bodies,
# which are license-gated (free academic registration). They are NOT committed
# here. Either (a) COPY them in at build time from a local `checkpoints/` you
# downloaded, or (b) mount a RunPod Network Volume at /app/GVHMR/inputs. The
# block below assumes option (a); comment it out for option (b).

FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    GVHMR_DIR=/app/GVHMR

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# 1) GVHMR + its Python deps (pinned by the repo's own requirements).
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /app/GVHMR
WORKDIR /app/GVHMR
RUN pip install -e . || pip install -r requirements.txt || true

# 2) Worker deps (RunPod SDK + frame sampling + video mux).
RUN pip install runpod opencv-python-headless imageio[ffmpeg] pillow numpy

# 3) License-gated checkpoints + SMPL bodies (option a — bake them in).
#    Place downloaded files under ./checkpoints/ next to this Dockerfile first.
#    Layout must match GVHMR docs/INSTALL.md (inputs/checkpoints/..., body_models/...).
# COPY checkpoints/ /app/GVHMR/inputs/

# 4) The handler + the SMPL converter.
WORKDIR /app/worker
COPY runpod_handler.py gvhmr_to_aist.py /app/worker/

CMD ["python", "-u", "/app/worker/runpod_handler.py"]
