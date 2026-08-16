# Architecture

MotionMind is a **deterministic motion pipeline** with a semantic understanding
layer and a realtime coaching runtime on top. This document explains each stage,
the data that flows between them, and where to get the models/datasets the code
expects.

## Data flow

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐
│   video     │──►│ pose extract │──►│  SMPL fitting │──►│  retarget to   │
│  (any clip) │   │  RTMPose /   │   │  GVHMR / WHAM │   │  VRM humanoid  │
└─────────────┘   │  MediaPipe   │   │  (world frame)│   │  rig           │
                  └──────────────┘   └───────────────┘   └───────┬────────┘
                                                                 │
                        ┌────────────────────────────────────────┘
                        ▼
┌────────────────┐   ┌────────────────┐   ┌───────────────────┐
│ quaternion fix │──►│ physics gate   │──►│ motion cache      │
│ (continuity)   │   │ (angle/vel)    │   │ (.json per clip)  │
└────────────────┘   └────────────────┘   └─────────┬─────────┘
                                                     │
        ┌────────────────────────────────────────────┘
        ▼
┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ motion index + │──►│ LLM choreographer  │──►│ browser avatar     │
│ FAISS / search │   │ (tool-calling)     │   │ three.js + VRM     │
└────────────────┘   └─────────┬──────────┘   └────────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │ Gemini Live voice  │  ◄── you talk, it teaches
                     │ (Azure fallback)   │
                     └────────────────────┘
```

## Stage 1 — Pose extraction (`src/pose_extraction/`)

Detects whole-body keypoints from each frame.

- **RTMPose (MMPose)** — 133-keypoint whole-body (body + hands + face), best
  accuracy. GPU recommended.
- **MediaPipe** — lightweight, runs on CPU / in the browser. Used for the live
  feedback loop.

Output: per-frame 2D/3D keypoint arrays.

## Stage 2 — SMPL fitting (`scripts/video_to_smpl/`)

Lifts 2D video into a **world-grounded 3D body** as SMPL/SMPL-X parameters
(`(T, 72)` pose + translation + fps).

- **GVHMR** (default) — gravity-aware, keeps feet on the floor, minimal skating.
  Has a static-camera flag for tripod footage. Best for dance/movement.
- **WHAM** (fallback) — for handheld / moving-camera footage.

Runners included: a **Colab notebook** (free T4), a **RunPod** handler, and a
**Modal** app. Output is a small `.pkl` you download and feed into the retarget
step.

## Stage 3 — Retarget to avatar (`src/avatar/`, `scripts/export_motion_json.py`)

Maps SMPL joints onto a VRM humanoid rig and converts axis-angle rotations to
per-bone local quaternions the browser can apply directly.

- `scripts/export_motion_json.py` — SMPL → VRM-bone quaternions
- `scripts/fix_quaternion_signs.py` — enforces hemisphere continuity (kills 360°
  spin artifacts)
- Also exports **BVH** (`export_bvh.py`) and **GLB** (`export_glb.py`) for
  Blender / game engines.

## Stage 4 — Physics safety gate (`coach/physics_validator.py`)

Defense in depth so an avatar never contorts impossibly:

1. **Offline validator** — scans every clip; failed clips never reach the avatar.
2. **Composer post-conditions** — every composed clip is re-validated before save.
3. **Runtime guard** — the browser `MotionPlayer` clamps each quaternion just
   before it writes to a bone; on a violation it freezes on the last safe frame.

```bash
python -m coach.physics_validator scan
```

## Stage 5 — Motion library + search (`coach/motion_index.py`, `coach/semantic_search.py`)

- `motion_index` scans the motion cache and exposes the safe-clip list to the API.
- `semantic_search` does natural-language lookup over the library (embeddings when
  `sentence-transformers` is installed; keyword fallback otherwise).
- `src/motion_processing/` stores motion vectors in HDF5 + a FAISS index for
  nearest-neighbour retrieval ("find moves like this one").

## Stage 6 — Motion understanding (`coach/choreographer/`)

- `ontology.yaml` — a knowledge graph of moves: genres, families, prerequisites.
- `agent.py` — an LLM tool-calling loop (Groq by default) that reasons about a
  clip and drives the avatar via tools.
- `tools.py` — the tool surface: `pick_clip`, `play`, `drill`, `slower`, `mirror`,
  `break_down`, `explain`, …
- `movement_composer.py` — blends/sequences clips into new routines.

This is the layer that turns raw joint data into *teachable understanding*.

## Stage 7 — Realtime coach (`coach/gemini_live.py`, `coach/server.py`)

- `server.py` — FastAPI + WebSocket. Serves the browser runtime and the clip API,
  and hosts the conversational loop (`/ws/agent`) and the voice loop (`/ws/voice`).
- `gemini_live.py` — bridges the browser mic/speaker to Google's Gemini Live API
  (native audio dialog) while **our** tool layer still drives the avatar — we
  never hand choreography to a black box. If `GEMINI_API_KEY` is unset it falls
  back to Azure STT → Groq → Azure TTS.
- `coach/static/` — three.js scene + `@pixiv/three-vrm` avatar + `motion_player.js`
  runtime that plays the retargeted motion with the runtime safety guard.

## Assets

The code is MIT; the following are downloaded separately under their own licenses.

| Asset | Where | License note |
|-------|-------|--------------|
| SMPL / SMPL-X body models | https://smpl.is.tue.mpg.de/ , https://smpl-x.is.tue.mpg.de/ | Free academic registration; non-commercial |
| GVHMR weights | https://github.com/zju3dv/GVHMR | See repo |
| WHAM weights | https://github.com/yohanshin/WHAM | See repo |
| HMR2 / 4D-Humans | https://github.com/shubham-goel/4D-Humans | See repo |
| CMU Mocap (optional) | http://mocap.cs.cmu.edu/ | Free |
| AIST++ (optional) | https://google.github.io/aistplusplus_dataset/ | CC BY 4.0 (conditions) |
| VRM avatars | e.g. VRoid Hub | Per-avatar license |

Large binary assets (models, motion caches, `.blend`, videos, VRMs) are
intentionally **git-ignored** — see [.gitignore](.gitignore). Place them in
`data/`, `coach/motion_cache/`, and `scripts/video_to_smpl/checkpoints/` after
cloning.

## What is intentionally NOT in this repo

This is a research/pipeline release. The consumer product's growth/ops layer
(analytics, email outreach, billing, cloud deploy scripts, SEO) is **not**
included. A few optional integration hooks remain (e.g. an external identity
backend via `STUDIOOS_API`); they are disabled by default. See [SECURITY.md](SECURITY.md).
