# MotionMind

**Teach an AI avatar to understand and perform any human movement — from a single video.**

MotionMind is an open-source pipeline that takes an ordinary video of a person
moving, lifts it into structured 3D motion (SMPL/SMPL-X body parameters),
retargets it onto any humanoid avatar (VRM), lets a language model *understand*
the motion semantically, and turns the whole thing into an interactive coach you
can **talk to in real time** (Gemini Live) to learn the movement yourself.

> Not another pixel-generator. MotionMind produces **structured, controllable
> motion** — joint rotations, SMPL params, BVH — so the avatar is fully
> deterministic and you own every frame. Diffusion video makes pretty pixels you
> can't control; MotionMind makes motion you can.

```
   video  ──►  extract SMPL  ──►  retarget to avatar  ──►  play in 3D
     🎥            🧍                    🤖                    🎬
                                                               │
                                              LLM understands the motion
                                              (kinematics, style, beats)
                                                               │
                                              Gemini Live talks you through it
                                                               │
                                                    👤 you learn the move
```

## Why this exists

Most "AI dance/motion" tools either (a) generate uncontrollable video frames, or
(b) lock the good parts behind a SaaS. MotionMind is the full deterministic
pipeline in the open:

- **Motion capture from wild video** — no mocap suit, no studio. Any phone clip.
- **Retarget to *your* character** — the same motion drives a stick figure, a
  cartoon, or a full VRM avatar.
- **Semantic motion understanding** — an LLM reasons about the move (what body
  parts, what style, what beat) and can break it into teachable steps.
- **Talk to it** — a realtime voice coach (Gemini Live, with an Azure fallback)
  that watches, narrates, corrects, and adapts.
- **Physics-safe** — every clip passes a joint-angle / velocity validator so the
  avatar never contorts into impossible poses.

## Architecture at a glance

```
                    ┌──────────────────────── pipeline (src/, scripts/) ───────────────────────┐
 video ─► pose ext ─► SMPL fit ─► retarget ─► quaternion fix ─► physics gate ─► motion cache (.json)
        (RTMPose /    (GVHMR /    (VRM rig)    (continuity)     (safety)          │
         MediaPipe)    WHAM)                                                      │
                                                                                  ▼
                    ┌──────────────────────── brain + runtime (coach/) ────────────────────────┐
 motion cache ─► motion index / FAISS ─► LLM choreographer (tool-calling) ─► avatar in browser
                  semantic search        + Gemini Live voice loop            (three.js + VRM)
```

Full details in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Repository layout

| Path | What it is |
|------|------------|
| `src/pose_extraction/` | Whole-body 2D/3D keypoint detection (RTMPose / MediaPipe) |
| `src/motion_processing/` | Normalize, smooth, embed, and store motion (HDF5 + FAISS) |
| `src/avatar/` | 2D skeleton, inverse kinematics, character rendering |
| `src/choreography/` | Motion mixing, SLERP/Bezier transitions, style transfer |
| `src/video/`, `src/scene/` | Video composition, effects, reel export |
| `src/pipeline.py` | High-level Python API tying the pipeline together |
| `scripts/video_to_smpl/` | **Video → SMPL** front-end (GVHMR/WHAM on Colab/RunPod/Modal) |
| `scripts/` | Extraction, retargeting, export (BVH/GLB/JSON), and QA scripts |
| `coach/choreographer/` | LLM tool-calling agent + move ontology that *understands* motion |
| `coach/gemini_live.py` | Realtime speech-to-speech coach bridge (Gemini Live) |
| `coach/motion_index.py` | Serves the safe-clip library to the browser |
| `coach/physics_validator.py` | Joint-angle / velocity / acceleration safety gate |
| `coach/semantic_search.py` | Natural-language search over the motion library |
| `coach/server.py` | FastAPI + WebSocket backend that drives the avatar |
| `coach/static/` | Browser avatar runtime (three.js + `@pixiv/three-vrm`) |

## The end-to-end flow

### 1. Video → SMPL (motion capture)
Run [`scripts/video_to_smpl/`](scripts/video_to_smpl/README.md). The fastest path
is the included Colab notebook (`gvhmr_colab_export.ipynb`): upload a clip, run
GVHMR on a free T4 GPU, and it exports a tiny `<name>.pkl` of SMPL parameters.
RunPod and Modal handlers are included for batch/production use.

```bash
# after you have <name>.pkl of SMPL params:
python scripts/video_to_smpl/build_clip.py --aist <name>.pkl --name my_move --preview
```

### 2. SMPL → avatar (retarget)
The SMPL body parameters are retargeted onto a VRM humanoid rig, quaternion signs
are made continuous, and the result is written to a motion cache the browser can play.

```bash
python scripts/export_motion_json.py --clip my_move       # SMPL → VRM-bone quaternions
python scripts/fix_quaternion_signs.py --clip my_move     # continuity
python -m coach.physics_validator scan                    # safety gate
```

### 3. Play it in 3D
Start the backend and open the browser runtime — the avatar performs the motion
in real time.

```bash
cp coach/.env.example coach/.env      # fill in keys (see below)
pip install -r coach/requirements.txt
python -m uvicorn coach.server:app --host 127.0.0.1 --port 8770 --reload
# open http://127.0.0.1:8770/
```

### 4. LLM understands the motion
The choreographer agent (`coach/choreographer/`) uses tool-calling over a move
**ontology** to reason about the clip — which body parts move, the style family,
prerequisites — and can break any move into teachable steps.

### 5. Gemini Live teaches you
With `GEMINI_API_KEY` set, the coach opens a realtime speech-to-speech channel:
it hears your voice, watches the avatar, narrates the steps, and adapts the pace.
Without a Gemini key it falls back to Azure STT → Groq LLM → Azure TTS.

## Quick start (pipeline only, no GPU)

```bash
git clone <your-fork-url> motionmind && cd motionmind
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python scripts/demo.py --style neon --trail        # render a demo, no GPU needed
```

## Configuration

All secrets are read from environment variables — **nothing is hardcoded**.
Copy the example files and fill in your own keys:

```bash
cp coach/.env.example coach/.env
cp .env.example .env         # optional, pipeline-side settings
```

| Variable | Purpose | Free tier? |
|----------|---------|-----------|
| `GROQ_API_KEY` | LLM brain (motion understanding / coaching) | ✅ groq.com |
| `AZURE_SPEECH_KEY` | Streaming STT + TTS voice | ✅ Azure F0 |
| `GEMINI_API_KEY` | Realtime speech-to-speech coach (optional) | ✅ aistudio.google.com |
| `HF_TOKEN` | Optional vision-language understanding | ✅ huggingface.co |

## Models & datasets you provide separately

The code is MIT, but the **models and motion data are not redistributed here**
(size + their own licenses). See [ARCHITECTURE.md](ARCHITECTURE.md#assets) for
download links: SMPL/SMPL-X bodies, GVHMR/WHAM weights, and (optionally) the
CMU / AIST++ motion libraries.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Good first areas: more
export targets (FBX/USD), additional retarget rigs, a cleaner standalone viewer,
and packaging the video→SMPL step as a one-command CLI.

## Security & responsible use

- Never commit real API keys. `.env` files are git-ignored; only `*.env.example`
  placeholders are tracked. See [SECURITY.md](SECURITY.md).
- Only capture motion from videos you have the rights to use.

## License

[MIT](LICENSE) for the code. Third-party models/datasets keep their own licenses.
