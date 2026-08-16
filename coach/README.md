# Dance.AI Coach — Pilot

Voice-driven AI dance coach. Hands-free.

## Stack

| Layer | Tech | Cost |
|---|---|---|
| STT (streaming) | Azure Speech (F0 free) | $0 — 5 hr/mo |
| TTS (streaming, with visemes) | Azure Speech Neural TTS (F0 free) | $0 — 0.5M chars/mo |
| LLM | Groq Llama-3.3-70B (free tier) | $0 |
| Vision (pose) | MediaPipe Pose Landmarker (browser) | $0 |
| Vision (semantic, optional) | Qwen2.5-VL-7B via HF Inference (free tier) | $0 |
| Avatar runtime | three.js + @pixiv/three-vrm | $0 |
| Backend | FastAPI + WebSocket | $0 |
| Motion DB | AIST++ (already on disk) | $0 |

## Physics safety (defense-in-depth)

1. **Offline validator** — `python -m coach.physics_validator scan` flags every clip in the motion DB. Failed clips never reach the avatar.
2. **Composer post-conditions** — every composed `.pkl` is re-validated before save.
3. **Runtime guard** — browser `MotionPlayer` clamps each quaternion just before bone write. If a value exceeds bounds, the avatar freezes on the last safe frame and a `motion.flagged` event fires.

### Dataset QA + deterministic auto-fix

Run this before shipping new motion packs (and periodically on the full DB):

```bash
python -m coach.ingestion.motion_qa_pipeline --fix
```

Outputs:
- `coach/reports/motion_qa_aist.json`
- `coach/reports/motion_qa_cmu.json`
- `coach/reports/motion_qa_aist_broken.json`
- `coach/reports/motion_qa_cmu_broken.json`

This pass:
- clamps absurd axis-angle magnitudes,
- floor-anchors root translation,
- recenters XY trajectory for studio framing,
- emits exact lists of clips that still warn/fail.

For already-ingested CMU packs that look slanted/lying in studio checks,
run a one-time orientation repair:

```bash
python -m coach.ingestion.fix_existing_cmu_orientation
```

## Setup

1. Create `.env` from `.env.example`:
   ```
   AZURE_SPEECH_KEY=...
   AZURE_SPEECH_REGION=eastus
   GROQ_API_KEY=...           # https://console.groq.com/keys (free)
   ```
2. Install Python deps:
   ```
   py -3.12 -m pip install -r requirements.txt
   ```
3. Run the backend:
   ```
   py -3.12 -m uvicorn coach.server:app --host 127.0.0.1 --port 8770 --reload
   ```
4. Open http://127.0.0.1:8770/ in Chrome/Edge.

## Directory layout

```
coach/
├── server.py              FastAPI + WS routes
├── physics_validator.py   Joint angle / velocity / accel checks
├── motion_index.py        Scans motion_db, exposes /api/motion/list
├── choreographer/
│   ├── agent.py           LLM tool-calling loop (Groq)
│   ├── tools.py           pick_clip, play, drill, slower, mirror, explain
│   ├── ontology.yaml      Move knowledge graph (genres, families, prereqs)
│   └── prompts.py         System prompt for the dance coach
├── speech/
│   ├── azure_token.py     Mint ephemeral tokens for browser Speech SDK
│   └── tts_visemes.py     Lipsync metadata helpers
└── static/
    ├── coach.html         App shell
    ├── coach.js           three.js + VRM + motion player + voice loop
    ├── motion_player.js   Drives VRM bones from .pkl frames + runtime guard
    └── azure_voice.js     STT/TTS browser bindings
```
