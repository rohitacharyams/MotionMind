# Contributing to MotionMind

Thanks for your interest! MotionMind is an open pipeline for turning video into
understandable, controllable avatar motion. Contributions of all sizes are welcome.

## Getting started

```bash
git clone <your-fork-url> motionmind && cd motionmind
python -m venv .venv && . .venv/Scripts/activate   # Windows (use bin/activate on *nix)
pip install -r requirements.txt          # pipeline
pip install -r coach/requirements.txt    # coach/runtime
cp coach/.env.example coach/.env         # add your own keys
```

Run the no-GPU demo to confirm your setup:

```bash
python scripts/demo.py --style neon --trail
```

## Where help is most useful

- **One-command video→SMPL CLI** wrapping `scripts/video_to_smpl/`.
- **More export targets** — FBX / USD / glTF animation.
- **More retarget rigs** beyond VRM (Mixamo, Rigify, UE Mannequin).
- **A clean standalone viewer** decoupled from the coach app.
- **Docs & examples** — sample clips, notebooks, tutorials.
- **Tests** for the physics validator and retarget math.

## Guidelines

- Keep the pipeline **deterministic**. Structured motion in, structured motion out.
- Never commit secrets. `.env` is git-ignored; use `*.env.example` for new vars.
- Don't add third-party analytics/telemetry to the open-source build.
- Respect dataset/model licenses; don't commit SMPL, CMU, AIST++, or Mixamo assets.
- Match the existing code style; keep PRs focused and described.

## Reporting bugs

Open an issue with steps to reproduce, your OS/GPU, and the relevant log output.
For security issues, see [SECURITY.md](SECURITY.md).
