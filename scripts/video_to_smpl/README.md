# video → SMPL → avatar (prototype)

Goal: take **any dance video**, extract a **SMPL motion file** from it on a
rented GPU, and load it into the existing VRM avatar / coach so the avatar
performs that dance. This is the only missing stage — the repo already ingests
SMPL for AIST++ clips, retargets to the VRM rig, fixes quaternions, and gates on
orientation. We just add a *video → SMPL* front end and reuse everything after.

```
dance video ──[GVHMR on RunPod]──▶ hmr4d_results.pt (SMPL-X, world frame)
        │
        ├─ gvhmr_to_aist.py ─────▶ my_clip.pkl   {smpl_poses (T,72), smpl_trans, fps}
        │                          (download this tiny file to your PC)
        │
        └─ build_clip.py ────────▶ export_motion_json.py  (SMPL → VRM-quat)
                                   fix_quaternion_signs    (continuity)
                                   physics_validator       (safety/orientation gate)
                                   coach/motion_cache/my_clip.json  ← avatar dances it
```

## Which model & why

- **GVHMR** (default) — world-grounded, gravity-aware → keeps feet on the floor,
  minimal skating/drift, avatar stays upright. Has a static-camera flag (`-s`)
  so tripod-shot tutorials skip the slow SLAM step. Best for dance.
- **WHAM** (fallback) — use for handheld / moving-camera footage.
- Both are free/open-source (MIT). The only wall is the **free** SMPL/SMPL-X
  academic registration (no payment).

Cost: an RTX 4090 on RunPod is ~$0.3–0.7/hr; a 15–30s clip processes in a few
minutes → pennies per clip.

---

## FASTEST path — free Colab (recommended for prototyping)

Skip RunPod entirely. Use [`gvhmr_colab_export.ipynb`](gvhmr_colab_export.ipynb)
— it's the **official GVHMR Colab** plus one export cell that writes the repo's
`.pkl` and downloads it:

1. Open the notebook in Google Colab, set **Runtime → GPU (T4)**.
2. Run cells 1–2 (install + checkpoints), skip DPVO (cell 1b) if your camera is static.
3. Upload your dance video (cell 3), run GVHMR with `-s` (cell 4).
4. Cell 5 converts `hmr4d_results.pt` → `<name>.pkl`; cell 6 downloads it.
5. Locally: `python scripts/video_to_smpl/build_clip.py --aist <name>.pkl --name <name> --preview`.

That converter cell is the inline twin of [`gvhmr_to_aist.py`](gvhmr_to_aist.py).
Use the RunPod path below only when you outgrow free Colab (batch jobs, longer
videos, no session timeouts).

---

## Part A — one-time pod setup (RunPod alternative)

1. Launch a RunPod pod: **RTX 4090 (24 GB)**, PyTorch 2.x / CUDA 12.1 template,
   with a **Network Volume mounted at `/workspace`**.
2. In the pod terminal, run [`runpod_setup.sh`](runpod_setup.sh) (clones GVHMR,
   installs deps). Then download GVHMR's checkpoints + SMPL(-X) bodies per its
   `docs/INSTALL.md` into `inputs/checkpoints/...` (persists on the volume).
3. Smoke-test GVHMR itself:
   ```bash
   python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s
   ```
   You should get `outputs/demo/tennis/hmr4d_results.pt` plus side-by-side
   render mp4s. If those look right, GVHMR is good.
4. Upload [`gvhmr_to_aist.py`](gvhmr_to_aist.py) to `/workspace/GVHMR/`.

## Part B — per video (the loop you repeat to VERIFY)

**On the pod:**

1. Upload your dance video to the pod (web file browser / `runpodctl send`).
2. Extract SMPL. Use `-s` if the camera is static (most tutorials):
   ```bash
   python tools/demo/demo.py --video inputs/my_dance.mp4 -s
   ```
   → `outputs/demo/my_dance/hmr4d_results.pt`.
   **First sanity check:** open the auto-generated `*_global.mp4` render. If the
   GVHMR mesh matches the dancer, the SMPL is good *before* you ever touch the
   avatar.
3. Convert to the repo's SMPL pkl:
   ```bash
   python gvhmr_to_aist.py --source gvhmr \
       --in outputs/demo/my_dance/hmr4d_results.pt \
       --out my_dance.pkl --fps 30
   ```
   It prints frames / finite / pose range / height range — a fast numeric check.
4. Download `my_dance.pkl` to your PC (it's small).

**On your PC (in `c:\dan`):**

5. Run the chain + safety gate, and render a preview:
   ```powershell
   python scripts/video_to_smpl/build_clip.py `
       --aist my_dance.pkl --name my_dance `
       --vrm data/models/extra/AliciaSolid.vrm `
       --preview
   ```
   - If it prints **`passed=True`** and installs to
     `coach/motion_cache/my_dance.json`, the extraction is clean.
   - `data/output_videos/v2smpl_my_dance.mp4` is the avatar performing it —
     **this is your visual proof** that a random video → correct SMPL worked.
6. Or view it live: start the coach and open the viewer; the clip is served at
   `/api/motion/data/my_dance.json` and appears in `/api/motion/list`.

## Verifying correctness (the checklist you asked for)

| # | Check | Where | Pass looks like |
|---|-------|-------|-----------------|
| 1 | GVHMR mesh tracks the dancer | pod `*_global.mp4` | mesh limbs follow the person |
| 2 | pkl is finite, sane ranges | `gvhmr_to_aist.py` stdout | `finite=True`, pose within ±3.2 rad |
| 3 | retarget succeeds | `build_clip.py` step 2 | JSON written, no error |
| 4 | physics/orientation gate | `build_clip.py` step 4 | `passed=True severity=ok` |
| 5 | avatar looks right | `--preview` mp4 | avatar upright, does the moves |

If **step 4 fails** or **step 5 looks lying-down / rotated**: re-run
`gvhmr_to_aist.py` with `--root-fix x-90` (or `x+90` / `y180`) and rebuild —
it's a cheap coordinate-frame retry. Jittery limbs → the video had occlusion /
fast spins; trim with `--trim-start/--trim-end` or pick cleaner footage.

## Known limits (honest)

- **No fingers.** SMPL/GVHMR body pose has no hand articulation. Your rig only
  has single `LeftHand`/`RightHand` bones, so this matches current fidelity.
- **Monocular depth ambiguity.** One camera can't perfectly recover limb depth;
  crossed limbs / heavy occlusion need cleaner footage. That's exactly what the
  step-4 gate is for.
- **Retarget was tuned on clean AIST++.** Video-derived SMPL is noisier; the
  sign-fix + gate handle most of it, but very noisy clips may need light
  smoothing before they look polished on the avatar.

## Files here

- [`gvhmr_to_aist.py`](gvhmr_to_aist.py) — GVHMR/WHAM output → repo SMPL pkl (run on pod).
- [`build_clip.py`](build_clip.py) — chains retarget → sign-fix → validate → install (run on PC).
- [`runpod_setup.sh`](runpod_setup.sh) — pod bootstrap.
