"""runpod_handler.py — RunPod Serverless worker: video → SMPL (+ keyframes).

This is the programmatic twin of the manual Colab/RunPod flow documented in
README.md. Deploy it as a **RunPod Serverless** endpoint (see Dockerfile in this
folder). Every request spins a worker from the pre-baked image — GVHMR and all
its deps + checkpoints are already installed, so there is NO per-request setup;
only inference runs. The endpoint scales to zero when idle.

Contract
--------
Input  (event["input"]):
    video_url      str   HTTPS url to the source video (preferred; e.g. a blob url)
    video_b64      str   OR base64 of the raw video bytes (small clips only)
    static_camera  bool  pass GVHMR `-s` (skip SLAM) for tripod/static shots (default True)
    fps            int   output motion fps written into the pkl (default 30)
    n_keyframes    int   evenly-spaced JPEG frames to return for step segmentation (default 12)
    root_fix       str   one of gvhmr_to_aist.ROOT_FIX keys (default "none")

Output (returned dict — RunPod delivers this to the webhook / poll result):
    ok             bool
    smpl_b64       str   base64 of the AIST-style SMPL .pkl (tiny; a few hundred KB)
    fps            float
    n_frames       int
    keyframes      [str] base64 JPEGs, evenly spaced across the clip
    keyframe_times [float] seconds for each keyframe (segmentation boundary hints)
    error          str   present only on failure

The retarget SMPL→VRM clip step is intentionally NOT done here — it runs on the
always-on dan box (coach `/api/learn/build`) which owns the VRM rig + coach libs.
This worker's only job is the GPU-bound extraction.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

# GVHMR checkout baked into the image. Override with env if you mount elsewhere.
GVHMR_DIR = Path(os.getenv('GVHMR_DIR', '/app/GVHMR'))
# gvhmr_to_aist.py is copied next to this handler in the image.
HERE = Path(__file__).resolve().parent
GVHMR_TO_AIST = HERE / 'gvhmr_to_aist.py'


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={'User-Agent': 'dance-runpod/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        f.write(r.read())


def _resolve_video(inp: dict, workdir: Path) -> Path:
    dst = workdir / 'input.mp4'
    if inp.get('video_url'):
        _download(str(inp['video_url']), dst)
    elif inp.get('video_b64'):
        dst.write_bytes(base64.b64decode(inp['video_b64']))
    else:
        raise ValueError('provide video_url or video_b64')
    if not dst.exists() or dst.stat().st_size == 0:
        raise ValueError('downloaded/decoded video is empty')
    return dst


def _trim_video(video: Path, workdir: Path, max_seconds: int) -> Path:
    """Trim the clip to the first `max_seconds` (reel-length cap). If ffmpeg is
    unavailable or the clip is already shorter, returns the original path."""
    if max_seconds <= 0:
        return video
    out = workdir / 'trimmed.mp4'
    try:
        proc = subprocess.run(
            ['ffmpeg', '-y', '-i', str(video), '-t', str(max_seconds),
             '-c', 'copy', str(out)],
            capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            # Stream-copy can fail on odd keyframes — fall back to a re-encode.
            proc = subprocess.run(
                ['ffmpeg', '-y', '-i', str(video), '-t', str(max_seconds),
                 str(out)],
                capture_output=True, text=True)
        if out.exists() and out.stat().st_size > 0:
            return out
    except Exception:
        pass
    return video


def _run_gvhmr(video: Path, static_camera: bool) -> Path:
    """Run the official GVHMR demo; return the hmr4d_results.pt path."""
    cmd = [sys.executable, 'tools/demo/demo.py', '--video', str(video)]
    if static_camera:
        cmd.append('-s')  # static camera → skip the slow SLAM stage
    proc = subprocess.run(cmd, cwd=str(GVHMR_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            'GVHMR demo failed:\n' + (proc.stderr or proc.stdout)[-4000:])
    stem = video.stem
    pt = GVHMR_DIR / 'outputs' / 'demo' / stem / 'hmr4d_results.pt'
    if not pt.exists():
        # Some GVHMR builds name the folder after the full filename; scan.
        cands = list((GVHMR_DIR / 'outputs' / 'demo').rglob('hmr4d_results.pt'))
        if not cands:
            raise RuntimeError('GVHMR produced no hmr4d_results.pt')
        pt = max(cands, key=lambda p: p.stat().st_mtime)
    return pt


def _to_smpl_pkl(pt: Path, out_pkl: Path, fps: int, root_fix: str) -> None:
    cmd = [sys.executable, str(GVHMR_TO_AIST), '--source', 'gvhmr',
           '--in', str(pt), '--out', str(out_pkl), '--fps', str(fps)]
    if root_fix and root_fix != 'none':
        cmd += ['--root-fix', root_fix]
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            'gvhmr_to_aist failed:\n' + (proc.stderr or proc.stdout)[-4000:])


def _pkl_stats(pkl: Path):
    import pickle
    import numpy as np
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    poses = np.asarray(d['smpl_poses'])
    return int(poses.reshape(len(poses), -1).shape[0]), float(d.get('fps', 30.0))


def _sample_keyframes(video: Path, n: int):
    """Return (list[base64 jpeg], list[seconds]) evenly spaced across the clip."""
    if n <= 0:
        return [], []
    try:
        import cv2  # type: ignore
    except Exception:
        return _sample_keyframes_imageio(video, n)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 0:
        cap.release()
        return _sample_keyframes_imageio(video, n)
    idxs = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
    frames, times = [], []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode('ascii'))
            times.append(round(idx / vfps, 3))
    cap.release()
    return frames, times


def _sample_keyframes_imageio(video: Path, n: int):
    try:
        import imageio.v3 as iio  # type: ignore
        import numpy as np
        from io import BytesIO
        from PIL import Image  # type: ignore
    except Exception:
        return [], []
    frames_all = list(iio.imiter(str(video)))
    total = len(frames_all)
    if total == 0:
        return [], []
    vfps = 30.0
    idxs = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
    frames, times = [], []
    for idx in idxs:
        buf = BytesIO()
        Image.fromarray(np.asarray(frames_all[idx])).convert('RGB').save(
            buf, format='JPEG', quality=80)
        frames.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        times.append(round(idx / vfps, 3))
    return frames, times


def process(inp: dict) -> dict:
    fps = int(inp.get('fps', 30))
    static_camera = bool(inp.get('static_camera', True))
    n_keyframes = int(inp.get('n_keyframes', 12))
    root_fix = str(inp.get('root_fix', 'none'))
    max_seconds = int(inp.get('max_seconds', 0) or 0)   # reel-length cap (0 = off)

    with tempfile.TemporaryDirectory(prefix='v2smpl_') as td:
        work = Path(td)
        video = _resolve_video(inp, work)
        if max_seconds > 0:
            video = _trim_video(video, work, max_seconds)
        pt = _run_gvhmr(video, static_camera)
        out_pkl = work / 'motion.pkl'
        _to_smpl_pkl(pt, out_pkl, fps, root_fix)
        n_frames, real_fps = _pkl_stats(out_pkl)
        keyframes, times = _sample_keyframes(video, n_keyframes)
        return {
            'ok': True,
            'smpl_b64': base64.b64encode(out_pkl.read_bytes()).decode('ascii'),
            'fps': real_fps,
            'n_frames': n_frames,
            'keyframes': keyframes,
            'keyframe_times': times,
        }


def handler(event: dict) -> dict:
    """RunPod serverless entrypoint."""
    try:
        return process((event or {}).get('input') or {})
    except Exception as e:  # noqa: BLE001
        return {'ok': False, 'error': str(e),
                'trace': traceback.format_exc()[-2000:]}


if __name__ == '__main__':
    # When RUNPOD is present, start the serverless loop. Otherwise allow a local
    # smoke test:  python runpod_handler.py '{"input":{"video_url":"..."}}'
    try:
        import runpod  # type: ignore
        runpod.serverless.start({'handler': handler})
    except ImportError:
        import json
        arg = sys.argv[1] if len(sys.argv) > 1 else '{"input":{}}'
        out = handler(json.loads(arg))
        out.pop('smpl_b64', None)  # keep local stdout readable
        out['keyframes'] = f'<{len(out.get("keyframes", []))} frames>'
        print(json.dumps(out, indent=2))
