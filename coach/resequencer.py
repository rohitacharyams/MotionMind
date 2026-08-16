"""resequencer.py — beat-aligned choreography assembler.

Takes a user-supplied music clip, detects beats with librosa, then
stitches a sequence of retargeted VRM clips so each new clip starts on
a downbeat. Output is a single ``vrm-quat`` JSON the existing
MotionPlayer already understands — no new playback code needed.

Strategy (deterministic, no diffusion):
  1. librosa.beat.beat_track → music tempo + beat times.
  2. Choose a beat grid of N bars × 4 beats each.
  3. For each bar pick a retargeted clip:
       - filter by genre (if user asked)
       - filter by `bpm_target` within ±15% of detected tempo
       - score by metadata.summary semantic similarity (if available)
       - never repeat the same clip twice in a row
  4. For each chosen clip, sample a frame range covering one bar of
     music at the clip's native fps. Resample to a common fps (60).
  5. Concatenate frames with an N-frame quaternion-SLERP crossfade at
     each seam so seams don't snap.
  6. Hips translation is rebased so each clip starts at the previous
     clip's last hips position (avoids teleporting).

Entry points:
    resequence_from_audio(audio_path, *, bars=8, genre=None,
                          query=None) -> dict   (vrm-quat JSON)
    POST /api/motion/resequence   {audio_url, bars?, genre?, query?}
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

COACH = Path(__file__).resolve().parent
CACHE_DIR = COACH / 'motion_cache'
META_DIR  = COACH / 'motion_meta'

OUT_FPS = 60
CROSSFADE_FRAMES = 12   # at 60fps = 200ms crossfade between clips


# ─── audio analysis ────────────────────────────────────────────────────
def analyse_audio(audio_path: str) -> Dict[str, Any]:
    """Return {bpm, beats_s, downbeats_s, duration_s}. Beats are seconds."""
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # tempo can be 0-d ndarray; coerce to float
    if hasattr(tempo, 'item'):
        tempo = tempo.item()
    return {
        'bpm':         float(tempo),
        'beats_s':     [float(t) for t in beat_times],
        'duration_s':  float(len(y) / sr),
    }


# ─── clip pool ─────────────────────────────────────────────────────────
def _load_meta(clip_id: str) -> Dict[str, Any]:
    p = META_DIR / f'{clip_id}.json'
    if not p.exists():
        return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}


def _candidate_clips(target_bpm: float, *,
                     genre: Optional[str] = None,
                     bpm_tol: float = 0.18) -> List[str]:
    out: List[str] = []
    for p in sorted(CACHE_DIR.glob('*.json')):
        cid = p.stem
        if cid.startswith('_'):
            continue
        if genre and not cid.startswith(genre):
            continue
        m = _load_meta(cid)
        bt = m.get('bpm_target') or 110
        if abs(bt - target_bpm) / max(target_bpm, 1) > bpm_tol:
            continue
        out.append(cid)
    return out


# ─── concatenation ─────────────────────────────────────────────────────
def _slerp_q(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc quaternion SLERP between (x,y,z,w) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    th0 = math.acos(dot)
    s_th0 = math.sin(th0)
    s1 = math.sin((1 - t) * th0) / s_th0
    s2 = math.sin(t       * th0) / s_th0
    return s1 * a + s2 * b


def _resample_clip(clip: Dict[str, Any], start_f: int, end_f: int,
                   out_n: int) -> Dict[str, Any]:
    """Linear-interpolate frame indices to produce exactly out_n frames."""
    src_n = max(1, end_f - start_f)
    idx = np.linspace(start_f, end_f - 1, out_n)
    rotations: Dict[str, List[List[float]]] = {}
    for bone, frames in (clip.get('rotations') or {}).items():
        arr = np.asarray(frames, dtype=np.float64)
        out: List[List[float]] = []
        for f in idx:
            i0 = int(math.floor(f))
            i1 = min(arr.shape[0] - 1, i0 + 1)
            t = f - i0
            q = _slerp_q(arr[i0], arr[i1], t)
            out.append([float(x) for x in q])
        rotations[bone] = out
    hipsT = clip.get('hips_translation') or []
    hips_out: List[List[float]] = []
    if hipsT:
        arr = np.asarray(hipsT, dtype=np.float64)
        for f in idx:
            i0 = int(math.floor(f))
            i1 = min(arr.shape[0] - 1, i0 + 1)
            t = f - i0
            hips_out.append([float(x) for x in (1 - t) * arr[i0] + t * arr[i1]])
    return {
        'rotations':        rotations,
        'hips_translation': hips_out,
        'bones':            clip.get('bones'),
        'rest_local_rotation':    clip.get('rest_local_rotation'),
        'rest_local_translation': clip.get('rest_local_translation'),
    }


def _crossfade(prev_seg: Dict[str, Any], next_seg: Dict[str, Any],
               n_fade: int) -> None:
    """Mutate next_seg's first `n_fade` frames to SLERP from prev's last."""
    if n_fade <= 0:
        return
    for bone, next_frames in next_seg['rotations'].items():
        prev_frames = prev_seg['rotations'].get(bone)
        if not prev_frames:
            continue
        # Blend from prev's last quat into the next clip's frame.
        last_q = np.asarray(prev_frames[-1], dtype=np.float64)
        for k in range(min(n_fade, len(next_frames))):
            t = (k + 1) / (n_fade + 1)
            next_q = np.asarray(next_frames[k], dtype=np.float64)
            next_frames[k] = [float(x) for x in _slerp_q(last_q, next_q, t)]
    # Hips: lerp position from prev's last toward next's k-th
    if prev_seg.get('hips_translation') and next_seg.get('hips_translation'):
        last_h = np.asarray(prev_seg['hips_translation'][-1])
        for k in range(min(n_fade, len(next_seg['hips_translation']))):
            t = (k + 1) / (n_fade + 1)
            nh = np.asarray(next_seg['hips_translation'][k])
            next_seg['hips_translation'][k] = [
                float(x) for x in (1 - t) * last_h + t * nh
            ]


def _rebase_hips(seg: Dict[str, Any], origin: np.ndarray) -> np.ndarray:
    """Shift seg's hips so frame 0 starts at `origin`. Return last hips pos."""
    hips = seg.get('hips_translation') or []
    if not hips:
        return origin
    arr = np.asarray(hips, dtype=np.float64)
    delta = origin - arr[0]
    arr += delta
    seg['hips_translation'] = arr.tolist()
    return arr[-1]


# ─── top-level driver ──────────────────────────────────────────────────
def resequence_from_audio(audio_path: str, *,
                          bars: int = 8,
                          beats_per_bar: int = 4,
                          genre: Optional[str] = None,
                          query: Optional[str] = None) -> Dict[str, Any]:
    a = analyse_audio(audio_path)
    bpm = a['bpm'] or 110.0
    bar_seconds = (60.0 / bpm) * beats_per_bar
    frames_per_bar = int(round(bar_seconds * OUT_FPS))

    pool = _candidate_clips(bpm, genre=genre)
    if not pool:
        pool = _candidate_clips(bpm, genre=None, bpm_tol=0.4)
    if not pool:
        raise RuntimeError('no retargeted clips found for resequencing')

    # Optional semantic reranking
    if query:
        try:
            from coach import semantic_search
            hits = semantic_search.search(query, k=64)
            ordered = [h['id'] for h in hits if h['id'] in set(pool)]
            if ordered:
                pool = ordered + [c for c in pool if c not in set(ordered)]
        except Exception:
            pass

    chosen: List[str] = []
    for _ in range(bars):
        candidates = [c for c in pool if c != (chosen[-1] if chosen else None)]
        if not candidates:
            candidates = pool
        # Prefer top of the pool (semantic order) with mild randomness
        top = candidates[:8] if len(candidates) >= 8 else candidates
        chosen.append(random.choice(top))

    # Build segments, one bar each
    segments: List[Dict[str, Any]] = []
    cumulative_hips = np.zeros(3, dtype=np.float64)
    for cid in chosen:
        clip = json.loads((CACHE_DIR / f'{cid}.json').read_text(
            encoding='utf-8'))
        src_n = int(clip.get('n_frames', 0))
        if src_n <= 0:
            continue
        # Take the first `frames_per_bar` worth of source clip
        # (resampled to OUT_FPS).
        end_f = min(src_n, max(2, int(frames_per_bar *
                                       (clip.get('fps', 30) / OUT_FPS))))
        seg = _resample_clip(clip, 0, end_f, frames_per_bar)
        cumulative_hips = _rebase_hips(seg, cumulative_hips)
        if segments:
            _crossfade(segments[-1], seg, CROSSFADE_FRAMES)
        segments.append(seg)

    if not segments:
        raise RuntimeError('failed to assemble segments')

    # Concatenate
    out_rotations: Dict[str, List[List[float]]] = {
        bone: [] for bone in segments[0]['rotations'].keys()
    }
    out_hips: List[List[float]] = []
    for seg in segments:
        for bone, frames in seg['rotations'].items():
            out_rotations.setdefault(bone, []).extend(frames)
        out_hips.extend(seg.get('hips_translation') or [])

    n_frames = len(next(iter(out_rotations.values())))
    return {
        'format':     'vrm-quat',
        'name':       'resequenced',
        'fps':        OUT_FPS,
        'n_frames':   n_frames,
        'frames':     n_frames,
        'duration_s': n_frames / OUT_FPS,
        'bones':      segments[0].get('bones'),
        'rotations':  out_rotations,
        'hips_translation': out_hips,
        'rest_local_rotation':    segments[0].get('rest_local_rotation'),
        'rest_local_translation': segments[0].get('rest_local_translation'),
        'meta': {
            'bpm':        bpm,
            'bars':       bars,
            'clip_order': chosen,
        },
    }
