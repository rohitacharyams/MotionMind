"""compare_2d.py — DTW-align student COCO-17 keypoints to the
precomputed reference (from `coach/motion_2d/*.npz`) and report a
per-keypoint normalised position error.

Why 2D and not 3D:
  - student keypoints come from MediaPipe Pose in the browser, which
    is rock-solid for 2D and noisy for 3D.
  - reference keypoints come from orthographic projection of the
    retargeted VRM-quat clip (see scripts/precompute_aist_2d_keypoints.py).
  - normalising both by torso length + recentring on the mid-hip
    makes comparison scale-, translation- and (mostly) viewpoint-
    invariant, as long as the student films front-on.

COCO-17 layout:
  0 nose  1 lEye  2 rEye  3 lEar  4 rEar
  5 lSh   6 rSh   7 lElb  8 rElb  9 lWri 10 rWri
 11 lHip 12 rHip 13 lKne 14 rKne 15 lAnk 16 rAnk
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Coachable keypoints (the ones whose error we actually surface to
# the LLM writer). Face landmarks are deliberately excluded — they're
# synthesised in the reference and we don't grade head pose anyway.
WATCH_KPT_IDX: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
WATCH_KPT_NAME: Tuple[str, ...] = (
    'left_shoulder', 'right_shoulder',
    'left_elbow',    'right_elbow',
    'left_wrist',    'right_wrist',
    'left_hip',      'right_hip',
    'left_knee',     'right_knee',
    'left_ankle',    'right_ankle',
)
NAME_OF = dict(zip(WATCH_KPT_IDX, WATCH_KPT_NAME))

L_HIP, R_HIP = 11, 12
L_SHO, R_SHO = 5, 6
EPS = 1e-6


def normalise(kp: np.ndarray) -> np.ndarray:
    """(T, 17, 3) → (T, 17, 2) in torso-units, centred on mid-hip.

    Per frame: subtract mid-hip; divide by per-frame torso length
    (mid-hip → mid-shoulder distance). Visibility column dropped
    (we re-attach it from the caller).
    """
    xy = kp[:, :, :2].astype(np.float64)
    mid_hip = 0.5 * (xy[:, L_HIP] + xy[:, R_HIP])
    mid_sho = 0.5 * (xy[:, L_SHO] + xy[:, R_SHO])
    torso = np.linalg.norm(mid_sho - mid_hip, axis=1)
    torso = np.where(torso < EPS, EPS, torso)
    centred = xy - mid_hip[:, None, :]
    return centred / torso[:, None, None]


def _energy_signal(norm_kp: np.ndarray) -> np.ndarray:
    """Per-frame motion-energy signal for DTW: mean ‖Δposition‖ across
    watched keypoints. Robust to absolute pose, sensitive to timing."""
    watched = norm_kp[:, list(WATCH_KPT_IDX), :]
    deltas = np.diff(watched, axis=0)
    energy = np.linalg.norm(deltas, axis=2).mean(axis=1)
    # Pad to original length.
    return np.concatenate([energy[:1], energy])


def _dtw_path(s: np.ndarray, t: np.ndarray,
              band_frac: float = 0.25) -> List[Tuple[int, int]]:
    """Sakoe-Chiba banded DTW. Returns aligned (i_ref, j_student) pairs."""
    n, m = len(s), len(t)
    band = max(8, int(band_frac * max(n, m)))
    INF = float('inf')
    cost = np.full((n + 1, m + 1), INF)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            c = abs(float(s[i - 1] - t[j - 1]))
            cost[i, j] = c + min(cost[i - 1, j],
                                 cost[i, j - 1],
                                 cost[i - 1, j - 1])
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = [(cost[i - 1, j - 1], (i - 1, j - 1)),
                (cost[i - 1, j],     (i - 1, j)),
                (cost[i, j - 1],     (i, j - 1))]
        prev.sort(key=lambda x: x[0])
        _, (i, j) = prev[0]
    path.reverse()
    return path


def load_reference(motion_2d_dir: Path, clip_id: str) -> np.ndarray:
    """Load and return raw (T, 17, 3) reference keypoints."""
    p = motion_2d_dir / f'{clip_id}.npz'
    if not p.exists():
        raise FileNotFoundError(f'no precomputed 2D for {clip_id}: {p}')
    return np.load(p)['keypoints']


def compare_2d(ref_kp: np.ndarray, stu_kp: np.ndarray,
               vis_thresh: float = 0.3) -> Dict[str, Any]:
    """Compare student vs reference COCO-17 keypoints.

    Both arrays are (T, 17, 3) — third channel = visibility (0..1).
    Reference visibility is synthetic (always 1.0 from projection);
    student visibility comes from MediaPipe. Frames where a student
    keypoint's visibility is below ``vis_thresh`` are excluded from
    that keypoint's error stats (but the frame is still aligned).
    """
    if ref_kp.ndim != 3 or stu_kp.ndim != 3 or ref_kp.shape[1] < 17 \
       or stu_kp.shape[1] < 17:
        return {'ok': False, 'reason': 'bad keypoint shape'}
    if ref_kp.shape[0] < 8 or stu_kp.shape[0] < 8:
        return {'ok': False, 'reason': 'too few frames'}

    ref_n = normalise(ref_kp)               # (Tr, 17, 2)
    stu_n = normalise(stu_kp)               # (Ts, 17, 2)
    stu_vis = stu_kp[:, :, 2].astype(np.float64)

    ref_e = _energy_signal(ref_n)
    stu_e = _energy_signal(stu_n)
    path = _dtw_path(ref_e, stu_e)

    # Per-keypoint Euclidean error along the alignment.
    per_kpt: Dict[str, Dict[str, float]] = {}
    per_kpt_timeline: Dict[str, List[float]] = {}
    frame_errors: List[float] = []
    for k in WATCH_KPT_IDX:
        errs: List[float] = []
        timeline: List[float] = []
        for (i, j) in path:
            v = stu_vis[j, k] if j < stu_vis.shape[0] else 0.0
            if v < vis_thresh:
                timeline.append(float('nan'))
                continue
            d = float(np.linalg.norm(ref_n[i, k] - stu_n[j, k]))
            errs.append(d)
            timeline.append(d)
        if not errs:
            continue
        per_kpt[NAME_OF[k]] = {
            'mean':    float(np.mean(errs)),
            'p90':     float(np.percentile(errs, 90)),
            'max':     float(np.max(errs)),
            'samples': len(errs),
        }
        per_kpt_timeline[NAME_OF[k]] = timeline

    if not per_kpt:
        return {'ok': False, 'reason': 'no visible keypoints'}

    # Total per-frame error (mean across visible watched keypoints).
    n_pairs = len(path)
    for fi in range(n_pairs):
        vals = [per_kpt_timeline[name][fi] for name in per_kpt
                if not math.isnan(per_kpt_timeline[name][fi])]
        frame_errors.append(float(np.mean(vals)) if vals else float('nan'))

    worst_frames_idx = sorted(
        ((e, fi) for fi, e in enumerate(frame_errors) if not math.isnan(e)),
        reverse=True,
    )[:5]
    worst_frames = [{'pair_index': fi,
                     'ref_frame':  int(path[fi][0]),
                     'student_frame': int(path[fi][1]),
                     'error':         float(e)}
                    for e, fi in worst_frames_idx]

    worst_kpts = sorted(per_kpt.items(), key=lambda kv: -kv[1]['mean'])[:4]
    mean_error = float(np.mean([v['mean'] for v in per_kpt.values()]))

    return {
        'ok':                True,
        'aligned_pairs':     n_pairs,
        'mean_error':        mean_error,        # in torso units
        'per_keypoint':      per_kpt,
        'worst_keypoints':   [{'keypoint': k, **v} for k, v in worst_kpts],
        'worst_frames':      worst_frames,
        'frame_errors':      frame_errors,      # for client-side highlight
        'student_frames':    int(stu_kp.shape[0]),
        'ref_frames':        int(ref_kp.shape[0]),
    }
