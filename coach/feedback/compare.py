"""compare.py — align student motion to reference and compute per-joint
angular error.

We pick a small set of "coachable" bones (shoulders, elbows, hips,
knees) and:
  1. For each bone in each motion, convert the quaternion stream into
     an axis-angle magnitude (instantaneous rotation amount) and the
     bone's local Y-axis swing direction. This gives us a low-dim
     signal robust to absolute orientation differences.
  2. DTW-align the two signals (across all watched bones simultaneously)
     with a 0.5x..2x speed band.
  3. For each aligned frame pair, compute the geodesic angle between
     student and reference quats per bone (in radians).
  4. Aggregate to per-bar statistics + a "worst bones" ranking.

We deliberately do NOT recompute world-space joint positions — that
would require running FK with a specific SMPL skeleton and adds
fragility. Bone-local quaternion deltas are the canonical metric.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

WATCH_BONES: Tuple[str, ...] = (
    'leftUpperArm',  'rightUpperArm',
    'leftLowerArm',  'rightLowerArm',
    'leftUpperLeg',  'rightUpperLeg',
    'leftLowerLeg',  'rightLowerLeg',
    'spine', 'chest', 'hips',
)


def _quat_to_axis_angle_mag(q: np.ndarray) -> float:
    """Return the rotation magnitude (radians) of a quat (x,y,z,w)."""
    w = max(-1.0, min(1.0, float(q[3])))
    return 2.0 * math.acos(abs(w))


def _quat_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest geodesic angle between two quaternions, in radians."""
    d = float(abs(np.dot(a, b)))
    d = max(-1.0, min(1.0, d))
    return 2.0 * math.acos(d)


def _signal_for(clip: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return per-bone array of frame-by-frame rotation magnitudes."""
    out: Dict[str, np.ndarray] = {}
    rots = clip.get('rotations') or {}
    for b in WATCH_BONES:
        frames = rots.get(b)
        if not frames:
            continue
        out[b] = np.array([_quat_to_axis_angle_mag(np.asarray(q))
                           for q in frames], dtype=np.float64)
    return out


def _dtw_path(s: np.ndarray, t: np.ndarray) -> List[Tuple[int, int]]:
    """Basic O(NM) DTW, band-constrained to ±20%. Returns aligned pairs."""
    n, m = len(s), len(t)
    band = max(8, int(0.2 * max(n, m)))
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
    # backtrack
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [(cost[i - 1, j - 1], (i - 1, j - 1)),
                   (cost[i - 1, j],     (i - 1, j)),
                   (cost[i, j - 1],     (i, j - 1))]
        choices.sort(key=lambda x: x[0])
        _, (i, j) = choices[0]
    path.reverse()
    return path


def compare(reference: Dict[str, Any],
            student:   Dict[str, Any]) -> Dict[str, Any]:
    """Time-align student to reference and return per-bone error stats."""
    ref_sig = _signal_for(reference)
    stu_sig = _signal_for(student)
    common = sorted(set(ref_sig) & set(stu_sig))
    if not common:
        return {'ok': False, 'reason': 'no shared bones'}

    # Use mean-across-bones signal for DTW (single 1-D series each).
    ref_mean = np.mean([ref_sig[b] for b in common], axis=0)
    stu_mean = np.mean([stu_sig[b] for b in common], axis=0)
    path = _dtw_path(ref_mean, stu_mean)

    # Per-bone geodesic error along the alignment
    ref_rots = reference.get('rotations') or {}
    stu_rots = student.get('rotations')   or {}
    per_bone: Dict[str, Dict[str, float]] = {}
    for b in common:
        rf = ref_rots.get(b); sf = stu_rots.get(b)
        if not (rf and sf):
            continue
        errs = []
        for (i, j) in path:
            if i >= len(rf) or j >= len(sf):
                continue
            errs.append(_quat_geodesic(np.asarray(rf[i]), np.asarray(sf[j])))
        if not errs:
            continue
        per_bone[b] = {
            'mean_deg': math.degrees(float(np.mean(errs))),
            'p90_deg':  math.degrees(float(np.percentile(errs, 90))),
            'max_deg':  math.degrees(float(np.max(errs))),
        }

    worst = sorted(per_bone.items(),
                   key=lambda kv: -kv[1]['mean_deg'])[:4]
    total_mean = float(np.mean([v['mean_deg'] for v in per_bone.values()])) \
                 if per_bone else 0.0
    return {
        'ok':              True,
        'aligned_pairs':   len(path),
        'mean_error_deg':  total_mean,
        'per_bone':        per_bone,
        'worst_bones':     [{'bone': b, **v} for b, v in worst],
        'student_frames':  student.get('n_frames'),
        'ref_frames':      reference.get('n_frames'),
    }
