"""motion_analyzer.py — derive plain-English beat cues from a clip.

The LLM cannot see the avatar; it only has the structured clip JSON.
For each retargeted vrm-quat clip we synthesise a short list of
8-count cues that describe WHICH BODY PART is doing the most work in
each count window and which way it moves. These cues feed into
`pick_and_play()` results so the coach can narrate the choreography
accurately ("count 3 — right arm sweeps up") instead of guessing.

Pure stdlib + numpy. ~5 ms per clip on cold cache.

Output schema (one entry per beat 1..8):
    {
        "beat": <int 1..8>,
        "frame_start": <int>,
        "frame_end": <int>,
        "cue":  "<plain-english one-liner>",
        "body_parts": ["<part>", ...]   # canonical labels
    }
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from coach import motion_index

CUES_DIR = Path(__file__).resolve().parent / 'motion_cues_v33'

# Bone → (canonical body-part label, English noun phrase).
# When a bone tops the velocity ranking in a count window we use the
# noun phrase verbatim in the cue. Keep these descriptions concrete
# — the LLM will weave them into a full coach line.
_BONE_LABELS: Dict[str, Dict[str, str]] = {
    'Hips':           {'part': 'hips',         'phrase': 'hips'},
    'Spine':          {'part': 'torso',        'phrase': 'torso'},
    'Spine2':         {'part': 'chest',        'phrase': 'chest'},
    'Neck':           {'part': 'neck',         'phrase': 'neck'},
    'Head':           {'part': 'head',         'phrase': 'head'},
    'LeftShoulder':   {'part': 'left_shoulder','phrase': 'left shoulder'},
    'RightShoulder':  {'part': 'right_shoulder','phrase': 'right shoulder'},
    'LeftArm':        {'part': 'left_arm',     'phrase': 'left arm'},
    'RightArm':       {'part': 'right_arm',    'phrase': 'right arm'},
    'LeftForeArm':    {'part': 'left_arm',     'phrase': 'left forearm'},
    'RightForeArm':   {'part': 'right_arm',    'phrase': 'right forearm'},
    'LeftHand':       {'part': 'left_hand',    'phrase': 'left hand'},
    'RightHand':      {'part': 'right_hand',   'phrase': 'right hand'},
    'LeftUpLeg':      {'part': 'left_leg',     'phrase': 'left leg'},
    'RightUpLeg':     {'part': 'right_leg',    'phrase': 'right leg'},
    'LeftLeg':        {'part': 'left_leg',     'phrase': 'left knee'},
    'RightLeg':       {'part': 'right_leg',    'phrase': 'right knee'},
    'LeftFoot':       {'part': 'left_foot',    'phrase': 'left foot'},
    'RightFoot':      {'part': 'right_foot',   'phrase': 'right foot'},
}

# Per-bone verb hints — DANCE-TEACHER VOCABULARY.
# v33: the previous engineering-talk ("forearm rotates inward",
# "knee plants down") was leaking into the coach's spoken lines.
# Real teachers say "arm up", "step back", "chest pop". Short,
# physical, and recognisable to a dancer — the LLM speaks these
# almost verbatim and they sound like a studio call.
_AXIS_VERBS: Dict[str, Dict[str, str]] = {
    'arm':  {'+x': 'arm up',     '-x': 'arm down',
             '+y': 'arm in',     '-y': 'arm out',
             '+z': 'reach',      '-z': 'pull back'},
    'leg':  {'+x': 'knee up',    '-x': 'drop',
             '+y': 'turn in',    '-y': 'turn out',
             '+z': 'step',       '-z': 'step back'},
    'torso':{'+x': 'lean back',  '-x': 'lean forward',
             '+y': 'twist',      '-y': 'twist',
             '+z': 'tilt',       '-z': 'tilt'},
    'head': {'+x': 'head up',    '-x': 'head down',
             '+y': 'head turn',  '-y': 'head turn',
             '+z': 'head tilt',  '-z': 'head tilt'},
    'hips': {'+x': 'hip back',   '-x': 'hip forward',
             '+y': 'hip swivel', '-y': 'hip swivel',
             '+z': 'hip pop',    '-z': 'hip pop'},
}


def _quat_delta_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions (radians)."""
    d = float(abs(np.dot(q0, q1)))
    d = max(-1.0, min(1.0, d))
    return 2.0 * math.acos(d)


def _dominant_axis(q0: np.ndarray, q1: np.ndarray) -> str:
    """Return '+x', '-x', '+y', '-y', '+z', or '-z' indicating which
    rotation axis the bone moved along most between two frames."""
    # Relative rotation q_rel = q0^-1 * q1; its xyz components are the
    # rotation axis scaled by sin(theta/2). Sign tells direction.
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    # Inverse of unit quaternion = conjugate.
    rx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
    ry = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
    rz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
    mags = [abs(rx), abs(ry), abs(rz)]
    i = int(np.argmax(mags))
    sign = '+' if (rx, ry, rz)[i] >= 0 else '-'
    axis = 'xyz'[i]
    return f'{sign}{axis}'


def _bone_family(bone: str) -> str:
    if 'Arm' in bone or 'Hand' in bone or 'Shoulder' in bone: return 'arm'
    if 'Leg' in bone or 'Foot' in bone:                       return 'leg'
    if bone in ('Spine', 'Spine2'):                            return 'torso'
    if bone in ('Head', 'Neck'):                               return 'head'
    return 'hips'


def _analyze_window(rotations: Dict[str, List[List[float]]],
                    f_start: int, f_end: int) -> Dict[str, Any]:
    """Find the top 1-2 movers across a frame window and describe."""
    f_end = min(f_end, max(f_start + 1,
                           min(len(v) for v in rotations.values())))
    if f_end <= f_start + 1:
        return {'movers': [], 'cue_parts': []}
    # Total angular travel per bone across the window.
    scores: List[tuple] = []
    for bone, frames in rotations.items():
        if bone not in _BONE_LABELS:
            continue
        if f_end > len(frames):
            continue
        total = 0.0
        for i in range(f_start, f_end - 1):
            q0 = np.asarray(frames[i],   dtype=np.float64)
            q1 = np.asarray(frames[i+1], dtype=np.float64)
            total += _quat_delta_angle(q0, q1)
        if total > 1e-3:
            # Direction = axis dominant in the start→end delta (not
            # per-frame avg — captures net travel).
            q_a = np.asarray(frames[f_start],  dtype=np.float64)
            q_b = np.asarray(frames[f_end-1],  dtype=np.float64)
            axis = _dominant_axis(q_a, q_b)
            scores.append((total, bone, axis))
    scores.sort(reverse=True)
    # v33: ONE phrase per beat (was 2). A real teacher calls ONE
    # cue per count — "step right", "chest pop", "arm up". Stacking
    # two clauses ("forearm rotates inward, left leg steps forward")
    # sounds like a robot reading a kinematics paper.
    pieces: List[str] = []
    parts: List[str] = []
    for total, bone, axis in scores:
        fam = _bone_family(bone)
        lbl = _BONE_LABELS[bone]
        verb = _AXIS_VERBS.get(fam, {}).get(axis, 'move')
        side = ''
        if 'left' in lbl['part']:
            side = 'left '
        elif 'right' in lbl['part']:
            side = 'right '
        # "arm up" / "step back" — short, dance-vocab. Side only when
        # the bone is laterally specific.
        phrase = f'{side}{verb}'.strip()
        pieces.append(phrase)
        parts.append(lbl['part'])
        break  # only the dominant mover
    return {'movers': pieces, 'cue_parts': parts}


def analyze_clip(clip_id: str, *, beats: int = 8,
                 cache: bool = True) -> Dict[str, Any]:
    """Compute 8-count cues for a clip. Result is cached on disk."""
    CUES_DIR.mkdir(exist_ok=True, parents=True)
    cache_path = CUES_DIR / f'{clip_id}.json'
    if cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass

    src = motion_index.get_cached_json(clip_id)
    if not src:
        return {'clip_id': clip_id, 'beats': [], 'error': 'no clip json'}
    data = json.loads(src.read_text(encoding='utf-8'))
    rotations = data.get('rotations') or {}
    if not rotations:
        return {'clip_id': clip_id, 'beats': [], 'error': 'no rotations'}
    n_frames = int(data.get('n_frames') or data.get('frames') or 0)
    if n_frames < beats * 2:
        beats = max(1, n_frames // 2)

    # For long clips, focus on the first ~8 counts of choreography
    # since the LLM teaches the OPENING of each move. Take the first
    # half of the clip (or all of it if short) and divide into N counts.
    span = min(n_frames, max(n_frames // 2, 60))
    step = max(1, span // beats)

    out_beats: List[Dict[str, Any]] = []
    for b in range(beats):
        f0 = b * step
        f1 = min(span, (b + 1) * step)
        info = _analyze_window(rotations, f0, f1)
        cue_text = ', '.join(info['movers']) if info['movers'] else 'hold'
        out_beats.append({
            'beat':        b + 1,
            'frame_start': f0,
            'frame_end':   f1,
            'cue':         cue_text,
            'body_parts':  info['cue_parts'],
        })

    # Top-line summary: which body parts dominate the whole window.
    family_totals: Dict[str, float] = {}
    for b in out_beats:
        for p in b['body_parts']:
            family_totals[p] = family_totals.get(p, 0.0) + 1.0
    dominant = sorted(family_totals.items(), key=lambda x: -x[1])[:3]
    summary_parts = [p for p, _ in dominant]

    result = {
        'clip_id':         clip_id,
        'n_frames':        n_frames,
        'fps':             float(data.get('fps') or 30),
        'beats':           out_beats,
        'dominant_parts':  summary_parts,
    }
    if cache:
        try:
            cache_path.write_text(json.dumps(result, indent=2),
                                  encoding='utf-8')
        except Exception:
            pass
    return result


@lru_cache(maxsize=512)
def cues_for(clip_id: str) -> Dict[str, Any]:
    """Memoised public entry point used by the choreographer tools."""
    return analyze_clip(clip_id)


# ── step segmentation (v107) ─────────────────────────────────────────
# Convert the uniform 8-beat cue track into a SMALL ordered list of
# named, teachable micro-steps so the coach can teach "step 1, step 2,
# step 3" instead of one blurred call. We merge consecutive beats that
# belong to the same body region (arms / legs / full), absorb "hold"
# beats into the step they follow, then cap to a handful of steps.
def _iso_group(body_parts: List[str]) -> List[str]:
    """Map fine body parts -> the coarse isolation group the avatar
    player understands ('arms' / 'legs'); [] means full body."""
    if not body_parts:
        return []
    p = body_parts[0]
    if 'arm' in p or 'hand' in p or 'shoulder' in p:
        return ['arms']
    if 'leg' in p or 'foot' in p:
        return ['legs']
    return []                                  # torso / head / hips → full


def _group_key(body_parts: List[str]) -> str:
    g = _iso_group(body_parts)
    return g[0] if g else 'full'


@lru_cache(maxsize=512)
def steps_for(clip_id: str, max_steps: int = 4) -> Dict[str, Any]:
    """Segment a clip into 2-5 named, teachable micro-steps.

    Returns {'clip_id', 'steps': [{step, name, cue, parts, body_parts,
    beat_start, beat_end, frame_start, frame_end}], 'n_steps'}.
    Each step is one coachable chunk in plain dance vocabulary, e.g.
    'Right arm up', 'Chest pop', 'Step back, freeze'. Empty steps list
    means the clip is a steady groove with no distinct segments — the
    coach should just ride it rather than fake a breakdown.
    """
    a = analyze_clip(clip_id)
    beats = a.get('beats') or []
    raw: List[Dict[str, Any]] = []
    for b in beats:
        cue = (b.get('cue') or '').strip()
        parts = b.get('body_parts') or []
        is_hold = (not cue or cue == 'hold' or not parts)
        if is_hold:
            # absorb a hold into the step it follows (a freeze/settle);
            # leading holds before any movement are dropped.
            if raw:
                raw[-1]['frame_end'] = b.get('frame_end', raw[-1]['frame_end'])
                raw[-1]['beat_end'] = b.get('beat', raw[-1]['beat_end'])
                raw[-1]['hold_after'] = True
            continue
        grp = _group_key(parts)
        # Step boundary rule (tuned for ~2-4 teachable steps):
        #   • same cue as the open step  -> extend it
        #   • same body region but the open step is still only 1 beat
        #     long -> extend (don't fragment into 1-beat slivers)
        #   • otherwise (cue changes after the step has body, or the
        #     body region changes) -> start a NEW step.
        if raw:
            cur = raw[-1]
            cur_beats = (cur['beat_end'] or 0) - (cur['beat_start'] or 0) + 1
            same_region = cur['_grp'] == grp
            if cur['cue'] == cue or (same_region and cur_beats < 2):
                cur['frame_end'] = b.get('frame_end', cur['frame_end'])
                cur['beat_end'] = b.get('beat', cur['beat_end'])
                if cue not in cur['_cues']:
                    cur['_cues'].append(cue)
                continue
        raw.append({
            '_grp': grp,
            '_cues': [cue],
            'cue': cue,
            'body_parts': list(parts),
            'beat_start': b.get('beat'),
            'beat_end': b.get('beat'),
            'frame_start': b.get('frame_start'),
            'frame_end': b.get('frame_end'),
        })

    # Cap to max_steps by merging the shortest (fewest-frame) step into
    # its previous neighbour until we're under the limit.
    def _span(s):
        return (s.get('frame_end') or 0) - (s.get('frame_start') or 0)

    while len(raw) > max_steps:
        # find smallest-span step index >= 1 (merge into previous)
        idx = min(range(1, len(raw)), key=lambda i: _span(raw[i]))
        prev = raw[idx - 1]
        cur = raw[idx]
        prev['frame_end'] = cur['frame_end']
        prev['beat_end'] = cur['beat_end']
        for c in cur['_cues']:
            if c not in prev['_cues']:
                prev['_cues'].append(c)
        raw.pop(idx)

    steps: List[Dict[str, Any]] = []
    for i, s in enumerate(raw, start=1):
        # Name: the first 1-2 distinct cues joined, Title-cased.
        cues = s['_cues'][:2]
        name = ', '.join(cues)
        name = name[:1].upper() + name[1:] if name else 'Hold'
        if s.get('hold_after') and 'freeze' not in name.lower():
            name = f'{name}, freeze'
        steps.append({
            'step': i,
            'name': name,
            'cue': ', '.join(cues) if cues else 'hold',
            'parts': _iso_group(s['body_parts']),
            'body_parts': s['body_parts'],
            'beat_start': s['beat_start'],
            'beat_end': s['beat_end'],
            'frame_start': s['frame_start'],
            'frame_end': s['frame_end'],
        })

    return {
        'clip_id': clip_id,
        'n_steps': len(steps),
        'steps': steps,
        'fps': a.get('fps', 30),
    }

