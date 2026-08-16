"""writer.py — turn a numeric diff into a 2-3-sentence coach note.

We hand the LLM both the diff stats AND the clip's teaching metadata
(`key_cues`, `common_mistakes`) so the feedback uses the actual coach
vocabulary instead of inventing it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

COACH = Path(__file__).resolve().parent.parent
load_dotenv(COACH / '.env')

GROQ_MODEL = os.getenv('GROQ_WRITER_MODEL', 'llama-3.1-8b-instant')
GROQ_KEY   = os.getenv('GROQ_API_KEY', '')


_SYS = """You are a dance instructor giving short, kind, actionable
feedback on a student's attempt. You receive:

  - the reference clip's teaching metadata (cues + common mistakes)
  - a per-bone angular error report (degrees) from DTW alignment
  - the worst 4 bones by mean error

Rules:
  - 2-3 short sentences total. No bullet lists.
  - Lead with one specific thing they did well (if mean_error_deg < 30).
  - Name ONE concrete bone/limb to fix using everyday language
    ("right elbow", "left knee", "shoulders" — not "leftLowerArm").
  - End with a single drill suggestion using their own clip
    ("loop counts 1-4 at half speed").
  - Never mention degrees or DTW. Use plain English.
  - If mean_error_deg > 60 the student is clearly off-step — gently
    suggest starting with the slower drill first.
"""


_BONE_NICE = {
    'leftUpperArm':  'left shoulder/upper arm',
    'rightUpperArm': 'right shoulder/upper arm',
    'leftLowerArm':  'left elbow/forearm',
    'rightLowerArm': 'right elbow/forearm',
    'leftUpperLeg':  'left hip/thigh',
    'rightUpperLeg': 'right hip/thigh',
    'leftLowerLeg':  'left knee/shin',
    'rightLowerLeg': 'right knee/shin',
    'spine':         'lower spine',
    'chest':         'upper chest',
    'hips':          'hips',
}


async def write_feedback(diff: Dict[str, Any],
                         meta: Dict[str, Any]) -> str:
    if not GROQ_KEY:
        return _fallback(diff, meta)
    from groq import AsyncGroq
    cx = AsyncGroq(api_key=GROQ_KEY)
    payload = {
        'clip_title':     meta.get('title', ''),
        'key_cues':       meta.get('key_cues', []),
        'common_mistakes': meta.get('common_mistakes', []),
        'mean_error_deg': round(diff.get('mean_error_deg', 0), 1),
        'worst_bones':    [
            {'bone': _BONE_NICE.get(b['bone'], b['bone']),
             'mean_deg': round(b['mean_deg'], 1)}
            for b in (diff.get('worst_bones') or [])[:4]
        ],
    }
    try:
        resp = await cx.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'system', 'content': _SYS},
                      {'role': 'user',
                       'content': json.dumps(payload, ensure_ascii=False)}],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or '').strip() \
               or _fallback(diff, meta)
    except Exception:
        return _fallback(diff, meta)


def _fallback(diff: Dict[str, Any], meta: Dict[str, Any]) -> str:
    worst = (diff.get('worst_bones') or [])
    if not worst:
        return ("Solid effort! Loop counts 1-4 at half speed to lock the "
                "timing in.")
    name = _BONE_NICE.get(worst[0]['bone'], worst[0]['bone'])
    mean = diff.get('mean_error_deg', 0)
    if mean > 60:
        return (f"You're a bit off the beat — try the slow drill first, "
                f"focusing on your {name}. We'll speed it up once it locks "
                f"in.")
    return (f"Nice flow on the timing — main thing to clean up is your "
            f"{name}. Loop counts 1-4 at half speed and exaggerate that "
            f"movement.")


# ─── 2D variant ────────────────────────────────────────────────────────
_KPT_NICE = {
    'left_shoulder':  'left shoulder',
    'right_shoulder': 'right shoulder',
    'left_elbow':     'left elbow',
    'right_elbow':    'right elbow',
    'left_wrist':     'left hand',
    'right_wrist':    'right hand',
    'left_hip':       'left hip',
    'right_hip':      'right hip',
    'left_knee':      'left knee',
    'right_knee':     'right knee',
    'left_ankle':     'left foot',
    'right_ankle':    'right foot',
}

_SYS_2D = """You are a dance instructor giving short, kind, actionable
feedback on a student's attempt. You receive:

  - the reference clip's teaching metadata (cues + common mistakes)
  - a per-body-part position-error report (normalised torso units;
    0.0 = perfect, 0.2 = noticeable, 0.5 = clearly wrong)
  - the worst 4 body parts by mean error

Rules:
  - 2-3 short sentences total. No bullet lists.
  - Lead with one specific thing they did well if mean_error < 0.25.
  - Name ONE concrete body part to fix using everyday language
    (the names you receive are already in plain English — use them).
  - End with a single drill suggestion using their own clip
    ("loop counts 1-4 at half speed").
  - Never mention numbers or normalisation. Use plain English.
  - If mean_error > 0.45 the student is clearly off-step — gently
    suggest starting with the slower drill first.
"""


async def write_feedback_2d(diff: Dict[str, Any],
                            meta: Dict[str, Any]) -> str:
    """LLM-narrated note from the 2D comparator's diff payload."""
    if not GROQ_KEY:
        return _fallback_2d(diff, meta)
    from groq import AsyncGroq
    cx = AsyncGroq(api_key=GROQ_KEY)
    payload = {
        'clip_title':       meta.get('title', ''),
        'key_cues':         meta.get('key_cues', []),
        'common_mistakes':  meta.get('common_mistakes', []),
        'mean_error':       round(diff.get('mean_error', 0.0), 3),
        'worst_body_parts': [
            {'body_part': _KPT_NICE.get(b['keypoint'], b['keypoint']),
             'mean':     round(b['mean'], 3)}
            for b in (diff.get('worst_keypoints') or [])[:4]
        ],
    }
    try:
        resp = await cx.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'system', 'content': _SYS_2D},
                      {'role': 'user',
                       'content': json.dumps(payload, ensure_ascii=False)}],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or '').strip() \
               or _fallback_2d(diff, meta)
    except Exception:
        return _fallback_2d(diff, meta)


def _fallback_2d(diff: Dict[str, Any], meta: Dict[str, Any]) -> str:
    worst = (diff.get('worst_keypoints') or [])
    mean = float(diff.get('mean_error', 0.0))
    if not worst:
        return ("Solid effort! Loop counts 1-4 at half speed to lock the "
                "timing in.")
    name = _KPT_NICE.get(worst[0]['keypoint'], worst[0]['keypoint'])
    if mean > 0.45:
        return (f"You're a bit off the beat — try the slow drill first, "
                f"watching your {name}. We'll speed it up once it locks "
                f"in.")
    return (f"Nice flow on the timing — main thing to clean up is your "
            f"{name}. Loop counts 1-4 at half speed and exaggerate that "
            f"movement.")
