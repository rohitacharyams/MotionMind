"""metadata.py — per-clip metadata layer.

Each retargeted clip in ``coach/motion_cache/<id>.json`` may have a
sidecar at ``coach/motion_meta/<id>.json`` carrying human-coachable
information:

    {
      "id":              "<clip id>",
      "title":           "Two-Step Drop",
      "genre":           "gLO",
      "bpm_target":      105,
      "counts":          8,
      "difficulty":      2,                         // 1..5
      "summary":         "A clean 8-count locking phrase with a stop, "
                         "point, and shoulder roll back to start.",
      "key_cues": [
        {"beat": 1, "cue": "Drop weight into right knee, soft bounce."},
        {"beat": 3, "cue": "Lock and freeze — pop the right arm out."},
        {"beat": 5, "cue": "Point left, eyes follow the line."},
        {"beat": 7, "cue": "Shoulder roll back, reset to neutral."}
      ],
      "common_mistakes": [
        "Letting the freeze go too long — re-engage on count 4.",
        "Pointing with a limp wrist; the line breaks."
      ],
      "muscle_focus":    ["quads", "lats", "obliques"],
      "prerequisite":    ["bounce", "weight-shift"],
      "vibe_tags":       ["funky", "playful", "old-school"],
      "tempo_hint":      "matches a 100-110 BPM funk track",
      "embedding":       null                       // populated by phase A.2
    }

This module exposes:
    list_meta()                  → dict of id → metadata (cached)
    get_meta(clip_id)            → metadata for one clip (or {})
    seed_meta_for_clip(clip_id)  → call Groq to draft metadata for a clip
    seed_all(limit=None)         → batch driver (CLI: python -m coach.metadata seed)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

COACH = Path(__file__).resolve().parent
load_dotenv(COACH / '.env')

CACHE_DIR = COACH / 'motion_cache'
META_DIR  = COACH / 'motion_meta'

# Seeder uses the cheap fast model — has its own daily token bucket so
# it can't drain the 70B bucket the live chat agent uses. Override with
# GROQ_SEEDER_MODEL env if a future model is better suited.
GROQ_MODEL = os.getenv('GROQ_SEEDER_MODEL', 'llama-3.1-8b-instant')
GROQ_KEY   = os.getenv('GROQ_API_KEY', '')

GENRE_NAMES = {
    'gBR': 'Breaking',   'gHO': 'House',       'gJB': 'Jazz Ballet',
    'gJS': 'Street Jazz', 'gKR': 'Krump',      'gLH': 'LA Hip-Hop',
    'gLO': 'Locking',    'gMH': 'Middle Hip-Hop', 'gPO': 'Popping',
    'gWA': 'Waacking',
}


# ─── reading ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def list_meta() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not META_DIR.exists():
        return out
    for p in META_DIR.glob('*.json'):
        try:
            out[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
    return out


def get_meta(clip_id: str) -> Dict[str, Any]:
    p = META_DIR / f'{clip_id}.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


# ─── learned-choreography steps (video-to-learn pipeline) ──────────────
def set_learned_steps(clip_id: str, steps: List[Dict[str, Any]],
                      title: str = '', total: int = 0,
                      unlocked: int = 0) -> Dict[str, Any]:
    """Persist the step breakdown of a user-learned choreography into the
    clip's meta sidecar so ``break_down`` teaches THOSE named steps (with the
    free/paid gate baked in via each step's ``locked`` flag).

    Called server-to-server by the host app after its GPT-4o segmentation. Merges
    into any existing meta and refreshes the module cache. ``steps`` items are
    ``{index, label, cue, start_s, end_s, locked?}``.
    """
    META_DIR.mkdir(parents=True, exist_ok=True)
    p = META_DIR / f'{clip_id}.json'
    meta: Dict[str, Any] = {}
    if p.exists():
        try:
            meta = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    norm: List[Dict[str, Any]] = []
    for i, s in enumerate(steps or [], 1):
        if not isinstance(s, dict):
            continue
        norm.append({
            'index': int(s.get('index', i)),
            'label': str(s.get('label') or f'Step {i}')[:80],
            'count': str(s.get('count') or '')[:24],
            'cue': str(s.get('cue') or '')[:280],
            'detail': str(s.get('detail') or '')[:280],
            'mistake': str(s.get('mistake') or '')[:280],
            'feel': str(s.get('feel') or '')[:48],
            'start_s': float(s.get('start_s') or 0.0),
            'end_s': float(s.get('end_s') or 0.0),
            'locked': bool(s.get('locked', False)),
        })
    meta['id'] = clip_id
    if title:
        meta['title'] = title
    meta['learned_steps'] = norm
    meta['learned_total'] = int(total or len(norm))
    meta['learned_unlocked'] = int(unlocked or sum(
        1 for s in norm if not s['locked']))
    meta['source'] = 'learned'
    p.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    try:
        list_meta.cache_clear()
    except Exception:
        pass
    return meta


# ─── seeding (LLM draft) ───────────────────────────────────────────────
_SEED_SYSTEM = """You are a dance instructor writing teaching notes for a
short choreography clip. You will receive: genre name, approximate BPM,
duration in seconds, frame count, and a brief description of what's in
the clip (which we infer from the file name and joint statistics).

Return STRICT JSON only. No prose. Fields:
- title           short 2-4 word name
- counts          integer, total counts the clip spans (8/16/32)
- difficulty      1..5
- summary         one sentence about what the clip teaches
- key_cues        array of {beat, cue} — 3 to 5 entries
- common_mistakes array of 2-3 short strings
- muscle_focus    array of 2-4 muscle group names
- prerequisite    array of 0-2 simple move names that should be learned first
- vibe_tags       array of 3-6 single-word tags
- tempo_hint      one short phrase

Match the genre's actual stylistic vocabulary. Locking should mention
locks/points/skeeters; Krump should mention chest pops/jabs/stomps;
Waacking should mention arm whips and poses; House should mention
jacking, footwork, and bounces. Don't use the wrong vocabulary.

Be terse and coachable — these are speech cues read aloud."""


def _ctx_for_clip(clip_id: str) -> Dict[str, Any]:
    """Pull the basic timing facts from the retargeted clip JSON."""
    p = CACHE_DIR / f'{clip_id}.json'
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding='utf-8'))
    n_frames = int(d.get('n_frames', 0))
    fps      = float(d.get('fps', 30))
    duration = round(n_frames / max(fps, 1), 2)
    genre_code = clip_id[:3] if clip_id[:3] in GENRE_NAMES else 'multistyle'
    return {
        'clip_id':   clip_id,
        'genre':     GENRE_NAMES.get(genre_code, 'Multi-style'),
        'duration':  duration,
        'n_frames':  n_frames,
        'fps':       fps,
        # Crude bpm guess for AIST clips: 60fps source, 8 counts ≈ 4s of music
        # → bpm ~ 120 by default; adjust per genre below
        'bpm_guess': {
            'gBR': 105, 'gHO': 125, 'gJB': 100, 'gJS': 110,
            'gKR': 100, 'gLH': 95,  'gLO': 105, 'gMH': 100,
            'gPO': 95,  'gWA': 110,
        }.get(genre_code, 110),
    }


async def seed_meta_for_clip(clip_id: str, *, overwrite: bool = False
                              ) -> Optional[Dict[str, Any]]:
    META_DIR.mkdir(exist_ok=True, parents=True)
    dest = META_DIR / f'{clip_id}.json'
    if dest.exists() and not overwrite:
        return None
    if not GROQ_KEY:
        raise RuntimeError('GROQ_API_KEY not set; cannot seed metadata.')
    from groq import AsyncGroq          # local import keeps cold-start fast
    cx = AsyncGroq(api_key=GROQ_KEY)
    ctx = _ctx_for_clip(clip_id)
    if not ctx:
        return None
    user = (f"Genre: {ctx['genre']}\n"
            f"Duration: {ctx['duration']} s\n"
            f"Frames: {ctx['n_frames']} at {ctx['fps']} fps\n"
            f"Approximate BPM: {ctx['bpm_guess']}\n"
            f"Clip id: {clip_id}\n"
            f"Now produce the JSON.")

    # Retry on 429 (token/RPM rate limits). Honour the message's
    # "try again in Xs" hint when we can parse it.
    for attempt in range(4):
        try:
            resp = await cx.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{'role': 'system', 'content': _SEED_SYSTEM},
                          {'role': 'user',   'content': user}],
                temperature=0.5,
                response_format={'type': 'json_object'},
            )
            break
        except Exception as e:
            msg = str(e)
            if '429' not in msg and 'rate_limit' not in msg.lower():
                raise
            # Per-day token cap — surface to caller so we stop the run.
            if 'tokens per day' in msg or 'TPD' in msg:
                raise RuntimeError('Groq daily token cap hit — '
                                   'resume tomorrow.') from e
            import re as _re
            m = _re.search(r'try again in ([0-9.]+)s', msg)
            wait = float(m.group(1)) + 0.5 if m else (2 ** attempt) * 5
            print(f'  rate-limited; sleeping {wait:.1f}s '
                  f'(attempt {attempt+1}/4)')
            await asyncio.sleep(wait)
    else:
        return None

    txt = resp.choices[0].message.content or '{}'
    try:
        draft = json.loads(txt)
    except json.JSONDecodeError:
        return None
    draft['id']         = clip_id
    draft['genre']      = clip_id[:3] if clip_id[:3] in GENRE_NAMES \
                                       else 'multistyle'
    draft['bpm_target'] = ctx['bpm_guess']
    draft['embedding']  = None       # filled in by phase A.2
    dest.write_text(json.dumps(draft, indent=2, ensure_ascii=False),
                    encoding='utf-8')
    return draft


async def _seed_all(limit: Optional[int] = None,
                    overwrite: bool = False,
                    concurrency: int = 4) -> None:
    ids = sorted(p.stem for p in CACHE_DIR.glob('*.json')
                 if not p.stem.startswith('_'))
    if limit:
        ids = ids[:limit]
    print(f'[meta] seeding {len(ids)} clips (concurrency={concurrency}, '
          f'overwrite={overwrite})')
    sem = asyncio.Semaphore(concurrency)
    n_ok = n_skip = n_fail = 0

    async def one(cid: str) -> None:
        nonlocal n_ok, n_skip, n_fail
        async with sem:
            try:
                r = await seed_meta_for_clip(cid, overwrite=overwrite)
                if r is None: n_skip += 1
                else:
                    n_ok += 1
                    print(f'  ok  {cid}  "{r.get("title", "?")}"')
            except Exception as e:
                n_fail += 1
                print(f'  FAIL {cid}: {e}')

    await asyncio.gather(*(one(cid) for cid in ids))
    print(f'[meta] done  ok={n_ok}  skip={n_skip}  fail={n_fail}')


# ─── CLI ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['seed'])
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--concurrency', type=int, default=4)
    a = p.parse_args()
    if a.cmd == 'seed':
        asyncio.run(_seed_all(limit=a.limit,
                              overwrite=a.overwrite,
                              concurrency=a.concurrency))
