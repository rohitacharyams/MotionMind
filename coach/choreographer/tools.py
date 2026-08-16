"""tools.py — tool definitions exposed to the LLM.

The LLM's tool_call output is translated into events streamed to the
browser, which then drives the MotionPlayer / mirror / drill loop.
We do NOT play motion on the server — the browser owns the avatar.
"""
from __future__ import annotations

import random
import re
from collections import deque
from typing import Any, Dict, List, Optional

from coach import motion_index
from coach import metadata as motion_metadata
from coach import motion_analyzer
from coach import semantic_search
from coach import session as coach_session

# v144: WARM-UP / MOBILITY TEACHING CURRICULUM. Rich per-movement teaching
# metadata (setup, teaching_script, breathing, sensation, mistakes) that lets
# the coach actually TEACH a warm-up instead of just naming it. Keyed by
# lesson_id, with a clip_to_lesson map from the 21 whitelisted CMU clips.
# Loaded once at import; failures are non-fatal (teaching just stays empty).
import json as _json
from pathlib import Path as _Path

_WARMUP_CURRICULUM: Dict[str, Any] = {}
_WARMUP_LESSON_BY_CLIP: Dict[str, Dict[str, Any]] = {}
try:
    _cur_path = _Path(__file__).resolve().parent.parent / 'motion_meta' / 'warmup_curriculum.json'
    _cur = _json.loads(_cur_path.read_text(encoding='utf-8'))
    _lessons_by_id = {l['lesson_id']: l for l in _cur.get('lessons', [])}
    _WARMUP_CURRICULUM = _cur
    for _cid, _lid in (_cur.get('clip_to_lesson') or {}).items():
        if _lid in _lessons_by_id:
            _WARMUP_LESSON_BY_CLIP[_cid] = _lessons_by_id[_lid]
except Exception:                                                # noqa: BLE001
    _WARMUP_CURRICULUM = {}
    _WARMUP_LESSON_BY_CLIP = {}


def get_warmup_lesson(clip_id: str) -> Optional[Dict[str, Any]]:
    """Return the rich teaching lesson for a warm-up/cooldown clip, or None."""
    return _WARMUP_LESSON_BY_CLIP.get(clip_id)


# Cross-SESSION variety. `state.played_clips` only dedupes within a
# single session, so every fresh session re-picked from the same
# sorted-first clips (the user saw "the same warm-up clips in almost
# every session"). This process-global ring buffer remembers the last
# clips served across ALL sessions in this server process so the picker
# can prefer ones the user hasn't seen recently. Big enough to rotate
# through the warmup pool, small enough that we never starve a small
# style genre.
_RECENT_SERVED: deque = deque(maxlen=40)


def _normalize_cues(kc):
    """v133: coerce key_cues into a [{beat,cue}] list. LLM seeders
    sometimes emit a dict {"1":"Heel Toe",...} or a flat list of
    strings; both used to fail break_down's dict-check and silently
    drop to the noisy segmenter. Normalize so authored moves always win."""
    if isinstance(kc, dict):
        out = []
        for i, (k, v) in enumerate(kc.items()):
            beat = int(k) if str(k).isdigit() else (1 + i * 4)
            out.append({'beat': beat, 'cue': str(v)})
        return out
    if isinstance(kc, list):
        out = []
        for i, c in enumerate(kc):
            if isinstance(c, dict) and c.get('cue'):
                out.append({'beat': c.get('beat', 1 + i * 4), 'cue': c['cue']})
            elif isinstance(c, str) and c.strip():
                out.append({'beat': 1 + i * 4, 'cue': c.strip()})
        return out
    return []


def _drop_just_served(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Never hand back the clip we JUST served. When a pool has been
    exhausted and recycled, the naive random pick can return the clip
    that is currently playing — so the coach narrates "let's try a
    different move" but the SAME clip replays (the user's "says it
    changed but plays the same one" bug). Drop the last-served id
    whenever at least one alternative remains."""
    if len(pool) <= 1:
        return pool
    last = _RECENT_SERVED[-1] if _RECENT_SERVED else None
    if last is None:
        return pool
    trimmed = [m for m in pool if m.get('id') != last]
    return trimmed or pool


def _choose_fresh(cands: List[Dict[str, Any]],
                  played_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Pick a clip from `cands`, preferring ones not served recently
    (across sessions) and not already played this session. Falls back
    to the full candidate list only when every option was recently
    used, so we rotate through the WHOLE pool instead of looping the
    first few. Records the pick in the cross-session buffer."""
    if not cands:
        return None
    recent = set(_RECENT_SERVED)
    pool = [m for m in cands if m.get('id') not in recent]
    if played_ids:
        fresher = [m for m in pool if m.get('id') not in played_ids]
        if fresher:
            pool = fresher
    if not pool:
        pool = cands
    pool = _drop_just_served(pool)
    pick = random.choice(pool)
    _RECENT_SERVED.append(pick.get('id'))
    return pick


try:
    from coach.choreographer.inverted_blacklist import INVERTED_BLACKLIST
except Exception:  # noqa: BLE001
    INVERTED_BLACKLIST = set()


def _warmup_clip_pool(all_clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The REAL warm-up / mobility / stretch pool = the verified-upright
    Mixamo clips in the curated stretch set (neck/shoulder/reach/squat/
    cardio). EXCLUDES floor work (plank/push-up/sit-up) which needs
    on-ground camera framing and is not a standing warm-up.

    v153: the old 'cmu' warm-up pool renders INVERTED, so every CMU clip
    gets swapped by the orientation gate to the safe fallback groove
    (gLO_sBM_cAll_d13_mLO0_ch01) — that groove is the "weird leaning-back
    walk" the user kept seeing when asking for a warm-up, and it's why the
    real Mixamo stretches never appeared. Warm-up/stretch/mobility requests
    must therefore draw from the curated Mixamo pool. Falls back to CMU
    only if no Mixamo clip is available (should never happen in prod)."""
    curated = set(coach_session._SW_UPPER + coach_session._SW_MID +
                  coach_session._SW_LOWER + coach_session._SW_DYNAMIC +
                  coach_session._SW_BREATH)
    mix = [m for m in all_clips
           if m.get('genre') == 'mixamo'
           and m['id'] in curated
           and m.get('safety') in ('ok', 'unknown')
           and m['id'] not in INVERTED_BLACKLIST]
    if mix:
        return mix
    # last-resort: any verified mixamo clip, then cmu
    mix_any = [m for m in all_clips
               if m.get('genre') == 'mixamo'
               and m.get('safety') in ('ok', 'unknown')
               and m['id'] not in INVERTED_BLACKLIST]
    if mix_any:
        return mix_any
    return [m for m in all_clips
            if m.get('genre') == 'cmu'
            and m.get('safety') in ('ok', 'unknown')
            and m['id'] not in INVERTED_BLACKLIST]


# Floor-exercise Mixamo clips (plank / push-up / sit-up). Kept OUT of the
# generic standing warm-up rotation (they need on-ground framing), but
# reachable when the user EXPLICITLY asks for them by name.
_FLOOR_CLIP_IDS = {
    'mixamo_plank', 'mixamo_start_plank', 'mixamo_end_plank',
    'mixamo_push_up', 'mixamo_idle_to_push_up', 'mixamo_jump_push_up',
    'mixamo_start_bicycle_sit_up', 'mixamo_end_bicycle_sit_up',
}
# Keywords that mean the user WANTS a floor exercise.
_FLOOR_KW = {
    'plank', 'planks', 'planking', 'push', 'pushup', 'pushups',
    'pushed', 'situp', 'situps', 'crunch', 'crunches', 'bicycle',
    'core', 'abs', 'floor', 'sit', 'ups',
}


def _floor_clip_pool(all_clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The Mixamo floor-exercise clips (plank/push-up/sit-up), verified
    and tagged pose_profile:'floor' so the client guard renders them."""
    return [m for m in all_clips
            if m.get('genre') == 'mixamo'
            and m['id'] in _FLOOR_CLIP_IDS
            and m.get('safety') in ('ok', 'unknown')
            and m['id'] not in INVERTED_BLACKLIST]



TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'pick_and_play',
            'description': (
                'ATOMIC: pick a motion clip from the DB AND start '
                'playing it on the avatar in one shot. THIS IS THE '
                'ONLY WAY to make the avatar dance. NEVER speak about '
                'a move without calling this first. Returns the actual '
                'clip id + title + key_cues + common_mistakes that you '
                'MUST use verbatim in your speech (never invent move '
                'names). Excludes clips you have already played this '
                'session so the user sees variety.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'genre': {
                        'type': 'string',
                        'enum': ['gBR', 'gHO', 'gJB', 'gJS', 'gKR',
                                 'gLH', 'gLO', 'gMH', 'gPO', 'gWA',
                                 'cmu'],
                        'description': 'Genre code. gBR=Breaking, '
                            'gHO=House, gJB=Ballet Jazz, gJS=Street '
                            'Jazz, gKR=Krump, gLH=LA Hip-Hop, '
                            'gLO=Locking, gMH=Middle Hip-Hop, '
                            'gPO=Popping, gWA=Waacking, '
                            'cmu=Basics & Warmups (CMU mocap: '
                            'walks, jogs, kicks, stretches, '
                            'posture, footwork drills).',
                    },
                    'query': {
                        'type': 'string',
                        'description': (
                            'IMPORTANT: pass the user\u2019s exact '
                            'wording when they describe a specific '
                            'move (e.g. "casual walk", "side kick", '
                            '"arm wave", "shoulder roll", "slow '
                            'shoulder bounce"). The catalog is '
                            'searched semantically and the best '
                            'title match within the genre wins. '
                            'OMIT this parameter only when the user '
                            'said something generic like "dance for '
                            'me" or "show me house".'),
                    },
                    'difficulty': {'type': 'integer', 'minimum': 1,
                                   'maximum': 5},
                    'bpm':   {'type': 'integer'},
                    'speed': {'type': 'number', 'minimum': 0.25,
                              'maximum': 1.5,
                              'description': 'Playback speed. '
                                  'Default 1.0. Pass 0.5 for a slow '
                                  'demo on the first run-through.'},
                    'loop':  {'type': 'boolean',
                              'description': 'Default TRUE \u2014 the '
                                  'avatar keeps dancing while you '
                                  'coach. Set false ONLY for a '
                                  'one-shot demo.'},
                    'mirror':{'type': 'boolean'},
                },
                'required': ['genre'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'pick_clip',
            'description': ('LEGACY \u2014 prefer pick_and_play. Choose '
                            'a motion clip from the DB without '
                            'playing it. You MUST call play() right '
                            'after, otherwise the avatar stays still.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'genre': {
                        'type': 'string',
                        'enum': ['gBR', 'gHO', 'gJB', 'gJS', 'gKR',
                                 'gLH', 'gLO', 'gMH', 'gPO', 'gWA',
                                 'cmu'],
                        'description': 'Genre code.',
                    },
                    'query': {
                        'type': 'string',
                        'description': (
                            'Optional free-text describing the '
                            'desired move (e.g. "casual walk", '
                            '"arm wave"). Used to semantically '
                            'narrow within the genre.'),
                    },
                    'difficulty': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                    'bpm':        {'type': 'integer'},
                },
                'required': ['genre'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'play',
            'description': 'Play the most-recently-picked clip on the avatar.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'speed':  {'type': 'number', 'minimum': 0.25, 'maximum': 1.5},
                    'mirror': {'type': 'boolean'},
                    'loop':   {'type': 'boolean'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'drill',
            'description': ('Loop an N-count slice of the current clip '
                            'for practice, with a slow-to-full speed '
                            'ramp. Optionally restrict to a specific '
                            'count window (e.g. start_count=1 end_count=8 '
                            'drills only counts 1-8).'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'counts':      {'type': 'integer', 'minimum': 4,
                                    'maximum': 32,
                                    'description': 'How many counts the '
                                    'drill window spans (8 = 1 phrase).'},
                    'repeats':     {'type': 'integer', 'minimum': 1,
                                    'maximum': 16},
                    'speed_start': {'type': 'number', 'minimum': 0.25,
                                    'maximum': 1.5,
                                    'description': 'Starting speed '
                                    '(default 0.5 = half speed).'},
                    'speed_end':   {'type': 'number', 'minimum': 0.25,
                                    'maximum': 1.5,
                                    'description': 'Ending speed '
                                    '(default 1.0 = full speed).'},
                    'start_count': {'type': 'integer', 'minimum': 1,
                                    'description': 'First count in the '
                                    'drill window (1-indexed).'},
                    'end_count':   {'type': 'integer', 'minimum': 1,
                                    'description': 'Last count in the '
                                    'drill window (inclusive).'},
                    'mirror_alternate': {'type': 'boolean',
                                    'description': 'Mirror every other '
                                    'repeat so the student practises '
                                    'both sides.'},
                },
            },
        },
    },
    {'type': 'function', 'function': {'name': 'slower',
        'description': 'Halve current playback speed.', 'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {'name': 'mirror',
        'description': 'Toggle mirror mode on the avatar.', 'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {'name': 'set_language',
        'description': ('Switch the language the coach speaks in. Use whenever the '
                        'user asks to change language, e.g. "hindi me baat karo", '
                        '"speak in english", "hinglish chalega", "talk to me in hindi".'),
        'parameters': {'type': 'object', 'properties': {
            'language': {'type': 'string', 'enum': ['english', 'hinglish', 'hindi']}},
            'required': ['language']}}},
    {'type': 'function', 'function': {'name': 'stop',
        'description': 'Stop the avatar.', 'parameters': {'type': 'object', 'properties': {}}}},
    {
        'type': 'function',
        'function': {
            'name': 'explain',
            'description': 'Brief 1-2 sentence tip spoken by the coach.',
            'parameters': {
                'type': 'object',
                'properties': {'topic': {'type': 'string'}},
                'required': ['topic'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_clips',
            'description': ('Find clips matching a free-text description '
                            'like "slow funky shoulder roll" or '
                            '"explosive krump chest pop". Returns the '
                            'top matches with id, title, and short '
                            'summary. Use this BEFORE pick_clip when the '
                            'user describes a vibe in words instead of '
                            'asking for a specific genre.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'genre': {
                        'type': 'string',
                        'enum': ['gBR', 'gHO', 'gJB', 'gJS', 'gKR',
                                 'gLH', 'gLO', 'gMH', 'gPO', 'gWA'],
                        'description': 'optional genre filter',
                    },
                    'k': {'type': 'integer', 'minimum': 1, 'maximum': 12},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'set_mood',
            'description': ('Set the avatar\'s facial expression to match '
                            'the emotional tone of the next thing you say. '
                            'Call this whenever the vibe shifts: hype the '
                            'user with happy/excited, focus a drill with '
                            'focused, soften a correction with relaxed.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'mood': {
                        'type': 'string',
                        'enum': ['happy', 'excited', 'relaxed',
                                 'focused', 'surprised', 'neutral'],
                    },
                },
                'required': ['mood'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'isolate',
            'description': (
                'Drive ONLY the named body part(s) on the avatar; pin '
                'the rest at bind-pose so the student can focus on one '
                'section of the move (arms-only, legs-only, etc.). Use '
                'this when the student says "I can\'t get the arms" or '
                "'show me just the footwork'. The current clip keeps "
                'looping in isolation. Call unisolate() to bring the '
                'full body back.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'parts': {
                        'type': 'array',
                        'items': {'type': 'string',
                                  'enum': ['arms', 'legs', 'torso',
                                           'head', 'hands', 'feet',
                                           'left', 'right']},
                        'description': 'One or more body-part groups.',
                    },
                },
                'required': ['parts'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'unisolate',
            'description': ('Clear any body-part isolation; the avatar '
                            'goes back to driving the whole body from '
                            'the clip.'),
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'break_down',
            'description': (
                'GUIDED STEP-BY-STEP BREAKDOWN — the PRIMARY tool for '
                'TEACHING. Use it WHENEVER the student wants to LEARN '
                '(e.g. "teach me hip hop", "I want to learn house", '
                '"how do I do this", "break it down", "step by step", '
                '"show me slowly", "what are the steps", "from the top"). '
                'The avatar walks through the move as NUMBERED micro-steps '
                '(step 1, step 2, step 3...) derived from the actual '
                'motion — each step isolates the body part it lives in and '
                'plays SLOW, then assembles full and rides at speed. Holds '
                'each step long enough to copy AND mirrors the steps into a '
                'clickable side rail the student can reference. If no clip '
                'is playing yet, pass `genre` (and optional `query`) and it '
                'will pick + load a clip for that style FIRST, then teach '
                'it — so one call handles "teach me <style>" end to end. '
                'Returns the ordered numbered steps so you can call each '
                'one by name as it plays. ONE call only — the browser '
                'drives the whole sequence.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'move_id': {
                        'type': 'string',
                        'description': 'Optional clip id; defaults to '
                            'the most-recently-played clip.',
                    },
                    'genre': {
                        'type': 'string',
                        'description': 'Dance style to teach when nothing '
                            'is playing yet (e.g. "hiphop", "house", '
                            '"locking", "popping", "waacking", "breaking").',
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Optional keyword to narrow the '
                            'clip pick within the style (e.g. "bounce").',
                    },
                    'stage_seconds': {
                        'type': 'number',
                        'minimum': 4, 'maximum': 16,
                        'description': 'Seconds per stage (default 8).',
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'resequence_to_music',
            'description': (
                'Tell the BROWSER to open its music picker so the student '
                'uploads an audio file. The browser then beat-detects it '
                'and stitches a custom routine from the catalog. Use when '
                'the student says things like "dance to this song", '
                '"choreograph to my track", "make a routine to <song>".'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'genre': {'type': 'string',
                              'enum': ['gBR','gHO','gJB','gJS','gKR',
                                       'gLH','gLO','gMH','gPO','gWA']},
                    'query': {'type': 'string',
                              'description': 'Free-text vibe to bias '
                              'clip selection (e.g. "smooth and slow").'},
                    'bars':  {'type': 'integer', 'minimum': 4,
                              'maximum': 32, 'default': 8},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'give_feedback',
            'description': (
                'Tell the BROWSER to open its student-video picker so the '
                'student can upload a clip of themselves dancing. The '
                'browser captures pose from the video, sends it to the '
                'backend, and the backend returns a per-bone error '
                'report and a coach note. Use when the student asks '
                '"how am I doing", "check my form", "rate my dance".'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'clip_id': {'type': 'string',
                                'description': 'Optional. Defaults to the '
                                'most-recently-played clip.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'live_feedback',
            'description': (
                'Open the LIVE mirror popup so the student can dance in '
                'front of their webcam and get real-time scoring on the '
                'current clip. Browser runs MediaPipe Pose locally and '
                'streams keypoints to /ws/feedback for DTW scoring. Use '
                'when the student says "watch me", "check my form '
                'live", "mirror me", "dance with me", "let me try".'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'clip_id': {'type': 'string',
                                'description': 'Optional. Defaults to the '
                                'most-recently-played clip.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'open_lessons',
            'description': (
                'Open the LEARN library PANEL — a browsable menu/list of '
                'Hip-Hop + House lessons. This ONLY opens a menu; it does '
                'NOT move the avatar. Use it ONLY when the student explicitly '
                'asks to SEE the menu/library/list of lessons ("show me the '
                'lessons", "what lessons are there", "open the lesson menu", '
                '"the curriculum"). DO NOT use this for "teach me X" or "break '
                'it down" or "show me a move" — for ANY request to actually '
                'LEARN or SEE a move, call break_down instead (that one moves '
                'the avatar). After opening this menu, do NOT say the avatar is '
                'dancing — nothing is moving yet.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'style': {'type': 'string', 'enum': ['hiphop', 'house'],
                              'description': 'Optional: focus one track.'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'open_lesson',
            'description': (
                'Open a SPECIFIC lesson\'s written reference card (what/why/'
                'cues/the one rule/music) in the Learn panel. This shows TEXT; '
                'it does NOT by itself move the avatar. Prefer break_down for '
                'actually teaching a move with the avatar. Use open_lesson only '
                'when the student wants to READ the notes for a named '
                'foundation ("show me the notes for the jack", "what are the '
                'cues for the two-step").'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'style': {'type': 'string', 'enum': ['hiphop', 'house']},
                    'move': {'type': 'string',
                             'description': 'The move / foundation name the '
                             'student asked for (free text).'},
                },
                'required': ['move'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'close_panel',
            'description': (
                'Close/dismiss any open on-screen panel — the steps rail, the '
                'lessons library, the chat log, or the live-mirror popup — so '
                'the student can see the avatar full-screen again. Use when the '
                'student says "close this", "hide the steps", "get rid of the '
                'menu", "close the panel", "let me just watch", or whenever a '
                'panel is in the way of the move you want them to see. Safe to '
                'call anytime; if nothing is open it does nothing.'),
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'open_profile',
            'description': (
                'Open the student\'s PROFILE page (streak, sessions, dance '
                'journey, preferences). Use when they ask "show my '
                'progress", "how many sessions have I done", "my streak", '
                '"my profile".'),
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
]


# ─── server-side execution of tools ────────────────────────────────────
class CoachState:
    """Mutable per-session state. Travels with the WS connection."""
    def __init__(self):
        self.current_clip: Optional[str] = None
        self.current_bpm: Optional[int] = None
        self.speed: float = 1.0
        self.mirror: bool = False
        # v84: Hinglish is the default coach language (user request).
        # Overridden by ?language= query param or a set_language WS msg.
        self.coach_language: str = 'hinglish'
        # Set by server.py at WS-accept time from the the host app JWT.
        # When the caller is anonymous these stay at the defaults.
        self.user_id: Optional[str] = None
        self.user_name: Optional[str] = None
        self.email_verified: bool = False
        self.tier: str = 'anon'        # anon | unverified | verified
        # Character context (set by browser via set_character WS msg).
        # The prompt builder reads these so the LLM stays in-character
        # AND knows which style is the home turf to default to.
        self.character_name: Optional[str] = None          # registry id
        self.character_display_name: Optional[str] = None  # "Kira"
        self.character_style: Optional[str] = None         # "House"
        # Move history — every clip the LLM picks gets appended here
        # so the system prompt can show "already shown" + the LLM
        # avoids repeats. Kept compact to avoid bloating the prompt.
        self.played_clips: List[Dict[str, Any]] = []
        # v197: set True when the student opens the Lessons panel so the
        # system prompt turns the coach into a proactive teacher/navigator.
        self.learning_intent: bool = False

    def remember_clip(self, clip_id: str, genre: str, summary: str = '') -> None:
        """Append clip to the played-clips memory. Dedupes consecutive
        repeats so a slower-replay of the same clip doesn't fill the list."""
        if not clip_id:
            return
        if self.played_clips and self.played_clips[-1].get('id') == clip_id:
            return
        self.played_clips.append({
            'id': clip_id, 'genre': genre, 'summary': summary,
        })
        # Keep last 32 — prompt only shows last 8 anyway.
        if len(self.played_clips) > 32:
            self.played_clips = self.played_clips[-32:]


def execute_tool(state: CoachState, name: str,
                 args: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a result dict that goes back to the LLM AND a browser
    event that drives the avatar."""
    if name == 'pick_clip':
        genre = args.get('genre')
        query = (args.get('query') or '').strip()
        all_clips = motion_index.list_motions()
        pool = [m for m in all_clips
                if m['genre'] == genre and m['safety'] in ('ok', 'unknown') and m['id'] not in INVERTED_BLACKLIST]
        # v153: warm-up/stretch/basics requests arrive with genre='cmu'
        # (see agent._GENRE_KW). The CMU pool is broken (renders inverted →
        # gated to the fallback groove = the "weird walk" bug), so serve the
        # verified Mixamo warm-up pool instead. Title-matching below still
        # works (Mixamo titles: "Neck Stretching", "Shrugging", "Slow Jog"…).
        if genre == 'cmu':
            pool = _warmup_clip_pool(all_clips)
            # v154: floor exercises (plank/push-up/sit-up) are kept out of
            # the generic standing rotation, but if the user EXPLICITLY
            # asks for one ("plank", "push ups", "bicycle crunches"), add
            # the floor clips so title-matching can reach them.
            _qtoks = set(re.findall(r'[a-z]+', (query or '').lower()))
            if _qtoks & _FLOOR_KW:
                _floor = _floor_clip_pool(all_clips)
                _have = {m['id'] for m in pool}
                pool = pool + [m for m in _floor if m['id'] not in _have]
        if not pool:
            return {
                'ok': False,
                'reason': f'no safe clips for {genre}',
                'browser_event': None,
            }        # Strongly prefer retargeted clips — those have proper VRM
        # bone-local quaternions and won't twist the avatar.
        retargeted = [m for m in pool if m.get('retargeted')]
        if retargeted:
            pool = retargeted
        # VARIETY: drop clips already shown this session so the user
        # sees fresh material. Fall back to the full pool if exhausted
        # (then the LLM is just spinning in this genre — recycle).
        played_ids = {c.get('id') for c in state.played_clips}
        fresh = [m for m in pool if m['id'] not in played_ids]
        if fresh:
            pool = fresh
        # v130: for TEACH, strongly prefer clips that carry authored
        # key_cues (real move names: "Heel Toe", "Bounce Down"). Clips
        # with no cues fall back to the joint-jitter segmenter, which
        # narrates nonsense ("left arm out, left arm in") and teaches
        # nothing. Bias the pool to cued clips so a lesson is always
        # clear; recycle to the full pool only if none are cued.
        if args.get('prefer_cued'):
            cued = [m for m in pool
                    if len(((motion_metadata.get_meta(m['id']) or {})
                            .get('key_cues')) or []) >= 2]
            if cued:
                pool = cued
        # QUERY MATCH: if the user asked for a specific move ("casual
        # walk", "arm wave", "posture warmup"), hard-filter the pool
        # by TITLE first (deterministic, doesn't depend on the
        # embedding model). Fail loud with ok:False if nothing matches
        # so the coach can say "I couldn't find that" instead of
        # silently rolling a random Casual Turn.
        pick = None
        if query:
            _STOP = {
                'show', 'me', 'please', 'a', 'the', 'some', 'play',
                'demo', 'do', 'can', 'you', 'could', 'now', 'just',
                'and', 'or', 'with', 'on', 'in', 'for', 'to', 'is',
                'it', 'this', 'that', 'one', 'something', 'give',
                'i', 'want', 'see', 'watch', 'teach', 'us', 'go',
                'try', 'lets', 'let', 'how', 'about', 'an', 'any',
                'really', 'good', 'nice', 'cool', 'next', 'another',
                'kind', 'type', 'something', 'practice', 'drill',
                # v24: GENRE keywords are not title-search content —
                # the genre filter already happened upstream. If a
                # user says "more house moves", "house" is intent
                # (already encoded in genre=gLH) not a title query.
                # Without this, the only gLH clip whose title contains
                # "house" ("House Bounce") gets picked every single
                # time and the user sees the same move repeated.
                'house', 'hiphop', 'hip', 'hop', 'pop', 'popping',
                'lock', 'locking', 'krump', 'krumping', 'break',
                'breaking', 'jazz', 'ballet', 'ballet-jazz',
                'middle', 'eastern', 'la', 'waack', 'waacking',
                'wave', 'moves', 'move', 'dance', 'dances',
                'style', 'styles', 'routine', 'choreography',
                'choreo',
            }
            # Synonym expansion: each content token may map to several
            # title vocabulary words. Keeps the user's natural English
            # ("warmup", "stretch", "posture") aligned with the clip
            # vocabulary ("Casual Stretch", "Arm Wave", "Side-to-Side").
            _SYNS = {
                'walk':    ['walk', 'stride', 'gait'],
                'walking': ['walk', 'stride', 'gait'],
                'jog':     ['jog', 'slow jog'],
                'jogging': ['jog'],
                'run':     ['jog', 'run'],
                'running': ['jog', 'run'],
                'wave':    ['wave'],
                'waves':   ['wave'],
                'waving':  ['wave'],
                'arm':     ['arm'],
                'arms':    ['arm'],
                'hand':    ['arm', 'hand'],
                'hands':   ['arm', 'hand'],
                'jump':    ['jump', 'hop', 'leap'],
                'jumping': ['jump', 'hop'],
                'hop':     ['hop', 'jump'],
                'leap':    ['leap', 'jump'],
                'kick':    ['kick'],
                'turn':    ['turn', 'pivot'],
                'turning': ['turn'],
                'pivot':   ['pivot', 'turn'],
                'spin':    ['turn', 'spin'],
                'twist':   ['twist', 'turn'],
                'stretch':   ['stretch', 'reach'],
                'stretches': ['stretch'],
                'stretching':['stretch'],
                'reach':     ['reach', 'stretch'],
                'reaching':  ['reach', 'stretch'],
                'plank':     ['plank'],
                'planks':    ['plank'],
                'planking':  ['plank'],
                'pushup':    ['push'],
                'pushups':   ['push'],
                'push':      ['push'],
                'situp':     ['sit', 'bicycle'],
                'situps':    ['sit', 'bicycle'],
                'crunch':    ['sit', 'bicycle'],
                'crunches':  ['sit', 'bicycle'],
                'bicycle':   ['bicycle', 'sit'],
                'core':      ['plank', 'sit', 'push'],
                'abs':       ['sit', 'bicycle'],
                'warmup':    ['stretch', 'warm', 'reach', 'sway',
                              'bob', 'swivel', 'step', 'shift'],
                'warm':      ['stretch', 'warm', 'reach', 'sway'],
                'warmups':   ['stretch', 'warm', 'reach'],
                'posture':   ['stand', 'pose', 'stretch', 'sway'],
                'balance':   ['stand', 'sway', 'shift', 'step'],
                'breath':    ['stand', 'sway'],
                'breathing': ['stand', 'sway'],
                'standing':  ['stand'],
                'bob':       ['bob'],
                'bobs':      ['bob'],
                'side':      ['side', 'side-to-side'],
                'sideways':  ['side', 'side-to-side'],
                'lateral':   ['side', 'side-to-side'],
                'step':      ['step'],
                'steps':     ['step'],
                'lunge':     ['lunge'],
                'lunges':    ['lunge'],
                'casual':    ['casual'],
                'slow':      ['slow'],
                'fast':      ['fast'],
                'head':      ['head'],
                'shoulder':  ['shoulder'],
                'hip':       ['hip', 'swivel'],
                'hips':      ['hip', 'swivel'],
                'swivel':    ['swivel'],
                'sway':      ['sway'],
                'shift':     ['shift'],
                'idle':      ['stand'],
                'still':     ['stand'],
                'rest':      ['stand'],
                'foot':      ['step', 'stride'],
                'feet':      ['step', 'stride'],
                'leg':       ['leg', 'step'],
                'legs':      ['leg', 'step'],
            }
            toks = [t for t in re.findall(r'[a-z]+', query.lower())
                    if t not in _STOP and len(t) >= 3]
            # v132: a genre-only ask ("waacking", "teach me breaking")
            # leaves ZERO content tokens (the style word is a stopword).
            # That used to dead-end at the bottom "no clip matches" guard
            # — the user saw "not in my deck" even though the pool was
            # full. With no specific move requested, just pick a fresh
            # clip from the (already genre+cued) pool.
            if not toks:
                pick = _choose_fresh(pool, played_ids)
            # v19 CMU-FIRST ROUTING. If the query contains a basics /
            # warmup / locomotion keyword, the user wants a CMU mocap
            # clip (walks, jogs, stretches, posture, footwork, hip
            # swivels). Route to the CMU pool BEFORE searching the
            # requested genre, otherwise "footwork warmup" while the
            # avatar is in House mode returns "House Bounce" because
            # the House title accidentally contains "step". User
            # complaint: "why our llm is unable to route the message
            # to correct motion ... Where are the corrected motions".
            _CMU_INTENT_KW = {
                'warmup', 'warmups', 'warm', 'basics', 'basic',
                'stretch', 'stretches', 'stretching', 'reach',
                'reaching', 'posture', 'balance', 'breathing',
                'breath', 'walk', 'walks', 'walking', 'gait',
                'stride', 'jog', 'jogging', 'wave', 'waves',
                'waving', 'sway', 'swivel', 'bob', 'lunge', 'lunges',
                'footwork', 'casual', 'simple', 'idle', 'rest',
                'standing', 'still', 'cooldown', 'cool',
            }
            if genre != 'cmu' and any(t in _CMU_INTENT_KW for t in toks):
                _cmu_all = _warmup_clip_pool(all_clips)
                _cmu_retarg = [m for m in _cmu_all if m.get('retargeted')]
                if _cmu_retarg:
                    _cmu_all = _cmu_retarg
                if _cmu_all:
                    _cmu_fresh = [m for m in _cmu_all
                                  if m['id'] not in played_ids]
                    pool = _cmu_fresh or _cmu_all
            # Per content-token, build the set of acceptable title
            # substrings (lowercase). Token "walk" → {"walk","stride",
            # "gait"}. Match if ANY substring appears in title.
            token_aliases = []
            for t in toks:
                token_aliases.append(set(_SYNS.get(t, [t])))
            # Strict-by-aliases: title must hit EVERY content token
            # via at least one of its aliases. So "casual walk" needs
            # both "casual" AND ("walk"|"stride"|"gait") in the title.
            def title_hits(title_lc, aliases_list):
                return all(any(a in title_lc for a in als)
                           for als in aliases_list) if aliases_list else False
            strict = [m for m in pool
                      if title_hits((m.get('title', '') or '').lower(),
                                    token_aliases)]
            # v24: re-apply dedupe AFTER title narrowing. The dedupe
            # above filtered the genre pool, but if the title query
            # only matches a handful of clips, the played-set might
            # still cover all of them and we'd loop the same one.
            # Strip just-played here too; recycle only if exhausted.
            if strict:
                strict_fresh = [m for m in strict if m['id'] not in played_ids]
                if strict_fresh:
                    strict = strict_fresh
                pick = _choose_fresh(strict, played_ids)
            else:
                # Relaxed: drop the rarest token (often a filler like
                # "drill", "thing") and retry once.
                if len(token_aliases) >= 2:
                    relaxed_aliases = token_aliases[:-1]
                    relaxed = [m for m in pool
                               if title_hits((m.get('title', '') or '').lower(),
                                             relaxed_aliases)]
                    if relaxed:
                        relaxed_fresh = [m for m in relaxed
                                         if m['id'] not in played_ids]
                        if relaxed_fresh:
                            relaxed = relaxed_fresh
                        pick = _choose_fresh(relaxed, played_ids)
                # v19b BEST-HIT FALLBACK for CMU intent. "start slow
                # jog warmup" → toks=[start,slow,jog,warmup]; no CMU
                # title contains "start", so strict & relaxed both
                # fail. Score each pool clip by how many of the
                # query's tokens (via aliases) appear in its title;
                # pick the best. Only fires when CMU intent is on
                # so we don't accidentally muddy genre searches.
                if pick is None and genre != 'cmu' and any(
                        t in _CMU_INTENT_KW for t in toks):
                    def title_score(title_lc, aliases_list):
                        return sum(1 for als in aliases_list
                                   if any(a in title_lc for a in als))
                    scored = []
                    for m in pool:
                        s = title_score(
                            (m.get('title', '') or '').lower(),
                            token_aliases)
                        if s > 0:
                            scored.append((s, m))
                    if scored:
                        scored.sort(key=lambda r: r[0], reverse=True)
                        top = scored[0][0]
                        best = [m for s, m in scored if s == top]
                        pick = _choose_fresh(best, played_ids)
                if pick is None:
                    # v15: CMU-POOL FALLBACK. Warmups, basics, walks,
                    # jogs, stretches, arm waves, hip swivels — these
                    # live in the 'cmu' genre, not in the avatar's
                    # home style genre (gHO/gLH/etc). If the user's
                    # query has any of the warmup/basic keywords AND
                    # we couldn't match the requested genre, retry
                    # against the cmu pool before giving up. This is
                    # what makes "start slow jog warmup", "footwork
                    # warmup", "simple hip swivel" actually find their
                    # CMU clips (cmu_02_02_02, cmu_02_02_08, etc.)
                    # instead of dead-ending with "I don't have that
                    # one in my deck".
                    _CMU_KW = {
                        'warmup', 'warmups', 'warm', 'basics', 'basic',
                        'stretch', 'stretches', 'stretching', 'reach',
                        'reaching', 'posture', 'balance', 'breathing',
                        'walk', 'walks', 'walking', 'gait', 'stride',
                        'jog', 'jogging', 'wave', 'waves', 'waving',
                        'sway', 'swivel', 'bob', 'lunge', 'lunges',
                        'footwork', 'casual', 'simple', 'idle',
                        'rest', 'standing', 'still',
                    }
                    if genre != 'cmu' and any(t in _CMU_KW for t in toks):
                        cmu_pool = _warmup_clip_pool(all_clips)
                        if cmu_pool:
                            cmu_fresh = [m for m in cmu_pool
                                         if m['id'] not in played_ids]
                            cmu_search = cmu_fresh or cmu_pool
                            cmu_strict = [m for m in cmu_search
                                          if title_hits(
                                              (m.get('title','') or '').lower(),
                                              token_aliases)]
                            if cmu_strict:
                                pick = _choose_fresh(cmu_strict, played_ids)
                            elif len(token_aliases) >= 2:
                                cmu_relaxed = [m for m in cmu_search
                                               if title_hits(
                                                   (m.get('title','') or '').lower(),
                                                   token_aliases[:-1])]
                                if cmu_relaxed:
                                    pick = _choose_fresh(cmu_relaxed, played_ids)
                            # If still no pick: any cmu clip whose
                            # vibe_tags include 'warmup' as a final
                            # safety net for purely intent-driven
                            # queries ("posture warmup drill").
                            if pick is None:
                                tagged = []
                                for m in cmu_search:
                                    meta = motion_metadata.get_meta(m['id']) or {}
                                    tags = set(meta.get('vibe_tags') or [])
                                    if tags & {'warmup', 'stretch', 'reach',
                                               'basic', 'basics'}:
                                        tagged.append(m)
                                if tagged:
                                    pick = _choose_fresh(tagged, played_ids)
                # v117: WARMUP/BASICS NEVER DEAD-ENDS. The CMU "Basics &
                # Warmups" pool is currently unavailable in the served
                # catalogue (0 cmu clips), so a bare "warmup" / "basic
                # warmup" / "cooldown" request used to title-miss in the
                # home genre and dead-end with the awful "I have no
                # warmup clips" line. For a GENERIC warmup/basics/cooldown
                # intent we always hand back a fresh clip from the home
                # pool — the calmest available thing — instead of failing.
                if pick is None and any(
                        t in {'warmup', 'warmups', 'warm', 'basics',
                              'basic', 'cooldown', 'cool', 'stretch',
                              'stretches', 'stretching', 'easy', 'gentle',
                              'slow', 'light'} for t in toks):
                    pick = _choose_fresh(pool, played_ids)
                if pick is None:
                    # No clip matches — DO NOT silently roll a random
                    # one. Return ok:False so the chat layer can say
                    # "I don't have a clip for that, try X / Y / Z".
                    # v34g: dedupe titles so the bubble doesn't read
                    # "try 'House Bounce', 'House Bounce', 'House Bounce'".
                    _seen = set()
                    sample = []
                    for m in pool:
                        t = m.get('title')
                        if not t or t in _seen:
                            continue
                        _seen.add(t)
                        sample.append(t)
                        if len(sample) >= 8:
                            break
                    return {
                        'ok': False,
                        'reason': (f'no clip in genre {genre!r} matches '
                                   f'query {query!r}'),
                        'query': query,
                        'sample_titles': sample,
                        'browser_event': None,
                    }
        if pick is None:
            pick = _choose_fresh(pool, played_ids)
        state.current_clip = pick['id']
        state.current_bpm = pick.get('bpm_target')
        meta = motion_metadata.get_meta(pick['id'])
        # Remember this pick so the system prompt shows it next turn
        # and the LLM stops repeating moves.
        state.remember_clip(pick['id'], pick.get('genre', genre),
                            meta.get('summary', ''))
        # Auto-derived per-beat cues from the actual joint motion —
        # tells the LLM WHAT IS ACTUALLY HAPPENING in the clip so it
        # can narrate accurately instead of hallucinating moves.
        try:
            cues = motion_analyzer.cues_for(pick['id'])
        except Exception:
            cues = {'beats': [], 'dominant_parts': []}
        # v130: if the clip has AUTHORED key_cues (real move names like
        # "Heel Toe", "Bounce Down"), feed THOSE to the LLM as auto_cues
        # instead of the joint-jitter segmenter ("left arm out") which
        # makes the coach narrate nonsense. Real dance vocabulary only.
        _kc = meta.get('key_cues') or []
        _kc = _normalize_cues(_kc)   # v133: tolerate dict/str-shaped cues
        if isinstance(_kc, list) and len(_kc) >= 2:
            cues = {'beats': [{'beat': c.get('beat', i + 1),
                               'cue': c.get('cue', '')}
                              for i, c in enumerate(_kc) if c.get('cue')],
                    'dominant_parts': cues.get('dominant_parts', [])}
        # v107: numbered, move-specific teaching steps for step-by-step
        # learning ("step 1, step 2..."). Empty when the clip is a steady
        # groove with no distinct segments.
        try:
            seg = motion_analyzer.steps_for(pick['id'])
            steps = seg.get('steps', [])
        except Exception:
            steps = []
        return {
            'ok': True,
            'clip_id': pick['id'],
            'duration_sec': pick['duration_sec'],
            'genre_name': pick['genre_name'],
            'title':       meta.get('title') or pick['id'],
            'difficulty':  meta.get('difficulty'),
            'summary':     meta.get('summary', ''),
            'key_cues':    meta.get('key_cues', []),
            'auto_cues':       cues.get('beats', []),
            'steps':           steps,
            'dominant_parts':  cues.get('dominant_parts', []),
            'common_mistakes': meta.get('common_mistakes', []),
            'music_url':   motion_index.music_url_for(pick['id']),
            'bpm_target':  pick.get('bpm_target') or meta.get('bpm_target'),
            'browser_event': {
                'type': 'avatar.load',
                'clip_id': pick['id'],
                'duration_sec': pick['duration_sec'],
                'music_url': motion_index.music_url_for(pick['id']),
                'bpm': pick.get('bpm_target') or meta.get('bpm_target'),
            },
        }

    if name == 'pick_and_play':
        # Atomic: do pick_clip + play in one call so the LLM cannot
        # forget the second step (which was leaving the avatar
        # stationary while it monologued).
        pick_res = execute_tool(state, 'pick_clip',
                                {'genre':      args.get('genre'),
                                 'query':      args.get('query'),
                                 'difficulty': args.get('difficulty'),
                                 'bpm':        args.get('bpm'),
                                 'prefer_cued': args.get('prefer_cued')})
        if not pick_res.get('ok'):
            return pick_res
        play_args = {'speed':  args.get('speed', 1.0),
                     'mirror': args.get('mirror', state.mirror),
                     'loop':   args.get('loop', True)}
        play_res = execute_tool(state, 'play', play_args)
        # Merge the two browser events; the agent/server fan them
        # out via the optional 'browser_events' list.
        load_evt = pick_res.get('browser_event')
        play_evt = play_res.get('browser_event')
        evts = [e for e in (load_evt, play_evt) if e]
        result = dict(pick_res)
        result['played'] = bool(play_res.get('ok'))
        result['speed']  = play_args['speed']
        result['mirror'] = play_args['mirror']
        result['loop']   = play_args['loop']
        # Keep single 'browser_event' for back-compat and add the
        # list form so server.py fans out both load + play.
        result['browser_event']  = load_evt
        result['browser_events'] = evts
        return result

    if name == 'play':
        if not state.current_clip:
            return {'ok': False, 'reason': 'no clip picked yet',
                    'browser_event': None}
        speed = float(args.get('speed', 1.0))
        mirror = bool(args.get('mirror', state.mirror))
        # Default to LOOP=TRUE so the avatar keeps dancing while the
        # coach speaks the 8-count breakdown over the top. The user
        # should NEVER see the avatar drop to rest mid-lesson; the LLM
        # has to explicitly pass loop=false (for a one-shot demo).
        loop = bool(args.get('loop', True))
        state.speed = speed
        state.mirror = mirror
        return {
            'ok': True,
            'browser_event': {
                'type': 'avatar.play',
                'clip_id': state.current_clip,
                'speed': speed, 'mirror': mirror, 'loop': loop,
                'music_url': motion_index.music_url_for(state.current_clip),
                'bpm': state.current_bpm,
            },
        }

    if name == 'drill':
        clip_id = state.current_clip
        meta = motion_metadata.get_meta(clip_id) if clip_id else {}
        counts     = int(args.get('counts', meta.get('counts', 8) or 8))
        repeats    = int(args.get('repeats', 4))
        speed_start = float(args.get('speed_start', 0.5))
        speed_end   = float(args.get('speed_end',   1.0))
        start_count = args.get('start_count')
        end_count   = args.get('end_count')
        mirror_alt  = bool(args.get('mirror_alternate', False))
        # Coach cues drawn from metadata for the drilled window.
        cues = []
        for c in meta.get('key_cues', []) or []:
            if not isinstance(c, dict): continue
            b = c.get('beat')
            if start_count and end_count and b is not None:
                if not (start_count <= b <= end_count): continue
            cues.append(c.get('cue', ''))
        cues = [c for c in cues if c][:4]
        return {
            'ok': True,
            'cues': cues,
            'browser_event': {
                'type':        'avatar.drill',
                'clip_id':     clip_id,
                'counts':      counts,
                'repeats':     repeats,
                'speed_start': speed_start,
                'speed_end':   speed_end,
                'start_count': start_count,
                'end_count':   end_count,
                'mirror_alternate': mirror_alt,
                'cues':        cues,
            },
        }

    if name == 'slower':
        state.speed = max(0.25, state.speed * 0.5)
        return {'ok': True, 'speed': state.speed,
                'browser_event': {'type': 'avatar.speed', 'speed': state.speed}}

    if name == 'mirror':
        state.mirror = not state.mirror
        return {'ok': True, 'mirror': state.mirror,
                'browser_event': {'type': 'avatar.mirror', 'mirror': state.mirror}}

    if name == 'set_language':
        lang = str(args.get('language') or 'hinglish').lower()
        if lang not in ('english', 'hinglish', 'hindi'):
            lang = 'hinglish'
        state.coach_language = lang
        return {'ok': True, 'language': lang,
                'browser_event': {'type': 'avatar.language', 'language': lang}}

    if name == 'stop':
        return {'ok': True, 'browser_event': {'type': 'avatar.stop'}}

    if name == 'explain':
        topic = args.get('topic', '')
        return {'ok': True, 'topic': topic, 'browser_event': None}

    if name == 'set_mood':
        mood = args.get('mood', 'relaxed')
        return {
            'ok': True, 'mood': mood,
            'browser_event': {'type': 'avatar.mood', 'mood': mood},
        }

    if name == 'isolate':
        parts = args.get('parts') or []
        if isinstance(parts, str):
            parts = [parts]
        return {
            'ok': True, 'parts': parts,
            'browser_event': {'type': 'avatar.isolate', 'parts': parts},
        }

    if name == 'unisolate':
        return {
            'ok': True,
            'browser_event': {'type': 'avatar.unisolate'},
        }

    if name == 'break_down':
        # GUIDED BREAKDOWN — one tool call kicks off a multi-stage
        # client-side sequence: legs slow → arms slow → full slow →
        # full normal. The browser owns the timing; the LLM just
        # narrates the stage names.
        clip_id = args.get('move_id') or state.current_clip
        # v212: SELF-SUFFICIENT TEACH. In live voice the model calls
        # break_down cold for "teach me hip hop" with NO clip loaded yet.
        # Instead of dead-ending ("no clip selected" → avatar frozen /
        # just chatter), pick + load a clip for the requested style right
        # here so a single call BOTH shows the move AND teaches its steps
        # (with the side rail). When a clip is already playing (text
        # fast-path already called pick_and_play), _prepick_events stays
        # empty and we never double-load.
        _prepick_events = []
        if not clip_id:
            _pp = execute_tool(state, 'pick_and_play',
                               {'genre': args.get('genre') or args.get('style'),
                                'query': args.get('query'),
                                'loop': True, 'speed': 0.85,
                                'prefer_cued': True})
            if _pp.get('ok'):
                _prepick_events = [e for e in (_pp.get('browser_events')
                                   or [_pp.get('browser_event')]) if e]
                clip_id = state.current_clip
        if not clip_id:
            return {'ok': False, 'reason': 'no clip selected yet',
                    'browser_event': None}
        stage_seconds = float(args.get('stage_seconds') or 8.0)
        stage_seconds = max(4.0, min(16.0, stage_seconds))
        meta = motion_metadata.get_meta(clip_id) or {}
        # v190: LEARNED choreographies (from the video→learn pipeline) carry an
        # authored `learned_steps` breakdown produced by the vision model, WITH
        # the free/paid gate baked in per step (`locked`). Prefer these: teach
        # each named step full-body at half speed, stop at the first locked step
        # and surface an upgrade prompt so the free tier gets the first N moves.
        learned = meta.get('learned_steps') or []
        if isinstance(learned, list) and learned:
            steps = []
            locked_count = 0
            for s in learned:
                if not isinstance(s, dict):
                    continue
                if s.get('locked'):
                    locked_count += 1
                    continue
                idx = int(s.get('index') or (len(steps) + 1))
                label = str(s.get('label') or f'Step {idx}')
                count = str(s.get('count') or '').strip()
                cue = str(s.get('cue') or '').strip()
                detail = str(s.get('detail') or '').strip()
                mistake = str(s.get('mistake') or '').strip()
                # Compose a rich, teacher-quality line the live coach narrates:
                #   "Step 3 — Chest Pop (on the 5-6). <do-this cue>. <body detail>.
                #    Watch out: <common mistake>."
                head = f"Step {idx} — {label}"
                if count:
                    head += f" (on the {count})"
                text = head + "."
                if cue:
                    text += f' {cue}'
                if detail:
                    text += f' {detail}'
                if mistake:
                    text += f' Watch out: {mistake}'
                steps.append({'parts': [], 'speed': 0.5, 'cue': text,
                              'step': idx,
                              # segment window + label for 100% avatar/voice sync:
                              # the browser windows the avatar to EXACTLY this
                              # move's [start_s,end_s] and shows this caption while
                              # the cue is spoken, so what's SAID == what's SHOWN.
                              'start_s': float(s.get('start_s') or 0.0),
                              'end_s': float(s.get('end_s') or 0.0),
                              'label': label, 'count': count})
            steps.append({'parts': [], 'speed': 0.5,
                          'cue': 'Now all together — slow, breathe, feel the count.',
                          'label': 'All together (slow)', 'count': ''})
            steps.append({'parts': [], 'speed': 1.0,
                          'cue': 'Full speed. Own it — you\'ve got this.',
                          'label': 'Full speed', 'count': ''})
            unlocked_n = len([s for s in steps if s.get('step')])
            result = {
                'ok': True,
                'clip_id': clip_id,
                'stage_seconds': stage_seconds,
                'n_steps': unlocked_n,
                'learned': True,
                'locked_count': locked_count,
                'upgrade_required': locked_count > 0,
                'steps': [{'cue': s['cue'], 'parts': s['parts'],
                           'speed': s['speed'], 'step': s.get('step'),
                           'start_s': s.get('start_s'), 'end_s': s.get('end_s'),
                           'label': s.get('label'), 'count': s.get('count')}
                          for s in steps],
                'title': meta.get('title') or clip_id,
                'browser_event': {
                    'type': 'avatar.breakdown',
                    'clip_id': clip_id,
                    'stage_seconds': stage_seconds,
                    'stages': steps,
                    'locked_count': locked_count,
                    'upgrade_required': locked_count > 0,
                },
            }
            # v212: if we auto-picked a clip above, load+play it BEFORE the
            # breakdown so the browser has motion data to window per step.
            if _prepick_events:
                result['browser_events'] = _prepick_events + [result['browser_event']]
            return result
        # v129: PREFER the human-authored key_cues (real dance vocabulary —
        # "Heel Toe", "Bounce Down", "Jacking Up", "Freeze") over the
        # joint-jitter segmenter, which mislabels footwork as "left arm out"
        # and teaches nothing. Each cue runs from its beat to the next, full
        # body at half speed so the student sees the named move on its count.
        kc = meta.get('key_cues') or []
        kc = _normalize_cues(kc)   # v133: tolerate dict/str-shaped cues
        counts = int(meta.get('counts') or 0) or None
        steps = []
        if isinstance(kc, list) and len(kc) >= 2 and all(
                isinstance(c, dict) and c.get('cue') for c in kc):
            n = len(kc)
            for i, c in enumerate(kc):
                b0 = int(c.get('beat') or (i + 1))
                b1 = int(kc[i + 1].get('beat')) if i + 1 < n else (
                    (counts or b0) + 1)
                steps.append({
                    'parts': [], 'speed': 0.5,
                    'cue': f"{c['cue']} — count {b0}.",
                    'step': i + 1,
                    'beat_start': b0, 'beat_end': max(b0, b1 - 1),
                })
            steps.append({'parts': [], 'speed': 0.5,
                          'cue': 'Now all together — slow.'})
            steps.append({'parts': [], 'speed': 1.0,
                          'cue': 'Full speed. Ride it.'})
        # v107: build REAL, move-specific numbered steps from the actual
        # motion (segmenter) instead of the old generic legs/arms/full
        # sequence. Each step isolates the body region it lives in and
        # plays slow, so the student learns "step 1, step 2, step 3..."
        # with the avatar demonstrating exactly that chunk.
        steps_info = {'steps': []}
        try:
            steps_info = motion_analyzer.steps_for(clip_id)
        except Exception:
            steps_info = {'steps': []}
        seg_steps = steps_info.get('steps') or []
        if not steps and seg_steps:
            for s in seg_steps:
                steps.append({
                    'parts': s.get('parts') or [],
                    'speed': 0.5,
                    'cue': f"Step {s['step']} — {s['name']}.",
                    'step': s['step'],
                    'frame_start': s.get('frame_start'),
                    'frame_end': s.get('frame_end'),
                })
            # then assemble + ride at full speed
            steps.append({'parts': [], 'speed': 0.5,
                          'cue': 'Now all together — slow.'})
            steps.append({'parts': [], 'speed': 1.0,
                          'cue': 'Full speed. Ride it.'})
        elif not steps:
            # Fallback: steady groove with no distinct segments → the
            # classic legs/arms/full progression.
            steps = [
                {'parts': ['legs'], 'speed': 0.5,
                 'cue': 'Legs only — half speed. Just the footwork.'},
                {'parts': ['arms'], 'speed': 0.5,
                 'cue': 'Now arms only — half speed. Feel the upper body.'},
                {'parts': [],      'speed': 0.5,
                 'cue': 'Put it together, slow. Everything at once.'},
                {'parts': [],      'speed': 1.0,
                 'cue': 'Full speed. Ride it.'},
            ]
        _bd_result = {
            'ok': True,
            'clip_id': clip_id,
            'stage_seconds': stage_seconds,
            'n_steps': len([s for s in steps if s.get('step')]),
            'steps': [{'cue': s['cue'], 'parts': s['parts'],
                       'speed': s['speed'], 'step': s.get('step')}
                      for s in steps],
            'title': meta.get('title') or clip_id,
            'browser_event': {
                'type': 'avatar.breakdown',
                'clip_id': clip_id,
                'stage_seconds': stage_seconds,
                'stages': steps,
            },
        }
        # v212: fan out the auto-picked load+play events (if any) ahead of
        # the breakdown so the avatar shows the clip, then teaches its steps.
        if _prepick_events:
            _bd_result['browser_events'] = _prepick_events + [_bd_result['browser_event']]
        return _bd_result

    if name == 'live_feedback':
        clip_id = args.get('clip_id') or state.current_clip
        if not clip_id:
            return {'ok': False, 'reason': 'no clip selected yet',
                    'browser_event': None}
        return {
            'ok': True, 'clip_id': clip_id,
            'browser_event': {
                'type': 'ui.open_live_feedback',
                'clip_id': clip_id,
            },
        }

    if name == 'resequence_to_music':
        return {
            'ok': True,
            'browser_event': {
                'type': 'ui.open_audio_picker',
                'genre': args.get('genre'),
                'query': args.get('query'),
                'bars':  int(args.get('bars', 8)),
            },
        }

    if name == 'give_feedback':
        clip_id = args.get('clip_id') or state.current_clip
        if not clip_id:
            return {'ok': False, 'reason': 'no clip selected yet',
                    'browser_event': None}
        return {
            'ok': True, 'clip_id': clip_id,
            'browser_event': {
                'type': 'ui.open_video_picker',
                'clip_id': clip_id,
            },
        }

    if name == 'open_lessons':
        return {
            'ok': True,
            'browser_event': {
                'type': 'ui.open_learn',
                'style': args.get('style'),
            },
        }

    if name == 'open_lesson':
        from coach import curriculum
        les = curriculum.find_lesson(args.get('style') or '',
                                     args.get('move') or '')
        if not les:
            return {
                'ok': False, 'reason': 'no matching lesson',
                'browser_event': {'type': 'ui.open_learn',
                                  'style': args.get('style')},
            }
        return {
            'ok': True, 'lesson_id': les['id'], 'lesson_title': les['title'],
            'browser_event': {
                'type': 'ui.open_lesson',
                'lesson_id': les['id'],
                'track_id': les['track_id'],
            },
        }

    if name == 'open_profile':
        return {
            'ok': True,
            'browser_event': {'type': 'ui.open_profile'},
        }

    if name == 'close_panel':
        # Dismiss whatever pane is open (steps rail / lessons / chat / mirror)
        # so the avatar is visible full-screen again. The browser closes all of
        # them; harmless if nothing is open.
        return {
            'ok': True,
            'browser_event': {'type': 'ui.close_panels'},
        }

    if name == 'search_clips':
        from coach import semantic_search
        query = (args.get('query') or '').strip()
        if not query:
            return {'ok': False, 'reason': 'empty query',
                    'browser_event': None}
        try:
            hits = semantic_search.search(
                query,
                k=int(args.get('k', 5)),
                genre=args.get('genre'),
            )
        except Exception as e:
            return {'ok': False, 'reason': f'search error: {e}',
                    'browser_event': None}
        # Auto-pick the top hit so the LLM can immediately call play().
        if hits:
            state.current_clip = hits[0]['id']
            top = hits[0]
            state.remember_clip(top['id'],
                                top.get('genre', '?'),
                                top.get('summary', '')[:80])
        return {
            'ok':       True,
            'query':    query,
            'results':  hits,
            'top_clip': hits[0]['id'] if hits else None,
            'browser_event': None,
        }

    # ── COACH-DRIVEN SESSION TOOLS ──────────────────────────────
    # The session is a phase machine (warmup → drill → routine →
    # cooldown) run on the SERVER: the WS background ticker ticks the
    # clock and emits browser events; these expose the control surface
    # so the server (or the LLM) can start / advance / pause / end it.
    if name == 'start_session':
        tpl_id = args.get('template_id')
        if not tpl_id:
            sty = (state.character_style or '').lower()
            sty_map = {
                'house': 'gHO', 'la-hiphop': 'gLH', 'hiphop': 'gLH',
                'krump': 'gKR', 'waacking': 'gWA', 'jazz': 'gJS',
                'breaking': 'gBR', 'b-boy': 'gBR', 'b-girl': 'gBR',
                'locking': 'gLO', 'popping': 'gPO',
            }
            genre = sty_map.get(sty, 'gHO')
            tpl_id = f'quick5_{genre}'
        tpl = coach_session.get_template(tpl_id)
        if tpl is None:
            return {'ok': False, 'reason': f'unknown template {tpl_id!r}',
                    'browser_event': None}
        state.session = coach_session.Session(template=tpl)
        snap = state.session.snapshot()
        return {
            'ok': True,
            'session': snap,
            'browser_event': {
                'type': 'session.started',
                'session': snap,
            },
        }

    if name == 'advance_phase':
        sess = state.session
        if sess is None or sess.finished:
            return {'ok': False, 'reason': 'no active session',
                    'browser_event': None}
        sess.advance()
        snap = sess.snapshot()
        return {
            'ok': True,
            'session': snap,
            'finished': sess.finished,
            'browser_event': {
                'type': ('session.finished' if sess.finished
                         else 'session.phase'),
                'session': snap,
            },
        }

    if name == 'pause_session':
        sess = state.session
        if sess is None or sess.finished:
            return {'ok': False, 'reason': 'no active session',
                    'browser_event': None}
        sess.pause()
        return {'ok': True, 'session': sess.snapshot(),
                'browser_event': {'type': 'session.paused',
                                  'session': sess.snapshot()}}

    if name == 'resume_session':
        sess = state.session
        if sess is None or sess.finished:
            return {'ok': False, 'reason': 'no active session',
                    'browser_event': None}
        sess.resume()
        return {'ok': True, 'session': sess.snapshot(),
                'browser_event': {'type': 'session.resumed',
                                  'session': sess.snapshot()}}

    if name == 'end_session':
        sess = state.session
        if sess is None:
            return {'ok': False, 'reason': 'no active session',
                    'browser_event': None}
        sess.end()
        snap = sess.snapshot()
        state.session = None
        return {'ok': True, 'session': snap,
                'browser_event': {'type': 'session.finished',
                                  'session': snap}}

    if name == 'session_status':
        sess = state.session
        if sess is None:
            return {'ok': True, 'active': False, 'browser_event': None}
        return {'ok': True, 'active': True,
                'session': sess.snapshot(),
                'browser_event': None}

    return {'ok': False, 'reason': f'unknown tool {name}',
            'browser_event': None}


# ── session-ticker clip director ────────────────────────────────────
# Used by the server's background session ticker (warmup → drill →
# combo → cooldown) to auto-pick the next clip for the active phase.
# Restored after a refactor accidentally dropped it (the WS agent
# imports it, so its absence broke the live coach connection).

def _clip_text(meta: Dict[str, Any], clip_id: str) -> str:
    title = str(meta.get('title') or '')
    summary = str(meta.get('summary') or '')
    tags = ' '.join(str(t) for t in (meta.get('vibe_tags') or []))
    return f'{clip_id} {title} {summary} {tags}'.lower()


def _is_good_warmup_clip(clip_id: str) -> bool:
    meta = motion_metadata.get_meta(clip_id) or {}
    text = _clip_text(meta, clip_id)
    bad = ('walk', 'jog', 'gait', 'stride', 'turn')
    good = ('wave', 'stretch', 'reach', 'sway', 'bounce',
            'circle', 'step', 'lunge', 'swivel')
    return not any(k in text for k in bad) and any(k in text for k in good)


def _is_good_cooldown_clip(clip_id: str) -> bool:
    meta = motion_metadata.get_meta(clip_id) or {}
    text = _clip_text(meta, clip_id)
    bad = ('walk', 'jog', 'gait', 'stride')
    good = ('stretch', 'sway', 'breath', 'slow', 'reach', 'wave')
    return not any(k in text for k in bad) and any(k in text for k in good)


def _is_clean_drill_clip(clip: Dict[str, Any]) -> bool:
    dur = float(clip.get('duration_sec') or clip.get('duration') or 0.0)
    if dur and dur < 3.0:
        return False
    meta = motion_metadata.get_meta(clip['id']) or {}
    text = _clip_text(meta, clip['id'])
    bad = ('walk', 'jog', 'stand', 'static', 'still', 'turn', 'stretch')
    return not any(k in text for k in bad)


def _move_key(clip_id: str) -> str:
    """Strip trailing _chXX suffix → move identity. Two AIST clips of
    the same choreography from different camera angles share a
    move_key, so the variety engine treats them as one move."""
    return re.sub(r'_ch\d+$', '', clip_id)


def _rotate_pick(candidates: List[Dict[str, Any]],
                 recent_session: set) -> Optional[Dict[str, Any]]:
    """Pick a clip preferring ones not served recently ACROSS sessions
    (the `_RECENT_SERVED` ring buffer) and not just-played this session.
    This is what makes warmups rotate through the WHOLE pool instead of
    repeating the same handful in every session."""
    if not candidates:
        return None
    cross = set(_RECENT_SERVED)
    fresh = [m for m in candidates
             if m['id'] not in cross and m['id'] not in recent_session]
    if not fresh:
        fresh = [m for m in candidates if m['id'] not in recent_session]
    if not fresh:
        fresh = candidates
    fresh = _drop_just_served(fresh)
    return random.choice(fresh)


def pick_session_clip(state: 'CoachState') -> Optional[Dict[str, Any]]:
    """Pick the next clip for the active session's current phase,
    honouring phase intent (warmup / drill_one_move / combo /
    freestyle / cooldown / rest). Returns a motion-index-shaped dict
    with extra ``role`` / ``title`` / ``music_url`` fields, or None for
    'rest' phases."""
    sess = state.session
    if sess is None or sess.finished:
        return None
    phase = sess.current
    if phase is None or phase.intent == 'rest':
        return None
    all_clips = motion_index.list_motions()
    style = phase.style or sess.template.style
    if style == 'cmu':
        pool = [m for m in all_clips
                if m.get('genre') == 'cmu'
                and m.get('safety') in ('ok', 'unknown')
                and m['id'] not in INVERTED_BLACKLIST]
    else:
        pool = [m for m in all_clips
                if m.get('genre') == style
                and m.get('safety') in ('ok', 'unknown')
                and m['id'] not in INVERTED_BLACKLIST]
    retarg = [m for m in pool if m.get('retargeted')]
    if retarg:
        pool = retarg
    if not pool:
        return None

    intent = phase.intent
    played = list(sess.played_clips)
    recent_session = set(played[-4:])

    def _finalize(pick: Dict[str, Any]) -> Dict[str, Any]:
        sess.played_clips.append(pick['id'])
        if len(sess.played_clips) > 64:
            sess.played_clips = sess.played_clips[-64:]
        _RECENT_SERVED.append(pick['id'])
        state.current_clip = pick['id']
        state.current_bpm = pick.get('bpm_target')
        meta = motion_metadata.get_meta(pick['id']) or {}
        state.remember_clip(pick['id'], pick.get('genre', ''),
                            meta.get('summary', ''))
        out = dict(pick)
        out['title'] = meta.get('title') or pick['id']
        out['summary'] = meta.get('summary', '')
        out['role'] = intent
        out['music_url'] = motion_index.music_url_for(pick['id'])
        # v83: slow the gentle phases so the movement is CLEAR (user
        # feedback: clips too fast, arms/legs unclear). Teaching drills
        # play slow too; combos a touch under tempo; freestyle full.
        out['speed'] = {
            'warmup': 0.6, 'cooldown': 0.55, 'drill_one_move': 0.7,
            'combo': 0.85, 'freestyle': 1.0,
        }.get(intent, 0.8)
        # v144: for warm-up / cooldown clips, attach the rich mobility
        # teaching lesson so the narration LLM can coach it properly
        # (calm breath-led cues, setup, sensation) instead of dance rhythm.
        if intent in ('warmup', 'cooldown'):
            _lesson = _WARMUP_LESSON_BY_CLIP.get(pick['id'])
            if _lesson:
                out['teaching'] = _lesson
        return out

    # PHASE-LEVEL curated pool override (e.g. Stretch & Warmup routine).
    phase_pool_ids = getattr(phase, 'clip_pool', None) or []
    if phase_pool_ids:
        wl_set = set(phase_pool_ids)
        curated = [m for m in all_clips
                   if m['id'] in wl_set
                   and m.get('safety') in ('ok', 'unknown')
                   and m['id'] not in INVERTED_BLACKLIST]
        retarg_curated = [m for m in curated if m.get('retargeted')]
        curated_pool = retarg_curated or curated
        if curated_pool:
            pick = _rotate_pick(curated_pool, recent_session)
            return _finalize(pick)

    # DRILL: lock onto one move for the whole phase (rotate camera
    # angles only, never to a different move).
    if intent == 'drill_one_move':
        anchor = getattr(sess, '_drill_anchor', None)
        drill_pool = [m for m in pool if _is_clean_drill_clip(m)] or pool
        if anchor is None:
            taught = {_move_key(p) for p in played}
            fresh_moves = [m for m in drill_pool
                           if _move_key(m['id']) not in taught]
            candidates = fresh_moves or drill_pool
            chosen = _rotate_pick(candidates, set()) or random.choice(candidates)
            anchor = _move_key(chosen['id'])
            sess._drill_anchor = anchor  # type: ignore[attr-defined]
        same_move = [m for m in drill_pool if _move_key(m['id']) == anchor]
        recent2 = set(played[-2:])
        fresh = [m for m in same_move if m['id'] not in recent2]
        bucket = fresh or same_move or drill_pool or pool
        pick = random.choice(bucket)

    # COMBO: rotate across DIFFERENT moves of this style.
    elif intent == 'combo':
        used_moves = {_move_key(p) for p in played[-6:]}
        diff = [m for m in pool if _move_key(m['id']) not in used_moves]
        pick = _rotate_pick(diff or pool, recent_session)

    # FREESTYLE: uniform random across style; dedupe + cross-session.
    elif intent == 'freestyle':
        pick = _rotate_pick(pool, recent_session)

    # WARMUP / COOLDOWN: curated gentle CMU pools. Warmup now draws from
    # the FULL union of the curated stretch pools (upper/mid/lower/
    # dynamic/breath) and rotates across sessions via _RECENT_SERVED so
    # the user stops seeing the same 3-4 clips every session.
    else:
        all_cmu = _warmup_clip_pool(all_clips)
        retarg_cmu = [m for m in all_cmu if m.get('retargeted')] or all_cmu
        if intent == 'warmup':
            wl = set(coach_session._SW_UPPER + coach_session._SW_MID +
                     coach_session._SW_LOWER + coach_session._SW_DYNAMIC +
                     coach_session._SW_BREATH)
            # The curated _SW_* pools were already hand-verified tilt-safe
            # in session.py, so use ALL of them — don't double-filter with
            # the keyword heuristic (that cut 20 good clips down to 8 and
            # is what made warmups feel repetitive). Only fall back to the
            # keyword filter when picking from the broad untrusted pool.
            tagged = [m for m in retarg_cmu if m['id'] in wl]
            if not tagged:
                tagged = [m for m in retarg_cmu if _is_good_warmup_clip(m['id'])]
        else:
            wl = set(coach_session._SW_BREATH + coach_session._SW_LOWER)
            tagged = [m for m in retarg_cmu if m['id'] in wl]
            if not tagged:
                tagged = [m for m in retarg_cmu
                          if _is_good_cooldown_clip(m['id'])]
        candidates = tagged or retarg_cmu
        pick = _rotate_pick(candidates, set(played[-3:]))

    if pick is None:
        return None
    return _finalize(pick)
