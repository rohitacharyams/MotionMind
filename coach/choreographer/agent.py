"""agent.py — Groq-backed dance coach agent.

Streams events to the browser:
    { type: 'assistant_text', text: '...' }
    { type: 'tool_call', name, args, result }
    { type: 'avatar_event', event: {...} }
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from coach.choreographer.prompts import system_prompt
from coach.choreographer.tools import (TOOL_SCHEMAS, CoachState, execute_tool)

GROQ_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
# Fallback chain — when the primary model hits its daily TPD limit
# (Groq free tier: 70B = 100K/day, 8B-instant = 500K/day, gemma2-9b
# also has a generous bucket) we transparently retry with the next
# model. This keeps the coach alive long after the 70B bucket drains.
GROQ_FALLBACKS = [m.strip() for m in os.getenv(
    'GROQ_FALLBACK_MODELS',
    'llama-3.1-8b-instant,gemma2-9b-it,llama-3.1-8b-instant'
).split(',') if m.strip()]

# Multi-key rotation — pool every key in GROQ_API_KEYS (comma-separated).
# When one key's daily bucket drains we transparently rotate to the next.
# Combined with the model fallback chain above this effectively removes
# the user-visible quota for casual sessions.
_keys_env = os.getenv('GROQ_API_KEYS', '')
GROQ_KEYS = [k.strip() for k in _keys_env.split(',') if k.strip()]
if GROQ_KEY and GROQ_KEY not in GROQ_KEYS:
    GROQ_KEYS.insert(0, GROQ_KEY)
if not GROQ_KEY and GROQ_KEYS:
    GROQ_KEY = GROQ_KEYS[0]

_state = CoachState()  # legacy fallback (single-user). Real per-WS
# state is now passed in to run_turn; this default is only used by
# scripts that call run_turn(history) with no state argument.

try:
    from groq import AsyncGroq
    # One client per key so we can swap by index without re-init cost.
    _clients = [AsyncGroq(api_key=k) for k in GROQ_KEYS] if GROQ_KEYS else []
    _client = _clients[0] if _clients else None
except Exception:                                              # noqa: BLE001
    _clients = []
    _client = None

# ─── Azure OpenAI primary (Visual Studio Enterprise subscription) ────
# When AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY are set, we use it
# as the PRIMARY LLM. Groq remains as the multi-key fallback so the
# coach never goes dark.
AOAI_ENDPOINT    = os.getenv('AZURE_OPENAI_ENDPOINT', '').rstrip('/')
AOAI_KEY         = os.getenv('AZURE_OPENAI_API_KEY', '')
AOAI_DEPLOYMENT  = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
AOAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
try:
    from openai import AsyncAzureOpenAI
    _aoai = AsyncAzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_KEY,
        api_version=AOAI_API_VERSION,
    ) if (AOAI_ENDPOINT and AOAI_KEY) else None
except Exception:                                              # noqa: BLE001
    _aoai = None


# ─── v81: LIVE SESSION NARRATOR (LLM-backed) ──────────────────────────
# Guided sessions used to speak hardcoded template strings -> robotic,
# repetitive, English-only, no questions. This generates ONE short,
# alive, in-character, in-LANGUAGE line per beat via the fast writer
# model. Returns None on any failure so the caller falls back to the
# scripted line (the session never goes silent).
SESSION_WRITER_MODEL = os.getenv('GROQ_WRITER_MODEL', 'llama-3.1-8b-instant')

_SESS_LANG_RULE = {
    'hinglish': ("Speak in HINGLISH — natural casual Hindi+English in Roman "
                 "script (e.g. 'chalo shoulders thoda loose karo, kaisa lag "
                 "raha hai?'). Keep movement words in English."),
    'hindi': ("Reply mostly in HINDI (Devanagari fine), keeping movement words "
              "(warmup, groove, stretch) in English."),
    'english': "Speak in warm, natural English.",
}


async def llm_session_line(*, kind, intent='', phase_title='', clip_title='',
                           clip_cues='', user_name='', language='english',
                           char_name='', char_style='', recent_lines=None,
                           ask_question=False, timeout=2.6, teaching=None):
    """Generate one short, engaging session narration line. kind is
    'phase' (new phase begins), 'clip' (a new move started), or
    'heartbeat' (gentle check-in during a lull). Returns the line or
    None (caller uses the scripted fallback)."""
    client = _client
    if client is None and _aoai is None:
        return None
    lang = (language or 'english').lower()
    if lang not in _SESS_LANG_RULE:
        lang = 'english'
    who = char_name or 'a warm, upbeat movement coach'
    style = f' ({char_style} specialist)' if char_style else ''
    name_bit = f" The person's name is {user_name}; use it RARELY — at most once in a few lines, and most lines should NOT say their name." if user_name else ''
    q_bit = (" END with a short, genuine question to the person (how it feels, "
             "are they with you, want it slower)." if ask_question else
             " Occasionally (not every time) ask them a quick question.")
    # v81b: never leak raw clip-ids / genre codes (e.g. "cmu", "gHO_...")
    # into the spoken line. Drop them from the title/cues before prompting.
    import re as _re
    def _clean_cue(x):
        if not x:
            return ''
        x = _re.sub(r'\b(?:cmu|g[A-Z]{2})[\w]*\b', '', str(x))
        x = _re.sub(r'\s{2,}', ' ', x).strip(' ,.-')
        return x
    clip_title = _clean_cue(clip_title)
    clip_cues = _clean_cue(clip_cues)
    kind_bit = {
        'phase': (f"You're moving the session into a new phase: '{phase_title}' "
                  f"({intent}). Announce the shift with real energy and warmth."),
        'clip': (f"A new move just started" + (f": '{clip_title}'." if clip_title else ".") +
                 (f" Coach it using these cues: {clip_cues}." if clip_cues else "") +
                 " React like you're moving WITH them."),
        'heartbeat': ("There's been a quiet moment mid-session. Keep them company "
                      "and motivated — like a friend right next to them."),
    }.get(kind, "Say a short, warm coaching line.")
    # v144: MOBILITY / WARM-UP TEACHING MODE. When a warm-up/cooldown clip
    # carries a rich teaching lesson, coach it like a real mobility trainer:
    # calm, breath-led, body-part + sensation focused — NOT the dance
    # 5-6-7-8 rhythm. Feed the authored setup/steps/breathing/sensation.
    _mobility = None
    if teaching and intent in ('warmup', 'cooldown'):
        try:
            _steps = teaching.get('teaching_script') or []
            _cue_lines = ' '.join(
                str(s.get('say', '')) for s in _steps if isinstance(s, dict))[:400]
            _miss = (teaching.get('common_mistakes') or [None])[0]
            _mobility = (
                f"You are guiding a GENTLE WARM-UP / MOBILITY move: "
                f"'{teaching.get('lesson', clip_title)}' "
                f"({teaching.get('body_part', '')}). "
                f"Setup: {teaching.get('setup', '')} "
                f"Coach it in a CALM, breath-led way using these cues: {_cue_lines} "
                f"Breathing: {teaching.get('breathing', '')} "
                f"They should feel: {teaching.get('target_sensation', '')} "
                + (f"Watch for: {_miss} " if _miss else '')
                + "Speak like a real mobility coach moving WITH them — slow and "
                + "reassuring, name the body part and the breath. NOT a dance count."
            )
        except Exception:
            _mobility = None
    if _mobility:
        kind_bit = _mobility
    sys = (
        f"You are {who}{style}, coaching someone LIVE through a movement session, "
        f"out loud. Real friend energy: warm, specific, a little playful, never "
        f"corny or robotic. ONE short spoken sentence, max ~18 words. "
        f"{_SESS_LANG_RULE[lang]} {name_bit}{q_bit} "
        f"NEVER mention being an AI. NEVER say internal codes like 'cmu' or clip "
        f"ids. NEVER use lists, numbers, or stage directions. "
        f"Output ONLY the spoken line."
    )
    recent = [r for r in (recent_lines or []) if r][-6:]
    avoid = (" Do NOT repeat any of these recent lines: " +
             " | ".join(recent)) if recent else ''
    usr = kind_bit + avoid
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=SESSION_WRITER_MODEL,
                messages=[{'role': 'system', 'content': sys},
                          {'role': 'user', 'content': usr}],
                temperature=0.85,
                max_tokens=60,
                timeout=timeout,
            ),
            timeout=timeout,
        ) if client is not None else None
        if resp is None and _aoai is not None:
            resp = await asyncio.wait_for(
                _aoai.chat.completions.create(
                    model=AOAI_DEPLOYMENT,
                    messages=[{'role': 'system', 'content': sys},
                              {'role': 'user', 'content': usr}],
                    temperature=0.85, max_tokens=60, timeout=timeout),
                timeout=timeout)
        if resp is None:
            return None
        line = (resp.choices[0].message.content or '').strip()
        line = _clean_speech(line)
        # one sentence, strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if len(line) > 160:
            line = line[:160].rsplit(' ', 1)[0] + '…'
        return line or None
    except Exception:                                          # noqa: BLE001
        return None



async def summarize_dialogue_memory(history, prior, user_name='',
                                    timeout=4.0):
    """Distil the running conversation (+ any prior memory) into a small,
    durable JSON memory the coach can reuse on future days. Returns a dict
    like {summary, facts:[...], goals:[...], session_count, updated_at} or
    the prior memory unchanged on failure (never raises)."""
    import datetime as _dt
    prior = prior if isinstance(prior, dict) else {}
    # Build a compact transcript (last ~24 turns) for the writer.
    turns = []
    for m in (history or [])[-24:]:
        role = str(m.get('role') or '').strip().lower()
        text = str(m.get('content') or '').strip()
        if role in ('user', 'assistant') and text:
            who = (user_name or 'Student') if role == 'user' else 'Coach'
            turns.append(f"{who}: {text[:240]}")
    if not turns:
        return prior or None
    transcript = "\n".join(turns)
    prior_json = json.dumps({
        'summary': prior.get('summary', ''),
        'facts': prior.get('facts', [])[:12],
        'goals': prior.get('goals', [])[:6],
    }, ensure_ascii=False)
    sys = (
        "You maintain a dance coach's long-term memory of ONE student. "
        "Merge the PRIOR memory with the NEW conversation into an updated, "
        "compact memory. Keep only durable, useful facts (their name, goals, "
        "favourite styles, skill level, injuries/limits, what they struggled "
        "with or enjoyed, mood patterns). Drop small talk. Output STRICT "
        "JSON only, no prose, with keys: summary (<=320 chars, warm 2-3 "
        "sentences a coach would note), facts (array of <=12 short strings), "
        "goals (array of <=6 short strings)."
    )
    usr = (f"PRIOR MEMORY:\n{prior_json}\n\nNEW CONVERSATION:\n{transcript}"
           "\n\nReturn the updated memory JSON.")
    client = _client
    raw = None
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=SESSION_WRITER_MODEL,
                messages=[{'role': 'system', 'content': sys},
                          {'role': 'user', 'content': usr}],
                temperature=0.2, max_tokens=400,
                response_format={'type': 'json_object'},
                timeout=timeout),
            timeout=timeout) if client is not None else None
        if resp is None and _aoai is not None:
            resp = await asyncio.wait_for(
                _aoai.chat.completions.create(
                    model=AOAI_DEPLOYMENT,
                    messages=[{'role': 'system', 'content': sys},
                              {'role': 'user', 'content': usr}],
                    temperature=0.2, max_tokens=400,
                    response_format={'type': 'json_object'},
                    timeout=timeout),
                timeout=timeout)
        if resp is not None:
            raw = (resp.choices[0].message.content or '').strip()
    except Exception:                                          # noqa: BLE001
        raw = None
    if not raw:
        return prior or None
    try:
        data = json.loads(raw)
    except Exception:                                          # noqa: BLE001
        # tolerate a stray ```json fence
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return prior or None
        try:
            data = json.loads(m.group(0))
        except Exception:                                      # noqa: BLE001
            return prior or None
    summary = str(data.get('summary') or prior.get('summary') or '').strip()[:320]
    facts = [str(x).strip()[:120] for x in (data.get('facts') or []) if str(x).strip()][:12]
    goals = [str(x).strip()[:120] for x in (data.get('goals') or []) if str(x).strip()][:6]
    if not (summary or facts or goals):
        return prior or None
    out = {
        'summary': summary,
        'facts': facts,
        'goals': goals,
        'session_count': int(prior.get('session_count') or 0) + 1,
        'updated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
    }
    return out



# llama-3.3 / 8B-instant sometimes EMIT a tool call as plain text
# instead of using the JSON tool-calls field, which makes Groq return
# 400 tool_use_failed. We salvage it by parsing both shapes the wild
# emits:  `<function=NAME{json}</function>`  and the more common
# `<function=NAME>{json}</function>` (extra `>` after the name).
_FN_TEXT_RE = re.compile(
    r'<function\s*=\s*([A-Za-z_][\w]*)\s*>?\s*(\{.*?\})\s*</function\s*>?',
    re.DOTALL)
# Strip any leftover function-tag debris from a visible chat bubble.
_FN_STRIP_RE = re.compile(
    r'<function\s*=[^<]*?</function\s*>?', re.DOTALL)


def _clean_speech(text: str) -> str:
    """Remove any `<function=...>` debris from a chat-visible string
    so the student never sees raw tool syntax leak into the bubble.
    Also collapses LLM "wall-of-text 8-count" breakdowns into a single
    bouncy call-line, because no real teacher reads a numbered list.
    """
    if not text:
        return text
    cleaned = _FN_STRIP_RE.sub('', text)
    # Also nuke orphan opens (model truncated before the close tag).
    cleaned = re.sub(r'<function\s*=.*$', '', cleaned, flags=re.DOTALL)
    cleaned = _compress_breakdown(cleaned)
    return cleaned.strip()


# v33b: collapse numbered "One, X, two, Y, three, Z..." breakdowns.
# The LLM keeps producing them despite the prompt rule because it's
# what its training data treats as "dance teaching". So we just rip
# them out at the boundary.
_COUNT_TOKEN_RE = re.compile(
    r'(?im)(?:^|[,\.\n;\u2014\-])\s*(?:and\s+)?'
    r'(one|two|three|four|five|six|seven|eight|1|2|3|4|5|6|7|8)\s*[,\u2014\-:]')
_FIVE_SIX_SEVEN_EIGHT_RE = re.compile(
    r'(?i)(?:and\s+)?five[\s,]+six[\s,]+seven[\s,]+eight[\s\.,!\u2014\-]*')


def _compress_breakdown(text: str) -> str:
    """If the LLM wrote a long numbered 8-count list, replace it with a
    short bouncy call-line. Triggered on 3+ numbered tokens in a
    message > 140 chars."""
    if not text or len(text) < 140:
        return text
    tokens = _COUNT_TOKEN_RE.findall(text)
    if len(tokens) < 3:
        return text
    # Find a "5,6,7,8" anchor; if absent, synthesize one.
    m = _FIVE_SIX_SEVEN_EIGHT_RE.search(text)
    if m:
        # Keep everything BEFORE the anchor (intro line like "Alright,
        # here we go —"), drop the numbered list AFTER it, replace with
        # a short bouncy call.
        intro = text[: m.start()].rstrip(' \u2014-,.;:').strip()
        tail  = "Five, six, seven, EIGHT — let's go! Ride that pocket."
        if intro:
            return f"{intro} {tail}"
        return tail
    # No 5,6,7,8 anchor — just truncate at the first numbered token.
    first = _COUNT_TOKEN_RE.search(text)
    if first:
        intro = text[: first.start()].rstrip(' \u2014-,.;:').strip()
        tail  = "Five, six, seven, EIGHT — let's go!"
        if intro and len(intro) > 20:
            return f"{intro} {tail}"
        return tail
    return text

# v33e (MVP rule R-P0-6): hard cap on every spoken response so the
# coach never monologues. ≤ 2 sentences / ≤ 220 chars by default.
# Long-form is reserved for explicit "explain in detail" intent which
# we don't auto-detect yet — the wall is unconditional for now.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?])\s+')


def _clip_response(text: str,
                   max_sentences: int = 2,
                   max_chars: int = 220) -> str:
    """Trim chat-bubble + TTS text to MVP cadence rules.

    1. Strip function-tag debris and compress numbered breakdowns
       (delegates to existing _clean_speech / _compress_breakdown).
    2. Keep first N sentences.
    3. Hard char cap with ellipsis if still over.
    """
    if not text:
        return text
    cleaned = _clean_speech(text)
    parts = _SENTENCE_SPLIT_RE.split(cleaned.strip())
    parts = [p for p in parts if p]
    if len(parts) > max_sentences:
        cleaned = ' '.join(parts[:max_sentences])
    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars].rstrip()
        # Prefer to cut at the last sentence-ender before the cap.
        for end in ('.', '!', '?'):
            i = cut.rfind(end)
            if i > max_chars - 80:
                cut = cut[:i + 1]
                break
        cleaned = cut.rstrip(' ,;:—-') + ('…' if not cleaned.startswith(cut) else '')
        if not cleaned.endswith(('.', '!', '?', '…')):
            cleaned += '…'
    return cleaned


# v33f-2: post-LLM DEFLECTION REPAIR. Groq llama-3.3 keeps producing
# stuck-record small-talk ("Pretty good — just vibing to some beats,
# how about you?") even when the prompt explicitly bans it and the
# user asked a direct question. When we detect that pattern AND the
# state knows what's actually playing, we replace the LLM output with
# a deterministic state-aware reply so the coach can never be caught
# saying "just vibing" when the user just asked "what step is this?".
_DEFLECT_PATTERNS = (
    'just vibing', 'just vibin', 'pretty good', 'how about you',
    'how about yourself', "what's up with you", 'just chilling',
    'just chillin', 'vibing to some', 'vibing with some',
    'sounds like you', 'in a thinking mood', 'groove of your own',
)
_QUESTION_HINTS = (
    'what', 'which', 'why', 'how ', 'who ', 'when', 'where',
    'name', 'called', 'doing', 'playing', 'tell me', '?',
)


def _looks_like_deflection(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _DEFLECT_PATTERNS)


def _looks_like_question(user_text: str) -> bool:
    if not user_text:
        return False
    low = user_text.lower()
    return any(h in low for h in _QUESTION_HINTS)


def _repair_deflection(text: str, state, user_text: str) -> str:
    """If the LLM gave a generic deflection to a real question, swap
    in a deterministic answer that uses the actual session state.

    v23: NEUTERED. Was hijacking off-topic chat questions ("how is the
    weather?", "how is the day going?") and replacing the LLM's reply
    with the current clip's metadata ("This one's a House clip — teaches
    basic house footwork…"). Users found this jarring: a chat question
    deserves a chat answer, not a clip card. Now a no-op; the LLM's
    natural reply passes through unchanged.
    """
    return text
    # --- legacy body kept for reference, unreachable ---
    if not _looks_like_deflection(text):
        return text
    if not _looks_like_question(user_text):
        return text
    cur = getattr(state, 'current_clip', None)
    played = getattr(state, 'played_clips', None) or []
    last = played[-1] if (played and isinstance(played[-1], dict)) else {}
    gid = last.get('genre') or ''
    summary = (last.get('summary') or '').strip()
    try:
        from coach.choreographer.prompts import GENRE_LABELS
        gname = GENRE_LABELS.get(gid, gid)
    except Exception:
        gname = gid
    if cur and gname:
        if summary:
            return (f"This one's a {gname} clip — {summary[:120]}. "
                    f"Want me to break it down?")
        return (f"Right now I'm running a {gname} clip ({cur}). "
                f"Want me to break it down step by step?")
    if gname:
        return (f"Nothing's looping yet — wanna try some {gname}? "
                f"I can start slow.")
    return ("Nothing's playing right now — pick a style and I'll start "
            "you off slow.")


# "Please try again in 1m38.496s." extracted from Groq 429 messages.
_RETRY_RE = re.compile(
    r'try again in [\d\.]+\s*(?:ms|s|m\d+\.?\d*s)', re.I)


def _salvage_text_toolcalls(text: str):
    """Return list of (name, args_dict) parsed from inline function tags."""
    out = []
    for m in _FN_TEXT_RE.finditer(text or ''):
        try:
            out.append((m.group(1), json.loads(m.group(2))))
        except Exception:
            continue
    return out


_BROWSER_KEYS = ('browser_event', 'browser_events')


def _strip_browser_keys(result: dict) -> dict:
    """Drop browser-only fields before echoing the tool result to the LLM."""
    return {k: v for k, v in result.items() if k not in _BROWSER_KEYS}


# ─── v33c: quick-intent fast-path ─────────────────────────────────────
# Maps a normalised user phrase / keyword to a genre code. Hitting any
# of these short-circuits the LLM round-trip entirely: we call
# pick_and_play(genre) directly and stream a canned one-liner so the
# avatar moves within ~300ms instead of after 5-8s of model latency.
import random as _random

_GENRE_KW = {
    # AIST genres. v23: added common typos + standalone-word aliases.
    'gBR': ('breaking', 'break dance', 'breakdance', 'breakdancing',
            'breakin', "breakin'", 'brakin', 'bboy', 'b-boy', 'b boy',
            'bgirl', 'b-girl', 'b girl'),
    'gHO': ('house', 'house dance', 'deep house', 'housey'),
    'gJB': ('ballet jazz', 'jazz ballet', 'classical jazz',
            'ballet', 'balle', 'balet'),
    'gJS': ('street jazz', 'jazz funk', 'jazz street', 'jazz',
            'jazzy', 'jaz'),
    'gKR': ('krump', 'krumping', 'krumpin', "krumpin'"),
    'gLH': ('la hip-hop', 'la hiphop', 'la-style', 'la style', 'hip hop',
            'hip-hop', 'hiphop', 'hip hops', 'hip-hops', 'hiphops',
            'hip hopping', 'hip hop dance', 'la hip hop'),
    'gLO': ('locking', 'lock', 'lockers', 'lockin', "lockin'",
            'lock dance'),
    'gMH': ('middle hip-hop', 'middle hop', 'mh', 'middle hiphop',
            'old school hip hop', 'old school hiphop'),
    'gPO': ('popping', 'pop', 'popper', 'robot', 'poppin', "poppin'",
            'pop dance'),
    'gWA': ('waacking', 'waack', 'whacking', 'whackin', 'wacking',
            'wackin', "wackin'", 'wackings', 'waacks'),
    # CMU pool — natural-English keywords for warmups/basics/walks.
    # Keep these broad: the deterministic title-match inside pick_clip
    # narrows them down. False positives here are fine; false negatives
    # silently dead-end the user (no avatar response, no chat reply).
    'cmu': ('basics', 'warmup', 'warmups', 'warm up', 'warm-up',
            'warming up', 'warm me up', 'warm me', 'loosen up',
            'limber up', 'mobility', 'mobilise', 'mobilize',
            'stretch', 'stretches', 'stretching',
            'reach', 'reaching', 'posture', 'balance', 'breathing',
            'walk', 'walks', 'walking', 'gait', 'stride',
            'jog', 'jogging', 'slow jog', 'casual walk', 'casual jog',
            'arm wave', 'arm waves', 'arm waving', 'wave', 'waves',
            'side step', 'side-to-side', 'side to side', 'sway',
            'head bob', 'head bobs', 'head bobbing',
            'hip swivel', 'hip sway', 'lunge', 'lunges',
            'technique', 'drill', 'drills', 'footwork drill',
            'footwork drills', 'casual', 'simple', 'basic',
            'plank', 'planks', 'push up', 'push ups', 'pushup',
            'pushups', 'sit up', 'sit ups', 'situp', 'situps',
            'crunch', 'crunches', 'bicycle', 'core', 'abs'),
}

# Phrases that mean "play SOMETHING, your pick".
_DANCE_GENERIC = {
    'dance for me', 'dance for me please', 'dance', 'dance please',
    'dance now', 'just dance', 'show me a move', 'show me something',
    'show me a dance', 'show me', 'demo', 'demo it', 'demo something',
    'play something', 'play a move', 'play a clip', 'go', 'go for it',
    'do it', 'do something',
}
_TEACH_GENERIC = {
    'teach me a move', 'teach me', 'teach me something', 'teach me a step',
    'teach me a routine', 'teach', 'break it down', 'show me how',
}


def _wants_teach(text: str) -> bool:
    """True when the user explicitly asked to LEARN (not just watch),
    so a styled ask like 'teach me house' triggers a real breakdown."""
    t = ' ' + (text or '').lower() + ' '
    return any(k in t for k in (
        'teach', 'break it down', 'break down', 'how to', 'how do i',
        'learn', 'step by step', 'show me how', 'walk me through'))

_CANNED_COUNT_INS = [
    "Cool — watch this. Three, two, one —",
    "Alright, here we go — five, six, seven, eight —",
    "Okay, lock in. And a-one, two —",
    "Watch me. Five, six, seven, EIGHT —",
    "Eyes up — here we go.",
    "Feel this — three, two, one —",
    "Got you — five, six, seven, eight —",
]
_CANNED_TEACH = [
    "Cool — I'll show it slow first. Watch the hips.",
    "Alright — same vibe, count it out with me. Five, six, seven, eight —",
    "Got you — slow demo, then we'll drill it. Eyes on the feet.",
    "Locking in — watch once, then we go together.",
]


def _detect_explicit_genre(text: str) -> Optional[str]:
    """If the user message mentions a known genre keyword, return its
    code. Longer/more specific phrases win."""
    if not text:
        return None
    t = ' ' + text.lower().strip() + ' '
    best = None  # (length_of_match, code)
    for code, kws in _GENRE_KW.items():
        for kw in kws:
            needle = ' ' + kw + ' '
            if needle in t:
                if best is None or len(kw) > best[0]:
                    best = (len(kw), code)
    if best:
        return best[1]
    # v-typo: TOKEN-PREFIX TOLERANCE. Whole-word matching above misses
    # doubled-letter / trailing-junk typos ("wackingg", "breakingg",
    # "poppin", "poppingg", "lockin"). For every distinct word token,
    # if a genre keyword of length >= 5 is a prefix of the token (or the
    # token is a prefix of the keyword, min 5 chars), treat it as a
    # match. Longest keyword still wins so "breaking" beats "break".
    # Restricting to >= 5 chars avoids short-keyword false hits
    # ("hop", "jazz", "krump" stay exact-only).
    tokens = re.findall(r"[a-z']+", text.lower())
    for tok in tokens:
        if len(tok) < 5:
            continue
        for code, kws in _GENRE_KW.items():
            for kw in kws:
                if len(kw) < 5:
                    continue
                if tok.startswith(kw) or kw.startswith(tok):
                    if best is None or len(kw) > best[0]:
                        best = (len(kw), code)
    return best[1] if best else None


def _default_genre_for_state(st: 'CoachState | None') -> str:
    """Pick a genre when the user said 'dance for me' without naming a
    style. Style is NOT tied to the character — any avatar can dance any
    style — so this always rolls a random dance genre instead of the
    avatar's 'home' style."""
    return _random.choice(
        ('gHO', 'gLH', 'gKR', 'gWA', 'gJS', 'gBR', 'gLO', 'gPO', 'gMH'))


def _try_fast_path(st: 'CoachState | None', user_msg: str):
    """Return (intent, genre) if this message can skip the LLM, or None.

    v34j: fast-path is intentionally RESTRICTED to the 4 chip-button
    labels ("Dance for me", "Teach me a move", etc.). Those are pure
    intent with zero content tokens — the user did not name a specific
    clip, so a random pick from the avatar's genre is the correct
    behaviour. EVERY OTHER user message — including short imperatives
    like "give me side-to-side arm warmup" or "start slow jog warmup" —
    goes to the LLM. The LLM calls `pick_and_play` with the actual
    query string so title-matching can find the clip the user asked for
    (or honestly admit no match instead of silently picking something
    unrelated).

    intent ∈ {'dance', 'teach'} -> call pick_and_play.
    """
    if not user_msg:
        return None
    norm = re.sub(r'[\.\!\?\,\;\:]+$', '', user_msg.strip().lower()).strip()
    if not norm:
        return None

    # Chip-button generic intents only. These are emitted by the four
    # quick-action buttons in the chat panel and never contain a
    # specific clip request.
    if norm in _DANCE_GENERIC:
        return ('dance', _default_genre_for_state(st), '')
    if norm in _TEACH_GENERIC:
        return ('teach', _default_genre_for_state(st), '')

    # v22: KEYWORD FAST-PATH. Short, unambiguous one-word/short-phrase
    # asks ("warmup", "stretch", "walk", "side to side", "jog",
    # "house", "breaking", "krump", etc.) used to fall through to the
    # LLM, which would sometimes just narrate cues without calling
    # pick_and_play (see v21 ALWAYS-SWITCH prompt rule — Groq models
    # ignore it ~30% of the time). Now: if the entire user message
    # (after stripping leading polite-fillers like "give me", "start",
    # "play", "do") is a registered _GENRE_KW keyword, bypass the LLM
    # and call pick_and_play directly with that keyword as the query.
    # Result: deterministic switch within ~300 ms, no model in the loop.
    stripped = re.sub(
        r'^(please\s+)?'
        r'(give\s+me\s+|gimme\s+|start\s+|play\s+|do\s+|show\s+me\s+|'
        r"let\'?s\s+do\s+|let\'?s\s+try\s+|let\'?s\s+go\s+with\s+|"
        r'lets\s+go\s+with\s+|i\s+want\s+|i\'?d\s+like\s+|'
        r'how\s+about\s+|can\s+(?:you|we)\s+(?:do\s+)?|do\s+a\s+|'
        r'try\s+|teach\s+me\s+|a\s+slow\s+|slow\s+|some\s+|a\s+)*'
        r'(.+?)'
        r'(\s+please|\s+now|\s+for\s+me)?$',
        r'\3', norm).strip()
    if stripped:
        for code, kws in _GENRE_KW.items():
            if stripped in kws:
                intent = 'teach' if (code == 'cmu' or _wants_teach(norm)) else 'dance'
                return (intent, code, stripped)

    # v23: SUBSTRING FALLBACK. If exact-equality didn't match, look
    # for ANY known genre keyword as a whole-word substring of the
    # message. Catches: "jazz?", "hip hop please", "some little
    # warmup", "lets do some wackingg", "how about a breakdance combo",
    # "give me side-to-side arm warmup", "warm me up with some house",
    # "lets go with some hip hop moves mixed with house jazz".
    # Longest match wins (so "hip hop" beats "hop", "breakdance" beats
    # "break"). Returns the full user msg as the query so pick_clip's
    # title-matching can refine within the genre pool.
    detected = _detect_explicit_genre(' ' + norm + ' ')
    if detected:
        intent = 'teach' if (detected == 'cmu' or _wants_teach(norm)) else 'dance'
        return (intent, detected, norm)

    return None


def _canned_response(intent: str, pick_result: dict) -> str:
    """Build a short canned bubble that names the picked clip."""
    title = pick_result.get('title') or pick_result.get('genre_name') or ''
    genre_name = pick_result.get('genre_name') or ''
    if intent == 'teach':
        line = _random.choice(_CANNED_TEACH)
    else:
        line = _random.choice(_CANNED_COUNT_INS)
    # Append the move name when it's not just the raw clip id.
    if title and not title.startswith(('g', 'cmu_')):
        return f"{line} {title}."
    if genre_name:
        return f"{line} A bit of {genre_name}."
    return line



def _collect_events(result: dict):
    """Return the list of avatar events from a tool result, preferring
    the plural `browser_events` list (e.g. from pick_and_play) over the
    singular `browser_event`.

    v108: EVERY event passes through motion_index.gate_event(), the
    single orientation chokepoint. No matter which path produced the
    clip (LLM pick, guided session, warmup/cooldown fallback, hardcoded
    pool, breakdown), an unverified / possibly-inverted clip_id is
    swapped for a guaranteed-upright fallback before it reaches the
    browser. This is the last line of defence so the user can NEVER see
    the avatar upside-down.
    """
    from coach import motion_index as _mi
    evts = result.get('browser_events')
    if evts:
        out = [_mi.gate_event(e) for e in evts if e]
    else:
        be = result.get('browser_event')
        out = [_mi.gate_event(be)] if be else []
    # Some breakdown events carry a nested 'stages' list; their clip_id
    # was already gated above, which is what the player loads.
    return [e for e in out if e]



# ─── v33d: streaming completion ──────────────────────────────────────
# Wraps an OpenAI/Groq chat completion call in stream=True mode and
# accumulates the deltas back into a non-streaming-shaped object so
# the existing tool-call logic below works unchanged.
#
# WHY: most chat-only replies (no tool call) take 1.5-3s end-to-end.
# With streaming we can paint the bubble incrementally as tokens
# arrive — typical TTFT is ~300ms. The first sentence boundary fires
# the TTS request, so audio starts ~700ms sooner than the buffered
# path.
#
# Behaviour:
#   - For content deltas: yields ('delta', text) UNTIL a tool_call
#     delta appears in the stream. After that, ('clear', None) is
#     yielded once and further content is buffered silently (we never
#     want to leak narration before a pick_and_play has had a chance
#     to start the avatar).
#   - At end of stream: yields ('done', synth_msg) where synth_msg
#     has .content (str) and .tool_calls (list | None) compatible
#     with the openai SDK's non-streaming response shape.

class _StreamFn:
    __slots__ = ('name', 'arguments')
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _StreamTC:
    __slots__ = ('id', 'type', 'function')
    def __init__(self, id_: str, fn: _StreamFn):
        self.id = id_
        self.type = 'function'
        self.function = fn

class _StreamMsg:
    __slots__ = ('content', 'tool_calls')
    def __init__(self, content: str, tool_calls):
        self.content = content
        self.tool_calls = tool_calls or None


async def _stream_chat(stream_resp, first_chunk_timeout: float = 22.0,
                       chunk_timeout: float = 30.0):
    """Async-generator that drains a streaming chat completion and
    yields ('delta', text) | ('clear', None) | ('done', _StreamMsg).
    Caller is responsible for invoking client.chat.completions.create
    with stream=True and passing the returned stream object here."""
    content_buf: List[str] = []
    tc_acc: Dict[int, Dict[str, Any]] = {}
    seen_tool = False
    deltas_emitted = False
    first = True
    aiter_ = stream_resp.__aiter__()
    while True:
        try:
            timeout = first_chunk_timeout if first else chunk_timeout
            chunk = await asyncio.wait_for(aiter_.__anext__(),
                                           timeout=timeout)
            first = False
        except StopAsyncIteration:
            break
        if not getattr(chunk, 'choices', None):
            continue
        delta = chunk.choices[0].delta
        # Tool-call accumulation. As soon as we see ANY tool_call delta
        # we retract any preview text already painted in the bubble so
        # the narration doesn't leak before the avatar moves.
        d_tcs = getattr(delta, 'tool_calls', None)
        if d_tcs:
            if not seen_tool and deltas_emitted:
                yield ('clear', None)
            seen_tool = True
            for tc in d_tcs:
                idx = getattr(tc, 'index', 0) or 0
                slot = tc_acc.setdefault(
                    idx, {'id': None, 'name': '', 'args': ''})
                if getattr(tc, 'id', None):
                    slot['id'] = tc.id
                fn = getattr(tc, 'function', None)
                if fn is not None:
                    if getattr(fn, 'name', None):
                        slot['name'] = fn.name
                    if getattr(fn, 'arguments', None):
                        slot['args'] += fn.arguments
        # Content stream — only emit deltas if no tool call seen yet.
        d_text = getattr(delta, 'content', None)
        if d_text:
            content_buf.append(d_text)
            if not seen_tool:
                deltas_emitted = True
                yield ('delta', d_text)
    # Build synthetic non-streaming-shaped message.
    if tc_acc:
        tool_calls = []
        for i, v in sorted(tc_acc.items()):
            call_id = v['id'] or f'call_stream_{i}'
            tool_calls.append(_StreamTC(
                id_=call_id,
                fn=_StreamFn(name=v['name'], arguments=v['args'] or '{}'),
            ))
    else:
        tool_calls = None
    yield ('done', _StreamMsg(''.join(content_buf), tool_calls))


async def run_turn(
    history: List[Dict[str, Any]],
    state: 'CoachState | None' = None,
    source: str = 'typed',
) -> AsyncIterator[Dict[str, Any]]:
    """One agent turn. `history` is mutated in place. `state` is the
    per-session CoachState; if not given, falls back to a module-level
    global (single-user mode). `source` is 'typed' (textbox / chip
    button) or 'voice' (STT) — only voice input is eligible for the
    STT noise gate."""
    st = state if state is not None else _state
    if _client is None and _aoai is None:
        yield {'type': 'assistant_text',
               'text': "I'm not wired up to an LLM yet — set "
                       "AZURE_OPENAI_API_KEY or GROQ_API_KEY and restart."}
        return

    # v33b: STOP-KEYWORD BYPASS. The user typing "stop", "stop dancing",
    # "pause", "cut", etc. must HALT the avatar, not trigger another
    # pick_and_play. The LLM was reading "stop dancing" as a topic
    # ("let's stop dancing this style and try another one") and queuing
    # a new clip. Route stop intents directly to the stop tool and
    # short-circuit the LLM round-trip.
    last_user = ''
    for m in reversed(history):
        if m.get('role') == 'user':
            last_user = (m.get('content') or '').strip().lower()
            break
    _STOP_INTENTS = {
        'stop', 'stop dancing', 'stop please', 'please stop',
        'pause', 'pause it', 'pause dancing', 'halt', 'cut',
        'cut it', "that's enough", 'thats enough', 'enough',
        'quit', 'end', 'shut up', 'be quiet', 'silence', 'no more',
        'stop the music', 'stop the dance', 'stop now',
    }
    # Strip trailing punctuation for matching.
    _user_norm = re.sub(r'[\.\!\?\,\;\:]+$', '', last_user).strip()
    if _user_norm in _STOP_INTENTS or _user_norm.startswith(('stop ', 'pause ')):
        result = execute_tool(st, 'stop', {})
        yield {'type': 'tool_call', 'name': 'stop', 'args': {},
               'result': _strip_browser_keys(result)}
        for be in _collect_events(result):
            yield {'type': 'avatar_event', 'event': be}
        msg = "Okay — stopped. Just say the word when you want to go again."
        history.append({'role': 'assistant', 'content': msg})
        yield {'type': 'assistant_text', 'text': _clip_response(msg)}
        return

    # v33f: STT-NOISE GATE. When music + TTS are playing the mic
    # frequently mis-recognises lyrics or breath as short fragments
    # like "They call the slab.", "On the same office.", "But I think.
    # But I think.". Sending those to the LLM as user turns wastes
    # tokens and produces nonsense replies. If the fragment has no
    # verb / question word / known genre / clear command, drop it.
    _GATE_KEYWORDS = (
        'dance', 'teach', 'show', 'play', 'stop', 'pause', 'slower',
        'faster', 'again', 'next', 'name', 'step', 'move', 'song',
        'music', 'beat', 'how', 'what', 'why', 'who', 'when', 'where',
        'help', 'try', 'tell', 'explain', 'mean', 'i ', " i'm", ' me',
        ' my', 'you ', 'your', 'we ', 'us ',
        'hi', 'hey', 'hello', 'yes', 'no', 'ok', 'okay', 'cool',
        'thanks', 'thank', 'love', 'like', 'want', 'need', 'ready',
        'house', 'hiphop', 'hip-hop', 'krump', 'lock', 'pop', 'jazz',
        'break', 'ballet', 'waack', 'middle', 'style', 'mirror',
    )
    _wc = len(_user_norm.split())
    # v34i: The keyword whitelist gate ("dance, teach, show, ...") was
    # silently dropping legitimate TYPED commands like "start slow jog
    # warmup" because no token matched the list. That whitelist is only
    # meaningful for VOICE input, where the mic may pick up TTS feedback
    # mid-utterance. Typed text (textbox / chip) is always intentional
    # and must never be silently gated. The duplicate-fragment check
    # (e.g. "but i think but i think") is content-based and runs for
    # both sources.
    _has_kw = any(k in (' ' + _user_norm + ' ') for k in _GATE_KEYWORDS)
    _is_dup = False
    if _wc >= 4:
        _half = _wc // 2
        _toks = _user_norm.split()
        if _toks[:_half] == _toks[_half:_half * 2]:
            _is_dup = True
    _voice_gated = (source == 'voice'
                    and 0 < _wc <= 7
                    and not _has_kw)
    if _voice_gated or _is_dup:
        # Silently drop the fragment — don't even emit a bubble. The
        # user didn't actually speak; the mic picked up the speakers.
        # Echoing "didn't catch that" every time would make the coach
        # feel paranoid. We just no-op this turn.
        return

    # v33c: QUICK-INTENT FAST-PATH. Chip buttons ("Dance for me",
    # "Teach me a move") and short direct asks ("show me house",
    # "krump", "popping please") have an unambiguous intent. Skip the
    # 2-round LLM dance entirely: call pick_and_play directly and emit
    # a canned count-in line. Cuts perceived latency from ~5-8s to
    # ~300ms because there is zero model round-trip.
    fast = _try_fast_path(st, last_user)
    if fast is not None:
        intent, genre, query = fast
        # v22: chip-button generic intents pass query='' (genre filter
        # picks a random clip). Keyword fast-path passes the actual
        # keyword ("warmup", "stretch", "walk", ...) so title-matching
        # inside pick_clip narrows the pool to the relevant clips.
        result = execute_tool(st, 'pick_and_play',
                              {'genre': genre,
                               'query': query,
                               'loop': True,
                               'prefer_cued': intent == 'teach',
                               'speed': 1.0 if intent == 'dance' else 0.85})
        if result.get('ok'):
            yield {'type': 'tool_call', 'name': 'pick_and_play',
                   'args': {'genre': genre, 'query': query},
                   'result': _strip_browser_keys(result)}
            for be in _collect_events(result):
                yield {'type': 'avatar_event', 'event': be}
            # v129: a TEACH ask must actually TEACH — break the clip into
            # its authored named steps (Heel Toe -> Bounce Down -> ...),
            # not just loop it. pick_and_play set state.current_clip.
            if intent == 'teach':
                bd = execute_tool(st, 'break_down', {'stage_seconds': 8})
                if bd.get('ok'):
                    yield {'type': 'tool_call', 'name': 'break_down',
                           'args': {}, 'result': _strip_browser_keys(bd)}
                    for be in _collect_events(bd):
                        yield {'type': 'avatar_event', 'event': be}
            text = _canned_response(intent, result)
            history.append({'role': 'assistant', 'content': text})
            yield {'type': 'assistant_text', 'text': _clip_response(text)}
            return
        # Pool genuinely empty for that genre (rare) — fall through
        # to the LLM, which has access to other tools / can suggest
        # a different style. No hardcoded keyword bubble here.

    messages = [{'role': 'system', 'content': system_prompt(st)}] + history

    # v33: hard cap of ONE pick_and_play per user turn. The LLM was
    # occasionally chaining pick_and_play in round 1, then ANOTHER
    # pick_and_play in round 2 (after the tool result came back) —
    # which queued a second clip on top of the first and the user
    # saw "two motions playing together". Track in a closure var
    # used by the tool dispatch below.
    pick_done_this_turn = {'flag': False}

    # ─── provider selection ──────────────────────────────────────────
    # Try Azure OpenAI first (faster + your VS Enterprise subscription
    # has plenty of TPM headroom). If it fails for any reason this
    # turn, fall through to the Groq multi-key fallback chain.
    aoai_dead = (_aoai is None)
    # Track which fallback model AND which API key we're on. Both reset
    # at the top of every turn; on a 429 we advance the model first,
    # then the key, before surrendering to the friendly quota modal.
    fb_idx = [-1]
    key_idx = [0]
    def _active_model():
        return GROQ_MODEL if fb_idx[0] < 0 else GROQ_FALLBACKS[fb_idx[0]]
    def _active_client():
        if not _clients: return _client
        return _clients[key_idx[0] % len(_clients)]

    # Up to 4 tool-call rounds per user turn.
    # v117 ANTI-STALL: track whether the coach said ANYTHING this turn
    # so we never end in dead silence (the "she just stops and I have to
    # pitch in" bug).
    spoke = False
    for _ in range(4):
        try:
            if not aoai_dead:
                stream = await asyncio.wait_for(
                    _aoai.chat.completions.create(
                        model=AOAI_DEPLOYMENT,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice='auto',
                        temperature=0.35,
                        max_tokens=400,
                        timeout=20,
                        stream=True,
                    ),
                    timeout=22,
                )
            else:
                stream = await asyncio.wait_for(
                    _active_client().chat.completions.create(
                        model=_active_model(),
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice='auto',
                        temperature=0.35,
                        max_tokens=400,
                        timeout=20,
                        stream=True,
                    ),
                    timeout=22,
                )
            # v33d: drain the stream, yielding delta events as content
            # arrives. The synthetic _StreamMsg this produces mirrors
            # the non-streaming response shape so the rest of the loop
            # below is unchanged.
            # v33e: hard cap on visible preview length. Once the live
            # bubble has grown past _STREAM_CAP chars we stop emitting
            # deltas (the model often keeps generating well past the
            # 2-sentence MVP rule). The final assistant_text emission
            # is still clipped via _clip_response so the bubble
            # finalises ≤ 220 chars even if preview hit 360.
            stream_msg = None
            _delta_chars = 0
            _STREAM_CAP = 360
            async for _kind, _payload in _stream_chat(stream):
                if _kind == 'delta':
                    if _delta_chars >= _STREAM_CAP:
                        continue
                    _delta_chars += len(_payload)
                    spoke = True
                    yield {'type': 'assistant_text_delta',
                           'text': _payload}
                elif _kind == 'clear':
                    _delta_chars = 0
                    yield {'type': 'assistant_text_clear'}
                elif _kind == 'done':
                    stream_msg = _payload
            # Build a minimal `resp`-shaped object so existing code
            # (resp.choices[0].message) still works.
            class _R:
                pass
            resp = _R()
            _C = _R()
            _C.message = stream_msg
            resp.choices = [_C]
        except Exception as e:                                  # noqa: BLE001
            # Azure OpenAI failed — mark it dead for the rest of this
            # turn and fall through to Groq on the next iteration of
            # the loop. We don't surface the error to the student;
            # they should never see provider plumbing.
            if not aoai_dead:
                aoai_dead = True
                if _client is not None:
                    continue                                    # retry with Groq
            # Groq returns 400 tool_use_failed when the model wrote a
            # <function=...> inline instead of using the tool-calls
            # field. Parse `failed_generation` and recover.
            body = getattr(e, 'body', None) or {}
            err  = (body.get('error') or {}) if isinstance(body, dict) else {}
            # --- Detect 429 / quota-exhausted and surface a friendly
            # event the browser can render as a modal instead of a raw
            # error line in chat. ---
            status = getattr(e, 'status_code', None)
            err_code = (err.get('code') or '').lower()
            err_type = (err.get('type') or '').lower()
            err_msg  = err.get('message') or str(e)
            is_quota = (
                status == 429
                or err_code == 'rate_limit_exceeded'
                or 'rate limit' in err_msg.lower()
                or 'tokens per day' in err_msg.lower()
            )
            if is_quota:
                # 1) advance through model fallbacks on the current key
                if fb_idx[0] + 1 < len(GROQ_FALLBACKS):
                    fb_idx[0] += 1
                    continue
                # 2) all models exhausted on this key — rotate to the
                # next API key and restart the model chain.
                if key_idx[0] + 1 < len(_clients):
                    key_idx[0] += 1
                    fb_idx[0] = -1
                    continue
                # 3) genuinely out of headroom across every key.
                retry_hint = ''
                m = _RETRY_RE.search(err_msg)
                if m:
                    retry_hint = m.group(0)
                yield {'type': 'llm_quota',
                       'scope': 'tokens_per_day' if 'tokens per day' in err_msg.lower() else 'rate',
                       'retry_hint': retry_hint,
                       'upgrade_url': os.getenv('UPGRADE_URL', '')}
                return
            failed = err.get('failed_generation') or ''
            if not failed:
                failed = str(e)
            salvaged = _salvage_text_toolcalls(failed)
            if not salvaged:
                # Generic, non-scary fallback for the chat bubble.
                yield {'type': 'assistant_text',
                       'text': "The AI is busy right now — give it "
                               "a few seconds and try again. (You can "
                               "still use the quick-action buttons.)"}
                return
            for name, args in salvaged:
                # v33: hard cap of ONE pick_and_play per turn.
                if name == 'pick_and_play' and pick_done_this_turn['flag']:
                    result = {'ok': False,
                              'reason': 'already picked a clip this turn; '
                                        'drill or chat instead'}
                else:
                    result = execute_tool(st, name, args)
                    if name == 'pick_and_play' and result.get('ok'):
                        pick_done_this_turn['flag'] = True
                yield {'type': 'tool_call', 'name': name, 'args': args,
                       'result': _strip_browser_keys(result)}
                for be in _collect_events(result):
                    yield {'type': 'avatar_event', 'event': be}
                # Echo as assistant+tool pair so next round has context.
                fake_id = f'call_salvage_{name}'
                messages.append({
                    'role': 'assistant', 'content': '',
                    'tool_calls': [{
                        'id': fake_id, 'type': 'function',
                        'function': {'name': name,
                                     'arguments': json.dumps(args)}}],
                })
                messages.append({
                    'role': 'tool', 'tool_call_id': fake_id, 'name': name,
                    'content': json.dumps(_strip_browser_keys(result)),
                })
            # Loop again so the model can narrate after the salvaged call.
            continue
        choice = resp.choices[0].message
        # If the model wrote inline <function=...> tags in content
        # (instead of using the tool_calls field), salvage them.
        inline = _salvage_text_toolcalls(choice.content or '')
        # Always strip function-tag debris before emitting visible text;
        # the salvage path may not catch every malformed variant.
        cleaned = _clean_speech(choice.content or '')
        # SPEECH ORDERING FIX:
        #   The LLM produces narration + tool_calls in the same response,
        #   but the browser was showing the bubble text BEFORE the
        #   avatar moved (because we yielded assistant_text first).
        #   That created the "voice talks about a move and only later
        #   the avatar starts doing it" feel.
        #
        #   Now we BUFFER the assistant speech and yield it AFTER the
        #   tool_calls + avatar_events have been sent, so the move
        #   starts first and the narration arrives just behind it
        #   — matching how a real coach calls counts.
        pending_speech = None
        if cleaned and not inline:
            pending_speech = cleaned
            history.append({'role': 'assistant', 'content': cleaned})
            messages.append({'role': 'assistant', 'content': cleaned})

        tool_calls = getattr(choice, 'tool_calls', None) or []
        if not tool_calls and inline:
            for name, args in inline:
                # v33: hard cap of ONE pick_and_play per turn.
                if name == 'pick_and_play' and pick_done_this_turn['flag']:
                    result = {'ok': False,
                              'reason': 'already picked a clip this turn; '
                                        'drill or chat instead'}
                else:
                    result = execute_tool(st, name, args)
                    if name == 'pick_and_play' and result.get('ok'):
                        pick_done_this_turn['flag'] = True
                yield {'type': 'tool_call', 'name': name, 'args': args,
                       'result': _strip_browser_keys(result)}
                for be in _collect_events(result):
                    yield {'type': 'avatar_event', 'event': be}
                fake_id = f'call_inline_{name}'
                messages.append({
                    'role': 'assistant', 'content': '',
                    'tool_calls': [{
                        'id': fake_id, 'type': 'function',
                        'function': {'name': name,
                                     'arguments': json.dumps(args)}}],
                })
                messages.append({
                    'role': 'tool', 'tool_call_id': fake_id, 'name': name,
                    'content': json.dumps(_strip_browser_keys(result)),
                })
            continue
        if not tool_calls:
            # No tool calls — flush any buffered speech now, then end.
            if pending_speech:
                _repaired = _repair_deflection(pending_speech, st, last_user)
                yield {'type': 'assistant_text',
                       'text': _clip_response(_repaired)}
                pending_speech = None
                spoke = True
            if not spoke:
                # v117 ANTI-STALL: never end the turn silent.
                yield {'type': 'assistant_text',
                       'text': "What do you want to work on next?"}
            return

        # Assistant message MUST be in the trace alongside its tool_calls.
        messages.append({
            'role': 'assistant',
            'content': choice.content or '',
            'tool_calls': [
                {'id': tc.id, 'type': 'function',
                 'function': {'name': tc.function.name,
                              'arguments': tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or '{}')
            except Exception:
                args = {}
            # v33: hard cap of ONE pick_and_play per turn.
            if name == 'pick_and_play' and pick_done_this_turn['flag']:
                result = {'ok': False,
                          'reason': 'already picked a clip this turn; '
                                    'drill or chat instead'}
            else:
                result = execute_tool(st, name, args)
                if name == 'pick_and_play' and result.get('ok'):
                    pick_done_this_turn['flag'] = True
            yield {'type': 'tool_call', 'name': name, 'args': args,
                   'result': _strip_browser_keys(result)}
            for be in _collect_events(result):
                yield {'type': 'avatar_event', 'event': be}
            messages.append({
                'role': 'tool',
                'tool_call_id': tc.id,
                'name': name,
                'content': json.dumps(_strip_browser_keys(result)),
            })
        # NOW flush the LLM's narration — after the avatar has been
        # told to move. The browser plays the move immediately on
        # receipt of avatar_event; the bubble + TTS arrives ~50-150ms
        # later which is what the user expects from a real coach.
        if pending_speech:
            _repaired = _repair_deflection(pending_speech, st, last_user)
            yield {'type': 'assistant_text',
                   'text': _clip_response(_repaired)}
            pending_speech = None
            spoke = True
    # Hit tool-call cap — if the whole turn was silent (only moves, no
    # narration), add a short keep-alive so the coach doesn't go quiet.
    if not spoke:
        yield {'type': 'assistant_text',
               'text': "Keep going — tell me how that felt or what's next."}
    return
