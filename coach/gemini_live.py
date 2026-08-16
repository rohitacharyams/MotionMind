"""Gemini Live (speech-to-speech) relay — Layer A1.

A thin server-side bridge between the browser mic/speaker and Google's
Gemini Live API (native audio dialog). The realtime model handles the
*talking* (hears tone, ~300 ms, barge-in, emotional prosody); our
existing tool layer (`choreographer/tools.py`) still drives the avatar —
we never hand choreography to a black box.

Wire protocol on /ws/voice (see server.py):

  browser → server
    • binary frames  = raw PCM16 mono @ 16 kHz mic audio
    • {type:'config', language, character:{...}}   (optional, first msg)
    • {type:'stop'}                                (end the session)

  server → browser
    • binary frames  = raw PCM16 mono @ 24 kHz coach audio (play it)
    • {type:'ready'}
    • {type:'transcript', role:'user'|'coach', text, final}
    • {type:'avatar_event', event:{...}}           (drives the VRM)
    • {type:'interrupted'}                          (model barge-in)
    • {type:'error', message}

Feature-flagged: if GEMINI_API_KEY is unset OR the google-genai SDK is
missing, `gemini_enabled()` is False and the browser falls back to the
existing Azure STT→Groq→Azure TTS pipeline.
"""
from __future__ import annotations

import os
import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional

# Input/output audio sample rates mandated by the Live API.
GEMINI_IN_RATE = 16000      # mic → model
GEMINI_OUT_RATE = 24000     # model → speaker

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')
# gemini-3.1-flash-live-preview (half-cascade live) is the validated
# model: streams audio AND reliably calls our avatar tools AND supports
# input/output transcription — all together. The native-audio dialog
# models sound warmer but 1011 on tool-calls, so they're not usable
# while the avatar needs to move. Overridable via env.
GEMINI_LIVE_MODEL = os.getenv('GEMINI_LIVE_MODEL', 'gemini-3.1-flash-live-preview')
# Prebuilt Gemini voice. Kore = warm/neutral; Puck/Charon/Aoede/Fenrir
# are the other options. Overridable per-deploy.
GEMINI_VOICE = os.getenv('GEMINI_LIVE_VOICE', 'Aoede')

# Lazy SDK import so a missing dep never breaks server import.
try:
    from google import genai                       # type: ignore
    from google.genai import types as genai_types   # type: ignore
    _SDK_OK = True
except Exception:                                    # noqa: BLE001
    genai = None              # type: ignore
    genai_types = None        # type: ignore
    _SDK_OK = False


def gemini_enabled() -> bool:
    """True only when both the SDK is importable AND a key is set."""
    return bool(_SDK_OK and GEMINI_API_KEY)


def gemini_status() -> Dict[str, Any]:
    return {
        'enabled': gemini_enabled(),
        'sdk': _SDK_OK,
        'has_key': bool(GEMINI_API_KEY),
        'model': GEMINI_LIVE_MODEL,
        'voice': GEMINI_VOICE,
    }


# ─── tool-schema conversion (OpenAI → Gemini) ──────────────────────────
# Gemini's FunctionDeclaration uses an OpenAPI-subset Schema. We keep
# only the fields it accepts and drop the rest so it never 400s on an
# unsupported keyword (e.g. it rejects $-prefixed / draft-7 fluff).
_SCHEMA_KEEP = ('type', 'description', 'enum', 'properties', 'required',
                'items', 'nullable')


def _clean_schema(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k not in _SCHEMA_KEEP:
            continue
        if k == 'properties' and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        elif k == 'items':
            out[k] = _clean_schema(v)
        else:
            out[k] = v
    # Gemini requires an explicit type on every object node.
    if 'properties' in out and 'type' not in out:
        out['type'] = 'object'
    return out


def gemini_tools() -> List[Any]:
    """Convert the existing OpenAI-style TOOL_SCHEMAS into a single
    Gemini Tool with all function declarations. Reuses the SAME tools
    the text agent uses so voice can drive the avatar identically."""
    from coach.choreographer.tools import TOOL_SCHEMAS
    decls: List[Dict[str, Any]] = []
    for t in TOOL_SCHEMAS:
        fn = t.get('function') or {}
        name = fn.get('name')
        if not name:
            continue
        params = _clean_schema(fn.get('parameters') or {'type': 'object',
                                                         'properties': {}})
        decls.append({
            'name': name,
            'description': (fn.get('description') or '')[:1024],
            'parameters': params,
        })
    return [{'function_declarations': decls}]


# ─── the relay ─────────────────────────────────────────────────────────
async def run_voice_session(ws, state, *, send_json, send_bytes,
                            recv) -> None:
    """Drive one /ws/voice session end-to-end.

    Args:
      ws        : the Starlette WebSocket (already accepted).
      state     : a CoachState (identity + character + language already set).
      send_json : async fn(dict) → send a JSON control msg to the browser.
      send_bytes: async fn(bytes) → send a binary audio frame to browser.
      recv      : async fn() → next browser message; returns
                  ('bytes', b'...') or ('text', '...') or ('disconnect', None).
    """
    if not gemini_enabled():
        await send_json({'type': 'error',
                         'message': 'gemini_live_unavailable'})
        return

    from coach.choreographer.prompts import system_prompt
    from coach.choreographer.tools import execute_tool
    from coach.choreographer.agent import _collect_events, _strip_browser_keys

    client = genai.Client(api_key=GEMINI_API_KEY)

    sys_text = system_prompt(state) + (
        "\n\n=== VOICE MODE (CONVERSATION FIRST) ===\n"
        "You are talking OUT LOUD to the person in real time — like a warm "
        "friend on a call. DEFAULT TO CONVERSATION: greet, listen, ask how "
        "they are, chat, joke, encourage. Keep replies SHORT and natural — "
        "one or two spoken sentences. DO NOT start dancing on your own and "
        "NEVER jump into counts ('five, six, seven, eight') unless they ASK. "
        "There are TWO different movement tools and it matters which you "
        "pick: use pick_and_play ONLY when they just want to WATCH / vibe / "
        "freestyle ('dance for me', 'show me something', 'just dance', "
        "'give me a move'). The MOMENT they want to LEARN ('teach me hip "
        "hop', 'I want to learn house', 'how do I do this', 'break it "
        "down', 'step by step', 'from the top') you MUST call break_down "
        "(NOT pick_and_play) \u2014 see STEP-BY-STEP TEACHING below. The rest of "
        "the time just be great company. Do not read lists or counts aloud."
        "\n\n=== NEVER DANCE WITHOUT THE TOOL (CRITICAL) ===\n"
        "The avatar ONLY moves when you call a movement tool (pick_and_play / "
        "break_down). So NEVER say you are doing a move, showing a step, or "
        "dancing UNLESS you have called the tool in the SAME turn. Do not "
        "announce a move before the tool result comes back. When the tool "
        "returns, it tells you the REAL clip now playing (its `title`, "
        "`summary`, `auto_cues`, `dominant_parts`). Describe THAT — the move "
        "that is actually on screen — not a move you imagined. If the person "
        "asked for something specific and the tool returned a different move, "
        "roll with what's playing ('here's a nice groove to start') rather "
        "than naming a move the avatar isn't doing."
        " ANSWER ANY general question normally — nutrition (e.g. protein in "
        "chicken), food, science, life, trivia — like a smart friend. NEVER "
        "say you can only talk about dance; that is wrong. After you answer, "
        "you may lightly anchor back to movement at a natural lull ('cool — "
        "wanna shake it out after?'), but never refuse to answer."
        "\n\n=== STEP-BY-STEP TEACHING (THE CORE OF LEARNING) ===\n"
        "When the person wants to LEARN anything — a style ('teach me hip "
        "hop', 'sikha do', 'I want to learn house') or a specific move "
        "('break it down', 'step by step', 'teach me slowly', 'I can't "
        "follow', 'what are the steps', 'from the top') — you TEACH, you do "
        "NOT just play a clip. Do it in this order:\n"
        "1) INTAKE (one short line): confirm what they want and set the "
        "plan, e.g. 'Love it — hip hop starts from the bounce, then we'll "
        "add a groove and a short combo. Ready?' Keep it to ONE sentence.\n"
        "2) CALL break_down. If a clip is already playing it breaks THAT "
        "down; if nothing is playing yet, pass the style as `genre` "
        "(e.g. genre='hiphop') and it will pick, load AND teach a clip for "
        "that style in one call. It drives the avatar through the move as "
        "real NUMBERED micro-steps — each slow and isolated — shows a "
        "clickable STEP RAIL on the side, and hands you back a `steps` "
        "list.\n"
        "3) Then teach OUT LOUD one step at a time, in time with the "
        "avatar: 'Step one — right arm up... step two — chest pop... step "
        "three — step back, freeze.' ONE step per breath, wait a beat "
        "between them so they can copy, and use the step names from the "
        "tool result. After the last step, ask if they want it again "
        "slower or at full speed before you speed up. If the `steps` list "
        "comes back empty, it's a steady groove with no distinct parts — "
        "say so and just ride it together, don't fake steps. NEVER answer a "
        "'teach me' ask by only calling pick_and_play — that just makes the "
        "avatar dance with no steps, which is exactly what we must avoid."
        "\n\n=== PANELS DO NOT MOVE THE AVATAR (CRITICAL) ===\n"
        "open_lessons and open_lesson ONLY open a menu / a page of text — they "
        "do NOT make the avatar move, and the student may close them without "
        "clicking anything. So NEVER call open_lessons for 'teach me', 'show "
        "me a move', 'break it down', or 'how do I do X' — for ALL of those, "
        "call break_down (it moves the avatar for real). Only use open_lessons "
        "if they literally ask to SEE the lesson menu/list. After opening ANY "
        "panel, do NOT say 'watch me', 'here I go', or 'I'm doing the move' — "
        "nothing is moving until you call break_down / pick_and_play / drill. "
        "If a panel is in the way, call close_panel to dismiss it so they can "
        "watch the avatar full-screen. You fully control the screen: you can "
        "open the steps, close the steps, close the chat, and start a move — "
        "use these tools so what you SAY always matches what they SEE."
        "\n\n=== SAY-MATCHES-SHOW (NEVER LIE ABOUT MOTION) ===\n"
        "Only claim the avatar is dancing / showing a step / doing a move when "
        "you have called a MOVEMENT tool (break_down, pick_and_play, or drill) "
        "in THIS SAME turn and it returned ok. If you did not call one, the "
        "avatar is STILL — so do not narrate motion; instead call the tool "
        "first, THEN describe what it's now doing."
        "\n\n=== YOU CAN SEE THEM ===\n"
        "When camera frames arrive you can SEE the dancer live. Use it like "
        "a real coach watching: react to what they're actually doing — their "
        "energy, posture, whether they're moving, tired, or nailing it. Give "
        "specific, warm, in-the-moment feedback (e.g. 'love that bounce, drop "
        "your shoulders a touch'). NEVER describe their appearance, clothing, "
        "room, or anything not about the dancing. If you can't see them "
        "clearly, just coach by ear — never mention the camera or that you're "
        "analysing video.")

    config: Dict[str, Any] = {
        'response_modalities': ['AUDIO'],
        'system_instruction': sys_text,
        'tools': gemini_tools(),
        'speech_config': {
            'voice_config': {
                'prebuilt_voice_config': {'voice_name': GEMINI_VOICE}
            }
        },
        'input_audio_transcription': {},
        'output_audio_transcription': {},
    }

    loop_done = asyncio.Event()

    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL,
                                            config=config) as session:
            await send_json({'type': 'ready', 'model': GEMINI_LIVE_MODEL,
                             'voice': GEMINI_VOICE})

            # v77: PROACTIVE GREETING. Don't wait for the user to speak
            # first — a real companion says hi. Trigger Gemini to greet
            # warmly, by first name when we know it, in the user's
            # language, then ask what's up.
            try:
                _nm = (getattr(state, 'user_name', None) or '').strip()
                _first = _nm.split(' ')[0] if _nm else ''
                # v88: weave in durable memory so voice remembers across days.
                _memhint = ''
                _mm = getattr(state, 'dialogue_memory', None)
                if isinstance(_mm, dict):
                    _bits = []
                    if _mm.get('summary'):
                        _bits.append(str(_mm['summary']))
                    if _mm.get('goals'):
                        _bits.append('goals: ' + ', '.join(_mm['goals'][:3]))
                    if _bits:
                        _memhint = (" You remember this person from before: "
                                    + ' | '.join(_bits)
                                    + ". Reference ONE relevant detail warmly, "
                                    "do not recite it all.")
                _g = (
                    "[SYSTEM: The user just opened the app and can hear you "
                    "now. Greet them warmly and BRIEFLY like a friend"
                    + (f", by their first name ({_first})" if _first else "")
                    + ". Ask what's going on / how they're doing. ONE short "
                    "spoken sentence. Match their chosen language "
                    "(e.g. casual Hinglish: 'Arre"
                    + (f" {_first}" if _first else "")
                    + ", kya chal raha hai?')." + _memhint
                    + " Just say hi and ask \u2014 do NOT start dancing or counting."
                    + " Do NOT mention being an AI.]")
                await session.send_client_content(
                    turns={'role': 'user', 'parts': [{'text': _g}]},
                    turn_complete=True)
            except Exception:                       # noqa: BLE001
                pass

            # ── browser → Gemini (mic audio + control) ────────────────
            async def pump_in() -> None:
                try:
                    while not loop_done.is_set():
                        kind, payload = await recv()
                        if kind == 'disconnect':
                            break
                        if kind == 'bytes' and payload:
                            await session.send_realtime_input(
                                audio=genai_types.Blob(
                                    data=payload,
                                    mime_type=f'audio/pcm;rate={GEMINI_IN_RATE}'))
                        elif kind == 'text':
                            try:
                                m = json.loads(payload)
                            except Exception:       # noqa: BLE001
                                continue
                            if m.get('type') == 'stop':
                                break
                            if m.get('type') == 'text' and m.get('text'):
                                # Let the user also type in voice mode.
                                await session.send_client_content(
                                    turns={'role': 'user',
                                           'parts': [{'text': m['text']}]},
                                    turn_complete=True)
                            elif m.get('type') == 'say' and m.get('text'):
                                # v198: SESSION NARRATION relay. The guided
                                # session (Azure-free now) sends its scripted
                                # coaching lines here so the ONE live voice
                                # speaks them. Voice it essentially verbatim,
                                # warmly, in the user's language, and NEVER
                                # call a movement tool (the session engine
                                # already drives the avatar) or add chatter.
                                _line = str(m['text']).strip()
                                if _line:
                                    await session.send_client_content(
                                        turns={'role': 'user', 'parts': [{'text': (
                                            "[NARRATION \u2014 read the coaching line "
                                            "below OUT LOUD, right now, word for "
                                            "word. Speak ONLY these words, keeping "
                                            "the exact wording and language "
                                            "(Hinglish/Hindi/English) as written. "
                                            "Do NOT translate, do NOT rephrase, do "
                                            "NOT summarise, do NOT add or drop "
                                            "anything, do NOT ask a question, do "
                                            "NOT comment, and do NOT call any tool. "
                                            "Just say it, warmly and in rhythm:\n"
                                            + _line)}]},
                                        turn_complete=True)
                            elif m.get('type') == 'image' and m.get('data'):
                                # v89: live camera frame -> Gemini vision.
                                try:
                                    import base64 as _b64
                                    _raw = _b64.b64decode(m['data'])
                                    _mt = m.get('mime') or 'image/jpeg'
                                    await session.send_realtime_input(
                                        media=genai_types.Blob(
                                            data=_raw, mime_type=_mt))
                                except Exception:       # noqa: BLE001
                                    pass
                finally:
                    loop_done.set()

            # ── Gemini → browser (audio + tools + transcripts) ────────
            async def pump_out() -> None:
                try:
                    while not loop_done.is_set():
                        turn = session.receive()
                        async for resp in turn:
                            await _handle_response(resp, session, state,
                                                   send_json, send_bytes,
                                                   execute_tool,
                                                   _collect_events,
                                                   _strip_browser_keys)
                except Exception as e:              # noqa: BLE001
                    if not loop_done.is_set():
                        await send_json({'type': 'error',
                                         'message': f'gemini_recv: {e!r}'})
                finally:
                    loop_done.set()

            await asyncio.gather(pump_in(), pump_out(),
                                 return_exceptions=True)
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        try:
            await send_json({'type': 'error',
                             'message': f'gemini_connect: {e!r}'})
        except Exception:                            # noqa: BLE001
            pass


async def _handle_response(resp, session, state, send_json, send_bytes,
                           execute_tool, _collect_events,
                           _strip_browser_keys) -> None:
    """Process one Gemini Live response chunk."""
    # 1) Audio out → browser speaker.
    data = getattr(resp, 'data', None)
    if data:
        await send_bytes(data)

    sc = getattr(resp, 'server_content', None)
    if sc is not None:
        # Barge-in: the model was interrupted by the user's voice.
        if getattr(sc, 'interrupted', None):
            await send_json({'type': 'interrupted'})
        # Live transcripts (captions).
        it = getattr(sc, 'input_transcription', None)
        if it is not None and getattr(it, 'text', None):
            await send_json({'type': 'transcript', 'role': 'user',
                             'text': it.text, 'final': False})
        ot = getattr(sc, 'output_transcription', None)
        if ot is not None and getattr(ot, 'text', None):
            await send_json({'type': 'transcript', 'role': 'coach',
                             'text': ot.text, 'final': False})

    # 2) Tool calls → run them on OUR engine, return results to Gemini,
    #    and forward avatar events to the browser so the VRM moves.
    tc = getattr(resp, 'tool_call', None)
    if tc is not None and getattr(tc, 'function_calls', None):
        responses = []
        # Movement-intent tools: when the coach's VOICE commits to a move
        # ("here's a spin!"), the avatar MUST actually move. If the picker
        # dead-ends (query didn't match the genre → ok:False, no event) we
        # were sending nothing, so she talked over a frozen avatar. For
        # these tools we GUARANTEE a real, upright clip plays and we hand
        # the ACTUAL clip's metadata back to Gemini so she narrates what is
        # truly on screen instead of the move she imagined.
        _MOVE_TOOLS = {'pick_and_play', 'pick_clip', 'play', 'break_down'}
        for fc in tc.function_calls:
            name = fc.name
            args = dict(fc.args or {})
            try:
                result = execute_tool(state, name, args)
            except Exception as e:                  # noqa: BLE001
                result = {'ok': False, 'reason': repr(e)}

            events = _collect_events(result)
            # Guaranteed-movement fallback for voice mode.
            if name in _MOVE_TOOLS and (not result.get('ok') or not events):
                fb = None
                try:
                    # Retry WITHOUT the unmatched query (keep genre if any),
                    # which falls through to a fresh pick from the pool.
                    fb = execute_tool(state, 'pick_and_play',
                                      {'genre': args.get('genre')})
                    if not fb.get('ok'):
                        # Last resort: no genre constraint at all.
                        fb = execute_tool(state, 'pick_and_play', {})
                except Exception as e:              # noqa: BLE001
                    fb = {'ok': False, 'reason': repr(e)}
                fb_events = _collect_events(fb) if fb else []
                if fb_events:
                    result, events = fb, fb_events

            # Drive the avatar.
            for be in events:
                await send_json({'type': 'avatar_event', 'event': be})
            # Remember played clips for variety (mirrors agent.py).
            try:
                if name in ('pick_and_play', 'pick_clip') and result.get('ok'):
                    cid = result.get('clip_id') or result.get('id')
                    if cid:
                        state.remember_clip(cid, args.get('genre', '?'),
                                            result.get('title', ''))
            except Exception:                        # noqa: BLE001
                pass
            responses.append(genai_types.FunctionResponse(
                id=getattr(fc, 'id', None),
                name=name,
                response=_strip_browser_keys(result)))
        try:
            await session.send_tool_response(function_responses=responses)
        except Exception:                            # noqa: BLE001
            pass
