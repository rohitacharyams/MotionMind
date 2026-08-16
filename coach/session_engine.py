"""session_engine.py — coach-driven guided-session driver (restored).

BACKGROUND / WHY THIS FILE EXISTS
---------------------------------
The guided-session engine (start a session -> auto-advance phases ->
auto-pick & play clips -> LLM narration) used to live *inline* inside
``server.py``'s ``/ws/agent`` handler. During the v18x/v19x refactor of
``server.py`` that whole engine was accidentally stripped out, so the
front-door buttons ("pick a style -> pick a length") sent a
``session.start`` WS message that the server silently dropped — the
avatar never started, which is exactly the "I click the timer and the
style but nothing happens" bug.

This module re-introduces that engine as a small, self-contained class so
the (newer) server only needs a few lines to wire it in. The ticker /
narration logic is a faithful port of the last known-good version, with
two deliberate changes:

  * GUEST-FIRST: ``session.start`` no longer requires sign-in. Anonymous
    users can run a full guided session; streak/history is still recorded
    for signed-in users by the server's own disconnect handler.
  * DECOUPLED: no direct telemetry / dialogue-memory calls — those stay
    in the server. The engine only needs ``ws`` + ``state``.

Wiring (in server.py's ws_agent):

    from coach.session_engine import SessionEngine
    _sess_engine = SessionEngine(ws, state, cancel_nudges=_cancel_nudges)
    ...
    # inside the receive loop, before the `user_text` branch:
    if await _sess_engine.handle(msg):
        continue
    ...
    # on disconnect / error / finally:
    await _sess_engine.stop()
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, Optional

from coach import motion_index
from coach import session as coach_session
from coach.choreographer import agent as _agent
from coach.choreographer.tools import execute_tool, pick_session_clip

LOG = logging.getLogger('coach.session_engine')

# Within-phase clip rotation intervals (seconds). Session clips play with
# loop:True so they never fire `clip_done`; without rotation a whole phase
# shows ONE looping move ("the same warmup repeats forever"). Rotate to a
# fresh clip after an intent-dependent interval so the user sees variety.
_CLIP_ROTATE_SEC = {
    'warmup':         12.0,
    'cooldown':       15.0,
    'combo':          11.0,
    'freestyle':      13.0,
    'drill_one_move': 16.0,
}


class SessionEngine:
    """Drives one guided session for a single /ws/agent connection."""

    def __init__(self, ws, state, *, cancel_nudges=None):
        self.ws = ws
        self.state = state
        # Optional async callable the server passes so the coach's idle
        # nudges shut up the moment a session takes the floor.
        self._cancel_nudges = cancel_nudges
        self._task: Optional[asyncio.Task] = None
        self._need_next_clip = False

    # ── message dispatch ────────────────────────────────────────────
    async def handle(self, msg: Dict[str, Any]) -> bool:
        """Handle a session.* control message. Returns True if this
        engine consumed the message (caller should `continue`)."""
        mt = msg.get('type')
        if mt == 'session.start':
            await self._on_start(msg)
            return True
        if mt == 'session.skip':
            res = execute_tool(self.state, 'advance_phase', {})
            await self._send({'type': 'session.phase',
                              'ok': res.get('ok'),
                              'session': res.get('session'),
                              'finished': res.get('finished')})
            if res.get('ok') and not res.get('finished'):
                self._need_next_clip = True
            return True
        if mt == 'session.pause':
            res = execute_tool(self.state, 'pause_session', {})
            await self._send({'type': 'session.paused',
                              'ok': res.get('ok'),
                              'session': res.get('session')})
            return True
        if mt == 'session.resume':
            res = execute_tool(self.state, 'resume_session', {})
            await self._send({'type': 'session.resumed',
                              'ok': res.get('ok'),
                              'session': res.get('session')})
            return True
        if mt == 'session.end':
            res = execute_tool(self.state, 'end_session', {})
            await self._send({'type': 'session.finished',
                              'ok': res.get('ok'),
                              'session': res.get('session')})
            await self.stop()
            return True
        if mt == 'session.clip_done':
            # Browser tells us the current clip finished; ticker picks next.
            self._need_next_clip = True
            return True
        if mt == 'session.list':
            await self._send({'type': 'session.catalog',
                              'templates': coach_session.list_templates()})
            return True
        return False

    async def _on_start(self, msg: Dict[str, Any]) -> None:
        # GUEST-FIRST: no sign-in / onboarding gate. Anyone can dance.
        tpl = msg.get('template_id')
        res = execute_tool(self.state, 'start_session', {'template_id': tpl})
        await self._send({'type': 'session.started',
                          'ok': res.get('ok'),
                          'session': res.get('session'),
                          'reason': res.get('reason')})
        if not res.get('ok'):
            return
        # Session owns the floor now — silence idle nudges.
        if self._cancel_nudges is not None:
            try:
                await self._cancel_nudges()
            except Exception:
                pass
        self._need_next_clip = True
        self._ensure_ticker()
        # Kick off LLM narration-pool generation in the background. The
        # ticker uses whatever pool is ready when it next picks a clip;
        # if still pending it falls back to the scripted pool.
        sess = self.state.session
        if sess is not None and sess.template is not None:
            async def _gen_pool(_sess=sess):
                try:
                    sty_code = _sess.template.style or 'cmu'
                    sty_disp = coach_session._STYLE_NAMES.get(sty_code, sty_code)
                    pool = await coach_session.generate_session_narration(
                        _sess.template, sty_code, sty_disp)
                    if pool and self.state.session is _sess:
                        _sess.llm_narration_pool = pool
                except Exception:
                    pass
            asyncio.create_task(_gen_pool())

    # ── lifecycle ───────────────────────────────────────────────────
    def _ensure_ticker(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._ticker())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    # ── helpers ─────────────────────────────────────────────────────
    async def _send(self, obj: Dict[str, Any]) -> None:
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass

    @staticmethod
    def _fallback_mood_for_clip(clip_id: str) -> str:
        cid = str(clip_id or '')
        if cid.startswith('cmu_'):
            return 'relaxed'
        if cid.startswith(('gKR', 'gBR')):
            return 'focused'
        if cid.startswith('gPO'):
            return 'surprised'
        return 'happy'

    def _heartbeat_line(self, intent: str) -> str:
        bank = {
            'warmup': [
                'How does that feel through your shoulders?',
                'Keep it loose — we are waking the body up, not rushing it.',
            ],
            'drill_one_move': [
                'Want that again, or should I mirror it for you?',
                'Catch the rhythm first — the shape comes right after.',
            ],
            'combo': [
                'Nice — stay with me through the transition.',
                'If that counts clean for you, we can add more bite next.',
            ],
            'freestyle': [
                'You can steal just one detail from this and make it yours.',
                'What part are you feeling most right now — feet, groove, or arms?',
            ],
            'cooldown': [
                'Let the shoulders drop and breathe all the way out.',
                'Take your time here — this part should feel easy.',
            ],
        }
        _hbl = (getattr(self.state, 'coach_language', 'hinglish') or 'hinglish').lower()
        _hb_hi = {
            'warmup': ['Shoulders mein kaisa lag raha hai?',
                       'Loose rakho — body ko jagaa rahe hain, jaldi nahi.'],
            'drill_one_move': ['Phir se karein, ya main mirror kar doon?',
                               'Pehle rhythm pakdo — shape baad mein aa jayega.'],
            'combo': ['Badhiya — transition ke through mere saath raho.',
                      'Saaf lag raha hai to thodi aur bite add karein.'],
            'freestyle': ['Bas ek detail steal karke apna bana lo.',
                          'Abhi kya feel ho raha hai — feet, groove, ya arms?'],
            'cooldown': ['Shoulders drop karo aur poori saans chhodo.',
                         'Time lo — yeh part easy lagna chahiye.'],
        }
        if _hbl in ('hinglish', 'hindi'):
            lines = _hb_hi.get(intent) or _hb_hi['freestyle']
        else:
            lines = bank.get(intent) or bank['freestyle']
        return random.choice(lines)

    async def emit_session_line(self, *, kind, fallback, source,
                                intent='', phase_title='', clip_title='',
                                clip_cues='', mood=None, ask_question=False,
                                teaching=None) -> None:
        """Speak a LIVE, LLM-generated session line (warm, in-character,
        in the user's language). Falls back to the scripted `fallback`
        string if the LLM is slow/unavailable so the session never goes
        silent. Fire-and-forget from the ticker."""
        state = self.state
        hist = getattr(state, 'narration_history', None)
        if hist is None:
            hist = []
            try:
                state.narration_history = hist
            except Exception:
                pass
        # Make name frequency deterministic: feed the name to the LLM on
        # ~1 of every 4 lines, strip it on the rest so it never leaks.
        _u_full = (getattr(state, 'user_name', '') or '').strip()
        _ni = getattr(state, '_narr_idx', 0)
        try:
            state._narr_idx = _ni + 1
        except Exception:
            pass
        _allow_name = (_ni % 4 == 0)
        _uname = _u_full if _allow_name else ''
        line = None
        try:
            line = await _agent.llm_session_line(
                kind=kind, intent=intent, phase_title=phase_title,
                clip_title=clip_title, clip_cues=clip_cues,
                teaching=teaching,
                user_name=_uname,
                language=(getattr(state, 'coach_language', 'hinglish') or 'hinglish'),
                char_name=(getattr(state, 'character_display_name', '') or ''),
                char_style=(getattr(state, 'character_style', '') or ''),
                recent_lines=hist, ask_question=ask_question)
        except Exception:
            line = None
        # When LLM narration is slow AND language isn't English, use a
        # short localized generic line rather than English fallback.
        _lang_fb = (getattr(state, 'coach_language', 'hinglish') or 'hinglish').lower()
        if not line and _lang_fb in ('hinglish', 'hindi'):
            _FB = {
                'hinglish': {
                    'warmup': ["Chalo, gently shuru karte hain — saans lo, body ko jagao.",
                               "Easy warm-up — mere saath flow karo, koi jaldi nahi."],
                    'cooldown': ["Bas relax, dheere dheere saans chhodo. Bahut badhiya.",
                                 "Cool down — shoulders drop karo, breathe out."],
                    'drill_one_move': ["Yeh move dekho — pehle slow, phir tempo pe.",
                                       "Chalo isko pakadte hain — mere saath try karo."],
                    'combo': ["Ab dono moves saath mein — beat ke saath chalo.",
                              "Combo time — transition pe mere saath raho."],
                    'freestyle': ["Ab tumhari baari — jo feel ho wahi karo!",
                                  "Freestyle — bas move karo, main saath hoon."],
                },
                'hindi': {
                    'warmup': ["Chalo dheere se shuru karein, saans lo.",
                               "Aasaan warm-up, mere saath flow karo."],
                    'cooldown': ["Ab relax karo, dheere saans chhodo."],
                    'drill_one_move': ["Yeh move dekho, pehle slow phir tempo."],
                    'combo': ["Ab dono moves saath, beat ke saath."],
                    'freestyle': ["Ab tumhari baari, jo feel ho wahi karo!"],
                },
            }
            _bank = _FB.get(_lang_fb, _FB['hinglish'])
            _opts = _bank.get(intent) or _bank.get('warmup')
            if _opts:
                line = random.choice(_opts)
        text = (line or fallback or '').strip()
        if not text:
            return
        # On non-name lines, scrub the first name if the model echoed it.
        if not _allow_name and _u_full and line:
            import re as _re_nm
            _first = _u_full.split()[0]
            text = _re_nm.sub(r'(?i)(?:^|,\s*|\s+)' + _re_nm.escape(_first) + r'\b[,!.\s]*',
                              ' ', text).strip()
            text = _re_nm.sub(r'\s{2,}', ' ', text).strip(' ,').strip()
            if not text:
                text = (fallback or '').strip()
            if not text:
                return
        try:
            hist.append(text)
            if len(hist) > 40:
                del hist[:-40]
        except Exception:
            pass
        await self._send({'type': 'assistant_text', 'text': text,
                          'source': source, 'mood': mood})

    # ── the driver ──────────────────────────────────────────────────
    async def _ticker(self) -> None:
        """Every 250ms: auto-advance phases when time elapses, rotate /
        pick the next clip, and keep the coach narrating so the avatar
        moves without the user typing. Cancelled on disconnect."""
        state = self.state
        ws = self.ws
        try:
            last_phase_idx = -1
            last_clip_at = 0.0
            while True:
                await asyncio.sleep(0.25)
                sess = state.session
                if sess is None or sess.finished or sess.paused:
                    continue
                now = asyncio.get_event_loop().time()
                # Mid-clip heartbeat so a long clip isn't silent.
                if (last_clip_at > 0 and not getattr(sess, '_clip_heartbeat_sent', False)
                        and sess.current is not None
                        and (now - last_clip_at) >= 14.0):
                    setattr(sess, '_clip_heartbeat_sent', True)
                    asyncio.create_task(self.emit_session_line(
                        kind='heartbeat',
                        intent=sess.current.intent,
                        fallback=self._heartbeat_line(sess.current.intent),
                        source='session_heartbeat', ask_question=True))
                # Within-phase rotation to a fresh clip.
                if (sess.current is not None and last_clip_at > 0
                        and not self._need_next_clip
                        and sess.remaining_phase() > 1.0):
                    _rot = _CLIP_ROTATE_SEC.get(sess.current.intent, 0.0)
                    if _rot and (now - last_clip_at) >= _rot:
                        self._need_next_clip = True
                # Phase transition?
                if sess.remaining_phase() <= 0.0:
                    sess.advance()
                    snap = sess.snapshot()
                    if sess.finished:
                        await self._send({'type': 'session.finished',
                                          'session': snap})
                        state.session = None
                        continue
                    if hasattr(sess, '_drill_anchor') and sess.current and \
                            sess.current.intent != 'drill_one_move':
                        try:
                            delattr(sess, '_drill_anchor')
                        except AttributeError:
                            pass
                    await self._send({'type': 'session.phase', 'session': snap})
                    cur = sess.current
                    if cur:
                        asyncio.create_task(self.emit_session_line(
                            kind='phase',
                            intent=getattr(cur, 'intent', ''),
                            phase_title=getattr(cur, 'label', ''),
                            clip_cues=getattr(cur, 'cue', ''),
                            fallback=(cur.voiceover or ''),
                            source='session_voiceover'))
                    self._need_next_clip = True
                    last_phase_idx = sess.phase_idx
                    last_clip_at = now
                    setattr(sess, '_clip_heartbeat_sent', False)
                # First tick of a new phase, or clip ended -> pick next.
                if self._need_next_clip or last_phase_idx != sess.phase_idx:
                    last_phase_idx = sess.phase_idx
                    self._need_next_clip = False
                    clip = pick_session_clip(state)
                    if clip is not None:
                        # Orientation safety net: never render inverted.
                        try:
                            _sid = clip.get('id')
                            _safe = motion_index.safe_clip(_sid)
                            if _safe != _sid:
                                clip = dict(clip)
                                clip['id'] = _safe
                                LOG.warning('session clip %s not verified-'
                                            'upright -> %s', _sid, _safe)
                        except Exception:
                            pass
                        mood = None
                        _music = clip.get('music_url') or motion_index.music_url_for(clip['id'])
                        await self._send({
                            'type': 'avatar_event',
                            'event': {
                                'type': 'avatar.load',
                                'clip_id': clip['id'],
                                'duration_sec': clip.get('duration_sec'),
                                'music_url': _music,
                                'bpm': clip.get('bpm_target'),
                                'session_role': clip.get('role'),
                            },
                        })
                        await self._send({
                            'type': 'avatar_event',
                            'event': {
                                'type': 'avatar.play',
                                'clip_id': clip['id'],
                                'speed': clip.get('speed', 1.0),
                                'mirror': getattr(state, 'mirror', False),
                                'loop': True,
                                'music_url': _music,
                                'bpm': clip.get('bpm_target'),
                            },
                        })
                        sess.clip_count_in_phase += 1
                        n = sess.clip_count_in_phase
                        _fb_entry = coach_session.next_narration(
                            sess, clip, recent_extra=getattr(
                                state, 'narration_history', None))
                        _fb_line = ''
                        if _fb_entry:
                            if isinstance(_fb_entry, str):
                                _fb_line = _fb_entry
                            else:
                                _fb_line = _fb_entry.get('text') or ''
                                mood = _fb_entry.get('mood')
                        if n >= 2 or sess.phase_idx > 0:
                            asyncio.create_task(self.emit_session_line(
                                kind='clip',
                                intent=getattr(sess.current, 'intent', '') if sess.current else '',
                                clip_title=(clip.get('title') or ''),
                                clip_cues=(clip.get('key_cues') or
                                           (getattr(sess.current, 'cue', '') if sess.current else '')),
                                fallback=_fb_line,
                                teaching=(clip.get('teaching') or None),
                                source='session_narration', mood=mood,
                                ask_question=(n % 3 == 0)))
                        await self._send({
                            'type': 'avatar_event',
                            'event': {
                                'type': 'avatar.mood',
                                'mood': mood or self._fallback_mood_for_clip(clip['id']),
                            },
                        })
                        last_clip_at = now
                        setattr(sess, '_clip_heartbeat_sent', False)
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001
            await self._send({'type': 'error',
                              'message': f'session_ticker: {e!r}'})
