"""session.py — coach-driven session phase machine (v27).

A Session walks the user through a structured workout: checkin →
warmup → drill → routine → freestyle → cooldown → summary. The
coach (not the user) drives the clock and the clip picks; the user
can chat / skip / pause but the avatar keeps moving.

This module owns the DATA MODEL only. The server's WebSocket loop is
the one that ticks the clock and emits browser events. The tools
layer reads Session.state to bias `pick_clip` toward the current
phase's intent.

Templates live in TEMPLATES below. A template is a list of phases
with target durations (seconds), an `intent` string used to bias
clip picking, and an optional `style` filter (genre code or 'cmu').
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Phase:
    name: str                    # 'warmup', 'drill', ...
    label: str                   # human-readable
    duration_sec: float          # target wall time in this phase
    intent: str                  # picker hint: 'warmup', 'drill_one_move', 'combo', 'freestyle', 'cooldown', 'rest'
    style: Optional[str] = None  # genre filter; None = use session.style
    cue: str = ''                # one-line coach prompt for the LLM
    voiceover: str = ''          # baked phrase the coach says when phase starts
    # v59: explicit clip pool override. When non-empty, the picker
    # MUST choose from these IDs (filtered for retargeted+safe).
    # Used by the Stretch & Warmup routine to hit specific body
    # parts in order (upper → mid → lower → dynamic → breath) instead
    # of relying on random whitelist draws.
    clip_pool: List[str] = field(default_factory=list)


@dataclass
class SessionTemplate:
    id: str
    title: str
    style: str                   # genre code, e.g. 'gHO'
    total_sec: float
    phases: List[Phase]
    description: str = ''


# ── Built-in templates ───────────────────────────────────────────
def _quick_start(style: str, style_name: str,
                 duration_min: int = 5) -> SessionTemplate:
    """A flexible quick session — 5, 10 or 20 minutes. Same arc,
    longer drills + freestyle as duration grows."""
    if duration_min == 10:
        # 10-min: longer drill + add freestyle
        phases = [
            Phase('checkin', 'Check-in', 10, 'rest', 'cmu',
                  cue='Greet the user warmly. Ask how they feel and what they want from these 10 minutes.',
                  voiceover="Hey, glad you're here. Let's get into it — 10 minutes, focused."),
            Phase('warmup', 'Warm up', 120, 'warmup', 'cmu',
                  cue='Gentle warm-up groove. Cue breathing. Mention you can chat any time.',
                  voiceover="Easy warm-up — match my breath and let the joints wake up.",
                  clip_pool=_SW_UPPER),
            Phase('drill', f'{style_name} drill', 240,
                  'drill_one_move', style,
                  cue=f'Teach ONE signature {style_name} move. Name it, give its origin in one line, then slow → tempo.',
                  voiceover=f"Now the move. I'll name it, show it slow, then take it up."),
            Phase('routine', 'Short combo', 150, 'combo', style,
                  cue='Chain 2–3 clips into a combo. Call counts. Ask if they want to repeat.',
                  voiceover="Let's chain pieces together — find the beat."),
            Phase('freestyle', 'Your turn', 60, 'freestyle', style,
                  cue='Hand the floor over. Encourage. Suggest one tweak (level, arms, half-time).',
                  voiceover="Your turn. Just move — I'll keep dancing with you."),
            Phase('cooldown', 'Cool down', 50, 'cooldown', 'cmu',
                  cue='Soft sway. Praise effort. Bring breath down.',
                  voiceover="Nice work. Soft sway, breathe it out.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Wrap', 20, 'rest', None,
                  cue='Recap: minutes moved, move name learned, one thing to remember. Encourage return.',
                  voiceover="That's a session. Come back tomorrow."),
        ]
        total = 650
    elif duration_min == 20:
        phases = [
            Phase('checkin', 'Check-in', 10, 'rest', 'cmu',
                  cue='Greet warmly. Ask their energy level and what they want from 20 minutes.',
                  voiceover="20 minutes — perfect. Tell me how you're feeling."),
            Phase('warmup', 'Warm up', 180, 'warmup', 'cmu',
                  cue='Full gentle warm-up groove. Breathing, ease the joints awake.',
                  voiceover="Twenty-minute session means a real warm-up. Stay with me.",
                  clip_pool=_SW_UPPER),
            Phase('drill', f'{style_name} drill — slow', 300,
                  'drill_one_move', style,
                  cue=f'Teach ONE signature {style_name} move slowly. Name it, origin, mechanics. Repeat 4–6 times.',
                  voiceover=f"This is our move for today. Let me tell you where it comes from."),
            Phase('drill', f'{style_name} drill — tempo', 240,
                  'drill_one_move', style,
                  cue='Same move, full tempo. Encourage. Offer ONE common mistake to avoid.',
                  voiceover="Same move, full speed now. Keep it clean."),
            Phase('routine', 'Combo', 240, 'combo', style,
                  cue='Build a 3-clip combo. Call counts. Repeat twice.',
                  voiceover="Now we layer. Three pieces — let's chain them."),
            Phase('freestyle', 'Freestyle', 180, 'freestyle', style,
                  cue='Hand it over. Suggest dynamics changes (level, arms, half-time, double-time).',
                  voiceover="Five minutes of just you. Make it yours."),
            Phase('cooldown', 'Cool down', 100, 'cooldown', 'cmu',
                  cue='Slow soft sway. Praise the work. Bring HR down.',
                  voiceover="Beautiful work. Cool it down with me.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Wrap', 30, 'rest', None,
                  cue='Recap moves taught, what to practise tomorrow, energy match.',
                  voiceover="That was a full session. See you tomorrow."),
        ]
        total = 1280
    else:
        # 5-min default
        phases = [
            Phase('checkin', 'Check-in', 10, 'rest', 'cmu',
                  cue='Greet warmly. Ask how they feel today in one short sentence.',
                  voiceover="Hey, glad you're here. Let's move."),
            Phase('warmup', 'Warm up', 60, 'warmup', 'cmu',
                  cue='Gentle sway warm-up. Cue breathing.',
                  voiceover="Easy warm-up — follow my breath.",
                  clip_pool=_SW_UPPER),
            Phase('drill', f'{style_name} drill', 100,
                  'drill_one_move', style,
                  cue=f'Teach ONE signature {style_name} move. Name it, one-line origin, then slow → tempo.',
                  voiceover=f"Now the move. Watch first, then join in."),
            Phase('routine', 'Short combo', 80, 'combo', style,
                  cue='Chain 2 different clips into a combo. Call counts.',
                  voiceover="Two moves together — find the beat."),
            Phase('cooldown', 'Cool down', 35, 'cooldown', 'cmu',
                  cue='Soft sway. Bring breath down.',
                  voiceover="Nice work. Breathe it out.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Session summary', 10, 'rest', None,
                  cue='Recap minutes moved + move name. Encourage return.',
                  voiceover="That's it — see you tomorrow."),
        ]
        total = 300
    return SessionTemplate(
        id=f'quick{duration_min}_{style}',
        title=f'Quick {style_name} — {duration_min} min',
        style=style,
        total_sec=float(total),
        description=(f'{duration_min} minutes with your coach. '
                     f'Warm up, learn a {style_name} move, combo, cool down.'),
        phases=phases,
    )


def _house_day1() -> SessionTemplate:
    """Day 1 of the House Foundations 14-day program."""
    return SessionTemplate(
        id='house_foundations_d1',
        title='House Foundations — Day 1: The Bounce',
        style='gHO',
        total_sec=600,
        description='Your first House lesson. Learn the foundational '
                    'bounce — it sits under every other House move.',
        phases=[
            Phase('checkin', 'Welcome', 20, 'rest', 'cmu',
                  cue='Introduce yourself as Kira. Explain Day 1 is The Bounce.',
                  voiceover="I'm Kira. Welcome to Day 1 of House."),
            Phase('warmup', 'Warm up', 90, 'warmup', 'cmu',
                  cue='Ankles, knees, hips — House lives in the legs.',
                  voiceover="Loosen the ankles, knees, hips."),
            Phase('drill', 'The Bounce — slow', 180,
                  'drill_one_move', 'gHO',
                  cue='Teach the bounce. Down on 1, up on 2. Half tempo.',
                  voiceover="The bounce. Down on one, up on two."),
            Phase('drill', 'The Bounce — to tempo', 120,
                  'drill_one_move', 'gHO',
                  cue='Same bounce, now full tempo. Breath stays even.',
                  voiceover="Same thing, full tempo. Keep breathing."),
            Phase('freestyle', 'Try it free', 90, 'freestyle', 'gHO',
                  cue='Let the music play. User bounces. Encourage.',
                  voiceover="Your turn. Just bounce."),
            Phase('cooldown', 'Cool down', 80, 'cooldown', 'cmu',
                  cue='Slow sway. Praise effort.',
                  voiceover="Beautiful. Soft sway, breathe it down."),
            Phase('summary', 'Wrap', 20, 'rest', None,
                  cue='Day 1 done. Tomorrow: the Jack.',
                  voiceover="Day 1 done. Tomorrow we learn the Jack."),
        ],
    )


# v59: hand-curated CMU clip pools for the Stretch & Warmup routine.
# Each pool targets a body region / energy level.
#
# v68 RE-AUDIT (2026-06-22): the previous "tilt-verified" claim was WRONG
# — it measured the normalized skeleton WITHOUT calling vrm.update(dt),
# so it under-reported the real on-screen tilt. Re-measured through the
# exact render path (player.update + vrm.update each frame) revealed
# several "warmup" clips that actually fold/cartwheel 30-71 deg forward —
# e.g. cmu_05_05_04 is a full aerial kick (leg straight up, head down),
# NOT a breath. Those are the "she's upside down" clips the user saw.
# REMOVED all clips whose max head-vs-vertical tilt exceeded ~25 deg:
#   cmu_02_02_05 (50), cmu_02_02_08 (31), cmu_105_105_12 (64),
#   cmu_02_02_09 (41), cmu_05_05_04 (71), cmu_106_106_18 (28).
# Every clip kept below is <= ~22 deg through the real render loop and
# redistributed so no pool is starved (the session warmup unions them).
# v80: the cmu warmup library is broken (155/193 clips render the
# avatar INVERTED/folded). Switched warmup pools to verified-clean,
# upright, mellow dance clips (locking / house / jazz / middle hip-hop)
# played as a gentle 'warm up with me' groove. The one genuinely-clean
# cmu clip (cmu_111_111_07) is kept for breath/cooldown.
# v147: REAL warm-up / mobility library — Mixamo stretch/squat/cardio clips
# (retargeted, verified upright, purpose-built for warm-ups — unlike the old
# CMU clips which rendered inverted, and the AIST dance grooves which were a
# stopgap). Grouped by body region so the guided routine flows
# upper -> mid -> lower -> dynamic -> breath.
_SW_UPPER = ['mixamo_neck_stretching', 'mixamo_shrugging',
             'mixamo_shoulder_rubbing', 'mixamo_one_shoulder_lean',
             'mixamo_arm_stretching']
_SW_MID = ['mixamo_reaching_out', 'mixamo_reaching_down', 'mixamo_leaning']
_SW_LOWER = ['mixamo_air_squat', 'mixamo_back_squat', 'mixamo_overhead_squat',
             'mixamo_crouch_to_stand']
_SW_DYNAMIC = ['mixamo_warming_up', 'mixamo_jumping_jacks', 'mixamo_slow_jog',
               'mixamo_jog_in_circle']
_SW_BREATH = ['mixamo_neck_stretching', 'mixamo_one_shoulder_lean',
              'mixamo_reaching_out', 'mixamo_shrugging']


# v108: HARD GUARANTEE — strip any clip that is not orientation-verified
# from every curated pool at import time, so a hand-edit can never
# reintroduce an inverted clip into the warmup/cooldown rotation. The
# verified set is the single source of truth (motion_index).
def _filter_verified(pool):
    try:
        from coach import motion_index as _mi
        kept = [c for c in pool if _mi.is_verified_upright(c)]
        return kept or pool        # never return empty (keep a groove)
    except Exception:
        return pool


_SW_UPPER = _filter_verified(_SW_UPPER)
_SW_MID = _filter_verified(_SW_MID)
_SW_LOWER = _filter_verified(_SW_LOWER)
_SW_DYNAMIC = _filter_verified(_SW_DYNAMIC)
_SW_BREATH = _filter_verified(_SW_BREATH)


def _stretch_warmup(duration_min: int = 10) -> SessionTemplate:
    """Pure stretch + warmup routine — no dance style required. Hits
    upper -> mid -> lower -> dynamic -> breath in a fixed arc. Curated
    CMU clip pools per phase so we know exactly what plays."""
    if duration_min == 5:
        phases = [
            Phase('checkin', 'Quick check-in', 12, 'rest', 'cmu',
                  cue='Greet warmly. Ask in ONE sentence what feels tight today.',
                  voiceover="Hey - five-minute reset. Where do you feel tight?"),
            Phase('warmup', 'Wake the arms', 60, 'warmup', 'cmu',
                  cue='Arm waves & reaches. Shoulders soft, breath long.',
                  voiceover="Wake up the shoulders. Big easy reaches.",
                  clip_pool=_SW_UPPER),
            Phase('warmup', 'Open the hips', 60, 'warmup', 'cmu',
                  cue='Hip swivels & side bends. Knees soft, ribs lifted.',
                  voiceover="Hips next. Let them swing.",
                  clip_pool=_SW_MID),
            Phase('warmup', 'Light feet', 70, 'warmup', 'cmu',
                  cue='Light jog / step-touch. Land soft, breath even.',
                  voiceover="Get the feet moving. Light and easy.",
                  clip_pool=_SW_LOWER),
            Phase('cooldown', 'Breath out', 50, 'cooldown', 'cmu',
                  cue='Slow waves, gentle sway. Long exhale, shoulders drop.',
                  voiceover="Cool it down. Long exhale.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Wrap', 8, 'rest', None,
                  cue='Praise. Suggest the 10-min version next time.',
                  voiceover="Done. Body says thanks."),
        ]
        total = 260
    elif duration_min == 15:
        phases = [
            Phase('checkin', 'Quick check-in', 15, 'rest', 'cmu',
                  cue='Greet warmly. Ask what feels tight today.',
                  voiceover="Fifteen minutes - proper warm-up. Where are you tight?"),
            Phase('warmup', 'Wake the arms', 120, 'warmup', 'cmu',
                  cue='Arm waves & reaches. Shoulders soft, breath long.',
                  voiceover="Start at the top. Big easy reaches.",
                  clip_pool=_SW_UPPER),
            Phase('warmup', 'Open the hips', 120, 'warmup', 'cmu',
                  cue='Hip swivels & side bends. Knees soft, ribs lifted.',
                  voiceover="Hips next. Swing through them.",
                  clip_pool=_SW_MID),
            Phase('warmup', 'Light feet', 180, 'warmup', 'cmu',
                  cue='Light jog / step-touch. Land soft, breath even.',
                  voiceover="Feet moving now. Light and easy.",
                  clip_pool=_SW_LOWER),
            Phase('warmup', 'Add some energy', 180, 'warmup', 'cmu',
                  cue='Side bounces, alt steps, dynamic stretch. Heart rate up a notch.',
                  voiceover="Bring the energy up a notch.",
                  clip_pool=_SW_DYNAMIC),
            Phase('cooldown', 'Breath out', 120, 'cooldown', 'cmu',
                  cue='Slow waves, gentle sway. Long exhale, drop shoulders.',
                  voiceover="Cool it down with me. Long exhales.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Wrap', 15, 'rest', None,
                  cue='Praise. Suggest coming back tomorrow.',
                  voiceover="Strong work. Come back tomorrow."),
        ]
        total = 750
    else:  # 10-min default
        phases = [
            Phase('checkin', 'Quick check-in', 15, 'rest', 'cmu',
                  cue='Greet warmly. Ask what feels tight today.',
                  voiceover="Ten minutes - warm-up time. Where are you tight?"),
            Phase('warmup', 'Wake the arms', 90, 'warmup', 'cmu',
                  cue='Arm waves & reaches. Shoulders soft, breath long.',
                  voiceover="Start at the top. Wake up the shoulders.",
                  clip_pool=_SW_UPPER),
            Phase('warmup', 'Open the hips', 90, 'warmup', 'cmu',
                  cue='Hip swivels & side bends. Knees soft, ribs lifted.',
                  voiceover="Hips next. Let them swing.",
                  clip_pool=_SW_MID),
            Phase('warmup', 'Light feet', 120, 'warmup', 'cmu',
                  cue='Light jog / step-touch. Land soft, breath even.',
                  voiceover="Get the feet moving. Light and easy.",
                  clip_pool=_SW_LOWER),
            Phase('warmup', 'Add some energy', 120, 'warmup', 'cmu',
                  cue='Side bounces, alt steps, dynamic stretch. Heart rate up.',
                  voiceover="Now bring the energy up.",
                  clip_pool=_SW_DYNAMIC),
            Phase('cooldown', 'Breath out', 90, 'cooldown', 'cmu',
                  cue='Slow waves, gentle sway. Long exhale.',
                  voiceover="Cool it down. Long exhale.",
                  clip_pool=_SW_BREATH),
            Phase('summary', 'Wrap', 15, 'rest', None,
                  cue='Praise. Note one thing they did well.',
                  voiceover="Nice work. Body says thanks."),
        ]
        total = 540
    return SessionTemplate(
        id=f'stretch_warmup_{duration_min}',
        title=f'Stretch & Warmup - {duration_min} min',
        style='cmu',
        total_sec=float(total),
        description=(f'{duration_min}-minute guided stretch + dynamic '
                     'warmup. No dance experience needed. Arms, hips, '
                     'feet, energy, breath.'),
        phases=phases,
    )


def get_template(template_id: str) -> Optional[SessionTemplate]:
    """Resolve a template id. 'quick{5|10|20}_<style>' resolves on the fly."""
    if template_id == 'house_foundations_d1':
        return _house_day1()
    # v59: dedicated Stretch & Warmup routines — no dance style.
    # Top-of-funnel niche: every user can do these, no learning curve,
    # safe to lead with. Three lengths: 5, 10, 15 minutes.
    for mins in (5, 10, 15):
        if template_id == f'stretch_warmup_{mins}':
            return _stretch_warmup(mins)
    name_map = {
        'gHO': 'House', 'gLH': 'LA-Style HipHop',
        'gKR': 'Krump', 'gWA': 'Waacking', 'gJS': 'Jazz',
        'gBR': 'Breaking', 'gLO': 'Locking', 'gPO': 'Popping',
        'gJB': 'Ballet Jazz', 'gMH': 'Middle HipHop',
    }
    for mins in (5, 10, 20):
        pfx = f'quick{mins}_'
        if template_id.startswith(pfx):
            style = template_id[len(pfx):]
            return _quick_start(style, name_map.get(style, style), mins)
    # Legacy: bare 'quick5_<style>' already handled above.
    return None


def list_templates() -> List[Dict[str, Any]]:
    """For UI: enumerate available session templates.
    Emits 5/10/20-min variants of the default styles + bonus programs."""
    out: List[Dict[str, Any]] = []
    # v59: Stretch & Warmup leads the catalogue — top-of-funnel niche.
    for mins in (5, 10, 15):
        t = get_template(f'stretch_warmup_{mins}')
        if t:
            out.append({
                'id': t.id, 'title': t.title, 'style': t.style,
                'total_sec': t.total_sec,
                'duration_min': mins,
                'description': t.description,
                'n_phases': len(t.phases),
                'category': 'stretch_warmup',
            })
    styles = ('gHO', 'gLH', 'gKR', 'gWA', 'gJS', 'gBR', 'gLO', 'gPO')
    for mins in (5, 10, 20):
        for sty in styles:
            t = get_template(f'quick{mins}_{sty}')
            if t:
                out.append({
                    'id': t.id, 'title': t.title, 'style': t.style,
                    'total_sec': t.total_sec,
                    'duration_min': mins,
                    'description': t.description,
                    'n_phases': len(t.phases),
                    'category': 'dance',
                })
    bonus = get_template('house_foundations_d1')
    if bonus:
        out.append({
            'id': bonus.id, 'title': bonus.title, 'style': bonus.style,
            'total_sec': bonus.total_sec, 'duration_min': 10,
            'description': bonus.description,
            'n_phases': len(bonus.phases),
            'category': 'program',
        })
    return out


# ── Runtime session state ────────────────────────────────────────
@dataclass
class Session:
    template: SessionTemplate
    phase_idx: int = 0
    started_at: float = field(default_factory=time.monotonic)
    phase_started_at: float = field(default_factory=time.monotonic)
    paused: bool = False
    paused_at: float = 0.0
    pause_accum: float = 0.0       # total time spent paused (across phases)
    phase_pause_accum: float = 0.0  # paused time in current phase only
    finished: bool = False
    # Per-phase telemetry (move ids played, etc.)
    history: List[Dict[str, Any]] = field(default_factory=list)
    # Variety bookkeeping: clips played in this session, by phase
    played_clips: List[str] = field(default_factory=list)
    # v29: rolling buffer of recently-spoken narration lines (dedupe).
    narration_recent: List[str] = field(default_factory=list)
    # v29: how many clips have been picked in current phase (lets us
    # decide WHEN to fire narration — every clip is too chatty).
    clip_count_in_phase: int = 0
    # v31: LLM-generated narration pool for this session. List of
    # {'text': str, 'mood': str, 'intent': str} dicts. Filled async
    # at session.start; falls back to static pool if generation fails.
    llm_narration_pool: List[Dict[str, Any]] = field(default_factory=list)
    llm_narration_cursor: int = 0

    @property
    def current(self) -> Optional[Phase]:
        if self.finished or self.phase_idx >= len(self.template.phases):
            return None
        return self.template.phases[self.phase_idx]

    def elapsed_total(self) -> float:
        if self.paused:
            return self.paused_at - self.started_at - self.pause_accum
        return time.monotonic() - self.started_at - self.pause_accum

    def elapsed_phase(self) -> float:
        if self.paused:
            return self.paused_at - self.phase_started_at - self.phase_pause_accum
        return time.monotonic() - self.phase_started_at - self.phase_pause_accum

    def remaining_phase(self) -> float:
        if self.current is None:
            return 0.0
        return max(0.0, self.current.duration_sec - self.elapsed_phase())

    def advance(self) -> Optional[Phase]:
        """Move to the next phase. Returns the new phase, or None if done."""
        self.history.append({
            'phase': self.template.phases[self.phase_idx].name,
            'elapsed': self.elapsed_phase(),
            'at': time.monotonic() - self.started_at,
        })
        self.phase_idx += 1
        if self.phase_idx >= len(self.template.phases):
            self.finished = True
            return None
        self.phase_started_at = time.monotonic()
        self.phase_pause_accum = 0.0
        self.clip_count_in_phase = 0
        return self.current

    def pause(self) -> None:
        if self.paused or self.finished:
            return
        self.paused = True
        self.paused_at = time.monotonic()

    def resume(self) -> None:
        if not self.paused:
            return
        delta = time.monotonic() - self.paused_at
        self.pause_accum += delta
        self.phase_pause_accum += delta
        self.paused = False

    def end(self) -> None:
        self.finished = True

    def snapshot(self) -> Dict[str, Any]:
        cur = self.current
        return {
            'template_id': self.template.id,
            'template_title': self.template.title,
            'style': self.template.style,
            'total_sec': self.template.total_sec,
            'phase_idx': self.phase_idx,
            'n_phases': len(self.template.phases),
            'phase_name': cur.name if cur else None,
            'phase_label': cur.label if cur else None,
            # Full ordered plan so the client can show a step/phase rail and
            # highlight the active one (the HUD only exposes the current phase).
            'plan': [
                {'name': p.name, 'label': p.label, 'intent': p.intent,
                 'cue': p.cue}
                for p in self.template.phases
            ],
            'current_phase': ({
                'intent': cur.intent,
                'name': cur.name,
                'label': cur.label,
            } if cur else None),
            'phase_duration_sec': cur.duration_sec if cur else 0.0,
            'phase_remaining_sec': self.remaining_phase(),
            'elapsed_total_sec': self.elapsed_total(),
            'paused': self.paused,
            'finished': self.finished,
            'played_clips': list(self.played_clips),
        }


# ── v29: per-clip narration pool ────────────────────────────────
# When the ticker picks a new clip mid-phase, we want the coach to
# say SOMETHING varied — not just the once-per-phase voiceover.
# Lines are short (≤ ~16 words), pickable in any order.

import random as _random  # noqa: E402

_NARRATION_POOL: Dict[str, List[str]] = {
    'warmup': [
        "Long, easy breath here.",
        "Let the joints wake up.",
        "Soft through the shoulders, jaw unclenched.",
        "Match my speed — no rush.",
        "Feel both feet flat on the floor.",
        "These are gentle stretches dancers actually use to start.",
        "Loose, not floppy.",
        "Open the chest, drop the shoulders.",
        "Nothing here should burn — pure mobility.",
        "Tiny range, full attention.",
    ],
    'drill_one_move': [
        "Watch the hips first — that's where the move lives.",
        "I'll do it slow, then take it up to tempo.",
        "Mirror me — same side, not opposite.",
        "Reset and try again. Reps are how this lands.",
        "This shape comes from {style_name} foundations.",
        "Stay loose in the knees — they're the suspension.",
        "Count it with me: one, two, three, four.",
        "Don't chase me — find your own tempo first.",
        "Smaller is better than sloppy. Shrink it if you need to.",
        "Feel that pulse? That's the move's heartbeat.",
        "Eyes up — find a focal point and let the body move under it.",
        "Slow is smooth, smooth is fast. Don't skip the slow rep.",
    ],
    'combo': [
        "Now we chain two pieces together.",
        "Transition cleanly — no rush between moves.",
        "Find the 'and' between counts.",
        "Same energy, different shape.",
        "Top of eight: reset, breathe, go.",
        "Treat the transition like its own move.",
        "If you lose it, just keep moving — pick up at the next 1.",
    ],
    'freestyle': [
        "Your turn — make it yours.",
        "Change the level: low, mid, high.",
        "Add an arm. See what happens.",
        "Half-time it — slow everything down.",
        "Don't think, just move.",
        "Repeat what feels good. That's how vocab is born.",
        "Steal a shape from me, change it, give it back.",
    ],
    'cooldown': [
        "Soft sway. Let the heart rate settle.",
        "Long exhale through the mouth.",
        "Drop the shoulders away from the ears.",
        "You did good work today.",
        "Notice where you feel warm — that's where you grew.",
        "Slow it down to half-speed, then half again.",
    ],
    'rest': [
        "Take a sip of water.",
        "Breathe and reset.",
    ],
}

_STYLE_NAMES: Dict[str, str] = {
    'gHO': 'House', 'gLH': 'LA-Style HipHop', 'gKR': 'Krump',
    'gWA': 'Waacking', 'gJS': 'Jazz', 'gBR': 'Breaking',
    'gLO': 'Locking', 'gPO': 'Popping', 'gJB': 'Ballet Jazz',
    'gMH': 'Middle HipHop', 'cmu': 'foundations',
}

# v30: per-style ORIGIN FACTS (vetted, accurate). Picked occasionally
# during drill phases so each style teaches its own history — not the
# same generic line every session.
_STYLE_FACTS: Dict[str, List[str]] = {
    'gHO': [  # House
        "House dance grew out of Chicago and New York club floors in the early 80s, alongside house music itself.",
        "The Jack is house's signature core — a chest ripple riding the four-on-the-floor.",
        "Footwork in house borrows from tap, salsa and African dance — it's a melting pot.",
        "Lofting is house's floorwork — fluid, low, almost like horizontal swimming.",
        "In house, the feet do the talking. The upper body just rides the bass.",
    ],
    'gLH': [  # Hip-Hop / LA-style
        "Hip-hop dance was born in the Bronx in the 70s at DJ Kool Herc's block parties.",
        "The bounce and the rock are the two roots of hip-hop — everything else branches from them.",
        "LA-style hip-hop layered choreography and musicality on top of the East Coast freestyle roots.",
        "Party rocking is the social side of hip-hop — it lives in cyphers and clubs, not just on stage.",
        "The two-step is hip-hop's universal language — every dancer has their own flavor of it.",
    ],
    'gKR': [  # Krump
        "Krump came out of South-Central LA in the early 2000s — Tommy the Clown and the RIZE doc put it on the map.",
        "Krump is built on four pillars: stomps, chest pops, arm swings, and jabs.",
        "The aggression in krump is emotional release, not anger — it's a conversation, not a fight.",
        "Krump sessions happen in 'fams' — crews that battle and raise each other up.",
        "A krump 'kill' is when one dancer's energy completely takes over the cypher.",
    ],
    'gWA': [  # Waacking
        "Waacking was born in 70s LA gay disco clubs — Tyrone Proctor and Arthur Goff shaped its vocabulary.",
        "The arms are everything in waacking — whips, poses, and points on the beat.",
        "Waacking takes from old Hollywood — Marlene Dietrich, Greta Garbo posing on the beat.",
        "Originally called 'punking,' the name was reclaimed and softened to waacking.",
        "Waacking is feminine energy and theatrical attitude — the dancer is always the star of their own movie.",
    ],
    'gJS': [  # Jazz
        "Jazz dance roots run from African American vernacular through the Charleston, Lindy and Fosse.",
        "Isolation — moving one body part while everything else stays still — came from jazz.",
        "Jack Cole codified what we now call 'theatrical jazz' in the mid-1900s.",
        "Jazz dance pulls from ballet for line but keeps the swing and syncopation alive.",
        "Every contemporary commercial style — from music videos to Broadway — borrows from jazz vocabulary.",
    ],
    'gBR': [  # Breaking
        "Breaking started at Kool Herc's 1973 Bronx parties — dancers went off during the 'break' in the record.",
        "The four pillars of breaking: toprock, footwork, power moves, and freezes.",
        "Toprock is your introduction — it sets your attitude before you ever hit the floor.",
        "Breaking became an Olympic sport in 2024 — same dance, same Bronx roots, bigger stage.",
        "A 'cypher' in breaking is the circle where dancers take turns — battles without judges.",
    ],
    'gLO': [  # Locking
        "Locking was invented by Don 'Campbellock' Campbell in late-1960s LA — partly by accident, on Soul Train.",
        "The lock — that frozen pose — is what makes the style. Hit it, hold it, move on.",
        "Locking is built on funk: James Brown, Sly Stone, and the Soul Train era.",
        "Points, wrist rolls, and the funky chicken are all locking signatures.",
        "Locking's character is playful — big smiles, big personality, audience interaction.",
    ],
    'gPO': [  # Popping
        "Popping came out of Fresno, California in the 70s — Boogaloo Sam and the Electric Boogaloos.",
        "A 'pop' is a quick contraction of a muscle on the beat — chest, arm, leg, neck.",
        "Popping is its own style — breaking, locking and popping are three different things.",
        "Animation, tutting, waving and boogaloo are all sub-styles that live inside popping.",
        "The robot predates popping — popping evolved out of robotic and funk styles in the 70s.",
    ],
    'gJB': [  # Ballet Jazz
        "Ballet jazz blends classical ballet's line with jazz dance's groove — Jack Cole pioneered the fusion.",
        "Eugene Loring's choreography brought ballet jazz into American musicals in the 1930s and 40s.",
        "You'll see ballet jazz everywhere on Broadway — it's the backbone of modern musical theatre dance.",
        "In ballet jazz the turnout is softer than classical, but the lines stay long and precise.",
    ],
}


def next_narration(sess: 'Session',
                   clip: Optional[Dict[str, Any]] = None,
                   recent_extra: Optional[List[str]] = None,
                   ) -> Optional[Dict[str, Any]]:
    """Return a varied narration entry for the current phase/clip.

    Returns a dict: {'text': str, 'mood': str|None} — or None.

    v31: Prefers the LLM-generated pool (sess.llm_narration_pool)
    when available, falling back to the static pool. The LLM pool is
    intent-filtered: only entries matching the current phase intent
    are eligible, so a drill line never plays during cooldown.
    """
    cur = sess.current
    if cur is None:
        return None
    sty = (sess.template.style if sess.template else None) or 'cmu'
    style_name = _STYLE_NAMES.get(sty, sty)
    recent_set = set(sess.narration_recent[-8:])
    if recent_extra:
        recent_set.update(recent_extra)

    # ── v31: try LLM-generated pool first ─────────────────────────
    llm_pool = sess.llm_narration_pool or []
    if llm_pool:
        eligible = [
            e for e in llm_pool
            if e.get('intent') in (cur.intent, 'any')
            and e.get('text') not in recent_set
        ]
        if eligible:
            entry = _random.choice(eligible)
            text = (entry.get('text') or '').replace('{style_name}', style_name)
            mood = entry.get('mood') or None
            sess.narration_recent.append(text)
            if len(sess.narration_recent) > 24:
                sess.narration_recent = sess.narration_recent[-24:]
            return {'text': text, 'mood': mood, 'source': 'llm'}

    # ── Fallback: static pool with optional style-specific facts ─
    pool: List[str] = list(_NARRATION_POOL.get(cur.intent) or [])
    use_facts = (
        cur.intent == 'drill_one_move'
        and sty in _STYLE_FACTS
        and _random.random() < 0.30
    )
    if use_facts:
        facts = _STYLE_FACTS[sty]
        pool = facts + [l for l in pool if '{style_name}' not in l]
    if not pool:
        return None
    candidates = [l for l in pool if l not in recent_set] or pool
    line = _random.choice(candidates)
    line = line.replace('{style_name}', style_name)
    sess.narration_recent.append(line)
    if len(sess.narration_recent) > 24:
        sess.narration_recent = sess.narration_recent[-24:]
    # Default mood by intent — gives avatar_life a hint even without LLM.
    intent_mood = {
        'warmup': 'relaxed', 'drill_one_move': 'focused',
        'combo': 'happy', 'freestyle': 'excited',
        'cooldown': 'relaxed', 'rest': 'relaxed',
    }.get(cur.intent)
    return {'text': line, 'mood': intent_mood, 'source': 'static'}


# ── v31: LLM-driven narration pool generation ────────────────────
# Once per session, ask the LLM to generate ~30 short coaching lines
# tagged with phase intent + mood. This kills repetition across days
# because the LLM produces fresh content each time.

_MOODS_ALLOWED = {'happy', 'relaxed', 'excited', 'focused', 'surprised',
                  'neutral'}
# Parse `[intent:warmup|mood:happy] line text...` or any subset.
import re as _re  # noqa: E402
_TAG_RE = _re.compile(
    r'^\s*\[([^\]]+)\]\s*(.+?)\s*$', _re.DOTALL)


def _parse_narration_line(raw: str,
                          default_intent: str = 'any') -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    # Strip leading list markers ("- ", "1. ", etc.)
    line = _re.sub(r'^\s*(?:[-*\u2022]|\d+[\.\)])\s+', '', raw.strip())
    intent = default_intent
    mood: Optional[str] = None
    m = _TAG_RE.match(line)
    if m:
        tags, body = m.group(1), m.group(2)
        for kv in tags.split('|'):
            k, _, v = kv.strip().partition(':')
            k = k.strip().lower()
            v = v.strip().lower()
            if k == 'intent' and v:
                intent = v
            elif k == 'mood' and v in _MOODS_ALLOWED:
                mood = v
        line = body.strip()
    # Strip trailing quotes / dangling punctuation noise.
    line = line.strip('"\u201c\u201d\u2018\u2019 ').strip()
    if not line or len(line) > 200:
        return None
    return {'text': line, 'intent': intent, 'mood': mood}


async def generate_session_narration(template: SessionTemplate,
                                     style_code: str,
                                     style_display: str,
                                     ) -> List[Dict[str, Any]]:
    """Ask the LLM for ~30 fresh narration lines for this session.

    Returns a list of {'text', 'intent', 'mood'} dicts. Returns []
    on any error — the caller should fall back to the static pool.
    Uses the same Azure OpenAI / Groq client the agent uses.
    """
    try:
        from coach.choreographer import agent as _agent
    except Exception:                                          # noqa: BLE001
        return []
    intents = []
    seen = set()
    for ph in template.phases:
        if ph.intent and ph.intent not in seen:
            intents.append(ph.intent)
            seen.add(ph.intent)
    intents_csv = ', '.join(intents) or 'warmup, drill_one_move, combo, cooldown'
    style_facts = _STYLE_FACTS.get(style_code, [])
    facts_block = ''
    if style_facts:
        facts_block = (
            'These are TRUE facts about ' + style_display
            + ' you may weave into drill lines (paraphrase, do NOT repeat verbatim):\n  - '
            + '\n  - '.join(style_facts[:5]) + '\n'
        )
    sys_prompt = (
        "You are writing micro-narration lines for a dance coach AI avatar. "
        "Each line is spoken between dance clips during a live session. "
        "Lines must be SHORT (≤ 18 words), warm, specific, and natural-sounding — "
        "like a real coach talking, not a motivational poster. "
        "NEVER use numbered 8-counts. NEVER use emoji. "
        "Each line MUST start with two tags in square brackets: "
        "[intent:X|mood:Y] where X is one of: " + intents_csv
        + " (use 'any' if generic), and Y is one of: happy, relaxed, excited, "
        "focused, surprised, neutral. "
        "Example: [intent:drill_one_move|mood:focused] Watch the hips first — that's where the move lives."
    )
    user_prompt = (
        f"Style: {style_display} ({style_code}). "
        f"Session phases: {intents_csv}. "
        f"Write exactly 30 narration lines, one per line, no blank lines, no preface, no closing. "
        f"Distribute them across the phases roughly proportional to phase duration. "
        f"{facts_block}"
        f"Mix coaching cues, encouragement, breath reminders, and 3-4 lines of genuine "
        f"{style_display} history or culture (paraphrased from the facts above). "
        f"Vary mood: most lines should be 'focused' or 'happy'; warmup/cooldown use 'relaxed'; "
        f"freestyle uses 'excited'. "
        f"Do NOT repeat any line."
    )
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    text: str = ''
    try:
        if getattr(_agent, '_aoai', None) is not None:
            import asyncio as _asyncio
            resp = await _asyncio.wait_for(
                _agent._aoai.chat.completions.create(
                    model=_agent.AOAI_DEPLOYMENT,
                    messages=messages,
                    temperature=0.85,
                    max_tokens=1200,
                    timeout=20,
                ),
                timeout=24,
            )
            text = (resp.choices[0].message.content or '') if resp.choices else ''
        elif getattr(_agent, '_client', None) is not None:
            import asyncio as _asyncio
            resp = await _asyncio.wait_for(
                _agent._client.chat.completions.create(
                    model=_agent.GROQ_MODEL,
                    messages=messages,
                    temperature=0.85,
                    max_tokens=1200,
                    timeout=20,
                ),
                timeout=24,
            )
            text = (resp.choices[0].message.content or '') if resp.choices else ''
        else:
            return []
    except Exception:                                          # noqa: BLE001
        return []
    out: List[Dict[str, Any]] = []
    for raw in (text or '').splitlines():
        parsed = _parse_narration_line(raw)
        if parsed:
            out.append(parsed)
    # Deduplicate while preserving order.
    seen_text: set = set()
    deduped: List[Dict[str, Any]] = []
    for e in out:
        if e['text'] in seen_text:
            continue
        seen_text.add(e['text'])
        deduped.append(e)
    return deduped

