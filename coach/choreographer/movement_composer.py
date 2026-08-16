"""movement_composer.py — generative dance phrases from existing clips.

The user has 400+ retargeted VRM-quat clips across 10 AIST dance
styles. Instead of always teaching the SAME canonical clip per move,
the composer stitches *new sequences* out of beat-aligned slices of
existing clips. Same source motion, fresh choreography — keeps users
engaged session after session without us shipping new mocap.

This module produces structured ``Phrase`` objects, NEVER pixels.
Every segment references ``(clip_id, frame_start, frame_end)`` and
the browser ``MotionPlayer`` plays them back-to-back using its
existing ``fromFrame``/``toFrame`` window support — no new playback
code needed.

Composition strategy
--------------------
1. Group clips by style code (``gHO``, ``gLH``, …) and load each
   clip's beat track from ``motion_cues_v33/<clip>.json``.
2. For a requested style + energy + length, pick 2-4 source clips
   from compatible style families (configurable). Compatible families
   are defined in :data:`STYLE_FAMILIES`.
3. From each source clip select a contiguous run of beats (default
   2 beats per segment so 4 segments = 8 counts).
4. Concatenate the segments. The browser plays segment N, the
   session ticker fires the next on ``avatar.clip_done``.

Determinism + caching
---------------------
Every composition is deterministic given a (style, seed) pair. The
manifest is persisted to ``coach/generated_movements/<style>/<hash>.json``
the first time it is produced so the LLM can replay exactly the same
phrase later (or grade it consistently).

Public API
----------
``compose_phrase(style, energy='medium', seed=None, beats=8,
                 allow_blend=True) -> Phrase``
``list_phrases(style=None) -> List[Phrase]``
``get_phrase(phrase_id) -> Optional[Phrase]``
``rebuild_index() -> int`` — wipes cache + rebuilds (admin only).
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from coach import motion_index

ROOT = Path(__file__).resolve().parents[1]
CUE_DIR = ROOT / 'motion_cues_v33'
OUT_DIR = ROOT / 'generated_movements'

# Compatible style families. Within a family the composer may pull
# segments from different style codes to produce a fusion phrase
# ("hip-hop + house" combos feel natural; "krump + ballet" does not).
STYLE_FAMILIES: Dict[str, List[str]] = {
    'hiphop':   ['gLH', 'gMH', 'gHO', 'gLO', 'gJB', 'gJS'],
    'street':   ['gBR', 'gKR', 'gWA', 'gPO'],
    'house':    ['gHO', 'gLH'],
    'breaking': ['gBR', 'gKR'],
    'popping':  ['gPO', 'gLO'],
    'waacking': ['gWA', 'gJB'],
    'cmu':      ['cmu'],
}

# How many counts the composer treats as one "musical 8-count".
PHRASE_BEATS = 8

# How many beats per segment by default. 2 beats per segment + 4
# segments = 8 counts = a tight 1-bar phrase at most BPMs.
DEFAULT_SEGMENT_BEATS = 2

# Bucket label per style code (drives the UI "style buckets").
STYLE_BUCKETS: Dict[str, str] = {
    'gBR': 'breaking', 'gHO': 'house', 'gJB': 'jazz_ballet',
    'gJS': 'street_jazz', 'gKR': 'krump', 'gLH': 'la_hiphop',
    'gLO': 'locking', 'gMH': 'middle_hiphop', 'gPO': 'popping',
    'gWA': 'waacking', 'cmu': 'basics',
}


@dataclass
class Segment:
    """One contiguous slice of a source clip."""
    clip_id: str
    frame_start: int
    frame_end: int
    beats: int = 0
    cue: str = ''
    body_parts: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    music_url: Optional[str] = None


@dataclass
class Phrase:
    """A composed dance phrase: ordered list of segments."""
    id: str
    style: str
    bucket: str
    energy: str
    beats: int
    duration_sec: float
    title: str
    summary: str
    segments: List[Segment]
    seed: int
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['segments'] = [asdict(s) for s in self.segments]
        return d


# ─── beat cue loading ────────────────────────────────────────────────
_CUES_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _load_beats(clip_id: str) -> List[Dict[str, Any]]:
    """Return the per-beat cue list for a clip, [] if no cues exist."""
    if clip_id in _CUES_CACHE:
        cached = _CUES_CACHE[clip_id]
        return list(cached.get('beats') or []) if cached else []
    fp = CUE_DIR / f'{clip_id}.json'
    if not fp.exists():
        _CUES_CACHE[clip_id] = None
        return []
    try:
        raw = json.loads(fp.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            _CUES_CACHE[clip_id] = raw
            return list(raw.get('beats') or [])
    except Exception:
        _CUES_CACHE[clip_id] = None
    return []


# ─── catalogue helpers ───────────────────────────────────────────────
_STYLE_INDEX: Dict[str, List[Dict[str, Any]]] = {}
_STYLE_INDEX_BUILT_AT: float = 0.0


def _build_style_index() -> Dict[str, List[Dict[str, Any]]]:
    """Group safe, retargeted clips by their style code.

    Result is cached in-process for 10 minutes — the underlying
    motion_index is ``@lru_cache(maxsize=1)`` so this is essentially
    free, but the explicit cache shields us if motion_index grows
    invalidation logic later.
    """
    global _STYLE_INDEX, _STYLE_INDEX_BUILT_AT
    if _STYLE_INDEX and (time.time() - _STYLE_INDEX_BUILT_AT) < 600:
        return _STYLE_INDEX
    out: Dict[str, List[Dict[str, Any]]] = {}
    for m in motion_index.list_motions():
        if not m.get('retargeted'):
            continue
        if m.get('safety') not in ('ok', 'unknown'):
            continue
        out.setdefault(m['genre'], []).append(m)
    _STYLE_INDEX = out
    _STYLE_INDEX_BUILT_AT = time.time()
    return out


def style_pool(style: str, allow_blend: bool = True) -> List[Dict[str, Any]]:
    """Return the clip pool eligible for a composition.

    If ``allow_blend`` is True and the style belongs to a family,
    siblings from the family are also included so the composer can
    fuse e.g. LA hip-hop with house. Family membership is conservative
    (see :data:`STYLE_FAMILIES`) so we never blend, say, breaking
    into a gentle warmup.
    """
    idx = _build_style_index()
    if style not in idx and style != 'cmu':
        return []
    pool: List[Dict[str, Any]] = list(idx.get(style) or [])
    if not allow_blend:
        return pool
    for fam_styles in STYLE_FAMILIES.values():
        if style in fam_styles:
            for sib in fam_styles:
                if sib == style:
                    continue
                # Cap family-imports so the home style still dominates
                # (no more than 25% of pool per sibling).
                cap = max(1, len(pool) // 4)
                pool.extend((idx.get(sib) or [])[:cap])
    return pool


# ─── core composition ───────────────────────────────────────────────
def _pick_beat_run(beats: List[Dict[str, Any]], want: int,
                   rng: random.Random) -> Optional[Tuple[int, int, int,
                                                          str, List[str]]]:
    """Pick a contiguous beat run of length ``want``.

    Returns (frame_start, frame_end, beats_taken, cue_summary,
    dominant_body_parts) or None if the clip is too short.
    """
    if len(beats) < want:
        if not beats:
            return None
        want = max(1, len(beats))
    # Skip the first beat (often a "pickup"/anacrusis) and the last
    # beat (often a final pose) when the clip has enough beats; the
    # middle of the clip carries the cleanest motion.
    start_lo = 1 if len(beats) > (want + 2) else 0
    start_hi = max(start_lo, len(beats) - want)
    start = rng.randint(start_lo, start_hi)
    chunk = beats[start:start + want]
    frame_start = int(chunk[0].get('frame_start') or 0)
    frame_end = int(chunk[-1].get('frame_end')
                    or chunk[-1].get('frame_start') or 0) + 1
    cue = ', '.join(str(b.get('cue') or '').strip()
                    for b in chunk if b.get('cue')).strip(', ')
    parts: List[str] = []
    for b in chunk:
        for p in (b.get('body_parts') or []):
            if p and p not in parts:
                parts.append(p)
    return frame_start, frame_end, want, cue, parts


def _energy_segments(energy: str) -> Tuple[int, int]:
    """How many segments + beats-per-segment for an energy level."""
    energy = (energy or 'medium').lower()
    if energy == 'low':
        return 2, 4  # 2 segments × 4 beats = 8 counts, calm
    if energy == 'high':
        return 4, 2  # 4 punchy 2-beat hits
    return 3, 3      # default — three 3-beat segments


def _phrase_id(style: str, segments: List[Segment]) -> str:
    sig = '|'.join(f'{s.clip_id}:{s.frame_start}-{s.frame_end}'
                   for s in segments)
    h = hashlib.sha1(f'{style}::{sig}'.encode('utf-8')).hexdigest()[:12]
    return f'mix_{style}_{h}'


def _segment_duration(clip: Dict[str, Any], frame_start: int,
                      frame_end: int) -> float:
    fps = max(1, int(clip.get('fps') or 30))
    return round(max(0, frame_end - frame_start) / fps, 2)


def _segment_title(seg: Segment, clip: Dict[str, Any]) -> str:
    base = clip.get('title') or clip['id']
    if seg.cue:
        return f'{base} — {seg.cue}'
    return base


def compose_phrase(style: str, energy: str = 'medium',
                   seed: Optional[int] = None,
                   beats: int = PHRASE_BEATS,
                   allow_blend: bool = True,
                   persist: bool = True) -> Optional[Phrase]:
    """Generate a new phrase for ``style``.

    Args:
        style: AIST style code (``gHO``, ``gLH``, …) or ``cmu``.
        energy: ``'low'`` | ``'medium'`` | ``'high'`` — controls how
            many segments + beats per segment.
        seed: Optional RNG seed for deterministic output. When None
            uses a time-based seed.
        beats: Target total beats. Defaults to 8 (one 8-count).
        allow_blend: Whether to import family-sibling clips.
        persist: Cache the phrase JSON under ``generated_movements``.

    Returns:
        A :class:`Phrase`, or ``None`` if the catalogue is empty for
        this style.
    """
    pool = style_pool(style, allow_blend=allow_blend)
    if not pool:
        return None
    n_segs, segment_beats = _energy_segments(energy)
    # Re-balance to hit the requested total beats roughly.
    if n_segs * segment_beats != beats:
        segment_beats = max(1, beats // n_segs)
    seed_v = int(seed) if seed is not None else int(time.time() * 1000)
    rng = random.Random(seed_v)
    rng.shuffle(pool)

    segments: List[Segment] = []
    used_clip_ids: set = set()
    for clip in pool:
        if len(segments) >= n_segs:
            break
        cid = clip['id']
        if cid in used_clip_ids:
            continue
        beat_list = _load_beats(cid)
        if not beat_list:
            continue
        run = _pick_beat_run(beat_list, segment_beats, rng)
        if not run:
            continue
        fs, fe, took, cue, parts = run
        seg = Segment(
            clip_id=cid,
            frame_start=fs,
            frame_end=fe,
            beats=took,
            cue=cue,
            body_parts=parts,
            duration_sec=_segment_duration(clip, fs, fe),
            music_url=motion_index.music_url_for(cid),
        )
        segments.append(seg)
        used_clip_ids.add(cid)

    # Fallback: if we couldn't find enough cued clips, fill the rest
    # by playing FULL clips at half-length. Keeps generation from
    # failing on niche styles with sparse cue coverage.
    if len(segments) < n_segs:
        for clip in pool:
            if len(segments) >= n_segs:
                break
            cid = clip['id']
            if cid in used_clip_ids:
                continue
            total = int(clip.get('frames') or 0)
            if total < 30:
                continue
            half = total // 2
            seg = Segment(
                clip_id=cid,
                frame_start=0,
                frame_end=half,
                beats=segment_beats,
                cue='',
                body_parts=[],
                duration_sec=_segment_duration(clip, 0, half),
                music_url=motion_index.music_url_for(cid),
            )
            segments.append(seg)
            used_clip_ids.add(cid)

    if not segments:
        return None

    bucket = STYLE_BUCKETS.get(style, 'mixed')
    total_dur = round(sum(s.duration_sec for s in segments), 2)
    title = f'{bucket.replace("_", " ").title()} remix · {energy}'
    summary = (
        f'{len(segments)} beat-aligned slices stitched into a '
        f'{int(sum(s.beats for s in segments))}-count phrase.')
    pid = _phrase_id(style, segments)
    phrase = Phrase(
        id=pid,
        style=style,
        bucket=bucket,
        energy=energy,
        beats=int(sum(s.beats for s in segments)),
        duration_sec=total_dur,
        title=title,
        summary=summary,
        segments=segments,
        seed=seed_v,
        created_at=time.time(),
    )
    if persist:
        _persist_phrase(phrase)
    return phrase


# ─── persistence ──────────────────────────────────────────────────────
def _phrase_path(style: str, phrase_id: str) -> Path:
    return OUT_DIR / style / f'{phrase_id}.json'


def _persist_phrase(phrase: Phrase) -> None:
    fp = _phrase_path(phrase.style, phrase.id)
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(json.dumps(phrase.to_dict(), ensure_ascii=False,
                                 indent=2), encoding='utf-8')
        _update_index(phrase)
    except Exception:
        pass


def _index_path() -> Path:
    return OUT_DIR / 'index.json'


def _read_index() -> Dict[str, Any]:
    p = _index_path()
    if not p.exists():
        return {'updated_at': '', 'buckets': {}, 'phrases': {}}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {'updated_at': '', 'buckets': {}, 'phrases': {}}


def _write_index(idx: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _index_path().write_text(json.dumps(idx, ensure_ascii=False, indent=2),
                              encoding='utf-8')


def _update_index(phrase: Phrase) -> None:
    idx = _read_index()
    idx['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                      time.gmtime())
    idx.setdefault('buckets', {}).setdefault(phrase.bucket, []).append(phrase.id)
    # dedupe bucket list
    seen, dedup = set(), []
    for pid in idx['buckets'][phrase.bucket]:
        if pid in seen:
            continue
        seen.add(pid)
        dedup.append(pid)
    idx['buckets'][phrase.bucket] = dedup[-200:]  # cap
    idx.setdefault('phrases', {})[phrase.id] = {
        'style': phrase.style, 'bucket': phrase.bucket,
        'energy': phrase.energy, 'beats': phrase.beats,
        'duration_sec': phrase.duration_sec, 'title': phrase.title,
    }
    _write_index(idx)


def get_phrase(phrase_id: str) -> Optional[Phrase]:
    """Load a cached phrase by id."""
    idx = _read_index()
    meta = (idx.get('phrases') or {}).get(phrase_id)
    if not meta:
        return None
    fp = _phrase_path(meta.get('style', ''), phrase_id)
    if not fp.exists():
        return None
    try:
        raw = json.loads(fp.read_text(encoding='utf-8'))
        segs = [Segment(**s) for s in (raw.get('segments') or [])]
        return Phrase(
            id=raw['id'], style=raw['style'], bucket=raw['bucket'],
            energy=raw['energy'], beats=int(raw['beats']),
            duration_sec=float(raw['duration_sec']),
            title=raw['title'], summary=raw['summary'],
            segments=segs, seed=int(raw['seed']),
            created_at=float(raw['created_at']),
        )
    except Exception:
        return None


def list_phrases(style: Optional[str] = None,
                 bucket: Optional[str] = None) -> List[Dict[str, Any]]:
    """List indexed phrase metadata, optionally filtered."""
    idx = _read_index()
    out: List[Dict[str, Any]] = []
    for pid, meta in (idx.get('phrases') or {}).items():
        if style and meta.get('style') != style:
            continue
        if bucket and meta.get('bucket') != bucket:
            continue
        out.append({'id': pid, **meta})
    return out


def buckets() -> Dict[str, List[str]]:
    """Return ``{bucket: [phrase_id, ...]}`` for the UI."""
    return dict((_read_index().get('buckets') or {}))


def prewarm(per_style: int = 6) -> int:
    """Pre-generate ``per_style`` phrases for every known style.

    Called at server startup so the UI / LLM always has fresh combos
    ready without paying generation latency on the first request.
    Returns the number of phrases produced this call.
    """
    made = 0
    seed_base = int(time.time())
    for sty in STYLE_BUCKETS:
        for i in range(per_style):
            for energy in ('low', 'medium', 'high'):
                ph = compose_phrase(sty, energy=energy,
                                    seed=seed_base + made,
                                    allow_blend=True, persist=True)
                if ph:
                    made += 1
    return made


def rebuild_index() -> int:
    """Drop the index file and rebuild it by walking ``OUT_DIR``.

    Returns the number of phrases re-indexed.
    """
    new_idx: Dict[str, Any] = {
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'buckets': {},
        'phrases': {},
    }
    if OUT_DIR.exists():
        for fp in OUT_DIR.rglob('*.json'):
            if fp.name == 'index.json':
                continue
            try:
                raw = json.loads(fp.read_text(encoding='utf-8'))
            except Exception:
                continue
            pid = raw.get('id')
            if not pid:
                continue
            new_idx['phrases'][pid] = {
                'style': raw.get('style'), 'bucket': raw.get('bucket'),
                'energy': raw.get('energy'), 'beats': raw.get('beats'),
                'duration_sec': raw.get('duration_sec'),
                'title': raw.get('title'),
            }
            new_idx['buckets'].setdefault(raw.get('bucket', 'mixed'),
                                          []).append(pid)
    _write_index(new_idx)
    return len(new_idx['phrases'])


def segments_as_browser_events(phrase: Phrase) -> List[Dict[str, Any]]:
    """Map a Phrase to the browser event list the WS handler emits.

    Each segment becomes an ``avatar.load`` event with the optional
    ``frame_start`` / ``frame_end`` fields the player already honours.
    The browser fires ``session.clip_done`` when each segment ends,
    so the server can step through the list one segment at a time.
    """
    out: List[Dict[str, Any]] = []
    for s in phrase.segments:
        out.append({
            'type': 'avatar.load',
            'clip_id': s.clip_id,
            'frame_start': int(s.frame_start),
            'frame_end': int(s.frame_end),
            'music_url': s.music_url,
            'composition_id': phrase.id,
        })
    return out
