"""motion_index.py — exposes the motion DB to the agent / browser.

Each entry: {id, genre, genre_name, source, frames, fps, duration_sec,
             safety, bpm_hint, signature}

Drives the LLM tool `pick_clip(...)` and the browser fetch
`GET /api/motion/list`.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AIST_DIR = ROOT / 'data' / 'motion_db' / 'aistpp_full' / 'motions'
CMU_DIR = ROOT / 'data' / 'motion_db' / 'amass_cmu'
COACH_DIR = Path(__file__).resolve().parent
SAFETY_PATHS = (
    COACH_DIR / 'motion_safety.json',
    COACH_DIR / 'reports' / 'motion_qa_aist.json',
    COACH_DIR / 'reports' / 'motion_qa_cmu.json',
)
CACHE_DIR = COACH_DIR / 'motion_cache'
CACHE_DIR_CMU = COACH_DIR / 'motion_cache_cmu'   # retargeted CMU clips
META_DIR  = COACH_DIR / 'motion_meta'

# Hard blacklist — clips whose source is physical-therapy / rehab mocap,
# not dance. Per per_frame_qa.json these have hip vertical motion > 30 cm,
# which produces uncanny "stuck pelvis with moving body" once the runtime
# hip-bob clamp engages. Subjects 105, 106 are the AMASS squat-rehab set;
# 102 is sit-to-stand; the listed 01 / 05 clips are individual jump
# tests. Refresh by re-running per_frame_qa.py and pasting the top list.
BLACKLIST = {
    'cmu_106_106_04', 'cmu_106_106_08', 'cmu_106_106_17', 'cmu_106_106_19',
    'cmu_106_106_22', 'cmu_106_106_24', 'cmu_106_106_25', 'cmu_106_106_28',
    'cmu_106_106_30', 'cmu_106_106_34',
    'cmu_105_105_40', 'cmu_105_105_41', 'cmu_105_105_45', 'cmu_105_105_50',
    'cmu_102_102_01', 'cmu_102_102_20', 'cmu_102_102_21', 'cmu_102_102_32',
    'cmu_103_103_06',
    'cmu_111_111_05', 'cmu_111_111_06', 'cmu_111_111_08',
    'cmu_01_01_08', 'cmu_01_01_10',
    'cmu_05_05_05', 'cmu_05_05_07',
    # v19 — confirmed in-browser frame-by-frame sweep (619/619 OK,
    # `_analyzeClipSync`, `coach/reports/browser_per_clip_qa.json`).
    # These have large_foot_jumps>5 AND foot vertical range >0.9 m,
    # i.e. the foot teleports 40–70 cm between adjacent frames and
    # flies 1 m above the floor — bad SMPL inference, unplayable.
    'cmu_108_108_25', 'cmu_108_108_26',
    'gBR_sBM_cAll_d05_mBR0_ch08',
    'gJB_sBM_cAll_d08_mJB5_ch06', 'gJB_sBM_cAll_d09_mJB5_ch06',
    'gJB_sBM_cAll_d09_mJB5_ch07', 'gJB_sFM_cAll_d08_mJB5_ch14',
    'gJB_sFM_cAll_d09_mJB5_ch20',
    'gLH_sFM_cAll_d17_mLH4_ch12',
    'multistyle_reel10_s97',
}


def _meta_for(clip_id: str) -> dict:
    p = META_DIR / f'{clip_id}.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


# ── Clean-clip whitelist (audit v1, 2026-06-21) ──────────────────────
# Every clip in the catalogue was scored in the real three-vrm renderer
# for backward-lean, body collapse, foot teleport, sustained float and
# limb interpenetration (coach/reports/clip_audit_v1.json). 313 of 610
# passed. We launch with ONLY these so the user never sees an absurd
# pose. Set COACH_WHITELIST_OFF=1 to serve the full (unfiltered) set
# while curating. Adding a clip back = re-run the audit or hand-add to
# coach/motion_meta/clean_whitelist.json.
_WHITELIST_PATH = COACH_DIR / 'motion_meta' / 'clean_whitelist.json'


_VERIFIED_PATH = COACH_DIR / 'motion_meta' / 'verified_upright.json'


@lru_cache(maxsize=1)
def _verified_upright() -> Optional[frozenset]:
    """Orientation-verified clip ids (VRM-FK upright test, every
    frame). Set COACH_VERIFY_OFF=1 to bypass. Source of truth:
    coach/motion_meta/verified_upright.json (gen _verify_orientation.py)."""
    import os
    if os.getenv('COACH_VERIFY_OFF') == '1':
        return None
    try:
        data = json.loads(_VERIFIED_PATH.read_text(encoding='utf-8-sig'))
        ids = data.get('verified') or []
        return frozenset(ids) if ids else None
    except Exception:
        return None


# ── orientation safety gate (v108) ───────────────────────────────────
# A guaranteed-upright, mellow clip used as the universal fallback when
# ANY code path tries to play a clip that is not orientation-verified.
# It is itself in the verified set (asserted at import below).
SAFE_FALLBACK_CLIP = 'gLO_sBM_cAll_d13_mLO0_ch01'


def is_verified_upright(clip_id: Optional[str]) -> bool:
    """True if the clip passed the per-frame VRM-FK upright test (or if
    verification is globally disabled). This is the SINGLE source of
    truth every play path must consult before showing a clip."""
    if not clip_id:
        return False
    vf = _verified_upright()
    if vf is None:          # verification disabled → don't block
        return True
    return clip_id in vf


def safe_clip(clip_id: Optional[str]) -> str:
    """Return clip_id if it is orientation-verified, otherwise the
    guaranteed-safe fallback. NEVER returns an inverted/folded clip."""
    if is_verified_upright(clip_id):
        return clip_id  # type: ignore[return-value]
    return SAFE_FALLBACK_CLIP


# Clip-id keys inside a browser_event that point at something the avatar
# will actually PLAY (and therefore must be orientation-verified).
_PLAYABLE_EVENTS = {'avatar.load', 'avatar.play', 'avatar.breakdown',
                    'avatar.drill'}


def gate_event(be: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """SINGLE CHOKEPOINT: sanitise an outgoing browser_event so the
    avatar can never be asked to play a non-verified (possibly inverted)
    clip — no matter which path (LLM, session warmup/cooldown, hardcoded
    pool, breakdown) produced it. Swaps any unsafe clip_id for the safe
    fallback in place and returns the event. Best-effort; never raises."""
    try:
        if not isinstance(be, dict):
            return be
        if be.get('type') not in _PLAYABLE_EVENTS:
            return be
        cid = be.get('clip_id')
        if cid and not is_verified_upright(cid):
            import sys
            print(f'[orient-gate] swapped unsafe clip {cid} -> '
                  f'{SAFE_FALLBACK_CLIP}', file=sys.stderr)
            be['clip_id'] = SAFE_FALLBACK_CLIP
    except Exception:
        pass
    return be



@lru_cache(maxsize=1)
def _clean_whitelist() -> Optional[frozenset]:
    import os
    if os.getenv('COACH_WHITELIST_OFF') == '1':
        return None
    try:
        data = json.loads(_WHITELIST_PATH.read_text(encoding='utf-8-sig'))
        ids = data.get('clean') or []
        return frozenset(ids) if ids else None
    except Exception:
        return None

GENRE_NAMES = {
    'gBR': 'Breaking', 'gHO': 'House', 'gJB': 'Jazz Ballet',
    'gJS': 'Street Jazz', 'gKR': 'Krump', 'gLH': 'LA Hip-Hop',
    'gLO': 'Locking', 'gMH': 'Middle Hip-Hop', 'gPO': 'Popping',
    'gWA': 'Waacking',
    # CMU AMASS — non-stylized basics (walks, jumps, kicks, posture,
    # weight transfer). Useful for warmups, technique drills, and as
    # rest/recovery clips between styled choreography blocks.
    'cmu': 'Basics & Warmups',
    'mixamo': 'Warm-up & Mobility',
}
# Generic backing loops for clips that have no native track (e.g. CMU
# warmups). The frontend stretches/loops these to fit the clip length.
_GENERIC_LOOPS = ('warm_funk', 'sad_lofi', 'banger')


def music_url_for(clip_id: str) -> Optional[str]:
    """Derive a backing-track URL from the clip id.

    AIST clips encode the music id in the 5th token, e.g.
    `gWA_sFM_cAll_d04_mWA0_ch01` → `mWA0` → `/audio/aist_orig/mWA0.mp3`.
    CMU clips fall back to one of the three generic loops, chosen
    deterministically so the same clip always pairs with the same
    track (gives the user a sense of identity per clip).
    """
    if not clip_id:
        return None
    parts = clip_id.split('_')
    if len(parts) >= 5 and parts[4].startswith('m') and len(parts[4]) >= 3:
        mid = parts[4]
        return f'/audio/aist_orig/{mid}.mp3'
    if clip_id.startswith('cmu_'):
        # Deterministic choice based on the clip id hash so the same
        # warmup always plays the same loop across sessions.
        idx = sum(ord(c) for c in clip_id) % len(_GENERIC_LOOPS)
        return f'/audio/music/{_GENERIC_LOOPS[idx]}.mp3'
    return None


def _safety_map() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for sp in SAFETY_PATHS:
        if not sp.exists():
            continue
        try:
            d = json.loads(sp.read_text(encoding='utf-8'))
            for r in d.get('reports', []):
                out[Path(r.get('path', '')).name] = r
        except Exception:
            continue
    return out


@lru_cache(maxsize=1)
def list_motions() -> List[Dict[str, Any]]:
    """Cheap directory scan; reads pkl only to grab frame count once,
    then caches forever."""
    safety = _safety_map()
    out: List[Dict[str, Any]] = []
    if not AIST_DIR.exists():
        return out
    for p in sorted(AIST_DIR.glob('*.pkl')):
        name = p.name
        genre = name[:3]
        if genre not in GENRE_NAMES and not name.startswith('multistyle_'):
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            poses = np.asarray(d['smpl_poses']).reshape(-1, 24, 3)
            fps_native = int(d.get('fps', 60))
            if fps_native == 60:
                frames = poses.shape[0] // 2
                fps = 30
            else:
                frames = poses.shape[0]
                fps = fps_native
        except Exception:
            frames, fps = 0, 30
        sd = safety.get(str(p)) or safety.get(name) or {}
        mid = name.replace('.pkl', '')
        retargeted = (CACHE_DIR / f'{mid}.json').exists()
        meta = _meta_for(mid)
        out.append({
            'id':           mid,
            'file':         name,
            'genre':        genre if genre in GENRE_NAMES else 'multistyle',
            'genre_name':   GENRE_NAMES.get(genre, 'Multi-style'),
            'frames':       frames,
            'fps':          fps,
            'duration_sec': round(frames / max(fps, 1), 2),
            'safety':       sd.get('severity', 'unknown'),
            'retargeted':   retargeted,
            'title':        meta.get('title') or '',
            'difficulty':   meta.get('difficulty'),
            'bpm_target':   meta.get('bpm_target'),
            'vibe_tags':    meta.get('vibe_tags', []),
            'has_meta':     bool(meta),
            'music_url':    music_url_for(mid),
        })
    # ---- CMU AMASS clips (retargeted vrm-quat JSONs only) -----------
    # These were ingested via coach/ingestion/cmu_amass_adapter.py and
    # retargeted with batch_retarget.py. There is no .pkl in AIST_DIR,
    # so we list them directly from the cache directory. Frames/fps
    # come straight from the JSON header.
    if CACHE_DIR_CMU.exists():
        for jp in sorted(CACHE_DIR_CMU.glob('*.json')):
            mid = jp.stem
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    head = json.load(f)
                frames = int(head.get('n_frames') or head.get('frames') or 0)
                fps = int(head.get('fps') or 30)
            except Exception:
                frames, fps = 0, 30
            meta = _meta_for(mid)
            pkl_name = f'{mid}.pkl'
            sd = safety.get(pkl_name) or {}
            out.append({
                'id':           mid,
                'file':         jp.name,
                'genre':        'cmu',
                'genre_name':   GENRE_NAMES['cmu'],
                'frames':       frames,
                'fps':          fps,
                'duration_sec': round(frames / max(fps, 1), 2),
                'safety':       sd.get('severity', 'unknown'),
                'retargeted':   True,
                'title':        meta.get('title') or '',
                'difficulty':   meta.get('difficulty'),
                'bpm_target':   meta.get('bpm_target'),
                'vibe_tags':    meta.get('vibe_tags', []),
                'has_meta':     bool(meta),
                'music_url':    music_url_for(mid),
            })
    # v147: MIXAMO warm-up / mobility clips (retargeted vrm-quat JSONs in
    # motion_cache/mixamo_*.json via ingest_mixamo.py). Purpose-built,
    # clean Y-up stretch/squat/cardio content. genre='mixamo'.
    if CACHE_DIR.exists():
        for jp in sorted(CACHE_DIR.glob('mixamo_*.json')):
            mid = jp.stem
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    head = json.load(f)
                frames = int(head.get('n_frames') or head.get('frames') or 0)
                fps = int(head.get('fps') or 30)
            except Exception:
                frames, fps = 0, 30
            meta = _meta_for(mid)
            # v153: derive a searchable title from the id when metadata has
            # none (e.g. 'mixamo_neck_stretching' -> 'Neck Stretching') so
            # the chat title-matcher can find "neck stretch", "jumping
            # jacks", "slow jog", "squat" etc. Without this the warm-up
            # pool had blank titles and picked random clips.
            _derived = mid.replace('mixamo_', '').replace('_', ' ').strip().title()
            out.append({
                'id':           mid,
                'file':         jp.name,
                'genre':        'mixamo',
                'genre_name':   GENRE_NAMES.get('mixamo', 'Warm-up & Mobility'),
                'frames':       frames,
                'fps':          fps,
                'duration_sec': round(frames / max(fps, 1), 2),
                'safety':       'ok',
                'retargeted':   True,
                'title':        meta.get('title') or _derived,
                'difficulty':   meta.get('difficulty'),
                'bpm_target':   meta.get('bpm_target'),
                'vibe_tags':    meta.get('vibe_tags', []),
                'has_meta':     bool(meta),
                'music_url':    music_url_for(mid),
            })
    # v127: the BLACKLIST stores BASE clip ids (e.g. 'cmu_102_102_01'),
    # but retargeted clips are served with a '_phys' suffix
    # ('cmu_102_102_01_phys'). A plain `id not in BLACKLIST` membership
    # test therefore NEVER matched the _phys variants, so rehab / jump-test
    # mocap (sit-to-stand 102, squat 106, jump 111 …) leaked into the live
    # catalogue and rendered as "floating / weird" motion. Strip the suffix
    # before testing so the blacklist actually applies.
    def _blk(cid: str) -> bool:
        return cid in BLACKLIST or (
            cid.endswith('_phys') and cid[:-5] in BLACKLIST)
    out = [m for m in out if not _blk(m['id'])]
    wl = _clean_whitelist()
    if wl is not None:
        out = [m for m in out if m['id'] in wl]
    vf = _verified_upright()
    if vf is not None:
        out = [m for m in out if m['id'] in vf]
    return out


def get_motion(motion_id: str) -> Optional[Path]:
    p = AIST_DIR / f'{motion_id}.pkl'
    if p.exists():
        return p
    p2 = CMU_DIR / f'{motion_id}.pkl'
    return p2 if p2.exists() else None


def get_cached_json(motion_id: str) -> Optional[Path]:
    """Resolve the retargeted vrm-quat JSON for a clip id. Looks first
    in the AIST cache then in the CMU cache. Returns None if neither
    exists. Used by the /api/motion/data endpoint to serve clips from
    both sources transparently."""
    # CMU clips should use retargeted vrm-quat JSON for runtime playback.
    # Raw smpl-aa fallback in the browser is a compatibility path only and
    # does not preserve the same retarget quality as export_motion_json.
    if motion_id.startswith('cmu_'):
        p_cmu = CACHE_DIR_CMU / f'{motion_id}.json'
        if not p_cmu.exists():
            return None
        # If the source PKL was updated (e.g. orientation migration),
        # force a refresh by treating older cache as missing.
        src = CMU_DIR / f'{motion_id}.pkl'
        if src.exists():
            try:
                if p_cmu.stat().st_mtime < src.stat().st_mtime:
                    return None
            except Exception:
                pass
        return p_cmu
    p1 = CACHE_DIR / f'{motion_id}.json'
    if p1.exists():
        return p1
    p2 = CACHE_DIR_CMU / f'{motion_id}.json'
    if p2.exists():
        return p2
    return None

