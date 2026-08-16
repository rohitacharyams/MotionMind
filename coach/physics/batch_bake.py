"""batch_bake.py — run the MuJoCo PD-tracking physics bake over a set
of clips and write ``<id>_phys.json`` next to each source.

Default target = the curated warmup/cooldown pools (the clips the coach
actually plays during the stretch/warmup phases). Pass ``--all-cmu`` to
bake every retargeted CMU clip, or ``--ids a,b,c`` for a custom set.

The browser/server prefers ``<id>_phys.json`` automatically once it
exists (see motion_index.get_cached_json), so re-running this is the
whole deployment step — no code change to roll out a re-bake.

Usage:
    python -m coach.physics.batch_bake               # warmup+cooldown pools
    python -m coach.physics.batch_bake --all-cmu
    python -m coach.physics.batch_bake --ids cmu_02_02_04,cmu_07_07_01
    python -m coach.physics.batch_bake --jobs 6      # parallel workers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
COACH = os.path.dirname(HERE)
ROOT = os.path.dirname(COACH)
sys.path.insert(0, ROOT)

CMU_CACHE = os.path.join(COACH, 'motion_cache_cmu')
AIST_CACHE = os.path.join(COACH, 'motion_cache')


def _warmup_cooldown_ids() -> List[str]:
    """The curated pools the coach plays during warmup/cooldown."""
    from coach import session as cs
    ids = set()
    for name in ('_SW_UPPER', '_SW_MID', '_SW_LOWER', '_SW_DYNAMIC',
                 '_SW_BREATH'):
        ids.update(getattr(cs, name, []) or [])
    return sorted(ids)


def _all_cmu_ids() -> List[str]:
    if not os.path.isdir(CMU_CACHE):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CMU_CACHE)
        if f.endswith('.json') and not f.endswith('_phys.json'))


def _all_aist_ids() -> List[str]:
    """Every retargeted AIST dance clip (excludes phys + smoketest)."""
    if not os.path.isdir(AIST_CACHE):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(AIST_CACHE)
        if f.endswith('.json') and not f.endswith('_phys.json')
        and not f.startswith('_'))


def _all_ids() -> List[str]:
    """Every clip in both caches — dance (AIST) + basics (CMU)."""
    return _all_aist_ids() + _all_cmu_ids()


def _resolve_src(clip_id: str) -> Optional[str]:
    for d in (CMU_CACHE, AIST_CACHE):
        p = os.path.join(d, f'{clip_id}.json')
        if os.path.exists(p):
            return p
    return None


def _fk_foot_slide(ref) -> float:
    """Planted-foot horizontal slide (cm) under plain FK — exactly what
    the browser plays. A foot is 'planted' when in the lowest contact
    band; per-foot, sum its horizontal motion while it stays planted
    (a lift/re-plant resets, so steps are not counted as slide)."""
    import numpy as np
    from coach.physics.pd_bake import vrm_fk
    poss = [vrm_fk(ref, f) for f in range(ref.n_frames)]
    minz = 1e9
    for wp in poss:
        minz = min(minz, wp['LeftFoot'][2], wp['RightFoot'][2])
    band = minz + 0.05
    total = 0.0
    for foot in ('LeftFoot', 'RightFoot'):
        prev = None
        for wp in poss:
            p = wp[foot]
            if p[2] < band:
                if prev is not None:
                    total += float(np.linalg.norm(p[:2] - prev[:2]))
                prev = p
            else:
                prev = None
    return total * 100.0


def _fk_min_bodyv(ref) -> float:
    """Minimum body-vertical (Head.y − Hips.y, metres) over all frames
    under plain VRM FK — exactly the three-vrm geometry (matches the
    browser to 0.1 mm). This is GROUNDING-INVARIANT (both shift together
    with any floor offset), so it is the honest detector of postural
    COLLAPSE / forward-fold that joint-angle tracking error misses: a
    bake can track every joint to <5° yet have the root pitch so the
    head drops toward / below the hips. Healthy upright ≈ 0.30-0.40;
    a fold drives it toward 0 or negative."""
    from coach.physics.pd_bake import vrm_fk
    mn = 1e9
    for f in range(ref.n_frames):
        wp = vrm_fk(ref, f)
        mn = min(mn, float(wp['Head'][1] - wp['Hips'][1]))
    return mn


def _bake_one(clip_id: str, force: bool,
              no_deslide: bool = False) -> Dict[str, object]:
    """Worker: bake a single clip → <id>_phys.json. Returns metrics."""
    from coach.physics.pd_bake import bake_clip, Reference, validate_fk
    src = _resolve_src(clip_id)
    if src is None:
        return {'id': clip_id, 'ok': False, 'reason': 'source not found'}
    out = os.path.join(os.path.dirname(src), f'{clip_id}_phys.json')
    if os.path.exists(out) and not force:
        return {'id': clip_id, 'ok': True, 'skipped': True, 'out': out}
    t0 = time.time()
    try:
        ref = Reference(src)
        fk = validate_fk(ref)
        if fk['max_err_m'] > 0.01:        # 1 cm gate — mapping must be sane
            return {'id': clip_id, 'ok': False,
                    'reason': f"FK gate failed {fk['max_err_m']*1000:.1f}mm"}
        # raw planted-foot slide (browser-equivalent) for the slide gate
        raw_slide = _fk_foot_slide(ref)
        m = bake_clip(src, out_path=out, verbose=False, no_deslide=no_deslide)
        m['id'] = clip_id
        m['fk_max_mm'] = round(fk['max_err_m'] * 1000, 3)
        m['sec'] = round(time.time() - t0, 1)
        m['raw_slide_cm'] = round(raw_slide, 1)
        # measure the BAKED output slide under plain FK (what the browser
        # plays). The de-slide should make this << raw.
        try:
            m['baked_slide_cm'] = round(_fk_foot_slide(Reference(out)), 1)
        except Exception:
            m['baked_slide_cm'] = -1.0
        # POSTURE: min body-vertical of the baked output (collapse gate)
        try:
            m['baked_min_bodyv'] = round(_fk_min_bodyv(Reference(out)), 3)
        except Exception:
            m['baked_min_bodyv'] = -1.0
        # QUALITY GATE: never ship a bake that is WORSE than the raw clip.
        # Two independent checks — a bake must pass BOTH:
        #  (1) PENETRATION must not regress (the physics is supposed to
        #      REMOVE self-collision, never add it).
        #  (2) TRACKING FIDELITY — the baked pose must still follow the
        #      choreography. Fast/aerial dance (ballet, waacking) can
        #      exceed what stiff PD can track; if the body drifts too far
        #      from the intended move it would teach the WRONG thing, so
        #      we reject and keep the raw clip. Slow warmups pass easily.
        raw_cm = float(m.get('raw_self_pen_max_cm') or 0.0)
        post_cm = float(m.get('post_self_pen_max_cm') or 0.0)
        mean_drift = float(m.get('mean_track_err_deg') or 0.0)
        max_drift = float(m.get('max_track_err_deg') or 0.0)
        raw_slide = float(m.get('raw_slide_cm') or 0.0)
        baked_slide = float(m.get('baked_slide_cm') or 0.0)
        min_bodyv = float(m.get('baked_min_bodyv', 1.0))
        reject = None
        if post_cm > raw_cm + 1.0:
            reject = f'pen regressed {raw_cm:.1f}->{post_cm:.1f}cm'
        elif min_bodyv < 0.25:
            reject = f'postural collapse: min bodyV {min_bodyv:.2f} (<0.25)'
        elif mean_drift > 18.0:
            reject = f'pose drift mean {mean_drift:.1f}deg (>18)'
        elif max_drift > 90.0:
            reject = f'pose drift spike {max_drift:.1f}deg (>90)'
        elif (not no_deslide) and baked_slide > raw_slide + 5.0:
            # Slide gate only applies when the baked de-slide is ON. With
            # de-slide OFF, runtime foot-IK handles slide, so we do NOT
            # reject on slide here (it would discard good physics poses).
            reject = (f'slide regressed {raw_slide:.0f}->'
                      f'{baked_slide:.0f}cm')
        if reject:
            try:
                os.remove(out)
            except OSError:
                pass
            m['rejected'] = True
            m['reject_reason'] = reject + ' (kept raw)'
        return m
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return {'id': clip_id, 'ok': False, 'reason': repr(e),
                'trace': traceback.format_exc()[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default=None,
                    help='comma-separated clip ids')
    ap.add_argument('--all-cmu', action='store_true')
    ap.add_argument('--all-aist', action='store_true',
                    help='bake every retargeted AIST dance clip')
    ap.add_argument('--all', action='store_true',
                    help='bake EVERY clip in both caches (dance + basics)')
    ap.add_argument('--force', action='store_true',
                    help='re-bake even if <id>_phys.json exists')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--mass', type=float, default=70.0)
    ap.add_argument('--no-deslide', action='store_true',
                    help='skip baked de-slide; runtime foot-IK handles slide')
    args = ap.parse_args()

    if args.ids:
        ids = [s.strip() for s in args.ids.split(',') if s.strip()]
    elif args.all:
        ids = _all_ids()
    elif args.all_aist:
        ids = _all_aist_ids()
    elif args.all_cmu:
        ids = _all_cmu_ids()
    else:
        ids = _warmup_cooldown_ids()

    print(f'[batch_bake] {len(ids)} clips, jobs={args.jobs}, '
          f'force={args.force}')
    t0 = time.time()
    results: List[Dict[str, object]] = []
    if args.jobs <= 1:
        for cid in ids:
            r = _bake_one(cid, args.force, args.no_deslide)
            results.append(r)
            _print_row(r)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_bake_one, cid, args.force, args.no_deslide): cid
                    for cid in ids}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                _print_row(r)

    ok = [r for r in results if r.get('ok')]
    baked = [r for r in ok if not r.get('skipped') and not r.get('rejected')]
    rejected = [r for r in results if r.get('rejected')]
    fail = [r for r in results if not r.get('ok')]
    print('\n=== SUMMARY ===')
    print(f'  total={len(results)}  baked={len(baked)}  '
          f'skipped={len(ok) - len(baked) - len(rejected)}  '
          f'rejected={len(rejected)}  failed={len(fail)}')
    if baked:
        import statistics as st
        pen_drop = [(r["raw_self_pen_max_cm"], r["post_self_pen_max_cm"])
                    for r in baked if 'raw_self_pen_max_cm' in r]
        if pen_drop:
            raw = st.mean(a for a, _ in pen_drop)
            post = st.mean(b for _, b in pen_drop)
            print(f'  mean self-pen depth: raw {raw:.2f}cm -> '
                  f'post {post:.2f}cm')
        trk = [r['mean_track_err_deg'] for r in baked
               if 'mean_track_err_deg' in r]
        if trk:
            print(f'  mean tracking error: {st.mean(trk):.1f}deg')
    for r in fail:
        print(f'  FAIL {r.get("id")}: {r.get("reason")}')
    for r in rejected:
        print(f'  REJECT {r.get("id")}: {r.get("reject_reason")}')
    print(f'  wall time: {time.time() - t0:.1f}s')

    # write a manifest for the deploy step
    man = os.path.join(HERE, 'bake_manifest.json')
    with open(man, 'w', encoding='utf-8') as f:
        json.dump({'baked': [r['id'] for r in baked],
                   'skipped': [r['id'] for r in ok
                               if r.get('skipped')],
                   'rejected': [r['id'] for r in rejected],
                   'failed': [r['id'] for r in fail],
                   'results': results}, f, indent=2)
    print(f'  manifest → {man}')


def _print_row(r: Dict[str, object]) -> None:
    if not r.get('ok'):
        print(f'  [FAIL] {r.get("id"):16s} {r.get("reason")}')
    elif r.get('skipped'):
        print(f'  [skip] {r.get("id"):16s} (exists)')
    elif r.get('rejected'):
        print(f'  [REJECT] {r.get("id"):16s} {r.get("reject_reason")}')
    else:
        print(f'  [ok] {r.get("id"):16s} '
              f'pen {r.get("raw_self_pen_max_cm")}->'
              f'{r.get("post_self_pen_max_cm")}cm  '
              f'trk {r.get("mean_track_err_deg")} '
              f'{r.get("sec")}s')


if __name__ == '__main__':
    main()
