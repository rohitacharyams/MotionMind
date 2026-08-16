"""Motion catalog QA report.

Reads coach/motion_meta/corrections.json (baked by analyze_motions.py)
and motion_meta/*.json (clip metadata), and prints/writes a quarantine
report for clips that are likely to look broken when played back in the
browser.

Run:  python c:\\dan\\qa_motions.py
Output:
  - stdout: human-readable summary
  - c:\\dan\\coach\\motion_meta\\qa_report.json   (machine-readable)

Categories:
  HIP_BOB        in_place + !is_jump + hip_y_range > 0.20 (squat / rehab
                 artefact baked into source mocap)
  HUGE_TRAVEL    !in_place + travel_m > 5.0 (mocap drifted off-stage)
  BAD_GROUND     |ground_offset_y| > 0.50 (offline FK miscalibrated, the
                 runtime falls back to live ground-offset, so this is a
                 warn-only category — informational, not blocking).
  DEAD_FRAMES    n_frames < 30 (under 1 second of motion — too short to
                 be useful as a teachable demo)
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict

ROOT = r'c:\dan\coach'
CORR = os.path.join(ROOT, 'motion_meta', 'corrections.json')
META_DIR = os.path.join(ROOT, 'motion_meta')
OUT  = os.path.join(ROOT, 'motion_meta', 'qa_report.json')


def title_for(clip_id: str) -> str:
    p = os.path.join(META_DIR, f'{clip_id}.json')
    if not os.path.exists(p):
        return clip_id
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f).get('title') or clip_id
    except Exception:
        return clip_id


def main() -> int:
    if not os.path.exists(CORR):
        print(f'no corrections.json at {CORR}; run analyze_motions.py first',
              file=sys.stderr)
        return 1
    with open(CORR, 'r', encoding='utf-8') as f:
        corr = json.load(f)

    cats: dict[str, list] = defaultdict(list)
    for cid, m in corr.items():
        if m.get('suppress_hip_bob'):
            cats['HIP_BOB'].append((cid, m['hip_y_range'], title_for(cid)))
        if (not m.get('in_place')) and m.get('travel_m', 0) > 5.0:
            cats['HUGE_TRAVEL'].append((cid, m['travel_m'], title_for(cid)))
        if abs(m.get('ground_offset_y', 0)) > 0.50:
            cats['BAD_GROUND'].append((cid, m['ground_offset_y'],
                                       title_for(cid)))
        if m.get('n_frames', 0) < 30:
            cats['DEAD_FRAMES'].append((cid, m['n_frames'], title_for(cid)))

    # Subjects with the most HIP_BOB hits — candidates for full
    # quarantine.
    subj_hits = defaultdict(int)
    for cid, *_ in cats['HIP_BOB']:
        # e.g. cmu_105_105_41 -> 105
        parts = cid.split('_')
        if len(parts) >= 3 and parts[0] == 'cmu':
            subj_hits[parts[1]] += 1

    print('=' * 70)
    print('MOTION CATALOG QA — corrections.json')
    print('=' * 70)
    print(f'total clips           : {len(corr)}')
    print(f'HIP_BOB (squat/rehab) : {len(cats["HIP_BOB"])}')
    print(f'HUGE_TRAVEL (off-stage): {len(cats["HUGE_TRAVEL"])}')
    print(f'BAD_GROUND (info only): {len(cats["BAD_GROUND"])}')
    print(f'DEAD_FRAMES (< 1 s)   : {len(cats["DEAD_FRAMES"])}')
    print()

    if cats['HIP_BOB']:
        print('--- HIP_BOB (hard-suppressed by motion_player v15) ---')
        for cid, v, title in sorted(cats['HIP_BOB'],
                                    key=lambda x: -x[1])[:30]:
            print(f'  {cid:35s}  range={v:.2f} m   {title}')
        print()

    if cats['HUGE_TRAVEL']:
        print('--- HUGE_TRAVEL ---')
        for cid, v, title in sorted(cats['HUGE_TRAVEL'],
                                    key=lambda x: -x[1])[:15]:
            print(f'  {cid:35s}  travel={v:.2f} m  {title}')
        print()

    if cats['DEAD_FRAMES']:
        print('--- DEAD_FRAMES ---')
        for cid, v, title in cats['DEAD_FRAMES'][:15]:
            print(f'  {cid:35s}  frames={v}  {title}')
        print()

    if subj_hits:
        print('--- CMU subjects ranked by HIP_BOB count ---')
        for subj, n in sorted(subj_hits.items(), key=lambda x: -x[1]):
            print(f'  subject {subj} : {n} bad clips')
        print()

    report = {
        'totals': {
            'clips': len(corr),
            'hip_bob': len(cats['HIP_BOB']),
            'huge_travel': len(cats['HUGE_TRAVEL']),
            'bad_ground': len(cats['BAD_GROUND']),
            'dead_frames': len(cats['DEAD_FRAMES']),
        },
        'categories': {k: [{'id': cid, 'value': v, 'title': t}
                           for cid, v, t in sorted(rows)]
                       for k, rows in cats.items()},
        'cmu_subject_hip_bob_hits': dict(subj_hits),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
