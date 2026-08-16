"""Fix quaternion hemisphere discontinuities across the motion cache.

For each clip, every bone's per-frame quaternions are made continuous:
if dot(q_prev, q_cur) < 0, flip the sign of q_cur. This does NOT change
the represented rotation (q and -q are the same orientation) but
guarantees three.js slerp picks the short path → no 140° single-frame
'jitter' jumps.

Idempotent. Writes in place. Backs up originals to *.json.bak on first run.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
CACHES = [
    ROOT / 'coach' / 'motion_cache',
    ROOT / 'coach' / 'motion_cache_cmu',
]


def fix_clip(path: Path) -> dict:
    bak = path.with_suffix('.json.bak')
    if not bak.exists():
        shutil.copy2(path, bak)

    data = json.loads(path.read_text(encoding='utf-8'))
    rots = data.get('rotations', {})
    if not rots:
        return {'clip': path.stem, 'flipped': 0, 'bones': 0}

    flipped = 0
    for bone, frames in rots.items():
        for i in range(1, len(frames)):
            q_prev = frames[i - 1]
            q_cur = frames[i]
            if len(q_prev) != 4 or len(q_cur) != 4:
                continue
            d = sum(a * b for a, b in zip(q_prev, q_cur))
            if d < 0.0:
                frames[i] = [-q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]]
                flipped += 1

    path.write_text(json.dumps(data))
    return {'clip': path.stem, 'flipped': flipped, 'bones': len(rots)}


def main():
    total = 0
    touched = 0
    for cache in CACHES:
        if not cache.exists():
            continue
        for f in sorted(cache.glob('*.json')):
            r = fix_clip(f)
            total += r['flipped']
            if r['flipped'] > 0:
                touched += 1
                print(f"{r['clip']}: flipped {r['flipped']} frames across {r['bones']} bones")
    print(f'\nDONE: {total} sign flips across {touched} clips')


if __name__ == '__main__':
    main()
