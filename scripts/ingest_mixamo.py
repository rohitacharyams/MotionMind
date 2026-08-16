"""ingest_mixamo.py — Blender: Mixamo FBX -> VRM-bone-local quaternion JSON.

Run:
  blender --background --python ingest_mixamo.py -- \
      --src c:\\dan\\mixamo_drop --out c:\\dan\\coach\\motion_cache

Output per file: motion_cache/mixamo_<slug>.json in the SAME schema the browser
MotionPlayer eats (rotations{bone:[[x,y,z,w]...]}, hips_translation, fps,
n_frames, rest_local_rotation, format:'vrm-quat').

IMPORTANT: this is a FIRST-CUT retarget. Mixamo bone-local axes may differ from
the VRM normalized-bone axes, so after ingesting the FIRST file we VERIFY it on
the live avatar (head-above-hips / feet-on-floor) and, if any bone is rotated
wrong, add a per-bone axis correction here — exactly the empirical method used to
diagnose the CMU up-axis bug. Do NOT assume it's correct until browser-verified.

Bone map: Mixamo names (minus 'mixamorig:') already match our format's source
bone names, so mapping is a straight prefix strip.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import bpy
import mathutils

# Source bone names the format/player expect (see motion_player VRM_BONE_MAP keys).
WANT = ['Hips', 'Spine', 'Spine2', 'Neck', 'Head',
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot',
        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
        'RightUpLeg', 'RightLeg', 'RightFoot']
# Mixamo also has 'Spine1'/'Spine2'; our 'Spine2' target is the chest. Prefer
# Spine2 if present, else Spine1.
ALIASES = {'Spine2': ['Spine2', 'Spine1']}

# Target (VRM normalized) parent chain, in WANT-name space. Used by the
# world-delta retarget: a bone's LOCAL quat = parent_worldDelta^-1 * own_worldDelta.
# Unmapped VRM intermediates (e.g. upperChest) stay at rest (identity world
# rotation) so the nearest MAPPED ancestor is the correct world parent.
PARENT = {
    'Hips': None,
    'Spine': 'Hips',
    'Spine2': 'Spine',
    'Neck': 'Spine2',
    'Head': 'Neck',
    'LeftShoulder': 'Spine2',
    'LeftArm': 'LeftShoulder',
    'LeftForeArm': 'LeftArm',
    'LeftHand': 'LeftForeArm',
    'LeftUpLeg': 'Hips',
    'LeftLeg': 'LeftUpLeg',
    'LeftFoot': 'LeftLeg',
    'RightShoulder': 'Spine2',
    'RightArm': 'RightShoulder',
    'RightForeArm': 'RightArm',
    'RightHand': 'RightForeArm',
    'RightUpLeg': 'Hips',
    'RightLeg': 'RightUpLeg',
    'RightFoot': 'RightLeg',
}

# Basis change Blender(Z-up) -> three.js/VRM(Y-up): Rx(-90deg). A rotation
# expressed in Blender world is re-expressed in three world by conjugation
# q3 = C * q * C^-1. mathutils.Quaternion is (w, x, y, z).
_C = mathutils.Quaternion((0.7071067811865476, -0.7071067811865476, 0.0, 0.0))
_CINV = _C.inverted()

# Clips that are DELIBERATE forward folds (spine goes near-horizontal on
# purpose). Tagged pose_profile:'fold' so the browser inversion guard
# tolerates the fold instead of swapping to the idle clip.
FOLD_CLIPS = {'mixamo_reaching_down', 'mixamo_warming_up'}

# Clips performed ON THE GROUND (plank / push-up / sit-up). Deliberately
# horizontal (cosTilt ~0), so tagged pose_profile:'floor' so the browser
# inversion guard doesn't swap them to the standing idle groove.
FLOOR_CLIPS = {
    'mixamo_plank', 'mixamo_start_plank', 'mixamo_end_plank',
    'mixamo_push_up', 'mixamo_idle_to_push_up', 'mixamo_jump_push_up',
    'mixamo_start_bicycle_sit_up', 'mixamo_end_bicycle_sit_up',
}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def slugify(name: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return f'mixamo_{s}'


def find_armature():
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            return o
    return None


def strip(n: str) -> str:
    return n.split(':')[-1]


def ingest(fbx: Path, out_dir: Path) -> dict:
    reset()
    # automatic_bone_orientation=True is ESSENTIAL: these FBX are downloaded
    # WITHOUT skin, so there is no bind mesh to anchor the rest pose. Blender's
    # DEFAULT import then collapses every bone to point +Z (arms up, legs up),
    # producing a bogus rest. The world-delta retarget measures rotations from
    # that rest, so a bogus rest sends arms overhead. With auto orientation
    # Blender points each bone at its child, recovering the true Mixamo T-pose
    # (arms out +X, legs down -Z) which MATCHES the VRM normalized rest — the
    # required precondition for the delta retarget to be correct.
    bpy.ops.import_scene.fbx(filepath=str(fbx), automatic_bone_orientation=True)
    arm = find_armature()
    if arm is None:
        return {'file': fbx.name, 'error': 'no armature'}
    scene = bpy.context.scene
    fps = scene.render.fps or 30
    # Use the imported ACTION's true frame range, not Blender's default scene
    # range (which is 1-250 for every file and would truncate/pad clips). FBX
    # import attaches the animation as an action on the armature; its
    # frame_range is the real clip length.
    f0, f1 = scene.frame_start, scene.frame_end
    act = None
    ad = arm.animation_data
    if ad and ad.action:
        act = ad.action
    if act is None:
        # fall back to any action in the file
        acts = list(bpy.data.actions)
        if acts:
            act = acts[0]
    if act is not None:
        fr = act.frame_range
        f0, f1 = int(round(fr[0])), int(round(fr[1]))
    n = max(0, f1 - f0 + 1)

    # Build name -> pose bone (stripped Mixamo prefix).
    pbones = {strip(pb.name): pb for pb in arm.pose.bones}

    def resolve(want):
        for cand in ALIASES.get(want, [want]):
            if cand in pbones:
                return pbones[cand]
        return None

    mw = arm.matrix_world

    def world_rot(pb):
        """Pose bone's WORLD-space rotation quaternion (Blender frame)."""
        return (mw @ pb.matrix).to_quaternion().normalized()

    def rest_world_rot(pb):
        """Pose bone's REST WORLD-space rotation quaternion (Blender frame)."""
        return (mw @ pb.bone.matrix_local).to_quaternion().normalized()

    # Pre-resolve bones + rest world rotations (rest is frame-independent).
    resolved = {w: resolve(w) for w in WANT}
    rest_w = {}
    for w in WANT:
        pb = resolved[w]
        rest_w[w] = rest_world_rot(pb) if pb is not None else mathutils.Quaternion()

    rot = {w: [] for w in WANT}
    hips_t = []
    hips_pb = resolved['Hips']
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        # WORLD-DELTA RETARGET (robust; ignores per-bone local-axis mismatch
        # between Mixamo's bone-aligned frames and VRM's world-aligned
        # normalized bones — the root cause of the old "arms stuck overhead"
        # bug). Steps per bone:
        #   D = Sanim_world * Srest_world^-1      (rotation from rest, Blender world)
        #   D3 = C * D * C^-1                     (re-express in three/VRM world)
        # Then LOCAL quat for the normalized bone = D3[parent]^-1 * D3[bone]
        # (target rest world = identity, so world anim == D3).
        d3 = {}
        for w in WANT:
            pb = resolved[w]
            if pb is None:
                d3[w] = mathutils.Quaternion()
                continue
            D = world_rot(pb) @ rest_w[w].inverted()
            d3[w] = (_C @ D @ _CINV).normalized()
        for w in WANT:
            p = PARENT[w]
            if p is None:
                qloc = d3[w]
            else:
                qloc = d3[p].inverted() @ d3[w]
            qloc.normalize()
            # v4: FACING RECONCILIATION. The C=Rx(-90) conversion yields data in
            # a +Z-facing three.js frame, but this VRM's normalized humanoid uses
            # the VRM -Z-facing convention. The net correction is a 180 deg yaw,
            # which for a local quat is a conjugation by Ry(180) == negate the x
            # and z components. Verified LIVE on the rig (arms-overhead bug -> arms
            # hang naturally at the sides). Without this every arm-centric clip
            # renders with the arms flipped up.
            rot[w].append([-qloc.x, qloc.y, -qloc.z, qloc.w])
        if hips_pb is not None:
            # world head position of the hips bone, in metres
            m = mw @ hips_pb.matrix
            loc = m.to_translation()
            hips_t.append([loc.x, loc.z, loc.y])   # Blender Z-up -> Y-up
        else:
            hips_t.append([0, 0.9, 0])

    out = {
        'id': slugify(fbx.stem),
        'source': 'mixamo',
        'fps': int(fps),
        'n_frames': n,
        'format': 'vrm-quat',
        'rotations': rot,
        'hips_translation': hips_t,
        'rest_local_rotation': {w: [0, 0, 0, 1] for w in WANT},
        'corrections': {'mixamo_ingest': 'v4_ry180'},
    }
    # Deliberate forward-fold stretches (toe-touch, standing hamstring
    # reach) bend the spine toward horizontal on purpose. The browser's
    # upright-companion guard treats that as "lying flat" and swaps the
    # clip away, so flag them so the guard tolerates the fold (coach.js
    # v152) while still catching true inversion / floor-sink.
    if out['id'] in FOLD_CLIPS:
        out['pose_profile'] = 'fold'
    elif out['id'] in FLOOR_CLIPS:
        out['pose_profile'] = 'floor'
    out_path = out_dir / f'{out["id"]}.json'
    out_path.write_text(json.dumps(out), encoding='utf-8')
    return {'file': fbx.name, 'id': out['id'], 'frames': n, 'fps': int(fps),
            'out': str(out_path)}


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    src = Path(a.src)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('*.fbx'))
    report = []
    for i, f in enumerate(files, 1):
        try:
            r = ingest(f, out_dir)
        except Exception as e:                                   # noqa: BLE001
            r = {'file': f.name, 'error': str(e)}
        report.append(r)
        print(f'[{i}/{len(files)}] {f.name} -> {r.get("id", "FAIL")} '
              f'({r.get("frames", "?")}f)')
    (src / '_ingest_report.json').write_text(json.dumps(report, indent=2))
    print('done; report ->', src / '_ingest_report.json')


if __name__ == '__main__':
    main()
