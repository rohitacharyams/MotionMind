"""Blender batch script: convert every motion_cache JSON clip to a .blend file.

Each output .blend contains:
  - an Armature 'Skeleton' built from rest_local_translation/rotation
  - a baked Action 'Motion' with per-frame quaternion keys on every bone
  - per-frame Hips location keys from hips_translation

Run:
  blender --background --python export_motions_to_blend.py -- \
      --src coach/motion_cache_cmu --out blend_exports/cmu

For ALL clips in both caches, use _export_all_blends.cmd instead.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_armature(rest_t: dict, rest_r: dict, parent_map: dict, name: str = 'Skeleton'):
    """Build an Armature from rest pose. parent_map: child_bone -> parent_bone."""
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Resolve world positions for each bone by walking the parent chain.
    world_pos: dict[str, mathutils.Vector] = {}
    def resolve(bone: str) -> mathutils.Vector:
        if bone in world_pos:
            return world_pos[bone]
        local = mathutils.Vector(rest_t.get(bone, (0, 0, 0)))
        parent = parent_map.get(bone)
        if parent and parent in rest_t:
            world_pos[bone] = resolve(parent) + local
        else:
            world_pos[bone] = local
        return world_pos[bone]

    edit_bones = arm_data.edit_bones
    created: dict[str, object] = {}
    for bone in rest_t.keys():
        head = resolve(bone)
        # Tail = head + small offset toward first child, else +Y
        children = [b for b, p in parent_map.items() if p == bone]
        if children:
            tail = resolve(children[0])
            if (tail - head).length < 1e-4:
                tail = head + mathutils.Vector((0, 0.05, 0))
        else:
            tail = head + mathutils.Vector((0, 0.05, 0))
        eb = edit_bones.new(bone)
        eb.head = head
        eb.tail = tail
        created[bone] = eb

    # Wire up parents.
    for bone, parent in parent_map.items():
        if bone in created and parent in created:
            created[bone].parent = created[parent]

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def bake_action(arm_obj, rotations: dict, hips_t: list, rest_r: dict, fps: int):
    """Key per-frame quaternion rotation + Hips location."""
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    n = len(hips_t) if hips_t else max((len(v) for v in rotations.values()), default=0)
    scene.frame_end = n

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    arm_obj.animation_data_create()
    action = bpy.data.actions.new('Motion')
    arm_obj.animation_data.action = action

    pose_bones = {pb.name: pb for pb in arm_obj.pose.bones}
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    for f in range(n):
        scene.frame_set(f + 1)
        # Per-bone rotations: track values are [x,y,z,w] (glTF order).
        # Blender quaternion is (w,x,y,z).
        for bone, frames in rotations.items():
            if bone not in pose_bones or f >= len(frames):
                continue
            q = frames[f]
            if len(q) != 4:
                continue
            pb = pose_bones[bone]
            # Anim rotations REPLACE bind. Blender pose-bone rotation is
            # relative to rest. Convert: pose_q = bind^-1 * anim_q.
            bind_xyzw = rest_r.get(bone, (0, 0, 0, 1))
            bind = mathutils.Quaternion((bind_xyzw[3], bind_xyzw[0], bind_xyzw[1], bind_xyzw[2]))
            anim = mathutils.Quaternion((q[3], q[0], q[1], q[2]))
            pb.rotation_quaternion = bind.inverted() @ anim
            pb.keyframe_insert(data_path='rotation_quaternion', frame=f + 1)
        # Hips world translation -> Hips pose-bone location (object-space).
        if hips_t and f < len(hips_t) and 'Hips' in pose_bones:
            hp = pose_bones['Hips']
            tx, ty, tz = hips_t[f]
            hp.location = mathutils.Vector((tx, ty, tz))
            hp.keyframe_insert(data_path='location', frame=f + 1)

    bpy.ops.object.mode_set(mode='OBJECT')
    return action


def infer_parents(skeleton: list | None) -> dict[str, str]:
    """skeleton entries look like {name, parent, ...}. Fall back to standard
    humanoid hierarchy if not provided."""
    if skeleton:
        out = {}
        for s in skeleton:
            if isinstance(s, dict) and s.get('parent'):
                out[s['name']] = s['parent']
        if out:
            return out
    # Standard humanoid fallback.
    return {
        'Spine': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1',
        'Neck': 'Spine2', 'Head': 'Neck',
        'LeftShoulder': 'Spine2', 'LeftArm': 'LeftShoulder',
        'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm',
        'RightShoulder': 'Spine2', 'RightArm': 'RightShoulder',
        'RightForeArm': 'RightArm', 'RightHand': 'RightForeArm',
        'LeftUpLeg': 'Hips', 'LeftLeg': 'LeftUpLeg', 'LeftFoot': 'LeftLeg', 'LeftToeBase': 'LeftFoot',
        'RightUpLeg': 'Hips', 'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg', 'RightToeBase': 'RightFoot',
    }


def export_one(json_path: Path, out_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding='utf-8'))
    rest_t = data.get('rest_local_translation', {})
    rest_r = data.get('rest_local_rotation', {})
    rotations = data.get('rotations', {})
    hips_t = data.get('hips_translation') or []
    fps = int(round(data.get('fps') or 30))
    parent_map = infer_parents(data.get('skeleton'))

    reset_scene()
    arm = build_armature(rest_t, rest_r, parent_map, name=json_path.stem)
    bake_action(arm, rotations, hips_t, rest_r, fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path.resolve()))
    return {
        'clip': json_path.stem,
        'bones': len(rotations),
        'frames': len(hips_t),
        'fps': fps,
        'out': str(out_path),
    }


def main():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--only', default='', help='comma list of clip stems to export')
    args = p.parse_args(argv)

    src = Path(args.src)
    out_dir = Path(args.out)
    only = {s.strip() for s in args.only.split(',') if s.strip()}
    files = sorted(src.glob('*.json'))
    if only:
        files = [f for f in files if f.stem in only]
    if args.limit:
        files = files[:args.limit]

    report = []
    for i, f in enumerate(files, 1):
        try:
            r = export_one(f, out_dir / f'{f.stem}.blend')
            report.append(r)
            print(f'[{i}/{len(files)}] {f.stem} -> {r["out"]} ({r["frames"]}f)')
        except Exception as e:
            print(f'[{i}/{len(files)}] {f.stem} FAIL: {e}')
            report.append({'clip': f.stem, 'error': str(e)})

    (out_dir / '_export_report.json').write_text(json.dumps(report, indent=2))
    print(f'wrote {out_dir / "_export_report.json"}')


if __name__ == '__main__':
    main()
