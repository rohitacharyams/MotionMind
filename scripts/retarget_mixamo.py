"""retarget_mixamo.py - Retarget Mixamo actions onto our characters.

Standard constraint+bake workflow:

  1. Open the library .blend (contains all mx_* actions).
  2. Import target character (VRM/GLB) -> target armature in T-pose.
  3. Import a Mixamo FBX as the source armature carrying one action.
  4. Detect the target rig style (Mixamo-named vs J_Bip_*) -> pick the
     appropriate bone alias map.
  5. For each (target_bone, source_bone) pair: add COPY_ROTATION
     (target_space='WORLD', owner_space='WORLD').
     For the hips bone: also add COPY_LOCATION with z-axis offset to
     plant the feet.
  6. Bake target armature visual -> new Action <char>_<slug>.
  7. Save into a per-character library .blend.

Run:
  blender --background --python retarget_mixamo.py -- \
      --char_name alicia \
      --char_path c:\\dan\\data\\models\\extra\\AliciaSolid.vrm \
      --src_dir c:\\dan\\data\\motion_raw\\mixamo \
      --out_blend c:\\dan\\data\\motion_library\\chars\\alicia.blend \
      --clips walking idle hip_hop_dancing pointing
      [--all]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy

# ---------------------------------------------------------------------------
# BONE ALIAS MAPS
# ---------------------------------------------------------------------------
# Standard Mixamo bone names (always with the "mixamorig:" prefix from FBX)
MIXAMO_BONES = {
    "hips":          "Hips",
    "spine":         "Spine",
    "spine1":        "Spine1",
    "spine2":        "Spine2",
    "neck":          "Neck",
    "head":          "Head",
    "l_shoulder":    "LeftShoulder",
    "l_upperarm":    "LeftArm",
    "l_lowerarm":    "LeftForeArm",
    "l_hand":        "LeftHand",
    "r_shoulder":    "RightShoulder",
    "r_upperarm":    "RightArm",
    "r_lowerarm":    "RightForeArm",
    "r_hand":        "RightHand",
    "l_upperleg":    "LeftUpLeg",
    "l_lowerleg":    "LeftLeg",
    "l_foot":        "LeftFoot",
    "l_toe":         "LeftToeBase",
    "r_upperleg":    "RightUpLeg",
    "r_lowerleg":    "RightLeg",
    "r_foot":        "RightFoot",
    "r_toe":         "RightToeBase",
}

# Same logical bones, named for VRoid-style rigs (Sample_A/C/G/K/P/Z)
VROID_BONES = {
    "hips":          "J_Bip_C_Hips",
    "spine":         "J_Bip_C_Spine",
    "spine1":        "J_Bip_C_Chest",
    "spine2":        "J_Bip_C_UpperChest",
    "neck":          "J_Bip_C_Neck",
    "head":          "J_Bip_C_Head",
    "l_shoulder":    "J_Bip_L_Shoulder",
    "l_upperarm":    "J_Bip_L_UpperArm",
    "l_lowerarm":    "J_Bip_L_LowerArm",
    "l_hand":        "J_Bip_L_Hand",
    "r_shoulder":    "J_Bip_R_Shoulder",
    "r_upperarm":    "J_Bip_R_UpperArm",
    "r_lowerarm":    "J_Bip_R_LowerArm",
    "r_hand":        "J_Bip_R_Hand",
    "l_upperleg":    "J_Bip_L_UpperLeg",
    "l_lowerleg":    "J_Bip_L_LowerLeg",
    "l_foot":        "J_Bip_L_Foot",
    "l_toe":         "J_Bip_L_ToeBase",
    "r_upperleg":    "J_Bip_R_UpperLeg",
    "r_lowerleg":    "J_Bip_R_LowerLeg",
    "r_foot":        "J_Bip_R_Foot",
    "r_toe":         "J_Bip_R_ToeBase",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _slug(s: str) -> str:
    s = Path(s).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0


def _import_char(path: Path):
    """Import VRM or GLB. Returns its armature object."""
    if path.suffix.lower() == ".vrm":
        try:
            bpy.ops.import_scene.vrm(filepath=str(path))
        except Exception:
            bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"No armature in {path.name}")
    return arms[-1]


def _detect_rig_style(arm) -> dict:
    """Pick the correct alias map by sniffing one bone."""
    names = {b.name for b in arm.pose.bones}
    if "J_Bip_C_Hips" in names:
        return VROID_BONES
    if "Hips" in names:
        return MIXAMO_BONES
    raise RuntimeError(f"Unknown rig style; bones include: {sorted(names)[:8]}")


def _import_mixamo_src(fbx_path: Path):
    """Import a Mixamo FBX and return (armature, action). The armature
    carries the action via animation_data."""
    objs_before = set(bpy.data.objects.keys())
    acts_before = set(bpy.data.actions.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        automatic_bone_orientation=False,
        ignore_leaf_bones=True,
        use_anim=True,
    )
    new_objs = [bpy.data.objects[n] for n in
                (set(bpy.data.objects.keys()) - objs_before)]
    new_acts = [bpy.data.actions[n] for n in
                (set(bpy.data.actions.keys()) - acts_before)]
    arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
    action = new_acts[0] if new_acts else None
    return arm, action, new_objs


def _delete_objs(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        try:
            o.select_set(True)
        except ReferenceError:
            pass
    if any(o for o in objs if o.name in bpy.data.objects):
        bpy.ops.object.delete(use_global=False)


def _world_z_of_bone(arm, bone_name: str) -> float:
    """World-space Z height of a bone's head in current pose."""
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return 0.0
    return float((arm.matrix_world @ pb.head).z)


def _add_constraints(target_arm, source_arm, alias_map: dict,
                     mixamo_prefix: str = "mixamorig:",
                     copy_hip_loc: bool = False) -> int:
    """Add COPY_ROTATION (world) on every mapped bone. Optionally also
    add COPY_LOCATION on hips (only useful for locomotion clips)."""
    n = 0
    for logical, tgt_name in alias_map.items():
        src_name = mixamo_prefix + MIXAMO_BONES[logical]
        tpb = target_arm.pose.bones.get(tgt_name)
        spb = source_arm.pose.bones.get(src_name)
        if tpb is None or spb is None:
            continue

        # Copy rotation as a *delta from rest pose*. LOCAL/LOCAL is the
        # robust choice when source & target share the same bone-axis
        # convention (Mixamo's Y-down-the-bone). It avoids fighting the
        # VRM rig's quirky world-rest orientation.
        cr = tpb.constraints.new("COPY_ROTATION")
        cr.target = source_arm
        cr.subtarget = src_name
        cr.target_space = "LOCAL"
        cr.owner_space = "LOCAL"
        n += 1

        if logical == "hips" and copy_hip_loc:
            cl = tpb.constraints.new("COPY_LOCATION")
            cl.target = source_arm
            cl.subtarget = src_name
            cl.target_space = "WORLD"
            cl.owner_space = "WORLD"
            cl.use_offset = True
            n += 1
    return n


def _clear_constraints(arm):
    for pb in arm.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)


def _bake_action(target_arm, frame_start: int, frame_end: int,
                 new_name: str) -> bpy.types.Action:
    """Bake visual transforms of the constrained target armature into a
    fresh Action.  Then strip constraints and rename the action."""
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")

    # nuke any existing action so the bake creates a fresh one
    if target_arm.animation_data and target_arm.animation_data.action:
        target_arm.animation_data.action = None

    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=False,
        bake_types={"POSE"},
    )

    bpy.ops.object.mode_set(mode="OBJECT")

    new_action = target_arm.animation_data.action
    new_action.name = new_name
    new_action.use_fake_user = True
    return new_action


def _action_frame_range(action) -> tuple[int, int]:
    s, e = action.frame_range
    return int(round(s)), int(round(e))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char_name", required=True)
    ap.add_argument("--char_path", required=True)
    ap.add_argument("--src_dir", required=True,
                    help="Directory of Mixamo .fbx files.")
    ap.add_argument("--out_blend", required=True,
                    help="Output per-character library .blend.")
    ap.add_argument("--manifest_in", default=None,
                    help="manifest.json from mixamo_import (for categories).")
    ap.add_argument("--manifest_out", default=None,
                    help="Per-character manifest output JSON.")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="Restrict to these clip slugs (default: all).")
    ap.add_argument("--all", action="store_true",
                    help="Retarget every FBX in --src_dir.")
    ap.add_argument("--hip_loc", action="store_true",
                    help="Bake hip translation for ALL clips.")
    ap.add_argument("--locomotion_clips", nargs="*", default=None,
                    help="Slugs that should bake hip translation "
                         "(applied per-clip; overrides --hip_loc).")
    args = ap.parse_args(_argv())

    src_dir = Path(args.src_dir)
    out_blend = Path(args.out_blend)
    out_blend.parent.mkdir(parents=True, exist_ok=True)

    fbx_files = sorted(src_dir.glob("*.fbx"))
    if args.clips and not args.all:
        wanted = set(args.clips)
        fbx_files = [f for f in fbx_files if _slug(f.name) in wanted]
    if not fbx_files:
        sys.exit("[!] no FBX clips selected")

    print(f"\n[retarget] character : {args.char_name}")
    print(f"[retarget] char file  : {args.char_path}")
    print(f"[retarget] clips      : {len(fbx_files)}")
    print(f"[retarget] out        : {out_blend}\n")

    # 1. Fresh scene with the target character imported once
    _reset()
    char_path = Path(args.char_path)
    target_arm = _import_char(char_path)
    target_arm.name = f"ARM_{args.char_name}"
    print(f"[retarget] imported target arm: {target_arm.name} "
          f"({len(target_arm.pose.bones)} bones)")

    alias_map = _detect_rig_style(target_arm)
    style = "vroid" if alias_map is VROID_BONES else "mixamo"
    print(f"[retarget] rig style  : {style}")

    # Sanity: report which logical bones we successfully resolve
    resolved = [k for k, v in alias_map.items()
                if v in target_arm.pose.bones]
    missing = [k for k in alias_map if k not in resolved]
    print(f"[retarget] bones ok   : {len(resolved)}/{len(alias_map)}")
    if missing:
        print(f"[retarget] MISSING    : {missing}")

    results = []

    for i, fbx in enumerate(fbx_files, 1):
        slug = _slug(fbx.name)
        action_name = f"{args.char_name}_{slug}"
        print(f"\n[{i:02d}/{len(fbx_files)}] {slug} -> {action_name}")

        # Snapshot to know what to delete after
        objs_before = set(bpy.data.objects.keys())

        # 2. Import source FBX (gives us src arm + action)
        src_arm, src_action, new_objs = _import_mixamo_src(fbx)
        if src_arm is None or src_action is None:
            print("  [!] import failed - skipped")
            continue

        # IMPORTANT: do NOT override src_arm.rotation_euler -- the FBX
        # importer applies the Y-up -> Z-up conversion as an object rotation,
        # and we need that for WORLD-space constraints to read correctly.

        # The Mixamo importer stripped 'mixamorig:' prefix? Detect.
        sample_bone = next(iter(src_arm.pose.bones)).name
        prefix = "mixamorig:" if sample_bone.startswith("mixamorig:") else ""

        f_start, f_end = _action_frame_range(src_action)

        # 3. Add constraints
        loco_set = set(args.locomotion_clips or [])
        is_loco = args.hip_loc or (slug in loco_set)
        n_constraints = _add_constraints(target_arm, src_arm, alias_map,
                                         mixamo_prefix=prefix,
                                         copy_hip_loc=is_loco)

        # 4. Bake
        try:
            new_action = _bake_action(target_arm, f_start, f_end, action_name)
            n_frames = f_end - f_start + 1
            print(f"  ok  baked {n_frames}f -> {new_action.name}"
                  f"  hip_loc={is_loco}")
            results.append({
                "clip": slug,
                "action": new_action.name,
                "n_frames": n_frames,
                "duration_s": round(n_frames / 30.0, 3),
                "hip_loc": is_loco,
            })
        except Exception as e:
            print(f"  [!] bake failed: {e}")
            _clear_constraints(target_arm)
            results.append({"clip": slug, "error": str(e)})

        # 5. Cleanup: delete imported source armature/meshes; orphan source
        # action stays referenced as fake_user? It's not needed; remove it
        # from data so the per-char blend stays clean.
        new_now = [bpy.data.objects[n] for n in
                   (set(bpy.data.objects.keys()) - objs_before)
                   if n in bpy.data.objects]
        _delete_objs(new_now)

        # Wipe orphan datablocks (source armatures, meshes, source action)
        for coll, attr_user in (
            (bpy.data.armatures, "users"),
            (bpy.data.meshes, "users"),
            (bpy.data.materials, "users"),
            (bpy.data.images, "users"),
            (bpy.data.actions, "users"),
        ):
            for db in list(coll):
                if getattr(db, attr_user) == 0 and db.name != action_name:
                    coll.remove(db, do_unlink=True)

    # Save the per-character library (only baked actions remain, all
    # fake-user'd). Keep the target armature too for spot-check renders.
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"\n[retarget] saved: {out_blend}")

    if args.manifest_out:
        manifest = {
            "schema": 1,
            "character": args.char_name,
            "char_path": str(char_path),
            "rig_style": style,
            "library_blend": str(out_blend),
            "fps": 30,
            "clips": results,
            "n_clips_ok": sum(1 for r in results if "error" not in r),
            "n_clips_failed": sum(1 for r in results if "error" in r),
        }
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[retarget] manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
