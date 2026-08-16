"""inspect_vrm_bones.py — Print bone names for a VRM/GLB armature.

Usage:
  & 'C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe' `
      --background --factory-startup `
      --python c:\\dan\\scripts\\inspect_vrm_bones.py -- `
      --char c:\\dan\\data\\models\\extra\\AliciaSolid.vrm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True)
    args = ap.parse_args(_argv())

    bpy.ops.wm.read_factory_settings(use_empty=True)

    p = Path(args.char)
    print(f"\n[inspect] {p.name}")
    if p.suffix.lower() == ".vrm":
        try:
            bpy.ops.import_scene.vrm(filepath=str(p))
        except Exception as e:
            print(f"  [!] vrm import failed ({e}); trying glb")
            bpy.ops.import_scene.gltf(filepath=str(p))
    else:
        bpy.ops.import_scene.gltf(filepath=str(p))

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        sys.exit("[!] no armature")
    arm = arms[0]
    print(f"\n[inspect] armature: {arm.name}  ({len(arm.pose.bones)} bones)\n")
    print("name | parent | head_local")
    print("-" * 72)
    for pb in arm.pose.bones:
        parent = pb.parent.name if pb.parent else "-"
        h = pb.bone.head_local
        print(f"{pb.name:36s} | {parent:30s} | "
              f"({h.x:+.2f},{h.y:+.2f},{h.z:+.2f})")


if __name__ == "__main__":
    main()
