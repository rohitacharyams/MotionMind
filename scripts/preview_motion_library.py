"""preview_motion_library.py — Render filmstrip thumbnails for every Mixamo clip.

For each .fbx in --src, open it in a fresh Blender scene, render N evenly-
spaced frames at low resolution using the Workbench engine (fast, no
raytracing), and stitch them into a single horizontal filmstrip PNG.

Then write an HTML contact sheet that displays all filmstrips with their
metadata, for fast visual review.

Usage (PowerShell):
  & 'C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe' `
      --background --factory-startup `
      --python c:\\dan\\scripts\\preview_motion_library.py -- `
      --src c:\\dan\\data\\motion_raw\\mixamo `
      --manifest c:\\dan\\data\\motion_library\\manifest.json `
      --out c:\\dan\\data\\motion_library\\previews
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy

W, H = 240, 320               # per-frame thumbnail size
N_THUMBS = 6                  # number of frames per filmstrip


def _slug(name: str) -> str:
    s = Path(name).stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _argv_after_dd() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _setup_render(out_dir: Path) -> None:
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.film_transparent = True
    # workbench shading: solid + matcap
    s = sc.display.shading
    s.light = "MATCAP"
    s.studio_light = "basic_grey.exr"
    s.color_type = "SINGLE"
    s.single_color = (0.55, 0.6, 0.7)
    s.show_shadows = True


def _setup_camera_for(arm: bpy.types.Object) -> bpy.types.Object:
    """Frame a 3/4 view on the armature's bounding box."""
    # estimate height from edit bones rest pose -> use pose head positions
    bpy.context.view_layer.update()
    # Place camera at +Y -X side, head-height target
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Target empty at hips height
    target = bpy.data.objects.new("PreviewTarget", None)
    bpy.context.collection.objects.link(target)
    target.location = (arm.location.x, arm.location.y, arm.location.z + 1.0)

    cam.location = (arm.location.x + 2.6,
                    arm.location.y - 3.4,
                    arm.location.z + 1.6)
    # Track to target
    tc = cam.constraints.new(type="TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    return cam


def _import_fbx(fbx_path: Path):
    objs_before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        automatic_bone_orientation=False,
        ignore_leaf_bones=True,
        use_anim=True,
    )
    new_objs = [bpy.data.objects[n] for n in
                (set(bpy.data.objects.keys()) - objs_before)]
    arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
    return arm, new_objs


def _frame_range_of(arm: bpy.types.Object) -> tuple[int, int]:
    ad = arm.animation_data
    if ad and ad.action:
        s, e = ad.action.frame_range
        return int(round(s)), int(round(e))
    return 1, 60


def _render_filmstrip(fbx_path: Path, frames_dir: Path) -> tuple[int, int]:
    """Render N evenly-spaced thumbnails into frames_dir/fNN.png.

    Returns (n_frames_total_in_clip, n_thumbs_rendered).
    Stitching is done by stitch_filmstrips.py in system Python (PIL).
    """
    _reset_scene()
    _setup_render(frames_dir)
    arm, _ = _import_fbx(fbx_path)
    if arm is None:
        return 0, 0
    f_start, f_end = _frame_range_of(arm)
    n = f_end - f_start + 1

    # add a ground plane for shadow grounding
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    _setup_camera_for(arm)

    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink(missing_ok=True)

    if n <= 1:
        frames = [f_start]
    else:
        frames = [int(round(f_start + i * (n - 1) / max(1, N_THUMBS - 1)))
                  for i in range(N_THUMBS)]
    rendered = 0
    for i, f in enumerate(frames):
        bpy.context.scene.frame_set(f)
        bpy.context.scene.render.filepath = str(frames_dir / f"f{i:02d}.png")
        bpy.ops.render.render(write_still=True)
        rendered += 1
    return n, rendered


def _write_index(out_dir: Path, manifest: dict, results: dict) -> Path:
    """Build a single-page contact-sheet HTML (metadata-focused)."""
    items = []
    for slug in sorted(manifest["clips"].keys()):
        m = manifest["clips"][slug]
        png = f"{slug}.png"
        ok = (out_dir / png).exists()
        items.append({
            "slug": slug,
            "png": png if ok else None,
            "category": m.get("category", "?"),
            "n_frames": m.get("n_frames", 0),
            "duration_s": m.get("duration_s", 0.0),
            "n_bones": m.get("n_bones", 0),
            "has_root": m.get("has_root_motion", False),
            "source_fbx": m.get("source_fbx", ""),
        })
    cats = sorted({i["category"] for i in items})
    n_total = len(items)
    total_frames = sum(i["n_frames"] for i in items)
    total_secs = round(sum(i["duration_s"] for i in items), 1)

    # Per-category sections
    sections = []
    for cat in cats:
        cat_items = [i for i in items if i["category"] == cat]
        cat_secs = round(sum(i["duration_s"] for i in cat_items), 1)
        cards = []
        for it in cat_items:
            strip = (f'<img class="strip" src="{it["png"]}" loading="lazy">'
                     if it["png"] else '')
            cards.append(
                f'<div class="card">'
                f'  <div class="slug">{it["slug"]}</div>'
                f'  {strip}'
                f'  <div class="dim">{it["n_frames"]}f &middot; '
                f'{it["duration_s"]}s &middot; {it["n_bones"]} bones'
                f'{"  &middot; +root" if it["has_root"] else ""}</div>'
                f'  <div class="src">mx_{it["slug"]}</div>'
                f'</div>'
            )
        sections.append(
            f'<section id="{cat}">'
            f'  <h2>{cat.upper()} <span class="cnt">{len(cat_items)} clips '
            f'&middot; {cat_secs}s</span></h2>'
            f'  <div class="grid">{"".join(cards)}</div>'
            f'</section>'
        )
    nav = " &middot; ".join(
        f'<a href="#{c}">{c} ({sum(1 for i in items if i["category"]==c)})</a>'
        for c in cats)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>dance.AI Studios - Motion Library</title>
<style>
 :root {{ --pink:#ff38a2; --bg:#0b0b12; --fg:#eaeaf0; --dim:#9aa; }}
 body {{ font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
         background:var(--bg); color:var(--fg); margin:0; padding:24px 32px; }}
 h1 {{ font-weight:900; letter-spacing:-0.02em; margin:0 0 4px; font-size:2em; }}
 .sub {{ color:var(--dim); margin-bottom:16px; }}
 nav  {{ position:sticky; top:0; padding:10px 0; background:var(--bg);
        border-bottom:1px solid #1a1a22; z-index:10; margin-bottom:8px; }}
 nav a {{ color:var(--pink); text-decoration:none; margin-right:14px; font-weight:600; }}
 nav a:hover {{ text-decoration:underline; }}
 section {{ margin:28px 0; }}
 h2 {{ margin:6px 0 14px; padding-top:10px; text-transform:uppercase;
       letter-spacing:0.08em; font-size:1.05em; }}
 .cnt {{ color:#666; font-weight:400; font-size:0.85em; margin-left:8px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr));
         gap:12px; }}
 .card {{ background:#15151c; border:1px solid #22222b; border-radius:10px;
         padding:12px 14px; transition:border-color .15s; }}
 .card:hover {{ border-color:var(--pink); }}
 .slug {{ font-weight:700; color:var(--fg); font-size:1.02em; }}
 .strip {{ display:block; max-width:100%; height:80px; object-fit:cover;
          margin:8px 0; border-radius:6px; background:#0e0e15; }}
 .dim  {{ color:var(--dim); font-size:0.82em; margin-top:6px; }}
 .src  {{ color:#556; font-size:0.72em; font-family:Consolas, monospace; margin-top:4px; }}
</style></head>
<body>
<h1>dance.AI Studios &mdash; Motion Library</h1>
<div class="sub">{n_total} clips &middot; {total_frames} frames total &middot;
   {total_secs}s of source motion @ 30 fps</div>
<nav>{nav}</nav>
{''.join(sections)}
</body></html>"""
    idx = out_dir / "index.html"
    idx.write_text(page, encoding="utf-8")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(_argv_after_dd())

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    fbx_files = sorted(src.glob("*.fbx"))
    print(f"[preview] {len(fbx_files)} clips to render -> {out_dir}")

    results = {}
    for i, fbx in enumerate(fbx_files, 1):
        slug = _slug(fbx.name)
        clip_dir = out_dir / "_frames" / slug
        print(f"\n[{i:02d}/{len(fbx_files)}] {slug}")
        try:
            n_frames, n_thumbs = _render_filmstrip(fbx, clip_dir)
            results[slug] = {"n_frames": n_frames, "n_thumbs": n_thumbs}
            print(f"  ok  {n_thumbs} thumbs -> _frames/{slug}/")
        except Exception as e:
            print(f"  [!] failed: {e}")
            results[slug] = {"error": str(e)}

    idx = _write_index(out_dir, manifest, results)
    print(f"\n[preview] wrote {idx}")
    print(f"[preview] now run: python scripts/stitch_filmstrips.py "
          f"--frames {out_dir / '_frames'} --out {out_dir}")


if __name__ == "__main__":
    main()
