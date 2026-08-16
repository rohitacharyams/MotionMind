"""build_motion_index.py — Generate the motion-library HTML index.

Reads manifest.json and writes a single-page contact sheet describing
every clip in the library, grouped by category. Pure-Python (no Blender).

Usage:
  python scripts/build_motion_index.py \
      --manifest data/motion_library/manifest.json \
      --out data/motion_library/previews
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_index(out_dir: Path, manifest: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        })
    cats = sorted({i["category"] for i in items})
    n_total = len(items)
    total_frames = sum(i["n_frames"] for i in items)
    total_secs = round(sum(i["duration_s"] for i in items), 1)

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
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    idx = _write_index(Path(args.out), manifest)
    print(f"[index] wrote {idx}")


if __name__ == "__main__":
    main()
