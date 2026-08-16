"""format_spec.py - Reel format spec & hook-driven templates.

A "reel" = ordered list of SHOTS rendered in Blender + a post pipeline
(ffmpeg) that adds text overlays, beat cuts, and music.

Each shot specifies:
  - scene_blend: scene to load (or "none" for studio bg)
  - char: character name (must exist in motion_library/chars/<name>.blend)
  - clip: action name (e.g. "alicia_hip_hop_dancing")
  - camera: one of {CAM_wide, CAM_med, CAM_close} from the scene
  - duration_s: shot length in seconds
  - clip_start_s: optional offset into the source clip
  - speed: 1.0 normal, 0.5 slow-mo, 2.0 fast
  - effect: optional {"freeze_first_frame": True, "fade_in": 0.2, ...}
  - face_camera: rotate char 180deg (default True)

A reel also defines:
  - title: filename stem
  - fps: typically 30
  - aspect: "9:16" (1080x1920) or "16:9" or "1:1"
  - overlays: list of timed text overlays (hook text, CTA, captions)
  - hook_pattern: string id for analytics/labeling
  - bgm: optional music file path

Hook patterns supported:
  - COLD_OPEN:    start mid-action at full speed, no preroll
  - TEXT_HOOK:    big question/promise text in shot 1 (0.0-1.5s)
  - FREEZE_REVEAL: freeze on dramatic frame, then unfreeze with whoosh
  - BEAT_DROP:    slow-mo intro -> normal speed on "drop" with text
  - PATTERN_INT:  rapid cut between 2-3 cameras to break attention
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# dataclasses                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Shot:
    scene_blend: str
    char: str
    clip: str
    camera: str = "CAM_med"
    duration_s: float = 2.0
    clip_start_s: float = 0.0
    speed: float = 1.0
    face_camera: bool = True
    lens: Optional[float] = None    # override camera focal length (mm)
    effect: dict = field(default_factory=dict)


@dataclass
class Overlay:
    """Text overlay drawn via ffmpeg drawtext."""
    text: str
    start_s: float
    end_s: float
    # placement
    x: str = "(w-text_w)/2"          # ffmpeg expr
    y: str = "h*0.18"                # default upper-third
    # styling
    fontsize: int = 96
    fontcolor: str = "white"
    box: bool = True
    boxcolor: str = "black@0.55"
    boxborderw: int = 24
    fontfile: str = "C\\:/Windows/Fonts/segoeuib.ttf"


@dataclass
class Reel:
    title: str
    hook_pattern: str
    shots: list[Shot]
    overlays: list[Overlay] = field(default_factory=list)
    fps: int = 30
    aspect: str = "9:16"            # "9:16" | "16:9" | "1:1"
    bgm: Optional[str] = None       # absolute path or None

    @property
    def resolution(self) -> tuple[int, int]:
        return {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1":  (1080, 1080),
        }[self.aspect]

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.shots)

    def to_json(self) -> str:
        d = {
            "title": self.title,
            "hook_pattern": self.hook_pattern,
            "fps": self.fps,
            "aspect": self.aspect,
            "bgm": self.bgm,
            "shots": [asdict(s) for s in self.shots],
            "overlays": [asdict(o) for o in self.overlays],
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Reel":
        d = json.loads(text)
        return cls(
            title=d["title"],
            hook_pattern=d["hook_pattern"],
            fps=d.get("fps", 30),
            aspect=d.get("aspect", "9:16"),
            bgm=d.get("bgm"),
            shots=[Shot(**s) for s in d["shots"]],
            overlays=[Overlay(**o) for o in d.get("overlays", [])],
        )


# --------------------------------------------------------------------------- #
# hook templates                                                              #
# --------------------------------------------------------------------------- #
SCENES = {
    "neon":    r"c:\dan\data\scenes\neon_street.blend",
    "rooftop": r"c:\dan\data\scenes\rooftop.blend",
    "subway":  r"c:\dan\data\scenes\subway.blend",
}


def template_text_hook_dance(char: str = "alicia",
                              scene: str = "neon") -> Reel:
    """A:  TEXT_HOOK  ->  reveal dance.

    0.0-1.5s: medium shot, character idle, big text "WAIT FOR IT..."
    1.5-5.5s: cut to wide, hip-hop dance
    5.5-8.0s: cut to close, last beats
    """
    scene_path = SCENES[scene]
    return Reel(
        title=f"hook_text__{char}_{scene}",
        hook_pattern="TEXT_HOOK",
        shots=[
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_idle", camera="CAM_med",
                 duration_s=1.5, clip_start_s=0.5),
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_wide",
                 duration_s=4.0, clip_start_s=0.0),
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_close",
                 duration_s=2.5, clip_start_s=4.0),
        ],
        overlays=[
            Overlay(text="WAIT FOR IT...", start_s=0.1, end_s=1.4,
                    fontsize=110),
            Overlay(text="HE GOES OFF", start_s=1.6, end_s=3.2,
                    fontsize=120, y="h*0.10"),
        ],
    )


def template_beat_drop(char: str = "alicia",
                        scene: str = "subway") -> Reel:
    """B:  BEAT_DROP  ->  slow-mo intro then normal speed on drop.

    0.0-2.0s: close shot, dance @ 0.4x speed, text "3...2...1"
    2.0-7.0s: wide shot, dance @ 1.0x, text "DROP"
    """
    scene_path = SCENES[scene]
    return Reel(
        title=f"hook_drop__{char}_{scene}",
        hook_pattern="BEAT_DROP",
        shots=[
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_close",
                 duration_s=2.0, clip_start_s=2.5, speed=0.4),
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_wide",
                 duration_s=5.0, clip_start_s=4.0, speed=1.0),
        ],
        overlays=[
            Overlay(text="3", start_s=0.1, end_s=0.7, fontsize=180),
            Overlay(text="2", start_s=0.75, end_s=1.30, fontsize=180),
            Overlay(text="1", start_s=1.35, end_s=1.95, fontsize=180),
            Overlay(text="DROP", start_s=2.05, end_s=3.5, fontsize=160,
                    y="h*0.12", fontcolor="yellow"),
        ],
    )


def template_freeze_reveal(char: str = "alicia",
                            scene: str = "rooftop") -> Reel:
    """C:  FREEZE_REVEAL.

    0.0-1.0s: medium shot, FROZEN on first frame of pose, text overlay
    1.0-3.0s: same frame range plays at normal speed
    3.0-7.0s: wide, full dance.
    """
    scene_path = SCENES[scene]
    return Reel(
        title=f"hook_freeze__{char}_{scene}",
        hook_pattern="FREEZE_REVEAL",
        shots=[
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_med",
                 duration_s=1.0, clip_start_s=2.0,
                 effect={"freeze": True}),
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_med",
                 duration_s=2.0, clip_start_s=2.0),
            Shot(scene_blend=scene_path, char=char,
                 clip=f"{char}_hip_hop_dancing", camera="CAM_wide",
                 duration_s=4.0, clip_start_s=4.0),
        ],
        overlays=[
            Overlay(text="POV: rooftop session", start_s=0.05, end_s=0.95,
                    fontsize=86, y="h*0.12"),
        ],
    )


def template_pattern_interrupt(char: str = "alicia",
                                scene: str = "neon") -> Reel:
    """D:  PATTERN_INT  ->  rapid 0.5s cuts between 3 cameras.

    First 2s: 4 quick cuts on idle/dance, then settles into wide.
    """
    scene_path = SCENES[scene]
    cams = ["CAM_close", "CAM_med", "CAM_close", "CAM_wide"]
    starts = [0.5, 1.0, 1.8, 2.2]
    shots = []
    for c, s in zip(cams, starts):
        shots.append(Shot(scene_blend=scene_path, char=char,
                          clip=f"{char}_hip_hop_dancing", camera=c,
                          duration_s=0.5, clip_start_s=s))
    shots.append(Shot(scene_blend=scene_path, char=char,
                      clip=f"{char}_hip_hop_dancing", camera="CAM_wide",
                      duration_s=5.0, clip_start_s=4.0))
    return Reel(
        title=f"hook_cuts__{char}_{scene}",
        hook_pattern="PATTERN_INT",
        shots=shots,
        overlays=[
            Overlay(text="STOP SCROLLING", start_s=0.05, end_s=1.95,
                    fontsize=110, y="h*0.10"),
            Overlay(text="watch the whole thing", start_s=2.05, end_s=3.5,
                    fontsize=58, y="h*0.86"),
        ],
    )


TEMPLATES = {
    "text_hook":     template_text_hook_dance,
    "beat_drop":     template_beat_drop,
    "freeze_reveal": template_freeze_reveal,
    "pattern_int":   template_pattern_interrupt,
}


# --------------------------------------------------------------------------- #
# CLI helper: dump a template to JSON                                         #
# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", choices=list(TEMPLATES.keys()),
                    required=True)
    ap.add_argument("--char", default="alicia")
    ap.add_argument("--scene", default="neon",
                    choices=list(SCENES.keys()))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reel = TEMPLATES[args.template](char=args.char, scene=args.scene)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(reel.to_json(), encoding="utf-8")
    print(f"[spec] wrote {args.out}  duration={reel.duration_s:.1f}s  "
          f"shots={len(reel.shots)}")


if __name__ == "__main__":
    main()
