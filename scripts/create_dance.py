"""
Create dance videos from stored motions.

Usage:
    # Simple single-motion video
    python scripts/create_dance.py --motions dance1 --style neon --output my_dance.mp4

    # Sequential mashup from multiple motions
    python scripts/create_dance.py --motions dance1,dance2,dance3 --style silhouette \
        --mix sequential --output mashup.mp4 --trail

    # Concept video with multiple characters
    python scripts/create_dance.py --motions dance1,dance2 --style neon,cartoon \
        --mode concept --layout side_by_side --output concept.mp4

    # Overlay on original video
    python scripts/create_dance.py --motions dance1 --mode overlay \
        --source-video original.mp4 --style neon --output overlay.mp4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DanceMotionPipeline


def main():
    parser = argparse.ArgumentParser(description="Create dance video from motions")
    parser.add_argument("--motions", required=True, help="Comma-separated motion IDs")
    parser.add_argument("--style", default="stick_figure",
                       help="Character style(s), comma-separated: stick_figure, silhouette, neon, cartoon")
    parser.add_argument("--output", default="dance_output.mp4", help="Output filename")
    parser.add_argument("--config", default="config/pipeline_config.yaml")

    # Mode
    parser.add_argument("--mode", default="dance", choices=["dance", "concept", "overlay"],
                       help="Video creation mode")

    # Dance mode options
    parser.add_argument("--mix", default="sequential",
                       choices=["sequential", "interleave", "layer_upper", "layer_lower"],
                       help="Motion mixing method")
    parser.add_argument("--trail", action="store_true", help="Enable motion trail")

    # Concept mode options
    parser.add_argument("--layout", default="side_by_side",
                       choices=["side_by_side", "overlapping", "grid"])
    parser.add_argument("--style-from", default=None,
                       help="Apply style transfer from this motion ID")

    # Overlay mode options
    parser.add_argument("--source-video", default=None, help="Source video for overlay mode")
    parser.add_argument("--opacity", type=float, default=0.6)
    parser.add_argument("--pip", action="store_true", help="Picture-in-picture mode")

    # Effects
    parser.add_argument("--fade", action="store_true", help="Fade in/out")
    parser.add_argument("--vignette", type=float, default=0, help="Vignette strength")
    parser.add_argument("--blur", type=int, default=0, help="Motion blur strength")
    parser.add_argument("--audio", default=None, help="Audio file to add")

    args = parser.parse_args()

    pipe = DanceMotionPipeline(args.config)
    motion_ids = [m.strip() for m in args.motions.split(",")]
    styles = [s.strip() for s in args.style.split(",")]

    effects = {}
    if args.fade:
        effects["fade"] = True
    if args.vignette > 0:
        effects["vignette"] = args.vignette
    if args.blur > 0:
        effects["motion_blur"] = args.blur

    if args.mode == "dance":
        output = pipe.create_dance_video(
            motion_ids=motion_ids,
            style=styles[0],
            output=args.output,
            mix_method=args.mix,
            motion_trail=args.trail,
            effects=effects or None,
            audio_path=args.audio,
        )

    elif args.mode == "concept":
        output = pipe.create_concept_video(
            motion_ids=motion_ids,
            styles=styles,
            layout=args.layout,
            output=args.output,
            style_transfer_from=args.style_from,
            effects=effects or None,
            audio_path=args.audio,
        )

    elif args.mode == "overlay":
        if not args.source_video:
            print("Error: --source-video required for overlay mode")
            return
        output = pipe.create_overlay_video(
            video_path=args.source_video,
            motion_id=motion_ids[0] if motion_ids else None,
            style=styles[0],
            output=args.output,
            opacity=args.opacity,
            pip=args.pip,
        )

    print(f"\nVideo saved: {output}")


if __name__ == "__main__":
    main()
