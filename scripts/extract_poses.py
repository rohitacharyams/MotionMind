"""
Extract poses from dance videos and store them.

Usage:
    python scripts/extract_poses.py --video path/to/video.mp4 --id my_dance
    python scripts/extract_poses.py --video path/to/video.mp4 --id my_dance --tags hip-hop,fast
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DanceMotionPipeline


def main():
    parser = argparse.ArgumentParser(description="Extract poses from dance video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--id", default=None, help="Motion ID (default: filename)")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Config path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames (-1 = all)")
    parser.add_argument("--person", type=int, default=0, help="Person index to track")
    args = parser.parse_args()

    pipe = DanceMotionPipeline(args.config)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    motion_id = pipe.ingest_video(
        args.video,
        motion_id=args.id,
        person_index=args.person,
        tags=tags,
        max_frames=args.max_frames,
    )

    print(f"\nDone! Motion stored as '{motion_id}'")
    print(f"Frames: {pipe.storage.load_motion(motion_id)['keypoints'].shape[0]}")
    print(f"Total motions in DB: {len(pipe.list_motions())}")


if __name__ == "__main__":
    main()
