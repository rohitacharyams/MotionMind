"""Pose extraction module — RTMPose (rtmlib) and MediaPipe backends."""
from .extractor import PoseExtractor
from .extractor_mediapipe import MediaPipePoseExtractor

__all__ = ["PoseExtractor", "MediaPipePoseExtractor"]
