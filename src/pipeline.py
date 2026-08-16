"""
Main pipeline orchestrator — ties all modules together.

Provides a high-level API for:
1. Extracting poses from dance videos
2. Processing and storing motion data
3. Creating choreography from multiple sources
4. Rendering animated character videos
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .pose_extraction import PoseExtractor
from .motion_processing import MotionNormalizer, MotionSmoother, MotionEmbedder, MotionStorage
from .avatar import AvatarRenderer, Skeleton2D, IKSolver2D, CharacterRig
from .choreography import MotionMixer, TransitionEngine, MotionStyleTransfer
from .video import VideoComposer, VideoEffects, VideoExporter
from .scene import ReelComposer

logger = logging.getLogger(__name__)


class DanceMotionPipeline:
    """Complete dance motion pipeline from video to animation.
    
    Usage:
        pipe = DanceMotionPipeline("config/pipeline_config.yaml")
        
        # Extract and store motion
        pipe.ingest_video("dance1.mp4", motion_id="dance1")
        
        # Create dance video
        pipe.create_dance_video(
            motion_ids=["dance1", "dance2"],
            style="neon",
            output="my_dance.mp4",
        )
    """

    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        device = self.config.get("pipeline", {}).get("device", "auto")

        # Initialize components (lazy for heavy ones)
        self._pose_extractor = None
        self._device = device

        self.normalizer = MotionNormalizer(self.config)
        self.smoother = MotionSmoother(self.config)
        self.embedder = MotionEmbedder(self.config)
        self.storage = MotionStorage(self.config)
        self.renderer = AvatarRenderer(self.config)
        self.mixer = MotionMixer(self.config)
        self.style_transfer = MotionStyleTransfer(self.config)
        self.composer = VideoComposer(self.config)
        self.effects = VideoEffects()
        self.exporter = VideoExporter(self.config)
        self.ik_solver = IKSolver2D()

        logger.info("Pipeline initialized (device=%s)", device)

    @property
    def pose_extractor(self) -> PoseExtractor:
        """Lazy-load pose extractor (heavy GPU models)."""
        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor(self.config, self._device)
        return self._pose_extractor

    # ──────────────────────────────────────────────────────────────
    # High-level API
    # ──────────────────────────────────────────────────────────────

    def ingest_video(
        self,
        video_path: str,
        motion_id: str | None = None,
        person_index: int = 0,
        tags: list[str] | None = None,
        max_frames: int = -1,
    ) -> str:
        """Extract poses from video, process, and store.
        
        Args:
            video_path: Path to input video.
            motion_id: ID for stored motion. Auto-generated if None.
            person_index: Which detected person to use (0 = primary).
            tags: Optional tags for searching motions later.
            max_frames: Max frames to process.
            
        Returns:
            motion_id of the stored motion.
        """
        if motion_id is None:
            motion_id = Path(video_path).stem

        logger.info("Ingesting video: %s -> motion_id='%s'", video_path, motion_id)

        # 1. Extract poses
        extraction = self.pose_extractor.extract_from_video(
            video_path, max_frames=max_frames
        )

        # 2. Get primary person
        keypoints = extraction["keypoints"][:, person_index]  # (T, 133, 2)
        scores = extraction["scores"][:, person_index]        # (T, 133)

        # 3. Normalize
        norm_result = self.normalizer.normalize(keypoints, scores)

        # 4. Smooth
        smoothed = self.smoother.smooth(norm_result["keypoints"], scores)

        # 5. Create embedding
        embedding = self.embedder.embed(smoothed)

        # 6. Store
        metadata = {
            "source_video": video_path,
            "fps": extraction["fps"],
            "frame_size": list(extraction["frame_size"]),
            "tags": tags or [],
        }
        self.storage.save_motion(
            motion_id, smoothed, scores, embedding, metadata
        )

        # Also store denormalization params for later rendering
        self.storage.save_motion(
            f"{motion_id}__params",
            np.stack([norm_result["hip_positions"],
                      np.broadcast_to(norm_result["scale_factors"][:, None], norm_result["hip_positions"].shape)]),
            scores,
            embedding,
            {"type": "normalization_params", "parent": motion_id},
        )

        logger.info(
            "Ingested '%s': %d frames, embedding shape %s",
            motion_id, smoothed.shape[0], embedding.shape
        )
        return motion_id

    def create_dance_video(
        self,
        motion_ids: list[str],
        style: str = "stick_figure",
        output: str = "dance_output.mp4",
        mix_method: str = "sequential",
        motion_trail: bool = False,
        effects: dict | None = None,
        audio_path: str | None = None,
    ) -> str:
        """Create a dance video from stored motions.
        
        Args:
            motion_ids: List of motion IDs to use.
            style: Character style ('stick_figure', 'silhouette', 'neon', 'cartoon').
            output: Output filename.
            mix_method: How to combine motions:
                'sequential' — play back-to-back with transitions
                'interleave' — alternate between clips at beat intervals
                'layer_upper' — base from first, upper body from rest
                'layer_lower' — base from first, lower body from rest
            motion_trail: Enable motion trail effect.
            effects: Optional effects config dict:
                'fade': bool, 'vignette': float, 'motion_blur': int,
                'color_grade': dict
            audio_path: Optional audio to add.
            
        Returns:
            Path to output video.
        """
        logger.info(
            "Creating dance video: motions=%s, style=%s, method=%s",
            motion_ids, style, mix_method
        )

        # Load motions
        clips = []
        scores_list = []
        for mid in motion_ids:
            data = self.storage.load_motion(mid)
            clips.append(data["keypoints"])
            scores_list.append(data["scores"])

        # Mix motions
        if mix_method == "sequential":
            mixed, mixed_scores = self.mixer.stitch_sequential(clips, scores_list)
        elif mix_method == "interleave":
            mixed = self.mixer.interleave_beats(clips)
            mixed_scores = None
        elif mix_method == "layer_upper":
            mixed = clips[0].copy()
            for overlay in clips[1:]:
                mixed = self.mixer.blend_layers(
                    mixed, overlay,
                    ["left_arm", "right_arm", "head"],
                )
            mixed_scores = scores_list[0]
        elif mix_method == "layer_lower":
            mixed = clips[0].copy()
            for overlay in clips[1:]:
                mixed = self.mixer.blend_layers(
                    mixed, overlay,
                    ["left_leg", "right_leg"],
                )
            mixed_scores = scores_list[0]
        else:
            mixed, mixed_scores = clips[0], scores_list[0]

        # Render
        frames = self.composer.compose_single(
            mixed, mixed_scores, style=style, motion_trail=motion_trail
        )

        # Apply effects
        if effects:
            frames = self._apply_effects(frames, effects)

        # Determine FPS from first motion's metadata
        fps = 30.0
        first_motion_meta = self.storage.load_motion(motion_ids[0]).get("metadata", {})
        if "fps" in first_motion_meta:
            fps = first_motion_meta["fps"]

        # Export
        return self.exporter.export_ffmpeg(
            frames, output, fps=fps, audio_path=audio_path
        )

    def create_concept_video(
        self,
        motion_ids: list[str],
        styles: list[str] | None = None,
        layout: str = "side_by_side",
        output: str = "concept_output.mp4",
        style_transfer_from: str | None = None,
        effects: dict | None = None,
        audio_path: str | None = None,
    ) -> str:
        """Create a concept dance video with multiple characters.
        
        Args:
            motion_ids: List of motion IDs.
            styles: Character style per motion (or single for all).
            layout: 'side_by_side', 'overlapping', 'grid'.
            output: Output filename.
            style_transfer_from: Apply style from this motion ID to all others.
            effects: Visual effects config.
            audio_path: Optional audio.
            
        Returns:
            Path to output video.
        """
        characters = []
        for i, mid in enumerate(motion_ids):
            data = self.storage.load_motion(mid)
            kps = data["keypoints"]

            # Apply style transfer if requested
            if style_transfer_from and style_transfer_from != mid:
                style_data = self.storage.load_motion(style_transfer_from)
                kps = self.style_transfer.transfer(kps, style_data["keypoints"])

            style = "stick_figure"
            if styles:
                style = styles[i] if i < len(styles) else styles[-1]

            characters.append({
                "keypoints": kps,
                "scores": data["scores"],
                "style": style,
                "label": mid,
            })

        frames = self.composer.compose_multi_character(characters, layout=layout)

        if effects:
            frames = self._apply_effects(frames, effects)

        fps = 30.0
        return self.exporter.export_ffmpeg(
            frames, output, fps=fps, audio_path=audio_path
        )

    def create_overlay_video(
        self,
        video_path: str,
        motion_id: str | None = None,
        style: str = "neon",
        output: str = "overlay_output.mp4",
        opacity: float = 0.6,
        pip: bool = False,
    ) -> str:
        """Create video with animated overlay on original footage.
        
        Args:
            video_path: Source video path.
            motion_id: Motion ID (extracts fresh if None).
            style: Character style for overlay.
            output: Output filename.
            opacity: Overlay opacity.
            pip: Use picture-in-picture mode instead of overlay.
            
        Returns:
            Path to output video.
        """
        if motion_id is None:
            motion_id = self.ingest_video(video_path)

        # Get raw (non-normalized) keypoints for overlay
        extraction = self.pose_extractor.extract_from_video(video_path)
        keypoints = extraction["keypoints"][:, 0]  # Primary person, pixel coords
        scores = extraction["scores"][:, 0]

        frames = self.composer.compose_with_source(
            video_path, keypoints, scores,
            style=style, overlay_opacity=opacity, pip_mode=pip,
        )

        return self.exporter.export_ffmpeg(
            frames, output, fps=extraction["fps"]
        )

    def create_reel(
        self,
        motion_ids: list[str],
        style: str = "neon",
        rig_name: str = "dancer_female",
        bg_preset: str = "studio_dark",
        output: str = "reel.mp4",
        mix_method: str = "sequential",
        audio_path: str | None = None,
        watermark: str | None = None,
        motion_trail: bool = False,
        format: str = "instagram_reel",
    ) -> str:
        """Create a social media reel from stored motions.

        Args:
            motion_ids: Motion IDs to stitch together.
            style: Character style.
            rig_name: Character rig (dancer_female, dancer_male, chibi, robot, shadow_dancer).
            bg_preset: Background preset (studio_dark, neon_club, stage_floor, etc.).
            output: Output file path.
            mix_method: How to combine motions (sequential, interleave).
            audio_path: Audio track to mux in.
            watermark: Watermark text (e.g. "@MotionMind").
            motion_trail: Enable motion trails.
            format: Reel format (instagram_reel, tiktok, youtube_short, instagram_square).

        Returns:
            Path to output file.
        """
        logger.info("Creating reel: motions=%s, style=%s, rig=%s", motion_ids, style, rig_name)

        # Load and mix motions
        clips = []
        scores_list = []
        for mid in motion_ids:
            data = self.storage.load_motion(mid)
            clips.append(data["keypoints"])
            scores_list.append(data["scores"])

        if mix_method == "sequential":
            mixed, mixed_scores = self.mixer.stitch_sequential(clips, scores_list)
        elif mix_method == "interleave":
            mixed = self.mixer.interleave_beats(clips)
            mixed_scores = None
        else:
            mixed, mixed_scores = clips[0], scores_list[0]

        # Determine FPS
        fps = 30.0
        first_meta = self.storage.load_motion(motion_ids[0]).get("metadata", {})
        if "fps" in first_meta:
            fps = first_meta["fps"]

        # Compose reel
        composer = ReelComposer(self.config)
        dims = composer.reel_dimensions()
        if format in dims:
            composer.REEL_WIDTH, composer.REEL_HEIGHT = dims[format]

        return composer.compose_reel(
            mixed,
            mixed_scores,
            style=style,
            rig_name=rig_name,
            bg_preset=bg_preset,
            output_path=output,
            fps=fps,
            audio_path=audio_path,
            watermark=watermark,
            motion_trail=motion_trail,
        )

    def find_similar_motions(
        self,
        query_motion_id: str | None = None,
        query_video: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Find motions similar to a query.
        
        Args:
            query_motion_id: Search by stored motion ID.
            query_video: Search by video file (extracts poses first).
            top_k: Number of results.
            
        Returns:
            List of result dicts with motion_id and similarity_score.
        """
        if query_motion_id:
            data = self.storage.load_motion(query_motion_id)
            embedding = data["embedding"]
        elif query_video:
            mid = self.ingest_video(query_video, motion_id="__query_temp")
            data = self.storage.load_motion(mid)
            embedding = data["embedding"]
            self.storage.delete_motion(mid)
        else:
            raise ValueError("Provide query_motion_id or query_video")

        return self.storage.search_similar(embedding, top_k=top_k)

    def list_motions(self) -> list[dict]:
        """List all stored motions."""
        return self.storage.list_motions()

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _apply_effects(self, frames: list[np.ndarray], effects: dict) -> list[np.ndarray]:
        """Apply configured visual effects to frames."""
        if effects.get("fade"):
            frames = self.effects.fade_in_out(frames)

        if effects.get("vignette"):
            frames = self.effects.vignette(frames, strength=effects["vignette"])

        if effects.get("motion_blur"):
            frames = self.effects.motion_blur(frames, strength=effects["motion_blur"])

        if effects.get("color_grade"):
            frames = self.effects.color_grade(frames, **effects["color_grade"])

        return frames

    @staticmethod
    def _default_config() -> dict:
        return {
            "pipeline": {"device": "auto"},
            "pose_extraction": {
                "mode": "performance",
            },
            "motion_processing": {
                "normalize_to_hip": True,
                "scale_to_unit": True,
                "min_confidence": 0.3,
                "interpolate_low_confidence": True,
                "smoothing": {
                    "enabled": True,
                    "method": "savgol",
                    "window_size": 7,
                    "poly_order": 3,
                },
            },
            "motion_storage": {
                "db_path": "data/motion_db",
                "embedding_dim": 256,
                "method": "temporal_pooling",
                "index_type": "Flat",
            },
            "avatar": {
                "skeleton_type": "body_17",
                "default_style": "stick_figure",
                "canvas_width": 1920,
                "canvas_height": 1080,
                "background_color": [0, 0, 0],
                "fps": 30,
                "styles": {
                    "stick_figure": {
                        "joint_color": [255, 255, 255],
                        "bone_color": [0, 200, 255],
                        "joint_radius": 6,
                        "bone_width": 3,
                    },
                    "silhouette": {
                        "fill_color": [255, 255, 255],
                        "outline_color": [0, 200, 255],
                        "outline_width": 2,
                    },
                    "neon": {
                        "glow_color": [0, 255, 200],
                        "core_color": [255, 255, 255],
                        "glow_radius": 15,
                        "bone_width": 4,
                    },
                    "cartoon": {
                        "skin_color": [255, 220, 185],
                        "outline_color": [40, 40, 40],
                        "outfit_color": [100, 149, 237],
                        "outline_width": 3,
                    },
                },
            },
            "choreography": {
                "transition_frames": 15,
                "transition_method": "slerp",
                "style_weight": 0.5,
            },
            "video_output": {
                "codec": "libx264",
                "quality": 18,
                "output_dir": "data/output_videos",
            },
        }
