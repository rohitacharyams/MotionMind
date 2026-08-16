"""
Motion mixer — combine movements from different dance sequences.

Supports:
- Sequential stitching (play clips back-to-back with transitions)
- Layered blending (combine upper body from one, lower from another)
- Temporal mixing (interleave beats from different sources)
"""

import numpy as np
from ..pose_extraction.utils import JOINT_GROUPS
from .transitions import TransitionEngine


class MotionMixer:
    """Mix and combine motion clips from different sources."""

    def __init__(self, config: dict):
        cfg = config.get("choreography", {})
        self.transition_frames = cfg.get("transition_frames", 15)
        self.transition_engine = TransitionEngine(config)

    def stitch_sequential(
        self,
        clips: list[np.ndarray],
        scores_list: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Stitch motion clips sequentially with smooth transitions.
        
        Args:
            clips: List of (T_i, K, 2) motion arrays.
            scores_list: Optional list of (T_i, K) score arrays.
            
        Returns:
            (merged_keypoints, merged_scores) tuple.
        """
        if len(clips) == 0:
            raise ValueError("No clips to stitch")
        if len(clips) == 1:
            scores = scores_list[0] if scores_list else None
            return clips[0], scores

        result = clips[0].copy()
        result_scores = scores_list[0].copy() if scores_list else None

        for i in range(1, len(clips)):
            clip = clips[i]
            clip_scores = scores_list[i] if scores_list else None

            # Create transition between end of result and start of clip
            transition = self.transition_engine.create_transition(
                result[-self.transition_frames:],
                clip[:self.transition_frames],
                n_frames=self.transition_frames,
            )

            # Splice: result[:-transition] + transition + clip[transition:]
            result = np.concatenate([
                result[:-self.transition_frames],
                transition,
                clip[self.transition_frames:],
            ], axis=0)

            if result_scores is not None and clip_scores is not None:
                # Interpolate scores through transition
                t_scores = np.linspace(0, 1, self.transition_frames)[:, None]
                score_transition = (
                    result_scores[-self.transition_frames:] * (1 - t_scores) +
                    clip_scores[:self.transition_frames] * t_scores
                )
                result_scores = np.concatenate([
                    result_scores[:-self.transition_frames],
                    score_transition,
                    clip_scores[self.transition_frames:],
                ], axis=0)

        return result, result_scores

    def blend_layers(
        self,
        base_clip: np.ndarray,
        overlay_clip: np.ndarray,
        body_parts: list[str],
        blend_weight: float = 1.0,
    ) -> np.ndarray:
        """Blend specific body parts from overlay onto base clip.
        
        Allows combining e.g. upper body from one dance with
        lower body from another.
        
        Args:
            base_clip: (T, K, 2) base motion.
            overlay_clip: (T, K, 2) overlay motion.
            body_parts: List of body part groups to take from overlay.
                e.g. ["left_arm", "right_arm", "head"]
            blend_weight: Weight for overlay (0=all base, 1=all overlay).
            
        Returns:
            (T, K, 2) blended motion.
        """
        T = min(len(base_clip), len(overlay_clip))
        result = base_clip[:T].copy()
        overlay = overlay_clip[:T]

        for part in body_parts:
            indices = JOINT_GROUPS.get(part, [])
            for idx in indices:
                if idx < result.shape[1] and idx < overlay.shape[1]:
                    result[:, idx] = (
                        (1 - blend_weight) * base_clip[:T, idx] +
                        blend_weight * overlay[:T, idx]
                    )

        return result

    def interleave_beats(
        self,
        clips: list[np.ndarray],
        beat_frames: int = 30,
    ) -> np.ndarray:
        """Interleave clips at beat intervals.
        
        Switches between clips every N frames, useful for
        creating mashup-style choreography.
        
        Args:
            clips: List of (T_i, K, 2) motion arrays.
            beat_frames: Frames per beat segment.
            
        Returns:
            (T, K, 2) interleaved motion.
        """
        if len(clips) == 0:
            raise ValueError("No clips to interleave")

        # Find total length
        max_T = max(c.shape[0] for c in clips)
        K = clips[0].shape[1]

        # Pad clips to same length (loop if needed)
        padded = []
        for clip in clips:
            if clip.shape[0] < max_T:
                n_repeat = (max_T // clip.shape[0]) + 1
                clip = np.tile(clip, (n_repeat, 1, 1))[:max_T]
            padded.append(clip[:max_T])

        result_segments = []
        clip_idx = 0
        t = 0

        while t < max_T:
            end = min(t + beat_frames, max_T)
            segment = padded[clip_idx % len(padded)][t:end]
            result_segments.append(segment)

            # Smooth transition at boundary
            if len(result_segments) > 1 and self.transition_frames > 0:
                trans_len = min(self.transition_frames, len(result_segments[-2]), len(segment))
                if trans_len > 1:
                    weights = np.linspace(0, 1, trans_len)[:, None, None]
                    prev_end = result_segments[-2][-trans_len:]
                    curr_start = segment[:trans_len]
                    blended = prev_end * (1 - weights) + curr_start * weights
                    result_segments[-2][-trans_len:] = blended

            clip_idx += 1
            t = end

        return np.concatenate(result_segments, axis=0)

    def loop_clip(
        self, clip: np.ndarray, n_loops: int, crossfade_frames: int = 10
    ) -> np.ndarray:
        """Loop a clip N times with crossfade at loop boundaries.
        
        Args:
            clip: (T, K, 2) motion.
            n_loops: Number of times to loop.
            crossfade_frames: Frames for crossfade at loop point.
            
        Returns:
            (T*n_loops - crossfade*(n_loops-1), K, 2) looped motion.
        """
        if n_loops <= 1:
            return clip

        clips = [clip] * n_loops
        return self.stitch_sequential(clips)[0]
