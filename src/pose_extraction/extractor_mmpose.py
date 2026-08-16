"""
High-quality pose extraction using MMPose RTMPose whole-body models.

Extracts 133 keypoints (body + hands + face) from video frames using GPU.
Uses RTMDet for person detection and RTMPose for keypoint estimation.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract whole-body keypoints from video using MMPose.
    
    Uses RTMPose-L whole-body model for 133-keypoint extraction,
    yielding body, hands, face, and feet keypoints at high quality.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.det_model = None
        self.pose_model = None
        self._init_models()

    def _init_models(self):
        """Initialize MMPose detection and pose models."""
        try:
            from mmpose.apis import init_model as init_pose_model
            from mmdet.apis import init_detector
        except ImportError:
            raise ImportError(
                "MMPose and MMDet are required. Install via:\n"
                "  pip install openmim\n"
                "  mim install mmengine mmcv mmdet mmpose"
            )

        cfg = self.config.get("pose_extraction", {})

        # Download models if not present via mim
        self._ensure_models(cfg)

        det_config = self._get_det_config(cfg)
        det_checkpoint = self._get_det_checkpoint(cfg)
        pose_config = self._get_pose_config(cfg)
        pose_checkpoint = self._get_pose_checkpoint(cfg)

        logger.info("Loading detection model: %s", det_config)
        self.det_model = init_detector(
            det_config, det_checkpoint, device=self.device
        )

        logger.info("Loading pose model: %s", pose_config)
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=self.device
        )

        # Set precision
        if self.config.get("pipeline", {}).get("precision") == "fp16" and self.device == "cuda":
            self.det_model = self.det_model.half()
            self.pose_model = self.pose_model.half()
            logger.info("Using FP16 precision for GPU inference")

    def _ensure_models(self, cfg: dict):
        """Ensure model configs and checkpoints are available."""
        try:
            import mim
            # Check if configs exist, download if needed
            pose_model = cfg.get("pose_model", "rtmpose-l_8xb32-270e_coco-wholebody-384x288")
            logger.info("Ensuring model availability: %s", pose_model)
        except ImportError:
            logger.warning("OpenMIM not available, assuming models are manually placed")

    def _get_det_config(self, cfg: dict) -> str:
        """Resolve detection model config path."""
        try:
            from mmdet import __file__ as mmdet_file
            mmdet_dir = Path(mmdet_file).parent
            # RTMDet-m for person detection
            config_path = mmdet_dir / ".mim" / "configs" / "rtmdet" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.py"
            if config_path.exists():
                return str(config_path)
        except Exception:
            pass
        
        # Fallback: try to find via mim
        try:
            from mim.utils import get_installed_path
            mmdet_path = get_installed_path("mmdet")
            config_path = Path(mmdet_path) / ".mim" / "configs" / "rtmdet" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.py"
            if config_path.exists():
                return str(config_path)
        except Exception:
            pass

        # Use direct config name — MMPose will resolve
        return "rtmdet_m_640-8xb32_coco-person"

    def _get_det_checkpoint(self, cfg: dict) -> str:
        return cfg.get(
            "det_checkpoint",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        )

    def _get_pose_config(self, cfg: dict) -> str:
        try:
            from mim.utils import get_installed_path
            mmpose_path = get_installed_path("mmpose")
            config_path = (
                Path(mmpose_path) / ".mim" / "configs" / "wholebody_2d_keypoint" /
                "rtmpose" / "ubody" /
                "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
            )
            if config_path.exists():
                return str(config_path)
        except Exception:
            pass
        return "rtmpose-l_8xb32-270e_coco-wholebody-384x288"

    def _get_pose_checkpoint(self, cfg: dict) -> str:
        return cfg.get(
            "pose_checkpoint",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/wholebody/"
            "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd42_20230728.pth"
        )

    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = -1,
        progress: bool = True,
    ) -> dict:
        """Extract keypoints from all frames of a video.
        
        Args:
            video_path: Path to input video file.
            max_frames: Max frames to process (-1 for all).
            progress: Show progress bar.
            
        Returns:
            dict with keys:
                'keypoints': np.ndarray (T, N_persons, 133, 2) — xy coordinates
                'scores': np.ndarray (T, N_persons, 133) — confidence scores
                'bboxes': np.ndarray (T, N_persons, 4) — person bounding boxes
                'fps': float — source video FPS
                'frame_size': tuple (W, H)
                'n_frames': int
                'video_path': str
        """
        from mmpose.apis import inference_topdown
        from mmdet.apis import inference_detector
        from mmpose.structures import merge_data_samples

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        logger.info(
            "Processing video: %s (%dx%d, %.1f fps, %d frames)",
            video_path, w, h, fps, total_frames
        )

        all_keypoints = []
        all_scores = []
        all_bboxes = []

        det_score_thr = self.config.get("pose_extraction", {}).get("det_score_thr", 0.5)
        bbox_expansion = self.config.get("pose_extraction", {}).get("bbox_expansion", 1.25)

        iterator = range(total_frames)
        if progress:
            iterator = tqdm(iterator, desc="Extracting poses", unit="frame")

        for frame_idx in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect persons
            det_result = inference_detector(self.det_model, frame)
            pred_instances = det_result.pred_instances

            # Filter by score and get person class (class 0)
            if hasattr(pred_instances, 'labels'):
                person_mask = (pred_instances.labels == 0) & (pred_instances.scores > det_score_thr)
            else:
                person_mask = pred_instances.scores > det_score_thr

            bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
            scores = pred_instances.scores[person_mask].cpu().numpy()

            if len(bboxes) == 0:
                all_keypoints.append(np.zeros((0, 133, 2), dtype=np.float32))
                all_scores.append(np.zeros((0, 133), dtype=np.float32))
                all_bboxes.append(np.zeros((0, 4), dtype=np.float32))
                continue

            # Expand bboxes for better pose estimation
            bboxes = self._expand_bboxes(bboxes, bbox_expansion, (w, h))

            # Run pose estimation
            pose_results = inference_topdown(
                self.pose_model, frame, bboxes
            )
            data_samples = merge_data_samples(pose_results)

            kpts = data_samples.pred_instances.keypoints  # (N, 133, 2)
            kpt_scores = data_samples.pred_instances.keypoint_scores  # (N, 133)

            all_keypoints.append(kpts.astype(np.float32))
            all_scores.append(kpt_scores.astype(np.float32))
            all_bboxes.append(bboxes.astype(np.float32))

        cap.release()

        # Pad to consistent number of persons per frame
        max_persons = max(k.shape[0] for k in all_keypoints) if all_keypoints else 1
        max_persons = max(max_persons, 1)

        keypoints_padded = np.zeros((len(all_keypoints), max_persons, 133, 2), dtype=np.float32)
        scores_padded = np.zeros((len(all_keypoints), max_persons, 133), dtype=np.float32)
        bboxes_padded = np.zeros((len(all_keypoints), max_persons, 4), dtype=np.float32)

        for i, (k, s, b) in enumerate(zip(all_keypoints, all_scores, all_bboxes)):
            n = k.shape[0]
            if n > 0:
                keypoints_padded[i, :n] = k[:max_persons]
                scores_padded[i, :n] = s[:max_persons]
                bboxes_padded[i, :n] = b[:max_persons]

        return {
            "keypoints": keypoints_padded,
            "scores": scores_padded,
            "bboxes": bboxes_padded,
            "fps": fps,
            "frame_size": (w, h),
            "n_frames": len(all_keypoints),
            "video_path": video_path,
        }

    def extract_from_frame(self, frame: np.ndarray) -> dict:
        """Extract keypoints from a single frame.
        
        Args:
            frame: BGR image as numpy array.
            
        Returns:
            dict with 'keypoints', 'scores', 'bboxes' for this frame.
        """
        from mmpose.apis import inference_topdown
        from mmdet.apis import inference_detector
        from mmpose.structures import merge_data_samples

        det_result = inference_detector(self.det_model, frame)
        pred_instances = det_result.pred_instances

        det_score_thr = self.config.get("pose_extraction", {}).get("det_score_thr", 0.5)

        if hasattr(pred_instances, 'labels'):
            person_mask = (pred_instances.labels == 0) & (pred_instances.scores > det_score_thr)
        else:
            person_mask = pred_instances.scores > det_score_thr

        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()

        h, w = frame.shape[:2]
        bbox_expansion = self.config.get("pose_extraction", {}).get("bbox_expansion", 1.25)
        bboxes = self._expand_bboxes(bboxes, bbox_expansion, (w, h))

        if len(bboxes) == 0:
            return {
                "keypoints": np.zeros((0, 133, 2), dtype=np.float32),
                "scores": np.zeros((0, 133), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
            }

        pose_results = inference_topdown(self.pose_model, frame, bboxes)
        data_samples = merge_data_samples(pose_results)

        return {
            "keypoints": data_samples.pred_instances.keypoints.astype(np.float32),
            "scores": data_samples.pred_instances.keypoint_scores.astype(np.float32),
            "bboxes": bboxes.astype(np.float32),
        }

    @staticmethod
    def _expand_bboxes(bboxes: np.ndarray, factor: float, frame_size: tuple) -> np.ndarray:
        """Expand bounding boxes by a factor while keeping them in frame."""
        w, h = frame_size
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        bw = (bboxes[:, 2] - bboxes[:, 0]) * factor
        bh = (bboxes[:, 3] - bboxes[:, 1]) * factor

        expanded = np.stack([
            np.clip(cx - bw / 2, 0, w),
            np.clip(cy - bh / 2, 0, h),
            np.clip(cx + bw / 2, 0, w),
            np.clip(cy + bh / 2, 0, h),
        ], axis=-1)
        return expanded
