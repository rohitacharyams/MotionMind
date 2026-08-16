"""
Pose extraction using rtmlib + ONNX Runtime with DirectML (AMD GPU).

Extracts 133 whole-body keypoints (body + hands + face + feet) per frame
using RTMPose ONNX models accelerated on AMD Radeon via DirectML.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract whole-body keypoints from video using rtmlib + DirectML.
    
    Uses RTMW whole-body model for 133-keypoint extraction via ONNX Runtime,
    accelerated on AMD GPU through the DirectML execution provider.
    """

    # Model quality modes (maps to rtmlib Wholebody.MODE)
    MODES = {
        "performance": "performance",   # rtmw-dw-x-l 384x288 (best quality)
        "balanced": "balanced",         # rtmw-dw-x-l 256x192
        "lightweight": "lightweight",   # rtmw-dw-l-m 256x192 (fastest)
    }

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self._wholebody = None

        # Determine device
        if device == "auto":
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self._use_dml = "DmlExecutionProvider" in providers
        else:
            self._use_dml = device in ("dml", "directml", "gpu", "cuda")

        self._mode = config.get("pose_extraction", {}).get("mode", "performance")
        if self._mode not in self.MODES:
            self._mode = "performance"

        logger.info(
            "PoseExtractor: mode=%s, DirectML=%s", self._mode, self._use_dml
        )

    @property
    def wholebody(self):
        """Lazy-load the rtmlib Wholebody model."""
        if self._wholebody is None:
            self._wholebody = self._init_model()
        return self._wholebody

    def _init_model(self):
        """Initialize rtmlib Wholebody with DirectML if available."""
        try:
            from rtmlib import Wholebody
        except ImportError:
            raise ImportError(
                "rtmlib is required. Install via: pip install rtmlib"
            )

        model = Wholebody(
            mode=self.MODES[self._mode],
            backend="onnxruntime",
            device="cpu",  # rtmlib device param; we patch ORT providers below
        )

        # Patch to DirectML for AMD GPU acceleration
        if self._use_dml:
            self._patch_to_directml(model)

        return model

    def _patch_to_directml(self, model):
        """Replace ORT sessions with DirectML-enabled sessions."""
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            logger.warning("DirectML not available, falling back to CPU")
            self._use_dml = False
            return

        for name in ("det_model", "pose_model"):
            model_obj = getattr(model, name)
            onnx_path = model_obj.onnx_model

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            model_obj.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )

            active = model_obj.session.get_providers()[0]
            logger.info("  %s -> %s", name, active)

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
                'keypoints': (T, N_persons, 133, 2) xy coordinates
                'scores': (T, N_persons, 133) confidence scores
                'fps': float — source video FPS
                'frame_size': (W, H)
                'n_frames': int
                'video_path': str
        """
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
            video_path, w, h, fps, total_frames,
        )

        all_keypoints = []
        all_scores = []

        iterator = range(total_frames)
        if progress:
            iterator = tqdm(iterator, desc="Extracting poses", unit="frame")

        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self.wholebody(frame)

            if len(keypoints) == 0:
                all_keypoints.append(np.zeros((0, 133, 2), dtype=np.float32))
                all_scores.append(np.zeros((0, 133), dtype=np.float32))
            else:
                all_keypoints.append(np.array(keypoints, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))

        cap.release()

        # Pad to consistent number of persons per frame
        max_persons = max(
            (k.shape[0] for k in all_keypoints), default=1
        )
        max_persons = max(max_persons, 1)

        n_frames = len(all_keypoints)
        kp_padded = np.zeros(
            (n_frames, max_persons, 133, 2), dtype=np.float32
        )
        sc_padded = np.zeros(
            (n_frames, max_persons, 133), dtype=np.float32
        )

        for i, (k, s) in enumerate(zip(all_keypoints, all_scores)):
            n = k.shape[0]
            if n > 0:
                kp_padded[i, :n] = k[:max_persons]
                sc_padded[i, :n] = s[:max_persons]

        return {
            "keypoints": kp_padded,
            "scores": sc_padded,
            "fps": fps,
            "frame_size": (w, h),
            "n_frames": n_frames,
            "video_path": video_path,
        }

    def extract_from_frame(self, frame: np.ndarray) -> dict:
        """Extract keypoints from a single BGR frame.
        
        Returns:
            dict with 'keypoints' (N, 133, 2) and 'scores' (N, 133).
        """
        keypoints, scores = self.wholebody(frame)

        if len(keypoints) == 0:
            return {
                "keypoints": np.zeros((0, 133, 2), dtype=np.float32),
                "scores": np.zeros((0, 133), dtype=np.float32),
            }

        return {
            "keypoints": np.array(keypoints, dtype=np.float32),
            "scores": np.array(scores, dtype=np.float32),
        }
