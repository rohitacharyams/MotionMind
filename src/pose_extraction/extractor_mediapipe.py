"""
Pose extraction using Google MediaPipe.

Uses MediaPipe Pose (BlazePose) for robust 33-keypoint body pose estimation,
then maps to COCO-WholeBody 133-keypoint format for compatibility with
the rest of the pipeline.

MediaPipe advantages:
- Very stable temporal tracking (built-in temporal smoothing)
- Works well on single-person videos
- Good occlusion handling
- Fast inference on CPU
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

#  MediaPipe BlazePose 33 landmarks → COCO-17 body mapping
#  BlazePose indices:
#    0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer,
#    4: right_eye_inner, 5: right_eye, 6: right_eye_outer,
#    7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right,
#    11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
#    15: left_wrist, 16: right_wrist, 17: left_pinky, 18: right_pinky,
#    19: left_index, 20: right_index, 21: left_thumb, 22: right_thumb,
#    23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee,
#    27: left_ankle, 28: right_ankle, 29: left_heel, 30: right_heel,
#    31: left_foot_index, 32: right_foot_index

# Map BlazePose → COCO-17 body keypoints
_BLAZE_TO_COCO17 = {
    0: 0,    # nose
    2: 1,    # left_eye (blazepose left_eye)
    5: 2,    # right_eye (blazepose right_eye)
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

# Map BlazePose → COCO feet keypoints (indices 17-22)
_BLAZE_TO_FEET = {
    31: 17,  # left_foot_index → left_big_toe
    31: 18,  # left_foot_index → left_small_toe (approx)
    29: 19,  # left_heel
    32: 20,  # right_foot_index → right_big_toe
    32: 21,  # right_foot_index → right_small_toe (approx)
    30: 22,  # right_heel
}


class MediaPipePoseExtractor:
    """Extract poses from video using MediaPipe Pose (BlazePose).

    Outputs 133-keypoint format compatible with the pipeline.
    Body (0-16) and feet (17-22) come from MediaPipe.
    Face (23-90) and hands (91-132) are estimated from body keypoints
    using simple heuristics (or left as low-confidence if MediaPipe
    Holistic is not used).
    """

    def __init__(self, config: dict, use_holistic: bool = False):
        self.config = config
        self.use_holistic = use_holistic

        pose_cfg = config.get("pose_extraction", {})
        self.model_complexity = pose_cfg.get("mediapipe_complexity", 2)
        self.min_detection_confidence = pose_cfg.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = pose_cfg.get("min_tracking_confidence", 0.5)
        self.static_image_mode = False

        self._pose = None
        self._holistic = None
        logger.info(
            "MediaPipePoseExtractor: complexity=%d, holistic=%s",
            self.model_complexity, self.use_holistic,
        )

    def _init_pose(self):
        """Lazy-init MediaPipe Pose."""
        import mediapipe as mp
        if self.use_holistic:
            self._holistic = mp.solutions.holistic.Holistic(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=True,
            )
        else:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=True,
                static_image_mode=self.static_image_mode,
            )

    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = -1,
        progress: bool = True,
    ) -> dict:
        """Extract keypoints from video using MediaPipe.

        Returns dict with:
            'keypoints': (T, 1, 133, 2) - xy in pixel coords
            'scores': (T, 1, 133) - confidence
            'fps': float
            'frame_size': (W, H)
            'n_frames': int
            'video_path': str
        """
        if self._pose is None and self._holistic is None:
            self._init_pose()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames > 0:
            total = min(total, max_frames)

        logger.info("Processing video: %s (%dx%d, %.1f fps, %d frames)",
                     video_path, W, H, fps, total)

        all_kps = []
        all_scores = []

        iterator = range(total)
        if progress:
            iterator = tqdm(iterator, desc="Extracting poses (MediaPipe)", unit="frame")

        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            kps, scores = self._process_frame(frame, W, H)
            all_kps.append(kps)
            all_scores.append(scores)

        cap.release()

        # Close MediaPipe resources
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

        n_frames = len(all_kps)
        kp_arr = np.array(all_kps, dtype=np.float32).reshape(n_frames, 1, 133, 2)
        sc_arr = np.array(all_scores, dtype=np.float32).reshape(n_frames, 1, 133)

        return {
            "keypoints": kp_arr,
            "scores": sc_arr,
            "fps": fps,
            "frame_size": (W, H),
            "n_frames": n_frames,
            "video_path": video_path,
        }

    def _process_frame(self, frame: np.ndarray, W: int, H: int):
        """Process a single frame, return (133, 2) keypoints and (133,) scores."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        kps_133 = np.zeros((133, 2), dtype=np.float32)
        scores_133 = np.zeros(133, dtype=np.float32)

        if self.use_holistic and self._holistic is not None:
            results = self._holistic.process(rgb)
            pose_landmarks = results.pose_landmarks
            face_landmarks = results.face_landmarks
            left_hand = results.left_hand_landmarks
            right_hand = results.right_hand_landmarks
        else:
            results = self._pose.process(rgb)
            pose_landmarks = results.pose_landmarks
            face_landmarks = None
            left_hand = None
            right_hand = None

        if pose_landmarks is None:
            return kps_133, scores_133

        lm = pose_landmarks.landmark

        # Map BlazePose 33 → COCO-17 body
        for blaze_idx, coco_idx in _BLAZE_TO_COCO17.items():
            if blaze_idx < len(lm):
                l = lm[blaze_idx]
                kps_133[coco_idx] = [l.x * W, l.y * H]
                scores_133[coco_idx] = l.visibility

        # Feet (17-22)
        feet_map = [
            (31, 17), (31, 18), (29, 19),  # left foot
            (32, 20), (32, 21), (30, 22),  # right foot
        ]
        for blaze_idx, coco_idx in feet_map:
            if blaze_idx < len(lm):
                l = lm[blaze_idx]
                kps_133[coco_idx] = [l.x * W, l.y * H]
                scores_133[coco_idx] = l.visibility

        # Face landmarks (23-90) from Holistic or estimate from body
        if face_landmarks is not None:
            # MediaPipe face mesh has 468 landmarks, map subset to 68-point format
            self._map_face_landmarks(face_landmarks, kps_133, scores_133, W, H)
        else:
            # Estimate face region from body keypoints
            self._estimate_face_from_body(kps_133, scores_133)

        # Hand landmarks (91-132) from Holistic or estimate from wrists
        if left_hand is not None:
            self._map_hand_landmarks(left_hand, kps_133, scores_133, W, H, offset=91)
        else:
            self._estimate_hand_from_wrist(kps_133, scores_133, wrist_idx=9, offset=91)

        if right_hand is not None:
            self._map_hand_landmarks(right_hand, kps_133, scores_133, W, H, offset=112)
        else:
            self._estimate_hand_from_wrist(kps_133, scores_133, wrist_idx=10, offset=112)

        return kps_133, scores_133

    def _map_face_landmarks(self, face_lm, kps, scores, W, H):
        """Map MediaPipe face mesh (468) to 68-point format (indices 23-90)."""
        # Standard 468→68 mapping (approximate):
        # Jaw: 0-16, Eyebrow L: 17-21, Eyebrow R: 22-26, Nose: 27-35
        # Eye L: 36-41, Eye R: 42-47, Lip outer: 48-59, Lip inner: 60-67
        mp468_to_68 = [
            # Jaw contour (0-16)
            10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400,
            # Left eyebrow (17-21)
            70, 63, 105, 66, 107,
            # Right eyebrow (22-26)
            336, 296, 334, 293, 300,
            # Nose bridge (27-30)
            168, 6, 197, 195,
            # Nose bottom (31-35)
            5, 4, 1, 19, 94,
            # Left eye (36-41)
            33, 160, 158, 133, 153, 144,
            # Right eye (42-47)
            362, 385, 387, 263, 373, 380,
            # Outer lip (48-59)
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78,
            # Inner lip (60-67)
            78, 95, 88, 178, 87, 14, 317, 402,
        ]

        fl = face_lm.landmark
        for i, mp_idx in enumerate(mp468_to_68):
            if mp_idx < len(fl):
                l = fl[mp_idx]
                kps[23 + i] = [l.x * W, l.y * H]
                scores[23 + i] = 0.8  # face mesh is generally reliable

    def _estimate_face_from_body(self, kps, scores):
        """Estimate basic face landmark positions from body keypoints."""
        nose = kps[0]
        l_eye = kps[1]
        r_eye = kps[2]
        l_ear = kps[3]
        r_ear = kps[4]

        if scores[0] < 0.3:
            return

        # Rough face region - place landmarks in a circle around nose
        eye_dist = np.linalg.norm(l_eye - r_eye)
        if eye_dist < 1:
            eye_dist = 30

        face_center = nose.copy()
        face_r = eye_dist * 1.5

        # Place 68 landmarks in a simple grid pattern
        for i in range(68):
            angle = (i / 68.0) * 2 * np.pi
            r = face_r * (0.3 + 0.7 * (i % 3) / 2)
            kps[23 + i] = face_center + r * np.array([np.cos(angle), np.sin(angle)])
            scores[23 + i] = 0.2  # low confidence for estimated

    def _map_hand_landmarks(self, hand_lm, kps, scores, W, H, offset):
        """Map MediaPipe hand landmarks (21) to pipeline format."""
        for i, l in enumerate(hand_lm.landmark):
            kps[offset + i] = [l.x * W, l.y * H]
            scores[offset + i] = 0.7

    def _estimate_hand_from_wrist(self, kps, scores, wrist_idx, offset):
        """Estimate hand landmark positions based on wrist and elbow."""
        wrist = kps[wrist_idx]
        if scores[wrist_idx] < 0.3:
            return

        # Elbow index
        elbow_idx = 7 if wrist_idx == 9 else 8
        elbow = kps[elbow_idx]

        forearm = wrist - elbow
        forearm_len = np.linalg.norm(forearm)
        if forearm_len < 1:
            return

        forearm_dir = forearm / forearm_len
        hand_len = forearm_len * 0.35
        perp = np.array([-forearm_dir[1], forearm_dir[0]])

        # Place 21 hand landmarks
        # 0=wrist, then 4 fingers + thumb with 4 joints each
        kps[offset + 0] = wrist
        scores[offset + 0] = scores[wrist_idx] * 0.5

        finger_dirs = [
            forearm_dir * 0.8 + perp * -0.6,  # thumb
            forearm_dir * 0.9 + perp * -0.2,  # index
            forearm_dir,                        # middle
            forearm_dir * 0.9 + perp * 0.2,   # ring
            forearm_dir * 0.8 + perp * 0.4,   # pinky
        ]

        joint_idx = 1
        for finger_i, fdir in enumerate(finger_dirs):
            fdir = fdir / (np.linalg.norm(fdir) + 1e-8)
            for j in range(4):
                t = (j + 1) / 4.0
                kps[offset + joint_idx] = wrist + fdir * hand_len * t
                scores[offset + joint_idx] = 0.15
                joint_idx += 1

        return

    def extract_from_frame(self, frame: np.ndarray) -> dict:
        """Extract keypoints from a single BGR frame."""
        if self._pose is None and self._holistic is None:
            self._init_pose()

        H, W = frame.shape[:2]
        kps, scores = self._process_frame(frame, W, H)

        return {
            "keypoints": kps.reshape(1, 1, 133, 2),
            "scores": scores.reshape(1, 1, 133),
        }
