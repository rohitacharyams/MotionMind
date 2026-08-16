"""coach.feedback — form feedback for student-uploaded videos.

End-to-end:
  1. Student uploads a phone video performing a clip (or freestyle).
  2. extract.py runs a mocap inference service that returns vrm-quat
     JSON in the same schema as our retargeted catalog. (Stub for now —
     calls out to an external GPU service; the local fallback assumes
     the user already produced a vrm-quat file with our own retargeter.)
  3. compare.py time-aligns student to reference with DTW on a few key
     joint angles, then computes per-joint per-frame angular error.
  4. writer.py asks an LLM to translate the numeric diff + the clip's
     metadata key_cues / common_mistakes into 2-3 actionable sentences.

Nothing here calls an external GPU at runtime by default — extract is
a stub. Plug in WHAM/HMR2/PoseGPT later.
"""
