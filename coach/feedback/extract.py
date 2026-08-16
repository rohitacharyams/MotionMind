"""extract.py — produce a vrm-quat JSON for a student-uploaded video.

For now this is a stub: it expects the caller to have already produced
a retargeted JSON from the student's video using the same pipeline as
``scripts/export_motion_json.py``. A real GPU-backed implementation
(WHAM, HMR2, PoseGPT) will replace this function later — its signature
will not change.

If the input is already a JSON file path with format='vrm-quat', we
just load and return it. Otherwise raise NotImplementedError so the
caller can fall back to a friendly error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def extract_from_video(video_path: str | Path) -> Dict[str, Any]:
    p = Path(video_path)
    if p.suffix.lower() == '.json':
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('format') == 'vrm-quat':
            return d
    raise NotImplementedError(
        'student-video mocap requires an external GPU service. '
        'For now, pre-extract via scripts/export_motion_json.py and '
        'pass the JSON path here. We will plug in WHAM/HMR2 later.'
    )
