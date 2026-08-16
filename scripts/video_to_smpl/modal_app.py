"""modal_app.py — Modal Serverless GPU worker: dance video → SMPL (+ keyframes).

Drop-in replacement for the RunPod worker (runpod_handler.py). Same GVHMR
pipeline, same output shape, same webhook contract — but on Modal, which:
  * scales to ZERO when idle (you pay only for GPU-seconds actually used), and
  * gives $30/month of free compute on the Starter plan (≈500+ short clips/mo).

Architecture (mirrors the RunPod async + webhook flow so the Flask side is
unchanged):

    the host app  ──POST /submit──▶  submit() web endpoint (CPU, instant)
                                       │  .spawn()
                                       ▼
                              _run_and_callback()  (GPU, GVHMR inference)
                                       │  when done
                                       ▼
                     POST {status, output} ──▶  the host app /api/learn/webhook/runpod

The `output` dict is byte-identical to the RunPod worker's, so
`finalize_from_worker()` needs no changes.

────────────────────────────────────────────────────────────────────────────
ONE-TIME SETUP (run locally, needs a free Modal account):

    pip install modal
    modal token new                      # opens browser, links your account

    # license-gated GVHMR checkpoints + SMPL/SMPL-X bodies go in a Modal Volume
    # (free academic registration — see GVHMR docs/INSTALL.md for the files):
    modal volume create gvhmr-checkpoints
    modal volume put gvhmr-checkpoints  ./checkpoints  /inputs

    modal deploy modal_app.py            # prints the submit endpoint URL

Then set on the host app App Service:
    GPU_PROVIDER=modal
    MODAL_ENDPOINT_URL=<the submit URL modal deploy printed>
    MODAL_SUBMIT_TOKEN=<any long random string; also set below as a Modal secret>

Cost: a ~60–90s clip runs ~60–120s on a T4 ≈ $0.01–0.02 per video. Idle = $0.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os

import modal

APP_NAME = 'dance-v2smpl'

# Build the GPU image from the SAME Dockerfile the RunPod worker uses, so GVHMR
# + gvhmr_to_aist.py + the frame sampler are all present and identical. The
# license-gated checkpoints are NOT baked in — they live in a Modal Volume
# (mounted at /app/GVHMR/inputs) so we never rebuild the image to update them.
image = (
    modal.Image.from_dockerfile('Dockerfile')
    .pip_install('requests', 'fastapi[standard]')
)

# Persisted checkpoints (SMPL bodies + GVHMR weights). Populate once with:
#   modal volume put gvhmr-checkpoints ./checkpoints /inputs
checkpoints = modal.Volume.from_name('gvhmr-checkpoints', create_if_missing=True)

app = modal.App(APP_NAME)

# Shared secret so only our backend can call the submit endpoint.
try:
    _submit_secret = modal.Secret.from_name('modal-submit-token')
except Exception:  # not created yet — deploy still works, auth just skipped
    _submit_secret = modal.Secret.from_dict({'MODAL_SUBMIT_TOKEN': ''})


def _process(inp: dict) -> dict:
    """Reuse the exact GVHMR pipeline from the RunPod worker."""
    import sys
    sys.path.insert(0, '/app/worker')
    from runpod_handler import process  # noqa: WPS433 (baked into the image)
    return process(inp)


@app.function(
    image=image,
    gpu='T4',                      # cheapest that fits GVHMR; bump to 'A10' if OOM
    volumes={'/app/GVHMR/inputs': checkpoints},
    timeout=900,                   # 15 min hard ceiling per clip
    retries=1,
)
def _run_and_callback(inp: dict, webhook_url: str) -> dict:
    """Run GVHMR on the GPU, then POST the result to the the host app webhook in
    the exact shape the RunPod webhook handler expects."""
    import requests
    try:
        out = _process(inp)
        status = 'COMPLETED' if out.get('ok') else 'FAILED'
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {'ok': False, 'error': str(e),
               'trace': traceback.format_exc()[-2000:]}
        status = 'FAILED'
    if webhook_url:
        try:
            requests.post(webhook_url, json={'status': status, 'output': out},
                          timeout=60)
        except Exception:
            pass
    return {'status': status, 'ok': out.get('ok', False)}


@app.function(image=image, secrets=[_submit_secret], timeout=60)
@modal.fastapi_endpoint(method='POST')
def submit(payload: dict):
    """CPU web endpoint the backend calls. Validates the shared token, spawns
    the GPU job (which fires the webhook when done), and returns instantly with
    a job id — mirroring RunPod's async /run."""
    from fastapi import HTTPException

    want = os.environ.get('MODAL_SUBMIT_TOKEN', '')
    got = (payload or {}).get('token', '')
    if want and got != want:
        raise HTTPException(status_code=403, detail='bad token')

    inp = (payload or {}).get('input') or {}
    webhook_url = (payload or {}).get('webhook') or ''
    if not (inp.get('video_url') or inp.get('video_b64')):
        raise HTTPException(status_code=400, detail='need video_url or video_b64')

    call = _run_and_callback.spawn(inp, webhook_url)
    return {'id': call.object_id, 'status': 'IN_QUEUE'}
