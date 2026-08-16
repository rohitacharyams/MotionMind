"""server.py — FastAPI backend for the AI Coach pilot.

Routes
------
GET  /                            → static/coach.html
GET  /static/*                    → JS / VRM / etc.
GET  /api/motion/list             → list of safe clips
GET  /api/motion/data/{id}.json   → axis-angle + trans for a clip (browser MotionPlayer eats this)
GET  /api/characters              → VRM registry
GET  /api/limits                  → physics limits JSON (browser runtime guard)
GET  /api/speech/token            → ephemeral Azure Speech token (browser SDK)
WS   /ws/agent                    → conversational loop: text in → tool calls + speech back
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import (FastAPI, File, Header, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coach import motion_index
from coach.physics_validator import (clamp_pose, export_limits_json,
                                     validate_motion)
from coach import metadata as motion_metadata
from coach import storage as journey_storage
from coach import notifications as coach_notifications

load_dotenv(Path(__file__).resolve().parent / '.env')

ROOT = Path(__file__).resolve().parents[1]
COACH = Path(__file__).resolve().parent
STATIC = COACH / 'static'
REGISTRY = ROOT / 'data' / 'characters' / 'registry.json'

# Optional analytics injection point. The open-source build ships with NO
# third-party analytics/ads. If you want to add your own (GA4, Amplitude,
# Plausible, ...), return your <script> tags from this function; they are
# injected into the served coach.html <head>.
def _head_inject() -> str:
    return ''

AZURE_KEY = os.getenv('AZURE_SPEECH_KEY', '')
AZURE_REGION = os.getenv('AZURE_SPEECH_REGION', 'eastus')
GROQ_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

app = FastAPI(title='Dance.AI Coach')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])

# v225: wire Azure Application Insights (was configured via
# APPLICATIONINSIGHTS_CONNECTION_STRING but NEVER instrumented → the AI
# component sat EMPTY). azure-monitor-opentelemetry auto-instruments FastAPI so
# every request (path, status, duration) flows to App Insights, giving us
# queryable per-URL 4xx/5xx + latency without any client SDK. Fully guarded: if
# the package or connection string is missing it's a silent no-op and boot is
# never affected.
try:
    if os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING', '').strip():
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(logger_name='dance-coach')
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        print('[appinsights] instrumented')
except Exception as _e:                                    # noqa: BLE001
    print(f'[appinsights] not enabled: {_e!r}')


@app.middleware('http')
async def _www_to_apex(request: 'Request', call_next):
    """301 www.dancecoach.fit -> dancecoach.fit (apex is canonical). Keeps SEO
    authority on ONE hostname and avoids duplicate-content. Only fires for the
    www host; everything else passes through untouched."""
    host = (request.headers.get('host') or '').lower()
    if host.startswith('www.dancecoach.fit'):
        from fastapi.responses import RedirectResponse as _RR
        url = request.url.replace(netloc='dancecoach.fit', scheme='https')
        return _RR(str(url), status_code=301)
    return await call_next(request)


@app.on_event('startup')
async def _warm_caches() -> None:
    """v33c: pre-load the heavy modules on container startup so the
    FIRST websocket message after a cold start doesn't pay the import
    + parse cost. Without this, the user types 'hi' and waits ~3-4 s
    for ontology.yaml + motion_index + motion_meta to lazy-load."""
    try:
        # Force motion index scan + caches.
        _ = motion_index.list_motions()
        # Pre-build the system-prompt template (loads + parses ontology.yaml).
        from coach.choreographer.prompts import system_prompt as _sp
        _ = _sp(None)
        # Pre-import the agent module so the OpenAI / Groq clients are
        # ready to fire on the first request.
        import coach.choreographer.agent  # noqa: F401
    except Exception as e:                                       # noqa: BLE001
        print(f'[startup] warm-up failed (non-fatal): {e}')


@app.on_event('startup')
async def _start_reminder_loop() -> None:
    """v185: reminders.py's reminder_loop() was fully built (docstring
    literally says 'started from server.py startup') but never actually
    started anywhere -- it's been dead code since it was written. Fires
    the daily-reminder background scan every 5 min. Exits silently on its
    own if neither Web Push nor Notification Hubs is configured (see
    reminders.reminder_loop docstring), so this is a safe no-op until
    those channels are set up."""
    try:
        from coach import reminders
        asyncio.create_task(reminders.reminder_loop(_journey_store))
    except Exception as e:                                       # noqa: BLE001
        print(f'[startup] reminder loop not started (non-fatal): {e}')


if STATIC.exists():
    # html=True → a directory URL like /static/blog/ auto-serves its
    # index.html (instead of 404 {"detail":"Not Found"}), so the blog is
    # readable at the clean folder URL, not just the explicit .html path.
    app.mount('/static', StaticFiles(directory=str(STATIC), html=True), name='static')

# v193: serve the blog at the CLEAN URL /dance/blog/ (no "static" in the path)
# for nicer, more shareable, SEO-friendly links. The old /static/blog/ URLs
# still resolve (the /static mount above stays), and each post's <link
# rel="canonical"> points at the /blog/ URL so search engines index one.
BLOG_DIR = STATIC / 'blog'
if BLOG_DIR.exists():
    app.mount('/blog', StaticFiles(directory=str(BLOG_DIR), html=True), name='blog')

# Backing-track audio. AIST clips reference one of the 60 mAA?.mp3 files
# (parsed from the clip id); CMU clips get one of 3 generic loops.
AUDIO_DIR = ROOT / 'data' / 'audio'
if AUDIO_DIR.exists():
    app.mount('/audio', StaticFiles(directory=str(AUDIO_DIR)), name='audio')


# ─── /mockInterviews — AI Mock Interviewer vertical (same app service) ──
# The interview product is a self-contained FastAPI app (interview/server.py)
# reusing the SAME Gemini Live S2S stack. We mount it here so studioos.fit
# can expose it at /mockInterviews on the SAME container as /dance — no new
# App Service. The mount is GUARDED: if the interview package isn't present
# in the image (older builds) or fails to import, the dance app is unaffected.
#
# nginx (studioos.fit) must forward the WHOLE prefix INCLUDING *.js — use
#   location ^~ /mockInterviews/ { proxy_pass http://<this-app>; <ws headers> }
# The ^~ is essential: it stops the bare-*.js regex hijack from stealing
# /mockInterviews/static/*.js (the interview frontend loads real .js files).
# ─── /mockInterviews — MOVED to its own app + domain (praxari.com) ──────
# The AI mock-interview product used to be MOUNTED here on the same container.
# It now lives on a fully separate app (praxari-app) at its own domain,
# https://praxari.com — so a coach deploy can never affect it and vice-versa.
# We keep the old studioos.fit/mockInterviews/* URLs working by permanently
# (301) redirecting them to the canonical praxari.com home, so shared links,
# bookmarks and search results don't break. No interview code runs here anymore.
MOCK_PREFIX = os.getenv('MOCK_PREFIX', '/mockInterviews')
PRAXARI_URL = os.getenv('PRAXARI_URL', 'https://praxari.com').rstrip('/')


async def _praxari_redirect(request: 'Request', subpath: str = ''):
    from fastapi.responses import RedirectResponse
    dest = PRAXARI_URL + ('/' + subpath if subpath else '/')
    q = request.url.query
    if q:
        dest += '?' + q
    return RedirectResponse(dest, status_code=301)


from fastapi import Request  # noqa: E402  (local import; keeps coach imports intact)
for _pfx in {MOCK_PREFIX, MOCK_PREFIX.lower()}:
    app.add_api_route(_pfx, _praxari_redirect, methods=['GET', 'HEAD'],
                      include_in_schema=False)
    app.add_api_route(_pfx + '/{subpath:path}', _praxari_redirect,
                      methods=['GET', 'HEAD'], include_in_schema=False)
print(f'[redirect] {MOCK_PREFIX}/* -> {PRAXARI_URL} (301)')


@app.get('/')
def index():
    f = STATIC / 'coach.html'
    if not f.exists():
        raise HTTPException(503, 'coach.html missing')
    inject = _head_inject()
    if inject:
        try:
            html = f.read_text(encoding='utf-8')
            html = html.replace('</head>', inject + '\n</head>', 1)
            return HTMLResponse(html)
        except Exception:                                        # noqa: BLE001
            pass
    return FileResponse(str(f))


# ─── Legal / compliance pages (Google Play REQUIRES reachable URLs) ────
# These are plain static HTML in coach/static/, but Play declares the clean
# extension-less URLs https://studioos.fit/dance/privacy and
# /dance/delete-account. The external nginx forwards /dance/<anything> to
# this app (it only special-cases *.js and the /static /asset /api /healthz
# prefixes), so a bare /privacy hits FastAPI — which previously had NO route
# and returned {"detail":"Not Found"}. Serving the files directly from these
# canonical routes makes the Play-declared URLs permanently valid. Do NOT
# replace these with redirects to another domain.
def _serve_static_page(filename: str):
    f = STATIC / filename
    if not f.exists():
        raise HTTPException(404, f'{filename} missing')
    # text/html + no-cache-ish so edits go live on next deploy without a
    # stale CDN copy; these pages must be "not editable / always reachable".
    return FileResponse(str(f), media_type='text/html; charset=utf-8')


@app.get('/privacy', include_in_schema=False)
@app.get('/privacy.html', include_in_schema=False)
def privacy_page():
    return _serve_static_page('privacy.html')


@app.get('/delete-account', include_in_schema=False)
@app.get('/delete-account.html', include_in_schema=False)
@app.get('/account-deletion', include_in_schema=False)
def delete_account_page():
    return _serve_static_page('delete-account.html')


@app.get('/terms', include_in_schema=False)
@app.get('/terms.html', include_in_schema=False)
def terms_page():
    return _serve_static_page('terms.html')


# ─── SEO landing pages (dancecoach.fit) ────────────────────────────────
# Static, crawlable HTML that ranks for high-intent queries and funnels into
# the coach. Canonicals point at dancecoach.fit. Served extension-less so the
# URLs are clean + shareable.
@app.get('/ai-dance-coach', include_in_schema=False)
def seo_ai_dance_coach():
    return _serve_static_page('seo/ai-dance-coach.html')


@app.get('/learn-hip-hop-free', include_in_schema=False)
def seo_learn_hip_hop_free():
    return _serve_static_page('seo/learn-hip-hop-free.html')


@app.get('/learn-to-dance-at-home', include_in_schema=False)
def seo_learn_to_dance_at_home():
    return _serve_static_page('seo/learn-to-dance-at-home.html')


@app.get('/learn-house-dance-free', include_in_schema=False)
def seo_learn_house_dance_free():
    return _serve_static_page('seo/learn-house-dance-free.html')


@app.get('/how-to-breakdance-for-beginners', include_in_schema=False)
def seo_how_to_breakdance():
    return _serve_static_page('seo/how-to-breakdance-for-beginners.html')


@app.get('/learn-popping-free', include_in_schema=False)
def seo_learn_popping_free():
    return _serve_static_page('seo/learn-popping-free.html')


@app.get('/learn-locking-free', include_in_schema=False)
def seo_learn_locking_free():
    return _serve_static_page('seo/learn-locking-free.html')


@app.get('/free-online-dance-classes', include_in_schema=False)
def seo_free_online_dance_classes():
    return _serve_static_page('seo/free-online-dance-classes.html')


@app.get('/dance-workout-at-home', include_in_schema=False)
def seo_dance_workout_at_home():
    return _serve_static_page('seo/dance-workout-at-home.html')


@app.get('/how-to-learn-dance-online', include_in_schema=False)
def seo_how_to_learn_dance_online():
    return _serve_static_page('seo/how-to-learn-dance-online.html')


@app.get('/robots.txt', include_in_schema=False)
def robots_txt():
    f = STATIC / 'robots.txt'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type='text/plain')


@app.get('/sitemap.xml', include_in_schema=False)
def sitemap_xml():
    f = STATIC / 'sitemap.xml'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type='application/xml')


@app.get('/ads.txt', include_in_schema=False)
def ads_txt():
    f = STATIC / 'ads.txt'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type='text/plain')


# ─── Brand assets at crawlable ROOT paths ──────────────────────────────
# Google's search-result favicon crawler fetches /favicon.ico (and ignores
# data-URI favicons entirely), and social scrapers fetch the og:image at its
# absolute URL. Serving these from the domain root (not just /static/) is what
# makes the logo appear in Google results and link previews. Explicit routes
# (NOT a catch-all) so they never shadow the app/API/WebSocket routes below.
def _serve_root_asset(fname: str, mt: str):
    f = STATIC / fname
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type=mt,
                        headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/favicon.ico', include_in_schema=False)
def favicon_ico():
    return _serve_root_asset('favicon.ico', 'image/x-icon')


@app.get('/favicon.svg', include_in_schema=False)
def favicon_svg():
    return _serve_root_asset('favicon.svg', 'image/svg+xml')


@app.get('/apple-touch-icon.png', include_in_schema=False)
@app.get('/apple-touch-icon-precomposed.png', include_in_schema=False)
def apple_touch_icon():
    return _serve_root_asset('apple-touch-icon.png', 'image/png')


@app.get('/icon-192.png', include_in_schema=False)
def icon_192():
    return _serve_root_asset('icon-192.png', 'image/png')


@app.get('/icon-512.png', include_in_schema=False)
def icon_512():
    return _serve_root_asset('icon-512.png', 'image/png')


@app.get('/favicon-96.png', include_in_schema=False)
def favicon_96():
    return _serve_root_asset('favicon-96.png', 'image/png')


@app.get('/og-cover.png', include_in_schema=False)
def og_cover_png():
    return _serve_root_asset('og-cover.png', 'image/png')


@app.get('/llms.txt', include_in_schema=False)
def llms_txt():
    return _serve_root_asset('llms.txt', 'text/plain; charset=utf-8')


# v225: catch two bare-root paths that older/cached clients still request and
# that were 404ing (contributing to the elevated 4xx rate). The PWA manifest is
# linked as static/manifest.json, but some browsers + previously-cached HTML ask
# for /manifest.json at the domain root; and an old installed service worker can
# re-request /service-worker.js at root. Serve both so they stop 404ing.
@app.get('/manifest.json', include_in_schema=False)
def manifest_root():
    return _serve_root_asset('manifest.json', 'application/manifest+json')


@app.get('/service-worker.js', include_in_schema=False)
def service_worker_root():
    f = STATIC / 'service-worker.js'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type='text/javascript',
                        headers={'Service-Worker-Allowed': '/',
                                 'Cache-Control': 'no-cache'})


# Workaround: the studioos.fit nginx reverse proxy intercepts ALL `*.js` URLs
# and serves them from its own filesystem (not forwarded to this container).
# We expose JS modules under an extension-less path so nginx forwards them.
@app.get('/m/{name}')
def js_module(name: str):
    if not name.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(404)
    f = STATIC / f'{name}.js'
    if not f.exists():
        raise HTTPException(404)
    headers = {}
    # v137: the service worker is served via this proxied /m/ route (the
    # external nginx only forwards /m/, /api/, /static/, /asset/, /healthz —
    # NOT a bare /sw.js). A SW served from /dance/m/ would normally be capped
    # to scope /dance/m/, but we need it to control the whole /dance/ app so
    # it can cache the avatar VRM + CDN libs. Service-Worker-Allowed widens
    # the permitted scope above the script path.
    if name in ('service-worker', 'sw'):
        headers['Service-Worker-Allowed'] = '/'
    else:
        # v-ux10: let the browser serve the JS from cache on warm/repeat visits
        # instead of re-downloading ~250 KB every load. FileResponse still sends
        # ETag + Last-Modified, so after max-age the browser revalidates (cheap
        # 304); stale-while-revalidate keeps the next load instant while it
        # refreshes in the background. Short max-age bounds staleness after a
        # deploy. The service worker (above) is intentionally never cached.
        headers['Cache-Control'] = 'public, max-age=300, stale-while-revalidate=86400'
    return FileResponse(str(f), media_type='text/javascript', headers=headers)


# v135: the ONE service worker (push notifications + on-device asset caching)
# lives in service-worker.js. A service worker's scope is limited to the path
# it is served from, so to control the whole /dance/ app we serve it at the
# app-root path /sw.js AND send Service-Worker-Allowed: / for the wider scope.
@app.get('/sw.js')
def service_worker():
    f = STATIC / 'service-worker.js'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(
        str(f), media_type='text/javascript',
        headers={'Service-Worker-Allowed': '/',
                 'Cache-Control': 'no-cache'})


# Same trick for binary stage assets (`stage.glb`). The nginx static-file
# regex is unpredictable; an extension-less path always proxies through.
@app.get('/asset/{name}')
def asset_blob(name: str):
    if not name.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(404)
    for ext, mt in (('.glb', 'model/gltf-binary'),
                    ('.gltf', 'model/gltf+json'),
                    ('.bin', 'application/octet-stream')):
        f = STATIC / f'{name}{ext}'
        if f.exists():
            return FileResponse(str(f), media_type=mt)
    raise HTTPException(404)


@app.get('/healthz')
def health():
    # v34: include safety summary so we can see at-a-glance how many
    # clips passed the physics validator.
    motions = motion_index.list_motions()
    safety_counts: Dict[str, int] = {}
    for m in motions:
        s = m.get('safety') or 'unknown'
        safety_counts[s] = safety_counts.get(s, 0) + 1
    return {
        'ok': True,
        'azure_speech': bool(AZURE_KEY),
        'groq': bool(GROQ_KEY),
        'motions': len(motions),
        'safety': safety_counts,
        'static_dir': str(STATIC),
        'static_files': sorted([p.name for p in STATIC.iterdir()]) if STATIC.exists() else None,
    }


@app.get('/api/voice/status')
def voice_status():
    """Tells the browser whether the live speech-to-speech path is
    available so it can show the toggle + fall back when it isn't.
    v120: only expose `enabled` — never leak the model/provider/voice."""
    try:
        from coach import gemini_live
        st = gemini_live.gemini_status()
        return {'enabled': bool(st.get('enabled'))}
    except Exception:                                            # noqa: BLE001
        return {'enabled': False}


@app.get('/api/limits')
def get_limits():
    return export_limits_json()


@app.get('/api/characters')
def get_characters():
    if not REGISTRY.exists():
        return {'characters': []}
    return json.loads(REGISTRY.read_text(encoding='utf-8'))


# ── Internal machine-to-machine API (studio-Os learn pipeline) ─────────────
# These are called server-to-server by the studio-Os backend after RunPod
# returns raw SMPL. They are guarded by a shared secret (DAN_INTERNAL_TOKEN),
# NOT a user JWT. If the token is unset the endpoints are disabled (503) so a
# misconfigured box never exposes the retarget/push surface publicly.
DAN_INTERNAL_TOKEN = os.getenv('DAN_INTERNAL_TOKEN', '')
BUILD_CLIP = ROOT / 'scripts' / 'video_to_smpl' / 'build_clip.py'


def _require_internal(token: str) -> None:
    if not DAN_INTERNAL_TOKEN:
        raise HTTPException(503, 'internal API disabled (DAN_INTERNAL_TOKEN unset)')
    if token != DAN_INTERNAL_TOKEN:
        raise HTTPException(401, 'bad internal token')


class BuildClipRequest(BaseModel):
    name: str                       # target clip id (motion_cache/<name>.json)
    smpl_b64: Optional[str] = None  # base64 of the AIST-style SMPL .pkl
    smpl_url: Optional[str] = None  # OR an https url to fetch the .pkl from
    vrm: Optional[str] = None       # override VRM rig path (else build_clip default)
    allow_fail: bool = False        # install even if the physics gate fails


@app.post('/api/learn/build')
async def learn_build(req: BuildClipRequest,
                      x_internal_token: str = Header(default='')):
    """Retarget a RunPod-produced SMPL .pkl into a playable coach clip.

    Runs the EXISTING build_clip.py chain (export → sign-fix → physics gate →
    install into motion_cache) and returns the metadata studio-Os needs to mark
    a LearnJob READY. This is the always-on-dan-box half of the pipeline."""
    _require_internal(x_internal_token)

    # Only allow safe clip ids (they become filenames + motion ids).
    name = ''.join(c for c in req.name if c.isalnum() or c in ('_', '-'))
    if not name:
        raise HTTPException(400, 'invalid name')

    import base64 as _b64
    tmp = Path(tempfile.mkdtemp(prefix='learn_build_'))
    try:
        pkl = tmp / f'{name}.pkl'
        if req.smpl_b64:
            pkl.write_bytes(_b64.b64decode(req.smpl_b64))
        elif req.smpl_url:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.get(req.smpl_url)
                r.raise_for_status()
                pkl.write_bytes(r.content)
        else:
            raise HTTPException(400, 'provide smpl_b64 or smpl_url')

        cmd = [sys.executable, str(BUILD_CLIP), '--aist', str(pkl), '--name', name]
        if req.vrm:
            cmd += ['--vrm', req.vrm]
        if req.allow_fail:
            cmd.append('--allow-fail')
        proc = await asyncio.to_thread(
            subprocess.run, cmd, cwd=str(ROOT),
            capture_output=True, text=True)
        installed = (COACH / 'motion_cache' / f'{name}.json')
        if proc.returncode != 0 or not installed.exists():
            tail = (proc.stderr or proc.stdout or '')[-1500:]
            return JSONResponse(
                {'ok': False, 'passed': False, 'error': tail}, status_code=422)

        data = json.loads(installed.read_text(encoding='utf-8'))
        # New clip on disk — drop the cached motion list so it appears at once.
        try:
            motion_index.list_motions.cache_clear()  # type: ignore[attr-defined]
        except Exception:                                            # noqa: BLE001
            pass
        return {
            'ok': True, 'passed': True, 'motion_id': name,
            'n_frames': int(data.get('n_frames') or 0),
            'fps': float(data.get('fps') or 0.0),
            'duration_s': float(data.get('duration_s') or 0.0),
        }
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class LearnStepsRequest(BaseModel):
    motion_id: str
    steps: List[Dict[str, Any]]     # [{index,label,cue,start_s,end_s,locked?}]
    title: Optional[str] = None
    total: int = 0
    unlocked: int = 0


@app.post('/api/learn/steps')
async def learn_steps(req: LearnStepsRequest,
                      x_internal_token: str = Header(default='')):
    """Store a learned choreography's step breakdown so the coach's break_down
    tool teaches those named steps with the free/paid gate baked in. Called by
    studio-Os after its vision-model segmentation (and again on upgrade)."""
    _require_internal(x_internal_token)
    name = ''.join(c for c in req.motion_id if c.isalnum() or c in ('_', '-'))
    if not name:
        raise HTTPException(400, 'invalid motion_id')
    meta = motion_metadata.set_learned_steps(
        name, req.steps, title=req.title or '',
        total=req.total, unlocked=req.unlocked)
    return {'ok': True, 'motion_id': name,
            'stored': len(meta.get('learned_steps') or []),
            'unlocked': meta.get('learned_unlocked')}
    user_id: str
    title: str
    body: str
    url: Optional[str] = None


@app.post('/api/internal/notify-push')
async def internal_notify_push(req: PushRequest,
                               x_internal_token: str = Header(default='')):
    """Fire a push notification (web-push / FCM) to a user's registered devices.
    Called by studio-Os when a learn job is READY. Best-effort; never raises."""
    _require_internal(x_internal_token)
    try:
        subs = _journey_store.list_push_subscriptions(req.user_id)
        sub_dicts = [r.get('subscription') for r in subs
                     if isinstance(r.get('subscription'), dict)]
        dev_rows = _journey_store.list_device_tokens(req.user_id)
        dev_tokens = [r.get('registration_id') for r in dev_rows
                      if r.get('registration_id')]
        res = await asyncio.to_thread(
            coach_notifications.send_to_user,
            req.user_id, req.title, req.body, req.url or '/',
            sub_dicts, f'user:{req.user_id}', None, dev_tokens)
        for tok in getattr(res, 'stale_tokens', []) or []:
            try:
                _journey_store.delete_device_token(req.user_id, tok)
            except Exception:                                        # noqa: BLE001
                pass
        return {'ok': True, **res.as_dict()}
    except Exception as e:                                           # noqa: BLE001
        return {'ok': False, 'error': str(e)}


@app.get('/api/me/progress')
async def get_me_progress(authorization: str = Header(default='')):
    """Streak / total minutes / sessions-completed for the signed-in user.
    The account menu (coach.js) has been fetching this since v69 — it just
    404'd every time because the route never existed. Never raises: any
    failure just yields zeroed progress so the UI shows dashes gracefully."""
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return {'user': None, 'progress': _default_progress()}
    u = await _fetch_studioos_user(token)
    if not u:
        return {'user': None, 'progress': _default_progress()}
    user_id = str(u.get('id') or u.get('user_id') or '')
    progress = _default_progress()
    profile = {}
    if user_id:
        try:
            doc = _journey_store.load_journey(user_id) or {}
            p = doc.get('progress')
            if isinstance(p, dict):
                progress.update(p)
            pr = doc.get('profile')
            if isinstance(pr, dict):
                profile = pr
        except Exception as e:                                    # noqa: BLE001
            print(f'[server] get_me_progress load failed: {e}', file=sys.stderr)
    return {'user': {'id': user_id,
                     'name': u.get('name') or u.get('display_name'),
                     'email': u.get('email') or '',
                     'created_at': u.get('created_at') or u.get('createdAt') or ''},
            'progress': progress,
            'profile': profile}


class _SaveSessionBody(BaseModel):
    minutes: float = 0
    template_id: str = ''
    style: str = ''


@app.post('/api/me/save-session')
async def save_session(body: _SaveSessionBody,
                       authorization: str = Header(default='')):
    """v220: persist a just-finished session to the now-signed-in user's
    journey. Called right after a delighted (👍) anonymous user creates an
    account from the end-of-session card, so the streak/minutes they just
    earned aren't lost. Auth via bearer token → studio-Os user. Best-effort."""
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return JSONResponse(status_code=401, content={'saved': False, 'error': 'no_token'})
    u = await _fetch_studioos_user(token)
    if not u:
        return JSONResponse(status_code=401, content={'saved': False, 'error': 'invalid_token'})
    user_id = str(u.get('id') or u.get('user_id') or '')
    minutes = 0.0
    try:
        minutes = max(0.0, min(180.0, float(body.minutes or 0)))
    except Exception:
        minutes = 0.0
    if user_id and minutes > 0:
        template_id = (body.template_id or '').strip()[:80]
        if not template_id and body.style:
            template_id = f'quick5_{body.style}'.strip()[:80]
        _record_session_complete(user_id, minutes, template_id)
        return {'saved': True}
    return {'saved': False, 'error': 'nothing_to_save'}


# ─── v227: "Upload your own video to learn" — same-origin proxy ────────
# The full learn-from-video pipeline already lives on studio-Os
# (POST /api/learn/submit: stores the video, creates a LearnJob, dispatches to
# the GPU pipeline or emails the owner, and later emails the user "your dance
# is ready"). But the coach is served cross-origin (dancecoach.fit) from that
# API, so the browser can't POST a big multipart video there directly without
# CORS/pre-flight pain. This proxy accepts the upload SAME-ORIGIN
# (dancecoach.fit/api/learn/upload) and forwards it to studio-Os with the
# user's bearer token — mirroring the /api/track proxy. Auth is enforced by
# studio-Os (@jwt_required on /submit); we require a token here too so an
# anonymous user is told to sign in BEFORE we buffer a large file.
@app.post('/api/learn/upload')
async def learn_upload(video: UploadFile = File(...),
                       title: str = '',
                       authorization: str = Header(default='')):
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return JSONResponse(status_code=401,
                            content={'ok': False, 'error': 'sign_in_required'})
    # Read + hard-cap the upload (120 MB) so a huge file can't exhaust memory.
    data = await video.read()
    size = len(data or b'')
    if size == 0:
        return JSONResponse(status_code=400,
                            content={'ok': False, 'error': 'empty_file'})
    if size > 120 * 1024 * 1024:
        return JSONResponse(status_code=413,
                            content={'ok': False, 'error': 'file_too_large',
                                     'message': 'Please upload a shorter clip '
                                                '(under ~120 MB / a reel).'})
    try:
        files = {'video': (video.filename or 'upload.mp4', data,
                           video.content_type or 'video/mp4')}
        form = {'rights_confirmed': '1', 'title': (title or '')[:120]}
        async with httpx.AsyncClient(timeout=180.0) as cx:
            r = await cx.post(
                f'{STUDIOOS_API}/api/learn/submit',
                headers={'Authorization': f'Bearer {token}'},
                files=files, data=form)
        try:
            payload = r.json()
        except Exception:                                        # noqa: BLE001
            payload = {}
        if r.status_code in (200, 201):
            return {'ok': True, 'job': payload.get('job'),
                    'eligibility': payload.get('eligibility'),
                    'message': "Got it! We're processing your video. You'll get "
                               "a notification and an email when your coach is "
                               "ready to teach you these steps — meanwhile, keep "
                               "dancing. \U0001F49C"}
        # 402 = free quota used / needs a plan; 429 = rate limited. Pass through
        # so the browser can show the right upsell / message.
        return JSONResponse(status_code=r.status_code,
                            content={'ok': False,
                                     'error': payload.get('error') or 'submit_failed',
                                     'upgrade_required': payload.get('upgrade_required'),
                                     'rate_limited': payload.get('rate_limited')})
    except Exception as e:                                        # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={'ok': False, 'error': f'upstream: {e}'})


# ─── v195: LEARN tab — structured Hip-Hop + House curriculum ──────────
# Landing users didn't know WHAT to learn. These routes power the Learn
# tab: a static curriculum (two tracks, foundational moves w/ pedagogy)
# plus per-user lesson progress persisted in the journey doc so learning
# compounds across sessions.
@app.get('/api/curriculum')
async def get_curriculum_route():
    """The two lesson tracks (Hip-Hop + House). Static, cacheable."""
    from coach import curriculum
    return curriculum.get_curriculum()


@app.get('/api/me/lessons')
async def get_me_lessons(authorization: str = Header(default='')):
    """Per-user lesson progress map {lesson_id: {status, best_score,
    worst_keypoint, attempts, updated_at}}. Anonymous -> empty (progress
    only saved for signed-in users)."""
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return {'lessons': {}}
    u = await _fetch_studioos_user(token)
    if not u:
        return {'lessons': {}}
    user_id = str(u.get('id') or u.get('user_id') or '')
    lessons: Dict[str, Any] = {}
    if user_id:
        try:
            doc = _journey_store.load_journey(user_id) or {}
            ls = doc.get('lessons')
            if isinstance(ls, dict):
                lessons = ls
        except Exception as e:                                    # noqa: BLE001
            print(f'[server] get_me_lessons load failed: {e}', file=sys.stderr)
    return {'lessons': lessons}


class LessonProgress(BaseModel):
    lesson_id: str
    status: str = ''            # 'learning' | 'practiced' | 'mastered'
    best_score: Optional[float] = None
    worst_keypoint: str = ''
    note: str = ''


@app.post('/api/me/lessons')
async def post_me_lessons(body: LessonProgress,
                          authorization: str = Header(default='')):
    """Record progress on one lesson. Merges into the journey doc's
    `lessons` map. Best-effort; never raises out."""
    from coach import curriculum
    if not curriculum.get_lesson(body.lesson_id):
        return {'ok': False, 'reason': 'unknown_lesson'}
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return {'ok': False, 'reason': 'sign_in_required'}
    u = await _fetch_studioos_user(token)
    if not u:
        return {'ok': False, 'reason': 'sign_in_required'}
    user_id = str(u.get('id') or u.get('user_id') or '')
    if not user_id:
        return {'ok': False, 'reason': 'no_user'}
    try:
        doc = _journey_store.load_journey(user_id) or {}
        doc.setdefault('user_id', user_id)
        lessons = doc.get('lessons')
        if not isinstance(lessons, dict):
            lessons = {}
        cur = lessons.get(body.lesson_id)
        if not isinstance(cur, dict):
            cur = {'status': '', 'attempts': 0, 'best_score': None}
        # Status only ever advances: learning < practiced < mastered.
        rank = {'': 0, 'learning': 1, 'practiced': 2, 'mastered': 3}
        if body.status and rank.get(body.status, 0) >= rank.get(cur.get('status', ''), 0):
            cur['status'] = body.status
        cur['attempts'] = int(cur.get('attempts') or 0) + 1
        if body.best_score is not None:
            prev = cur.get('best_score')
            cur['best_score'] = (body.best_score if prev is None
                                 else max(float(prev), float(body.best_score)))
        if body.worst_keypoint:
            cur['worst_keypoint'] = body.worst_keypoint
        if body.note:
            cur['note'] = body.note
        cur['updated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        lessons[body.lesson_id] = cur
        doc['lessons'] = lessons
        doc['updated_at'] = cur['updated_at']
        _journey_store.save_journey(user_id, doc)
        return {'ok': True, 'lesson': cur}
    except Exception as e:                                        # noqa: BLE001
        print(f'[server] post_me_lessons failed: {e}', file=sys.stderr)
        return {'ok': False, 'reason': 'save_failed'}


class EntryEvent(BaseModel):
    referrer: str = ''
    utm_source: str = ''
    utm_medium: str = ''
    utm_campaign: str = ''
    path: str = ''


def _is_internal_entry(ev: 'EntryEvent') -> bool:
    """v202: True for our OWN deploy/verification traffic, which must NOT be
    logged as a real user entry event (it was polluting the /api/entry-event
    numbers — e.g. ?cb=vNNN cache-busts fired by post-deploy curl checks,
    localhost:5173 dev-preview referrers, and the verify-script UTM). Real
    users never carry any of these markers."""
    src = (ev.utm_source or '').strip().lower()
    if src in ('verify-script', 'verify', 'deploy', 'healthcheck', 'smoke'):
        return True
    ref = (ev.referrer or '').lower()
    if 'localhost' in ref or '127.0.0.1' in ref:
        return True
    path = (ev.path or '').lower()
    # deploy cache-bust marker: ?cb=v200voice, ?cb=v201verif, &cb=v2 ...
    if 'cb=v' in path:
        return True
    return False


@app.post('/api/entry-event')
async def post_entry_event(ev: EntryEvent, authorization: str = Header(default='')):
    """v179: records WHERE a fresh /dance page load came from (referrer +
    UTM). Nothing previously captured this at all — there was no way to
    tell how much of the (tiny) traffic reaching this app came from the
    studioOS homepage vs direct/search/Android. Fire-and-forget from the
    client; never raises, never blocks page load.

    v202: our own deploy/verification hits are dropped (not logged) so the
    entry-event numbers reflect only real users."""
    if _is_internal_entry(ev):
        return {'ok': True, 'skipped': 'internal'}
    user_id = ''
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if token:
        u = await _fetch_studioos_user(token)
        if u:
            user_id = str(u.get('id') or u.get('user_id') or '')
    try:
        journey_storage.log_entry_event(
            ROOT, referrer=ev.referrer, utm_source=ev.utm_source,
            utm_medium=ev.utm_medium, utm_campaign=ev.utm_campaign,
            path=ev.path, user_id=user_id)
    except Exception as e:                                        # noqa: BLE001
        print(f'[server] post_entry_event failed: {e}', file=sys.stderr)
    return {'ok': True}


# ─── skill events (v186) — the (attempt, instruction, outcome) corpus ──
class SkillEvent(BaseModel):
    session_id: str = ''
    clip_id: str = ''
    event_kind: str = ''          # 'attempt' | 'instruction' | 'attempt_score'
    attempt_index: int = 0
    score: Optional[float] = None
    mean_error: Optional[float] = None
    worst_keypoint: str = ''
    instruction: str = ''
    instruction_source: str = ''  # 'llm' | 'live_feedback' | 'canned'
    consent: bool = False
    meta: Optional[Dict[str, Any]] = None


@app.post('/api/skill-event')
async def post_skill_event(ev: SkillEvent, authorization: str = Header(default='')):
    """v186: append one coaching-beat (a learner attempt score and/or the
    instruction given) to the durable skill-event corpus. This is the
    research moat — (attempt, instruction, outcome) triples that only a
    deployed product can collect. Fire-and-forget from the client; never
    raises. Only stores when consent=true (the client passes the user's
    data-sharing consent flag)."""
    if not ev.consent:
        # Respect consent strictly — no consent, no capture. Still 200 so
        # the client fire-and-forget never sees an error.
        return {'ok': True, 'stored': False, 'reason': 'no_consent'}
    user_id = await _user_id_from_auth(authorization)
    try:
        journey_storage.log_skill_event(
            ROOT, user_id=user_id, session_id=ev.session_id,
            clip_id=ev.clip_id, event_kind=ev.event_kind,
            attempt_index=ev.attempt_index, score=ev.score,
            mean_error=ev.mean_error, worst_keypoint=ev.worst_keypoint,
            instruction=ev.instruction, instruction_source=ev.instruction_source,
            consent=ev.consent, meta=ev.meta)
    except Exception as e:                                        # noqa: BLE001
        print(f'[server] post_skill_event failed: {e}', file=sys.stderr)
    return {'ok': True, 'stored': True}


# ─── notifications (v185) ──────────────────────────────────────────────
# coach.js has been calling /api/notifications/vapid-key + /api/me/push/
# subscribe since v69/v135 and coach/notifications.py + coach/reminders.py
# were fully built -- but NONE of these routes were ever registered, and
# the notifications module was never even imported. Every subscribe
# attempt 404'd silently. This wires the whole thing up for real.
async def _user_id_from_auth(authorization: str) -> str:
    token = ''
    if authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token:
        return ''
    u = await _fetch_studioos_user(token)
    if not u:
        return ''
    return str(u.get('id') or u.get('user_id') or '')


@app.get('/api/notifications/vapid-key')
def get_vapid_key():
    """Public VAPID key the browser needs to call pushManager.subscribe().
    Empty string when Web Push isn't configured -- the browser already
    handles that gracefully (see coach.js _enablePush)."""
    return {'public_key': coach_notifications.public_vapid_key()}


@app.get('/api/notifications/diag')
def get_notifications_diag():
    """Operator-facing snapshot of which push channels are actually live."""
    return coach_notifications.diagnostics()


@app.get('/api/diag/funnel')
def get_funnel_diag(hours: int = 168,
                    x_internal_token: str = Header(default='')):
    """v202: drop-off report — of everyone who opened the coach WS, how many
    started a session, how many completed, and the duration-bucket breakdown
    of when they left (so 'bounced in <20s without starting' is a concrete
    number). Gated by the internal token (same as the build API)."""
    _require_internal(x_internal_token)
    return journey_storage.funnel_summary(ROOT, hours=hours)



class PushSubscribeBody(BaseModel):
    subscription: Dict[str, Any]
    tz: str = ''
    preferred_hour: int = 18


@app.post('/api/me/push/subscribe')
async def push_subscribe(body: PushSubscribeBody,
                         authorization: str = Header(default='')):
    user_id = await _user_id_from_auth(authorization)
    if not user_id:
        raise HTTPException(401, 'sign in required')
    try:
        _journey_store.save_push_subscription(user_id, body.subscription)
        # Persist tz/preferred_hour onto the journey so reminders.py's
        # _should_notify() can compute the user's local send window.
        doc = _journey_store.load_journey(user_id) or {}
        doc.setdefault('user_id', user_id)
        prefs = dict(doc.get('notifications') or {})
        if body.tz:
            prefs['tz'] = body.tz
        prefs['preferred_hour'] = int(body.preferred_hour)
        doc['notifications'] = prefs
        _journey_store.save_journey(user_id, doc)
    except Exception as e:                                        # noqa: BLE001
        raise HTTPException(500, f'subscribe failed: {e}')
    return {'ok': True}


class PushUnsubscribeBody(BaseModel):
    endpoint: str


@app.post('/api/me/push/unsubscribe')
async def push_unsubscribe(body: PushUnsubscribeBody,
                           authorization: str = Header(default='')):
    user_id = await _user_id_from_auth(authorization)
    if not user_id:
        raise HTTPException(401, 'sign in required')
    try:
        _journey_store.delete_push_subscription(user_id, body.endpoint)
    except Exception as e:                                        # noqa: BLE001
        raise HTTPException(500, f'unsubscribe failed: {e}')
    return {'ok': True}


@app.post('/api/me/push/test')
async def push_test(authorization: str = Header(default='')):
    """Fire one real notification at the caller right now -- lets the
    user (or us) confirm the whole channel actually delivers, instead of
    trusting config alone."""
    user_id = await _user_id_from_auth(authorization)
    if not user_id:
        raise HTTPException(401, 'sign in required')
    try:
        subs = _journey_store.list_push_subscriptions(user_id)
        sub_dicts = [r.get('subscription') for r in subs
                    if isinstance(r.get('subscription'), dict)]
        dev_rows = _journey_store.list_device_tokens(user_id)
        dev_tokens = [r.get('registration_id') for r in dev_rows
                      if r.get('registration_id')]
        res = coach_notifications.send_to_user(
            user_id, 'Dance.AI', "Just testing — this is what a reminder "
            "looks like!", '/dance', sub_dicts, f'user:{user_id}',
            device_tokens=dev_tokens)
        # Prune any FCM tokens the send reported as dead.
        for tok in res.stale_tokens:
            try:
                _journey_store.delete_device_token(user_id, tok)
            except Exception:                                    # noqa: BLE001
                pass
        return res.as_dict()
    except Exception as e:                                        # noqa: BLE001
        raise HTTPException(500, f'test send failed: {e}')


class DeviceRegisterBody(BaseModel):
    registration_id: str
    platform: str = 'android'
    locale: str = ''
    tz: str = ''


@app.post('/api/me/devices/register')
async def devices_register(body: DeviceRegisterBody,
                           authorization: str = Header(default='')):
    """Register a native FCM/APNs device token (Android app / future iOS)
    so Azure Notification Hubs can fan out native push to it. No-ops
    gracefully if the Hub isn't configured yet -- the token is still
    stored so sends can start working the moment it is."""
    user_id = await _user_id_from_auth(authorization)
    if not user_id:
        raise HTTPException(401, 'sign in required')
    try:
        _journey_store.save_device_token(
            user_id, body.registration_id, body.platform,
            body.locale, body.tz)
    except Exception as e:                                        # noqa: BLE001
        raise HTTPException(500, f'device register failed: {e}')
    return {'ok': True, 'hub_enabled': coach_notifications.hub_enabled()}


@app.get('/api/vrm/{name}')
def get_vrm(name: str):
    """Serve a VRM file by character name (registry lookup)."""
    reg = json.loads(REGISTRY.read_text(encoding='utf-8'))
    match = next((c for c in reg.get('characters', [])
                  if c.get('name') == name), None)
    if not match:
        raise HTTPException(404, f'no character {name}')
    vrm_path = ROOT / match['vrm']
    if not vrm_path.exists():
        raise HTTPException(404, f'vrm file missing: {match["vrm"]}')
    # v76: cache aggressively. VRMs are ~14 MB and immutable per character;
    # without this the browser re-downloads the whole model on every visit
    # (the #1 load-time cost). 1-year immutable cache => repeat loads are
    # instant from disk. Bust by renaming the file if a model ever changes.
    return FileResponse(
        str(vrm_path), media_type='model/gltf-binary',
        headers={'Cache-Control': 'public, max-age=31536000, immutable'})


# ─── inline auth (login / signup popup) ───────────────────────────────
# The dance coach used to bounce users to studioos.fit/login and rely on
# them navigating back — which broke (token landed on a different page /
# the WS never re-read it, so the user looped through "please sign in").
# These thin server-side proxies forward to the studio-Os auth API so the
# browser can sign in / sign up from an in-page popup with NO cross-origin
# CORS and NO full-page redirect. We return the upstream JSON + status so
# the browser can grab {access_token} and store it exactly like the main
# studio-Os SPA does.
class _AuthLoginBody(BaseModel):
    email: str
    password: str


class _AuthRegisterBody(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    phone: Optional[str] = None


def _upstream_json(r: 'httpx.Response') -> Any:
    try:
        return r.json()
    except Exception:
        return {'message': (r.text or '').strip()[:200]}


@app.post('/api/track', include_in_schema=False)
async def track_proxy(request: 'Request'):
    """Forward coach telemetry to the studio-Os backend.

    On studioos.fit/dance the external nginx proxied /api/track to
    api.studioos.fit. On the dancecoach.fit apex there is NO nginx, so the
    browser's POST /api/track hit FastAPI directly and 404'd — losing all
    picker/onboarding/funnel telemetry (dance_style_viewed, dance_ready, etc.).
    This route restores it by forwarding the raw JSON body to the studio-Os
    /api/track endpoint. Fire-and-forget shape: always returns 204 so the
    beacon never surfaces an error in the console.
    """
    try:
        raw = await request.body()
    except Exception:
        raw = b''
    try:
        async with httpx.AsyncClient(timeout=6.0) as cx:
            await cx.post(f'{STUDIOOS_API}/api/track', content=raw,
                          headers={'Content-Type': 'application/json'})
    except Exception:
        pass  # telemetry is best-effort; never block or error the client
    return JSONResponse(status_code=204, content=None)


@app.post('/api/auth/login')
async def auth_login(body: _AuthLoginBody):
    """Proxy a login to studio-Os. Returns {access_token, refresh_token,
    user} on success (200) or the upstream error status."""
    try:
        async with httpx.AsyncClient(timeout=12.0) as cx:
            r = await cx.post(
                f'{STUDIOOS_API}/api/auth/login',
                json={'email': body.email.strip().lower(),
                      'password': body.password},
            )
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={'message': f'auth upstream error: {e}'})
    return JSONResponse(status_code=r.status_code,
                        content=_upstream_json(r))


@app.post('/api/auth/register')
async def auth_register(body: _AuthRegisterBody):
    """Proxy a dancer sign-up to studio-Os. Creates a customer account and
    returns the same {access_token, ...} shape as login on success."""
    payload: Dict[str, Any] = {
        'email': body.email.strip().lower(),
        'password': body.password,
        'user_type': 'customer',
        'signup_source': 'dancer_popup',
    }
    if body.name:
        payload['name'] = body.name.strip()
    if body.phone:
        payload['phone'] = body.phone.strip()
    try:
        async with httpx.AsyncClient(timeout=12.0) as cx:
            r = await cx.post(f'{STUDIOOS_API}/api/auth/register',
                              json=payload)
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={'message': f'auth upstream error: {e}'})
    return JSONResponse(status_code=r.status_code,
                        content=_upstream_json(r))


class _AuthRefreshBody(BaseModel):
    refresh_token: str


@app.post('/api/auth/refresh')
async def auth_refresh(body: _AuthRefreshBody):
    """v185: proxy a refresh_token exchange to studio-Os. coach.js has
    been STORING refresh_token since login/register but never had
    anywhere to send it -- an expired access_token just silently
    degraded the session to anonymous with no re-login prompt. Mirrors
    the login/register proxy shape: {access_token, refresh_token, user}
    on success, or the upstream error status/body passed straight
    through so the client can fall back to a clean sign-in if this
    endpoint doesn't match studio-Os's actual contract."""
    try:
        async with httpx.AsyncClient(timeout=12.0) as cx:
            r = await cx.post(
                f'{STUDIOOS_API}/api/auth/refresh',
                json={'refresh_token': body.refresh_token},
            )
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={'message': f'auth upstream error: {e}'})
    return JSONResponse(status_code=r.status_code,
                        content=_upstream_json(r))


@app.get('/api/motion/list')
def motion_list():
    return {'motions': motion_index.list_motions()}


@app.get('/api/motion/variety')
def motion_variety(genre: str, exclude: str = '', limit: int = 6):
    """Fresh clips in a genre for FREE-DANCE auto-rotation (so the avatar
    stops looping the same move). Returns ONLY safe, retargeted, orientation-
    verified clips (never an inverted/unsafe one), excluding `exclude`,
    shuffled. Empty list is a valid answer (caller just keeps the current
    loop)."""
    import random
    g = (genre or '').strip()
    picks = []
    for m in motion_index.list_motions():
        if m['id'] == exclude:
            continue
        if g and m.get('genre') != g:
            continue
        if not m.get('retargeted'):
            continue                      # no VRM clip cached → can't play
        if m.get('safety') == 'fail':
            continue
        if not motion_index.is_verified_upright(m['id']):
            continue                      # never rotate to an upside-down clip
        picks.append({'clip_id': m['id'],
                      'music_url': m.get('music_url'),
                      'bpm': m.get('bpm_target')})
    random.shuffle(picks)
    return {'genre': g, 'clips': picks[:max(1, min(int(limit), 12))]}


@app.get('/api/motion/meta/{motion_id}.json')
def motion_meta(motion_id: str):
    """Per-clip teaching metadata (title, cues, mistakes...) — drives
    the coach's voice tips and the user-facing clip card."""
    m = motion_metadata.get_meta(motion_id)
    if not m:
        raise HTTPException(404, f'no metadata for {motion_id}')
    return m


@app.get('/api/motion/search')
def motion_search(q: str, k: int = 8, genre: Optional[str] = None):
    """Free-text semantic search over clip metadata."""
    from coach import semantic_search
    return {'query': q, 'results': semantic_search.search(q, k=k, genre=genre)}


@app.post('/api/motion/resequence')
async def motion_resequence(
    audio: UploadFile = File(...),
    bars: int = 8,
    genre: Optional[str] = None,
    query: Optional[str] = None,
):
    """Upload an audio file; receive a vrm-quat JSON whose frames are
    a beat-aligned concatenation of retargeted clips. The browser can
    play it through the existing MotionPlayer."""
    import tempfile, os as _os
    from coach import resequencer
    suffix = _os.path.splitext(audio.filename or '.wav')[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        out = resequencer.resequence_from_audio(
            path, bars=int(bars), genre=genre, query=query)
    except Exception as e:
        raise HTTPException(500, f'resequence failed: {e}')
    finally:
        try: _os.unlink(path)
        except Exception: pass
    return out


@app.post('/api/feedback/compare')
async def feedback_compare(
    clip_id: str,
    student_motion: UploadFile = File(...),
):
    """Compare a student's pre-extracted vrm-quat JSON against the
    reference clip and return a 2-3-sentence coach note plus per-bone
    error stats. The student JSON must already be in vrm-quat format
    (use scripts/export_motion_json.py to produce it from a video)."""
    import json as _json
    from coach import metadata as _md
    from coach.feedback import compare as _cmp, writer as _wr
    ref_path = MOTION_CACHE / f'{clip_id}.json'
    if not ref_path.exists():
        raise HTTPException(404, f'no retargeted clip {clip_id}')
    raw = await student_motion.read()
    try:
        student = _json.loads(raw)
    except Exception as e:
        raise HTTPException(400, f'bad student JSON: {e}')
    if student.get('format') != 'vrm-quat':
        raise HTTPException(400, 'student motion must be vrm-quat')
    reference = _json.loads(ref_path.read_text(encoding='utf-8'))
    diff = _cmp.compare(reference, student)
    if not diff.get('ok'):
        raise HTTPException(400, f"compare failed: {diff.get('reason')}")
    meta = _md.get_meta(clip_id)
    note = await _wr.write_feedback(diff, meta)
    return {'ok': True, 'clip_id': clip_id, 'note': note, 'diff': diff}


# ─── 2D feedback (preferred — student sends COCO-17 keypoints from
#     in-browser MediaPipe Pose; we compare against a precomputed
#     reference projection of the same clip).
class StudentKpPayload(BaseModel):
    clip_id:   str
    fps:       float = 30.0
    keypoints: List[List[List[float]]]   # (T, 17, 3)


@app.post('/api/feedback/compare2d')
async def feedback_compare_2d(payload: StudentKpPayload):
    """Compare a student's 2D keypoint stream (COCO-17, normalised image
    coords with visibility) against the precomputed reference.

    The browser is expected to run MediaPipe Pose on the student's
    uploaded video and POST one keypoint frame per video frame."""
    import numpy as _np
    from coach import metadata as _md
    from coach.feedback import compare_2d as _c2, writer as _wr
    ref_path = MOTION_2D / f'{payload.clip_id}.npz'
    if not ref_path.exists():
        raise HTTPException(404, f'no 2D reference for {payload.clip_id} '
                                 f'(run scripts/precompute_aist_2d_keypoints.py)')
    try:
        stu_kp = _np.asarray(payload.keypoints, dtype=_np.float32)
    except Exception as e:
        raise HTTPException(400, f'bad keypoints array: {e}')
    if stu_kp.ndim != 3 or stu_kp.shape[1] < 17 or stu_kp.shape[2] < 3:
        raise HTTPException(400,
            f'keypoints must be (T,17,3); got {stu_kp.shape}')
    ref_kp = _np.load(ref_path)['keypoints']
    diff = _c2.compare_2d(ref_kp, stu_kp)
    if not diff.get('ok'):
        raise HTTPException(400, f"compare failed: {diff.get('reason')}")
    meta = _md.get_meta(payload.clip_id)
    note = await _wr.write_feedback_2d(diff, meta)
    # Strip the dense frame_errors timeline from the response by
    # default — it's ~30 KB per minute. Caller can include=timeline
    # to opt in later if we want to draw a heatmap.
    return {'ok': True, 'clip_id': payload.clip_id, 'note': note,
            'mean_error':       diff['mean_error'],
            'aligned_pairs':    diff['aligned_pairs'],
            'worst_keypoints':  diff['worst_keypoints'],
            'worst_frames':     diff['worst_frames'],
            'per_keypoint':     diff['per_keypoint']}


MOTION_CACHE = COACH / 'motion_cache'
MOTION_CACHE_CMU = COACH / 'motion_cache_cmu'
MOTION_2D    = COACH / 'motion_2d'

# v14: per-clip baked corrections (yaw / in-place / ground offset / jump
# flag). Generated offline by `analyze_motions.py`; loaded once at import
# time so every `/api/motion/data` response can attach the clip's record
# under `body['corrections']` for the browser MotionPlayer to consume.
_CORRECTIONS_PATH = COACH / 'motion_meta' / 'corrections.json'
try:
    _CORRECTIONS: Dict[str, Dict[str, Any]] = json.loads(
        _CORRECTIONS_PATH.read_text(encoding='utf-8'))
    print(f'[motion_corrections] loaded {len(_CORRECTIONS)} clips '
          f'from {_CORRECTIONS_PATH}')
except Exception as _e:                                              # noqa: BLE001
    _CORRECTIONS = {}
    print(f'[motion_corrections] no corrections.json ({_e}); '
          'runtime heuristics will be used')
EXPORT_SCRIPT = ROOT / 'scripts' / 'export_motion_json.py'
SMPL_PKL = ROOT / 'data' / 'models' / 'smpl_raw' / 'smpl' / 'models' / 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
VRM_DEFAULT = ROOT / 'data' / 'models' / 'extra' / 'AvatarSample_K.vrm'


def _maybe_retarget_cmu_on_demand(motion_id: str, src_pkl: Path) -> Optional[Path]:
    """Build CMU vrm-quat cache lazily for clips whose cache is missing/stale.

    This keeps runtime playback on the high-quality retarget path even when
    a full background batch rebuild has not completed yet.
    """
    if not motion_id.startswith('cmu_'):
        return None
    if not EXPORT_SCRIPT.exists() or not VRM_DEFAULT.exists() or not SMPL_PKL.exists():
        return None
    out = MOTION_CACHE_CMU / f'{motion_id}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable or 'python'
    try:
        r = subprocess.run(
            [py, str(EXPORT_SCRIPT),
             '--aist', str(src_pkl),
             '--vrm', str(VRM_DEFAULT),
             '--smpl_pkl', str(SMPL_PKL),
             '--out', str(out)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode == 0 and out.exists():
            return out
        tail = (r.stderr or r.stdout or '').strip().splitlines()[-2:]
        print(f'[motion_data] on-demand retarget fail {motion_id}: {tail}')
    except Exception as e:                                           # noqa: BLE001
        print(f'[motion_data] on-demand retarget exc {motion_id}: {e}')
    return None


@app.get('/api/motion/data/{motion_id}.json')
def motion_data(motion_id: str):
    """Serve motion data for the browser MotionPlayer.

    Prefers retargeted VRM-bone-local quaternion JSON if present in
    ``coach/motion_cache/<id>.json`` (produced by
    ``scripts/export_motion_json.py``). Falls back to raw SMPL axis-angle
    if no retargeted file exists.

    Cached shape (preferred, ``format: 'vrm-quat'``):
        { format, fps, frames, bones[], rotations:{bone:[[x,y,z,w]...]},
          hips_translation:[[x,y,z]...], rest_local_rotation:{bone:[x,y,z,w]} }

    Fallback shape (``format: 'smpl-aa'``):
        { format, fps, frames, poses[T*24*3], trans[T*3] }
    """
    # 1. preferred path: pre-retargeted bone-local quaternions
    #    (looks in motion_cache/ then motion_cache_cmu/)
    cached = motion_index.get_cached_json(motion_id)
    if cached is None and motion_id.startswith('cmu_'):
        src = motion_index.get_motion(motion_id)
        if src is not None:
            cached = _maybe_retarget_cmu_on_demand(motion_id, src)
    if cached is not None and cached.exists():
        try:
            body = json.loads(cached.read_text(encoding='utf-8'))
            body['format'] = 'vrm-quat'
            body['id'] = motion_id
            body['frames'] = int(body.get('n_frames', 0))
            body['safety'] = {'passed': True, 'severity': 'ok',
                              'cached': True}
            corr = _CORRECTIONS.get(motion_id)
            if corr is not None:
                body['corrections'] = corr
            return JSONResponse(body)
        except Exception as e:
            # fall through to raw if cache is corrupt
            print(f'[motion_data] cache read fail {motion_id}: {e}')

    p = motion_index.get_motion(motion_id)
    if p is None:
        raise HTTPException(404, f'no motion {motion_id}')
    with open(p, 'rb') as f:
        d = pickle.load(f)
    poses = np.asarray(d['smpl_poses'], dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(d['smpl_trans'], dtype=np.float32).reshape(-1, 3)
    fps = int(d.get('fps', 60))
    if fps == 60:
        poses = poses[::2]
        trans = trans[::2]
        fps = 30
    scaling = float(np.asarray(d.get('smpl_scaling', [1.0])).flatten()[0])
    if scaling != 1.0:
        trans = trans / max(scaling, 1e-6)

    # Server-side hard clamp before shipping (defense layer 2)
    poses = clamp_pose(poses)

    report = validate_motion(poses, trans, fps=fps, path=p.name)

    body = {
        'id':     motion_id,
        'format': 'smpl-aa',
        'fps':    fps,
        'frames': int(poses.shape[0]),
        'poses':  poses.reshape(-1).tolist(),
        'trans':  trans.reshape(-1).tolist(),
        'safety': {
            'passed':   report.passed,
            'severity': report.severity,
            'max_joint_speed_rad_s': report.max_joint_speed_rad_s,
            'max_pelvis_speed_m_s': report.max_pelvis_speed_m_s,
            'n_violations': len(report.violations),
        },
    }
    # v25: attach corrections on the SMPL fallback path too. The
    # vrm-quat path was already attaching them; the SMPL path silently
    # dropped them, so the 13 RUNAWAY_XZ clips (which only ship as
    # raw SMPL because no retargeted JSON exists) never received the
    # in_place override and continued to teleport. Same dict shape
    # the browser MotionPlayer reads on both code paths.
    corr = _CORRECTIONS.get(motion_id)
    if corr is not None:
        body['corrections'] = corr
    return JSONResponse(body)


@app.get('/api/speech/token')
async def speech_token():
    """Mint an ephemeral Azure Speech token for the browser SDK so we
    never expose the raw subscription key to the client."""
    if not AZURE_KEY:
        raise HTTPException(503, 'AZURE_SPEECH_KEY not configured')
    url = f'https://{AZURE_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken'
    async with httpx.AsyncClient(timeout=10) as cx:
        r = await cx.post(url, headers={'Ocp-Apim-Subscription-Key': AZURE_KEY})
    if r.status_code != 200:
        raise HTTPException(502, f'azure token mint failed: {r.text[:200]}')
    return {'token': r.text, 'region': AZURE_REGION}


# ─── conversational loop ───────────────────────────────────────────────
@app.websocket('/ws/agent')
async def ws_agent(ws: WebSocket):
    """Browser sends {type:'user_text', text:'...'}; we run the agent
    loop and stream back {type:'assistant_text'|'tool_call'|'done'}.

    Voice in: the browser runs Azure STT, sends recognised text here.
    Voice out: the browser runs Azure TTS on the assistant text.

    Auth: the WebSocket URL may carry ``?token=<JWT>``. When present
    we call back to studio-Os to identify the user. Missing / invalid
    tokens are NOT a 401 — the session just runs in anonymous mode
    (matching the static viewer's behaviour).
    """
    from coach.choreographer.agent import run_turn
    from coach.choreographer.tools import CoachState
    from coach.session_engine import SessionEngine

    await ws.accept()
    history: List[Dict[str, Any]] = []
    state = CoachState()    # PER-WS state — isolates Alice from Bob.

    # Optional studio-Os auth lookup.
    token = ws.query_params.get('token')
    if token:
        await _identify_user(state, token)

    # v178: reconnect real session/streak tracking (see _record_session_complete).
    # Lazily create a journey doc for a brand-new signed-in user so the
    # account menu has something other than dashes from the very first visit.
    _connect_ts = time.time()
    _session_recorded = False

    # v202: FUNNEL instrumentation — see storage.log_funnel_event. Every WS
    # gets a stable id so ws_connected / session_started / session_completed /
    # ws_closed rows can be stitched into one visit and the drop-off point
    # (bounced in 5s? started then quit? completed?) becomes queryable.
    _funnel_sid = uuid.uuid4().hex
    _funnel = {'started': False, 'engaged': False, 'msgs': 0, 'closed': False}
    try:
        journey_storage.log_funnel_event(
            ROOT, event='ws_connected', session_id=_funnel_sid,
            user_id=state.user_id or '', anon=not bool(state.user_id))
    except Exception:                                             # noqa: BLE001
        pass

    if state.user_id:
        try:
            _doc = _journey_store.load_journey(state.user_id)
            if not _doc:
                _journey_store.save_journey(state.user_id, {
                    'user_id': state.user_id,
                    'version': 1,
                    'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'progress': _default_progress(),
                })
        except Exception as e:                                    # noqa: BLE001
            print(f'[server] journey lazy-create failed: {e}', file=sys.stderr)

    def _finish_session():
        """Record this connection as one completed session (best-effort,
        idempotent per-WS). Skips tiny/instant connections (<20s) so
        page-reload noise doesn't inflate the streak/session count."""
        nonlocal _session_recorded
        if _session_recorded or not state.user_id:
            return
        _session_recorded = True
        elapsed_min = (time.time() - _connect_ts) / 60.0
        if elapsed_min < (20.0 / 60.0):
            return
        _record_session_complete(state.user_id, elapsed_min,
                                 template_id=getattr(state, 'last_template_id', ''))
        try:
            journey_storage.log_funnel_event(
                ROOT, event='session_completed', session_id=_funnel_sid,
                user_id=state.user_id, anon=False,
                template_id=getattr(state, 'last_template_id', '') or '',
                duration_sec=round(elapsed_min * 60.0, 1))
        except Exception:                                         # noqa: BLE001
            pass

    def _log_ws_closed():
        """Fire ONE ws_closed funnel row carrying how long the visit lasted
        and whether the user ever engaged / started a session. This is the
        row that reveals 'they bounce in N seconds without starting'."""
        if _funnel['closed']:
            return
        _funnel['closed'] = True
        try:
            journey_storage.log_funnel_event(
                ROOT, event='ws_closed', session_id=_funnel_sid,
                user_id=state.user_id or '', anon=not bool(state.user_id),
                template_id=getattr(state, 'last_template_id', '') or '',
                duration_sec=round(time.time() - _connect_ts, 1),
                meta={'started': _funnel['started'],
                      'engaged': _funnel['engaged'],
                      'msgs': _funnel['msgs']})
        except Exception:                                         # noqa: BLE001
            pass


    # ── BARGE-IN INFRASTRUCTURE ───────────────────────────────────────
    # The browser used to be a strict request/response client: while
    # the LLM was generating, the WS read loop was blocked on
    # `async for event in run_turn(...)`. That meant the user could
    # NOT interrupt the coach mid-monologue — even if they yelled,
    # the AI kept talking until done.
    #
    # New model: each turn runs as its OWN asyncio.Task. The WS
    # message loop runs concurrently and can cancel that task at
    # any time. The browser sends {type:'user_interrupt'} the moment
    # mic detects speech OR the user submits a new text — this turn
    # is cancelled, any remaining events are dropped, TTS is told
    # to stop (browser side), and the next user_text starts fresh.
    import asyncio

    turn_task: Optional[asyncio.Task] = None

    async def _run_turn_and_stream(turn_history, source='typed'):
        """Run the LLM turn, forwarding every event to the WS. Catches
        CancelledError so an interrupted turn dies cleanly."""
        try:
            async for event in run_turn(turn_history, state=state, source=source):
                await ws.send_json(event)
            await ws.send_json({'type': 'done'})
        except asyncio.CancelledError:
            # Browser already cancelled TTS locally; just notify so it
            # can clear any "thinking" indicator.
            try:
                await ws.send_json({'type': 'interrupted'})
            except Exception:
                pass
            raise

    async def _cancel_current_turn():
        nonlocal turn_task
        if turn_task and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except (asyncio.CancelledError, Exception):
                pass
        turn_task = None

    try:
        await ws.send_json({'type': 'hello', 'model': GROQ_MODEL,
                            'has_groq': bool(GROQ_KEY)})

        # ── v34: COACH-LED IDLE NUDGES ────────────────────────────
        # If the user stays silent after the greeting, push a series
        # of friendly nudges so the conversation never dead-ends. The
        # browser already speaks an in-character greeting on connect;
        # these nudges layer on top. Any user_text cancels the chain.
        _NUDGES = [
            (14.0,
             "Still there? Tell me how your day is going — or just "
             "tap one of the chips and I'll start us moving."),
            (28.0,
             "No pressure — wanna try a slow warm-up? Just say 'warm "
             "me up' and I'll guide you in."),
            (48.0,
             "I'll be right here whenever you're ready. Hit "
             "'Dance for me' and I'll do the rest."),
        ]
        nudge_task: Optional[asyncio.Task] = None

        async def _idle_nudges():
            try:
                for delay, line in _NUDGES:
                    await asyncio.sleep(delay)
                    await ws.send_json({
                        'type': 'assistant_text',
                        'text': line,
                        'source': 'idle_nudge',
                    })
            except asyncio.CancelledError:
                return
            except Exception:
                return

        nudge_task = asyncio.create_task(_idle_nudges())

        async def _cancel_nudges():
            nonlocal nudge_task
            if nudge_task and not nudge_task.done():
                nudge_task.cancel()
                try:
                    await nudge_task
                except (asyncio.CancelledError, Exception):
                    pass
            nudge_task = None

        # v192: guided-session engine (restored). Handles session.start/
        # skip/pause/resume/end/clip_done and drives phases + clips +
        # narration. Without this, the front-door "pick a style -> pick a
        # length" buttons sent session.start into the void (the engine had
        # been stripped from this file) and nothing ever started.
        _sess_engine = SessionEngine(ws, state, cancel_nudges=_cancel_nudges)

        while True:
            msg = await ws.receive_json()
            mt = msg.get('type')
            # Route every session.* control message to the engine.
            if isinstance(mt, str) and mt.startswith('session.'):
                # v219: KILL idle nudges the instant a session control arrives.
                # Previously the "No pressure — say 'warm me up'" nudge (28s
                # timer) could fire mid-session because cancellation relied on
                # the engine and raced the timer. Cancelling here guarantees the
                # coach never nudges "get moving" while a session is running.
                await _cancel_nudges()
                if mt == 'session.start':
                    # Remember what the user chose so the completed-session
                    # record (and their profile history) can show the style,
                    # e.g. "Hip-Hop · 10 min" instead of just "10 min".
                    try:
                        state.last_template_id = msg.get('template_id') or ''
                    except Exception:
                        pass
                    # v202: funnel — the user actually STARTED a guided
                    # session (tapped a length / front-door). This is the
                    # key activation event the funnel was blind to.
                    _funnel['started'] = True
                    _funnel['engaged'] = True
                    try:
                        journey_storage.log_funnel_event(
                            ROOT, event='session_started',
                            session_id=_funnel_sid,
                            user_id=state.user_id or '',
                            anon=not bool(state.user_id),
                            template_id=state.last_template_id or '')
                    except Exception:                             # noqa: BLE001
                        pass
                if mt == 'session.end':
                    # Also record the session for streak/minutes (best-effort).
                    _finish_session()
                await _sess_engine.handle(msg)
                continue
            if mt == 'set_character':
                state.character_name         = msg.get('name')
                state.character_display_name = msg.get('display_name')
                state.character_style        = msg.get('style')
                continue
            if mt == 'user_interrupt':
                # User started speaking / typing — kill any in-flight
                # LLM turn so the coach shuts up and listens.
                await _cancel_current_turn()
                await _cancel_nudges()
                continue
            if mt == 'ui_event':
                # v197: the browser tells us the student interacted with a
                # surface (e.g. opened the Lessons panel). We flip the
                # learning_intent flag so the system prompt turns the coach
                # into a proactive teacher/navigator on the NEXT turn. No
                # LLM call here — purely context.
                ev = (msg.get('event') or '').strip()
                if ev == 'opened_learn':
                    state.learning_intent = True
                elif ev == 'closed_learn':
                    state.learning_intent = False
                continue
            if mt != 'user_text':
                continue
            user_text = (msg.get('text') or '').strip()
            if not user_text:
                continue
            # v34i: 'typed' (textbox/chip) vs 'voice' (STT). Only voice
            # may be filtered by the STT noise gate; typed text is
            # always intentional and must reach the LLM.
            source = (msg.get('source') or 'typed').strip().lower()
            if source not in ('typed', 'voice'):
                source = 'typed'
            # v202: funnel — any real user message counts as engagement.
            _funnel['engaged'] = True
            _funnel['msgs'] += 1
            # New user input ALWAYS supersedes whatever the coach was
            # mid-generating. Cancel first, then start fresh.
            await _cancel_current_turn()
            await _cancel_nudges()
            history.append({'role': 'user', 'content': user_text})
            turn_task = asyncio.create_task(
                _run_turn_and_stream(list(history), source=source))
    except WebSocketDisconnect:
        await _cancel_current_turn()
        await _cancel_nudges()
        try:
            await _sess_engine.stop()
        except Exception:
            pass
        _finish_session()
        _log_ws_closed()
        return
    except Exception as e:                                       # noqa: BLE001
        await _cancel_current_turn()
        await _cancel_nudges()
        try:
            await _sess_engine.stop()
        except Exception:
            pass
        _finish_session()
        _log_ws_closed()
        try:
            await ws.send_json({'type': 'error', 'message': repr(e)})
        except Exception:
            pass


# ─── Layer A1: GEMINI LIVE speech-to-speech ───────────────────────────
# Native audio dialog: the browser streams mic PCM here, we relay to
# Gemini Live, and stream the coach's spoken audio back. Tool calls map
# onto the SAME choreographer tools the text agent uses so voice can
# drive the avatar. Feature-flagged: if GEMINI_API_KEY isn't set the
# browser stays on the Azure STT→Groq→Azure TTS path.
@app.websocket('/ws/voice')
async def ws_voice(ws: WebSocket):
    from starlette.websockets import WebSocketState
    from coach.choreographer.tools import CoachState
    from coach import gemini_live

    await ws.accept()
    if not gemini_live.gemini_enabled():
        try:
            await ws.send_json({'type': 'error',
                                'message': 'gemini_live_unavailable'})
        finally:
            await ws.close()
        return

    state = CoachState()
    token = ws.query_params.get('token')
    if token:
        try:
            await _identify_user(state, token)
        except Exception:                                        # noqa: BLE001
            pass
    # Character + language may ride as query params (set by the browser
    # before opening the socket) so the very first spoken turn is in
    # character + the right language.
    state.character_name         = ws.query_params.get('character') or None
    state.character_display_name = ws.query_params.get('display_name') or None
    state.character_style        = ws.query_params.get('style') or None
    lang = (ws.query_params.get('language') or '').strip().lower()
    if lang in ('english', 'hinglish', 'hindi'):
        state.coach_language = lang

    async def _send_json(obj):
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json(obj)

    async def _send_bytes(b):
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_bytes(b)

    async def _recv():
        try:
            m = await ws.receive()
        except Exception:                                        # noqa: BLE001
            return ('disconnect', None)
        if m.get('type') == 'websocket.disconnect':
            return ('disconnect', None)
        if m.get('bytes') is not None:
            return ('bytes', m['bytes'])
        if m.get('text') is not None:
            return ('text', m['text'])
        return ('text', '')

    try:
        await gemini_live.run_voice_session(
            ws, state, send_json=_send_json, send_bytes=_send_bytes,
            recv=_recv)
    except WebSocketDisconnect:
        pass
    except Exception as e:                                       # noqa: BLE001
        try:
            await _send_json({'type': 'error', 'message': repr(e)})
        except Exception:                                        # noqa: BLE001
            pass
    finally:
        try:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:                                        # noqa: BLE001
            pass


# ─── v34: LIVE FEEDBACK over WebSocket ────────────────────────────────
# Browser MediaPipe Pose → /ws/feedback → DTW-lite per-frame score.
# Lightweight on purpose: we don't run global DTW (the playback is
# live and looping); we use the avatar's current playback frame as
# the alignment anchor and just diff the student's posture against
# the reference frame at that index. A short sliding window smooths
# the score so single-frame jitter doesn't make the gauge twitch.
@app.websocket('/ws/feedback')
async def ws_feedback(ws: WebSocket):
    """Live pose comparison.

    Client → server messages:
      {type:'start',    clip_id:'gHO_...'}                   open session
      {type:'frame',    ref_frame:42, kp:[[x,y,v]...17]}     stream frame
      {type:'stop'}                                          end session

    Server → client messages:
      {type:'ready',       frames, fps}     reference loaded; start streaming
      {type:'unavailable', clip_id, reason} no precomputed reference
      {type:'score',       score, worst, mean_err}           every ~500ms
      {type:'done',        avg_score}                        after 'stop'
    """
    import numpy as _np
    from coach.feedback.compare_2d import (normalise as _norm,
                                           WATCH_KPT_IDX as _WK,
                                           NAME_OF as _NM)
    await ws.accept()
    ref_norm: Optional[_np.ndarray] = None            # (Tref, 17, 2)
    Tref = 0
    score_window: List[float] = []
    kpt_acc: Dict[int, List[float]] = {}              # per-kpt windowed err
    all_scores: List[float] = []
    last_emit = 0.0
    EMIT_INTERVAL = 0.5
    WINDOW_LEN = 30
    BAD = 0.8

    async def _emit_score():
        nonlocal last_emit
        if not score_window:
            return
        mean_err = float(_np.mean(score_window[-WINDOW_LEN:]))
        score = max(0.0, min(100.0, (1.0 - mean_err / BAD) * 100.0))
        # Worst keypoint over the same window.
        worst_name = None
        worst_val = -1.0
        for k, errs in kpt_acc.items():
            if not errs:
                continue
            mv = float(_np.mean(errs[-WINDOW_LEN:]))
            if mv > worst_val:
                worst_val = mv
                worst_name = _NM.get(k, str(k))
        all_scores.append(score)
        await ws.send_json({'type': 'score',
                            'score': round(score, 1),
                            'mean_err': round(mean_err, 3),
                            'worst': worst_name})
        last_emit = asyncio.get_event_loop().time()

    try:
        while True:
            msg = await ws.receive_json()
            mt = msg.get('type')
            if mt == 'start':
                clip_id = msg.get('clip_id') or ''
                ref_path = MOTION_2D / f'{clip_id}.npz'
                if not ref_path.exists():
                    await ws.send_json({'type': 'unavailable',
                                        'clip_id': clip_id,
                                        'reason': 'no 2D reference precomputed'})
                    # Keep socket open in case browser sends a different
                    # clip_id; the browser typically just closes.
                    continue
                try:
                    ref_kp = _np.load(ref_path)['keypoints']
                    ref_norm = _norm(ref_kp)
                    Tref = ref_norm.shape[0]
                    score_window.clear()
                    kpt_acc.clear()
                    all_scores.clear()
                    await ws.send_json({'type': 'ready',
                                        'frames': int(Tref),
                                        'fps': 30})
                except Exception as e:                          # noqa: BLE001
                    await ws.send_json({'type': 'unavailable',
                                        'clip_id': clip_id,
                                        'reason': f'load error: {e}'})
                continue
            if mt == 'frame':
                if ref_norm is None or Tref == 0:
                    continue
                kp_raw = msg.get('kp') or []
                if len(kp_raw) < 17:
                    continue
                ref_frame = int(msg.get('ref_frame') or 0) % Tref
                try:
                    stu = _np.asarray(kp_raw, dtype=_np.float32).reshape(1, -1, 3)
                except Exception:
                    continue
                if stu.shape[1] < 17:
                    continue
                try:
                    stu_n = _norm(stu)[0]    # (17, 2)
                except Exception:
                    continue
                r = ref_norm[ref_frame]      # (17, 2)
                # Per-watched-keypoint Euclidean dist (torso units).
                errs: List[float] = []
                for k in _WK:
                    v = stu[0, k, 2]
                    if v < 0.3:
                        continue
                    d = float(_np.linalg.norm(r[k] - stu_n[k]))
                    errs.append(d)
                    kpt_acc.setdefault(k, []).append(d)
                    if len(kpt_acc[k]) > 4 * WINDOW_LEN:
                        kpt_acc[k] = kpt_acc[k][-WINDOW_LEN:]
                if not errs:
                    continue
                frame_err = float(_np.mean(errs))
                score_window.append(frame_err)
                if len(score_window) > 4 * WINDOW_LEN:
                    score_window = score_window[-WINDOW_LEN:]
                now = asyncio.get_event_loop().time()
                if now - last_emit >= EMIT_INTERVAL:
                    await _emit_score()
                continue
            if mt == 'stop':
                avg = float(_np.mean(all_scores)) if all_scores else 0.0
                try:
                    await ws.send_json({'type': 'done',
                                        'avg_score': round(avg, 1),
                                        'frames': len(all_scores)})
                except Exception:
                    pass
                return
    except WebSocketDisconnect:
        return
    except Exception as e:                                      # noqa: BLE001
        try:
            await ws.send_json({'type': 'error', 'message': repr(e)})
        except Exception:
            pass


# ─── studio-Os identity lookup ────────────────────────────────────────
# Optional external identity backend. We delegate JWT verification to an
# external service by calling its /api/me endpoint with the Bearer token.
# Unset by default → the coach runs fully anonymous / standalone. Set the
# STUDIOOS_API (or your own) env var to enable authenticated features.
STUDIOOS_API = os.getenv('STUDIOOS_API', '').rstrip('/')


async def _fetch_studioos_user(token: str) -> Optional[Dict[str, Any]]:
    """Call studio-Os /api/me with the given bearer token. Returns the
    user payload dict, or None on any failure / non-200 (never raises)."""
    if not token:
        return None
    try:
        async with httpx.AsyncClient(timeout=4.0) as cx:
            r = await cx.get(
                f'{STUDIOOS_API}/api/me',
                headers={'Authorization': f'Bearer {token}'},
            )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        # studio-Os reachability issues should never block the session.
        return None


async def _identify_user(state, token: str) -> None:
    u = await _fetch_studioos_user(token)
    if not u:
        return
    state.user_id        = str(u.get('id') or u.get('user_id') or '')
    state.user_name      = u.get('name') or u.get('display_name')
    state.email_verified = bool(u.get('email_verified'))
    state.tier           = 'verified' if state.email_verified \
                           else ('unverified' if state.user_id else 'anon')


# ─── user journey / progress (streak, minutes, sessions) ──────────────
# v178: this used to be wired via a one-off patch to server.py that got
# lost in a later rewrite — coach/storage.py's JourneyStore has been
# sitting fully built but NEVER CALLED from the live app. No journey has
# been created/updated since. Reconnecting it here: minimal, best-effort,
# never raises out to the WS/HTTP caller.
_journey_store = journey_storage.get_store(ROOT)


def _default_progress() -> Dict[str, Any]:
    return {
        'sessions_completed': 0,
        'total_minutes': 0,
        'current_streak_days': 0,
        'last_session_date': '',
        'last_session_ts': '',
        'recent_sessions': [],
    }


# Friendly style names keyed by the genre code embedded in a template id
# (quick10_gHO, stretch_warmup_10, etc.) so the profile can show
# "House · 10 min" instead of a raw template id.
_STYLE_LABELS = {
    'gLH': 'Hip-Hop', 'gMH': 'Hip-Hop', 'gHO': 'House', 'gLO': 'Locking',
    'gWA': 'Waacking', 'gBR': 'Breaking', 'gPO': 'Popping', 'gKR': 'Krump',
    'gJS': 'Jazz',
}


def _style_from_template(template_id: str) -> str:
    """Best-effort friendly style label from a template id. Returns '' if
    unknown so the caller can fall back gracefully."""
    tid = str(template_id or '')
    if not tid:
        return ''
    if tid.startswith('stretch_warmup') or 'warmup' in tid:
        return 'Warm-up'
    for code, label in _STYLE_LABELS.items():
        if code in tid:
            return label
    return ''


def _record_session_complete(user_id: str, minutes: float,
                             template_id: str = '') -> None:
    """Best-effort: bump sessions_completed/total_minutes/streak for
    ``user_id`` and persist. Never raises — a storage hiccup should
    never take down a WS connection."""
    if not user_id or minutes <= 0:
        return
    try:
        doc = _journey_store.load_journey(user_id) or {}
        doc.setdefault('user_id', user_id)
        doc.setdefault('version', 1)
        progress = doc.get('progress')
        if not isinstance(progress, dict):
            progress = _default_progress()
        today_iso = datetime.now(timezone.utc).date().isoformat()
        last_date = str(progress.get('last_session_date') or '')
        streak = int(progress.get('current_streak_days') or 0)
        if last_date == today_iso:
            pass  # already practiced today — streak unchanged
        elif last_date and (datetime.now(timezone.utc).date()
                            - datetime.fromisoformat(last_date).date()).days == 1:
            streak += 1
        else:
            streak = 1
        progress['current_streak_days'] = streak
        progress['sessions_completed'] = int(progress.get('sessions_completed') or 0) + 1
        progress['total_minutes'] = round(
            float(progress.get('total_minutes') or 0) + minutes, 1)
        progress['last_session_date'] = today_iso
        progress['last_session_ts'] = datetime.now(timezone.utc).isoformat(
            timespec='seconds')
        recent = progress.get('recent_sessions')
        if not isinstance(recent, list):
            recent = []
        entry = {'date': today_iso, 'minutes': round(minutes, 1)}
        style = _style_from_template(template_id)
        if style:
            entry['style'] = style
        if template_id:
            entry['template_id'] = template_id
        entry['ts'] = progress['last_session_ts']
        recent.append(entry)
        progress['recent_sessions'] = recent[-20:]
        doc['progress'] = progress
        doc['updated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        _journey_store.save_journey(user_id, doc)
    except Exception as e:                                       # noqa: BLE001
        print(f'[server] _record_session_complete failed: {e}', file=sys.stderr)
