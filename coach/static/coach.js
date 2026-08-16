// coach.js — three.js scene, VRM avatar, voice loop, agent WS client.

import * as THREE from 'three';
import { GLTFLoader }    from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader }   from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// Sibling modules — relative paths so the bundle works behind any
// reverse-proxy prefix (e.g. studioos.fit/dance/static/...).
import { MotionPlayer } from './motion_player';
import { AzureVoice }   from './azure_voice';
import { AvatarLife }   from './avatar_life';
import { LiveVoice }  from './live_voice';

// APP_BASE = URL prefix the coach is mounted under (e.g. '/dance').
// Computed in coach.html before this module loads. Used for fetch() /
// WebSocket calls that need absolute paths.
const APP_BASE = (typeof window !== 'undefined' && window.APP_BASE) || '';

// v215: stamp module-load time so we can measure boot→interactive load latency
// for the cold-start bounce analysis. Prefer navigationStart if available.
try {
  if (typeof window !== 'undefined' && !window.__coachBootT0) {
    window.__coachBootT0 = Date.now();
  }
} catch (e) {}

// ── Auth token bridge ────────────────────────────────────────────────
// The Studio OS CUSTOMER app stores its JWT under 'customer_token', but this
// coach historically reads 'token'. When the coach is opened from the app
// (e.g. /app/coach?job=...) the user is already signed in as a customer, so
// mirror customer_token -> token (and keep it fresh) so the coach doesn't
// wrongly show a "sign in" wall. Same origin => localStorage is shared.
(function bridgeAuthToken() {
  try {
    const sync = () => {
      try {
        // Access token: customer_token -> token.
        const cust = localStorage.getItem('customer_token');
        const cur = localStorage.getItem('token');
        if (cust && cust !== cur) localStorage.setItem('token', cust);
        // Refresh token: customer_refresh_token -> refresh_token. WITHOUT this,
        // the coach's 1-hour access token expires and the silent refresh in
        // _tryRefreshToken() (which reads 'refresh_token') has nothing to use,
        // so it wrongly shows "session expired — sign in again". The customer
        // app persists the 30-day refresh under 'customer_refresh_token'.
        const crt = localStorage.getItem('customer_refresh_token');
        const rt = localStorage.getItem('refresh_token');
        if (crt && crt !== rt) localStorage.setItem('refresh_token', crt);
      } catch (e) {}
    };
    sync();
    window.addEventListener('storage', (e) => {
      if (!e || e.key === 'customer_token' || e.key === 'customer_refresh_token' || e.key === null) sync();
    });
  } catch (e) {}
})();

// ── Production log guardrail ─────────────────────────────────────────
// The coach streams a lot of internal diagnostics (voice/engagement/agent
// state). Those must never leak to end users in the browser console. In
// production we silence console.log/info/debug/warn; genuine errors still
// surface. Opt back into verbose logging on localhost or with ?debug=1
// (or localStorage 'coach.debug' = '1') while developing.
(function guardConsole() {
  try {
    const h = (typeof location !== 'undefined' && location.hostname) || '';
    const isLocal = h === 'localhost' || h === '127.0.0.1' || h === '' || h.endsWith('.local');
    let dbg = false;
    try {
      dbg = /[?&]debug=1\b/.test(location.search) ||
            localStorage.getItem('coach.debug') === '1';
    } catch (e) {}
    if (isLocal || dbg) return;   // keep full console for devs
    const noop = function () {};
    ['log', 'info', 'debug', 'warn', 'trace', 'table', 'group', 'groupEnd', 'groupCollapsed']
      .forEach((m) => { try { console[m] = noop; } catch (e) {} });
  } catch (e) {}
})();

const $ = (id) => document.getElementById(id);
const log = $('log');

function _fmtTs(d) {
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function _latencyTag(cls) {
  // For coach messages, surface the round-trip from the last user
  // send so the user can see exactly how long the LLM took.
  if (cls !== 'coach') return '';
  const t0 = window.__lastSentAt;
  if (!t0) return '';
  const ms = Date.now() - t0;
  // Single round per user send — clear so the next coach reply
  // doesn't show a stale latency.
  window.__lastSentAt = 0;
  if (ms < 0 || ms > 600000) return '';
  return ` · ${(ms / 1000).toFixed(1)}s`;
}
// v70: strip raw inline tool-call debris (e.g.
// `<function=set_mood{"mood":"excited"}</function>` or a truncated
// `<function=...` open) that the smaller writer model occasionally
// streams into the narration, so it never reaches a chat bubble.
function _stripFnTags(text) {
  if (!text) return text;
  return String(text)
    .replace(/<function\s*=[\s\S]*?<\/function\s*>?/gi, '')
    .replace(/<function\s*=[\s\S]*$/i, '')
    .trim();
}
function addMsg(text, cls) {
  // Remove the empty-state hint on first real message.
  const empty = log.querySelector('.empty');
  if (empty) empty.remove();
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  // Timestamp + (for coach) round-trip latency. Lets the user spot
  // when the LLM is being slow or silent so they don't think the
  // page is frozen.
  const ts = document.createElement('span');
  ts.className = 'ts';
  ts.textContent = _fmtTs(new Date()) + _latencyTag(cls);
  el.appendChild(ts);
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = text;
  el.appendChild(body);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  // Mirror coach / user messages as a floating bubble overlay on the
  // stage so the user never has to open the drawer just to see what
  // the coach said. The bubble auto-fades; the drawer keeps history.
  if (cls === 'coach' || cls === 'user') {
    if (text && text.trim()) showBubble(text, cls);
    if (cls === 'coach' && !isDrawerOpen()) bumpChatBadge();
  }
  return el;
}

// ── floating message bubble (IG-style overlay above the dock) ───────
const _bubbleWrap = $('bubble-wrap');
let _bubbleTimer = null;
function showBubble(text, cls) {
  if (!_bubbleWrap) return;
  // v108b: never render an empty bubble — that left a blank dot pill
  // hovering above the dock (worst on mobile) when a streaming coach
  // line was created before any text arrived.
  if (!text || !text.trim()) return;
  const el = document.createElement('div');
  el.className = 'bubble ' + cls;
  el.textContent = text;
  _bubbleWrap.appendChild(el);
  // Keep at most 2 bubbles visible (recent user + recent coach).
  while (_bubbleWrap.children.length > 2) {
    _bubbleWrap.firstElementChild.remove();
  }
  // Tapping a bubble opens the full chat drawer.
  el.addEventListener('click', openDrawer);
  // Auto-fade after 9s of inactivity.
  if (_bubbleTimer) clearTimeout(_bubbleTimer);
  _bubbleTimer = setTimeout(() => {
    [..._bubbleWrap.children].forEach(b => b.classList.add('fading'));
    setTimeout(() => {
      [..._bubbleWrap.children].forEach(b => { if (b.classList.contains('fading')) b.remove(); });
    }, 900);
  }, 9000);
}
function bumpChatBadge() {
  const b = $('chat-badge'); if (!b) return;
  const n = (parseInt(b.textContent, 10) || 0) + 1;
  b.textContent = String(n);
  b.classList.add('show');
}
function clearChatBadge() {
  const b = $('chat-badge'); if (!b) return;
  b.textContent = '0'; b.classList.remove('show');
}

function setStatus(s) {
  const el = $('status');
  el.textContent = s;
  el.classList.toggle('live',
    s === 'connected' || s === 'ready' || s === 'live' || s === 'coach connected');
}

// v214: lightweight picker/onboarding telemetry → /api/track (nginx →
// api.studioos.fit). Whitelisted events only. Best-effort, silent on fail.
function _coachTrack(event, props) {
  try {
    // v224 CRITICAL FIX: mint + persist a stable anonymous client id here if
    // one doesn't exist yet. Previously we only READ 'dance.cid' — but it was
    // only ever SET by session_summary.js at the END of a session. So a fresh
    // anonymous visitor fired every early funnel event (dance_ready,
    // dance_session_started, dance_voice_started…) with cid='' → the backend
    // got visitor_id=None → and SILENTLY DROPPED them as suspected bots. That
    // made dancecoach.fit's anonymous traffic (the majority) invisible in our
    // funnel while Clarity still recorded the sessions. Minting the id here
    // means every coach event now carries a visitor id and gets persisted.
    let cid = '';
    try {
      cid = localStorage.getItem('dance.cid') || '';
      if (!cid) {
        cid = 'c_' + Math.random().toString(36).slice(2, 10) +
              Date.now().toString(36).slice(-4);
        localStorage.setItem('dance.cid', cid);
      }
    } catch (e) {}
    const body = JSON.stringify({ event, props: props || {}, cid,
                                  path: location.pathname, ts: Date.now() });
    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/track', new Blob([body], { type: 'application/json' }));
    } else {
      fetch('/api/track', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body, keepalive: true, credentials: 'omit' }).catch(() => {});
    }
  } catch (e) {}
  // v215: also tag Microsoft Clarity (already loaded on /dance) so these funnel
  // events show up on the session replay timeline + are filterable in Clarity.
  try {
    if (window.clarity) {
      window.clarity('event', event);
      if (props && typeof props === 'object') {
        for (const k in props) {
          if (props[k] == null) continue;
          window.clarity('set', k, String(props[k]).slice(0, 64));
        }
      }
    }
  } catch (e) {}
  // v216: forward to Amplitude when enabled (server injects the SDK only if
  // DANCE_AMPLITUDE_API_KEY is set). Gives rich funnels/retention + longer
  // session-replay retention than Clarity's 3-day window.
  try {
    if (window.amplitude && typeof window.amplitude.track === 'function') {
      window.amplitude.track(event, props || {});
    }
  } catch (e) {}
}
try { window.__coachTrack = _coachTrack; } catch (e) {}

// v214: FIRST-VISIT "HOW IT WORKS" onboarding. Data showed ~44 visits but only
// ~2 motion plays — people landed on the style picker and left without tapping,
// a classic "I don't know what this does" bounce. This one-time overlay explains
// the loop in 3 lines and hands them a big "Start with Hip-Hop" CTA that taps the
// chip for them. Shown once (localStorage), skippable, never blocks returning users.
const _ONBOARD_KEY = 'coach.onboarded.v1';
function _maybeShowOnboarding() {
  try { if (localStorage.getItem(_ONBOARD_KEY)) return; } catch (e) {}
  if (document.getElementById('coach-onboard')) return;
  const ov = document.createElement('div');
  ov.id = 'coach-onboard';
  ov.innerHTML =
    '<div class="cob-card" role="dialog" aria-label="How it works">' +
      '<button class="cob-x" aria-label="Close">\u00d7</button>' +
      '<div class="cob-badge">\u2728 Free \u00b7 No signup to try</div>' +
      '<h2 class="cob-title">Learn any dance \u2014 in 3D</h2>' +
      '<div class="cob-steps">' +
        '<div class="cob-step"><span class="cob-n">1</span><div><b>Pick a style</b><br><small>Hip-Hop or House to start.</small></div></div>' +
        '<div class="cob-step"><span class="cob-n">2</span><div><b>Watch &amp; copy the coach</b><br><small>It breaks every move down slowly, from any angle.</small></div></div>' +
        '<div class="cob-step"><span class="cob-n">3</span><div><b>Talk to it</b><br><small>Tap \ud83c\udfa7 and ask \u201cteach me slower\u201d \u2014 it answers like a real coach.</small></div></div>' +
      '</div>' +
      '<button class="cob-cta" id="cob-start">Start with Hip-Hop \u2192</button>' +
      '<button class="cob-skip" id="cob-skip">I\u2019ll explore myself</button>' +
    '</div>';
  document.body.appendChild(ov);
  _ensureOnboardStyles();
  requestAnimationFrame(() => ov.classList.add('show'));
  try { _coachTrack('dance_onboard_shown', {}); } catch (e) {}
  const done = (started) => {
    try { localStorage.setItem(_ONBOARD_KEY, '1'); } catch (e) {}
    ov.classList.remove('show');
    setTimeout(() => { try { ov.remove(); } catch (e) {} }, 260);
    if (started) {
      try {
        _coachTrack('dance_onboard_start', { style: 'gLH' });
        const chip = document.querySelector('#ss2-styles .ss2-tile[data-genre="gLH"]');
        if (chip) chip.click();
      } catch (e) {}
    } else {
      try { _coachTrack('dance_onboard_skip', {}); } catch (e) {}
    }
  };
  ov.querySelector('#cob-start').addEventListener('click', () => done(true));
  ov.querySelector('#cob-skip').addEventListener('click', () => done(false));
  ov.querySelector('.cob-x').addEventListener('click', () => done(false));
  ov.addEventListener('click', (e) => { if (e.target === ov) done(false); });
}
let _onboardStyles = false;
function _ensureOnboardStyles() {
  if (_onboardStyles) return; _onboardStyles = true;
  const css = `
  #coach-onboard{position:fixed;inset:0;z-index:9800;display:flex;align-items:center;
    justify-content:center;padding:18px;background:rgba(6,4,14,.66);
    backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);opacity:0;
    transition:opacity .22s ease;}
  #coach-onboard.show{opacity:1;}
  #coach-onboard .cob-card{position:relative;width:100%;max-width:360px;box-sizing:border-box;
    background:linear-gradient(165deg,#1c1330,#241546);border:1px solid rgba(192,97,255,.28);
    border-radius:20px;padding:24px 22px 18px;color:#f4f1fb;font-family:inherit;
    box-shadow:0 24px 70px rgba(0,0,0,.55);transform:translateY(10px) scale(.98);
    transition:transform .22s cubic-bezier(.2,.8,.25,1);}
  #coach-onboard.show .cob-card{transform:none;}
  #coach-onboard .cob-x{position:absolute;top:12px;right:12px;background:none;border:none;
    color:#9b93b4;font-size:24px;line-height:1;cursor:pointer;padding:4px;border-radius:8px;}
  #coach-onboard .cob-x:hover{background:rgba(255,255,255,.06);color:#fff;}
  #coach-onboard .cob-badge{display:inline-block;font-size:11px;font-weight:800;
    text-transform:uppercase;letter-spacing:.4px;color:#e9b8ff;background:rgba(192,97,255,.16);
    border:1px solid rgba(192,97,255,.3);padding:4px 10px;border-radius:999px;margin-bottom:12px;}
  #coach-onboard .cob-title{margin:0 0 16px;font-size:23px;font-weight:850;letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#e7d8ff);-webkit-background-clip:text;background-clip:text;
    -webkit-text-fill-color:transparent;}
  #coach-onboard .cob-steps{display:flex;flex-direction:column;gap:12px;margin-bottom:20px;}
  #coach-onboard .cob-step{display:flex;gap:12px;align-items:flex-start;text-align:left;}
  #coach-onboard .cob-step b{font-size:14.5px;}
  #coach-onboard .cob-step small{color:#b0a8c9;font-size:12.5px;line-height:1.45;}
  #coach-onboard .cob-n{flex:0 0 26px;height:26px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:13px;font-weight:800;color:#fff;
    background:linear-gradient(135deg,#7c3aed,#db2777);}
  #coach-onboard .cob-cta{width:100%;padding:14px;border:none;border-radius:12px;font-size:15.5px;
    font-weight:800;cursor:pointer;color:#fff;font-family:inherit;
    background:linear-gradient(135deg,#7c3aed,#db2777);transition:filter .15s;}
  #coach-onboard .cob-cta:hover{filter:brightness(1.08);}
  #coach-onboard .cob-skip{width:100%;margin-top:8px;padding:8px;background:none;border:none;
    color:#9b93b4;font-size:12.5px;cursor:pointer;font-family:inherit;}
  #coach-onboard .cob-skip:hover{color:#d8d2ea;}`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
}
try { window.__maybeShowOnboarding = _maybeShowOnboarding; } catch (e) {}

function setFlag(visible, info) {
  // Soft flags = guard clamped a single bone for one frame but motion
  // kept playing. Log only; never paint the scary "motion flagged"
  // pill or post a chat message.
  if (info && info.soft) {
    console.debug('[coach] soft motion flag', info);
    return;
  }
  $('flag').style.display = visible ? 'block' : 'none';
  if (visible && info) addMsg('⚠ flagged: ' + JSON.stringify(info), 'flag');
}

// ── Client-side motion cache for instant re-play ─────────────────────
// /api/motion/data/<id>.json responses are deterministic per clip-id;
// memoise them so the second time the user (or coach) requests the
// same move it starts in < 50ms instead of waiting on a server fetch.
const _motionCache = new Map();
async function fetchMotion(clipId) {
  if (_motionCache.has(clipId)) return _motionCache.get(clipId);
  const p = fetch(APP_BASE + '/api/motion/data/' + clipId + '.json')
    .then(r => { if (!r.ok) throw new Error('motion ' + r.status); return r.json(); });
  _motionCache.set(clipId, p);
  try { return await p; }
  catch (e) { _motionCache.delete(clipId); throw e; }
}

// ── Friendly AI-quota modal (replaces raw RateLimitError lines) ───────
function showQuotaModal({ scope = 'rate', retry_hint = '', upgrade_url } = {}) {
  // Idempotent: if a modal is already open, just refresh its text.
  let bd = document.getElementById('quota-modal');
  if (!bd) {
    bd = document.createElement('div');
    bd.id = 'quota-modal';
    bd.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.55);' +
      'display:flex;align-items:center;justify-content:center;z-index:9999;';
    bd.innerHTML =
      '<div style="background:#1b1b22;color:#eee;border:1px solid #333;' +
      'border-radius:10px;max-width:420px;padding:22px 24px;font:14px/1.5 system-ui;">' +
        '<div style="font-size:16px;font-weight:600;margin-bottom:8px;">' +
          'Daily AI quota reached' +
        '</div>' +
        '<div id="quota-body" style="opacity:.85;margin-bottom:14px;"></div>' +
        '<div style="display:flex;gap:8px;justify-content:flex-end;">' +
          '<button id="quota-close" style="background:#2a2a33;color:#ddd;' +
          'border:1px solid #444;border-radius:6px;padding:6px 12px;' +
          'cursor:pointer;">Dismiss</button>' +
          '<a id="quota-upgrade" target="_blank" rel="noopener" ' +
          'style="background:#5b8def;color:#fff;text-decoration:none;' +
          'border-radius:6px;padding:6px 12px;">Upgrade</a>' +
        '</div>' +
      '</div>';
    document.body.appendChild(bd);
    bd.querySelector('#quota-close').addEventListener('click',
      () => bd.remove());
    bd.addEventListener('click', (ev) => { if (ev.target === bd) bd.remove(); });
  }
  const isDaily = scope === 'tokens_per_day';
  const body = isDaily
    ? "You've used today's free AI quota. " +
      'Coaching will resume tomorrow, or upgrade for unlimited access.'
    : "We're sending requests a bit too fast. " +
      (retry_hint ? ('Please ' + retry_hint + '.')
                  : 'Please wait a moment and try again.');
  bd.querySelector('#quota-body').textContent = body;
  const link = bd.querySelector('#quota-upgrade');
  // Upgrade target is configured by the deployer; no default business URL.
  link.href = upgrade_url || '#';
  link.style.display = isDaily ? 'inline-block' : 'none';
}

// ─── three.js scene ───────────────────────────────────────────────────
const canvas = $('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
// v32: ACES + 1.0 exposure was crushing the MToon mid-tones to
// black during dance — switch to Neutral tone-mapping (preserves
// saturation in shadows) and bump exposure so the avatar reads
// clearly even when she rotates her back to the key light.
renderer.toneMapping = THREE.NeutralToneMapping || THREE.ACESFilmicToneMapping;
// v109b: exposure pulled back (was 1.12). Combined with the lighter
// light rig below, the avatar's actual skin texture (#EFCCBD warm
// beige) now reads instead of clipping every channel to white.
renderer.toneMappingExposure = 0.80;
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
// v34d: enable real shadows for grounded contact — the missing
// contact shadow under the avatar was the #1 reason renders read as
// "floating in void" instead of standing on a stage (Blender parity
// complaint). PCFSoft + tight 5×5 m frustum keeps perf cost low
// since the camera never sees outside the immediate stage circle.
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
// Dance.AI studio look. v-contrast-fix: the old near-black background
// (0x07020f) + a single bright key made a high-contrast "black void with
// v115: Dance.AI look — deep indigo->black backdrop (like the brand
// mockup) with a neon-violet floor glow behind the avatar. Darker than
// before so the warm avatar + the purple floor light pop.
// v118: matched to the studio room's dark-plum walls so anything beyond
// the room edges blends into the same near-black plum (no bright seam).
scene.background = new THREE.Color(0x0d0a16);
scene.fog = new THREE.Fog(0x0d0a16, 18, 42);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

// v118: the procedural neon backdrop was removed — the Dance.AI studio
// room GLB (loaded below) now provides the real walls, purple LED strips,
// mirror, posters and ceiling lights, so the cosmetic gradient planes are
// no longer needed.

// Adaptive camera framing — narrow/portrait screens need a wider FOV
// and a slightly closer camera, otherwise the avatar reads as tiny and
// the stage backdrop dominates the canvas (the dance-ai-viewer used
// the same pattern).
function _aspect() {
  return (canvas.clientWidth || window.innerWidth) /
         (canvas.clientHeight || window.innerHeight);
}
function _pickFov() {
  const a = _aspect();
  if (a >= 1.4) return 38;        // wide desktop
  if (a >= 1.0) return 40;        // square / small landscape
  if (a >= 0.7) return 42;        // portrait tablet
  return 44;                       // phone portrait
}
function _pickCamZ() {
  // v118: pulled back so the studio room (walls, LED strips, mirror,
  // posters) is visible around the avatar, not just her body.
  // v120: room is now ~1.7x bigger, so sit a touch further back to show
  // the wider studio while keeping the avatar the clear hero.
  // v123: user felt the opening framing was too far — pull in ~1 unit so
  // the avatar reads bigger while the studio still shows around her.
  // v124: pull in a touch more per user request.
  const a = _aspect();
  if (a >= 1.4) return 4.0;
  if (a >= 1.0) return 4.3;
  if (a >= 0.7) return 4.5;
  return 4.7;                      // phone portrait
}
const camera = new THREE.PerspectiveCamera(_pickFov(), 1, 0.05, 100);
camera.position.set(0, 1.5, _pickCamZ());
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.95, -0.4); controls.enableDamping = true;
// v120: keep the user INSIDE the studio — they were zooming/orbiting out
// of the room ("get out of it"). Clamp the zoom range and stop the orbit
// from going under the floor or behind the back wall.
controls.minDistance = 3.0;
controls.maxDistance = 7.5;
controls.maxPolarAngle = Math.PI * 0.52;   // don't drop below the floor
controls.minPolarAngle = Math.PI * 0.20;   // don't fly to a top-down view
controls.enablePan = false;                  // panning let them slide outside

// v31: brighten the studio so the avatar doesn't darken into a
// near-silhouette the moment she rotates away from camera. The
// previous rig had key+fill+rim all pointing FROM behind, so any
// pose where she faced the camera left her front in toon-shader
// shadow. We add an explicit front-fill + raise the hemi.
// v32: previous rig had key+fill BEHIND the avatar, so her front
// was always in MToon shade-band shadow. Flip the layout: key now
// FROM THE FRONT, secondary fills wrap the body so no facing reads
// as silhouette. Also bumped hemisphere ambient hard.
// v109b: the avatar kept rendering WHITE even after the user set a warm
// skin tone in VRoid — the light rig was so strong it clipped every skin
// channel to 1.0 (pure white). Pulled the whole rig down so total
// irradiance on the body is ~1.0, letting the real texture colour show.
// Still ONE key light for shading/form so limbs read.
scene.add(new THREE.HemisphereLight(0xe8e0ff, 0x241a38, 0.20));
scene.add(new THREE.AmbientLight(0xffffff, 0.06));
const key = new THREE.DirectionalLight(0xfff0dd, 0.95);
key.position.set(-2.0, 4.5, 4.5); scene.add(key);
// v34d: key light casts the contact shadow. Tight ortho frustum
// around the avatar disc keeps the shadow map crisp (1024 is plenty
// at this scale). Bias trims acne on MToon meshes.
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.left = -2.5;
key.shadow.camera.right = 2.5;
key.shadow.camera.top = 3.0;
key.shadow.camera.bottom = -0.5;
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 12.0;
key.shadow.bias = -0.0006;
key.shadow.normalBias = 0.02;
key.shadow.radius = 4;
const fill = new THREE.DirectionalLight(0xdaeaff, 0.35);
fill.position.set(3.5, 3.5, 3.0); scene.add(fill);
// v-backlight-fix: the back rim light (was at 0,3,-4) threw a bright
// round specular hot-spot onto the StudioBackWall directly behind the
// avatar, which washed out the scene and made the dancer hard to read.
// The front/fill/belly lights already wrap the body, so the rim isn't
// needed for separation — removed it entirely.
// Front-fill — lights the SIDE FACING the camera so MToon's shade
// band doesn't swallow the entire body when she dances toward us.
const frontFill = new THREE.DirectionalLight(0xfff4e8, 0.50);
frontFill.position.set(0, 2.2, 6.0); scene.add(frontFill);
// Belly-up fill — kills the dark band on the lower torso / thighs
// that the screenshots kept showing.
const bellyFill = new THREE.DirectionalLight(0xffe8d6, 0.22);
bellyFill.position.set(0, 0.4, 5.5); scene.add(bellyFill);

// v34d: shadow-catcher disc at floor level. Invisible plane that
// only renders the cast shadow — gives the avatar a soft contact
// shadow even if the stage GLB hasn't loaded or doesn't include a
// proper ground mesh. ShadowMaterial is the cleanest way: alpha = 0
// everywhere except where a shadow falls. Slightly above y=0 to win
// the z-fight against the stage floor.
const shadowCatcher = new THREE.Mesh(
  new THREE.CircleGeometry(2.6, 64),
  new THREE.ShadowMaterial({ opacity: 0.38, transparent: true })
);
shadowCatcher.rotation.x = -Math.PI / 2;
shadowCatcher.position.y = 0.001;
shadowCatcher.receiveShadow = true;
scene.add(shadowCatcher);

// ─── Dance.AI studio stage (mirror + brand sign + floor) ──────────────
// The studio backdrop GLB lives at /asset/stage (extension-less so the
// studioos.fit nginx proxy never intercepts it). Loaded asynchronously
// so a slow CDN never blocks the avatar render. If the load fails we
// silently fall back to a plain disc floor.
const stageLoader = new GLTFLoader();
try {
  const draco = new DRACOLoader();
  draco.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/libs/draco/gltf/');
  draco.setDecoderConfig({ type: 'js' });
  stageLoader.setDRACOLoader(draco);
} catch (e) { /* draco optional */ }

let stageRoot = null;
stageLoader.load(APP_BASE + '/asset/studio_room', (gltf) => {
  // v118: load the purpose-built Dance.AI studio (5x4m room with dark
  // plum walls, warm-wood floor, purple LED strips along every floor/wall
  // seam + vertical back corners, full-length mirror, motivational
  // posters, plant, curtains, and 4 ceiling spotlights). The avatar
  // spawns at the baked-in `avatar_spawn_marker` (~origin).
  // v120: the base GLB is only 5x4m so the camera left the room almost
  // immediately ("very small"). Scale the whole room up so it reads as a
  // big, wide studio around the (real-size) avatar. The floor is at y=0
  // and we scale from the origin, so the floor stays at the avatar's feet.
  const ROOM_SCALE = 1.7;
  stageRoot = gltf.scene;
  stageRoot.position.set(0, 0, 0);
  stageRoot.scale.setScalar(ROOM_SCALE);
  const ledLight = [];
  stageRoot.traverse(o => {
    if (!o.isMesh) return;
    const n = (o.name || '').toLowerCase();
    const mat = o.material;
    const mn = (mat && mat.name ? mat.name : '').toLowerCase();
    // Hide the translucent debug spawn marker.
    if (n.includes('spawn') || mn.includes('spawn')) { o.visible = false; return; }
    o.castShadow = false;
    o.receiveShadow = n.includes('floor');
    if (!mat) return;
    // Purple LED strips: make them glow hard (neon look). No bloom pass,
    // so we ALSO drop point lights along them below for real illumination.
    if (mn.includes('led') || mn.includes('purple_led')) {
      mat.emissive = new THREE.Color(0xc850ff);
      mat.emissiveIntensity = 4.0;
      mat.toneMapped = false;
      mat.needsUpdate = true;
    } else if (mn.includes('ceiling_light')) {
      mat.emissive = new THREE.Color(0xffe6b0);
      mat.emissiveIntensity = 1.8;
      mat.toneMapped = false;
      mat.needsUpdate = true;
    } else if (mn.includes('mirror')) {
      // keep it a dim tinted glass so it doesn't throw a hot glare.
      mat.metalness = 0.55; mat.roughness = 0.18;
      if ('envMapIntensity' in mat) mat.envMapIntensity = 0.5;
      mat.needsUpdate = true;
    } else {
      // Matte walls / floor / ceiling: cut the env-map so the IBL doesn't
      // wash the dark-plum walls to pale lavender — keeps the room moody.
      if ('envMapIntensity' in mat) { mat.envMapIntensity = 0.18; mat.needsUpdate = true; }
    }
  });
  scene.add(stageRoot);

  // Real purple illumination from the LED strips (emissive alone casts no
  // light without a bloom pass). Low + around the room edges so the floor
  // and the avatar's sides pick up the neon wash, front stays true-colour.
  // Split the back seam into two so there's no single hot blob behind her.
  // Positions/ranges scale with the room so the wash still reaches the walls.
  const S = ROOM_SCALE;
  const addLED = (x, y, z, intensity, dist) => {
    const p = new THREE.PointLight(0xa845ff, intensity, dist * S, 2.0);
    p.position.set(x * S, y * S, z * S); scene.add(p); ledLight.push(p);
  };
  addLED(-1.4, 0.16, -1.85, 2.0, 6);  // back floor seam (left half)
  addLED(1.4, 0.16, -1.85, 2.0, 6);   // back floor seam (right half)
  addLED(-2.25, 0.18, 0, 2.4, 6);     // left floor seam
  addLED(2.25, 0.18, 0, 2.4, 6);      // right floor seam
  addLED(-2.25, 1.6, -1.85, 1.8, 6);  // back-left vertical
  addLED(2.25, 1.6, -1.85, 1.8, 6);   // back-right vertical
  // Warm ceiling spots — soft downlight so the room isn't flat.
  const addSpot = (x, z) => {
    const s = new THREE.PointLight(0xfff0d6, 1.3, 6 * S, 2.0);
    s.position.set(x * S, 2.55 * S, z * S); scene.add(s);
  };
  addSpot(-1.3, -1.1); addSpot(1.3, -1.1);
  addSpot(-1.3, 0.7);  addSpot(1.3, 0.7);
}, undefined, (err) => {
  console.warn('[coach] studio room load failed; using plain floor', err);
  // Fallback floor so the avatar isn't floating in void.
  const g = new THREE.Mesh(
    new THREE.CircleGeometry(3, 64),
    new THREE.MeshStandardMaterial({ color: 0x130b22, roughness: 0.92 }));
  g.rotation.x = -Math.PI / 2; scene.add(g);
});

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // Re-pick FOV + Z on aspect change so portrait flips still frame
  // the avatar nicely.
  const newFov = _pickFov();
  if (Math.abs(camera.fov - newFov) > 0.5) camera.fov = newFov;
  const newZ = _pickCamZ();
  if (Math.abs(camera.position.z - newZ) > 0.05) camera.position.z = newZ;
  // v118: the studio room is a fixed real-world size — do NOT rescale it
  // on resize (that warped the walls). The camera FOV/Z above handles
  // portrait vs landscape framing of the avatar inside the room.
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas); resize();

// ─── VRM loader ───────────────────────────────────────────────────────
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let currentVrm = null;
let player = null;
let life = null;
let limits = null;
let voice = new AzureVoice();
let voiceOn = true;
let micOn = false;

// ─── engagement bridge ───────────────────────────────────────────────
// Lets the behavioral director (engagement.js) speak through the coach
// voice + read coaching state without importing this module. Kept tiny.
window.__coach = {
  // Speak a short reactive line. Gated by the caller; we just play it.
  say(text, opts = {}) {
    if (!text || !voiceOn) return Promise.resolve();
    try { if (life) life.setSpeaking(true); } catch (e) {}
    try { addMsg(text, 'coach'); } catch (e) {}
    return voice.speak(text, { rate: opts.rate, pitch: opts.pitch })
      .catch(() => {})
      .finally(() => { try { if (life) life.setSpeaking(false); } catch (e) {} });
  },
  isSpeaking() { try { return !!(life && life.speaking); } catch (e) { return false; } },
  inSession() { return !!window.__inSession; },
  get player() { return player; },
  get life() { return life; },
  get bpm() { try { return life?._idleBpm || window.__lastBpm || 0; } catch (e) { return 0; } },
};

// ─── backing-track music ─────────────────────────────────────────────
// One <audio> element re-used for every clip. We duck the music
// volume while the coach is talking so the TTS stays intelligible,
// then ramp back up between sentences.
const MUSIC_BASE_VOLUME = 0.35;   // headroom for voice on top
const MUSIC_DUCK_VOLUME = 0.08;   // barely audible under coach speech
const _music = (typeof Audio !== 'undefined') ? new Audio() : null;
if (_music) {
  _music.loop = true;
  _music.volume = MUSIC_BASE_VOLUME;
  _music.preload = 'auto';
  _music.crossOrigin = 'anonymous';
}
let _musicSrc = '';
function playMusic(url) {
  if (!_music || !url) return;
  const full = url.startsWith('http') ? url : (APP_BASE + url);
  if (_musicSrc !== full) {
    _musicSrc = full;
    _music.src = full;
  }
  _music.volume = MUSIC_BASE_VOLUME;
  // Browsers gate autoplay on user gesture — but by the time the
  // user is sending the coach messages they've already clicked, so
  // .play() succeeds. Silent-catch the rare exception.
  _music.play().catch(() => {});
}
function stopMusic() {
  if (!_music) return;
  try { _music.pause(); _music.currentTime = 0; } catch {}
  _musicSrc = '';
}
function duckMusic(on) {
  if (!_music) return;
  _music.volume = on ? MUSIC_DUCK_VOLUME : MUSIC_BASE_VOLUME;
}
window.__music = { play: playMusic, stop: stopMusic, duck: duckMusic };

// ─── pending-motion (voice/motion sync) ─────────────────────────────
// When an `avatar.play` event arrives we don't kick the loop yet —
// we wait for the coach's voice line to actually start speaking so
// what the teacher SAYS matches what the avatar IS DOING at that
// exact moment.
let _pendingMotion = null;
let _pendingMotionTimer = null;
async function _preloadClip(clipId) {
  // Fetch the JSON + stand the avatar at frame 0 so there's no
  // perceptible delay when we flip the switch to "play".
  try {
    const data = await fetchMotion(clipId);
    if (!player || !data) return;
    player.load(data);
  } catch {}
}
function _flushPendingMotion() {
  if (!_pendingMotion) return;
  const p = _pendingMotion;
  _pendingMotion = null;
  clearTimeout(_pendingMotionTimer);
  _pendingMotionTimer = null;
  if (p.music_url) playMusic(p.music_url);
  loadAndPlayClip(p.clip_id, p.opts);
}

// ─── "Start dancing" launcher visibility (Q1) ───────────────────────
// The #ss2-reopen pill re-opens the style picker. It should be visible
// ONLY when the front door is dismissed AND the avatar is idle — never
// while she is actually dancing (that read as "why press Start when she's
// already dancing?"). Single source of truth, safe to call from anywhere.
function _syncReopenBtn() {
  try {
    const b = document.getElementById('ss2-reopen');
    if (!b) return;
    const dancing = !!(window.__player && window.__player.playing);
    // v217: also hide while a guided session is active OR the steps rail is
    // showing a breakdown — the user already chose what to do, so a second
    // "Start dancing" button read as confusing/redundant ("why start again?").
    const inSession = !!window.__inSession;
    let railOpen = false;
    try { const r = document.getElementById('coach-rail'); railOpen = !!(r && r.classList.contains('show')); } catch (e) {}
    const show = (window.__frontDoorOpen === false) && !dancing && !inSession && !railOpen;
    b.style.display = show ? 'flex' : 'none';
  } catch (e) {}
}

// ─── free-dance auto-variety (Q3) ───────────────────────────────────
// In free-dance (NOT a guided session), a single clip loops forever, so
// she "repeats the same moves". This rotates to a fresh clip in the SAME
// style every ~22 s of uninterrupted looping. It yields instantly to any
// new user prompt / session / drill (those call _cancelVariety()).
let _varietyTimer = null;
const VARIETY_MS = 22000;
function _cancelVariety() {
  if (_varietyTimer) { clearTimeout(_varietyTimer); _varietyTimer = null; }
}
function _scheduleVariety() {
  _cancelVariety();
  // Only auto-rotate in free-dance: not during a guided session (its own
  // engine drives rotation) and only while a clip is actually looping.
  if (window.__inSession) return;
  const lp = window.__lastPlay;
  if (!lp || !lp.opts || !lp.opts.loop) return;
  if (!(window.__player && window.__player.playing)) return;
  _varietyTimer = setTimeout(_rotateVariety, VARIETY_MS);
}
async function _rotateVariety() {
  _varietyTimer = null;
  // Re-check we're still free-dancing an active loop.
  if (window.__inSession) return;
  const lp = window.__lastPlay;
  if (!lp || !lp.opts || !lp.opts.loop) return;
  if (!(window.__player && window.__player.playing)) return;
  const cur = lp.clipId || '';
  const genre = cur.slice(0, 3);   // clip ids are genre-prefixed (gHO_, gKR_…)
  try {
    const r = await fetch(APP_BASE + '/api/motion/variety?genre='
      + encodeURIComponent(genre) + '&exclude=' + encodeURIComponent(cur));
    if (!r.ok) { _scheduleVariety(); return; }
    const j = await r.json();
    const next = (j.clips && j.clips[0]) || null;
    // If a session started or she stopped while we were fetching, abort.
    if (!next || window.__inSession
        || !(window.__player && window.__player.playing)) {
      _scheduleVariety(); return;
    }
    if (next.music_url) { try { playMusic(next.music_url); } catch {} }
    loadAndPlayClip(next.clip_id, { ...lp.opts, loop: true });
  } catch (e) {
    _scheduleVariety();
  }
}

async function loadVrm(name) {
  setStatus('loading ' + name + '…');
  const url = APP_BASE + '/api/vrm/' + encodeURIComponent(name);
  // Cache raw VRM bytes per character so swaps after the first time
  // are instant (no network round-trip). VRMs are ~5-15 MB each — a
  // handful in memory is fine on every modern device.
  if (!window.__vrmBlobs) window.__vrmBlobs = new Map();
  let parseUrl = window.__vrmBlobs.get(name);
  if (!parseUrl) {
    try {
      let buf = null;
      // v-ux10: reuse the avatar bytes prefetched at page-parse time
      // (window.__preFetch.vrm) for the default character, so the avatar
      // appears with no extra network wait on first load.
      if (window.__preVrmName === name && window.__preFetch && window.__preFetch.vrm) {
        try { buf = await window.__preFetch.vrm; } catch (e) { buf = null; }
      }
      if (!buf) {
        const r = await fetch(url);
        if (!r.ok) throw new Error('vrm ' + r.status);
        buf = await r.arrayBuffer();
      }
      const blob = new Blob([buf], { type: 'model/gltf-binary' });
      parseUrl = URL.createObjectURL(blob);
      window.__vrmBlobs.set(name, parseUrl);
    } catch (e) {
      // Fall back to direct URL load (loader handles it).
      parseUrl = url;
    }
  }
  return new Promise((ok, fail) => {
    loader.load(parseUrl, async (gltf) => {
      const vrm = gltf.userData.vrm;
      VRMUtils.rotateVRM0(vrm);
      if (currentVrm) {
        scene.remove(currentVrm.scene);
        VRMUtils.deepDispose(currentVrm.scene);
      }
      currentVrm = vrm;
      scene.add(vrm.scene);
      // v34d: cast shadows from every avatar mesh so the contact
      // shadow on the floor reads as a real grounded silhouette.
      vrm.scene.traverse((o) => {
        if (o.isMesh || o.isSkinnedMesh) {
          o.castShadow = true;
          o.receiveShadow = false;   // self-shadow on MToon = noisy
        }
      });
      // v31: MToon shaders ship with a hard shade-band that turns
      // half the body pitch-black when the avatar rotates away from
      // the key light. Lift the shadeColor toward the base color and
      // soften the shading shift so the dance side isn't a silhouette.
      try {
        vrm.scene.traverse((o) => {
          if (!o.isMesh) return;
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          for (const m of mats) {
            if (!m) continue;
            // v32: MToon material — push HARD toward the lit
            // color so the dance side never silhouettes. Earlier
            // 0.55 lerp was still leaving the body half-black.
            // v84: bring back a SOFT light->dark gradient so the
            // avatar has depth/volume instead of a flat 'shiny' look.
            // We still lift the shade enough that the dance side never
            // goes pitch-black, but not so far that it reads as flat.
            if (m.shadeColorFactor && m.shadeColorFactor.lerp) {
              m.shadeColorFactor.lerp({ r: 1, g: 1, b: 1 }, 0.62);
            }
            // Toony=0 keeps the band SMOOTH (a gradient, not a hard
            // 2-tone). A mild negative shift moves the terminator so
            // most of the body is lit but a soft shaded side remains.
            if ('shadingShiftFactor' in m) m.shadingShiftFactor = -0.18;
            if ('shadingToonyFactor' in m) m.shadingToonyFactor = 0.0;
            // Partial GI equalisation = some directional wrap (depth)
            // without the old fully-flat look.
            if ('giEqualizationFactor' in m) m.giEqualizationFactor = 0.55;
            // Make sure the lit/shade emissive aren't muted.
            if (m.outlineWidthFactor !== undefined) m.outlineWidthFactor = 0.0;
          }
        });
      } catch (e) { /* non-MToon material — fine */ }
      player = new MotionPlayer(vrm, limits);
      try { window.__player = player; window.__vrm = vrm; } catch(_) {}
      player.onflag = (info) => setFlag(true, info);
      // Settle into a natural standing pose (frame 0 of a calm clip)
      // BEFORE AvatarLife snapshots its rest pose. Otherwise the rest
      // pose is the VRM authoring T-pose and idle looks like a scarecrow.
      await applyIdleStance().catch(() => {});
      life = new AvatarLife(vrm, camera);
      life.attach(player);
      life.setMood('relaxed');
      setStatus('coach connected');
      // v215: LOAD-TIME TELEMETRY. Measure boot→interactive so we can rule out
      // (or confirm) slow cold-starts as a bounce cause. window.__coachBootT0 is
      // stamped at script start (coach.html). Fire once.
      try {
        if (!window.__coachReadyFired) {
          window.__coachReadyFired = true;
          const t0 = window.__coachBootT0 || 0;
          const loadMs = t0 ? Math.max(0, Date.now() - t0) : Math.round(performance.now());
          _coachTrack('dance_ready', { load_ms: loadMs });
          if (loadMs > 8000) _coachTrack('dance_load_slow', { load_ms: loadMs });
        }
      } catch (e) {}
      // v34c: mic + audio on by default. Auto-click the mic button
      // shortly after the studio is ready so the user gets the
      // browser permission prompt without having to hunt for it.
      // Browsers require a user-gesture for getUserMedia in some
      // configs — in that case the click() is a no-op and the user
      // can still click the mic button manually. Audio context is
      // resumed on the first real user interaction (the WebAudio
      // autoplay policy can't be bypassed).
      try {
        if (!window.__autoMicTried) {
          window.__autoMicTried = true;
          setTimeout(() => {
            // v90: if live voice is the default voice mode, don't also
            // fire the Azure mic — S2S owns the mic and would conflict.
            if (window.__s2sDefault) return;
            try {
              const m = document.getElementById('mic');
              if (m && !m.classList.contains('rec')) m.click();
            } catch (e) {}
          }, 600);
          // Resume audio context on the very first user gesture.
          const resume = () => {
            try {
              if (window.__audioCtx && window.__audioCtx.state === 'suspended') {
                window.__audioCtx.resume();
              }
            } catch (e) {}
            window.removeEventListener('pointerdown', resume, true);
            window.removeEventListener('keydown', resume, true);
          };
          window.addEventListener('pointerdown', resume, true);
          window.addEventListener('keydown', resume, true);
        }
      } catch (e) { /* ignore */ }
      // VRM mesh + lighting are now on the scene. Fade the splash
      // out so the user sees the studio instead of the bare floor.
      try {
        const sp = document.getElementById('splash');
        if (sp && !sp.classList.contains('fadeout')) {
          sp.classList.add('fadeout');
          setTimeout(() => sp.classList.add('gone'), 650);
        }
      } catch (e) { /* no splash present, fine */ }
      // v156: the FIRST time the avatar actually renders, tell the outer
      // "Warming up your studio" boot loader we're really done — instead of
      // it always waiting out the 45s fake-progress safety timeout. Guarded
      // so character swaps later in the session don't re-trigger it.
      try {
        if (!window.__vrmBootDone) {
          window.__vrmBootDone = true;
          window.__boot?.set(97, 'Studio ready!');
          window.__boot?.done();
        }
      } catch (e) {}
      ok(vrm);
    }, undefined, fail);
  });
}

// ─── motion fetch + drive ─────────────────────────────────────────────
const IDLE_CLIP_ID = 'gLO_sBM_cAll_d13_mLO0_ch01';   // calm locking stance

async function applyIdleStance() {
  if (!player) return;
  // Procedural relaxed-standing pose — arms hanging at sides, soft
  // elbow bend. Vastly more natural than freezing on frame 0 of any
  // dance clip (those start in stance with arms wide).
  player.applyRestPose();
}

async function loadAndPlayClip(clipId, opts = {}) {
  setFlag(false);
  let data;
  try { data = await fetchMotion(clipId); }
  catch (e) { addMsg('motion fetch failed', 'flag'); return; }
  if (data.safety && data.safety.severity === 'fail') {
    addMsg('clip ' + clipId + ' failed safety pre-check', 'flag'); return;
  }
  player.load(data);
  player.play(opts);
  setPlayPauseLabel(false);
  // Remember the active clip so we can resume from the same point
  // when the user swaps character mid-dance.
  window.__lastPlay = { clipId, opts, startedAt: Date.now() };
  _syncReopenBtn();       // Q1: hide the "Start dancing" pill while dancing
  _scheduleVariety();     // Q3: rotate to a fresh clip after a while (free-dance)
}

// ─── drill controller ────────────────────────────────────────────────
// Runs N repeats of a count window, ramping speed_start → speed_end,
// optionally mirroring alternate reps, and speaking a coach cue
// between loops if metadata provided one.
let _drillToken = 0;

// ─── v34: GUIDED BREAKDOWN ────────────────────────────────────────────
// Sequence: legs slow → arms slow → full slow → full normal. Each stage
// isolates the named parts, sets speed, narrates the cue via TTS, and
// waits stage_seconds before advancing.
let _breakdownToken = 0;

// v210: SYNCED STEP CAPTION. The single guarantee that "what's said == what's
// shown": when a step's segment starts playing on the avatar, we flip this
// caption to that step's name+count in the SAME tick. The learner always reads
// the label of the move they are literally watching, even if TTS/live-voice
// audio leads or lags by a beat.
function showStepCaption(label, count) {
  try {
    const el = document.getElementById('step-caption');
    if (!el) return;
    if (!label) { el.classList.remove('show'); el.innerHTML = ''; return; }
    el.innerHTML = `<span class="sc-label"></span>` +
                   (count ? `<span class="sc-count"></span>` : '');
    el.querySelector('.sc-label').textContent = label;
    if (count) el.querySelector('.sc-count').textContent = 'count ' + count;
    el.classList.add('show');
  } catch (e) {}
}
function hideStepCaption() { showStepCaption(null); }

async function runBreakdown(evt) {
  const token = ++_breakdownToken;
  const stages = Array.isArray(evt.stages) ? evt.stages : [];
  const stageSec = Math.max(4, Math.min(16, evt.stage_seconds || 8));
  if (!player) return;
  const fps = (player.data && (player.data.fps)) || 30;
  const nFrames = (player.data && (player.data.n_frames || player.data.frames)) || 0;
  // Mirror the steps into the universal side rail so the learner can see the
  // whole plan and jump around — in every surface the coach runs (standalone
  // page, Coach tab). Embedded StudioOS lessons use the parent's own rail.
  try { renderCoachRail(stages, evt.title || evt.style || ''); } catch (e) {}
  // v223 SAFETY NET — "said it's moving but it isn't". break_down is supposed
  // to have already loaded+started the clip (via pick_and_play), but if that
  // hand-off didn't flush (e.g. the pending-motion voice-sync never fired, or
  // the clip loaded a beat late) the avatar can sit still while the coach
  // narrates the move. Force the loaded clip to actually PLAY here so the
  // avatar is GUARANTEED to be moving the moment a breakdown begins.
  try {
    if (player) {
      player._idleHold = false;
      const notMoving = !player.playing;
      if (notMoving && player.data) {
        player.play({ speed: 0.85, loop: true });
      } else if (notMoving && evt.clip_id) {
        // No clip data loaded yet — load then play the breakdown's clip.
        try { loadAndPlayClip(evt.clip_id, { loop: true, speed: 0.85 }); } catch (e) {}
      }
    }
  } catch (e) {}
  // In live-voice mode Gemini narrates; otherwise Azure TTS speaks the cue and
  // we hold each step until the audio finishes so nothing bleeds into the next
  // move. Either way the WINDOW + CAPTION below are locked to the avatar.
  const s2s = _s2sOn;

  for (let _i = 0; _i < stages.length; _i++) {
    const stage = stages[_i];
    if (token !== _breakdownToken) { hideStepCaption(); return; }
    try { setCoachRailActive(_i); } catch (e) {}

    // 1) WINDOW the avatar to EXACTLY this move's segment (if the learned step
    //    carries start_s/end_s). This is what makes "shown == said": the avatar
    //    loops only the chest-pop while we talk about the chest-pop.
    const hasSeg = typeof stage.start_s === 'number' &&
                   typeof stage.end_s === 'number' && stage.end_s > stage.start_s;
    if (hasSeg && nFrames > 0) {
      const fromFrame = Math.max(0, Math.floor(stage.start_s * fps));
      const toFrame = Math.min(nFrames, Math.ceil(stage.end_s * fps));
      try {
        player._idleHold = false;
        player.play({ speed: stage.speed || 0.5, loop: true,
                      fromFrame, toFrame });
      } catch {}
    } else {
      try { player.isolate(stage.parts && stage.parts.length ? stage.parts : null); } catch {}
      try { player.speed = stage.speed || 1.0; } catch {}
    }

    // 2) CAPTION flips in the same tick the segment starts → visual sync.
    if (stage.label) showStepCaption(stage.label, stage.count || '');

    // 3) Narrate. Non-live: speak now and HOLD until the audio ends (so the
    //    spoken move and the shown move stay together). Live: Gemini already
    //    narrates; we still hold a musical minimum so the move is seen enough.
    if (stage.cue) {
      addMsg(stage.cue, 'tool');
      let spoke = null;
      if (voiceOn && voice && !s2s) {
        try { spoke = voice.speak(stage.cue, {}); } catch {}
      }
      // Hold = the longer of: the spoken line finishing, or the move looping
      // ~twice, capped by stageSec so a long cue can't stall the flow.
      const segDur = hasSeg ? Math.max(1.2, (stage.end_s - stage.start_s) /
                                             Math.max(0.25, stage.speed || 0.5)) : stageSec;
      const minHold = Math.min(stageSec, Math.max(2.5, segDur * 2));
      const holders = [new Promise((ok) => setTimeout(ok, minHold * 1000))];
      if (spoke && spoke.then) holders.push(spoke.catch(() => {}));
      await Promise.all(holders);
    } else {
      await new Promise((ok) => setTimeout(ok, stageSec * 1000));
    }
  }
  if (token !== _breakdownToken) return;
  try { player.unisolate(); } catch {}
  try { player.speed = 1.0; } catch {}
  hideStepCaption();
  try { markCoachRailComplete(); } catch (e) {}
  addMsg('breakdown complete — try it with me', 'tool');
}

// ─── UNIVERSAL STEP RAIL ──────────────────────────────────────────────
// Whenever the agent breaks a dance into steps (any style chip or an ask
// like "teach me this"), we mirror those steps into a clickable right-side
// pane. Clicking a step cancels the auto-sequence and re-teaches JUST that
// move (windowed segment on loop + narrated cue), so the learner controls
// the pace. Hidden in embedded StudioOS lessons — the parent owns that rail.
let _railStages = [];
let _railActive = -1;
// 'teach' rails are clickable (jump to a move); 'session' rails mirror the
// guided-session phase plan and are read-only (the server drives progression).
let _railMode = 'teach';

function _railWideDock() {
  try { return window.matchMedia('(min-width: 1024px)').matches; } catch (e) { return false; }
}

function renderCoachRail(stages, title) {
  if (_embeddedLesson) return;
  _railMode = 'teach';
  _railStages = Array.isArray(stages) ? stages : [];
  _railActive = -1;
  const list = document.getElementById('coach-rail-list');
  const toggle = document.getElementById('coach-rail-toggle');
  const countEl = document.getElementById('coach-rail-count');
  const titleEl = document.getElementById('coach-rail-title');
  const subEl = document.getElementById('coach-rail-sub');
  if (!list) return;
  if (titleEl) titleEl.textContent = title ? String(title) : "Today's breakdown";
  if (subEl) subEl.textContent = "Tap any step — I'll play it and explain.";
  list.innerHTML = '';
  if (!_railStages.length) { closeCoachRail(); if (toggle) toggle.classList.remove('show'); return; }
  _railStages.forEach((stage, i) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'cr-step';
    item.setAttribute('role', 'listitem');
    item.dataset.idx = String(i);
    const num = document.createElement('span');
    num.className = 'cr-num';
    num.textContent = String(i + 1);
    const body = document.createElement('div');
    body.className = 'cr-body';
    const titleRow = document.createElement('div');
    titleRow.className = 'cr-title';
    const label = document.createElement('b');
    label.textContent = stage.label || ('Step ' + (i + 1));
    titleRow.appendChild(label);
    if (stage.count) {
      const c = document.createElement('span');
      c.className = 'cr-count';
      c.textContent = 'count ' + stage.count;
      titleRow.appendChild(c);
    }
    body.appendChild(titleRow);
    if (stage.cue) {
      const cue = document.createElement('p');
      cue.className = 'cr-cue';
      cue.textContent = stage.cue;
      body.appendChild(cue);
    }
    const play = document.createElement('span');
    play.className = 'cr-play';
    play.textContent = '▶';
    item.appendChild(num);
    item.appendChild(body);
    item.appendChild(play);
    item.addEventListener('click', () => _railJump(i));
    list.appendChild(item);
  });
  if (countEl) countEl.textContent = String(_railStages.length);
  if (toggle) toggle.classList.add('show');
  // Wide screens dock the rail open; phones reveal a "Steps" pill to open it.
  if (_railWideDock()) openCoachRail(); else if (toggle) toggle.classList.remove('docked');
}

function setCoachRailActive(i) {
  _railActive = i;
  const list = document.getElementById('coach-rail-list');
  if (!list) return;
  Array.from(list.children).forEach((el, idx) => {
    el.classList.toggle('active', idx === i);
    if (idx < i) el.classList.add('done');
  });
  const active = list.children[i];
  if (active && active.scrollIntoView) {
    try { active.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (e) {}
  }
}

function markCoachRailComplete() {
  const list = document.getElementById('coach-rail-list');
  if (!list) return;
  Array.from(list.children).forEach((el) => el.classList.add('done'));
}

// Guided style/warm-up sessions: mirror the server's phase plan into the rail
// as a read-only "here's your session" pane. The server drives progression, so
// these steps aren't click-to-jump — we just highlight the active phase.
function renderSessionRail(session) {
  if (_embeddedLesson || !session) return;
  const plan = Array.isArray(session.plan) ? session.plan : [];
  _railMode = 'session';
  _railStages = plan;
  _railActive = -1;
  const list = document.getElementById('coach-rail-list');
  const toggle = document.getElementById('coach-rail-toggle');
  const countEl = document.getElementById('coach-rail-count');
  const titleEl = document.getElementById('coach-rail-title');
  const subEl = document.getElementById('coach-rail-sub');
  if (!list) return;
  if (titleEl) titleEl.textContent = session.template_title || 'Your session';
  if (subEl) subEl.textContent = 'Your coach guides each part — follow along.';
  list.innerHTML = '';
  if (!plan.length) { closeCoachRail(); if (toggle) toggle.classList.remove('show'); return; }
  plan.forEach((phase, i) => {
    const item = document.createElement('div');
    item.className = 'cr-step';
    item.setAttribute('role', 'listitem');
    const num = document.createElement('span');
    num.className = 'cr-num';
    num.textContent = String(i + 1);
    const body = document.createElement('div');
    body.className = 'cr-body';
    const titleRow = document.createElement('div');
    titleRow.className = 'cr-title';
    const label = document.createElement('b');
    label.textContent = phase.label || phase.name || ('Part ' + (i + 1));
    titleRow.appendChild(label);
    body.appendChild(titleRow);
    if (phase.cue) {
      const cue = document.createElement('p');
      cue.className = 'cr-cue';
      cue.textContent = phase.cue;
      body.appendChild(cue);
    }
    item.appendChild(num);
    item.appendChild(body);
    list.appendChild(item);
  });
  if (countEl) countEl.textContent = String(plan.length);
  if (toggle) toggle.classList.add('show');
  if (session.phase_idx != null) setCoachRailActive(session.phase_idx);
  if (_railWideDock()) openCoachRail();
}

function openCoachRail() {
  const rail = document.getElementById('coach-rail');
  const toggle = document.getElementById('coach-rail-toggle');
  const app = document.getElementById('app');
  if (!rail) return;
  rail.classList.add('show');
  rail.setAttribute('aria-hidden', 'false');
  if (toggle) { toggle.setAttribute('aria-expanded', 'true'); toggle.classList.add('docked'); }
  if (app && _railWideDock()) app.classList.add('rail-open');
}

function closeCoachRail() {
  const rail = document.getElementById('coach-rail');
  const toggle = document.getElementById('coach-rail-toggle');
  const app = document.getElementById('app');
  if (!rail) return;
  rail.classList.remove('show');
  rail.setAttribute('aria-hidden', 'true');
  if (toggle) { toggle.setAttribute('aria-expanded', 'false'); toggle.classList.remove('docked'); }
  if (app) app.classList.remove('rail-open');
}

async function _railJump(i) {
  const stage = _railStages[i];
  if (!stage || !player) return;
  // Cancel any running auto-sequence so it doesn't fight the manual jump.
  const token = ++_breakdownToken;
  setCoachRailActive(i);
  const fps = (player.data && player.data.fps) || 30;
  const nFrames = (player.data && (player.data.n_frames || player.data.frames)) || 0;
  // v217: steps come in TWO shapes — learned choreography carries a seconds
  // window (start_s/end_s); the auto-segmenter carries a FRAME window
  // (frame_start/frame_end). Previously only the seconds shape was handled, so
  // clicking a segmenter step (Breaking / any guided-session breakdown) did
  // nothing. Resolve either shape to a [fromFrame,toFrame] window.
  let fromFrame = null, toFrame = null;
  if (typeof stage.start_s === 'number' && typeof stage.end_s === 'number' && stage.end_s > stage.start_s) {
    fromFrame = Math.max(0, Math.floor(stage.start_s * fps));
    toFrame = Math.ceil(stage.end_s * fps);
  } else if (typeof stage.frame_start === 'number' && typeof stage.frame_end === 'number' && stage.frame_end > stage.frame_start) {
    fromFrame = Math.max(0, Math.floor(stage.frame_start));
    toFrame = Math.ceil(stage.frame_end);
  } else if (typeof stage.beat_start === 'number' && nFrames > 0) {
    // v219: guided-session key_cues breakdowns carry a BEAT window
    // (beat_start/beat_end) — NOT seconds/frames. Without this, every step fell
    // through to "play the whole clip", so tapping any step looked identical
    // ("same step"). Map beats → frames using the max beat across all steps as
    // the clip's total count, so each step isolates its own slice of the move.
    let maxBeat = 0;
    for (const s of (_railStages || [])) {
      if (typeof s.beat_end === 'number') maxBeat = Math.max(maxBeat, s.beat_end);
      if (typeof s.beat_start === 'number') maxBeat = Math.max(maxBeat, s.beat_start);
    }
    if (maxBeat > 0) {
      const bEnd = (typeof stage.beat_end === 'number') ? stage.beat_end : stage.beat_start;
      fromFrame = Math.max(0, Math.floor((stage.beat_start - 1) / maxBeat * nFrames));
      toFrame = Math.min(nFrames, Math.ceil(bEnd / maxBeat * nFrames));
    }
  }
  if (nFrames > 0 && toFrame != null) toFrame = Math.min(nFrames, toFrame);
  const hasSeg = fromFrame != null && toFrame != null && toFrame > fromFrame;
  if (hasSeg) {
    try {
      player._idleHold = false;
      player.play({ speed: stage.speed || 0.5, loop: true, fromFrame, toFrame });
      if (typeof player.poseFrame === 'function') player.poseFrame(fromFrame);
    } catch (e) {}
  } else {
    // No window on this step (e.g. "all together" / "full speed") — isolate the
    // named parts if any, else play the whole clip at the step's speed so the
    // click ALWAYS produces visible motion instead of silently no-op'ing.
    try {
      player._idleHold = false;
      if (stage.parts && stage.parts.length) {
        player.isolate(stage.parts);
      } else {
        player.unisolate();
      }
      player.play({ speed: stage.speed || 0.75, loop: true });
    } catch (e) {}
  }
  if (stage.label) showStepCaption(stage.label, stage.count || '');
  const narration = [stage.cue, stage.detail, stage.mistake ? ('Watch out: ' + stage.mistake) : '']
    .filter(Boolean).join(' ');
  if (narration) {
    addMsg(narration, 'tool');
    if (voiceOn && voice && !_s2sOn && token === _breakdownToken) {
      try { await voice.speak(narration, {}); } catch (e) {}
    }
  }
  // On phones the pane covers the avatar — auto-close after the pick so the
  // learner can watch the move, then reopen from the pill.
  if (!_railWideDock()) closeCoachRail();
}

// Wire the rail's open/close controls once the DOM is ready.
(function initCoachRailControls() {
  const bind = () => {
    const toggle = document.getElementById('coach-rail-toggle');
    const close = document.getElementById('coach-rail-close');
    const back = document.getElementById('coach-rail-back');
    if (toggle) toggle.addEventListener('click', () => {
      const rail = document.getElementById('coach-rail');
      if (rail && rail.classList.contains('show')) closeCoachRail(); else openCoachRail();
    });
    if (close) close.addEventListener('click', closeCoachRail);
    // v217: "Pick another style" — stop whatever's running, close the rail, and
    // re-open the style picker so the learner can switch (e.g. Hip-Hop → Breaking)
    // instead of being stranded on one breakdown with no way back.
    if (back) back.addEventListener('click', () => {
      try { ++_breakdownToken; } catch (e) {}
      try { if (window.__player && window.__player.stop) window.__player.stop(); } catch (e) {}
      // End any running guided session via its HUD "End" control if present.
      try { const endBtn = document.getElementById('sess-end') || document.getElementById('session-end'); if (endBtn) endBtn.click(); } catch (e) {}
      try { window.__inSession = false; window.__sessionHUD && window.__sessionHUD.hide && window.__sessionHUD.hide(); } catch (e) {}
      try { hideStepCaption(); } catch (e) {}
      closeCoachRail();
      try { _coachTrack('dance_rail_back', {}); } catch (e) {}
      if (typeof window.__openFrontDoor === 'function') { try { window.__openFrontDoor(); return; } catch (e) {} }
      // Fallback: click the reopen pill which re-opens the picker.
      try { const rb = document.getElementById('ss2-reopen'); if (rb) { rb.style.display = 'flex'; rb.click(); } } catch (e) {}
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else { bind(); }
})();

// ─── StudioOS embedded choreography lesson bridge ──────────────────
// The parent lesson owns navigation and teaching notes. The coach keeps the
// avatar loaded and accepts exact segment commands without iframe reloads.
const _embeddedLesson = (() => {
  try { return new URLSearchParams(location.search).get('embedded') === 'lesson'; }
  catch (e) { return false; }
})();
if (_embeddedLesson) document.documentElement.classList.add('lesson-embedded');

function _lessonParentAllowed(evt) {
  if (!_embeddedLesson || evt.source !== window.parent) return false;
  try {
    const parentOrigin = new URL(document.referrer).origin;
    return evt.origin === parentOrigin;
  } catch (e) {
    return evt.origin === location.origin;
  }
}

function _notifyLessonParent(type, detail = {}) {
  if (!_embeddedLesson || !window.parent || window.parent === window) return;
  try {
    const target = document.referrer ? new URL(document.referrer).origin : location.origin;
    window.parent.postMessage({ source: 'danceai.coach', type, ...detail }, target);
  } catch (e) {}
}

let _lessonCommandToken = 0;
let _pendingLessonCommand = null;

async function _teachEmbeddedStep(step, speed) {
  if (!player || !step) return;
  if (!player.data) {
    _pendingLessonCommand = { step, speed };
    return;
  }
  const commandToken = ++_lessonCommandToken;
  ++_breakdownToken;
  const requestedMotion = String(step.motion_id || '').trim();
  const loadedMotion = window.__lastPlay?.clipId || '';
  if (requestedMotion && requestedMotion !== loadedMotion) {
    _notifyLessonParent('lesson.step.loading', { index: step.index });
    try {
      const data = await fetchMotion(requestedMotion);
      if (commandToken !== _lessonCommandToken) return;
      player.load(data);
      window.__lastPlay = { clipId: requestedMotion, opts: {} };
    } catch (e) {
      _notifyLessonParent('lesson.error', {
        index: step.index,
        message: 'This move could not be loaded. Please try again.',
      });
      return;
    }
  }
  const fps = Number(player.data.fps) || 30;
  const nFrames = Number(player.data.n_frames || player.data.frames) || 0;
  const fromFrame = Math.max(0, Math.floor(Number(step.start_s || 0) * fps));
  const toFrame = nFrames > 0
    ? Math.min(nFrames, Math.ceil(Number(step.end_s || 0) * fps))
    : Math.ceil(Number(step.end_s || 0) * fps);
  if (toFrame <= fromFrame) return;

  const playbackSpeed = Math.max(0.25, Math.min(1.25, Number(speed) || 0.5));
  // CRITICAL: clear the idle-hold latch. On deep-link arrival (and after any
  // safety fallback) `_idleHold` is left true, which makes the render loop snap
  // the avatar back to the rest pose on EVERY frame — so play() runs but the
  // avatar looks frozen on one pose. Releasing it here lets the segment animate.
  try { player._idleHold = false; } catch (e) {}
  player.play({ speed: playbackSpeed, loop: true, fromFrame, toFrame });
  // Apply the first frame immediately. This prevents a stale pose while the
  // browser waits for its next animation frame and makes step changes obvious.
  try { if (typeof player.poseFrame === 'function') player.poseFrame(fromFrame); } catch (e) {}
  showStepCaption(step.label || `Step ${step.index || ''}`, step.count || '');
  _notifyLessonParent('lesson.step.started', { index: step.index, speed: playbackSpeed });

  const narration = [step.cue, step.detail, step.mistake ? `Watch out: ${step.mistake}` : '']
    .filter(Boolean).join(' ');
  if (narration) {
    addMsg(narration, 'tool');
    if (voiceOn && voice && !_s2sOn) {
      try { await voice.speak(narration, {}); } catch (e) {}
    }
  }
}

window.addEventListener('message', (evt) => {
  if (!_lessonParentAllowed(evt)) return;
  const msg = evt.data || {};
  if (msg.source !== 'studioos.learn') return;
  if (msg.type === 'lesson.ping') {
    if (player && player.data) _notifyLessonParent('lesson.ready');
  } else if (msg.type === 'lesson.step') {
    _teachEmbeddedStep(msg.step, msg.speed);
  } else if (msg.type === 'lesson.pause') {
    try { player.pause(); } catch (e) {}
    _notifyLessonParent('lesson.paused');
  } else if (msg.type === 'lesson.resume') {
    try { player.resume(); } catch (e) {}
    _notifyLessonParent('lesson.resumed');
  } else if (msg.type === 'lesson.replay') {
    try { player.restart(); } catch (e) {}
    _notifyLessonParent('lesson.resumed');
  } else if (msg.type === 'lesson.mirror') {
    try { player.mirror = !!msg.enabled; } catch (e) {}
    _notifyLessonParent('lesson.mirrored', { enabled: !!msg.enabled });
  } else if (msg.type === 'lesson.feedback') {
    let clipId = '';
    try { clipId = new URLSearchParams(location.search).get('motion') || ''; } catch (e) {}
    openLiveFeedback(clipId);
  }
});

// ─── v34: LIVE FEEDBACK POPUP (MediaPipe Pose, in-page) ──────────────
// Modal over the stage: webcam tile, score gauge, "worst body part"
// cue. MediaPipe Pose from CDN; streams keypoints to /ws/feedback.
let _liveFB = null;
async function openLiveFeedback(clipId) {
  if (_liveFB) {
    try { _liveFB.ws?.send(JSON.stringify(
      { type: 'start', clip_id: clipId })); } catch {}
    return;
  }
  const popup = $('feedback-popup');
  const video = $('fb-video');
  const scoreEl = $('fb-score');
  const tipEl   = $('fb-tip');
  const stopBtn = $('fb-stop');
  if (!popup || !video || !scoreEl) {
    addMsg('live mirror UI missing', 'flag'); return;
  }
  popup.classList.add('show');
  scoreEl.textContent = '—';
  tipEl.textContent   = 'starting camera…';

  // v186: one-time consent for movement-data capture (research corpus).
  // We ask ONCE, honestly, and remember the choice. If they decline, the
  // live mirror still works fully — we just never log skill events.
  try {
    if (localStorage.getItem('coach.skill.consent') === null) {
      const ok = window.confirm(
        'Help improve the AI coach?\n\n' +
        'We can learn from your attempts (pose data only — never your ' +
        'video, which stays on your device) to make coaching better for ' +
        'everyone. You can change this anytime. Allow?');
      _setSkillConsent(!!ok);
    }
  } catch (e) {}

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360, facingMode: 'user' },
      audio: false,
    });
  } catch (e) {
    tipEl.textContent = 'camera blocked: ' + e.message;
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});

  tipEl.textContent = 'loading pose model…';
  if (!window.__poseReady) {
    await new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js';
      s.onload = ok; s.onerror = fail;
      document.head.appendChild(s);
    }).catch((e) => { tipEl.textContent = 'pose lib error'; throw e; });
    window.__poseReady = true;
  }
  const pose = new window.Pose({
    locateFile: (f) => 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/' + f,
  });
  pose.setOptions({
    modelComplexity: 0, smoothLandmarks: true,
    enableSegmentation: false, minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host + APP_BASE + '/ws/feedback');
  _liveFB = { ws, stream, stopped: false, lastSend: 0 };
  // v186: this live-feedback session IS a stream of learner ATTEMPTS at a
  // specific move, each with a measured score + which body part was worst,
  // plus the instruction the coach shows in response. Capturing that
  // (attempt, instruction, outcome) stream is the research corpus. We
  // throttle to ~1 logged beat / 2s (the raw score fires every ~500ms —
  // too noisy to store every one) and only log with the user's consent.
  _liveFB.clipId = clipId;
  _liveFB.sessionId = window.__sessionId || ('fb_' + Date.now().toString(36));
  _liveFB.attemptIndex = 0;
  _liveFB.lastLog = 0;
  ws.onopen = () => {
    tipEl.textContent = 'syncing reference…';
    try { ws.send(JSON.stringify({ type: 'start', clip_id: clipId })); } catch {}
  };
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === 'ready') {
      tipEl.textContent = 'go! mirror what you see.';
    } else if (m.type === 'unavailable') {
      tipEl.textContent = 'no live reference for this clip yet';
      scoreEl.textContent = 'n/a';
    } else if (m.type === 'score') {
      scoreEl.textContent = String(Math.round(m.score));
      const tip = m.score >= 75 ? 'looking great' :
                  m.score >= 50 ? 'getting there — watch ' + (m.worst || 'the timing') :
                                  'try the ' + (m.worst || 'pose') + ' again';
      tipEl.textContent = tip;
      // v186: log this coaching beat (attempt score + the instruction we
      // just gave) into the skill corpus, throttled + consent-gated.
      try {
        const now = performance.now();
        if (now - (_liveFB.lastLog || 0) >= 2000) {
          _liveFB.lastLog = now;
          _liveFB.attemptIndex++;
          _logSkillEvent({
            session_id: _liveFB.sessionId,
            clip_id: _liveFB.clipId,
            event_kind: 'attempt_score',
            attempt_index: _liveFB.attemptIndex,
            score: (typeof m.score === 'number') ? m.score : null,
            mean_error: (typeof m.mean_err === 'number') ? m.mean_err : null,
            worst_keypoint: m.worst || '',
            instruction: tip,
            instruction_source: 'live_feedback',
          });
        }
      } catch (e) { /* logging must never break the live loop */ }
    } else if (m.type === 'done') {
      tipEl.textContent = 'session avg ' + m.avg_score;
    }
  };
  ws.onerror = () => { tipEl.textContent = 'feedback ws error'; };

  // MediaPipe BlazePose 33-landmark → COCO-17 mapping.
  const BP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,
                      23, 24, 25, 26, 27, 28];
  pose.onResults((res) => {
    if (_liveFB?.stopped || !res.poseLandmarks) return;
    const lms = res.poseLandmarks;
    const kp = BP_TO_COCO.map((bi) => {
      const p = lms[bi];
      if (!p) return [0, 0, 0];
      return [p.x, p.y, p.visibility ?? 1.0];
    });
    let refFrame = 0;
    try { refFrame = (player && Math.floor(player.frame || 0)) || 0; } catch {}
    if (!refFrame) refFrame = Math.floor(performance.now() / (1000/30));
    const now = performance.now();
    if (now - _liveFB.lastSend < 80) return;
    _liveFB.lastSend = now;
    try {
      ws.send(JSON.stringify({ type: 'frame', ref_frame: refFrame, kp }));
    } catch {}
  });

  const tickFB = async () => {
    if (_liveFB?.stopped) return;
    try { await pose.send({ image: video }); } catch {}
    setTimeout(tickFB, 90);
  };
  tickFB();

  if (stopBtn) stopBtn.onclick = () => closeLiveFeedback();
}

function closeLiveFeedback() {
  if (!_liveFB) return;
  _liveFB.stopped = true;
  try { _liveFB.ws?.send(JSON.stringify({ type: 'stop' })); } catch {}
  try { _liveFB.ws?.close(); } catch {}
  try { _liveFB.stream?.getTracks().forEach((t) => t.stop()); } catch {}
  _liveFB = null;
  const popup = $('feedback-popup');
  if (popup) popup.classList.remove('show');
}

async function runDrill(opts) {
  const token = ++_drillToken;             // cancels any prior drill
  const {
    clip_id, counts = 8, repeats = 4,
    speed_start = 0.5, speed_end = 1.0,
    start_count = null, end_count = null,
    mirror_alternate = false, cues = [],
  } = opts;
  if (!clip_id) return;
  // Fetch clip to compute frame window.
  const r = await fetch(APP_BASE + '/api/motion/data/' + clip_id + '.json');
  if (!r.ok) { addMsg('drill fetch failed', 'flag'); return; }
  const data = await r.json();
  if (token !== _drillToken) return;
  player.load(data);
  const fps = data.fps || 30;
  const nFrames = data.n_frames || data.frames || 0;
  // We don't know the music BPM precisely, but counts→frames is well
  // approximated by spreading `counts` over `counts/8` of the clip when
  // no explicit window is given.
  let fromF = 0, toF = nFrames;
  if (start_count && end_count && counts > 0) {
    const totalCounts = Math.max(counts, end_count);
    fromF = Math.floor(nFrames * (start_count - 1) / totalCounts);
    toF   = Math.floor(nFrames *  end_count        / totalCounts);
  } else {
    // Use first `counts` worth of the clip (assume clip ≈ 8-count).
    toF = Math.floor(nFrames * Math.min(1, counts / 8));
  }
  toF = Math.max(fromF + 4, toF);
  const cueList = cues.slice();
  for (let i = 0; i < repeats; i++) {
    if (token !== _drillToken) return;
    const t = repeats > 1 ? i / (repeats - 1) : 1;
    const speed = speed_start + (speed_end - speed_start) * t;
    const mirror = mirror_alternate ? (i % 2 === 1) : false;
    player.play({
      speed, mirror, loop: false,
      fromFrame: fromF, toFrame: toF,
    });
    if (life) life.setMood(i === repeats - 1 ? 'happy' : 'focused');
    // Wait for clip-window to finish (with safety timeout).
    const dur = (toF - fromF) / fps / Math.max(speed, 0.1);
    await new Promise((ok) => {
      let done = false;
      player.onend = () => { if (!done) { done = true; ok(); } };
      setTimeout(() => { if (!done) { done = true; ok(); } },
                 (dur + 0.4) * 1000);
    });
    player.onend = null;
    if (token !== _drillToken) return;
    // Speak a cue between reps (skip on last).
    if (i < repeats - 1 && cueList.length && voiceOn && voice) {
      const cue = cueList[i % cueList.length];
      if (cue) try { await voice.speak(cue, {}); } catch (e) {}
    }
  }
}

// ─── 8-count metronome ────────────────────────────────────────────────
// Lights up the current beat-count on top of the stage so students can
// learn the move on counts. Strict BPM-derived — never random.
//   beat   = floor(elapsed * bpm / 60)
//   count  = (beat % 8) + 1
// We hide the strip when not dancing or when the clip has no bpm.
// IMPORTANT: declared BEFORE `tick()` is invoked so the first frame's
// updateCounts() call doesn't hit a temporal-dead-zone ReferenceError.
const _countsEl = document.getElementById('counts');
const _countSpans = _countsEl ? [..._countsEl.querySelectorAll('span')] : [];
let _lastCountShown = -1;

// ─── render loop ──────────────────────────────────────────────────────
const clock = new THREE.Clock();

// v141: AUTO-FRAME GUARD. Some clips still carry residual root motion (or a
// bad retarget) that drifts the avatar sideways / floats her upward until
// she leaves the fixed frame. Instead of trusting every clip to stay put,
// the camera now gently FOLLOWS her hips horizontally + vertically, and
// temporarily ZOOMS OUT (by raising the orbit min-distance floor) whenever
// her head or feet approach the edge of the frame. Fully additive: it only
// nudges the OrbitControls target + min-distance, so the user can still
// orbit/zoom, and it relaxes back to the default framing once she re-centres.
const _AF_BASE_MIN = 3.0;
const _afHips = new THREE.Vector3();
const _afHead = new THREE.Vector3();
const _afFoot = new THREE.Vector3();
const _afTmp = new THREE.Vector3();
// v155: the camera used to FOLLOW the hips EVERY frame, so it drifted
// constantly and felt jittery even when she was perfectly in frame (user:
// "the frame is moving too much... keep this only when she goes out of
// frame"). Now it holds a fixed resting target and ONLY reacts when a body
// point approaches the frame edge — a dead-zone. Inside the comfort band it
// gently settles back to the resting home and default zoom, so the shot is
// steady.
let _afHome = null;            // resting camera target, captured once at boot
const _AF_EDGE = 0.80;         // |NDC| past this ⇒ she's near the frame edge → react
const _AF_RELAX = 0.55;        // |NDC| inside this ⇒ safe → settle back home
// v155: floor-camera state. Plank/push-up/sit-up read as a flat blob from
// the standing camera (which looks DOWN from ~torso height). In floor mode
// we drop the camera to a low 3/4 angle aimed at the body on the ground,
// then restore the standing framing when the clip ends.
let _floorCamActive = false;
let _floorCamSavedY = null;
function _autoFrame(dt) {
  if (!currentVrm || !currentVrm.humanoid) return;
  const H = currentVrm.humanoid;
  const hips = H.getNormalizedBoneNode('hips');
  if (!hips) return;
  hips.getWorldPosition(_afHips);
  if (_afHome === null) _afHome = controls.target.clone();
  const k = 1 - Math.pow(0.02, dt);
  // ── FLOOR-MODE FRAMING ──────────────────────────────────────────────
  const floorMode = (() => {
    try { return !!(player && player.data && player.data.pose_profile === 'floor'); }
    catch (e) { return false; }
  })();
  if (floorMode) {
    if (!_floorCamActive) { _floorCamSavedY = camera.position.y; _floorCamActive = true; }
    const kf = 1 - Math.pow(0.05, dt);
    // Aim at the body lying on the ground.
    controls.target.x += (_afHips.x - controls.target.x) * kf;
    controls.target.z += (_afHips.z - controls.target.z) * kf;
    controls.target.y += (0.30 - controls.target.y) * kf;
    // Drop the camera low for a near-eye-level 3/4 angle + pull a touch
    // closer so the horizontal body fills the frame.
    camera.position.y += (0.85 - camera.position.y) * kf;
    controls.minDistance += (2.6 - controls.minDistance) * kf;
    return;
  }
  if (_floorCamActive) {
    // Exiting floor mode — restore the standing camera height + zoom, then
    // hand control back to the normal dead-zone framing.
    const kf = 1 - Math.pow(0.05, dt);
    if (_floorCamSavedY != null) {
      camera.position.y += (_floorCamSavedY - camera.position.y) * kf;
    }
    controls.target.y += (_afHome.y - controls.target.y) * kf;
    controls.minDistance += (_AF_BASE_MIN - controls.minDistance) * kf;
    if (_floorCamSavedY == null || Math.abs(camera.position.y - _floorCamSavedY) < 0.03) {
      _floorCamActive = false;
    }
    return;
  }
  // Worst-case frame excursion: project head / feet / hips to NDC and take
  // the largest |x|/|y|. Head+hips check both axes; feet check vertical only.
  let maxNdc = 0;
  const measure = (boneName, useX) => {
    const b = H.getNormalizedBoneNode(boneName);
    if (!b) return;
    b.getWorldPosition(_afTmp); _afTmp.project(camera);
    maxNdc = Math.max(maxNdc, Math.abs(_afTmp.y));
    if (useX) maxNdc = Math.max(maxNdc, Math.abs(_afTmp.x));
  };
  measure('head', true);
  measure('leftFoot', false);
  measure('rightFoot', false);
  measure('hips', true);
  if (maxNdc > _AF_EDGE) {
    // Near/over the edge — follow her hips and zoom out just enough to
    // recapture her. This is the ONLY time the camera actively moves.
    controls.target.x += (_afHips.x - controls.target.x) * k;
    controls.target.z += ((_afHips.z - 0.4) - controls.target.z) * k;
    const ty = _afHips.y + 0.15;
    controls.target.y += (ty - controls.target.y) * (k * 0.6);
    const over = maxNdc - _AF_EDGE;
    const curDist = camera.position.distanceTo(controls.target);
    const floor = Math.min(controls.maxDistance, curDist + over * 3.0);
    if (floor > controls.minDistance) controls.minDistance = floor;
  } else if (maxNdc < _AF_RELAX) {
    // Comfortably in frame — gently drift the target BACK to its resting
    // home + default zoom so the shot stays still instead of chasing her.
    controls.target.x += (_afHome.x - controls.target.x) * (k * 0.3);
    controls.target.y += (_afHome.y - controls.target.y) * (k * 0.3);
    controls.target.z += (_afHome.z - controls.target.z) * (k * 0.3);
    controls.minDistance += (_AF_BASE_MIN - controls.minDistance) * (k * 0.5);
  }
  // Between RELAX and EDGE = dead zone: hold everything steady (no move).
}

function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  _autoFrame(dt);                    // keep her in frame before the orbit solve
  controls.update();
  if (player) player.update(dt);     // dance writes skeleton first
  // When a clip has ended or been stopped, player.update() early-returns
  // and the whole body stays frozen on the last frame — legs included —
  // so idle sway (which only touches hips/spine/head) leaves her tilted
  // with feet floating. Restore the grounded standing pose each idle
  // frame; AvatarLife then composes breath/sway on top. NOT done while
  // PAUSED (pause holds the current pose so the user can study a move).
  if (player && player._idleHold) { try { player.applyRestPose(); } catch {} }
  if (life)   life.update(dt);       // then face/eyes/breath/visemes
  // After AvatarLife sets the idle hip position, drop/lift the body so
  // the feet rest on the floor (foot-lock only runs during playback).
  if (player && player._idleHold) { try { player._groundForIdle(); } catch {} }
  if (currentVrm) currentVrm.update(dt);
  _faceCameraLock(dt);               // v75: keep her facing the user
  updateCounts();                    // 8-count metronome over the stage
  // v72: speaking-camera zoom REMOVED — it sometimes dollied in far
  // enough to push her out of frame. Keep the camera locked on the
  // full-body framing at all times (the user can still orbit).
  renderer.render(scene, camera);
}

// v72: SPEAKING CAMERA REMOVED. The talk-time dolly occasionally moved
// in far enough (or combined with a user orbit) to lose her from the
// frame, so it's gone. The camera now stays on the fixed full-body
// framing set up at boot; lip-sync + mouth-gain still convey speech.

// v75: FACE-CAMERA YAW LOCK. CMU/AIST mocap clips bake an arbitrary
// global heading into the hips, so the avatar frequently ends up with
// her BACK or SIDE to the camera. For a talking companion that's wrong
// — she should look at the user. Each frame we measure her face azimuth
// (hips local -Z) and ease the rig's yaw so her face points at the
// camera, with a small flattering 3/4 angle. Runs in BOTH playback and
// idle. Smoothed so in-clip turns still read briefly, then settle back.
let _faceTmpQ = null, _faceTmpV = null;
const FACE_TARGET_AZ = -0.16;        // rad (~ -9 deg): gentle 3/4 turn
let _invertT = 0;                    // s the avatar has been inverted
let _tiltT = 0;                      // v112: s the body has been lying/tilted
let _invertFallbackActive = false;   // v108: cooldown after a safe-clip swap
function _faceCameraLock(dt) {
  if (!currentVrm || !currentVrm.humanoid || !currentVrm.scene) return;
  const hum = currentVrm.humanoid;
  const hips = hum.getNormalizedBoneNode('hips');
  if (!hips) return;
  if (!_faceTmpQ) { _faceTmpQ = new THREE.Quaternion(); _faceTmpV = new THREE.Vector3(); }
  // ── v75 INVERSION GUARD ──────────────────────────────────────────
  // Some mocap clips (esp. breaking / aerial kicks) fold or invert the
  // avatar so her head drops below her hips — the "upside-down inside
  // the floor" bug. For a talking companion that's never acceptable.
  // If a playing clip keeps her torso inverted past ~0.25 s, abort it
  // and fall back to the safe grounded idle. Never fires on upright
  // dance (head always well above hips), so normal moves are untouched.
  try {
    if (player && player.playing) {
      const head = hum.getNormalizedBoneNode('head');
      if (head) {
        currentVrm.scene.updateMatrixWorld(true);
        const hp = _faceTmpV, hd = new THREE.Vector3();
        hips.getWorldPosition(hp); head.getWorldPosition(hd);
        // v112: measure the BODY AXIS (hips -> head), not just whether the
        // head is below the hips. A clip that lays her HORIZONTAL has the
        // head and hips at the SAME height (head NOT below hips), so the
        // old check missed it entirely — that's the "she's lying flat"
        // bug the user kept seeing. cosTilt = verticalness of the spine:
        //   1.0 = standing straight up, 0 = lying flat, < 0 = upside-down.
        const vx = hd.x - hp.x, vy = hd.y - hp.y, vz = hd.z - hp.z;
        const L = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1e-6;
        const cosTilt = vy / L;
        // v152: some warm-up/mobility clips are DELIBERATE forward folds
        // (toe-touch, standing hamstring reach). Those bend the spine
        // toward horizontal on purpose, which the upright-companion guard
        // otherwise treats as "lying flat" and swaps away — so the coach
        // kept replacing a legit stretch with the idle clip. When the
        // loaded clip is flagged pose_profile:'fold' we relax the mild
        // "tilted" swap (a forward fold is expected) while KEEPING the
        // hard protections: true inversion (head well below hips /
        // upside-down) and sinking through the floor still swap out.
        // v154: pose_profile:'floor' (plank / push-up / sit-up) is
        // DELIBERATELY horizontal — cosTilt ~0 — so it must be treated
        // the same relaxed way, otherwise the guard swaps every floor
        // exercise to the idle groove (the "why can't planks show" bug).
        const foldOk = (() => {
          try {
            const pp = player && player.data && player.data.pose_profile;
            return pp === 'fold' || pp === 'floor';
          } catch (e) { return false; }
        })();
        // Hard inversion (head at/below hips) OR sunk through the floor:
        // fire FAST so the user barely glimpses it. For fold clips a deep
        // bend can dip cosTilt slightly negative at the bottom, so only a
        // clear upside-down (< -0.35) counts as inversion there.
        const inverted = (cosTilt < (foldOk ? -0.35 : 0.0)) || (hp.y < -0.4);
        // Lying / heavily tilted (> ~65° off vertical): also unacceptable
        // for an upright companion. Allow a brief lean (debounce 0.16 s)
        // so a normal dance bend doesn't trip it, but a sustained
        // horizontal pose swaps to the safe clip. Fold clips skip this
        // mild check entirely (the fold IS the move).
        const tilted = !foldOk && cosTilt < 0.42;
        _invertT = inverted ? (_invertT + dt) : 0;
        _tiltT = tilted ? (_tiltT + dt) : 0;
        // v108/v112: FALL BACK to a guaranteed-upright clip (not just
        // freeze). This watches the ACTUAL rendered avatar (after
        // vrm.update), so it catches ANY bad clip no matter how it was
        // picked or whether the server-side verifier missed it.
        if ((_invertT > 0.06 || _tiltT > 0.16) && !_invertFallbackActive) {
          _invertT = 0;
          _tiltT = 0;
          _invertFallbackActive = true;
          try { _pendingMotion = null; if (_pendingMotionTimer) { clearTimeout(_pendingMotionTimer); _pendingMotionTimer = null; } } catch (e) {}
          try { player.stop(); } catch (e) {}
          try { player._idleHold = true; player.applyRestPose(); } catch (e) {}
          try { stopMusic(); } catch (e) {}
          // Swap to the safe fallback clip so she keeps dancing cleanly
          // rather than freezing (unless we were already on it). A short
          // cooldown prevents thrashing if the fallback itself ever dips.
          try {
            const onSafe = window.__lastPlay && window.__lastPlay.clipId === IDLE_CLIP_ID;
            if (!onSafe) {
              console.warn('[inversion] swapping to safe clip', IDLE_CLIP_ID);
              loadAndPlayClip(IDLE_CLIP_ID, { loop: true }).catch(() => {});
            }
          } catch (e) {}
          setTimeout(() => { _invertFallbackActive = false; }, 1500);
        }
      }
    } else { _invertT = 0; }
  } catch (e) { _invertT = 0; }
  hips.getWorldQuaternion(_faceTmpQ);
  // VRM face points along the bone's local -Z (verified empirically).
  _faceTmpV.set(0, 0, -1).applyQuaternion(_faceTmpQ);
  const faceAz = Math.atan2(_faceTmpV.x, _faceTmpV.z);
  let delta = FACE_TARGET_AZ - faceAz;
  while (delta >  Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  const k = Math.min(1, dt * 6);     // ease ~6/s, settle in ~0.3 s
  currentVrm.scene.rotation.y += delta * k;
}

function updateCounts() {
  // v80: the 8-count dance-class metronome is removed (this is a
  // movement companion, not a choreography drill). Keep the strip
  // permanently hidden.
  if (_countsEl && _countsEl.classList.contains('show')) {
    _countsEl.classList.remove('show');
  }
  return;
  // eslint-disable-next-line no-unreachable
  if (!_countsEl) return;
  if (!player || !player.playing || !player.data) {
    if (_countsEl.classList.contains('show')) {
      _countsEl.classList.remove('show');
      _countSpans.forEach(s => s.classList.remove('on'));
      _lastCountShown = -1;
    }
    return;
  }
  // Pull BPM from clip metadata. AIST clips ship with bpm in data.meta;
  // fall back to 120 if absent so the metronome still helps.
  const bpm = (player.data.meta?.bpm) || player.data.bpm || 120;
  const fps = player.data.fps || 30;
  const beatLen = (60 / bpm) * fps;             // frames per beat
  const beat = Math.floor(player.frame / beatLen);
  const count = (beat % 8) + 1;
  if (count !== _lastCountShown) {
    _lastCountShown = count;
    _countSpans.forEach((s, i) => s.classList.toggle('on', i + 1 === count));
  }
  if (!_countsEl.classList.contains('show')) _countsEl.classList.add('show');
}

// Kick off the render loop now that `_countsEl` / `_countSpans` exist.
tick();

// ─── bootstrap: limits + characters + healthz ─────────────────────────
async function bootstrap() {
  // v224: ENTRY beacon. `dance_visited` (top of the coach funnel) is injected
  // by studioos.fit's NGINX sub_filter on /dance loads — but dancecoach.fit has
  // NO nginx, so a coach open there NEVER fired dance_visited. Result: our funnel
  // was blind to dancecoach.fit's top-of-funnel (only Clarity saw those sessions).
  // Fire it here, ONCE, on boot for nginx-less hosts (dancecoach.fit / localhost),
  // BEFORE any await so it registers even if the health/VRM load later fails.
  // Skipped on *.studioos.fit where nginx already injects it (avoids double count).
  try {
    const _host = (location.hostname || '').toLowerCase();
    const _nginxless = _host.indexOf('studioos.fit') === -1;
    if (_nginxless && !window.__danceVisitedFired) {
      window.__danceVisitedFired = true;
      _coachTrack('dance_visited', { from: 'coach_boot', host: _host });
    }
  } catch (e) {}
  // v178: reuse the limits/characters fetches kicked off in coach.html's
  // inline health-gated loader (they started BEFORE this module even
  // loaded, in parallel with the healthz poll) instead of issuing brand
  // new requests here — shaves a full serialized round-trip off the
  // critical path to loadVrm().
  const pre = window.__preFetch || {};
  const [h, lim, chars] = await Promise.all([
    fetch(APP_BASE + '/healthz').then(r => r.json()),
    pre.limits || fetch(APP_BASE + '/api/limits').then(r => r.json()),
    pre.characters || fetch(APP_BASE + '/api/characters').then(r => r.json()),
  ]);
  limits = lim;
  // v156: REAL boot-loader milestone. Previously window.__boot.done() was
  // NEVER called anywhere — the "Warming up your studio" overlay just faked
  // its way to ~92% and sat there until a hardcoded 45s safety timeout force-
  // closed it, no matter how fast (or slow) the app actually was underneath.
  // Now we report real progress at each real milestone (see loadVrm() for the
  // done() call once the avatar is actually on screen).
  try { window.__boot?.set(35, 'Assembling the studio\u2026'); } catch (e) {}
  // Health details kept in console for debugging only — not user-visible.
  console.debug('[coach] health', h);
  const sel = $('char');
  const allowed = new Set(['nova', 'kira', 'aki', 'celeste', 'gio', 'pax', 'zane']);
  // registry uses different ids; show display_name, value = registry name
  let firstName = null;
  for (const c of (chars.characters || [])) {
    if (allowed.size && !allowed.has(c.profile?.display_name?.toLowerCase()))
      continue;
    const o = document.createElement('option');
    o.value = c.name;
    o.dataset.voice = c.profile?.voice || 'en-US-AriaNeural';
    o.textContent  = (c.profile?.display_name || c.name) +
                     '  ·  ' + (c.profile?.style || '');
    sel.appendChild(o);
    if (!firstName) firstName = c.name;
  }
  if (!sel.options.length) {           // allow-list missed; show everything
    for (const c of (chars.characters || [])) {
      const o = document.createElement('option');
      o.value = c.name;
      o.dataset.voice = c.profile?.voice || 'en-US-AriaNeural';
      o.textContent = c.profile?.display_name || c.name;
      sel.appendChild(o);
      if (!firstName) firstName = c.name;
    }
  }
  sel.value = firstName;
  // Prefer Nova (mymodel2) — the user-provided custom VRM (recolored) — as
  // the default boot character. Fall back to the first registry entry.
  const preferDefault = Array.from(sel.options).find(o =>
    /mymodel2\b/i.test(o.value)) || Array.from(sel.options).find(o =>
    /nova/i.test(o.textContent) || /mymodel1\b/i.test(o.value));
  if (preferDefault) sel.value = preferDefault.value;
  voice.setVoice(sel.selectedOptions[0]?.dataset.voice);
  sel.addEventListener('change', async () => {
    voice.setVoice(sel.selectedOptions[0]?.dataset.voice);
    // Snapshot what's currently dancing so we can resume after swap.
    // IMPORTANT: only resume if the player was actively playing. If
    // motion had already stopped/flagged we'd reload a stale frame
    // and the new character ends up frozen in a half-pose.
    const lp = window.__lastPlay;
    const wasPlaying = !!(player && player.playing && !player.flagged);
    const resumeFrame = wasPlaying ? ((player.frame) | 0) : 0;
    await loadVrm(sel.value);
    if (lp && wasPlaying && resumeFrame > 0) {
      // Re-fetch + play from the same frame so the new character
      // picks up the choreography where the old one left off.
      try {
        const data = await fetchMotion(lp.clipId);
        player.load(data);
        player.play({ ...(lp.opts || {}), fromFrame: resumeFrame });
        window.__lastPlay = lp;
      } catch (e) { /* fall back to idle stance */ }
    }
  });
  try { window.__boot?.set(55, 'Loading your avatar\u2026'); } catch (e) {}
  await loadVrm(sel.value);
  // Stash chosen character's profile so the WS-open handler can fire
  // an in-character greeting (intro line + signature dance) as soon
  // as the socket connects.
  const chosen = (chars.characters || []).find(c => c.name === sel.value);
  window.__charProfile = chosen ? chosen.profile : null;
  window.__charSlug    = chosen ? chosen.name    : null;

  // Deep-link: /dance/?motion=<clip_id> (e.g. from a READY "learn from video"
  // job in the app). Auto-load + play that choreography once the avatar is up.
  try {
    const qp = new URLSearchParams(location.search);
    const motionId = qp.get('motion');
    if (motionId && player) {
      const data = await fetchMotion(motionId);
      player.load(data);
      window.__lastPlay = { clipId: motionId, opts: { speed: 1.0, loop: true } };
      if (_embeddedLesson) {
        // The parent waits for an explicit Start lesson click. Loading the
        // motion is enough; do not loop the whole choreography on arrival.
        try { player.applyRestPose(); } catch (e) {}
        _notifyLessonParent('lesson.ready', { motion: motionId });
        if (_pendingLessonCommand) {
          const pending = _pendingLessonCommand;
          _pendingLessonCommand = null;
          _teachEmbeddedStep(pending.step, pending.speed);
        }
      } else {
        player.play({ speed: 1.0, loop: true });
      }
      // If this is a LEARNED choreography being resumed (the app passes
      // `job`/`step`), auto-kick the step-by-step breakdown so the coach
      // starts teaching the segmented steps immediately instead of just
      // looping the clip. The agent's break_down tool now teaches the
      // stored learned_steps (with the free/paid gate baked in).
      const stepParam = parseInt(qp.get('step') || '0', 10);
      if (!_embeddedLesson && (qp.get('job') || stepParam > 0) && !window.__learnResumeKicked) {
        window.__learnResumeKicked = true;
        const ask = stepParam > 1
          ? `Let's continue this choreography — break it down step by step from step ${stepParam}.`
          : `Teach me this choreography step by step, from the top.`;
        try { sendUser(ask, 'deeplink'); } catch (e) {}
      }
    }
  } catch (e) { console.warn('[coach] motion deep-link failed', e); }
}

// ─── coach greeting (proactive intro on first connection) ────────────
// When the WS opens we don't wait for the student to type; the avatar
// greets them in-character with their intro line + a short slice of
// their signature dance. This is the "real coach walks in the room"
// moment — way more inviting than a blank canvas.
let _greetedThisSession = false;

// A small bank of natural greeting variations so the user doesn't
// hear the exact same canned line on every page load. We rotate
// through {name, style} templates and pick one at random.
//
// v31: split into FIRST-MEET vs RETURNING variants. A real coach
// introduces themselves the first time they meet you ("Hi I'm Kira,
// great to see you") but doesn't keep re-introducing themselves
// every session ("Hey, ready when you are"). Local-storage flag
// `coach.met` tracks whether this browser has met this coach before.
function pickGreetingLine(displayName, style, charSlug) {
  const n  = (displayName || 'your coach');
  const st = (style || '').toLowerCase();
  const stShort = st || 'this style';
  // Has this browser met this coach before?
  let metBefore = false;
  const metKey = 'coach.met.' + (charSlug || 'default');
  try { metBefore = !!localStorage.getItem(metKey); } catch (e) {}

  // v86: greeting now LEADS the conversation (asks how they feel /
  // what they want) AND respects the chosen language so a Hinglish user
  // is greeted in Hinglish, not English.
  const lang = (window.__coachLang || 'hinglish');
  // v192: keep the greeting SHORT and warm — a quick "hi, how are you?"
  // and nothing more. The old greeting dumped the whole style catalogue
  // in one breath, which felt like being talked at. The front door
  // already shows the choices, so the coach just needs to say hello.
  const BANKS = {
    hinglish: {
      first: [
        `Hi, main ${n}. Kaise ho?`,
        `Arre hi — ${n} here. Kaisa chal raha hai?`,
        `Hello! Main ${n} hoon. Kaise ho aaj?`,
      ],
      returning: [
        `Wapas aa gaye — kaise ho?`,
        `Hey, phir mile! Kaisa chal raha hai?`,
        `Welcome back. Kaise ho?`,
      ],
    },
    hindi: {
      first: [
        `हाय, मैं ${n}। कैसे हो?`,
        `नमस्ते — मैं ${n}। कैसे हो आज?`,
      ],
      returning: [
        `वापस आ गए — कैसे हो?`,
        `हेय, फिर मिले! कैसे हो?`,
      ],
    },
    english: {
      first: [
        `Hi, I'm ${n}. How are you?`,
        `Hey — ${n} here. How's it going?`,
        `Hello! I'm ${n}. How are you today?`,
      ],
      returning: [
        `Hey, you're back. How are you?`,
        `Good to see you again. How's it going?`,
        `Welcome back. How are you?`,
      ],
    },
  };
  const _bank = BANKS[lang] || BANKS.english;
  const pool = metBefore ? _bank.returning : _bank.first;
  let idx = Math.floor(Math.random() * pool.length);
  try {
    const lastKey = 'coach.greet.idx.' + (metBefore ? 'r' : 'f');
    const last = parseInt(sessionStorage.getItem(lastKey) || '-1', 10);
    if (last === idx && pool.length > 1) idx = (idx + 1) % pool.length;
    sessionStorage.setItem(lastKey, String(idx));
  } catch (e) { /* private mode */ }
  // Stamp the "met" flag AFTER we picked, so a brand-new visitor
  // gets a real intro on this page-load, and subsequent reloads
  // get the casual returning-coach lines.
  try { localStorage.setItem(metKey, '1'); } catch (e) {}
  return pool[idx];
}

let _greetRetries = 0;
async function playCoachGreeting() {
  if (_greetedThisSession) return;
  // v190: don't "talk at" cold users. While the button-first start screen
  // (front door) is up, the coach stays quiet — the greeting only fires
  // once the user dismisses it (picks a session or taps "chat with coach").
  if (window.__frontDoorOpen) return;
  const prof = window.__charProfile;
  if (!prof) {
    // v87: char profile not loaded yet — retry instead of permanently
    // skipping (the old code set the greeted flag before this check, so
    // a slow profile load meant the coach NEVER greeted / led the chat).
    if (_greetRetries++ < 40) {
      setTimeout(() => playCoachGreeting().catch(() => {}), 300);
    }
    return;
  }
  _greetedThisSession = true;
  // Skip the spoken greeting on quick revisits (refresh within
  // ~10min) so it doesn't feel like a chatbot starting over every
  // time the user F5s. The chat message bubble still shows so the
  // session has a clear opening line.
  let skipVoice = false;
  try {
    const ts = parseInt(sessionStorage.getItem('coach.greet.ts') || '0', 10);
    if (ts && Date.now() - ts < 10 * 60 * 1000) skipVoice = true;
    sessionStorage.setItem('coach.greet.ts', String(Date.now()));
  } catch (e) { /* private mode */ }
  // v90: when live voice is the default voice, IT speaks the greeting —
  // don't also speak via Azure TTS (would double-greet). Bubble + wave
  // clip still show immediately.
  if (window.__s2sDefault) skipVoice = true;
  const line = pickGreetingLine(prof.display_name, prof.style, window.__charSlug);
  // Show the bubble + push into chat history immediately — the user
  // can read the greeting even if their browser is still blocking TTS.
  addMsg(line, 'coach');
  if (!skipVoice) showBubble('🔊 Volume up — your coach talks!', 'hint');
  // v91b: greet with the PROCEDURAL wave only. The CMU "Wave Hello" clips
  // (cmu_105_105_53 / _15) are BOTH in the inverted-render blacklist — they
  // flip the avatar upside-down. The procedural wave is hand-authored arm
  // motion (shoulder lift + elbow bend + hand wave) and physically CANNOT
  // invert, so it's the only safe greeting. setMood adds a smile.
  if (life) {
    try { if (life.setMood) life.setMood('happy', 0.6); } catch (e) {}
    try { life.playWave({ side: 'right', duration: 2.8 }); } catch (e) {}
  }
  if (!voiceOn || skipVoice) return;

  // Browser autoplay policies block TTS until a user gesture. If the
  // user hasn't tapped yet, calling speak() will reject silently and
  // the user never hears "Hi". Wait for the unlock instead.
  let _greetingSpoken = false;
  const speakNow = async () => {
    if (_greetingSpoken) return;     // guard against multiple unlock events
    // v90: if S2S became the default after this greeting was set up,
    // let live voice do the talking — don't double-greet via Azure TTS.
    if (window.__s2sDefault) { _greetingSpoken = true; return; }
    _greetingSpoken = true;
    try {
      if (life) life.stopVisemes();
      if (life) life.setSpeaking(true);
      console.info('[voice] greeting speak start:', line.slice(0, 60));
      await voice.speak(line, {
        onviseme: (v) => { if (life) life.pushViseme(v); },
      });
      if (life) life.stopVisemes();
      if (life) life.setSpeaking(false);
      console.info('[voice] greeting speak done');
    } catch (e) {
      if (life) life.setSpeaking(false);
      console.warn('[voice] greeting failed:', e?.message || e);
    }
    // Greeting is intentionally NOT followed by an auto-dance. The
    // coach asks "Wanna dance?" and waits for the user's reply (chat
    // or one of the quick-action buttons). Auto-playing here used
    // to make the avatar dance unprompted before the user said a
    // single word.
  };

  if (_audioUnlocked) {
    speakNow();
  } else {
    console.info('[voice] greeting deferred — waiting for user gesture');
    // NOTE: each ['pointerdown','touchstart','keydown'] listener has
    // its own `once:true`, so without a shared flag the first user
    // tap fires pointerdown AND a follow-up keystroke fires keydown
    // and the greeting plays twice. The _greetingSpoken guard inside
    // speakNow above stops the second invocation.
    const onUnlock = () => { speakNow(); };
    ['pointerdown', 'touchstart', 'keydown'].forEach(ev =>
      window.addEventListener(ev, onUnlock, { once: true, passive: true }));
  }
}

// ─── agent WS ─────────────────────────────────────────────────────────
let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  // If studio-Os dropped a JWT in localStorage (same pattern the
  // existing /dance viewer uses) attach it so the server can identify
  // the user. Anonymous sessions just omit the token.
  const params = new URLSearchParams();
  try {
    const tok = localStorage.getItem('token');
    if (tok) params.set('token', tok);
  } catch (e) { /* private mode */ }
  // v139: carry the chosen language ON THE URL so EVERY (re)connect keeps it.
  // The server builds a fresh CoachState per socket (default 'hinglish') and
  // reads ?language=. Without this, a reconnect (tab background / mobile
  // network blip) silently reverted the coach to Hinglish while the browser's
  // voice stayed on the chosen locale -> "Hinglish text, English accent".
  try {
    const cl = window.__coachLang || localStorage.getItem('coach_lang') || 'hinglish';
    if (cl) params.set('language', cl);
  } catch (e) {}
  const qs = params.toString() ? ('?' + params.toString()) : '';
  ws = new WebSocket(proto + '//' + location.host + APP_BASE + '/ws/agent' + qs);
  ws.onopen = () => {
    setStatus('connected');
    // Flush anything the user typed while the WS was reconnecting.
    try {
      const q = window.__wsQueue || [];
      window.__wsQueue = [];
      for (const msg of q) ws.send(JSON.stringify(msg));
    } catch (e) {}
    // Tell the server which character was picked so the LLM stays
    // in-character + defaults to that style. Anonymous registries
    // (no profile) just skip this.
    const prof = window.__charProfile;
    if (prof) {
      try {
        ws.send(JSON.stringify({
          type: 'set_character',
          name: window.__charSlug || null,
          display_name: prof.display_name || null,
          style: prof.style || prof.tagline || null,
        }));
      } catch (e) {}
    }
    // v139: ALWAYS (re)assert the chosen language on every open, even if the
    // URL param already set it — belt-and-suspenders so a reconnect can never
    // leave the coach on the server-side 'hinglish' default.
    try {
      const cl = window.__coachLang || localStorage.getItem('coach_lang') || 'hinglish';
      ws.send(JSON.stringify({ type: 'set_language', language: cl }));
    } catch (e) {}
    // Fire the in-character greeting once the socket is up. Awaits
    // VRM + bootstrap because bootstrap() calls connect() after both.
    setTimeout(() => playCoachGreeting().catch(() => {}), 250);
  };
  ws.onclose = () => { setStatus('disconnected — retrying…');
                       setTimeout(connect, 1200); };
  ws.onerror = () => setStatus('ws error');
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    handleEvent(m);
  };
}

// When the tab is backgrounded, mobile browsers often suspend the
// WebSocket without firing onclose — the user comes back to a dead
// connection. On visibility change, check the socket and reconnect
// if it's not OPEN.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  try {
    // Stale CONNECTING sockets count as dead too — some mobile
    // browsers leave the WS in CONNECTING forever after a long
    // background period.
    const dead = !ws ||
      ws.readyState === WebSocket.CLOSED ||
      ws.readyState === WebSocket.CLOSING ||
      ws.readyState === WebSocket.CONNECTING;
    if (dead) {
      console.info('[ws] reconnecting after tab return');
      try { ws && ws.close(); } catch (e) {}
      // Do NOT replay the greeting on reconnect — it interrupts
      // whatever the user came back to do.
      connect();
    }
  } catch (e) { /* ignore */ }
  // Also re-check sign-in state in case the user signed in on
  // another tab while away.
  try {
    const btn = $('signin');
    if (btn) btn.textContent =
      localStorage.getItem('token') ? 'Account' : 'Sign in';
  } catch (e) {}
  // v185: this is also the moment an Android app resumes from the
  // background — the natural point to notice an expired token instead
  // of silently running anonymous for the rest of the session.
  try { _verifySessionOrPrompt(); } catch (e) {}
});

async function handleEvent(m) {
  if (m.type === 'hello') {
    // Server-config warnings are for the operator, not the end-user.
    if (!m.ai_ready) console.warn('[coach] AI not configured on server');
    return;
  }
  // ─── guided-session routing ────────────────────────────────────────
  // The server drives a phase machine (warmup → drill → combo → cool
  // down) and emits session.* events. Forward each to the analytics bus
  // (start-hero + summary subscribe there) and drive the on-screen HUD.
  // Without this, clicking a session length started the session on the
  // server but nothing appeared on screen.
  try { window.DanceAnalytics?.onEvent?.(m); } catch (e) {}
  if (m.type === 'auth_required') {
    addMsg(m.message || 'Please sign in to start a guided session.', 'flag');
    try { showSignInModal(m.message); } catch (e) {}
    return;
  }
  if (m.type === 'onboarding_required') {
    addMsg(m.message || 'Finish quick setup first.', 'flag');
    return;
  }
  if (m.type === 'session.started') {
    if (m.ok === false) {
      const why = m.reason === 'sign_in_required'
        ? 'Please sign in to start a guided session.'
        : (m.reason === 'onboarding_required'
            ? 'Finish quick setup first.'
            : 'Could not start the session.');
      addMsg(why, 'flag');
      if (m.reason === 'sign_in_required') {
        try { showSignInModal(why); } catch (e) {}
      }
      return;
    }
    if (m.session) { try { window.__sessionHUD?.show(m.session); } catch (e) {} }
    window.__inSession = true;
    try { renderSessionRail(m.session); } catch (e) {}
    try { _cancelVariety(); } catch {}   // Q3: session engine drives rotation
    // v198: ONE voice EVERYWHERE. If live voice (Gemini) is the default and
    // not opted out, it NARRATES the guided session too — the session's
    // scripted lines are relayed to it (see the assistant_text handler) so
    // there is a single, consistent voice. Azure stays silent. If Gemini is
    // already live (free-talk), we just MUTE its mic so it stops listening
    // and only narrates; otherwise we start it in narration-only mode (no
    // mic prompt, no reacting to the backing music). If live voice is off or
    // opted out, Azure narrates exactly as before.
    try {
      let optOut = false;
      try { optOut = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
      const wantGemini = !!window.__s2sDefault && !optOut
        && (window.__liveVoiceAvailable || _s2sOn);
      if (wantGemini) {
        window.__sessionVoiceGemini = true;
        _s2sUserStopped = false;
        if (_s2sOn) {
          try { _gv && _gv.setMicEnabled && _gv.setMicEnabled(false); } catch (e) {}
        } else {
          try { _startS2S({ mic: false }); } catch (e) {}
        }
        voiceOn = false;   // Azure stays silent; Gemini is the single voice
        // v201: LESSON BARGE-IN. The lesson music leaks into the mic, so raise
        // the barge-in gate (only a real, louder voice interrupts). If the mic
        // permission is ALREADY granted, open it automatically so the user can
        // "just talk" hands-free; otherwise leave it to the 🎙 button (which
        // prompts on tap) so first-timers aren't hit with a surprise prompt.
        setTimeout(() => {
          try {
            if (!_gv) return;
            _gv.setBargeGate && _gv.setBargeGate(0.14);
            if (navigator.permissions && navigator.permissions.query) {
              navigator.permissions.query({ name: 'microphone' }).then((st) => {
                if (st && st.state === 'granted' && _gv && !micOn) {
                  _gv.enableMic().then((ok) => {
                    if (ok) { micOn = true; try { $('mic').classList.add('rec'); $('mic').title = 'Listening — tap to mute'; } catch (e) {} }
                  });
                }
              }).catch(() => {});
            }
          } catch (e) {}
        }, 900);
      } else {
        window.__sessionVoiceGemini = false;
        _s2sUserStopped = true;   // block S2S auto-reconnect mid-session
        if (_s2sOn && typeof _stopS2S === 'function') {
          window.__resumeS2SAfterSession = true;
          _stopS2S(true);
        }
        voiceOn = true;
      }
    } catch (e) {}
    try { window.__engagement?.onSession?.('started', m.session); } catch (e) {}
    if (window.matchMedia('(max-width: 760px)').matches) closeDrawer();
    return;
  }
  if (m.type === 'session.phase' || m.type === 'session.paused' ||
      m.type === 'session.resumed') {
    if (m.session) { try { window.__sessionHUD?.update(m.session); } catch (e) {} }
    if (m.type === 'session.phase' && m.session && m.session.phase_idx != null) {
      try { if (_railMode === 'session') setCoachRailActive(m.session.phase_idx); } catch (e) {}
    }
    try { window.__engagement?.onSession?.(m.type.split('.')[1], m.session); } catch (e) {}
    // v69: when the session pauses, make sure the backing music + the
    // coach's voice are actually stopped (not just the phase clock).
    if (m.type === 'session.paused') {
      try { stopMusic(); } catch (e) {}
      try { voice?.cancelSpeak({ silenceMs: 4000 }); } catch (e) {}
      try { if (life) { life.setSpeaking(false); life.stopVisemes?.(); } } catch (e) {}
      // v94: actually PAUSE the avatar's looping clip — previously only
      // the phase clock + audio stopped while the dancer kept moving.
      try { if (player) { player.pause?.(); } } catch (e) {}
      try { setPlayPauseLabel(true); } catch (e) {}
    }
    if (m.type === 'session.resumed') {
      try { if (player) { player.resume?.(); } } catch (e) {}
      try { setPlayPauseLabel(false); } catch (e) {}
    }
    return;
  }
  if (m.type === 'session.finished') {
    try { window.__sessionHUD?.hide(); } catch (e) {}
    window.__inSession = false;
    window.__sessionEndedAt = Date.now();   // v97b: ignore late session clips
    try { if (_railMode === 'session') { markCoachRailComplete(); closeCoachRail(); } } catch (e) {}
    // v80: when the session ends, EVERYTHING stops — music, the coach's
    // voice, lip-sync, AND the avatar's motion. Previously only the HUD
    // hid while the music kept thumping and the avatar kept dancing.
    try { stopMusic(); } catch (e) {}
    try { voice?.cancelSpeak({ silenceMs: 4000 }); } catch (e) {}
    try { if (life) { life.setSpeaking(false); life.stopVisemes?.(); } } catch (e) {}
    try { if (player) { player.stop(); player._idleHold = true; player.applyRestPose(); } } catch (e) {}
    try { setPlayPauseLabel(true); } catch (e) {}
    // v198: if live voice NARRATED the session, hand control back to normal
    // full-duplex free-talk by re-enabling the mic (if one is open). When the
    // session ran Gemini in narration-only mode there is no open mic, so the
    // user can tap 🎧 to start talking; the coach's voice keeps working.
    try {
      if (window.__sessionVoiceGemini) {
        window.__sessionVoiceGemini = false;
        if (_gv && _s2sOn && typeof _gv.setMicEnabled === 'function') {
          _gv.setMicEnabled(true);
        }
        // v201: restore the sensitive free-talk barge-in gate now the lesson
        // music has stopped, and reset the 🎙 button label.
        try { _gv && _gv.setBargeGate && _gv.setBargeGate(0.08); } catch (e) {}
      }
    } catch (e) {}
    // v109b: resume live voice if we paused it for the session
    // (and the user hasn't opted out of live voice).
    try {
      let optOut = false;
      try { optOut = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
      if (window.__resumeS2SAfterSession && !optOut && !_s2sOn
          && typeof _startS2S === 'function') {
        window.__resumeS2SAfterSession = false;
        setTimeout(() => { try { _startS2S(); } catch (e) {} }, 600);
      }
    } catch (e) {}
    try { window.__engagement?.onSession?.('finished', m.session); } catch (e) {}
    return;
  }
  if (m.type === 'interrupted') {
    // Server confirmed our barge-in landed: it cancelled the in-flight
    // turn. Nothing else to do — TTS already silenced locally.
    console.info('[ws] turn interrupted');
    return;
  }
  // v33d: streaming preview. The agent yields content tokens as they
  // arrive from the LLM (TTFT ~300ms vs ~1.5s buffered). We paint
  // them into a "live" coach bubble so the user sees Kira typing
  // immediately. TTS still waits for `assistant_text` so audio stays
  // ordered with avatar moves.
  if (m.type === 'assistant_text_delta') {
    if (!window._liveCoach) {
      // Remove the empty-state hint on first real message.
      const empty = log.querySelector('.empty');
      if (empty) empty.remove();
      const el = document.createElement('div');
      el.className = 'msg coach live';
      const ts = document.createElement('span');
      ts.className = 'ts';
      ts.textContent = _fmtTs(new Date()) + _latencyTag('coach');
      el.appendChild(ts);
      const body = document.createElement('span');
      body.className = 'body';
      el.appendChild(body);
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
      window._liveCoach = { el, body, text: '' };
    }
    window._liveCoach.text += m.text;
    // v70: the model occasionally streams a raw inline tool call
    // (e.g. `<function=set_mood{"mood":"excited"}</function>`) into
    // the narration. The server cleans the FINAL bubble, but on the
    // inline-salvage path no clean final arrives and the raw tag is
    // left on screen. Strip the debris from the live preview so the
    // student never sees raw tool syntax.
    window._liveCoach.body.textContent = _stripFnTags(window._liveCoach.text);
    log.scrollTop = log.scrollHeight;
    return;
  }
  if (m.type === 'assistant_text_clear') {
    // Tool round started — retract the preview so narration doesn't
    // appear before the avatar moves.
    if (window._liveCoach) {
      try { window._liveCoach.el.remove(); } catch {}
      window._liveCoach = null;
    }
    return;
  }
  if (m.type === 'assistant_text') {
    // v95: MODE SEPARATION. The /ws/agent socket fires idle "nudge"
    // lines ("tap a chip…") that clash with the OTHER two modes. Drop
    // them when live voice (S2S) is on OR a guided session is running —
    // those modes own the conversation and the nudges are confusing
    // (e.g. asking the user to tap chips mid-session).
    if (m.source === 'idle_nudge' && (_s2sOn || window.__inSession)) {
      return;
    }
    // v97b: drop late session narration that lands right after the user
    // ended the session (the ticker queues a few lines before it stops).
    if ((m.source === 'session_voiceover' || m.source === 'session_heartbeat'
         || m.source === 'session_narration')
        && !window.__inSession && window.__sessionEndedAt
        && (Date.now() - window.__sessionEndedAt) < 5000) {
      return;
    }
    // v70: belt-and-braces — strip any inline tool-call debris before
    // it reaches the bubble, the floating overlay, OR the TTS voice.
    if (m.text) m.text = _stripFnTags(m.text);
    // v69: DUPLICATE-SPEECH GUARD. Occasionally two sources (a phase
    // voiceover + a per-clip narration, or a retried turn) emit the
    // same line back-to-back and the coach says it twice. Drop an
    // identical line that arrives within 7 s of the last one so she
    // never repeats herself.
    const _norm = (m.text || '').trim().toLowerCase();
    const _now = Date.now();
    if (_norm && window.__lastSpoken && window.__lastSpoken.text === _norm &&
        (_now - window.__lastSpoken.at) < 7000) {
      return;
    }
    if (_norm) window.__lastSpoken = { text: _norm, at: _now };
    // Promote any live-streamed preview to a finalised bubble.
    if (window._liveCoach) {
      window._liveCoach.el.classList.remove('live');
      if (window._liveCoach.body) {
        window._liveCoach.body.textContent = m.text;
      } else {
        window._liveCoach.el.textContent = m.text;
      }
      // Mirror to the floating bubble overlay (addMsg normally does
      // this, but we're reusing the existing element).
      try { showBubble(m.text, 'coach'); } catch {}
      if (!isDrawerOpen()) try { bumpChatBadge(); } catch {}
      window._liveCoach = null;
    } else {
      addMsg(m.text, 'coach');
    }
    // The voice line is starting → start the deferred motion now so
    // the spoken counts line up with the visual choreography.
    _flushPendingMotion();
    // v108: NEVER run two voices at once. When live voice owns the audio
    // (it is actually LIVE), DO NOT also speak this line via Azure TTS.
    // v198: single voice = Gemini EVERYWHERE. During a guided session the
    // scripted narration lines are RELAYED to the live coach so SHE speaks
    // them (free-talk lines are already voiced by Gemini's own audio, so
    // they must NOT be relayed again). One voice, no clash.
    const _s2sActive = _s2sOn;
    const _isSessionLine = (m.source === 'session_voiceover'
      || m.source === 'session_heartbeat' || m.source === 'session_narration');
    if (_s2sActive && _isSessionLine && _gv && typeof _gv.speakText === 'function') {
      try { _gv.speakText(m.text); } catch (e) {}
    }
    if (voiceOn && !_s2sActive) {
      try {
        if (life) life.stopVisemes();
        if (life) life.setSpeaking(true);
        console.info('[voice] speak start:', m.text.slice(0, 60));
        duckMusic(true);
        await voice.speak(m.text, {
          onviseme: (v) => { if (life) life.pushViseme(v); },
        });
        if (life) life.stopVisemes();
        if (life) life.setSpeaking(false);
        duckMusic(false);
        console.info('[voice] speak done');
      } catch (e) {
        if (life) life.setSpeaking(false);
        duckMusic(false);
        console.warn('[voice] speak failed:', e?.message || e);
      }
    } else if (life) {
      // Even without TTS, give a brief talking-gesture pulse so the
      // chat bubble doesn't appear over a frozen statue.
      life.setSpeaking(true);
      setTimeout(() => life.setSpeaking(false),
                 Math.min(4000, 600 + (m.text?.length || 0) * 35));
    }
    return;
  }
  if (m.type === 'tool_call') {
    // Tool calls are an internal mechanism — never shown to the user.
    console.debug('[coach.tool]', m.name, m.args, '→', m.result);
    return;
  }
  if (m.type === 'llm_quota') {
    showQuotaModal(m);
    return;
  }
  if (m.type === 'avatar_event') {
    const e = m.event;
    if (e.type === 'avatar.load') {
      // no-op; we load+play together when we get avatar.play
      window.__pendingClip = e.clip_id;
    } else if (e.type === 'avatar.play') {
      // v97b: if the user just ENDED a session, ignore late session-ticker
      // avatar.play events that the server queued before it stopped —
      // otherwise the avatar restarts dancing right after 'End'.
      if (window.__sessionEndedAt && (Date.now() - window.__sessionEndedAt) < 4000
          && !window.__inSession) {
        return;
      }
      window.__sendToken++;           // cancel the stuck-pose watchdog
      // On phones, close the drawer so the user can actually watch
      // the dance they asked for.
      if (window.matchMedia('(max-width: 760px)').matches) closeDrawer();
      // v33: HARD STOP any previously-pending clip (and its 1.5 s
      // fallback timer) before queuing this one. Without this, a
      // second avatar.play arriving mid-turn could race the first
      // and the user saw "two motions playing together" on screen.
      if (_pendingMotionTimer) {
        clearTimeout(_pendingMotionTimer);
        _pendingMotionTimer = null;
      }
      _pendingMotion = null;
      // Also stop any music that's currently playing — the new
      // clip will start its own backing track when speech begins.
      try { stopMusic(); } catch {}
      // SYNC: load the clip + cue the music NOW but DEFER starting
      // playback until the coach's voice begins. That way the
      // narrated "and 1, 2, 3..." lines up with the avatar actually
      // BEING at counts 1, 2, 3 — instead of the avatar already 4
      // counts ahead by the time TTS audio starts. If no voice line
      // arrives within 1.5 s we fall back to starting immediately
      // (some turns are motion-only).
      _pendingMotion = {
        clip_id: e.clip_id, music_url: e.music_url,
        opts: { speed: e.speed, mirror: e.mirror, loop: e.loop },
      };
      // v33c item 7: infer per-genre mood from the clip id so the
      // face matches what the body is doing. Krump→focused, Waacking
      // /Locking→happy, Popping→surprised, CMU drills→relaxed, rest
      // default to happy.
      if (life) {
        const cid = String(e.clip_id || '');
        let mood = 'happy';
        if (cid.startsWith('cmu_')) mood = 'relaxed';
        else if (cid.startsWith('gKR') || cid.startsWith('gBR')) mood = 'focused';
        else if (cid.startsWith('gPO')) mood = 'surprised';
        else if (cid.startsWith('gWA') || cid.startsWith('gLO')) mood = 'happy';
        try { life.setMood(mood); } catch {}
        // v33d item 5: feed BPM into the idle beat-bounce so the
        // avatar bobs in time with the backing track even when not
        // dancing (between clips, mid-explanation, etc.).
        if (typeof e.bpm === 'number' && e.bpm > 30 && e.bpm < 220) {
          life._idleBpm = e.bpm;
        }
      }
      _preloadClip(e.clip_id);
      // v69: snappier transitions. Previously we held the queued clip for
      // up to 1.5 s waiting for a voice line to sync narrated counts — but
      // on motion-only turns (most session rotations) that read as the
      // avatar "getting stuck" between moves. 700 ms is still enough for a
      // voice line to land and flush early; if none comes, the new clip
      // starts ~0.8 s sooner so the hand-off feels live.
      _pendingMotionTimer = setTimeout(() => { _flushPendingMotion(); }, 700);
    } else if (e.type === 'avatar.drill') {
      runDrill(e);                       // fire-and-forget; cancels prior
    } else if (e.type === 'avatar.speed') {
      if (player) player.speed = e.speed;
    } else if (e.type === 'avatar.mirror') {
      if (player) player.mirror = e.mirror;
    } else if (e.type === 'avatar.stop') {
      if (player) player.stop();
      setPlayPauseLabel(true);
      stopMusic();
      // v33d item 9: brief flourish gesture so we don't snap-cut
      // straight to the rest pose.
      if (life) try { life.playFlourish({ duration: 1.2 }); } catch {}
    } else if (e.type === 'avatar.mood') {
      if (life) life.setMood(e.mood);
    } else if (e.type === 'avatar.language') {
      // v86: coach switched language by voice/LLM (set_language tool).
      const _lang = (e.language || 'english').toLowerCase();
      window.__coachLang = _lang;
      try { localStorage.setItem('coach_lang', _lang); } catch (e2) {}
      try { voice.setLanguage(_lang); } catch (e2) {}
      const _sel = document.getElementById('lang-select');
      if (_sel) _sel.value = _lang;
      const _names = { english: 'English', hinglish: 'Hinglish', hindi: 'Hindi' };
      try { addMsg('Coach language set to ' + (_names[_lang] || _lang) + '.', 'sys'); } catch (e2) {}
    } else if (e.type === 'avatar.isolate') {
      // Body-part isolation: only drive the named bones/groups; pin
      // the rest at bind-pose so the user sees the move broken down.
      if (player) player.isolate(e.parts);
      addMsg(
        'isolating: ' + (Array.isArray(e.parts) ? e.parts.join(', ') :
                         (e.parts || 'cleared')),
        'tool'
      );
    } else if (e.type === 'avatar.unisolate') {
      if (player) player.unisolate();
      addMsg('isolation cleared', 'tool');
    } else if (e.type === 'avatar.breakdown') {
      // v34: GUIDED BREAKDOWN — sequence isolate→speed→narrate per stage.
      runBreakdown(e).catch((err) => console.warn('[breakdown]', err));
    } else if (e.type === 'ui.open_audio_picker') {
      window.__resequenceOpts = {
        genre: e.genre || null, query: e.query || null,
        bars:  e.bars  || 8,
      };
      $('audio-input').click();
    } else if (e.type === 'ui.open_video_picker') {
      window.__feedbackClipId = e.clip_id;
      $('video-input').click();
    } else if (e.type === 'ui.open_live_feedback') {
      openLiveFeedback(e.clip_id).catch((err) =>
        addMsg('live mirror error: ' + (err?.message || err), 'flag'));
    } else if (e.type === 'ui.open_learn') {
      // v197: the coach's brain opened the Lessons panel for the student.
      try { window.__llmOpenLearn && window.__llmOpenLearn(e.style || null); } catch (er) {}
    } else if (e.type === 'ui.open_lesson') {
      try { window.__llmOpenLesson && window.__llmOpenLesson(e.lesson_id); } catch (er) {}
    } else if (e.type === 'ui.open_profile') {
      try { window.__openProfilePage && window.__openProfilePage(); } catch (er) {}
    } else if (e.type === 'ui.close_panels') {
      // v223: the coach's brain dismissed all panes so the student can watch
      // the avatar full-screen. Close the steps rail, chat drawer, lessons
      // library and live-mirror popup — each guarded, harmless if not open.
      try { window.__closeAllPanels && window.__closeAllPanels(); } catch (er) {}
    }
    return;
  }
  if (m.type === 'error') addMsg('error: ' + m.message, 'flag');
}

function sendUser(text, source) {
  if (!text) return;
  // v34i: tag the source so the backend can apply the STT noise
  // gate ONLY to voice-derived turns. Typed text (textbox submit or
  // chip button) is by definition intentional and must NEVER be
  // silently dropped by the keyword gate — that was the root cause
  // of typed lines like "start slow jog warmup" getting no reply.
  if (source !== 'voice') source = 'typed';
  // ── Duplicate-submit guard ────────────────────────────────────────
  // Several input paths can dispatch the SAME utterance back-to-back:
  //   • STT `onpartial` fills the text field, user presses Enter →
  //     form submit fires → sendUser. Then STT `onfinal` fires with
  //     the same text → sendUser AGAIN.
  //   • Some browsers fire submit + button-click for one tap.
  //   • Quick-action buttons hit twice on double-tap.
  // Each duplicate dispatches a second `user_interrupt` which CANCELS
  // the LLM reply to the first message, leaving the user with no
  // response. Drop any identical text within 2.5 s of the last send.
  const now = Date.now();
  if (window.__lastSentText === text &&
      now - (window.__lastSentAt || 0) < 2500) {
    console.info('[ws] duplicate sendUser suppressed:', text.slice(0, 40));
    return;
  }
  window.__lastSentText = text;
  window.__lastSentAt   = now;
  // Barge-in: any new user input cancels whatever the coach was
  // saying / about to say. cancelSpeak shuts up the TTS immediately;
  // the server-side user_interrupt cancels the in-flight LLM stream.
  try { voice?.cancelSpeak({ silenceMs: 2000 }); } catch {}
  addMsg(text, 'user');
  // If the WS isn't ready yet (initial load, mid-reconnect after a
  // tab switch, slow network), queue the message instead of silently
  // dropping it. The queue is flushed in ws.onopen.
  if (!ws || ws.readyState !== 1) {
    window.__wsQueue = window.__wsQueue || [];
    window.__wsQueue.push({ type: 'user_interrupt' });
    window.__wsQueue.push({ type: 'user_text', text, source });
    setStatus('reconnecting… your message is queued');
    // Kick a reconnect if the socket is dead or closing.
    try {
      if (!ws || ws.readyState === WebSocket.CLOSED ||
          ws.readyState === WebSocket.CLOSING) {
        connect();
      }
    } catch (e) {}
  } else {
    try { ws.send(JSON.stringify({ type: 'user_interrupt' })); } catch {}
    ws.send(JSON.stringify({ type: 'user_text', text, source }));
  }
  // ── Stuck-pose watchdog ────────────────────────────────────────────
  // When the LLM responds with text only and never dispatches an
  // avatar.play tool-call (rate-limit hiccup, model refusing, etc.)
  // the avatar would otherwise stay frozen in whatever weird mid-clip
  // pose it was last in. We give the agent 8 seconds to actually
  // start motion; if it doesn't, snap the avatar back to the natural
  // rest stance so the user never sees an "absurd position".
  const myToken = ++window.__sendToken;
  setTimeout(() => {
    if (myToken !== window.__sendToken) return;       // newer turn took over
    if (!player) return;
    if (player.playing && !player.flagged) return;    // motion already running
    // No motion arrived → quietly reset to standing rest.
    try { player.applyRestPose(); } catch {}
    setPlayPauseLabel(true);
  }, 8000);
}
if (!window.__sendToken) window.__sendToken = 0;

$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  // v184: belt-and-suspenders — CSS pointer-events:none already blocks
  // focusing #text once the anon wall is active, but if the field was
  // already focused the instant the wall triggered, Enter/keyboard submit
  // can still fire. Block it here too instead of trusting CSS alone.
  if (typeof _anonWallActive !== 'undefined' && _anonWallActive) {
    try { $('text').blur(); } catch (e2) {}
    try { showSignInModal('Sign in to keep chatting and dancing with me.'); } catch (e2) {}
    return;
  }
  const t = $('text').value.trim();
  if (t) { sendUser(t); $('text').value = ''; }
});
// NOTE: the send button is type=submit inside #composer, so clicking
// it already fires the submit handler above. A second click listener
// here would call sendUser TWICE — the user would see their message
// echoed in two bubbles and the server would receive two user_text
// messages back-to-back; the second user_interrupt cancels the LLM
// response to the first, leaving the coach silent. Do NOT add a
// separate click handler on #send.

// ─── voice buttons ────────────────────────────────────────────────────
$('mic').addEventListener('click', async () => {
  // v201: when live voice (Gemini) owns the audio — including during a
  // guided lesson that started output-only — the 🎙 button toggles LESSON
  // BARGE-IN (open/close Gemini's mic so you can talk to interrupt), NOT the
  // separate Azure STT path (running both mics clashes).
  if (_s2sOn && _gv) {
    try {
      if (!micOn) {
        const ok = await _gv.enableMic();
        if (!ok) { addMsg('Mic blocked — allow microphone access to talk.', 'flag'); return; }
        micOn = true;
        $('mic').classList.add('rec');
        $('mic').title = 'Listening — tap to mute';
        try { addMsg('🎙 Go ahead — talk any time, I\u2019m listening.', 'sys'); } catch (e) {}
      } else {
        _gv.disableMic();
        micOn = false;
        $('mic').classList.remove('rec');
        $('mic').title = 'Tap to talk';
      }
    } catch (e) { addMsg('mic error: ' + (e && e.message || e), 'flag'); }
    return;
  }
  try {
    if (!micOn) {
      voice.onpartial = (t) => {
        $('text').value = t;
        // First partial of a new utterance → user started talking.
        // Barge in: cut the coach off mid-sentence and tell the
        // server to abort the in-flight turn so the LLM listens
        // instead of continuing its previous train of thought.
        if (t && !window.__bargedThisUtt) {
          window.__bargedThisUtt = true;
          try { voice.cancelSpeak({ silenceMs: 2500 }); } catch {}
          try { ws?.send(JSON.stringify({ type: 'user_interrupt' })); } catch {}
        }
        if (!t) window.__bargedThisUtt = false;
      };
      voice.onfinal   = (t) => {
        $('text').value = '';
        window.__bargedThisUtt = false;
        sendUser(t, 'voice');
      };
      await voice.startListening();
      micOn = true;
      $('mic').classList.add('rec');
      $('mic').title = 'Stop listening';
    } else {
      await voice.stopListening();
      micOn = false;
      $('mic').classList.remove('rec');
      $('mic').title = 'Voice input';
    }
  } catch (e) {
    addMsg('mic error: ' + e.message, 'flag');
  }
});
$('tts').addEventListener('click', () => {
  voiceOn = !voiceOn;
  $('tts').classList.toggle('live', voiceOn);
  $('tts').textContent = voiceOn ? '🔊' : '🔇';
  $('tts').title = voiceOn ? 'Voice on' : 'Voice off';
});

// ─── live voice (speech-to-speech) ──────────────────────────────────
// A1: real-time voice chat. The coach HEARS you (tone, energy), talks
// back in < ~500 ms, and you can talk over her. Tool calls still drive
// the avatar through the same engine. Feature-flagged on the server
// (the live-voice key); the 🎧 button only appears when /api/voice/status
// reports enabled. Falls back to the Azure 🎙 path otherwise.
let _gv = null;          // live voiceVoice instance
let _s2sOn = false;
let _s2sCoachLine = null;
let _s2sRetries = 0;         // v100: transient-drop reconnect budget
let _s2sUserStopped = false; // v100: true only when the USER taps to stop

function _setLiveBtn(state) {
  const b = $('live-voice'); if (!b) return;
  b.classList.toggle('rec', state === 'live' || state === 'connecting');
  b.classList.toggle('live-on', state === 'live');
  b.classList.toggle('live-connecting', state === 'connecting');
  b.setAttribute('aria-pressed', state === 'live' ? 'true' : 'false');
  b.title = state === 'live' ? 'Live voice ON — tap to stop'
          : state === 'connecting' ? 'Connecting…'
          : 'Live voice chat (talk naturally)';
  // v198: a real status pill above the dock so the connection state reads
  // as a product, not just a coloured icon.
  const pill = $('voice-status');
  if (pill) {
    if (state === 'connecting') {
      pill.innerHTML = '<span class="vs-dot"></span>Connecting voice…';
      pill.className = 'vs-connecting show';
    } else if (state === 'live') {
      pill.innerHTML = '<span class="vs-dot"></span>Live voice on';
      pill.className = 'vs-live show';
    } else {
      pill.className = '';
    }
  }
  // v220: keep the persistent 🎧 ring in sync with every state change.
  try { _syncVoiceCue(); } catch (e) {}
}

// ─── one-tap "turn on voice" hint (Q2) ──────────────────────────────
// Browsers block auto-starting the mic without a user gesture, so live
// voice can't turn itself on at load. Instead, once the user dismisses
// the front door, if live voice is armed-but-off we surface a small
// TAPPABLE hint on the existing status pill. One tap = one gesture =
// voice goes live. Shown at most once per page load; never during a
// guided session (Azure narrates those).
let _voiceHintShown = false;
// v220: PERSISTENT headphones cue. The transient pill hint had fragile timing
// (it fired at a fixed delay before /api/voice/status had set
// __liveVoiceAvailable, so on slow loads it silently bailed and never came
// back). This adds an ALWAYS-visible pulsing ring on the 🎧 button whenever
// live voice is available but OFF — so the "you can talk" affordance is never
// missed, independent of the pill's timing. Cleared the moment voice goes on.
function _syncVoiceCue() {
  try {
    const btn = document.getElementById('live-voice');
    if (!btn) return;
    let off = false;
    try { off = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
    const avail = (window.__s2sDefault || window.__liveVoiceAvailable);
    const show = !!avail && !_s2sOn && !off && btn.style.display !== 'none';
    btn.classList.toggle('voice-cue', show);
  } catch (e) {}
}

function _maybeVoiceHint(_tries) {
  _tries = _tries || 0;
  // Always keep the persistent 🎧 ring in sync (independent of the pill).
  _syncVoiceCue();
  if (_s2sOn) return;
  let s2sOff = false;
  try { s2sOff = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
  if (s2sOff) return;
  // v220: voice availability is set ASYNC after /api/voice/status resolves.
  // If it isn't ready yet, RETRY (up to ~12s) instead of bailing forever —
  // that was why the hint "didn't come every time" on slow/cold loads.
  if (!window.__s2sDefault && !window.__liveVoiceAvailable) {
    if (_tries < 24) setTimeout(() => _maybeVoiceHint(_tries + 1), 500);
    return;
  }
  if (_voiceHintShown) return;   // pill shown once per session (see reset below)
  const pill = document.getElementById('voice-status');
  const btn = document.getElementById('live-voice');
  if (!pill || !btn) return;
  _voiceHintShown = true;
  try { _coachTrack('dance_voice_hint_shown', {}); } catch (e) {}
  pill.innerHTML = '<span class="vs-dot"></span>🎧 Tap to talk to your coach';
  pill.className = 'vs-connecting show';
  pill.style.cursor = 'pointer';
  const go = () => {
    pill.removeEventListener('click', go);
    pill.style.cursor = '';
    try { btn.click(); } catch (e) {}         // starts S2S with this gesture
  };
  pill.addEventListener('click', go);
  // Auto-dismiss the hint after a while if ignored (unless it already went
  // live, which _setLiveBtn will have repainted). Longer window so it's seen.
  setTimeout(() => {
    if (!_s2sOn && pill.className.indexOf('vs-connecting') !== -1) {
      pill.className = '';
      pill.style.cursor = '';
      try { pill.removeEventListener('click', go); } catch (e) {}
    }
  }, 14000);
}

async function _startS2S(opts) {
  opts = opts || {};
  // v198: live voice (Gemini) is now the ONE voice EVERYWHERE — including
  // guided sessions, which run it in narration-only mode (mic:false). So we
  // no longer bail out during a session; we just pick the right mic mode.
  if (_s2sOn) return;
  _s2sOn = true;
  try { _coachTrack('dance_voice_started', {}); } catch (e) {}
  const _micWanted = (opts.mic !== false);
  _s2sUserStopped = false;   // v100: fresh start, allow auto-reconnect
  _setLiveBtn('connecting');
  // Stop the Azure half-duplex paths so they don't fight live voice.
  try { if (micOn) { await voice.stopListening(); micOn = false; $('mic').classList.remove('rec'); } } catch (e) {}
  try { voice.cancelSpeak({ silenceMs: 200 }); } catch (e) {}
  const _prevVoiceOn = voiceOn;
  _gv = new LiveVoice({
    appBase: APP_BASE,
    getQuery: () => {
      const q = {};
      try { const t = localStorage.getItem('token'); if (t) q.token = t; } catch (e) {}
      const prof = window.__charProfile;
      if (window.__charSlug) q.character = window.__charSlug;
      if (prof) { q.display_name = prof.display_name || ''; q.style = prof.style || prof.tagline || ''; }
      try { q.language = localStorage.getItem('coach_lang') || 'hinglish'; } catch (e) {}
      return q;
    },
  });
  // v89: if the camera is already on, let the live coach SEE the dancer.
  try { if (_camStream) _gv.setCameraSource($('mirrorcam')); } catch (e) {}
  _gv.onAvatarEvent = (event) => { try { handleEvent({ type: 'avatar_event', event }); } catch (e) {} };
  _gv.onSpeaking = (on) => { try { if (life) life.setSpeaking(on); } catch (e) {} };
  _gv.onTranscript = (role, text, final) => {
    if (!text) return;
    if (role === 'coach') {
      if (!_s2sCoachLine) { _s2sCoachLine = addMsg('', 'coach'); _s2sCoachLine._t = ''; }
      _s2sCoachLine._t += text;
      const body = _s2sCoachLine.querySelector('.body');
      if (body) body.textContent = _s2sCoachLine._t;
      // v108b: show the floating stage bubble ONCE the line is complete
      // (no per-token thrash, and no blank placeholder dot before text).
      if (final) {
        try { if (_s2sCoachLine._t.trim()) showBubble(_s2sCoachLine._t, 'coach'); } catch (e) {}
        _s2sCoachLine = null;
      }
    } else {
      addMsg(text, 'user');
    }
  };
  _gv.onState = (state, info) => {
    _setLiveBtn(state);
    if (state === 'live') {
      _s2sRetries = 0;   // healthy connection resets the retry budget
      addMsg('🎧 Live voice on — just talk to me.', 'sys');
      voiceOn = false;
    }
    if (state === 'error') {
      const msg = (info && info.message) || '';
      // v100: live voice often drops with a transient 1008/abort after a
      // lull. Don't scare the user with 'unavailable' — silently reconnect
      // a few times. Only fall back to normal voice if reconnects fail.
      const transient = /1008|abort|gemini_recv|recv|timeout|deadline/i.test(msg);
      if (transient && !_s2sUserStopped && _s2sRetries < 3) {
        _s2sRetries++;
        try { _gv && _gv.stop(); } catch (e) {}
        _gv = null; _s2sOn = false;
        _setLiveBtn('connecting');
        setTimeout(() => { if (!_s2sUserStopped) { try { _startS2S(); } catch (e) {} } }, 800);
        return;
      }
      addMsg('Voice paused — tap 🎧 to resume.', 'sys');
      _stopS2S(_prevVoiceOn);
    }
    if (state === 'ended') {
      _s2sCoachLine = null;
      // v100: if live voice ended on its own (not a user tap) while we still
      // wanted it live, reconnect so the conversation never dead-ends.
      if (!_s2sUserStopped && _s2sOn && _s2sRetries < 3) {
        _s2sRetries++;
        _gv = null; _s2sOn = false;
        setTimeout(() => { if (!_s2sUserStopped) { try { _startS2S(); } catch (e) {} } }, 800);
      }
    }
  };
  try {
    await _gv.start({ mic: _micWanted });
  } catch (e) {
    addMsg('mic error: ' + (e && e.message || e), 'flag');
    _stopS2S(_prevVoiceOn);
  }
}

function _stopS2S(restoreVoiceOn) {
  _s2sOn = false;
  _s2sCoachLine = null;
  try { _gv && _gv.stop(); } catch (e) {}
  _gv = null;
  if (typeof restoreVoiceOn === 'boolean') voiceOn = restoreVoiceOn;
  _setLiveBtn('ended');
}

(function initLiveVoice() {
  const btn = $('live-voice'); if (!btn) return;
  // Probe server capability; reveal the button only when S2S is live.
  fetch(APP_BASE + '/api/voice/status')
    .then(r => r.json())
    .then(s => {
      if (!s || !s.enabled) return;
      btn.style.display = '';
      // v141: live voice (Gemini "Aoede", the 🎧 headphones mode) is now the
      // DEFAULT and PRIMARY voice — the user prefers the natural full-duplex
      // conversation. Azure Neerja TTS is kept ONLY as an automatic fallback
      // (if S2S fails to connect) or if the user EXPLICITLY turns live voice
      // off. So we default S2S ON unless the user opted out.
      let optOut = false;
      try { optOut = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
      if (optOut) {
        window.__s2sDefault = false;       // Azure TTS owns the voice
        try { _setLiveBtn('ended'); } catch (e) {}
        return;
      }
      window.__s2sDefault = true;   // tells the Azure auto-mic to stand down
      // v109b: show the 🎧 button as ACTIVE by default so it's clear live
      // voice is the default mode (the user kept thinking it was off).
      // Browsers still require ONE user gesture before the mic can open,
      // so we ALSO arm on the first interaction AND on the session/hero
      // buttons. The visual 'rec' state makes it read as on immediately.
      window.__liveVoiceAvailable = true;   // "Just talk" can use live voice
      // v220: now that voice is confirmed available, light up the persistent
      // 🎧 ring so the "you can talk" cue is visible immediately (and re-run
      // the hint check in case it was waiting on this).
      try { _syncVoiceCue(); } catch (e) {}
      try { _maybeVoiceHint(); } catch (e) {}
      // v193: DO NOT auto-start live voice on a random tap. That was the
      // root of BOTH "voice comes instantly" AND the two-voice clash — the
      // same tap that launched an Azure-narrated guided session also started
      // Gemini, so both talked over each other. Voice is now strictly opt-in:
      // the "Just talk" button or the 🎧 toggle. The first user gesture only
      // unlocks audio playback (needed for Azure/music autoplay policies).
      try { _setLiveBtn('ended'); } catch (e) {}
      const _unlockOnce = () => {
        try { unlockAudio(); } catch (e) {}
        window.removeEventListener('pointerdown', _unlockOnce, true);
        window.removeEventListener('keydown', _unlockOnce, true);
        window.removeEventListener('touchstart', _unlockOnce, true);
      };
      window.addEventListener('pointerdown', _unlockOnce, true);
      window.addEventListener('keydown', _unlockOnce, true);
      window.addEventListener('touchstart', _unlockOnce, true);
    })
    .catch(() => {});
  btn.addEventListener('click', () => {
    // First user gesture also unlocks audio playback.
    try { unlockAudio(); } catch (e) {}
    if (_s2sOn) {
      // User explicitly turned live voice OFF — clear the opt-in so the
      // next visit uses the default Azure (Neerja) voice again.
      try { localStorage.removeItem('coach.s2s.on'); localStorage.setItem('coach.s2s.off', '1'); } catch (e) {}
      window.__s2sDefault = false;
      _s2sUserStopped = true;   // v100: block auto-reconnect
      _stopS2S(true);
    } else {
      // User explicitly opted IN to live voice — remember it for next time.
      try { localStorage.setItem('coach.s2s.on', '1'); localStorage.removeItem('coach.s2s.off'); } catch (e) {}
      _startS2S();
    }
  });
})();

// ─── inline auth (login / signup popup) ──────────────────────────────
// No more bouncing to studioos.fit/login and reloading — the user used
// to land back here without a usable token and loop forever. Instead we
// open an in-page popup that posts to our own /api/auth/{login,register}
// proxy (which forwards to studio-Os), stash the returned access_token in
// localStorage['token'], reconnect the WS with it, and resume whatever the
// user was doing — all without leaving the page.

// Action to retry once the user successfully signs in (e.g. the session
// length they tapped). Set by the caller before the gate appears.
window.__pendingAuthAction = window.__pendingAuthAction || null;

function _isSignedIn() {
  try { return !!localStorage.getItem('token'); } catch (e) { return false; }
}

// v185: refresh_token has been STORED on every login/register since this
// modal was written, but nothing ever used it. When the studio-Os access
// token expired, _identify_user on the server just silently failed and
// the session quietly ran anonymous for the rest of the visit — no error,
// no re-login prompt, just a confusing loss of streak/history. This pair
// of functions detects that and either recovers silently (refresh) or
// prompts cleanly (re-login), instead of the previous silent failure.
async function _tryRefreshToken() {
  let refreshTok = '';
  try { refreshTok = localStorage.getItem('refresh_token') || ''; } catch (e) {}
  if (!refreshTok) return false;
  try {
    const r = await fetch(APP_BASE + '/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshTok }),
    });
    if (!r.ok) return false;
    const data = await r.json().catch(() => null);
    const tok = data && (data.access_token || data.token);
    if (!tok) return false;
    try {
      localStorage.setItem('token', tok);
      if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
    } catch (e) {}
    return true;
  } catch (e) { return false; }
}

let _verifyingSession = false;
async function _verifySessionOrPrompt() {
  // Anonymous visitors are not signed in at all — nothing to verify, and
  // must NEVER be prompted to sign in just because of this check (that's
  // the anon-wall's job, on its own timer).
  if (!_isSignedIn() || _verifyingSession) return;
  _verifyingSession = true;
  try {
    let tok = '';
    try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    const r = await fetch(APP_BASE + '/api/me/progress', {
      headers: { Authorization: 'Bearer ' + tok },
    }).catch(() => null);
    if (!r || !r.ok) return;   // network hiccup — don't punish the user
    const d = await r.json().catch(() => null);
    if (d && d.user) return;  // token still valid, nothing to do
    // Token rejected server-side. Try a silent refresh first.
    if (await _tryRefreshToken()) {
      try { ws && ws.close(); } catch (e) {}
      try { connect(); } catch (e) {}
      return;
    }
    // Refresh failed too (or no refresh_token) — this is the fix for the
    // silent-anonymous-degrade bug: clear stale tokens and show a clean,
    // honest re-login prompt instead of just continuing as anonymous.
    try {
      localStorage.removeItem('token'); localStorage.removeItem('refresh_token');
      localStorage.removeItem('user_name'); localStorage.removeItem('user_email');
    } catch (e) {}
    try { _refreshAccountBtn(); } catch (e) {}
    try {
      showSignInModal('Your session expired — sign in again to keep your streak and history.');
    } catch (e) {}
  } finally {
    _verifyingSession = false;
  }
}
// Periodic safety net in case the tab stays open (and visible) for a
// long single session without ever backgrounding.
setInterval(() => { try { _verifySessionOrPrompt(); } catch (e) {} }, 5 * 60 * 1000);

function _applyAuthSuccess(data) {
  // studio-Os returns { access_token, refresh_token, user }.
  const tok = data && (data.access_token || data.token);
  if (!tok) return false;
  try {
    localStorage.setItem('token', tok);
    if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
    const u = data.user || {};
    if (u.name)  localStorage.setItem('user_name', u.name);
    if (u.email) localStorage.setItem('user_email', u.email);
  } catch (e) {}
  _refreshAccountBtn();
  try { window.__refreshPushOptin && window.__refreshPushOptin(); } catch (e) {}
  try { window.__postNativeToken && window.__postNativeToken(); } catch (e) {}
  try { _releaseAnonWall(); } catch (e) {}
  // v220: if a delighted anonymous user just signed up from the end-of-session
  // 👍 card, persist the session they earned while anonymous so their streak +
  // minutes carry over to the new account, then celebrate + show their profile.
  try {
    const raw = localStorage.getItem('coach.pendingSave');
    if (raw) {
      const p = JSON.parse(raw);
      localStorage.removeItem('coach.pendingSave');
      // Only honour a recent pending save (last 30 min) to avoid stale replays.
      if (p && (Date.now() - (p.at || 0)) < 1800000 && (p.minutes || p.elapsed_sec)) {
        const mins = p.minutes || Math.max(1, Math.round((p.elapsed_sec || 0) / 60));
        fetch(APP_BASE + '/api/me/save-session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + tok },
          body: JSON.stringify({ minutes: mins, template_id: p.template || '',
                                 style: p.style || '' }),
        }).then((r) => r.json()).then((res) => {
          if (res && res.saved) {
            try { _showProgressSaved(mins); } catch (e) {}
          }
        }).catch(() => {});
      }
    }
  } catch (e) {}
  // Reconnect the agent socket so the server identifies the user on this
  // very session (the token rides as ?token= on the WS URL).
  try { ws && ws.close(); } catch (e) {}
  try { connect(); } catch (e) {}
  return true;
}

// v220: celebratory "your progress was saved" confirmation, shown right after a
// delighted 👍 user creates an account and their earned session is persisted.
// Opens their profile so they SEE the carried-over streak/minutes immediately.
function _showProgressSaved(mins) {
  try {
    const prev = document.getElementById('progress-saved-ov');
    if (prev) prev.remove();
    const ov = document.createElement('div');
    ov.id = 'progress-saved-ov';
    ov.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:10000',
      'display:flex', 'align-items:center', 'justify-content:center',
      'padding:20px', 'background:rgba(7,2,15,.72)',
      'backdrop-filter:blur(8px)', '-webkit-backdrop-filter:blur(8px)',
      'font:14px/1.5 -apple-system,BlinkMacSystemFont,Inter,Segoe UI,Roboto,sans-serif',
    ].join(';');
    const m = Math.max(1, Math.round(mins || 1));
    ov.innerHTML =
      '<div style="max-width:360px;width:92%;text-align:center;background:linear-gradient(165deg,#1c1330,#241546);' +
      'border:1px solid rgba(192,97,255,.35);border-radius:20px;padding:28px 24px;box-shadow:0 24px 70px rgba(0,0,0,.55);color:#f4f1fb">' +
        '<div style="font-size:46px;margin-bottom:8px">🎉</div>' +
        '<div style="font-size:20px;font-weight:800;margin-bottom:8px">Progress saved!</div>' +
        '<div style="font-size:14px;color:#c9c0e0;margin-bottom:20px">Your ' + m + '-minute session is in your account — ' +
        'your streak has started. Come back tomorrow to keep it going. 🔥</div>' +
        '<button id="psv-see" style="width:100%;background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;border:0;border-radius:12px;padding:13px;font-weight:800;font-size:15px;cursor:pointer;margin-bottom:8px">See my progress →</button>' +
        '<button id="psv-close" style="width:100%;background:none;border:0;color:#9b93b4;font-size:13px;cursor:pointer;padding:6px">Keep dancing</button>' +
      '</div>';
    document.body.appendChild(ov);
    requestAnimationFrame(() => { try { _fireConfetti(); } catch (e) {} });
    const close = () => { try { ov.remove(); } catch (e) {} };
    ov.querySelector('#psv-close').addEventListener('click', close);
    ov.querySelector('#psv-see').addEventListener('click', () => {
      close();
      try { (window.__openProfilePage || openProfilePage)(); } catch (e) {}
    });
    ov.addEventListener('click', (e) => { if (e.target === ov) close(); });
  } catch (e) {}
}

// Tiny dependency-free confetti burst for the progress-saved celebration.
function _fireConfetti() {
  try {
    const N = 44;
    const colors = ['#c061ff', '#ec4899', '#f59e0b', '#22c55e', '#38bdf8', '#fff'];
    const box = document.createElement('div');
    box.style.cssText = 'position:fixed;inset:0;z-index:10001;pointer-events:none;overflow:hidden';
    document.body.appendChild(box);
    for (let i = 0; i < N; i++) {
      const p = document.createElement('div');
      const size = 6 + Math.random() * 6;
      const left = 50 + (Math.random() * 40 - 20);
      const dx = (Math.random() * 240 - 120) + 'px';
      const dur = 900 + Math.random() * 900;
      p.style.cssText =
        'position:absolute;top:38%;left:' + left + '%;width:' + size + 'px;height:' + size + 'px;' +
        'background:' + colors[i % colors.length] + ';border-radius:' + (Math.random() < .5 ? '50%' : '2px') + ';' +
        'opacity:1;transform:translate(0,0) rotate(0deg);' +
        'transition:transform ' + dur + 'ms cubic-bezier(.2,.7,.3,1), opacity ' + dur + 'ms ease-out';
      box.appendChild(p);
      requestAnimationFrame(() => {
        p.style.transform = 'translate(' + dx + ',' + (60 + Math.random() * 40) + 'vh) rotate(' + (Math.random() * 720 - 360) + 'deg)';
        p.style.opacity = '0';
      });
    }
    setTimeout(() => { try { box.remove(); } catch (e) {} }, 2200);
  } catch (e) {}
}
// v226: expose the confetti so session_summary.js can celebrate a COMPLETED
// session (the "you did it" moment that makes finishing feel rewarding and
// nudges the save-progress signup right after).
try { window.__fireConfetti = _fireConfetti; } catch (e) {}
// Unsigned-in visitors get a time-boxed free preview (~2.5 min), tracked
// via a PERSISTED first-seen timestamp (not a per-page-load timer) so
// refreshing the page can't reset the clock. Once the wall triggers, the
// composer / quick-action chips / session-start buttons go dim + inert
// until the user signs in — but voice (mic + live-voice) is DELIBERATELY
// left completely untouched: neither this wall nor the existing WS
// reconnect in _applyAuthSuccess ever touches _gv/voice/mic state (voice
// runs on its own independent connection — see _startS2S), so an
// in-progress or newly-started voice conversation keeps working straight
// through the sign-in modal appearing AND through a successful sign-in.
const ANON_WALL_MS = 150000; // 2.5 minutes
const ANON_FIRST_SEEN_KEY = 'coach.anon.firstSeenAt';
let _anonWallActive = false;

function _anonWallLockedEls() {
  const ids = ['composer', 'quick', 'session-start'];
  return ids.map((id) => document.getElementById(id)).filter(Boolean);
}

function _triggerAnonWall() {
  if (_anonWallActive || _isSignedIn()) return;
  // v226: NEVER interrupt an in-progress guided session. Hitting a signup wall
  // mid-dance is the worst possible moment — it kills the very completion we're
  // trying to drive. If the dancer is in a session, defer the wall and re-check
  // after it ends (the summary card then carries the save-progress invite, a
  // far better conversion point than a hard mid-session block).
  if (window.__inSession) {
    setTimeout(_triggerAnonWall, 20000);
    return;
  }
  _anonWallActive = true;
  _anonWallLockedEls().forEach((el) => {
    el.classList.add('anon-locked');
    if (!el.querySelector(':scope > .anon-lock-overlay')) {
      const ov = document.createElement('div');
      ov.className = 'anon-lock-overlay';
      ov.addEventListener('click', () => {
        try { showSignInModal('Sign in to keep chatting and dancing with me.'); } catch (e) {}
      });
      el.appendChild(ov);
    }
  });
  const note = document.getElementById('anon-wall-note');
  if (note) note.classList.add('show');
  try {
    showSignInModal('Free to try \u2014 no login needed to start. Sign in only to '
                    + 'save your progress and keep dancing. Your voice chat keeps '
                    + 'running either way.');
  } catch (e) {}
}

function _releaseAnonWall() {
  if (!_anonWallActive) return;
  _anonWallActive = false;
  _anonWallLockedEls().forEach((el) => {
    el.classList.remove('anon-locked');
    const ov = el.querySelector(':scope > .anon-lock-overlay');
    if (ov) ov.remove();
  });
  const note = document.getElementById('anon-wall-note');
  if (note) note.classList.remove('show');
}

(function initAnonWall() {
  if (_isSignedIn()) return;
  let firstSeen = 0;
  try {
    firstSeen = parseInt(localStorage.getItem(ANON_FIRST_SEEN_KEY) || '0', 10);
    if (!firstSeen) {
      firstSeen = Date.now();
      localStorage.setItem(ANON_FIRST_SEEN_KEY, String(firstSeen));
    }
  } catch (e) { firstSeen = Date.now(); }
  const remaining = ANON_WALL_MS - (Date.now() - firstSeen);
  // If the trial was already used up on a prior visit, gate almost
  // immediately, but give the avatar's greeting a moment to play first.
  setTimeout(_triggerAnonWall, remaining > 0 ? remaining : 1500);
})();

document.getElementById('anon-wall-signin')?.addEventListener('click', () => {
  try { showSignInModal('Sign in to keep chatting and dancing with me.'); } catch (e) {}
});

// One injected stylesheet so the auth UI uses real :focus / :hover /
// ::placeholder states (inline styles can't) and reads like a standard
// sign-in dialog instead of an ad-hoc box.
let _authStylesInjected = false;
function _ensureAuthStyles() {
  if (_authStylesInjected) return;
  _authStylesInjected = true;
  const css = `
  .authm-backdrop{position:fixed;inset:0;z-index:9700;display:none;
    align-items:center;justify-content:center;padding:16px;
    background:rgba(6,4,14,.6);backdrop-filter:blur(6px);
    -webkit-backdrop-filter:blur(6px);}
  .authm-backdrop.show{display:flex;}
  .authm-card{width:100%;max-width:380px;box-sizing:border-box;
    background:#181527;border:1px solid rgba(255,255,255,.08);
    border-radius:16px;padding:24px;color:#f4f1fb;
    font-family:inherit;box-shadow:0 20px 60px rgba(0,0,0,.5);}
  .authm-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;}
  .authm-title{margin:0;font-size:20px;font-weight:650;letter-spacing:-.01em;}
  .authm-close{background:none;border:none;color:#9b93b4;font-size:24px;
    line-height:1;cursor:pointer;padding:4px;border-radius:8px;transition:background .15s,color .15s;}
  .authm-close:hover{background:rgba(255,255,255,.06);color:#fff;}
  .authm-sub{margin:0 0 18px;font-size:13.5px;color:#a59fbd;line-height:1.5;}
  .authm-field{margin-bottom:14px;}
  .authm-label{display:block;font-size:12.5px;font-weight:550;color:#b7b0cf;margin-bottom:6px;}
  .authm-input{width:100%;box-sizing:border-box;padding:11px 13px;font-size:14.5px;
    color:#fff;background:#0f0d1c;border:1px solid rgba(255,255,255,.12);
    border-radius:10px;outline:none;transition:border-color .15s,box-shadow .15s;font-family:inherit;}
  .authm-input::placeholder{color:#6b6485;}
  .authm-input:focus{border-color:#8b5cf6;box-shadow:0 0 0 3px rgba(139,92,246,.25);}
  .authm-err{display:none;color:#ff8a9b;font-size:13px;margin:-4px 0 12px;line-height:1.4;}
  .authm-err.show{display:block;}
  .authm-submit{width:100%;padding:12px;border:none;border-radius:10px;
    font-size:15px;font-weight:650;cursor:pointer;color:#fff;font-family:inherit;
    background:linear-gradient(135deg,#7c3aed,#db2777);transition:filter .15s,opacity .15s;}
  .authm-submit:hover{filter:brightness(1.08);}
  .authm-submit:disabled{opacity:.65;cursor:default;}
  .authm-foot{text-align:center;margin-top:18px;font-size:13.5px;color:#a59fbd;}
  .authm-link{color:#b794ff;text-decoration:none;font-weight:600;cursor:pointer;}
  .authm-link:hover{text-decoration:underline;}
  /* account dropdown menu (signed-in) */
  .acct-menu{position:fixed;z-index:9700;min-width:200px;background:#181527;
    border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:6px;
    box-shadow:0 16px 44px rgba(0,0,0,.5);display:none;font-family:inherit;}
  .acct-menu.show{display:block;}
  .acct-id{padding:9px 12px 10px;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:6px;}
  .acct-id .nm{font-size:13.5px;font-weight:600;color:#f4f1fb;}
  .acct-id .em{font-size:12px;color:#9b93b4;margin-top:2px;word-break:break-all;}
  .acct-stats{display:flex;gap:6px;padding:4px 6px 10px;}
  .acct-stat{flex:1;text-align:center;background:rgba(255,255,255,.04);border-radius:9px;padding:8px 4px;}
  .acct-stat .v{font-size:17px;font-weight:700;color:#f4f1fb;line-height:1;}
  .acct-stat .l{font-size:10.5px;color:#9b93b4;margin-top:4px;}
  .acct-item{display:block;width:100%;text-align:left;background:none;border:none;
    color:#e9e5f5;font-size:14px;padding:9px 12px;border-radius:8px;cursor:pointer;font-family:inherit;}
  .acct-item:hover{background:rgba(255,255,255,.07);}
  .acct-item.danger{color:#ff8a9b;}
  `;
  const s = document.createElement('style');
  s.textContent = css;
  document.head.appendChild(s);
}

function _refreshAccountBtn() {
  try {
    const btn = $('signin'); if (!btn) return;
    btn.textContent = _isSignedIn() ? 'Account' : 'Sign in';
  } catch (e) {}
}

let _authModalEl = null;
function _buildAuthModal() {
  if (_authModalEl) return _authModalEl;
  _ensureAuthStyles();
  const wrap = document.createElement('div');
  wrap.id = 'auth-modal';
  wrap.className = 'authm-backdrop';
  wrap.setAttribute('role', 'dialog');
  wrap.setAttribute('aria-modal', 'true');
  wrap.setAttribute('aria-labelledby', 'auth-title');
  wrap.innerHTML =
    '<div class="authm-card" role="document">' +
      '<div class="authm-head">' +
        '<h2 id="auth-title" class="authm-title">Sign in</h2>' +
        '<button id="auth-x" class="authm-close" aria-label="Close">\u00d7</button>' +
      '</div>' +
      '<p id="auth-sub" class="authm-sub">Free to try — no login needed to start. Sign in to save your progress.</p>' +
      '<div id="auth-name-row" class="authm-field" style="display:none;">' +
        '<label class="authm-label" for="auth-name">Name</label>' +
        '<input id="auth-name" class="authm-input" type="text" autocomplete="name" placeholder="Your name">' +
      '</div>' +
      '<div class="authm-field">' +
        '<label class="authm-label" for="auth-email">Email</label>' +
        '<input id="auth-email" class="authm-input" type="email" autocomplete="email" placeholder="you@example.com">' +
      '</div>' +
      '<div class="authm-field">' +
        '<label class="authm-label" for="auth-pass">Password</label>' +
        '<input id="auth-pass" class="authm-input" type="password" autocomplete="current-password" placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022">' +
      '</div>' +
      '<div id="auth-err" class="authm-err" role="alert"></div>' +
      '<button id="auth-submit" class="authm-submit" type="button">Sign in</button>' +
      '<div class="authm-foot">' +
        '<span id="auth-toggle-text">New here?</span> ' +
        '<a id="auth-toggle" class="authm-link" role="button" tabindex="0">Create an account</a>' +
      '</div>' +
    '</div>';
  document.body.appendChild(wrap);
  _authModalEl = wrap;
  // Close on backdrop click / × / Escape — but NOT on card click.
  wrap.addEventListener('click', (e) => { if (e.target === wrap) _hideAuthModal(); });
  wrap.querySelector('#auth-x').addEventListener('click', _hideAuthModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && wrap.classList.contains('show')) _hideAuthModal();
  });
  wrap.querySelectorAll('input').forEach((inp) => {
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') _submitAuth(); });
  });
  wrap.querySelector('#auth-submit').addEventListener('click', _submitAuth);
  const tog = wrap.querySelector('#auth-toggle');
  const flip = (e) => { e.preventDefault(); _setAuthMode(wrap.__mode === 'login' ? 'register' : 'login'); };
  tog.addEventListener('click', flip);
  tog.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') flip(e); });
  return wrap;
}

function _setAuthMode(mode) {
  const w = _authModalEl; if (!w) return;
  w.__mode = mode;
  const isReg = mode === 'register';
  w.querySelector('#auth-title').textContent      = isReg ? 'Create account' : 'Dance.AI \u2014 learn any dance';
  w.querySelector('#auth-submit').textContent     = isReg ? 'Sign up' : 'Sign in';
  w.querySelector('#auth-name-row').style.display = isReg ? 'block' : 'none';
  w.querySelector('#auth-toggle-text').textContent = isReg ? 'Already have an account?' : 'New here?';
  w.querySelector('#auth-toggle').textContent      = isReg ? 'Sign in' : 'Create an account';
  w.querySelector('#auth-pass').setAttribute('autocomplete', isReg ? 'new-password' : 'current-password');
  _setAuthErr('');
}

function _setAuthErr(msg) {
  const el = _authModalEl && _authModalEl.querySelector('#auth-err');
  if (!el) return;
  el.textContent = msg || '';
  el.classList.toggle('show', !!msg);
}

function _hideAuthModal() {
  if (_authModalEl) _authModalEl.classList.remove('show');
}

async function _submitAuth() {
  const w = _authModalEl; if (!w) return;
  const mode  = w.__mode || 'login';
  const email = (w.querySelector('#auth-email').value || '').trim();
  const pass  = w.querySelector('#auth-pass').value || '';
  const name  = (w.querySelector('#auth-name').value || '').trim();
  if (!email || !pass) { _setAuthErr('Enter your email and password.'); return; }
  if (mode === 'register' && pass.length < 6) {
    _setAuthErr('Password must be at least 6 characters.'); return;
  }
  const btn = w.querySelector('#auth-submit');
  const orig = btn.textContent;
  btn.disabled = true; btn.textContent = (mode === 'register' ? 'Creating…' : 'Signing in…');
  _setAuthErr('');
  try {
    const path = mode === 'register' ? '/api/auth/register' : '/api/auth/login';
    const payload = mode === 'register'
      ? { email, password: pass, name: name || undefined }
      : { email, password: pass };
    const r = await fetch(APP_BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    let data = {};
    try { data = await r.json(); } catch (e) {}
    if (r.ok && _applyAuthSuccess(data)) {
      _hideAuthModal();
      const pending = window.__pendingAuthAction;
      const pendingAt = window.__pendingAuthActionAt || 0;
      window.__pendingAuthAction = null;
      window.__pendingAuthActionAt = 0;
      // v93: only auto-run the pending action (e.g. start a session) if the
      // user set it in the last 2 min by literally tapping a session chip.
      // A plain Account sign-in must NEVER auto-launch a session.
      const fresh = (Date.now() - pendingAt) < 120000;
      if (fresh && typeof pending === 'function') setTimeout(() => { try { pending(); } catch (e) {} }, 400);
      return;
    }
    if (r.status === 401) {
      _setAuthErr('Invalid email or password.');
    } else if (r.status === 400 || r.status === 409) {
      const up = data && (data.error || data.message);
      _setAuthErr(mode === 'register'
        ? (up || 'Could not create account — that email may already be registered.')
        : 'Invalid email or password.');
    } else {
      _setAuthErr((data && (data.error || data.message)) || 'Something went wrong. Please try again.');
    }
  } catch (e) {
    _setAuthErr('Network error — please try again.');
  } finally {
    btn.disabled = false; btn.textContent = orig;
  }
}

// Public entry point used by the auth-gate handlers + the session buttons.
function showSignInModal(message, mode) {
  const w = _buildAuthModal();
  _setAuthMode(mode || 'login');
  if (message) w.querySelector('#auth-sub').textContent = message;
  w.classList.add('show');
  setTimeout(() => { try { w.querySelector('#auth-email').focus(); } catch (e) {} }, 50);
}
try { window.__showSignInModal = showSignInModal; } catch (e) {}

// Account dropdown menu (standard pattern instead of a window.confirm).
let _acctMenuEl = null;
function _closeAcctMenu() {
  if (_acctMenuEl) _acctMenuEl.classList.remove('show');
}
function _toggleAccountMenu() {
  _ensureAuthStyles();
  const btn = $('signin'); if (!btn) return;
  if (!_acctMenuEl) {
    _acctMenuEl = document.createElement('div');
    _acctMenuEl.className = 'acct-menu';
    _acctMenuEl.setAttribute('role', 'menu');
    document.body.appendChild(_acctMenuEl);
    document.addEventListener('click', (e) => {
      if (_acctMenuEl && _acctMenuEl.classList.contains('show') &&
          e.target !== btn && !_acctMenuEl.contains(e.target)) _closeAcctMenu();
    });
  }
  if (_acctMenuEl.classList.contains('show')) { _closeAcctMenu(); return; }
  let name = '', email = '';
  try { name = localStorage.getItem('user_name') || ''; email = localStorage.getItem('user_email') || ''; } catch (e) {}
  _acctMenuEl.innerHTML =
    (name || email
      ? '<div class="acct-id">' +
          (name ? '<div class="nm">' + _esc(name) + '</div>' : '<div class="nm">Signed in</div>') +
          (email ? '<div class="em">' + _esc(email) + '</div>' : '') +
        '</div>'
      : '<div class="acct-id"><div class="nm">Signed in</div></div>') +
    '<div id="acct-stats" class="acct-stats">' +
      '<div class="acct-stat"><div class="v" id="st-streak">\u2013</div><div class="l">\ud83d\udd25 streak</div></div>' +
      '<div class="acct-stat"><div class="v" id="st-mins">\u2013</div><div class="l">\u23f1 mins</div></div>' +
      '<div class="acct-stat"><div class="v" id="st-sess">\u2013</div><div class="l">\u2713 sessions</div></div>' +
    '</div>' +
    '<a class="acct-item" id="acct-contact" role="menuitem" href="mailto:you@example.com?subject=MotionMind%20support" style="display:block;text-decoration:none;">\u2709\ufe0f Contact us</a>' +
    '<button class="acct-item danger" id="acct-signout" role="menuitem">Sign out</button>';
  _acctMenuEl.querySelector('#acct-signout').addEventListener('click', () => {
    _closeAcctMenu();
    try {
      localStorage.removeItem('token'); localStorage.removeItem('refresh_token');
      localStorage.removeItem('user_name'); localStorage.removeItem('user_email');
    } catch (e) {}
    location.reload();
  });
  // v69: pull the streak / minutes / sessions from the server and fill
  // the stat row (they're tracked per-user but were never surfaced).
  (async () => {
    try {
      let tok = '';
      try { tok = localStorage.getItem('token') || ''; } catch (e) {}
      if (!tok) return;
      const r = await fetch(APP_BASE + '/api/me/progress', {
        headers: { Authorization: 'Bearer ' + tok },
      });
      if (!r.ok) return;
      const d = await r.json();
      const p = (d && d.progress) || {};
      const setv = (id, val) => { const el = _acctMenuEl && _acctMenuEl.querySelector('#' + id); if (el) el.textContent = String(val); };
      setv('st-streak', (p.current_streak_days || 0));
      setv('st-mins',   (p.total_minutes || 0));
      setv('st-sess',   (p.sessions_completed || 0));
      // Personalise the name if the server knows it and we didn't.
      if (d.user && d.user.name && !name) {
        const nm = _acctMenuEl.querySelector('.acct-id .nm');
        if (nm) nm.textContent = d.user.name;
      }
    } catch (e) { /* leave dashes */ }
  })();
  // Anchor under the button, right-aligned.
  const r = btn.getBoundingClientRect();
  _acctMenuEl.style.top = (r.bottom + 8) + 'px';
  _acctMenuEl.style.right = Math.max(8, window.innerWidth - r.right) + 'px';
  _acctMenuEl.style.left = 'auto';
  _acctMenuEl.classList.add('show');
}

function _esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ─── v194: full-screen PROFILE PAGE ───────────────────────────────────
// A real account surface (replaces the cramped dropdown). Fetches
// /api/me/progress and renders identity, streak/stats, dance-journey
// history, personalization, and a contact card. Opened from the top-right
// "Account" button for signed-in users.
const STYLE_EMOJI = {
  'Hip-Hop': '🔥', 'House': '🏠', 'Locking': '🔒', 'Waacking': '👐',
  'Breaking': '🌀', 'Popping': '🤖', 'Krump': '💥', 'Jazz': '🎷',
  'Warm-up': '🧘',
};

function _relativeDay(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso + (iso.length <= 10 ? 'T00:00:00' : ''));
    const today = new Date();
    const dayMs = 24 * 60 * 60 * 1000;
    const diff = Math.floor((today.setHours(0, 0, 0, 0) -
                             new Date(d).setHours(0, 0, 0, 0)) / dayMs);
    if (diff === 0) return 'Today';
    if (diff === 1) return 'Yesterday';
    if (diff > 1 && diff < 7) return `${diff} days ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (e) { return iso; }
}

function _renderProfile(data) {
  const user = (data && data.user) || {};
  const p = (data && data.progress) || {};
  const prof = (data && data.profile) || {};
  const name = user.name || (() => { try { return localStorage.getItem('user_name') || ''; } catch (e) { return ''; } })() || 'Dancer';
  const email = user.email || (() => { try { return localStorage.getItem('user_email') || ''; } catch (e) { return ''; } })() || '';
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  const setHTML = (id, v) => { const el = document.getElementById(id); if (el) el.innerHTML = v; };

  set('pp-name', name);
  set('pp-email', email);
  set('pp-avatar', (name.trim()[0] || '🙂').toUpperCase());
  if (user.created_at) {
    set('pp-since', 'Member since ' + _relativeDay(String(user.created_at).slice(0, 10)));
  } else { set('pp-since', ''); }

  set('pp-streak', p.current_streak_days || 0);
  set('pp-sessions', p.sessions_completed || 0);
  set('pp-minutes', Math.round(p.total_minutes || 0));

  const recent = Array.isArray(p.recent_sessions) ? p.recent_sessions.slice() : [];
  // "This week" = sessions in the last 7 days.
  const now = Date.now();
  const weekCount = recent.filter((s) => {
    const t = s.ts ? Date.parse(s.ts) : (s.date ? Date.parse(s.date) : NaN);
    return !isNaN(t) && (now - t) < 7 * 24 * 60 * 60 * 1000;
  }).length;
  set('pp-week', weekCount);

  if (p.last_session_date) {
    set('pp-lastseen', 'Last: ' + _relativeDay(p.last_session_date));
  } else {
    set('pp-lastseen', '');
  }

  // History (newest first)
  const hist = document.getElementById('pp-history');
  if (hist) {
    if (!recent.length) {
      hist.innerHTML = '<div class="pp-empty">No sessions yet — pick a style and start moving!</div>';
    } else {
      const rows = recent.slice().reverse().map((s) => {
        const style = s.style || 'Dance session';
        const emoji = STYLE_EMOJI[style] || '💃';
        const when = _relativeDay(s.date || (s.ts || '').slice(0, 10));
        const mins = Math.round(s.minutes || 0);
        return '<div class="pp-hrow">' +
          `<div class="pp-hemoji">${emoji}</div>` +
          '<div class="pp-hmain">' +
            `<div class="pp-hstyle">${_esc(style)}</div>` +
            `<div class="pp-hdate">${_esc(when)}</div>` +
          '</div>' +
          `<div class="pp-hmin">${mins} min</div>` +
        '</div>';
      }).join('');
      hist.innerHTML = rows;
    }
  }

  // Personalization (best-effort from profile; falls back to inferred).
  const goalMap = { dance: 'Learn to dance', fitness: 'Stay active', fun: 'Have fun', choreography: 'Choreography', social: 'Social / confidence' };
  const lvlMap = { beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' };
  const goal = prof.goal || (Array.isArray(prof.fun_focus) ? prof.fun_focus[0] : '') || '';
  set('pp-goal', goalMap[goal] || (goal ? _titleCase(goal) : '—'));
  set('pp-level', lvlMap[prof.level] || (prof.level ? _titleCase(prof.level) : '—'));
  // Favourite style = most frequent in history, else profile preference.
  let favStyle = prof.favorite_style || prof.style || '';
  if (!favStyle && recent.length) {
    const counts = {};
    recent.forEach((s) => { if (s.style && s.style !== 'Warm-up') counts[s.style] = (counts[s.style] || 0) + 1; });
    favStyle = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0] || '';
  }
  set('pp-favstyle', favStyle ? _titleCase(favStyle) : '—');
  const len = prof.session_minutes || prof.session_length || '';
  set('pp-favlen', len ? (String(len).replace(/[^0-9]/g, '') || len) + ' min' : '—');
}

function _titleCase(s) {
  return String(s).replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function openProfilePage() {
  const page = document.getElementById('profile-page');
  if (!page) return;
  page.hidden = false;
  page.setAttribute('aria-hidden', 'false');
  // Render immediately with whatever we know locally, then refresh from API.
  _renderProfile({});
  (async () => {
    let tok = '';
    try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    if (!tok) return;
    try {
      const r = await fetch(APP_BASE + '/api/me/progress', {
        headers: { Authorization: 'Bearer ' + tok },
      });
      if (!r.ok) return;
      const d = await r.json();
      _renderProfile(d);
      // cache name/email for offline render next time
      try {
        if (d.user && d.user.name) localStorage.setItem('user_name', d.user.name);
        if (d.user && d.user.email) localStorage.setItem('user_email', d.user.email);
      } catch (e) {}
    } catch (e) { /* leave locally-rendered values */ }
  })();
}

function closeProfilePage() {
  const page = document.getElementById('profile-page');
  if (!page) return;
  page.setAttribute('aria-hidden', 'true');
  page.hidden = true;
}

function _profileSignOut() {
  try {
    localStorage.removeItem('token'); localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_name'); localStorage.removeItem('user_email');
  } catch (e) {}
  location.reload();
}

(function wireProfilePage() {
  const back = document.getElementById('pp-back');
  if (back) back.addEventListener('click', closeProfilePage);
  const so1 = document.getElementById('pp-signout');
  if (so1) so1.addEventListener('click', _profileSignOut);
  const so2 = document.getElementById('pp-signout-2');
  if (so2) so2.addEventListener('click', _profileSignOut);
  // Esc closes the page.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const page = document.getElementById('profile-page');
      if (page && !page.hidden) closeProfilePage();
    }
  });
})();
try { window.__openProfilePage = openProfilePage; } catch (e) {}

// ─── v195: LEARN panel — structured Hip-Hop + House lessons ───────────
// Turns the app from "random session" into an academy. Fetches
// /api/curriculum + the user's /api/me/lessons progress, renders two
// tracks of foundational-move lessons, and on start plays the move's
// demo clip then kicks the user into a drill (existing session/feedback).
(function initLearn() {
  const panel = document.getElementById('learn-panel');
  const openBtn = document.getElementById('learn-btn');
  if (!panel || !openBtn) return;
  const tracksEl = document.getElementById('lp-tracks');
  const detail = document.getElementById('lp-detail');
  const detailScroll = document.getElementById('lp-detail-scroll');
  const detailTrack = document.getElementById('lp-detail-track');

  let _curriculum = null;
  let _progress = {};   // { lesson_id: {status, best_score, worst_keypoint, note} }

  const BADGE = {
    '': ['New', 'lp-b-new'], learning: ['Learning', 'lp-b-learning'],
    practiced: ['Practiced', 'lp-b-practiced'], mastered: ['Mastered', 'lp-b-mastered'],
  };
  const RANK = { '': 0, learning: 1, practiced: 2, mastered: 3 };

  function openPanel(opts) {
    opts = opts || {};
    panel.hidden = false;
    panel.setAttribute('aria-hidden', 'false');
    requestAnimationFrame(() => panel.classList.remove('lp-hidden'));
    if (detail) detail.hidden = true;
    // v197: tell the coach's brain the student wants to LEARN so its next
    // reply is a proactive teacher/navigator (system prompt learning block).
    if (!opts.fromLLM) _sendUiEvent('opened_learn');
    ensureData().then(() => {
      // If a specific lesson was requested (e.g. by the LLM), open it once
      // curriculum data is ready.
      if (opts.lessonId) _openLessonById(opts.lessonId);
    });
  }
  function closePanel() {
    panel.classList.add('lp-hidden');
    panel.setAttribute('aria-hidden', 'true');
    setTimeout(() => { panel.hidden = true; }, 320);
    _sendUiEvent('closed_learn');
  }

  // Fire a lightweight context signal to the coach WS (no LLM turn — the
  // server just flips a flag read by the system prompt on the next turn).
  function _sendUiEvent(event) {
    try {
      const payload = { type: 'ui_event', event };
      if (ws && ws.readyState === 1) ws.send(JSON.stringify(payload));
      else { window.__wsQueue = window.__wsQueue || []; window.__wsQueue.push(payload); }
    } catch (e) {}
  }

  // Open a specific lesson by id (used by the LLM's open_lesson tool).
  function _openLessonById(lessonId) {
    if (!_curriculum) return;
    for (const track of _curriculum.tracks) {
      const les = (track.lessons || []).find((l) => l.id === lessonId);
      if (les) { openLesson(track, les); return; }
    }
  }

  async function ensureData() {
    if (!_curriculum) {
      try {
        const r = await fetch(APP_BASE + '/api/curriculum');
        _curriculum = await r.json();
      } catch (e) { _curriculum = { tracks: [] }; }
    }
    await refreshProgress();
    renderTracks();
  }

  async function refreshProgress() {
    let tok = '';
    try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    if (!tok) { _progress = {}; return; }
    try {
      const r = await fetch(APP_BASE + '/api/me/lessons', {
        headers: { Authorization: 'Bearer ' + tok },
      });
      const d = await r.json();
      _progress = (d && d.lessons) || {};
    } catch (e) { _progress = {}; }
  }

  function trackPct(track) {
    const total = track.lessons.length || 1;
    let done = 0;
    for (const les of track.lessons) {
      const st = (_progress[les.id] || {}).status || '';
      if (RANK[st] >= RANK.practiced) done += 1;
    }
    return Math.round((done / total) * 100);
  }

  function renderTracks() {
    if (!tracksEl || !_curriculum) return;
    tracksEl.innerHTML = '';
    for (const track of _curriculum.tracks) {
      const wrap = document.createElement('div');
      wrap.className = 'lp-track';
      const pct = trackPct(track);
      let html =
        '<div class="lp-track-head">' +
          `<div class="lp-track-emoji">${track.emoji || '💃'}</div>` +
          '<div>' +
            `<div class="lp-track-name">${_esc(track.name)}</div>` +
            `<div class="lp-track-tag">${_esc(track.tagline || '')}</div>` +
          '</div>' +
        '</div>' +
        `<div class="lp-track-bar"><i style="width:${pct}%"></i></div>`;
      wrap.innerHTML = html;
      track.lessons.forEach((les, i) => {
        const prog = _progress[les.id] || {};
        const [label, cls] = BADGE[prog.status || ''] || BADGE[''];
        const btn = document.createElement('button');
        btn.className = 'lp-lesson';
        btn.type = 'button';
        btn.innerHTML =
          `<div class="lp-lesson-emoji">${les.emoji || '•'}</div>` +
          '<div class="lp-lesson-main">' +
            `<div class="lp-lesson-title">${i + 1}. ${_esc(les.title)}</div>` +
            `<div class="lp-lesson-sub">${_esc(les.level || '')} · ${_esc(les.one_liner || '')}</div>` +
          '</div>' +
          `<span class="lp-lesson-badge ${cls}">${label}</span>`;
        btn.addEventListener('click', () => openLesson(track, les));
        wrap.appendChild(btn);
      });
      tracksEl.appendChild(wrap);
    }
  }

  function openLesson(track, les) {
    if (!detail || !detailScroll) return;
    detailTrack.textContent = track.name;
    const prog = _progress[les.id] || {};
    const m = les.music || {};
    const bpm = Array.isArray(m.bpm) ? `${m.bpm[0]}–${m.bpm[1]} BPM` : '';
    let lastNote = '';
    if (prog.worst_keypoint) {
      lastNote = `<div class="lp-d-lastnote">Last time your <b>${_esc(prog.worst_keypoint)}</b> needed work — let's clean that up.</div>`;
    } else if (prog.note) {
      lastNote = `<div class="lp-d-lastnote">${_esc(prog.note)}</div>`;
    }
    detailScroll.innerHTML =
      `<div class="lp-d-emoji">${les.emoji || '•'}</div>` +
      `<div class="lp-d-title">${_esc(les.title)}</div>` +
      `<div class="lp-d-one">${_esc(les.one_liner || '')}</div>` +
      lastNote +
      // v219: CTA FIRST. Users complained the "Learn it" button was buried under
      // a wall of text ("so many things come, so confusing"). Put the primary
      // action right at the top so one tap starts teaching; the reference detail
      // (what/why/how) sits below for anyone who wants to read it.
      '<div class="lp-d-actions lp-d-actions-top">' +
        '<button class="lp-d-btn" id="lp-demo">▶ Learn it — step by step</button>' +
        '<button class="lp-d-btn secondary" id="lp-drill">🎯 Practice with feedback</button>' +
      '</div>' +
      section('What it is', `<div class="lp-d-body">${_esc(les.what || '')}</div>`) +
      section('Why it matters', `<div class="lp-d-body">${_esc(les.purpose || '')}</div>`) +
      section('How to do it', '<ul class="lp-d-cues">' +
        (les.cues || []).map((c) => `<li>${_esc(c)}</li>`).join('') + '</ul>') +
      section('The one rule (invariant)',
        `<div class="lp-d-invariant">⚠️ ${_esc(les.invariant || '')}</div>`) +
      section('Music', '<div class="lp-d-music">' +
        (m.feel ? `<span class="lp-d-chip">${_esc(m.feel)}</span>` : '') +
        (bpm ? `<span class="lp-d-chip">${bpm}</span>` : '') +
        (m.hit ? `<span class="lp-d-chip">${_esc(m.hit)}</span>` : '') +
        '</div>');
    detail.hidden = false;
    detailScroll.scrollTop = 0;
    document.getElementById('lp-demo').addEventListener('click', () => playDemo(track, les));
    document.getElementById('lp-drill').addEventListener('click', () => startDrill(track, les));
  }

  function section(title, inner) {
    return `<div class="lp-d-sec"><div class="lp-d-h">${title}</div>${inner}</div>`;
  }

  // Play the move's demo clip on the avatar (resolve to a real clip; fall
  // back to first clip of the genre if the specific id isn't in catalog).
  // v213: a "Watch the move" that only LOOPED the clip taught nothing and
  // showed no steps pane — the #1 confusion. Now it plays the move AND kicks
  // the step-by-step breakdown (numbered steps + side rail) via the proven
  // deeplink→break_down path, so selecting a move ALWAYS opens structured
  // teaching with the steps rail the learner can follow.
  async function playDemo(track, les) {
    try { unlockAudio(); } catch (e) {}
    _greetedThisSession = true;          // teaching, not greeting
    const clipId = await resolveClip(les.demo_clip, les.genre || track.genre);
    if (!clipId) { try { addMsg('Demo clip unavailable right now.', 'sys'); } catch (e) {} return; }
    closePanel();
    // Narrate the move, then play the clip.
    try {
      addMsg(`🎬 ${les.title} — ${les.one_liner} Remember the one rule: ${les.invariant}`, 'coach');
    } catch (e) {}
    try { loadAndPlayClip(clipId, { loop: true }); } catch (e) {}
    markProgress(les.id, 'learning');
    // Kick the structured step-by-step breakdown so the steps rail appears and
    // the coach teaches each part slowly, then combines — like a real class.
    // v223: do NOT force the chat drawer open — the coach's lines already show
    // as floating stage bubbles, and auto-opening the full chat log covers the
    // avatar (users complained it "opens for no reason").
    try {
      sendUser(`Teach me ${les.title} step by step, from the top — break it into parts.`, 'deeplink');
    } catch (e) {}
  }

  // Start a drill: begin a guided session in the lesson's genre so the
  // user practices with music + the coach, then (if they use the camera)
  // the existing 2D feedback scores them.
  function startDrill(track, les) {
    try { unlockAudio(); } catch (e) {}
    _greetedThisSession = true;
    closePanel();
    // Set the analyze reference so "record & analyze" compares to this style.
    resolveClip(les.demo_clip, les.genre || track.genre).then((clipId) => {
      if (clipId) { try { window.__feedbackClipId = clipId; } catch (e) {} }
    });
    try {
      addMsg(`🎯 ${les.drill_prompt || ('Let\'s drill ' + les.title + '.')}`, 'coach');
    } catch (e) {}
    // Kick off a short guided session in this genre (5 min quick session).
    try { startGuidedSessionTemplate(`quick5_${les.genre || track.genre}`, 5); } catch (e) {}
    markProgress(les.id, 'practiced');
  }

  // Resolve a lesson's demo clip id against the live catalog; fall back to
  // the first clip of the genre (same pattern as the analyze flow).
  let _catalog = null;
  async function resolveClip(preferredId, genre) {
    if (!_catalog) {
      try {
        const r = await fetch(APP_BASE + '/api/motion/list');
        const d = await r.json();
        _catalog = d.motions || [];
      } catch (e) { _catalog = []; }
    }
    if (preferredId && _catalog.some((c) => c.id === preferredId)) return preferredId;
    const hit = _catalog.find((c) => (c.id || '').startsWith(genre || ''));
    return hit ? hit.id : preferredId;
  }

  function markProgress(lessonId, status) {
    // optimistic local update so badges move immediately
    const cur = _progress[lessonId] || {};
    if (RANK[status] >= RANK[cur.status || '']) cur.status = status;
    _progress[lessonId] = cur;
    let tok = '';
    try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    if (!tok) return;   // progress only persists for signed-in users
    try {
      fetch(APP_BASE + '/api/me/lessons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + tok },
        body: JSON.stringify({ lesson_id: lessonId, status }),
        keepalive: true,
      }).catch(() => {});
    } catch (e) {}
  }

  openBtn.addEventListener('click', () => openPanel());
  document.getElementById('lp-close').addEventListener('click', closePanel);
  document.getElementById('lp-back').addEventListener('click', () => { detail.hidden = true; renderTracks(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !panel.hidden) {
      if (!detail.hidden) { detail.hidden = true; } else { closePanel(); }
    }
  });
  try { window.__openLearn = openPanel; } catch (e) {}
  // v197: hooks the LLM's navigation tools (ui.open_learn / ui.open_lesson)
  // call so the coach can DRIVE the app for the student.
  try {
    window.__llmOpenLearn = (style) => openPanel({ fromLLM: true, style });
    window.__llmOpenLesson = (lessonId) => openPanel({ fromLLM: true, lessonId });
    window.__closeLearn = () => { try { closePanel(); } catch (e) {} };
  } catch (e) {}

  // ── one-time Learn coachmark + button pulse ─────────────────────────
  // New users don't know what "Learn" is. Pulse the CTA and float a hint
  // under it for ~4s on first visit, then it fades and never nags again.
  (function learnCoachmark() {
    let seen = false;
    try { seen = localStorage.getItem('learn.hinted') === '1'; } catch (e) {}
    if (!seen) openBtn.classList.add('lp-pulse');
    // NOTE: declare hint/timers up-front so hideHint() never touches a
    // block-scoped binding that's still in the temporal dead zone (the early
    // `return` below skipped `const hint` and made clicking Learn throw
    // "Cannot access 'hint' before initialization").
    let hint = null, _t1, _t2;
    function hideHint() {
      if (hint) hint.classList.remove('show');
      clearTimeout(_t1); clearTimeout(_t2);
    }
    // Stop pulsing + mark seen the first time they open Learn.
    openBtn.addEventListener('click', () => {
      openBtn.classList.remove('lp-pulse');
      try { localStorage.setItem('learn.hinted', '1'); } catch (e) {}
      hideHint();
    });
    if (seen) return;
    hint = document.getElementById('learn-coach');
    if (!hint) return;
    function place() {
      const r = openBtn.getBoundingClientRect();
      hint.style.top = (r.bottom + 10) + 'px';
      // right-align the hint's arrow (22px from its right) under the button
      hint.style.right = Math.max(8, window.innerWidth - r.right - 4) + 'px';
      hint.style.left = 'auto';
    }
    // Show shortly after the studio is ready (after the front door / boot).
    _t1 = setTimeout(() => {
      if (openBtn.offsetParent === null) return;   // topbar not visible yet
      place();
      hint.classList.add('show');
      window.addEventListener('resize', place);
      _t2 = setTimeout(hideHint, 4600);
    }, 1600);
    hint.addEventListener('click', () => { hideHint(); openPanel(); });
  })();
})();

// v186: skill-event capture — the (attempt, instruction, outcome) research
// corpus. Consent-gated: we only send if the user has opted into helping
// improve the coach with their (anonymized, pose-only — never video)
// movement data. `_skillConsent()` reads a localStorage flag set by the
// one-time consent notice shown before the live mirror starts.
function _skillConsent() {
  try { return localStorage.getItem('coach.skill.consent') === '1'; }
  catch (e) { return false; }
}
function _setSkillConsent(on) {
  try { localStorage.setItem('coach.skill.consent', on ? '1' : '0'); } catch (e) {}
}
function _logSkillEvent(ev) {
  if (!_skillConsent()) return;   // strict: no consent, no capture
  try {
    let tok = '';
    try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    const body = JSON.stringify(Object.assign({ consent: true }, ev));
    // keepalive so an in-flight beat survives the popup closing.
    fetch(APP_BASE + '/api/skill-event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + tok },
      body, keepalive: true,
    }).catch(() => {});
  } catch (e) { /* never break the caller */ }
}

// v182: "Share Dance.AI" — the highest-leverage distribution lever we can
// ship from this codebase alone: turn every session into a chance for the
// user to invite someone else. Uses the native Web Share sheet on mobile
// (WhatsApp/Instagram/SMS one-tap) and falls back to clipboard-copy on
// desktop. The shared link carries UTM params so /api/entry-event (v179)
// can attribute exactly how many new visitors arrive via user-to-user
// shares vs. homepage/ads/direct.
function _shareUrl() {
  const base = APP_BASE + '/';
  const abs = new URL(base, location.origin).toString();
  return abs + '?utm_source=share&utm_medium=user_share&utm_campaign=session_share';
}
async function _handleShareClick() {
  const url = _shareUrl();
  const title = 'Dance.AI — AI dance coach';
  const text = 'I\u2019m dancing with an AI coach right now \u2014 come try it, it\u2019s free:';
  try {
    if (navigator.share) {
      await navigator.share({ title, text, url });
      try { window.DanceAnalytics?.track?.('dance.share.completed', { via: 'web_share' }); } catch (e) {}
      return;
    }
  } catch (e) {
    // user cancelled the native share sheet — not an error, just stop.
    if (e && e.name === 'AbortError') return;
  }
  // Desktop / unsupported browsers: copy the link instead.
  try {
    await navigator.clipboard.writeText(url);
    const el = $('status');
    const prev = el.textContent;
    el.textContent = 'Link copied! \ud83d\udd17';
    setTimeout(() => { if (el.textContent === 'Link copied! \ud83d\udd17') el.textContent = prev; }, 2200);
    try { window.DanceAnalytics?.track?.('dance.share.completed', { via: 'clipboard' }); } catch (e) {}
  } catch (e) { /* clipboard denied — silently give up, non-critical */ }
}
$('share-btn')?.addEventListener('click', () => {
  try { window.DanceAnalytics?.track?.('dance.share.clicked', {}); } catch (e) {}
  _handleShareClick();
});

// ─── v198: top-bar overflow (⋯) menu ────────────────────────────────
(function moreMenu() {
  const btn = $('more-btn');
  const menu = $('more-menu');
  if (!btn || !menu) return;
  const close = () => {
    if (menu.hidden) return;
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  };
  const open = () => {
    menu.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
  };
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.hidden ? open() : close();
  });
  // close after picking Share / Reload (any menu button)
  menu.addEventListener('click', (e) => {
    if (e.target.closest('.mm-item')) close();
  });
  // don't close when interacting with the language <select> inside the menu
  menu.addEventListener('click', (e) => { e.stopPropagation(); });
  document.addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
})();

$('signin')?.addEventListener('click', (e) => {
  e.stopPropagation();
  if (_isSignedIn()) { openProfilePage(); return; }
  showSignInModal('Sign in to your account.');
});
(function refreshSignInLabel() {
  const apply = () => {
    try {
      const btn = $('signin'); if (!btn) return;
      btn.textContent = localStorage.getItem('token') ? 'Account' : 'Sign in';
    } catch (e) {}
  };
  apply();
  // If sign-in happens in another tab/window (studioos.fit/login),
  // the storage event fires here so we can flip the button label
  // without a full page reload.
  window.addEventListener('storage', (e) => {
    if (!e || e.key === 'token' || e.key === null) apply();
  });
  // Also pick up a token that came back as ?token=... in the URL
  // (some auth flows return via query param rather than localStorage).
  try {
    const u = new URL(location.href);
    const t = u.searchParams.get('token');
    if (t && !localStorage.getItem('token')) {
      localStorage.setItem('token', t);
      u.searchParams.delete('token');
      history.replaceState({}, '', u.toString());
      apply();
    }
  } catch (e) {}
})();

// ─── language selector (English / Hinglish / Hindi) ──────────────────
// v69: pick the coach's reply language. Switches the Azure voice locale
// AND tells the server (→ LLM system prompt) to reply in that language.
// Persisted in localStorage.
(function languageSelector() {
  const LANGS = [
    { id: 'english',  label: 'English' },
    { id: 'hinglish', label: 'Hinglish' },
    { id: 'hindi',    label: '\u0939\u093f\u0902\u0926\u0940' },
  ];
  let saved = 'hinglish';
  try { saved = localStorage.getItem('coach_lang') || 'hinglish'; } catch (e) {}
  window.__coachLang = saved;
  try { voice.setLanguage(saved); } catch (e) {}
  // Tell the server as soon as the socket opens (flushed from __wsQueue).
  if (saved !== 'english') {
    window.__wsQueue = window.__wsQueue || [];
    window.__wsQueue.push({ type: 'set_language', language: saved });
  }
  const head = $('lang-mount') || $('drawer-head');
  if (head) {
    const sel = document.createElement('select');
    sel.id = 'lang-select';
    sel.setAttribute('aria-label', 'Coach language');
    for (const l of LANGS) {
      const o = document.createElement('option');
      o.value = l.id; o.textContent = l.label;
      if (l.id === saved) o.selected = true;
      sel.appendChild(o);
    }
    // v120: language now lives in the top bar (next to Sign in), not the
    // chat drawer. If the topbar mount exists, just append; otherwise fall
    // back to the old drawer-head placement.
    if (head.id === 'lang-mount') {
      head.appendChild(sel);
    } else {
      sel.style.cssText =
        'margin-left:auto;margin-right:8px;background:#1a1726;color:#e9e5f5;' +
        'border:1px solid rgba(255,255,255,.14);border-radius:8px;padding:4px 8px;' +
        'font-size:12.5px;font-family:inherit;cursor:pointer;';
      const closeBtn = $('drawer-close');
      if (closeBtn) head.insertBefore(sel, closeBtn); else head.appendChild(sel);
    }
    sel.addEventListener('change', () => {
      const lang = sel.value || 'english';
      window.__coachLang = lang;
      try { localStorage.setItem('coach_lang', lang); } catch (e) {}
      try { voice.setLanguage(lang); } catch (e) {}
      const payload = { type: 'set_language', language: lang };
      if (ws && ws.readyState === 1) {
        try { ws.send(JSON.stringify(payload)); } catch (e) {}
      } else {
        window.__wsQueue = window.__wsQueue || [];
        window.__wsQueue.push(payload);
      }
      const names = { english: 'English', hinglish: 'Hinglish', hindi: 'Hindi' };
      try { addMsg('Coach language set to ' + (names[lang] || lang) + '.', 'sys'); } catch (e) {}
    });
  }
})();

// ─── speed slider ────────────────────────────────────────────────────
const _speed = $('speed');
const _speedVal = $('speed-val');
if (_speed) {
  _speed.addEventListener('input', () => {
    const v = parseFloat(_speed.value);
    _speedVal.textContent = v.toFixed(2).replace(/0$/, '') + '×';
    if (player) player.speed = v;
    if (window.__lastPlay) window.__lastPlay.opts = {
      ...(window.__lastPlay.opts || {}), speed: v };
  });
}

// ─── stage mirror toggle (left-right flip, not webcam) ──────────────
const _mirrorBtn = $('mirror-btn');
if (_mirrorBtn) {
  _mirrorBtn.addEventListener('click', () => {
    const on = !_mirrorBtn.classList.contains('live');
    _mirrorBtn.classList.toggle('live', on);
    if (player) player.mirror = on;
    if (window.__lastPlay) window.__lastPlay.opts = {
      ...(window.__lastPlay.opts || {}), mirror: on };
  });
}

// ─── play / pause / stop (single button toggles) ────────────────────
const _playPause = $('playpause');
function setPlayPauseLabel(paused) {
  if (!_playPause) return;
  _playPause.classList.toggle('paused', paused);
  // Icon-only on the new dock; ARIA carries the semantic.
  _playPause.textContent = paused ? '▶' : '❚❚';
  _playPause.setAttribute('aria-label', paused ? 'Play' : 'Pause');
  _playPause.title = paused ? 'Play' : 'Pause';
  _syncReopenBtn();   // Q1: reflect dancing/idle on the launcher pill
}
setPlayPauseLabel(false);
if (_playPause) {
  _playPause.addEventListener('click', () => {
    if (!player) return;
    if (player.playing) {
      // v31: "pause" means STOP EVERYTHING — motion, music, AND the
      // coach's voice. Previously only motion paused, so the user
      // had a frozen avatar while the music kept thumping and the
      // TTS kept talking. That's not what a stop button means.
      player.pause();
      try { stopMusic(); } catch {}
      try { voice?.cancelSpeak({ silenceMs: 3000 }); } catch {}
      // Also tell the server to abort the in-flight LLM turn so
      // we don't get a flood of buffered avatar.play events the
      // moment the user hits stop.
      try { ws?.send(JSON.stringify({ type: 'user_interrupt' })); } catch {}
      try { _cancelVariety(); } catch {}   // Q3: no auto-rotation while paused
      try { hideStepCaption(); } catch {}  // v210: clear the step caption on stop
      if (life) { try { life.setSpeaking(false); life.stopVisemes(); } catch {} }
      setPlayPauseLabel(true);
    } else if (player.data) {
      // Resume only if a clip is loaded; otherwise replay last.
      if (player.frame > 0) {
        player.resume();
      } else if (window.__lastPlay) {
        loadAndPlayClip(window.__lastPlay.clipId, window.__lastPlay.opts);
      }
      setPlayPauseLabel(false);
    } else if (window.__lastPlay) {
      loadAndPlayClip(window.__lastPlay.clipId, window.__lastPlay.opts);
      setPlayPauseLabel(false);
    }
  });
}

// ─── refresh button (hard reload) ────────────────────────────────────
$('refresh')?.addEventListener('click', () => {
  // location.reload(true) is deprecated but still works on most engines;
  // bypassing the SW + cache is what the user expects from a "refresh".
  try { location.reload(true); } catch { location.reload(); }
});

// ─── chat drawer toggle ──────────────────────────────────────────────
const _drawer = $('drawer');
const _drawerBackdrop = $('drawer-backdrop');
function isDrawerOpen() { return _drawer && _drawer.classList.contains('open'); }
function openDrawer() {
  if (!_drawer) return;
  _drawer.classList.add('open');
  _drawer.setAttribute('aria-hidden', 'false');
  if (_drawerBackdrop) _drawerBackdrop.classList.add('open');
  clearChatBadge();
  // Focus the input so users can start typing immediately on desktop.
  setTimeout(() => { try { $('text')?.focus({ preventScroll: true }); } catch {} }, 80);
}
function closeDrawer() {
  if (!_drawer) return;
  _drawer.classList.remove('open');
  _drawer.setAttribute('aria-hidden', 'true');
  if (_drawerBackdrop) _drawerBackdrop.classList.remove('open');
}
$('chat-toggle')?.addEventListener('click', () => {
  if (isDrawerOpen()) closeDrawer(); else openDrawer();
});
$('drawer-close')?.addEventListener('click', closeDrawer);
_drawerBackdrop?.addEventListener('click', closeDrawer);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isDrawerOpen()) closeDrawer();
});

// v223: single entry-point the LLM's close_panel tool (ui.close_panels) uses
// to clear the screen — closes the steps rail, chat drawer, lessons library
// and live-mirror popup so the student sees the avatar full-screen. Each is
// guarded so a missing/closed pane is a harmless no-op.
window.__closeAllPanels = function () {
  try { closeCoachRail(); } catch (e) {}
  try { closeDrawer(); } catch (e) {}
  try { window.__closeLearn && window.__closeLearn(); } catch (e) {}
  try { closeLiveFeedback(); } catch (e) {}
};

// ─── audio unlock (iOS / strict autoplay policies) ───────────────────
// On iOS Safari, the SpeechSynthesizer's default speaker output is
// blocked until a user gesture unlocks the audio context. Play a
// 1-frame silent buffer on the first tap so subsequent TTS responses
// can actually be heard.
let _audioUnlocked = false;
function unlockAudio() {
  if (_audioUnlocked) return;
  _audioUnlocked = true;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) {
      const ctx = new Ctx();
      const buf = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      try { src.start(0); } catch {}
      if (ctx.resume) ctx.resume().catch(() => {});
    }
  } catch (e) { /* best-effort */ }
}
['pointerdown', 'touchstart', 'keydown'].forEach(ev =>
  window.addEventListener(ev, unlockAudio, { once: true, passive: true })
);

// v139: "Let's start" CTA (start_hero card). One clear user tap that both
// unlocks audio playback (so the coach can speak) AND starts voice INPUT
// (the Azure STT mic, so she can hear the user). The greeting itself
// auto-plays off the same tap via the deferred-unlock listeners above.
window.addEventListener('dance:lets-start', () => {
  try { unlockAudio(); } catch (e) {}
  try {
    if (window.__audioCtx && window.__audioCtx.state === 'suspended') {
      window.__audioCtx.resume();
    }
  } catch (e) {}
  // Start the mic so the user can talk. Live voice (S2S) is the default
  // mode and owns its own mic, so when it's active we start S2S; only fall
  // back to the Azure STT mic if the user explicitly turned live voice off.
  try {
    if (window.__s2sDefault) {
      if (typeof _startS2S === 'function' && !_s2sOn) _startS2S();
    } else {
      const m = document.getElementById('mic');
      if (m && !m.classList.contains('rec')) m.click();
    }
  } catch (e) {}
}, { once: true });

// ─── first-visit tour ────────────────────────────────────────────────
const TOUR_KEY = 'dance_coach_tour_v1';
function shouldShowTour() {
  try { return !localStorage.getItem(TOUR_KEY); } catch { return false; }
}
function markTourSeen() {
  try { localStorage.setItem(TOUR_KEY, '1'); } catch {}
}
function openTour() { $('tour')?.classList.add('show'); }
function closeTour() { $('tour')?.classList.remove('show'); markTourSeen(); }
$('tour-skip')?.addEventListener('click', closeTour);
$('tour-go')?.addEventListener('click', () => {
  closeTour();
  // Unlock audio on first visit. v223: do NOT auto-open the chat drawer — the
  // user asked us to stop the chat window popping open on its own. Coach lines
  // still appear as floating stage bubbles; tap the chat button to see the log.
  unlockAudio();
});
if (shouldShowTour()) {
  // Defer until after the avatar's greeting so the tour doesn't
  // hide the welcome moment. ~3.5s gives the avatar time to wave +
  // start the signature dance.
  setTimeout(openTour, 3500);
}

// ─── quick-action chips ───────────────────────────────────────────────
// v183: replaces the previously dead #quick .chip wiring (the old
// markup with data-act="dance"/"teach"/"song"/"form" no longer exists
// in coach.html — this querySelectorAll silently matched nothing).
// The new #quick row is the "I can teach:" capability chips — each one
// is a REAL, guaranteed-to-work prompt (matches agent.py's genre
// keyword map) so users have concrete things to tap instead of typing
// out-of-catalog move names and getting stuck.
document.querySelectorAll('#quick .chip[data-genre-prompt]').forEach(btn => {
  btn.addEventListener('click', () => {
    if (window.matchMedia('(max-width: 760px)').matches) closeDrawer();
    sendUser(btn.dataset.genrePrompt);
  });
});

// ─── session-start buttons (5 / 10 / 20 min) ──────────────────────────
// These were never wired, so clicking them (or the start-hero overlay,
// which forwards its click here) did nothing. Bind them to send a
// `session.start` to the agent with a full template_id derived from the
// active character's style, e.g. quick10_gHO. The server's session.start
// handler + get_template() already understand this format.
const STYLE_GENRE = {
  'house': 'gHO',
  'la-style hiphop': 'gLH', 'la hip-hop': 'gLH', 'hip-hop': 'gLH',
  'hiphop': 'gLH', 'hip hop': 'gLH',
  'krump': 'gKR', 'waacking': 'gWA', 'waack': 'gWA',
  'jazz': 'gJS', 'street jazz': 'gJS',
  'breaking': 'gBR', 'b-boy': 'gBR', 'b-girl': 'gBR', 'breakdance': 'gBR',
  'locking': 'gLO', 'popping': 'gPO',
};
// Style is NOT tied to the character — ANY character can teach ANY style.
// Each guided session rolls a random dance genre so the same avatar
// teaches different styles across sessions. These are the genres with
// safe, retargeted clips (cmu is warmup-only, so excluded here).
const RANDOM_STYLE_GENRES = [
  'gHO', 'gLH', 'gKR', 'gWA', 'gJS', 'gBR', 'gLO', 'gPO', 'gMH',
];
function pickRandomGenre() {
  return RANDOM_STYLE_GENRES[
    Math.floor(Math.random() * RANDOM_STYLE_GENRES.length)];
}
function startGuidedSession(mins) {
  const m = (mins === 10 || mins === 20) ? mins : 5;
  // Remember this exact action so we can resume it right after the user
  // signs in via the popup (no reload, no re-tapping).
  window.__pendingAuthAction = () => startGuidedSession(m);
  window.__pendingAuthActionAt = Date.now();  // v93: freshness stamp
  // Fast path: if we already know the user isn't signed in, open the
  // popup immediately instead of waiting for the server to bounce us.
  if (!_isSignedIn()) {
    showSignInModal('Sign in to start your guided session.');
    return;
  }
  // Random style every session, independent of which character is loaded.
  const genre = pickRandomGenre();
  const templateId = `quick${m}_${genre}`;
  _sendGuidedStart(templateId, m);
}

// v190: start a session with an EXPLICIT template (chosen from the button-
// first start screen) instead of a random genre. Handles the auth gate +
// resume-after-signin the same way startGuidedSession does. `templateId`
// is either `quick{5|10|20}_<genre>` or `stretch_warmup_{5|10|15}`.
function startGuidedSessionTemplate(templateId, mins) {
  const m = parseInt(mins, 10) || 5;
  // v191: GUEST-FIRST. The front door promises "no signup needed to try",
  // so start the session immediately even for anonymous users — the coach
  // WS accepts anonymous connections. The old hard sign-in wall here was
  // exactly why tapping a length felt like "nothing happens" (a modal the
  // user didn't want) and killed activation. We still remember the action
  // so a later sign-in (to SAVE streak/history) can resume seamlessly.
  window.__pendingAuthAction = () => startGuidedSessionTemplate(templateId, m);
  window.__pendingAuthActionAt = Date.now();
  _sendGuidedStart(templateId, m);
}

// Shared WS send for both guided-session entry points.
function _sendGuidedStart(templateId, m) {
  const payload = { type: 'session.start', template_id: templateId, minutes: m };
  if (!ws || ws.readyState !== 1) {
    window.__wsQueue = window.__wsQueue || [];
    window.__wsQueue.push(payload);
    setStatus('connecting… your session will start shortly');
    try {
      if (!ws || ws.readyState === WebSocket.CLOSED ||
          ws.readyState === WebSocket.CLOSING) connect();
    } catch (e) {}
  } else {
    try { ws.send(JSON.stringify(payload)); } catch (e) {}
  }
  // On phones, close the drawer so the user sees the avatar start moving.
  if (window.matchMedia('(max-width: 760px)').matches) closeDrawer();
}
document.querySelectorAll('#session-start .ss-btn[data-act^="session-"]')
  .forEach((btn) => {
    btn.addEventListener('click', () => {
      const mins = parseInt(btn.dataset.act.replace('session-', ''), 10) || 5;
      // v215: use the GUEST-FIRST path (same as the #start-screen front
      // door). The old startGuidedSession() opened a sign-in modal and
      // RETURNED for anonymous users, so tapping a length "did nothing" —
      // a known activation killer that contradicted the v191 guest-first
      // decision applied everywhere else. Start immediately; a later
      // sign-in still resumes/saves via __pendingAuthAction.
      const genre = pickRandomGenre();
      startGuidedSessionTemplate(`quick${mins}_${genre}`, mins);
    });
  });

// Relay the session HUD's control buttons (pause / resume / skip / end)
// back to the agent. The HUD dispatches `session:cmd:<cmd>` window events.
// v69: pause/skip/end must ALSO silence the locally-playing audio
// (backing music + the coach's TTS voice) immediately — the server
// round-trip only pauses the phase ticker, so without this the avatar
// froze while the music kept thumping and the voice kept talking. The
// user pressed "pause" and heard no change. Stop it here, client-side.
function _silenceLocalAudio({ stopMotion = false } = {}) {
  try { stopMusic(); } catch (e) {}
  try { voice?.cancelSpeak({ silenceMs: 4000 }); } catch (e) {}
  try { if (life) { life.setSpeaking(false); life.stopVisemes?.(); } } catch (e) {}
  if (stopMotion) { try { player?.pause?.(); } catch (e) {} }
}
['pause', 'resume', 'skip', 'end'].forEach((cmd) => {
  window.addEventListener('session:cmd:' + cmd, () => {
    // Kill the emitting audio the instant the user taps a control.
    if (cmd === 'pause') _silenceLocalAudio({ stopMotion: true });
    else if (cmd === 'skip') _silenceLocalAudio();
    else if (cmd === 'end') {
      // v94: END must stop EVERYTHING on the spot (audio + the avatar's
      // looping clip) and hide the HUD immediately — don't wait for the
      // server's session.finished round-trip, which made End feel broken.
      window.__sessionEndedAt = Date.now();   // v97b: ignore late session clips
      _silenceLocalAudio({ stopMotion: true });
      try { if (player) { player.stop(); player.applyRestPose?.(); } } catch (e) {}
      try { window.__sessionHUD?.hide(); } catch (e) {}
      window.__inSession = false;
    }
    const payload = { type: 'session.' + cmd };
    if (!ws || ws.readyState !== 1) {
      window.__wsQueue = window.__wsQueue || [];
      window.__wsQueue.push(payload);
    } else {
      try { ws.send(JSON.stringify(payload)); } catch (e) {}
    }
  });
});

// ─── mirror-cam (webcam overlay) ──────────────────────────────────────
let _camStream = null;
$('mirror-toggle').addEventListener('click', async () => {
  const v = $('mirrorcam');
  const btn = $('mirror-toggle');
  if (_camStream) {
    _camStream.getTracks().forEach(t => t.stop());
    _camStream = null;
    v.srcObject = null;
    v.classList.remove('live');
    btn.classList.remove('live');
    try { if (_gv) _gv.setCameraSource(null); } catch (e) {}
    return;
  }
  try {
    _camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360, facingMode: 'user' },
      audio: false,
    });
    v.srcObject = _camStream;
    v.classList.add('live');
    btn.classList.add('live');
    // v89: if live voice is on, start streaming frames so she can see you.
    try {
      if (_gv) {
        _gv.setCameraSource(v);
        addMsg('📹 Camera on — I can see you now, let\'s move!', 'sys');
      }
    } catch (e) {}
  } catch (e) {
    addMsg('webcam error: ' + e.message, 'flag');
  }
});

// ─── audio → resequencer ──────────────────────────────────────────────
$('audio-input').addEventListener('change', async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (!file) return;
  const opts = window.__resequenceOpts || { bars: 8 };
  // If the user picked a video, extract the audio track into a WAV
  // blob in-browser before uploading. This keeps the backend
  // single-codepath (audio-only) and works for any video container
  // the browser can decode (mp4, webm, mov…).
  let uploadFile = file;
  let uploadName = file.name;
  if (file.type.startsWith('video/')) {
    addMsg(`🎬 extracting audio from "${file.name}"…`, 'sys');
    try {
      uploadFile = await _extractAudioFromVideo(file);
      uploadName = file.name.replace(/\.[^.]+$/, '') + '.wav';
    } catch (e) {
      addMsg('audio extraction failed: ' + e.message, 'flag');
      return;
    }
  }
  addMsg(`🎵 generating routine to "${uploadName}" (${opts.bars} bars)…`,
         'sys');
  const fd = new FormData();
  fd.append('audio', uploadFile, uploadName);
  fd.append('bars', String(opts.bars || 8));
  if (opts.genre) fd.append('genre', opts.genre);
  if (opts.query) fd.append('query', opts.query);
  try {
    const r = await fetch(APP_BASE + '/api/motion/resequence', {
      method: 'POST', body: fd });
    if (!r.ok) {
      addMsg('resequence failed: ' + r.status + ' ' +
             await r.text(), 'flag');
      return;
    }
    const motion = await r.json();
    addMsg(`🎵 routine ready · ${motion.n_frames} frames · ` +
           `${(motion.duration_s || 0).toFixed(1)}s`, 'sys');
    player.load(motion);
    player.play({ speed: 1.0, loop: false });
    // Also play the song so avatar dances to it. Use the ORIGINAL
    // file (video keeps its visuals off-screen but the audio plays).
    try { window.__currentAudio?.pause?.(); } catch {}
    const audio = new Audio(URL.createObjectURL(file));
    audio.play().catch(() => {});
    window.__currentAudio = audio;
  } catch (e) {
    addMsg('resequence error: ' + e.message, 'flag');
  }
});

/** Decode a video file's audio track into a WAV Blob using the Web
 *  Audio API. Works for any container the browser can demux.        */
async function _extractAudioFromVideo(file) {
  const arrayBuf = await file.arrayBuffer();
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  // First decode with a regular AudioContext to learn duration + rate.
  const probeCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuf = await probeCtx.decodeAudioData(arrayBuf.slice(0));
  probeCtx.close?.();
  // Convert to a 16-bit PCM WAV blob (stereo or mono, 44.1k preserved).
  return _audioBufferToWavBlob(audioBuf);
}

function _audioBufferToWavBlob(buf) {
  const numCh = Math.min(2, buf.numberOfChannels);
  const sr = buf.sampleRate;
  const samples = buf.length;
  const bytesPerSample = 2;
  const blockAlign = numCh * bytesPerSample;
  const dataSize = samples * blockAlign;
  const ab = new ArrayBuffer(44 + dataSize);
  const dv = new DataView(ab);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, 'RIFF');
  dv.setUint32(4, 36 + dataSize, true);
  wstr(8, 'WAVE'); wstr(12, 'fmt ');
  dv.setUint32(16, 16, true);   // PCM chunk size
  dv.setUint16(20, 1, true);    // PCM format
  dv.setUint16(22, numCh, true);
  dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * blockAlign, true);
  dv.setUint16(32, blockAlign, true);
  dv.setUint16(34, 16, true);   // bits per sample
  wstr(36, 'data');
  dv.setUint32(40, dataSize, true);
  // Interleave channels.
  const chs = [];
  for (let c = 0; c < numCh; c++) chs.push(buf.getChannelData(c));
  let off = 44;
  for (let i = 0; i < samples; i++) {
    for (let c = 0; c < numCh; c++) {
      let s = Math.max(-1, Math.min(1, chs[c][i]));
      dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
  }
  return new Blob([ab], { type: 'audio/wav' });
}

// ─── video → 2D pose extraction → /api/feedback/compare2d ────────────
//
// MediaPipe Pose returns 33 BlazePose landmarks per frame in
// normalised image coords [0..1] (x,y,visibility). We re-pack the
// 17 COCO landmarks the server expects:
//   nose, lEye, rEye, lEar, rEar,
//   lShoulder, rShoulder, lElbow, rElbow, lWrist, rWrist,
//   lHip, rHip, lKnee, rKnee, lAnkle, rAnkle
const MP_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,
                      23, 24, 25, 26, 27, 28];

let _poseLandmarker = null;
async function _initPoseLandmarker() {
  if (_poseLandmarker) return _poseLandmarker;
  const vision = await import('@mediapipe/tasks-vision');
  const { FilesetResolver, PoseLandmarker } = vision;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  _poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' +
        'pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.4,
    minPosePresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
  return _poseLandmarker;
}

async function _extractStudentKeypoints(file, onProgress) {
  // Load the file into a hidden <video> + tick frame by frame.
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  await new Promise((res, rej) => {
    video.onloadedmetadata = res;
    video.onerror = () => rej(new Error('video decode failed'));
  });
  const fps = 30;     // sampling rate — fine for any consumer phone clip.
  const total = Math.max(1, Math.floor(video.duration * fps));
  const lm = await _initPoseLandmarker();
  const out = [];                            // (T, 17, 3)
  for (let i = 0; i < total; i++) {
    const t = (i + 0.5) / fps;
    video.currentTime = Math.min(t, Math.max(0, video.duration - 0.001));
    await new Promise(r => video.onseeked = r);
    const res = lm.detectForVideo(video, performance.now());
    const lms = (res.landmarks && res.landmarks[0]) || null;
    const frame = new Array(17);
    if (lms) {
      for (let k = 0; k < 17; k++) {
        const src = lms[MP_TO_COCO17[k]];
        frame[k] = [src.x, src.y, src.visibility ?? 1.0];
      }
    } else {
      // no detection → zeros + zero visibility (compare_2d ignores them).
      for (let k = 0; k < 17; k++) frame[k] = [0, 0, 0];
    }
    out.push(frame);
    if (onProgress && i % 10 === 0) onProgress(i, total);
  }
  URL.revokeObjectURL(url);
  return { fps, keypoints: out };
}

$('video-input').addEventListener('change', async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (!file) return;
  const clipId = window.__feedbackClipId;
  if (!clipId) {
    addMsg('pick a clip first, then I can grade you', 'flag');
    return;
  }
  addMsg(`🎬 received "${file.name}" — extracting your pose now…`, 'sys');
  const progressMsg = addMsg('  0%', 'sys');     // reuse for updates
  let extracted;
  try {
    extracted = await _extractStudentKeypoints(file, (i, total) => {
      if (progressMsg) {
        progressMsg.textContent = `  ${Math.floor(100*i/total)}% ` +
                                  `(${i}/${total} frames)`;
      }
    });
  } catch (e) {
    addMsg('pose extraction failed: ' + e.message, 'flag');
    return;
  }
  if (progressMsg) progressMsg.textContent = '  pose ready, asking coach…';
  try {
    const r = await fetch(APP_BASE + '/api/feedback/compare2d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clip_id:   clipId,
        fps:       extracted.fps,
        keypoints: extracted.keypoints,
      }),
    });
    if (!r.ok) {
      const err = await r.text();
      addMsg(`feedback error ${r.status}: ${err.slice(0, 200)}`, 'flag');
      return;
    }
    const out = await r.json();
    addMsg('💬 ' + out.note, 'coach');
    const worst = (out.worst_keypoints || []).slice(0, 3)
      .map(w => `${w.keypoint} (${w.mean.toFixed(2)})`).join(', ');
    if (worst) {
      addMsg(`(worst spots: ${worst} — mean error ` +
             `${out.mean_error.toFixed(2)} torso-units)`, 'sys');
    }
  } catch (e) {
    addMsg('feedback request failed: ' + e.message, 'flag');
  }
});

// ─── v227: UPLOAD A VIDEO TO LEARN ─────────────────────────────────────
// A visible "➕" on the dock. The user uploads a short clip; we hand it to the
// studio-Os learn pipeline (POST /api/learn/upload → /api/learn/submit) which
// stores it, creates a job, emails the OWNER to process it, and later emails
// the user "your dance is ready". GATED by sign-in with context carry: an
// anonymous tap opens the sign-in modal and, on success, resumes the picker on
// the SAME page (no reload) via the existing __pendingAuthAction pattern.
(function initLearnUpload() {
  const btn = document.getElementById('upload-video');
  const input = document.getElementById('learn-video-input');
  if (!btn || !input) return;

  function openPicker() { try { input.click(); } catch (e) {} }

  function gatedOpen() {
    try { unlockAudio(); } catch (e) {}
    if (!_isSignedIn()) {
      // Carry context: after a successful sign-in, resume the picker right here.
      window.__pendingAuthAction = openPicker;
      window.__pendingAuthActionAt = Date.now();
      try {
        showSignInModal(
          "Sign in to upload your own video \u2014 I'll learn the moves and "
          + "teach you the steps. You'll stay right here.", 'register');
      } catch (e) {}
      return;
    }
    openPicker();
  }
  btn.addEventListener('click', gatedOpen);

  input.addEventListener('change', async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = '';
    if (!file) return;
    if (!_isSignedIn()) { gatedOpen(); return; }
    if (file.size > 120 * 1024 * 1024) {
      _showUploadCard('big');
      return;
    }
    let tok = ''; try { tok = localStorage.getItem('token') || ''; } catch (e) {}
    _showUploadCard('uploading');
    try {
      const fd = new FormData();
      fd.append('video', file, file.name || 'upload.mp4');
      fd.append('title', (file.name || '').replace(/\.[^.]+$/, '').slice(0, 120));
      const r = await fetch(APP_BASE + '/api/learn/upload', {
        method: 'POST',
        headers: { Authorization: 'Bearer ' + tok },
        body: fd,
      });
      let data = {}; try { data = await r.json(); } catch (e) {}
      if (r.ok && data && data.ok) {
        try { _coachTrack('dance_submitted', { kind: 'learn_upload' }); } catch (e) {}
        _showUploadCard('done', data.message);
      } else if (r.status === 402 || (data && data.upgrade_required)) {
        _showUploadCard('upsell');
      } else if (r.status === 429 || (data && data.rate_limited)) {
        _showUploadCard('rate');
      } else if (r.status === 401) {
        gatedOpen();
      } else {
        _showUploadCard('error', (data && data.error) || '');
      }
    } catch (e) {
      _showUploadCard('error', 'network');
    }
  });

  // Simple full-screen card for each state. One at a time.
  function _showUploadCard(state, msg) {
    try {
      const prev = document.getElementById('learn-upload-ov');
      if (prev) prev.remove();
    } catch (e) {}
    const ov = document.createElement('div');
    ov.id = 'learn-upload-ov';
    ov.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:10000',
      'display:flex', 'align-items:center', 'justify-content:center',
      'padding:20px', 'background:rgba(7,2,15,.72)',
      'backdrop-filter:blur(8px)', '-webkit-backdrop-filter:blur(8px)',
      'font:14px/1.5 -apple-system,BlinkMacSystemFont,Inter,Segoe UI,Roboto,sans-serif',
    ].join(';');
    let icon = '\u23F3', title = '', body = '', primary = 'Keep dancing', showClose = true;
    if (state === 'uploading') {
      icon = '\u23F3'; title = 'Uploading your video\u2026';
      body = 'Hang tight \u2014 sending it over. Please keep this open.';
      primary = ''; showClose = false;
    } else if (state === 'done') {
      icon = '\uD83C\uDF89'; title = "Got it! We're on it.";
      body = msg || "We're processing your video. You'll get a notification and "
        + "an email when your coach is ready to teach you these steps \u2014 "
        + "meanwhile, learn something else. \uD83D\uDC9C";
      primary = 'Learn something now';
    } else if (state === 'upsell') {
      icon = '\uD83D\uDC83'; title = "You've used your free video";
      body = 'Your first video is on us. To keep turning your own clips into '
        + 'step-by-step lessons, grab a Learner or Pro plan \u2014 more videos, '
        + 'longer clips and a finer breakdown.';
      primary = 'Maybe later';
    } else if (state === 'rate') {
      icon = '\u23F0'; title = 'Whoa, slow down!';
      body = "You've uploaded a couple today already. Try again tomorrow \u2014 "
        + 'your coach needs time to build each lesson properly.';
      primary = 'Got it';
    } else if (state === 'big') {
      icon = '\uD83D\uDCC1'; title = 'That clip is a bit large';
      body = 'Please upload a shorter video (a reel-length clip works best). '
        + 'Short, clear moves are easiest for me to break down.';
      primary = 'OK';
    } else {
      icon = '\uD83D\uDE15'; title = "Couldn't upload that";
      body = 'Something went wrong sending your video. Please try again in a '
        + 'moment' + (msg ? ' (' + msg + ')' : '') + '.';
      primary = 'Close';
    }
    const spinner = (state === 'uploading')
      ? '<div style="width:34px;height:34px;margin:0 auto 10px;border:3px solid rgba(192,97,255,.25);border-top-color:#c061ff;border-radius:50%;animation:luSpin .8s linear infinite"></div>'
      : '<div style="font-size:46px;margin-bottom:8px">' + icon + '</div>';
    ov.innerHTML =
      '<style>@keyframes luSpin{to{transform:rotate(360deg)}}</style>' +
      '<div style="max-width:380px;width:92%;text-align:center;background:linear-gradient(165deg,#1c1330,#241546);'
      + 'border:1px solid rgba(192,97,255,.35);border-radius:20px;padding:26px 22px;box-shadow:0 24px 70px rgba(0,0,0,.55);color:#f4f1fb">'
      + spinner
      + '<div style="font-size:19px;font-weight:800;margin-bottom:8px">' + title + '</div>'
      + '<div style="font-size:13.5px;color:#c9c0e0;margin-bottom:18px">' + body + '</div>'
      + (primary ? ('<button id="lu-primary" style="width:100%;background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;border:0;border-radius:12px;padding:13px;font-weight:800;font-size:15px;cursor:pointer">' + primary + '</button>') : '')
      + '</div>';
    document.body.appendChild(ov);
    if (state === 'done') { try { window.__fireConfetti && window.__fireConfetti(); } catch (e) {} }
    const close = () => { try { ov.remove(); } catch (e) {} };
    const p = ov.querySelector('#lu-primary');
    if (p) p.addEventListener('click', () => {
      close();
      // "Learn something now" → reopen the style picker so they don't idle.
      if (state === 'done') { try { (window.__openFrontDoor || function(){})(); } catch (e) {} }
    });
    if (showClose) ov.addEventListener('click', (e) => { if (e.target === ov) close(); });
  }
})();

bootstrap().then(() => {
  // Bootstrap is done — profile is set, VRM is loaded. If the WS
  // already opened (eager connect, below) the onopen handler ran
  // before we knew which character to greet as. So we replay the
  // set_character + greeting here, idempotently.
  try {
    const prof = window.__charProfile;
    if (prof && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'set_character',
        name: window.__charSlug || null,
        display_name: prof.display_name || null,
        style: prof.style || prof.tagline || null,
      }));
    }
  } catch (e) { /* ws not ready, that's fine */ }
  // _greetedThisSession guards against double-firing.
  setTimeout(() => playCoachGreeting().catch(() => {}), 200);
  // Make sure we have a WS even if eager-connect never fired (rare).
  if (!ws) connect();
  // v185: catch a stale/expired token from a PRIOR visit right away,
  // instead of only on tab-return or the 5-min interval.
  setTimeout(() => { try { _verifySessionOrPrompt(); } catch (e) {} }, 3000);
}).catch(err => {
  setStatus('bootstrap failed');
  addMsg('bootstrap: ' + err.message, 'flag');
  // v156: don't strand the user behind the "Warming up your studio"
  // overlay for the full 45s if bootstrap genuinely failed — surface the
  // error message on the loader and let it close so they at least see
  // the chat/error state underneath instead of a frozen progress bar.
  try { window.__boot?.fail('Something went wrong loading the studio.'); } catch (e) {}
  try { window.__boot?.done(); } catch (e) {}
});

// ─── eager WS connect ───────────────────────────────────────────────
// VRM meshes are ~10-20 MB and take 1-3s to load. The WebSocket
// handshake itself takes only ~150ms. Without this, the user stares
// at a "loading sample_k…" screen for the entire VRM download with
// no WS connected — so their first message can't be sent. Kick the
// WS off in parallel; bootstrap() will replay set_character + greet
// once the character profile is known.
setTimeout(() => {
  try {
    if (!ws || ws.readyState >= WebSocket.CLOSING) connect();
  } catch (e) { /* bootstrap fallback will retry */ }
}, 50);


// ─── v93: PRODUCT-CLARITY DOCK (5 buttons) + RESTART ──────────────────
// Too many controls confused users. Keep only what a first-time user
// needs: Camera, Mic, Live-talk, Chat, and a Restart. Hide the rest
// (pause/speed/mirror-choreography/voice-out/character-select) — their
// JS stays wired so nothing breaks, they're just not shown.
(function simplifyDock() {
  // v95: Live S2S is the ONLY voice mode and it's default-on, so it owns
  // the mic AND the audio. Hide the separate Azure mic (🎙) too — it's
  // redundant and confusing. Dock = Camera, Live(🎧), Restart, Chat.
  const hideIds = ['playpause', 'speed-wrap', 'mirror-btn', 'tts', 'char-wrap', 'mic'];
  for (const id of hideIds) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  }
  // Add a Restart button into the right-hand control group, before chat.
  const chat = document.getElementById('chat-toggle');
  if (chat && chat.parentElement && !document.getElementById('restart-btn')) {
    const b = document.createElement('button');
    b.id = 'restart-btn';
    b.className = 'dock-btn';
    b.type = 'button';
    b.title = 'Start over — fresh conversation';
    b.setAttribute('aria-label', 'Restart conversation');
    b.textContent = '↻';
    chat.parentElement.insertBefore(b, chat);
    b.addEventListener('click', () => {
      try { unlockAudio(); } catch (e) {}
      restartConversation();
    });
  }
})();

// --- v96: WEB PUSH (daily reminders) ---------------------------------
function _urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(base64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}
let _swReg = null;
async function _registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return null;
  if (_swReg) return _swReg;
  try {
    // v137: register the SINGLE service worker (push + asset caching) via the
    // proxied /m/service-worker route (nginx only forwards /m/, not /sw.js),
    // and claim the whole app scope (APP_BASE + '/') so it can cache the heavy
    // avatar VRM + CDN libs. The server sends Service-Worker-Allowed: / so the
    // broader scope is permitted from the /dance/m/ script path.
    _swReg = await navigator.serviceWorker.register(
      APP_BASE + '/m/service-worker', { scope: APP_BASE + '/' });
    return _swReg;
  } catch (e) { console.warn('[sw] register failed', e); return null; }
}
async function _isPushSubscribed() {
  try {
    const reg = await _registerServiceWorker();
    if (!reg) return false;
    const sub = await reg.pushManager.getSubscription();
    return !!sub;
  } catch (e) { return false; }
}
async function _enablePush() {
  try {
    if (!('Notification' in window)) { addMsg('Reminders aren\u2019t available here yet — open Dance.AI in Chrome to turn them on.', 'sys'); return false; }
    const perm = await Notification.requestPermission();
    if (perm !== 'granted') { addMsg('Notifications blocked. Enable them in browser settings to get reminders.', 'flag'); return false; }
    const reg = await _registerServiceWorker();
    if (!reg) { addMsg('Could not set up notifications.', 'flag'); return false; }
    let pub = '';
    try { const r = await fetch(APP_BASE + '/api/notifications/vapid-key'); const d = await r.json(); pub = d.public_key || ''; } catch (e) {}
    if (!pub) { addMsg('Notifications not configured yet.', 'flag'); return false; }
    let sub = await reg.pushManager.getSubscription();
    if (!sub) {
      sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: _urlBase64ToUint8Array(pub),
      });
    }
    const tz = (Intl.DateTimeFormat().resolvedOptions().timeZone) || 'UTC';
    const token = (function(){ try { return localStorage.getItem('token'); } catch(e){ return null; } })();
    const r2 = await fetch(APP_BASE + '/api/me/push/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (token || '') },
      body: JSON.stringify({ subscription: sub.toJSON ? sub.toJSON() : sub, tz, preferred_hour: 18 }),
    });
    if (!r2.ok) { addMsg('Could not save your reminder. Try again later.', 'flag'); return false; }
    addMsg('Daily reminders on. I will nudge you to dance!', 'sys');
    try { localStorage.setItem('coach.push.on', '1'); } catch (e) {}
    return true;
  } catch (e) { console.warn('[push] enable failed', e); addMsg('Notification setup failed.', 'flag'); return false; }
}
(function initPush() {
  _registerServiceWorker();
  // v185: the old #push-optin was a small card tucked inside the chat
  // drawer -- invisible unless the drawer happened to be open, which is
  // exactly why nobody ever saw/used it (0 subscriptions in prod). This
  // now shows as an actual centered popup (reuses the same modal CSS as
  // the sign-in dialog via _ensureAuthStyles) triggered RIGHT AFTER a
  // successful login (see _applyAuthSuccess -> window.__refreshPushOptin)
  // as well as on page load for already-signed-in returning visitors.
  let _pushModalEl = null;
  function _buildPushModal() {
    if (_pushModalEl) return _pushModalEl;
    _ensureAuthStyles();
    const wrap = document.createElement('div');
    wrap.id = 'push-modal';
    wrap.className = 'authm-backdrop';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'true');
    wrap.innerHTML =
      '<div class="authm-card" role="document">' +
        '<div class="authm-head">' +
          '<h2 class="authm-title">Stay in the loop \ud83d\udd14</h2>' +
          '<button id="push-modal-x" class="authm-close" aria-label="Close">\u00d7</button>' +
        '</div>' +
        '<p class="authm-sub">Want reminders about your next session, ' +
          'new moves, and updates as we keep improving Dance.AI? Allow ' +
          'notifications and I\u2019ll keep you posted \u2014 never spammy.</p>' +
        '<button id="push-modal-enable" class="authm-submit" type="button">Allow notifications</button>' +
        '<div class="authm-foot"><a id="push-modal-dismiss" class="authm-link" role="button" tabindex="0">Not now</a></div>' +
      '</div>';
    document.body.appendChild(wrap);
    _pushModalEl = wrap;
    wrap.addEventListener('click', (e) => { if (e.target === wrap) _hidePushModal(); });
    wrap.querySelector('#push-modal-x').addEventListener('click', _hidePushModal);
    wrap.querySelector('#push-modal-enable').addEventListener('click', async () => {
      const btn = wrap.querySelector('#push-modal-enable');
      const orig = btn.textContent;
      btn.textContent = 'Enabling\u2026'; btn.disabled = true;
      const ok = await _enablePush();
      btn.disabled = false; btn.textContent = orig;
      if (ok) _hidePushModal();
    });
    wrap.querySelector('#push-modal-dismiss').addEventListener('click', () => {
      try { localStorage.setItem('coach.push.dismissed', '1'); } catch (e) {}
      _hidePushModal();
    });
    return wrap;
  }
  function _hidePushModal() {
    if (_pushModalEl) _pushModalEl.classList.remove('show');
  }
  async function maybeShow() {
    // v135: web push needs the Notification + PushManager APIs. The Android
    // app's WebView doesn't have them (yet -- native push is separate, see
    // devices/register), so never show this popup there.
    const pushSupported = ('Notification' in window) &&
                          ('serviceWorker' in navigator) &&
                          ('PushManager' in window);
    if (!pushSupported) return;
    let dismissed = false, on = false;
    try { dismissed = localStorage.getItem('coach.push.dismissed') === '1'; } catch (e) {}
    try { on = localStorage.getItem('coach.push.on') === '1'; } catch (e) {}
    const signedIn = (function(){ try { return !!localStorage.getItem('token'); } catch(e){ return false; } })();
    if (!signedIn || dismissed || on) return;
    const already = await _isPushSubscribed();
    if (already) return;
    if (Notification.permission === 'denied') return; // don't nag if OS-blocked
    _buildPushModal().classList.add('show');
  }
  setTimeout(maybeShow, 2500);
  window.__refreshPushOptin = maybeShow;
})();

// ─── Native push (Android Capacitor wrapper) ───────────────────────────
// The Play-Store app loads this same site inside a Capacitor WebView. Web
// Push (above) does NOT deliver when that app is fully backgrounded, so on
// native we use FCM instead: ask the OS for a device token via the
// @capacitor/push-notifications plugin, then POST it to /api/me/devices/
// register so the backend can fan out native pushes through FCM v1.
// This is a no-op in a normal browser (no Capacitor bridge present).
(function initNativePush() {
  function _cap() {
    try { return (window.Capacitor && Capacitor.isNativePlatform &&
                  Capacitor.isNativePlatform()) ? window.Capacitor : null; }
    catch (e) { return null; }
  }
  const cap = _cap();
  if (!cap) return;                       // browser / PWA -> Web Push path
  const PN = cap.Plugins && cap.Plugins.PushNotifications;
  if (!PN) { console.warn('[native-push] plugin not installed'); return; }

  let _fcmToken = '';
  try { _fcmToken = localStorage.getItem('coach.fcm.token') || ''; } catch (e) {}

  async function _postToken() {
    let jwt = ''; try { jwt = localStorage.getItem('token') || ''; } catch (e) {}
    if (!_fcmToken || !jwt) return;       // need both device token + sign-in
    const tz = (Intl.DateTimeFormat().resolvedOptions().timeZone) || 'UTC';
    const locale = (navigator.language || '').slice(0, 16);
    try {
      const r = await fetch(APP_BASE + '/api/me/devices/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json',
                   'Authorization': 'Bearer ' + jwt },
        body: JSON.stringify({ registration_id: _fcmToken,
                               platform: 'android', locale, tz }),
        keepalive: true,
      });
      if (r.ok) { try { localStorage.setItem('coach.fcm.sent', '1'); } catch (e) {} }
    } catch (e) { console.warn('[native-push] register failed', e); }
  }
  // Re-POST the token after a successful login (called from _applyAuthSuccess).
  window.__postNativeToken = _postToken;

  PN.addListener('registration', (t) => {
    _fcmToken = (t && t.value) || '';
    try { localStorage.setItem('coach.fcm.token', _fcmToken); } catch (e) {}
    _postToken();
  });
  PN.addListener('registrationError', (e) => {
    console.warn('[native-push] registration error', e);
  });
  // Tapping a delivered notification: honour a {url} data payload.
  PN.addListener('pushNotificationActionPerformed', (ev) => {
    try {
      const url = ev && ev.notification && ev.notification.data &&
                  ev.notification.data.url;
      if (url && typeof url === 'string') {
        if (/^https?:/i.test(url)) location.href = url;
        else location.href = APP_BASE + (url.startsWith('/') ? url : '/' + url);
      }
    } catch (e) {}
  });

  (async function _boot() {
    try {
      let perm = await PN.checkPermissions();
      if (perm.receive === 'prompt' || perm.receive === 'prompt-with-rationale') {
        perm = await PN.requestPermissions();
      }
      if (perm.receive === 'granted') {
        await PN.register();              // -> fires 'registration' listener
      } else {
        console.warn('[native-push] permission not granted:', perm.receive);
      }
    } catch (e) { console.warn('[native-push] boot failed', e); }
  })();
})();

// Complete restart of the conversation session: stop live voice + any
// guided session, silence audio, clear the chat, re-greet, reconnect.
function restartConversation() {
  try { if (typeof _stopS2S === 'function') _stopS2S(true); } catch (e) {}
  // End any active guided session, both via HUD relay and direct WS.
  try { window.dispatchEvent(new CustomEvent('session:cmd:end')); } catch (e) {}
  try { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'session.end' })); } catch (e) {}
  try { if (typeof _silenceLocalAudio === 'function') _silenceLocalAudio({ stopMotion: true }); } catch (e) {}
  try { if (player) player.stop(); } catch (e) {}
  // Wipe the chat log + any floating bubbles.
  try { if (log) log.innerHTML = ''; } catch (e) {}
  try { const bw = document.getElementById('bubble-wrap'); if (bw) bw.innerHTML = ''; } catch (e) {}
  try { clearChatBadge(); } catch (e) {}
  // Allow the coach to greet again on the fresh connection.
  _greetedThisSession = false;
  try { sessionStorage.removeItem('coach.greet.ts'); } catch (e) {}
  // Reconnect a clean socket.
  try { ws && ws.close(); } catch (e) {}
  try { connect(); } catch (e) {}
  addMsg('Fresh start — new conversation.', 'sys');
  setTimeout(() => { try { playCoachGreeting().catch(() => {}); } catch (e) {} }, 600);
}

// ─── v190: button-first START SCREEN controller (the front door) ──────
// Big style buttons -> duration -> dance. Record-&-analyze is first-class
// (camera optional). Chat/voice are opt-in ("Just chat with the coach").
// This replaces the "AI talks at you on open" cold-start that was killing
// activation (8 signups, ~0 sessions). Wires into the EXISTING guided-
// session (startGuidedSessionTemplate) and video-analysis
// (/api/feedback/compare2d via #video-input) machinery — no new backend.
(function initStartScreen() {
  const scr = document.getElementById('start-screen');
  if (!scr) return;
  // StudioOS renders its own lesson cockpit in embedded mode. Never initialize
  // the generic "What do you want to dance?" front door over that lesson.
  if (_embeddedLesson) {
    window.__frontDoorOpen = false;
    scr.classList.add('ss2-hidden');
    scr.setAttribute('aria-hidden', 'true');
    scr.style.pointerEvents = 'none';
    const embeddedReopen = document.getElementById('ss2-reopen');
    if (embeddedReopen) embeddedReopen.style.display = 'none';
    return;
  }
  window.__frontDoorOpen = true;   // keeps playCoachGreeting() quiet on load

  const stepStyles = document.getElementById('ss2-styles');
  const stepDur = document.getElementById('ss2-duration');
  const stepAnalyze = document.getElementById('ss2-analyze-pick');
  const durRow = document.getElementById('ss2-durrow');
  const durTitle = document.getElementById('ss2-dur-title');
  const analyzeGrid = document.getElementById('ss2-analyze-grid');
  const reopen = document.getElementById('ss2-reopen');

  const STYLE_LABEL = {
    gLH: 'Hip-Hop', gHO: 'House', gLO: 'Locking', gWA: 'Waacking',
    gBR: 'Breaking', gPO: 'Popping', gKR: 'Krump', gJS: 'Jazz',
    gMH: 'Middle Hip-Hop',
  };
  const ANALYZE_STYLES = [
    ['gLH', '🔥', 'Hip-Hop'], ['gHO', '🏠', 'House'],
    ['gLO', '🔒', 'Locking'], ['gWA', '👐', 'Waacking'],
    ['gBR', '🌀', 'Breaking'], ['gPO', '🤖', 'Popping'],
    ['gKR', '💥', 'Krump'], ['gJS', '🎷', 'Jazz'],
  ];

  function showStep(step) {
    for (const s of [stepStyles, stepDur, stepAnalyze]) {
      if (s) s.hidden = (s !== step);
    }
  }
  function openDoor() {
    window.__frontDoorOpen = true;
    scr.style.pointerEvents = '';
    scr.classList.remove('ss2-hidden');
    scr.setAttribute('aria-hidden', 'false');
    if (reopen) reopen.style.display = 'none';
    showStep(stepStyles);
    // v-ux9: the blocking full-screen "how it works" overlay had 0% CTR — it
    // was a speed bump. Replaced by a dismissible inline tip inside the door
    // (see _initInlineTip). Nothing to pop here anymore.
  }
  try { window.__openFrontDoor = openDoor; } catch (e) {}
  function closeDoor() {
    window.__frontDoorOpen = false;
    scr.classList.add('ss2-hidden');
    scr.setAttribute('aria-hidden', 'true');
    setTimeout(() => { scr.style.pointerEvents = 'none'; }, 360);
    _syncReopenBtn();          // Q1: only show the pill when she's idle
    try { _maybeVoiceHint(); } catch (e) {}   // Q2: one-tap "turn on voice"
  }
  // v191: "peek" dismiss — hide the panel to reveal the avatar WITHOUT
  // committing to anything. Unlike closeDoor(), this keeps __frontDoorOpen
  // TRUE so the coach stays quiet (no greeting, no live-voice auto-arm)
  // until the user actually picks a path. Reopen via the 🏠 Start menu.
  function collapseDoor() {
    scr.classList.add('ss2-hidden');
    scr.setAttribute('aria-hidden', 'true');
    setTimeout(() => { scr.style.pointerEvents = 'none'; }, 360);
    _syncReopenBtn();          // Q1: respect dancing state
  }

  // Step 2: duration chooser. kind = 'warmup' (5/10/15) or 'session' (5/10/20).
  function showDurations(kind, label, opts) {
    durTitle.textContent = label ? `${label} — how long?` : 'How long?';
    durRow.innerHTML = '';
    const lens = kind === 'warmup' ? [5, 10, 15] : [5, 10, 20];
    for (const m of lens) {
      const b = document.createElement('button');
      b.className = 'ss2-dur';
      b.type = 'button';
      b.innerHTML = `<b>${m}</b><span>min</span>`;
      b.addEventListener('click', () => {
        _greetedThisSession = true;   // a chosen session never auto-greets
        try { unlockAudio(); } catch (e) {}
        const tid = kind === 'warmup'
          ? `stretch_warmup_${m}`
          : `quick${m}_${opts.genre}`;
        startGuidedSessionTemplate(tid, m);
        closeDoor();
      });
      durRow.appendChild(b);
    }
    showStep(stepDur);
  }

  // Step 1: style tiles.
  stepStyles.querySelectorAll('.ss2-tile').forEach((tile) => {
    tile.addEventListener('click', () => {
      // v214: instrument the picker so we can SEE first-tap engagement (the
      // funnel was blind here — only guided-session starts fired events, so
      // picker/learn taps looked like a 0% bounce). Fire a whitelisted event.
      try {
        const _g = tile.dataset.warmup ? 'warmup' : (tile.dataset.genre || 'unknown');
        _coachTrack('dance_style_viewed', { style: _g, from: 'front_door' });
      } catch (e) {}
      if (tile.dataset.warmup) {
        showDurations('warmup', 'Warm-up', {});
        return;
      }
      let g = tile.dataset.genre;
      const surprise = (g === 'random');
      if (surprise) g = pickRandomGenre();
      // v213: Hip-Hop & House are our carefully-crafted STRUCTURED tracks.
      // Selecting them opens the Learn panel (lessons + step-by-step rail)
      // instead of the generic duration→session flow, so teaching is the
      // same everywhere on the platform. Other styles keep the quick session.
      if (!surprise && (g === 'gLH' || g === 'gHO')) {
        try { unlockAudio(); } catch (e) {}
        try { collapseDoor(); } catch (e) {}
        const openLearn = () => { try { (window.__openLearn || function(){})(); } catch (e) {} };
        openLearn();
        setTimeout(openLearn, 180);
        return;
      }
      showDurations('session', surprise ? 'Surprise' : (STYLE_LABEL[g] || 'Dance'),
                    { genre: g });
    });
  });

  // Back buttons return to the style grid.
  scr.querySelectorAll('.ss2-back').forEach((b) =>
    b.addEventListener('click', () => showStep(stepStyles)));

  // Collapse (✕) — peek at the avatar without starting anything.
  const collapseBtn = document.getElementById('ss2-collapse');
  if (collapseBtn) collapseBtn.addEventListener('click', collapseDoor);

  // v192: LESSONS-first — the primary CTA opens the structured curriculum
  // (Learn panel) instead of the style→duration quick flow. This makes "what
  // should I do?" obvious: follow a guided lesson. Collapses the front door so
  // the Learn panel is visible over the avatar.
  const learnBtn = document.getElementById('ss2-learn-btn');
  if (learnBtn) learnBtn.addEventListener('click', () => {
    try { unlockAudio(); } catch (e) {}
    try { collapseDoor(); } catch (e) {}
    const openLearn = () => { try { (window.__openLearn || function(){})(); } catch (e) {} };
    // Open immediately, and once more after the door's collapse settles so a
    // slow first paint can never swallow the tap.
    openLearn();
    setTimeout(openLearn, 180);
  });

  // "Just chat with the coach" — the ONLY path that opts into the AI voice.
  document.getElementById('ss2-chat-btn').addEventListener('click', () => {
    try { unlockAudio(); } catch (e) {}
    closeDoor();
    try { openDrawer(); } catch (e) {}
    _greetedThisSession = false;
    // v193: "Just talk" = the live-voice (Gemini) full-duplex conversation
    // when available (the user's preferred natural voice). Falls back to the
    // Azure-spoken greeting if live voice is off/unavailable.
    let s2sOff = false;
    try { s2sOff = localStorage.getItem('coach.s2s.off') === '1'; } catch (e) {}
    if (window.__liveVoiceAvailable && !s2sOff) {
      setTimeout(() => { try { _startS2S(); } catch (e) {} }, 200);
    } else {
      setTimeout(() => { try { playCoachGreeting().catch(() => {}); } catch (e) {} }, 250);
    }
  });

  // Record-&-analyze: pick which style you're dancing -> set the reference
  // clip -> open the file/camera picker (existing #video-input handler runs
  // pose extraction + /api/feedback/compare2d).
  let _analyzeClips = null;
  async function ensureAnalyzeClips() {
    if (_analyzeClips) return _analyzeClips;
    try {
      const r = await fetch(APP_BASE + '/api/motion/list');
      const d = await r.json();
      _analyzeClips = d.motions || [];
    } catch (e) { _analyzeClips = []; }
    return _analyzeClips;
  }
  function clipForGenre(clips, genre) {
    const hit = clips.find((c) => (c.id || '').startsWith(genre));
    return hit ? hit.id : null;
  }
  document.getElementById('ss2-analyze-btn').addEventListener('click', async () => {
    showStep(stepAnalyze);
    analyzeGrid.innerHTML = '<div class="ss2-sub">loading styles…</div>';
    const clips = await ensureAnalyzeClips();
    analyzeGrid.innerHTML = '';
    for (const [g, emoji, name] of ANALYZE_STYLES) {
      const clipId = clipForGenre(clips, g);
      if (!clipId) continue;
      const b = document.createElement('button');
      b.className = 'ss2-tile';
      b.type = 'button';
      b.innerHTML = `<span class="ss2-emoji">${emoji}</span>` +
                    `<span class="ss2-name">${name}</span>`;
      b.addEventListener('click', () => {
        _greetedThisSession = true;
        window.__feedbackClipId = clipId;
        addMsg(`📹 Record or upload a short clip of you dancing ${name} — ` +
               `I'll compare you to a pro and break down exactly what to fix.`,
               'sys');
        closeDoor();
        try { document.getElementById('video-input').click(); } catch (e) {}
      });
      analyzeGrid.appendChild(b);
    }
    if (!analyzeGrid.children.length) {
      analyzeGrid.innerHTML =
        '<div class="ss2-sub">No reference clips available right now — ' +
        'try a guided session instead.</div>';
    }
  });

  if (reopen) reopen.addEventListener('click', openDoor);
  showStep(stepStyles);

  // v225 DEAD-CLICK FIX: Clarity showed ~12% of sessions have "dead clicks" —
  // taps that hit nothing. On this app the #1 culprit is tapping the AVATAR /
  // empty stage when idle (front door closed, nothing playing) expecting
  // something to happen. Convert that dead tap into a useful action: re-open
  // the style picker. Guarded so it NEVER fires during a drag (OrbitControls
  // rotate), while dancing, in a session, or when a pane/door is already open.
  try {
    const _stage = document.getElementById('stage');
    if (_stage) {
      let _dx = 0, _dy = 0, _sx = 0, _sy = 0, _down = false;
      _stage.addEventListener('pointerdown', (e) => {
        _down = true; _sx = e.clientX; _sy = e.clientY; _dx = 0; _dy = 0;
      }, { passive: true });
      _stage.addEventListener('pointermove', (e) => {
        if (!_down) return;
        _dx = Math.max(_dx, Math.abs(e.clientX - _sx));
        _dy = Math.max(_dy, Math.abs(e.clientY - _sy));
      }, { passive: true });
      _stage.addEventListener('pointerup', () => {
        const wasTap = _down && _dx < 8 && _dy < 8;   // not a drag
        _down = false;
        if (!wasTap) return;
        const dancing = !!(window.__player && window.__player.playing);
        const inSession = !!window.__inSession;
        const doorOpen = window.__frontDoorOpen === true;
        let paneOpen = false;
        try {
          paneOpen = ['coach-rail', 'drawer', 'learn-panel'].some((id) => {
            const el = document.getElementById(id);
            return el && (el.classList.contains('show') || el.classList.contains('open') ||
                          (id === 'learn-panel' && !el.hidden));
          });
        } catch (e) {}
        // Only rescue a genuinely IDLE tap — never interrupt playback/panes.
        if (!dancing && !inSession && !doorOpen && !paneOpen) {
          try { _coachTrack('dance_more_styles_opened', { from: 'stage_tap_rescue' }); } catch (e) {}
          try { openDoor(); } catch (e) {}
        }
      }, { passive: true });
    }
  } catch (e) {}

  // ── v-ux9: SIMPLIFIED front door ──────────────────────────────────────
  // One dominant CTA ("Dance for me now") that delivers value in ONE tap —
  // no style pick, no "how long?" — because 89% of visitors were bouncing at
  // the decision screen. Styles + duration + record are tucked behind "More".
  const _TIP_KEY = 'coach.tip.dismissed.v1';
  function _initInlineTip() {
    const tip = document.getElementById('ss2-tip');
    if (!tip) return;
    let seen = false;
    try { seen = localStorage.getItem(_TIP_KEY) === '1'; } catch (e) {}
    if (seen) { tip.style.display = 'none'; return; }
    const x = document.getElementById('ss2-tip-x');
    if (x) x.addEventListener('click', () => {
      tip.style.display = 'none';
      try { localStorage.setItem(_TIP_KEY, '1'); } catch (e) {}
    });
  }
  _initInlineTip();

  // Lightweight non-blocking toast — guides the first-timer without a modal.
  function _coachToast(msg, ms) {
    try {
      let t = document.getElementById('coach-toast');
      if (!t) {
        t = document.createElement('div');
        t.id = 'coach-toast';
        t.style.cssText =
          'position:fixed;left:50%;bottom:104px;transform:translateX(-50%) translateY(12px);' +
          'z-index:9700;max-width:88vw;padding:12px 16px;border-radius:14px;' +
          'background:linear-gradient(135deg,#1c1330,#241546);color:#f4f1fb;' +
          'border:1px solid rgba(192,97,255,.4);box-shadow:0 12px 34px rgba(0,0,0,.5);' +
          'font-size:13.5px;font-weight:650;text-align:center;opacity:0;pointer-events:none;' +
          'transition:opacity .25s ease, transform .25s ease;';
        document.body.appendChild(t);
      }
      t.textContent = msg;
      requestAnimationFrame(() => {
        t.style.opacity = '1';
        t.style.transform = 'translateX(-50%) translateY(0)';
      });
      clearTimeout(window.__coachToastT);
      window.__coachToastT = setTimeout(() => {
        t.style.opacity = '0';
        t.style.transform = 'translateX(-50%) translateY(12px)';
      }, ms || 4200);
    } catch (e) {}
  }

  // PRIMARY: "Teach me to dance" → start a STRUCTURED guided Hip-Hop session
  // (warmup → teach a move step-by-step → build a combo → cool down) in ONE
  // tap. No style pick, no "how long?" — it teaches instead of randomly
  // dancing. Learn (curriculum) stays a separate secondary path.
  const instantBtn = document.getElementById('ss2-instant');
  if (instantBtn) instantBtn.addEventListener('click', () => {
    _greetedThisSession = true;   // a chosen action never auto-greets
    try { unlockAudio(); } catch (e) {}
    try {
      _coachTrack('dance_style_viewed', { style: 'gLH', from: 'teach_start' });
      _coachTrack('dance_instant_start', { mode: 'guided_session' });
    } catch (e) {}
    closeDoor();
    // Structured 5-min guided Hip-Hop session via the agent's session engine.
    try { startGuidedSessionTemplate('quick5_gLH', 5); } catch (e) {}
    _coachToast("I'll teach you — step by step. 🎧 Tap the mic anytime to talk.", 5200);
    // v220: re-arm the once-per-session pill so it shows for THIS session, and
    // kick the retrying hint (it now waits for voice availability internally).
    _voiceHintShown = false;
    try { _syncVoiceCue(); } catch (e) {}
    try { setTimeout(() => { try { _maybeVoiceHint(); } catch (e) {} }, 1500); } catch (e) {}
  });

  // "More styles & options" toggle reveals the full grid + record button.
  const moreToggle = document.getElementById('ss2-more-toggle');
  const moreWrap = document.getElementById('ss2-more-wrap');
  if (moreToggle && moreWrap) moreToggle.addEventListener('click', () => {
    const show = moreWrap.hidden;
    moreWrap.hidden = !show;
    moreToggle.textContent = show ? 'Fewer options ▴' : 'More styles & options ▾';
    if (show) { try { _coachTrack('dance_more_styles_opened', {}); } catch (e) {} }
  });

  // v-ux9: onboarding is now the inline tip (_initInlineTip above); the old
  // blocking overlay is retired.
})();

