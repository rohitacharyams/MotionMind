// wavetune.js — v38 live-tuner for the procedural hello pose
//
// Open ?preview=wavetune on the live site. Drag any slider, the
// avatar's pose updates IN REAL TIME (no envelope, no decay). When
// it looks right, hit "Save as hello" — params go to localStorage
// (`dance.hello.tuned`) AND a `hello.tuned.save` telemetry event so
// the next ship can bake them in.
//
// Why this exists: prior versions of playWave() / playBigWave()
// CALLED THE SAME apply(name, axis, angle) helper once per axis. The
// helper resets the bone from rest each call, so every multi-axis
// pose ended up with only its LAST axis applied. That's why every
// "hello" since v33 looked weak — the arm-LIFT axis was always
// overwritten by the hand-WIGGLE axis. v38 fixed compose, this
// panel lets the founder dial in the actual numbers without me
// guessing rig sign conventions.

(function () {
  const KEY = 'dance.hello.tuned';

  function qsGet(k) {
    try { return new URLSearchParams(location.search).get(k); }
    catch { return null; }
  }
  if (qsGet('preview') !== 'wavetune') return;

  // ── slider definitions ───────────────────────────────────────────
  // Each slider: id → { label, min, max, step, init, hint }
  // Values are radians (1 rad ≈ 57°). Init values are my best guess
  // for "overhead Statue-of-Liberty wave"; founder will adjust.
  const SLIDERS = [
    { id: 'upX',      label: 'Shoulder forward/up (X)',  min: -2.8, max: 2.8, step: 0.05, init: -1.6,
      hint: 'negative usually lifts the arm forward then up' },
    { id: 'upY',      label: 'Shoulder twist (Y)',       min: -1.5, max: 1.5, step: 0.05, init:  0.0,
      hint: 'rotates the arm around its long axis (palm in / out)' },
    { id: 'upZ',      label: 'Shoulder out (Z) × side',  min: -2.0, max: 2.0, step: 0.05, init: -1.0,
      hint: 'multiplied by +1 for right, -1 for left. Sign convention varies by rig.' },
    { id: 'lowX',     label: 'Elbow bend (X)',           min: -2.5, max: 0.5, step: 0.05, init: -1.0,
      hint: 'negative bends the forearm toward the chest' },
    { id: 'lowY',     label: 'Forearm twist baseline (Y)', min: -1.0, max: 1.0, step: 0.05, init: 0.0,
      hint: 'offset for the hand wiggle' },
    { id: 'wigAmp',   label: 'Hand wiggle amplitude',    min: 0.0,  max: 1.5, step: 0.05, init: 0.85,
      hint: 'how far the hand swings side-to-side' },
    { id: 'wigHz',    label: 'Hand wiggle speed (Hz)',   min: 0.5,  max: 4.0, step: 0.1,  init: 2.3,
      hint: 'cycles per second' },
    { id: 'headTilt', label: 'Head tilt (Z) × side',     min: -0.6, max: 0.6, step: 0.02, init: 0.18,
      hint: 'tilt head toward the wave-side shoulder' },
    { id: 'headTurn', label: 'Head turn (Y) × side',     min: -0.6, max: 0.6, step: 0.02, init: 0.20,
      hint: 'turn head toward camera' },
  ];

  function loadSaved() {
    try { return JSON.parse(localStorage.getItem(KEY) || 'null'); }
    catch { return null; }
  }
  function saveTuned(p) {
    try { localStorage.setItem(KEY, JSON.stringify(p)); } catch {}
    try {
      const body = JSON.stringify({
        event: 'hello.tuned.save',
        cid: localStorage.getItem('dance.cid') || 'c_anon',
        path: location.pathname,
        ts: Date.now(),
        props: p,
      });
      if (navigator.sendBeacon) {
        navigator.sendBeacon('/api/track',
          new Blob([body], { type: 'application/json' }));
      } else {
        fetch('/api/track', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body, keepalive: true, credentials: 'omit',
        }).catch(() => {});
      }
    } catch {}
  }

  // ── style ────────────────────────────────────────────────────────
  const css = `
  #wavetune {
    position: fixed; top: 12px; right: 12px; bottom: 12px;
    width: min(380px, 92vw); z-index: 9999;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #fff;
    background: rgba(10,12,18,0.86);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 14px 14px 12px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.55);
    overflow-y: auto;
  }
  #wavetune h3 { margin: 0 0 4px; font-size: 14px; letter-spacing: 0.3px; }
  #wavetune .wt-sub { font-size: 11px; opacity: 0.7; margin: 0 0 10px; }
  #wavetune .wt-row { margin: 8px 0 4px; }
  #wavetune .wt-row label { display: block; font-size: 11px;
    font-weight: 700; margin-bottom: 2px; }
  #wavetune .wt-row .wt-val { float: right; opacity: 0.7;
    font-family: ui-monospace, Menlo, monospace; }
  #wavetune .wt-row input[type=range] { width: 100%; }
  #wavetune .wt-hint { font-size: 10px; opacity: 0.55; margin-top: 1px; }
  #wavetune .wt-toggles { display: flex; gap: 6px;
    margin: 10px 0 6px; flex-wrap: wrap; }
  #wavetune .wt-toggles button {
    flex: 1; padding: 6px 8px; border-radius: 8px;
    background: rgba(255,255,255,0.08); color: #fff;
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 11px; font-weight: 600; cursor: pointer; }
  #wavetune .wt-toggles button.on {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    border-color: #16a34a; }
  #wavetune .wt-cta { display: flex; gap: 6px; margin-top: 12px; }
  #wavetune .wt-cta button {
    flex: 1; padding: 10px 8px; border-radius: 9px;
    font-weight: 800; font-size: 12px; cursor: pointer;
    border: 0; }
  #wavetune .wt-cta .wt-save { background: #fff; color: #0a0c12; }
  #wavetune .wt-cta .wt-play { background: #3b82f6; color: #fff; }
  #wavetune .wt-cta .wt-reset { background: rgba(255,255,255,0.10);
    color: #fff; border: 1px solid rgba(255,255,255,0.20); }
  #wavetune .wt-status { font-size: 11px; opacity: 0.7;
    margin-top: 8px; text-align: center; min-height: 14px; }
  `;
  const st = document.createElement('style');
  st.appendChild(document.createTextNode(css));
  document.head.appendChild(st);

  // ── DOM ─────────────────────────────────────────────────────────
  const card = document.createElement('div');
  card.id = 'wavetune';
  card.innerHTML = `
    <h3>Wave tuner</h3>
    <div class="wt-sub">Drag sliders. Avatar updates live. Save when it looks like a hello.</div>
    <div class="wt-toggles">
      <button id="wt-side-r" class="on">Right hand</button>
      <button id="wt-side-l">Left hand</button>
      <button id="wt-both">Both arms: off</button>
    </div>
    <div id="wt-sliders"></div>
    <div class="wt-cta">
      <button class="wt-play" id="wt-play">▶ Play 4s</button>
      <button class="wt-reset" id="wt-reset">Reset</button>
      <button class="wt-save" id="wt-save">Save as hello</button>
    </div>
    <div class="wt-status" id="wt-status">live preview ON</div>
  `;
  document.body.appendChild(card);

  // Hide the start hero so it doesn't cover the avatar.
  try {
    const h = document.querySelector('#start-hero');
    if (h) h.style.display = 'none';
  } catch {}

  // ── build sliders ────────────────────────────────────────────────
  const sliderDiv = card.querySelector('#wt-sliders');
  const inputs = {};
  const saved = loadSaved();

  SLIDERS.forEach((s) => {
    const init = (saved && typeof saved[s.id] === 'number') ? saved[s.id] : s.init;
    const wrap = document.createElement('div');
    wrap.className = 'wt-row';
    wrap.innerHTML = `
      <label>${s.label} <span class="wt-val" id="wt-val-${s.id}">${init.toFixed(2)}</span></label>
      <input type="range" id="wt-in-${s.id}"
             min="${s.min}" max="${s.max}" step="${s.step}" value="${init}">
      <div class="wt-hint">${s.hint}</div>
    `;
    sliderDiv.appendChild(wrap);
    const inp = wrap.querySelector('input');
    const val = wrap.querySelector('.wt-val');
    inputs[s.id] = inp;
    inp.addEventListener('input', () => {
      val.textContent = (+inp.value).toFixed(2);
      apply();
    });
  });

  let side = 'right';
  let bothArms = false;

  const sideR = card.querySelector('#wt-side-r');
  const sideL = card.querySelector('#wt-side-l');
  const both  = card.querySelector('#wt-both');
  sideR.onclick = () => { side = 'right'; sideR.classList.add('on'); sideL.classList.remove('on'); apply(); };
  sideL.onclick = () => { side = 'left';  sideL.classList.add('on'); sideR.classList.remove('on'); apply(); };
  both.onclick  = () => { bothArms = !bothArms; both.classList.toggle('on', bothArms);
                          both.textContent = 'Both arms: ' + (bothArms ? 'on' : 'off'); apply(); };

  function currentPose() {
    const p = { side, bothArms };
    SLIDERS.forEach((s) => { p[s.id] = +inputs[s.id].value; });
    return p;
  }

  const status = card.querySelector('#wt-status');

  function apply() {
    if (!window.__danceLife) { status.textContent = 'avatar runtime not ready'; return; }
    const p = currentPose();
    try { window.__danceLife.holdWavePose(p); status.textContent = 'live preview ON — pose updating'; }
    catch (e) { status.textContent = 'error: ' + (e.message || e); }
  }

  card.querySelector('#wt-play').onclick = () => {
    if (!window.__danceLife) return;
    const p = currentPose();
    try {
      window.__danceLife.clearWavePose();
      window.__danceLife.setTunedPose(p);
      // Use playBigWave if pose looks "big" (large upX), else playWave.
      const big = Math.abs(p.upX) > 1.0 || Math.abs(p.upZ) > 0.6;
      if (big) {
        window.__danceLife.setMood && window.__danceLife.setMood('happy', 0.75);
        window.__danceLife.playBigWave({ side: p.side, duration: 4.0 });
      } else {
        window.__danceLife.playWave({ side: p.side, duration: 3.0 });
      }
      status.textContent = 'playing 4s preview…';
      // After play, restore live-hold mode so sliders keep working.
      setTimeout(() => { window.__danceLife.setTunedPose(null); apply(); }, 4200);
    } catch (e) { status.textContent = 'error: ' + (e.message || e); }
  };

  card.querySelector('#wt-reset').onclick = () => {
    SLIDERS.forEach((s) => {
      inputs[s.id].value = s.init;
      card.querySelector('#wt-val-' + s.id).textContent = s.init.toFixed(2);
    });
    apply();
    status.textContent = 'reset to defaults';
  };

  card.querySelector('#wt-save').onclick = () => {
    const p = currentPose();
    saveTuned(p);
    try { window.__danceLife.setTunedPose(p); } catch {}
    status.textContent = 'saved! params posted to telemetry';
  };

  // Wait for the avatar runtime then push the initial pose.
  function waitAndApply(tries = 60) {
    if (window.__danceLife) { apply(); return; }
    if (tries <= 0) { status.textContent = 'avatar runtime never appeared'; return; }
    setTimeout(() => waitAndApply(tries - 1), 200);
  }
  waitAndApply();
})();
