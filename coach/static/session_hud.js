// session_hud.js — v27 coach-driven session overlay.
//
// Injects a top-center HUD that shows the current session phase,
// a ring timer, and skip/pause/end controls. Listens for custom
// events dispatched by coach.js (`session:started`, `session:phase`,
// `session:paused`, `session:resumed`, `session:finished`) and
// emits its own (`session:cmd:*`) that coach.js relays to the WS.
//
// Public API on window.__sessionHUD:
//   show(snapshot)        — open the overlay with a session snapshot
//   update(snapshot)      — update phase/timer
//   hide()                — remove from DOM
//   isOpen()              — bool
//
// CSS is inlined so this script is drop-in.

(function () {
  if (window.__sessionHUD) return;

  const STYLE = `
  #session-hud {
    position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
    z-index: 9000; pointer-events: auto;
    background: rgba(12,14,22,0.86); color: #fff;
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
    padding: 10px 14px; display: flex; align-items: center; gap: 12px;
    font-family: -apple-system, "Inter", system-ui, sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    min-width: 320px; max-width: 92vw;
  }
  #session-hud .sh-ring { position: relative; width: 46px; height: 46px; flex: 0 0 46px; }
  #session-hud .sh-ring svg { transform: rotate(-90deg); display: block; }
  #session-hud .sh-ring .sh-ring-bg { stroke: rgba(255,255,255,0.12); }
  #session-hud .sh-ring .sh-ring-fg { stroke: #6ad8ff; transition: stroke-dashoffset 0.25s linear; }
  #session-hud .sh-ring .sh-ring-num {
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
  }
  #session-hud .sh-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1 1 auto; }
  #session-hud .sh-phase { font-size: 14px; font-weight: 600; line-height: 1.1; }
  #session-hud .sh-sub { font-size: 11px; opacity: 0.65; }
  #session-hud .sh-btns { display: flex; gap: 6px; }
  #session-hud button.sh-btn {
    background: rgba(255,255,255,0.08); color: #fff;
    border: 1px solid rgba(255,255,255,0.12); border-radius: 8px;
    padding: 5px 10px; cursor: pointer; font-size: 12px;
    transition: background 0.15s;
  }
  #session-hud button.sh-btn:hover { background: rgba(255,255,255,0.16); }
  #session-hud button.sh-btn.danger:hover { background: rgba(255,80,80,0.25); }
  #session-hud .sh-pill {
    background: rgba(106,216,255,0.12); border: 1px solid rgba(106,216,255,0.35);
    color: #6ad8ff; border-radius: 999px; padding: 2px 8px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  /* v226: overall-progress bar — the #1 completion motivator. A thin fill under
     the HUD showing how far through the WHOLE session the dancer is, so they can
     see they're nearly done and push to finish (was: only a per-phase ring). */
  #session-hud .sh-progress {
    position: absolute; left: 12px; right: 12px; bottom: 5px; height: 3px;
    border-radius: 999px; background: rgba(255,255,255,0.10); overflow: hidden;
  }
  #session-hud .sh-progress > i {
    display: block; height: 100%; width: 0%;
    background: linear-gradient(90deg,#7c3aed,#ec4899);
    border-radius: 999px; transition: width 0.3s linear;
  }
  #session-hud { padding-bottom: 14px; }
  @media (max-width: 520px) {
    #session-hud { min-width: 0; width: calc(100vw - 24px); padding: 8px 10px 14px; gap: 8px; }
    #session-hud .sh-sub { display: none; }
  }
  `;

  function mkSvg() {
    const ns = 'http://www.w3.org/2000/svg';
    const r = 20, c = 2 * Math.PI * r;
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', '46'); svg.setAttribute('height', '46');
    svg.setAttribute('viewBox', '0 0 46 46');
    const bg = document.createElementNS(ns, 'circle');
    bg.setAttribute('cx', '23'); bg.setAttribute('cy', '23');
    bg.setAttribute('r', String(r)); bg.setAttribute('fill', 'none');
    bg.setAttribute('stroke-width', '3'); bg.classList.add('sh-ring-bg');
    const fg = document.createElementNS(ns, 'circle');
    fg.setAttribute('cx', '23'); fg.setAttribute('cy', '23');
    fg.setAttribute('r', String(r)); fg.setAttribute('fill', 'none');
    fg.setAttribute('stroke-width', '3'); fg.setAttribute('stroke-linecap', 'round');
    fg.setAttribute('stroke-dasharray', String(c));
    fg.setAttribute('stroke-dashoffset', '0');
    fg.classList.add('sh-ring-fg');
    svg.appendChild(bg); svg.appendChild(fg);
    return { svg, fg, c };
  }

  function ensureStyle() {
    if (document.getElementById('session-hud-style')) return;
    const s = document.createElement('style');
    s.id = 'session-hud-style'; s.textContent = STYLE;
    document.head.appendChild(s);
  }

  let el = null, ring = null, secsEl = null, phaseEl = null, subEl = null;
  let snap = null, tickHandle = null, paused = false;

  function fmt(s) {
    s = Math.max(0, Math.round(s));
    const m = Math.floor(s / 60), r = s % 60;
    return m + ':' + String(r).padStart(2, '0');
  }

  function build() {
    ensureStyle();
    el = document.createElement('div');
    el.id = 'session-hud';
    el.innerHTML = '';
    const ringWrap = document.createElement('div');
    ringWrap.className = 'sh-ring';
    ring = mkSvg();
    ringWrap.appendChild(ring.svg);
    const num = document.createElement('div');
    num.className = 'sh-ring-num'; secsEl = num;
    ringWrap.appendChild(num);
    el.appendChild(ringWrap);

    const info = document.createElement('div');
    info.className = 'sh-info';
    const top = document.createElement('div');
    top.style.display = 'flex'; top.style.alignItems = 'center'; top.style.gap = '6px';
    const pill = document.createElement('span'); pill.className = 'sh-pill';
    pill.textContent = 'Session'; top.appendChild(pill);
    el._pill = pill;
    phaseEl = document.createElement('span'); phaseEl.className = 'sh-phase';
    top.appendChild(phaseEl);
    info.appendChild(top);
    subEl = document.createElement('div'); subEl.className = 'sh-sub';
    info.appendChild(subEl);
    el.appendChild(info);

    const btns = document.createElement('div'); btns.className = 'sh-btns';
    function btn(label, cmd, danger) {
      const b = document.createElement('button');
      b.className = 'sh-btn' + (danger ? ' danger' : '');
      b.textContent = label;
      b.addEventListener('click', () => {
        if (cmd === 'pause/resume') {
          window.dispatchEvent(new CustomEvent(
            paused ? 'session:cmd:resume' : 'session:cmd:pause'));
        } else {
          window.dispatchEvent(new CustomEvent('session:cmd:' + cmd));
        }
      });
      return b;
    }
    const pauseBtn = btn('⏸', 'pause/resume');
    pauseBtn.setAttribute('title', 'Pause / resume');
    btns.appendChild(pauseBtn);
    const skipBtn = btn('Skip', 'skip');
    skipBtn.setAttribute('title', 'Next phase');
    btns.appendChild(skipBtn);
    btns.appendChild(btn('End', 'end', true));
    el.appendChild(btns);

    // v226: overall-progress bar pinned to the bottom of the HUD.
    const bar = document.createElement('div');
    bar.className = 'sh-progress';
    const barFill = document.createElement('i');
    bar.appendChild(barFill);
    el.appendChild(bar);
    el._barFill = barFill;

    document.body.appendChild(el);
    el._pauseBtn = pauseBtn;
  }

  function tick() {
    if (!snap || !el) return;
    const total = snap.phase_duration_sec || 1;
    let remaining = snap.phase_remaining_sec || 0;
    if (!paused) {
      // Locally decay so the ring is smooth between server updates.
      const now = performance.now();
      const dt = (now - (snap._lastTick || now)) / 1000;
      snap._lastTick = now;
      remaining = Math.max(0, remaining - dt);
      snap.phase_remaining_sec = remaining;
      // v226: also advance the overall elapsed so the session progress bar
      // creeps smoothly (capped at total) between server phase updates.
      if (snap.total_sec) {
        snap.elapsed_total_sec = Math.min(
          snap.total_sec, (snap.elapsed_total_sec || 0) + dt);
      }
    }
    const elapsed = Math.max(0, total - remaining);
    const frac = Math.min(1, Math.max(0, elapsed / total));
    if (ring) ring.fg.setAttribute(
      'stroke-dashoffset', String(ring.c * (1 - frac)));
    if (secsEl) secsEl.textContent = fmt(remaining);
    // v226: drive the overall-session progress bar. Prefer the server's
    // elapsed_total/total; locally advance (see above) so it creeps smoothly.
    if (el && el._barFill) {
      const tSec = snap.total_sec || 0;
      if (tSec > 0) {
        const overall = Math.min(1, Math.max(0, (snap.elapsed_total_sec || 0) / tSec));
        el._barFill.style.width = (overall * 100).toFixed(1) + '%';
      }
    }
  }

  function update(s) {
    if (!s) return;
    if (!el) build();
    snap = Object.assign({}, s, { _lastTick: performance.now() });
    paused = !!s.paused;
    if (phaseEl) phaseEl.textContent = s.phase_label || s.phase_name || '—';
    // v226: pill shows WHICH step of how many, so the dancer knows exactly how
    // far they've come and how little is left — a strong nudge to finish.
    if (el && el._pill) {
      let total = 0;
      if (Array.isArray(s.plan)) total = s.plan.length;
      else if (typeof s.total_phases === 'number') total = s.total_phases;
      const idx = (typeof s.phase_idx === 'number') ? s.phase_idx : -1;
      el._pill.textContent = (total > 0 && idx >= 0)
        ? ('Step ' + (idx + 1) + '/' + total) : 'Session';
    }
    if (subEl) {
      const total = Math.round(s.total_sec || 0);
      const elapsed = Math.round(s.elapsed_total_sec || 0);
      const ti = s.template_title || '';
      subEl.textContent = ti
        ? `${ti} · ${fmt(elapsed)} / ${fmt(total)}`
        : `${fmt(elapsed)} / ${fmt(total)}`;
    }
    if (el._pauseBtn) el._pauseBtn.textContent = paused ? '▶' : '⏸';
    if (tickHandle == null) {
      tickHandle = setInterval(tick, 250);
    }
    tick();
  }

  function hide() {
    if (tickHandle != null) { clearInterval(tickHandle); tickHandle = null; }
    if (el && el.parentNode) el.parentNode.removeChild(el);
    el = null; ring = null; secsEl = null; phaseEl = null; subEl = null;
    snap = null; paused = false;
  }

  window.__sessionHUD = {
    show: update, update, hide,
    isOpen: () => !!el,
  };
})();
