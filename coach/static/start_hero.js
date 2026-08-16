// start_hero.js — v35
// Bottom-anchored "tap a length" card. Designed so the AVATAR stays
// visible (no dim overlay) — the AI is the hero, the card is the
// nudge. Triggers a `dance:hero-shown` window event so coach.js can
// fire an attention move (wave + flourish) in sync.
//
// Flow:
//   1. Show 0.6s after splash fades.
//   2. Click length → forwards to existing #session-start .ss-btn.
//   3. Hide on session.started, re-appear after summary closes.

(function () {
  const DISMISS_KEY = 'dance.hero.dismissed';
  let heroEl = null;
  let shown = false;
  let sessionActive = false;

  function build() {
    if (heroEl) return heroEl;
    const STYLE = `
    /* v35: card pinned to bottom on mobile (avatar stays visible),
       dock to bottom-right on desktop. No backdrop dim — never hide
       the AI. */
    #start-hero {
      position: fixed; left: 0; right: 0; bottom: 0; top: auto;
      z-index: 70;
      display: none; justify-content: center; align-items: flex-end;
      padding: 0 12px 16px;
      pointer-events: none;
      font: 14px/1.5 -apple-system, BlinkMacSystemFont,
            "Inter", "Segoe UI", Roboto, sans-serif;
      opacity: 0; transition: opacity .35s ease, transform .35s ease;
      transform: translateY(20px);
    }
    #start-hero.show { display: flex; opacity: 1; transform: none; }
    /* v77: the full-width container must NEVER swallow taps meant for the
       dock buttons beneath it (mic / voice / chat). Only the card itself
       is interactive. */
    #start-hero { pointer-events: none; }
    #start-hero .sh-card { pointer-events: auto; }
    #start-hero .sh-card {
      max-width: 420px; width: 100%;
      background: rgba(15,12,28,.78);
      backdrop-filter: blur(14px) saturate(140%);
      -webkit-backdrop-filter: blur(14px) saturate(140%);
      border: 1px solid rgba(192,97,255,.30);
      border-radius: 18px;
      padding: 12px 14px 12px;
      box-shadow: 0 12px 40px rgba(0,0,0,.55),
                  0 0 0 1px rgba(192,97,255,.10) inset;
      animation: shIn .5s cubic-bezier(.2,.8,.25,1);
      position: relative;
    }
    @keyframes shIn {
      from { opacity: 0; transform: translateY(24px) scale(.97); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    #start-hero .sh-close {
      position: absolute; top: 6px; right: 8px;
      background: transparent; border: none;
      color: rgba(255,255,255,.55); width: 26px; height: 26px;
      font-size: 18px; line-height: 1; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      border-radius: 50%;
    }
    #start-hero .sh-close:hover { color: #fff; background: rgba(255,255,255,.10); }
    #start-hero .sh-greet {
      margin: 0 0 8px; line-height: 1.25;
    }
    #start-hero .sh-greet-hi {
      display: block; font-size: 19px; font-weight: 800; letter-spacing: -.02em;
      color: #fff;
    }
    #start-hero .sh-greet-sub {
      display: block; font-size: 13px; font-weight: 600; margin-top: 1px;
      background: linear-gradient(90deg, #ffb84d, #ff5aa0 55%, #c061ff);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; color: transparent;
    }
    #start-hero .sh-title {
      font-size: 14.5px; font-weight: 600; letter-spacing: -.01em;
      margin: 0 0 8px; color: #fff;
      display: flex; align-items: center; gap: 8px;
    }
    #start-hero .sh-title .sh-pulse {
      width: 8px; height: 8px; border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 0 rgba(34,197,94,.7);
      animation: shPulse 1.8s ease-out infinite;
    }
    @keyframes shPulse {
      0%   { box-shadow: 0 0 0 0 rgba(34,197,94,.55); }
      70%  { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
      100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }
    #start-hero .sh-row {
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;
      margin-bottom: 8px;
    }
    #start-hero .sh-btn {
      background: linear-gradient(180deg, rgba(192,97,255,.22),
                                          rgba(255,90,160,.12));
      border: 1px solid rgba(192,97,255,.45);
      color: #fff;
      border-radius: 12px;
      padding: 10px 6px 9px;
      cursor: pointer;
      text-align: center; line-height: 1.1;
      transition: transform .12s, background .15s, border-color .15s;
      -webkit-tap-highlight-color: transparent;
    }
    #start-hero .sh-btn:hover, #start-hero .sh-btn:active {
      background: linear-gradient(180deg, rgba(192,97,255,.38),
                                          rgba(255,90,160,.22));
      border-color: rgba(192,97,255,.75);
      transform: translateY(-1px);
    }
    #start-hero .sh-btn .sh-n {
      display: block; font-size: 22px; font-weight: 700; letter-spacing: -.02em;
    }
    #start-hero .sh-btn .sh-u {
      display: block; font-size: 9.5px; opacity: .8; margin-top: 1px;
      text-transform: uppercase; letter-spacing: .08em;
    }
    #start-hero .sh-foot {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11px; color: rgba(255,255,255,.55);
      margin-top: 2px;
    }
    #start-hero .sh-link {
      background: transparent; border: none; color: #c061ff;
      font-size: 11px; padding: 2px 4px; cursor: pointer;
      font-weight: 500;
    }
    #start-hero .sh-link:hover { color: #d684ff; text-decoration: underline; }
    #start-hero .sh-modes { display: flex; flex-direction: column; gap: 8px; margin: 4px 0 6px; }
    #start-hero .sh-hint {
      font-size: 12.5px; line-height: 1.4; color: rgba(255,255,255,.66);
      font-style: italic; margin: 2px 0 10px;
    }
    #start-hero .sh-mode {
      display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
      text-align: left; width: 100%;
      padding: 12px 14px; border-radius: 14px; cursor: pointer;
      background: rgba(255,255,255,.05);
      border: 1px solid rgba(192,97,255,.28);
      color: #fff; transition: background .15s ease, border-color .15s ease, transform .1s ease;
    }
    #start-hero .sh-mode:hover { background: rgba(192,97,255,.16); border-color: rgba(192,97,255,.55); }
    #start-hero .sh-mode:active { transform: scale(.985); }
    #start-hero .sh-mode-emoji { font-size: 20px; line-height: 1; }
    #start-hero .sh-mode-main { font-size: 15px; font-weight: 700; letter-spacing: -.01em; }
    #start-hero .sh-mode-sub { font-size: 11.5px; color: rgba(255,255,255,.62); font-weight: 500; }
    #start-hero .sh-back {
      background: transparent; border: none; color: rgba(255,255,255,.7);
      font-size: 22px; line-height: 1; cursor: pointer; margin-right: 4px;
      width: 24px; height: 24px; border-radius: 50%;
    }
    #start-hero .sh-back:hover { color: #fff; background: rgba(255,255,255,.10); }

    /* v139: primary "Let's start" CTA. One clear tap that unlocks audio
       (so the coach can speak) AND starts the mic (so she can hear you) —
       browsers require a user gesture for both, so we give the user an
       obvious thing to tap instead of silently waiting. */
    #start-hero .sh-cta {
      width: 100%; display: flex; align-items: center; justify-content: center;
      gap: 8px; margin: 2px 0 10px;
      padding: 13px 14px; border-radius: 14px; cursor: pointer;
      font-size: 15.5px; font-weight: 800; letter-spacing: -.01em; color: #fff;
      border: 1px solid rgba(192,97,255,.55);
      background: linear-gradient(180deg, #c061ff, #ff5aa0);
      box-shadow: 0 8px 24px rgba(192,97,255,.35);
      -webkit-tap-highlight-color: transparent;
      transition: transform .12s ease, box-shadow .15s ease, filter .15s ease;
    }
    #start-hero .sh-cta:hover { filter: brightness(1.06); }
    #start-hero .sh-cta:active { transform: scale(.98); }

    /* v77: on phones the dock lives at the very bottom; lift the
       start-hero card above it so it never covers the controls. */
    @media (max-width: 759px) {
      #start-hero { padding: 0 12px 84px; }
    }
    @media (min-width: 760px) {
      #start-hero { justify-content: flex-end; padding: 0 24px 24px; }
      #start-hero .sh-card { max-width: 340px; }
    }
    `;
    const s = document.createElement('style');
    s.id = 'start-hero-style';
    s.textContent = STYLE;
    document.head.appendChild(s);

    heroEl = document.createElement('div');
    heroEl.id = 'start-hero';
    heroEl.setAttribute('role', 'dialog');
    heroEl.setAttribute('aria-modal', 'false');
    heroEl.setAttribute('aria-labelledby', 'sh-title');
    heroEl.innerHTML = `
      <div class="sh-card">
        <button class="sh-close" aria-label="Dismiss">×</button>
        <!-- STEP 1: talk-first home -->
        <div class="sh-step sh-step-mode">
          <div class="sh-greet">
            <span class="sh-greet-hi" id="sh-greet-hi">Hi!</span>
            <span class="sh-greet-sub">Ready to move?</span>
          </div>
          <div id="sh-title" class="sh-title">
            <span class="sh-pulse"></span>
            I'm listening — just talk
          </div>
          <div class="sh-hint">
            Say hi out loud and I'll answer, hands-free. Or tap below for a
            guided dance.
          </div>
          <button class="sh-cta" data-act="lets-start">
            <span>▶</span><span>Let's start — say hi!</span>
          </button>
          <div class="sh-modes">
            <button class="sh-mode" data-mode="session">
              <span class="sh-mode-emoji">\u{1f57a}</span>
              <span class="sh-mode-main">Start a dance session</span>
              <span class="sh-mode-sub">Guided: warm up \u2192 learn \u2192 combo \u2192 cool down</span>
            </button>
          </div>
          <div class="sh-foot">
            <button class="sh-link" data-act="tour">How it works</button>
          </div>
        </div>
        <!-- STEP 2: pick a length -->
        <div class="sh-step sh-step-len" hidden>
          <div class="sh-title">
            <button class="sh-back" aria-label="Back">\u2039</button>
            How long?
          </div>
          <div class="sh-row">
            <button class="sh-btn" data-mins="5"  aria-label="5 minute session">
              <span class="sh-n">5</span><span class="sh-u">min</span>
            </button>
            <button class="sh-btn" data-mins="10" aria-label="10 minute session">
              <span class="sh-n">10</span><span class="sh-u">min</span>
            </button>
            <button class="sh-btn" data-mins="20" aria-label="20 minute session">
              <span class="sh-n">20</span><span class="sh-u">min</span>
            </button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(heroEl);

    const stepMode = heroEl.querySelector('.sh-step-mode');
    const stepLen = heroEl.querySelector('.sh-step-len');
    function showStep(which) {
      if (which === 'len') { stepMode.hidden = true; stepLen.hidden = false; }
      else { stepMode.hidden = false; stepLen.hidden = true; }
    }
    // STEP 1 modes.
    const ctaBtn = heroEl.querySelector('.sh-cta[data-act="lets-start"]');
    if (ctaBtn) ctaBtn.addEventListener('click', () => {
        // v139: THE gesture. Unlock audio + start the mic (both need a user
        // gesture on mobile). coach.js listens for this and does the work;
        // the greeting also auto-plays off this same tap.
        try { if (typeof unlockAudio === 'function') unlockAudio(); } catch (e) {}
        try { window.dispatchEvent(new CustomEvent('dance:lets-start')); } catch (e) {}
        hide(true);
      });
    heroEl.querySelector('.sh-mode[data-mode="session"]')
      .addEventListener('click', () => {
        // v109b: a guided session is the AZURE-NARRATED coaching mode.
        // Do NOT also start Gemini S2S here — running both made two
        // voices talk over each other (the "volume pumps / says things
        // twice" bug). One voice owns audio per mode: session => Azure,
        // "Just talk" => Gemini. We still unlock audio so Azure can play.
        try { if (typeof unlockAudio === 'function') unlockAudio(); } catch (e) {}
        showStep('len');
      });
    // v113: TALK-FIRST. Live voice ("just talk") is the DEFAULT and is
    // always on (it auto-arms on the first gesture via initLiveVoice).
    // We no longer show a separate "Just talk" card; the only explicit
    // choice on the home card is starting a guided dance session. The
    // 'talk' button was removed, so its old click handler is gone too —
    // any tap anywhere already counts as the gesture that starts S2S.
    // STEP 2 lengths -> forward to the drawer chips.
    heroEl.querySelectorAll('.sh-btn[data-mins]').forEach(b => {
      b.addEventListener('click', () => {
        const mins = b.getAttribute('data-mins');
        const target = document.querySelector(
          `#session-start .ss-btn[data-act="session-${mins}"]`);
        if (target) target.click();
        else console.warn('[hero] no target for', mins);
        hide();
      });
    });
    const backBtn = heroEl.querySelector('.sh-back');
    if (backBtn) backBtn.addEventListener('click', () => showStep('mode'));
    heroEl.querySelector('.sh-close').addEventListener('click', () => {
      hide(true);
    });
    heroEl.querySelector('[data-act="tour"]').addEventListener('click', () => {
      const tour = document.getElementById('tour');
      if (tour) tour.classList.add('show');
    });
    // v115: personalised, time-of-day greeting (matches the brand mockup).
    try {
      const h = new Date().getHours();
      const part = h < 5 ? 'Hello' : h < 12 ? 'Good morning'
                 : h < 17 ? 'Good afternoon' : h < 22 ? 'Good evening' : 'Hello';
      const sun  = (h >= 5 && h < 18) ? ' \u2600\uFE0F' : ' \u2728';
      let nm = '';
      try { nm = (localStorage.getItem('user_name') || '').trim(); } catch (e) {}
      nm = nm ? (', ' + nm.split(' ')[0]) : '';
      const el = heroEl.querySelector('#sh-greet-hi');
      if (el) el.textContent = part + nm + '!' + sun;
    } catch (e) {}
    // Always reset to the mode step when (re)shown.
    showStep('mode');
    return heroEl;
  }

  function show() {
    if (sessionActive) return;
    // v100: never re-show once dismissed/chosen this session (the splash
    // watcher AND a 12s fallback both call show() -> the card appeared twice).
    try {
      const d = parseInt(localStorage.getItem(DISMISS_KEY) || '0', 10);
      if (d && (Date.now() - d) < 30 * 60 * 1000) return;
    } catch (e) {}
    if (shown) return;
    build();
    shown = true;
    requestAnimationFrame(() => heroEl.classList.add('show'));
    // v35: tell coach.js to fire an attention move so the avatar
    // visibly notices the user the moment the hero appears.
    try {
      window.dispatchEvent(new CustomEvent('dance:hero-shown'));
    } catch {}
  }

  function hide(persistDismiss) {
    if (!heroEl) return;
    heroEl.classList.remove('show');
    shown = false;
    if (persistDismiss) {
      try { localStorage.setItem(DISMISS_KEY, String(Date.now())); } catch {}
    }
  }

  function watchSplash() {
    const splash = document.getElementById('splash');
    if (!splash) { setTimeout(show, 1000); return; }
    const tick = () => {
      if (splash.classList.contains('fadeout') ||
          splash.classList.contains('gone')) {
        setTimeout(show, 600);
        return;
      }
      setTimeout(tick, 150);
    };
    tick();
    setTimeout(() => { if (!shown && !sessionActive) show(); }, 12000);
  }

  function watchForSummaryClose() {
    let waited = 0;
    const poll = setInterval(() => {
      waited += 250;
      const card = document.getElementById('dance-summary-card');
      if (!card) {
        clearInterval(poll);
        setTimeout(() => { if (!sessionActive) show(); }, 400);
      } else if (waited > 60000) {
        clearInterval(poll);
      }
    }, 250);
  }

  function onWsEvent(m) {
    if (!m || !m.type) return;
    if (m.type === 'session.started') {
      sessionActive = true;
      hide();
    } else if (m.type === 'session.finished') {
      sessionActive = false;
      watchForSummaryClose();
    }
  }

  window.DanceStartHero = { show, hide, onWsEvent };
  if (window.DanceAnalytics && typeof window.DanceAnalytics.onEvent === 'function') {
    const orig = window.DanceAnalytics.onEvent.bind(window.DanceAnalytics);
    window.DanceAnalytics.onEvent = function (m) {
      try { onWsEvent(m); } catch {}
      return orig(m);
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', watchSplash);
  } else {
    watchSplash();
  }
})();
