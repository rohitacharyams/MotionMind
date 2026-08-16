// hello_preview.js — v37 greeting-clip A/B viewer
//
// Use: open ?preview=hello on the live site. Tap a candidate, the
// avatar plays it on the same camera the real users see. When you
// pick a winner, hit "Use this hello" — id is saved to localStorage
// AND emitted to /api/track so the next ship can wire it as default.
//
// v37 fix: dropped the four rigging-test "arm wave" clips (subjects
// 02/05 — those are mocap calibration arms, not greetings). Added
// the actual literal "Wave Hello" mocap (cmu_105_105_53) and a
// procedural BIG two-arm wave with smile + head turn.

(function () {
  const KEY = 'dance.greet.pick';

  // ── candidates ───────────────────────────────────────────────────
  //   type 'clip'  → played through MotionPlayer (real mocap asset)
  //   type 'proc'  → run a function on `life` (procedural — no asset)
  const CANDIDATES = [
    { type: 'proc', id: 'p_big', title: 'P · BIG WAVE',
      blurb: 'Procedural: both arms up, head turn, smile. The most "Pixar" option — instant attention grab.',
      run: (life) => { try { life && life.setMood && life.setMood('happy', 0.7); } catch {}
                       try { life && life.playBigWave && life.playBigWave({ duration: 4.0 }); } catch {} } },
    { type: 'clip', id: 'cmu_105_105_53', title: '1 · Wave Hello (mocap)',
      blurb: 'Real motion-capture greeting: right arm up, head tilt + turn, both arms reach out. 8 counts.' },
    { type: 'clip', id: 'cmu_105_105_15', title: '2 · Casual Wave',
      blurb: 'Calmer single-arm wave from the same actor as Wave Hello.' },
    { type: 'proc', id: 'p_small', title: 'P · Small wave',
      blurb: 'Old v33 procedural — single right arm, chest height. Subtle, what was shipped.',
      run: (life) => { try { life && life.playWave && life.playWave({ side: 'right', duration: 2.6 }); } catch {} } },
    { type: 'clip', id: 'cmu_01_01_01', title: '3 · Hip Swivel (legacy)',
      blurb: 'Old v36 default. Mostly hip movement, very little arm — kept for comparison.' },
  ];

  function qsGet(k) {
    try { return new URLSearchParams(location.search).get(k); }
    catch { return null; }
  }
  if (qsGet('preview') !== 'hello') return;   // gated by URL

  // ── style ────────────────────────────────────────────────────────
  const css = `
  #hello-preview {
    position: fixed; left: 50%; bottom: 18px;
    transform: translateX(-50%);
    z-index: 9999;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #fff; pointer-events: auto;
    background: rgba(10,12,18,0.78);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px 16px 12px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.45);
    width: min(94vw, 520px);
  }
  #hello-preview .hp-row { display: grid;
    grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0 10px; }
  #hello-preview button.hp-cand {
    padding: 10px 8px; border-radius: 10px;
    background: rgba(255,255,255,0.08);
    color: #fff; border: 1px solid rgba(255,255,255,0.18);
    font-size: 13px; font-weight: 600; cursor: pointer;
  }
  #hello-preview button.hp-cand.hp-active {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    border-color: #16a34a; color: #fff; }
  #hello-preview button.hp-pick {
    width: 100%; padding: 12px 14px; border-radius: 10px;
    background: #fff; color: #0a0c12; border: 0;
    font-weight: 800; font-size: 14px; cursor: pointer;
    box-shadow: 0 2px 8px rgba(255,255,255,0.18);
  }
  #hello-preview .hp-title { font-size: 14px; font-weight: 800;
    letter-spacing: 0.2px; margin: 0 0 4px; text-align: center; }
  #hello-preview .hp-blurb { font-size: 12px; opacity: 0.78;
    margin: 0 0 6px; text-align: center; min-height: 30px; }
  #hello-preview .hp-meta { font-size: 11px; opacity: 0.55;
    text-align: center; margin-top: 6px; }
  `;
  const st = document.createElement('style');
  st.id = 'hello-preview-style';
  st.appendChild(document.createTextNode(css));
  document.head.appendChild(st);

  // ── DOM ─────────────────────────────────────────────────────────
  const card = document.createElement('div');
  card.id = 'hello-preview';
  card.innerHTML = `
    <div class="hp-title">Pick the avatar's hello</div>
    <div class="hp-blurb" id="hp-blurb">Tap a candidate. The avatar
       plays it on loop. When you find the one that feels most alive,
       hit "Use this hello".</div>
    <div class="hp-row" id="hp-row"></div>
    <button class="hp-pick" id="hp-pick">Use this hello</button>
    <div class="hp-meta" id="hp-meta">no pick yet</div>
  `;
  document.body.appendChild(card);

  const row = card.querySelector('#hp-row');
  const blurb = card.querySelector('#hp-blurb');
  const meta = card.querySelector('#hp-meta');
  let active = null;

  CANDIDATES.forEach((c) => {
    const b = document.createElement('button');
    b.className = 'hp-cand'; b.textContent = c.title;
    b.dataset.id = c.id;
    b.onclick = () => playCandidate(c);
    row.appendChild(b);
  });

  function highlight(id) {
    row.querySelectorAll('button.hp-cand').forEach((el) => {
      el.classList.toggle('hp-active', el.dataset.id === id);
    });
  }

  // Hide the start hero so it doesn't cover the avatar in preview mode.
  function hideHero() {
    try {
      const h = document.querySelector('#start-hero');
      if (h) h.style.display = 'none';
      const ss = document.querySelector('#start-hero-style');
      // leave style sheet — only the element matters
    } catch {}
  }

  function waitForRuntime(cb, tries = 40) {
    // coach.js exposes loadAndPlayClip as __danceLoadAndPlay and the
    // procedural avatar as __danceLife once player is ready.
    const haveClip = typeof window.__danceLoadAndPlay === 'function';
    const haveLife = !!window.__danceLife;
    if (haveClip || haveLife) return cb();
    if (tries <= 0) {
      blurb.textContent = 'avatar runtime not ready — refresh the page';
      return;
    }
    setTimeout(() => waitForRuntime(cb, tries - 1), 250);
  }

  function playCandidate(c) {
    hideHero();
    active = c;
    highlight(c.id);
    blurb.textContent = c.blurb;
    meta.textContent = 'playing: ' + c.id;
    waitForRuntime(() => {
      try {
        if (c.type === 'proc') {
          c.run(window.__danceLife);
        } else {
          window.__danceLoadAndPlay(c.id, { loop: true, speed: 1.0 });
        }
      } catch (e) {
        blurb.textContent = 'failed to play ' + c.id + ' — ' +
                            (e && e.message ? e.message : e);
      }
    });
  }

  card.querySelector('#hp-pick').onclick = () => {
    if (!active) { meta.textContent = 'pick a candidate first'; return; }
    try { localStorage.setItem(KEY, active.id); } catch {}
    meta.textContent = 'saved: ' + active.id + ' — telemetry sent';
    try {
      const body = JSON.stringify({
        event: 'greet.pick',
        cid: localStorage.getItem('dance.cid') || 'c_anon',
        path: location.pathname,
        ts: Date.now(),
        props: { clip_id: active.id, title: active.title,
                 type: active.type },
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
  };

  // No auto-play in v37 — the founder picks. Auto-play of a bad clip
  // is what made v36 look broken. Card is visible immediately; on
  // each tap we wait for the runtime to expose __danceLife /
  // __danceLoadAndPlay then trigger.
})();
