// cliptest.js — v41 in-browser clip tester.
//
// Open ?preview=cliptest on the live site to get a small panel that
// lets you PLAY ANY clipId on the live avatar in 1 second. Use this
// to verify motions yourself instead of relying on me to render
// stick figures locally. Type a clip id, hit play. Hit a preset
// button to skip the typing.
//
// Why this exists: I've shipped multiple wrong "wave" clips because
// I trusted auto-generated metadata files instead of seeing the
// motion. This panel lets you SEE every clip on the real rig.

(function () {
  function qsGet(k) {
    try { return new URLSearchParams(location.search).get(k); }
    catch { return null; }
  }
  if (qsGet('preview') !== 'cliptest') return;

  // Presets: the new wave + a few visually distinct clips for engagement.
  // (Quant-verified candidates from _wave_scan_v40.json + manually
  // chosen variety for the dance demos.)
  const PRESETS = [
    ['wave_hello_mx',  'Hello Wave (Mixamo)'],
    ['cmu_105_105_53', 'Old "Wave Hello" (mislabelled)'],
    ['cmu_49_49_08',   'Top-scoring CMU wave-ish'],
    ['cmu_106_106_16', 'CMU #2 arm motion'],
    ['cmu_06_06_11',   'CMU #3 arm motion'],
    ['cmu_105_105_15', 'Casual Wave (CMU)'],
    ['cmu_105_105_01', 'Casual Walk'],
    ['cmu_105_105_05', 'Side-to-Side Bounce'],
    ['cmu_105_105_07', 'Casual Turn'],
  ];

  const css = `
  #cliptest {
    position: fixed; right: 12px; top: 12px; z-index: 9999;
    width: min(380px, 92vw);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #fff;
    background: rgba(10,12,18,0.86);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 14px 14px 12px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.55);
    max-height: calc(100vh - 24px); overflow-y: auto;
  }
  #cliptest h3 { margin: 0 0 6px; font-size: 14px; letter-spacing: 0.3px; }
  #cliptest .ct-sub { font-size: 11px; opacity: 0.7; margin: 0 0 10px; }
  #cliptest .ct-input { display: flex; gap: 6px; margin-bottom: 8px; }
  #cliptest input[type=text] {
    flex: 1; padding: 7px 8px; border-radius: 7px;
    background: rgba(255,255,255,0.10); color: #fff;
    border: 1px solid rgba(255,255,255,0.18);
    font: 13px ui-monospace, Menlo, monospace;
  }
  #cliptest button.ct-go {
    padding: 7px 14px; border-radius: 7px;
    background: #fff; color: #0a0c12; font-weight: 800; cursor: pointer;
    border: 0; font-size: 13px;
  }
  #cliptest .ct-toggle { display: flex; gap: 4px; margin: 4px 0 6px; }
  #cliptest .ct-toggle button {
    flex: 1; padding: 5px 6px; border-radius: 6px;
    background: rgba(255,255,255,0.08); color: #fff;
    border: 1px solid rgba(255,255,255,0.18);
    font: 600 11px system-ui; cursor: pointer;
  }
  #cliptest .ct-toggle button.on {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    border-color: #16a34a;
  }
  #cliptest .ct-list { display: flex; flex-direction: column; gap: 4px; }
  #cliptest .ct-list button {
    padding: 8px 10px; border-radius: 7px;
    background: rgba(255,255,255,0.08); color: #fff;
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 12px; font-weight: 600; cursor: pointer;
    text-align: left;
  }
  #cliptest .ct-list button:hover { background: rgba(255,255,255,0.18); }
  #cliptest .ct-list .ct-id {
    font: 11px ui-monospace, Menlo, monospace; opacity: 0.7;
    display: block; margin-top: 2px;
  }
  #cliptest .ct-status { font-size: 11px; opacity: 0.7;
    text-align: center; margin-top: 8px; min-height: 14px;
    font-family: ui-monospace, Menlo, monospace; }
  `;
  const st = document.createElement('style');
  st.appendChild(document.createTextNode(css));
  document.head.appendChild(st);

  const card = document.createElement('div');
  card.id = 'cliptest';
  card.innerHTML = `
    <h3>Clip tester</h3>
    <div class="ct-sub">Type any clipId or tap a preset. The avatar
       plays it on the same rig users see.</div>
    <div class="ct-input">
      <input id="ct-in" type="text" placeholder="wave_hello_mx" value="wave_hello_mx">
      <button class="ct-go" id="ct-go">Play</button>
    </div>
    <div class="ct-toggle">
      <button id="ct-loop">loop: off</button>
      <button id="ct-face" class="on">+ face hello: on</button>
    </div>
    <div class="ct-list" id="ct-list"></div>
    <div class="ct-status" id="ct-status">idle</div>
  `;
  document.body.appendChild(card);

  // Hide hero so we can see avatar
  try {
    const h = document.querySelector('#start-hero');
    if (h) h.style.display = 'none';
  } catch {}

  let loopOn = false;
  let faceOn = true;
  const loopBtn = card.querySelector('#ct-loop');
  const faceBtn = card.querySelector('#ct-face');
  loopBtn.onclick = () => { loopOn = !loopOn; loopBtn.classList.toggle('on', loopOn);
                            loopBtn.textContent = 'loop: ' + (loopOn ? 'on' : 'off'); };
  faceBtn.onclick = () => { faceOn = !faceOn; faceBtn.classList.toggle('on', faceOn);
                            faceBtn.textContent = '+ face hello: ' + (faceOn ? 'on' : 'off'); };

  const status = card.querySelector('#ct-status');
  const input = card.querySelector('#ct-in');

  function play(id) {
    id = (id || '').trim();
    if (!id) { status.textContent = 'enter a clip id'; return; }
    input.value = id;
    if (typeof window.__danceLoadAndPlay !== 'function') {
      status.textContent = 'avatar runtime not ready, try again in 2s'; return;
    }
    status.textContent = 'playing: ' + id + (loopOn ? '  (loop)' : '');
    try {
      if (faceOn && window.__danceLife?.playHello) {
        window.__danceLife.playHello({ duration: 2.4 });
      }
      window.__danceLoadAndPlay(id, { loop: loopOn, speed: 1.0 });
    } catch (e) {
      status.textContent = 'FAIL: ' + (e?.message || e);
    }
  }

  card.querySelector('#ct-go').onclick = () => play(input.value);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') play(input.value); });

  const list = card.querySelector('#ct-list');
  PRESETS.forEach(([id, title]) => {
    const b = document.createElement('button');
    b.innerHTML = title + `<span class="ct-id">${id}</span>`;
    b.onclick = () => play(id);
    list.appendChild(b);
  });
})();
