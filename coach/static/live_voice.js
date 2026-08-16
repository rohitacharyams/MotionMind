// live_voice.js — live voice (speech-to-speech) browser client.
//
// Captures the mic as PCM16 @ 16 kHz, streams it over /ws/voice to the
// server-side live voice relay, and plays back the coach's spoken
// audio (PCM16 @ 24 kHz). Tool calls arrive as {type:'avatar_event'}
// and drive the VRM exactly like the text path. Falls back to the
// existing Azure pipeline when live isn't available server-side.
//
// Public API:
//   const gv = new LiveVoice({ appBase, getQuery });
//   gv.onAvatarEvent = (event) => {...};
//   gv.onTranscript  = (role, text, final) => {...};
//   gv.onSpeaking    = (bool) => {...};   // drives face/visemes hint
//   gv.onState       = (state, info) => {...}; // 'connecting'|'live'|'ended'|'error'
//   await gv.start();   // asks for mic, opens socket
//   gv.stop();

const IN_RATE = 16000;
const OUT_RATE = 24000;

export class LiveVoice {
  constructor(opts = {}) {
    this.appBase = opts.appBase || '';
    this.getQuery = opts.getQuery || (() => ({}));
    this.onAvatarEvent = null;
    this.onTranscript = null;
    this.onSpeaking = null;
    this.onState = null;

    this.ws = null;
    this.active = false;
    // v198: mic can be disabled (guided-session NARRATION mode) so the coach
    // is OUTPUT-ONLY — she voices the scripted session lines without opening
    // the mic (no permission prompt) or reacting to the backing music.
    this._micEnabled = (opts.mic !== false);
    // v201: barge-in RMS gate. During a guided lesson the backing music leaks
    // into the mic, so we raise this so only a real voice (louder) interrupts
    // the coach; free-talk keeps the sensitive default.
    this._bargeGate = 0.08;
    // v198: lines to speak that arrived before the socket was ready.
    this._sayQueue = [];
    this._micCtx = null;
    this._micStream = null;
    this._micNode = null;
    this._micSource = null;
    this._playCtx = null;
    this._playGain = null;      // v155: master gain so we can duck the coach
    this._playHead = 0;
    this._sources = new Set();
    this._speaking = false;
    this._speakTimer = null;
    // v139: echo-guard state (mute mic while the coach speaks).
    this._micMuted = false;
    this._muteMicUntil = 0;
    // v155: voice- duck state. While the coach is speaking AND the user
    // starts talking over her, we lower HER playback volume so the user
    // can hear themselves / be heard, instead of both voices fighting at
    // full volume ("it gets inaudible").
    this._ducked = false;
    this._duckUntil = 0;
    this._duckTimer = null;
    // v89: optional live camera -> live vision.
    this._camVideo = null;
    this._camTimer = null;
    this._camCanvas = null;
    this.frameMs = opts.frameMs || 1200;   // ~0.8 fps, plenty for coaching
  }

  // Attach (or detach) the camera <video> whose frames we stream to the
  // coach so she can see the dancer. Pass null to stop sending frames.
  setCameraSource(videoEl) {
    this._camVideo = videoEl || null;
    if (this._camVideo && this.active && this.ws && this.ws.readyState === 1) {
      this._startCameraFrames();
    } else if (!this._camVideo) {
      this._stopCameraFrames();
    }
  }

  _startCameraFrames() {
    if (this._camTimer) return;
    if (!this._camCanvas) this._camCanvas = document.createElement('canvas');
    this._camTimer = setInterval(() => {
      try {
        const vid = this._camVideo;
        if (!vid || !this.active || !this.ws || this.ws.readyState !== 1) return;
        const w = vid.videoWidth, h = vid.videoHeight;
        if (!w || !h) return;
        // Downscale to ~320px wide to keep frames tiny.
        const sc = Math.min(1, 320 / w);
        const cw = Math.max(1, Math.round(w * sc));
        const ch = Math.max(1, Math.round(h * sc));
        const cv = this._camCanvas;
        cv.width = cw; cv.height = ch;
        const ctx = cv.getContext('2d');
        ctx.drawImage(vid, 0, 0, cw, ch);
        const url = cv.toDataURL('image/jpeg', 0.5);
        const b64 = url.slice(url.indexOf(',') + 1);
        if (b64) this.ws.send(JSON.stringify({ type: 'image', mime: 'image/jpeg', data: b64 }));
      } catch (e) { /* skip this frame */ }
    }, this.frameMs);
  }

  _stopCameraFrames() {
    if (this._camTimer) { clearInterval(this._camTimer); this._camTimer = null; }
  }

  _setState(s, info) { try { this.onState && this.onState(s, info); } catch (e) {} }
  _setSpeaking(on) {
    if (this._speaking === on) return;
    this._speaking = on;
    // v139: ECHO GUARD. Desktop Chrome cancels the coach's playback from the
    // mic, but the Android WebView's AEC does NOT cancel audio played via the
    // AudioContext — so the mic hears the coach and Gemini treats it as the
    // user talking (she interrupts herself + answers herself). While she is
    // speaking we STOP sending mic frames, and keep muting for a short
    // hangover after she finishes to swallow the room echo tail.
    if (on) {
      this._micMuted = true;
    } else {
      this._muteMicUntil = Date.now() + 350;  // hangover ms
    }
    try { this.onSpeaking && this.onSpeaking(on); } catch (e) {}
  }

  async start(opts = {}) {
    if (this.active) return;
    this.active = true;
    // v198: NARRATION mode (mic:false) opens the socket only — the coach
    // speaks relayed session lines but never listens. Full-duplex free-talk
    // (default) also opens the mic.
    if (opts.mic === false) this._micEnabled = false;
    this._setState('connecting');
    // v221 LATENCY FIX: open the socket IMMEDIATELY so the server↔Gemini
    // handshake runs CONCURRENTLY with the (often multi-second) mic-permission
    // prompt, instead of serially after it. The mic only feeds audio once
    // it's open; the socket doesn't need it to establish 'ready'. This shaves
    // the whole mic-grant time off perceived connect latency.
    this._openSocket();
    if (this._micEnabled) {
      try {
        await this._openMic();
      } catch (e) {
        this.active = false;
        try { this.ws && this.ws.close(); } catch (_e) {}
        this.ws = null;
        this._setState('error', { message: 'mic: ' + (e && e.message || e) });
        throw e;
      }
    }
  }

  // v198: enable/disable the mic on a LIVE session (used to stop listening
  // during a guided lesson so the coach only narrates, then resume after).
  // Only gates an already-open mic; it does not lazily open one.
  setMicEnabled(on) { this._micEnabled = (on !== false); }

  // v201: LESSON BARGE-IN. Lazily OPEN the mic on an already-live session so
  // the user can talk to interrupt mid-lesson, even though the lesson started
  // output-only (no prompt). Safe to call repeatedly. Returns a promise that
  // resolves true if the mic is live, false if permission was denied.
  async enableMic() {
    this._micEnabled = true;
    if (this._micNode || this._micStream) return true;   // already open
    if (!this.active) return false;
    try { await this._openMic(); return true; }
    catch (e) { this._micEnabled = false; return false; }
  }

  // v201: stop listening but keep the session (coach keeps narrating).
  disableMic() {
    this._micEnabled = false;
    this._teardownMic();
  }

  // v201: raise/lower the barge-in loudness gate (0..1). Higher = only a
  // clearly-louder voice interrupts (used while lesson music is playing).
  setBargeGate(v) {
    const n = Number(v);
    if (isFinite(n) && n > 0 && n < 1) this._bargeGate = n;
  }

  // v198: ask the live coach to SPEAK a scripted line in the SAME single
  // voice (used to narrate guided sessions). Queues until the socket is
  // ready. The server wraps it so Gemini voices it without calling tools.
  speakText(text) {
    const line = (text == null ? '' : String(text)).trim();
    if (!line) return false;
    if (this.ws && this.ws.readyState === 1) {
      try { this.ws.send(JSON.stringify({ type: 'say', text: line })); return true; }
      catch (e) { return false; }
    }
    this._sayQueue.push(line);
    return true;
  }

  stop() {
    this.active = false;
    this._setSpeaking(false);
    this._stopCameraFrames();
    try { this.ws && this.ws.close(); } catch (e) {}
    this.ws = null;
    this._teardownMic();
    this._teardownPlayback();
    this._setState('ended');
  }

  // ── mic capture ───────────────────────────────────────────────────
  async _openMic() {
    this._micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    // Request a 16 kHz context so the browser resamples the mic for us.
    // If the engine ignores the hint we down-sample manually below.
    this._micCtx = new Ctx({ sampleRate: IN_RATE });
    if (this._micCtx.state === 'suspended') { try { await this._micCtx.resume(); } catch (e) {} }
    this._micSource = this._micCtx.createMediaStreamSource(this._micStream);
    const node = this._micCtx.createScriptProcessor(4096, 1, 1);
    this._micNode = node;
    const ctxRate = this._micCtx.sampleRate;
    node.onaudioprocess = (ev) => {
      if (!this.active || !this.ws || this.ws.readyState !== 1) return;
      // v198: mic gated off (narration-only mode) — don't forward audio.
      if (this._micEnabled === false) return;
      const input = ev.inputBuffer.getChannelData(0);
      // v141: echo guard that STILL allows barge-in. While the coach is
      // speaking (or the short hangover after), we only forward mic audio
      // that's LOUD enough to be the user talking over her. Her own voice
      // leaking into the mic (already attenuated by AEC + speaker distance,
      // and absent entirely on headphones) stays below the gate, so she
      // never hears herself — but if the user really talks, it passes and
      // she can be interrupted. When she's silent, everything passes.
      if (this._speaking || (this._muteMicUntil && Date.now() < this._muteMicUntil)) {
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        const rms = Math.sqrt(sum / input.length);
        // v155: DUCK-ON-BARGE-IN. If the user is clearly talking while the
        // coach speaks, drop her volume so the two voices don't clash and
        // the user stays intelligible. Uses a slightly lower threshold than
        // the echo gate so ducking kicks in a touch before full barge-in.
        if (this._speaking && rms >= 0.05) this._duckPlayback();
        if (rms < this._bargeGate) return;   // below barge-in threshold => echo/music
      }
      const pcm = (ctxRate === IN_RATE)
        ? _floatToPCM16(input)
        : _floatToPCM16(_downsample(input, ctxRate, IN_RATE));
      try { this.ws.send(pcm); } catch (e) {}
    };
    this._micSource.connect(node);
    // ScriptProcessor only fires when connected to a destination.
    node.connect(this._micCtx.destination);
  }

  _teardownMic() {
    try { this._micNode && (this._micNode.onaudioprocess = null); } catch (e) {}
    try { this._micNode && this._micNode.disconnect(); } catch (e) {}
    try { this._micSource && this._micSource.disconnect(); } catch (e) {}
    try { this._micStream && this._micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { this._micCtx && this._micCtx.close(); } catch (e) {}
    this._micNode = this._micSource = this._micStream = this._micCtx = null;
  }

  // ── playback ──────────────────────────────────────────────────────
  _ensurePlayback() {
    if (this._playCtx) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this._playCtx = new Ctx({ sampleRate: OUT_RATE });
    // v155: all coach audio flows through a master gain so we can smoothly
    // duck her when the user talks over her (see _duckPlayback).
    this._playGain = this._playCtx.createGain();
    this._playGain.gain.value = 1.0;
    this._playGain.connect(this._playCtx.destination);
    this._playHead = 0;
  }

  // v155: lower the coach's playback volume for a short window because the
  // user is speaking over her. Ramps down fast, then auto-restores after a
  // hangover once the user stops. Cheap to call every mic frame.
  _duckPlayback() {
    this._duckUntil = Date.now() + 600;   // keep ducked while user keeps talking
    if (!this._playCtx || !this._playGain) return;
    if (!this._ducked) {
      this._ducked = true;
      try {
        const now = this._playCtx.currentTime;
        this._playGain.gain.cancelScheduledValues(now);
        this._playGain.gain.setTargetAtTime(0.28, now, 0.06);
      } catch (e) {}
    }
    if (this._duckTimer) return;
    const tick = () => {
      if (Date.now() >= this._duckUntil) {
        this._duckTimer = null;
        this._ducked = false;
        if (this._playCtx && this._playGain) {
          try {
            const now = this._playCtx.currentTime;
            this._playGain.gain.cancelScheduledValues(now);
            this._playGain.gain.setTargetAtTime(1.0, now, 0.12);
          } catch (e) {}
        }
      } else {
        this._duckTimer = setTimeout(tick, 120);
      }
    };
    this._duckTimer = setTimeout(tick, 120);
  }

  _playChunk(arrayBuf) {
    this._ensurePlayback();
    const ctx = this._playCtx;
    if (ctx.state === 'suspended') { try { ctx.resume(); } catch (e) {} }
    const i16 = new Int16Array(arrayBuf);
    if (!i16.length) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = ctx.createBuffer(1, f32.length, OUT_RATE);
    buf.getChannelData(0).set(f32);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    // v155: route through the master gain (created in _ensurePlayback) so
    // the coach can be ducked when the user talks over her.
    src.connect(this._playGain || ctx.destination);
    const now = ctx.currentTime;
    const startAt = Math.max(now, this._playHead);
    src.start(startAt);
    this._playHead = startAt + buf.duration;
    this._sources.add(src);
    src.onended = () => {
      this._sources.delete(src);
      if (this._sources.size === 0) this._setSpeaking(false);
    };
    this._setSpeaking(true);
    // v141: DO NOT rely on src.onended alone to clear the speaking flag.
    // On mobile WebViews a briefly-suspended AudioContext can drop the
    // 'ended' events entirely, leaving _speaking stuck TRUE forever — which
    // (with the echo guard) permanently mutes the mic so the coach can
    // never hear the user again ("she can't hear my voice"). Belt-and-
    // suspenders: also clear speaking on a timer sized to the queued audio.
    if (this._speakTimer) { clearTimeout(this._speakTimer); this._speakTimer = null; }
    const remainMs = Math.max(0, (this._playHead - ctx.currentTime) * 1000) + 150;
    this._speakTimer = setTimeout(() => {
      this._speakTimer = null;
      // Only clear if playback has actually drained (no fresh chunk queued
      // past 'now'); otherwise a later chunk will reschedule this timer.
      if (!this._playCtx || this._playHead <= this._playCtx.currentTime + 0.05) {
        this._sources.clear();
        this._setSpeaking(false);
      }
    }, remainMs);
  }

  _flushPlayback() {
    // Barge-in: stop everything queued so the coach goes silent NOW.
    if (this._speakTimer) { clearTimeout(this._speakTimer); this._speakTimer = null; }
    for (const s of this._sources) { try { s.stop(); } catch (e) {} }
    this._sources.clear();
    this._playHead = this._playCtx ? this._playCtx.currentTime : 0;
    // v155: reset any active duck so the NEXT coach turn starts full volume.
    if (this._duckTimer) { clearTimeout(this._duckTimer); this._duckTimer = null; }
    this._ducked = false;
    this._duckUntil = 0;
    if (this._playGain && this._playCtx) {
      try { this._playGain.gain.cancelScheduledValues(this._playCtx.currentTime);
            this._playGain.gain.value = 1.0; } catch (e) {}
    }
    this._setSpeaking(false);
  }

  _teardownPlayback() {
    this._flushPlayback();
    try { this._playGain && this._playGain.disconnect(); } catch (e) {}
    this._playGain = null;
    try { this._playCtx && this._playCtx.close(); } catch (e) {}
    this._playCtx = null;
  }

  // ── socket ────────────────────────────────────────────────────────
  _openSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const q = this.getQuery() || {};
    const params = new URLSearchParams();
    for (const k of Object.keys(q)) { if (q[k] != null && q[k] !== '') params.set(k, q[k]); }
    const qs = params.toString();
    const url = proto + '//' + location.host + this.appBase + '/ws/voice' + (qs ? ('?' + qs) : '');
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') { this._playChunk(ev.data); return; }
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.type === 'ready') {
        this._setState('live', m);
        if (this._camVideo) this._startCameraFrames();
        // v198: flush any narration lines queued before the socket opened.
        if (this._sayQueue && this._sayQueue.length) {
          const q = this._sayQueue.splice(0);
          for (const line of q) {
            try { this.ws.send(JSON.stringify({ type: 'say', text: line })); } catch (e) {}
          }
        }
      }
      else if (m.type === 'avatar_event') { try { this.onAvatarEvent && this.onAvatarEvent(m.event); } catch (e) {} }
      else if (m.type === 'transcript') { try { this.onTranscript && this.onTranscript(m.role, m.text, m.final); } catch (e) {} }
      else if (m.type === 'interrupted') { this._flushPlayback(); }
      else if (m.type === 'error') { this._setState('error', m); }
    };
    ws.onclose = () => { if (this.active) { this.active = false; this._setState('ended'); } };
    ws.onerror = () => { this._setState('error', { message: 'ws error' }); };
  }
}

// Float32 [-1,1] → little-endian PCM16 ArrayBuffer.
function _floatToPCM16(f32) {
  const out = new DataView(new ArrayBuffer(f32.length * 2));
  for (let i = 0; i < f32.length; i++) {
    let s = Math.max(-1, Math.min(1, f32[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out.buffer;
}

// Linear-interp down-sample (fallback when the AudioContext ignores
// the 16 kHz hint and hands us 44.1/48 kHz frames).
function _downsample(f32, fromRate, toRate) {
  if (toRate >= fromRate) return f32;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(f32.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, f32.length - 1);
    const frac = idx - i0;
    out[i] = f32[i0] * (1 - frac) + f32[i1] * frac;
  }
  return out;
}
