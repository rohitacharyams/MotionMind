// gemini_voice.js — Gemini Live (speech-to-speech) browser client.
//
// Captures the mic as PCM16 @ 16 kHz, streams it over /ws/voice to the
// server-side Gemini Live relay, and plays back the coach's spoken
// audio (PCM16 @ 24 kHz). Tool calls arrive as {type:'avatar_event'}
// and drive the VRM exactly like the text path. Falls back to the
// existing Azure pipeline when Gemini isn't available server-side.
//
// Public API:
//   const gv = new GeminiVoice({ appBase, getQuery });
//   gv.onAvatarEvent = (event) => {...};
//   gv.onTranscript  = (role, text, final) => {...};
//   gv.onSpeaking    = (bool) => {...};   // drives face/visemes hint
//   gv.onState       = (state, info) => {...}; // 'connecting'|'live'|'ended'|'error'
//   await gv.start();   // asks for mic, opens socket
//   gv.stop();

const IN_RATE = 16000;
const OUT_RATE = 24000;

export class GeminiVoice {
  constructor(opts = {}) {
    this.appBase = opts.appBase || '';
    this.getQuery = opts.getQuery || (() => ({}));
    this.onAvatarEvent = null;
    this.onTranscript = null;
    this.onSpeaking = null;
    this.onState = null;

    this.ws = null;
    this.active = false;
    this._micCtx = null;
    this._micStream = null;
    this._micNode = null;
    this._micSource = null;
    this._playCtx = null;
    this._playHead = 0;
    this._sources = new Set();
    this._speaking = false;
    this._speakTimer = null;
    // v89: optional live camera -> Gemini vision.
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
    try { this.onSpeaking && this.onSpeaking(on); } catch (e) {}
  }

  async start() {
    if (this.active) return;
    this.active = true;
    this._setState('connecting');
    try {
      await this._openMic();
    } catch (e) {
      this.active = false;
      this._setState('error', { message: 'mic: ' + (e && e.message || e) });
      throw e;
    }
    this._openSocket();
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
      const input = ev.inputBuffer.getChannelData(0);
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
    this._playHead = 0;
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
    src.connect(ctx.destination);
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
  }

  _flushPlayback() {
    // Barge-in: stop everything queued so the coach goes silent NOW.
    for (const s of this._sources) { try { s.stop(); } catch (e) {} }
    this._sources.clear();
    this._playHead = this._playCtx ? this._playCtx.currentTime : 0;
    this._setSpeaking(false);
  }

  _teardownPlayback() {
    this._flushPlayback();
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
      if (m.type === 'ready') { this._setState('live', m); if (this._camVideo) this._startCameraFrames(); }
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
