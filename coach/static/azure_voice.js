// azure_voice.js — browser STT (continuous) + TTS using Azure Speech SDK.
// Tokens are minted by the backend so we never expose the key.

let SDK = null;
async function loadSDK() {
  if (SDK) return SDK;
  await new Promise((ok, fail) => {
    const s = document.createElement('script');
    s.src = 'https://aka.ms/csspeech/jsbrowserpackageraw';
    s.onload = ok; s.onerror = fail;
    document.head.appendChild(s);
  });
  SDK = window.SpeechSDK;
  return SDK;
}

async function getToken() {
  const base = (typeof window !== 'undefined' && window.APP_BASE) || '';
  const r = await fetch(base + '/api/speech/token');
  if (!r.ok) throw new Error('speech token fetch failed: ' + r.status);
  return r.json();
}

export class AzureVoice {
  constructor() {
    this.recognizer = null;
    this.synth = null;
    // Ava Multilingual = Azure's most natural conversational voice
    // (breathy, warm, light laugh, micro-pauses). Falls back to
    // EmmaMultilingual / JennyMultilingual if the user prefers.
    this.voice = 'en-US-AvaMultilingualNeural';
    this.style = 'chat';      // chat | cheerful | excited | friendly
    // v69: reply language. xml:lang + voice swap. Hinglish uses an
    // Indian-English voice (reads Latin-script Hindi words naturally);
    // Hindi uses a native hi-IN voice.
    this.lang = 'en-US';
    this._useStyle = true;    // mstts express-as only on en-US voices
    // v30: slightly faster + a touch lower pitch removes the "TTS
    // newsreader" flavor. Ava already breathes; this just keeps her
    // moving so latency feels lower too.
    this._baseRate = '+8%';   // en-US Ava rate (with the lively chat style)
    this.rate = this._baseRate;
    this.pitch = '-2%';
    this.onfinal = null;
    this.onpartial = null;
    this.listening = false;
    this._tok = null;
  }

  setVoice(name) { if (name) this.voice = name; }
  setStyle(style) { if (style) this.style = style; }

  // v69: switch reply language → voice + xml:lang.
  setLanguage(language) {
    const l = (language || 'english').toLowerCase();
    if (l === 'hindi') {
      this.voice = 'hi-IN-SwaraNeural';
      this.lang = 'hi-IN';
      this._useStyle = false;     // hi-IN voices don't take the 'chat' style
      // v73: the en-IN / hi-IN voices reject mstts:express-as 'chat',
      // so without that lively style they read at a slower, more
      // measured pace than Ava. Push the prosody rate up to keep the
      // cadence close to the brisk English voice.
      this.rate = '+16%';
    } else if (l === 'hinglish') {
      // Indian-English voice: natural reading of Latin-script Hinglish.
      this.voice = 'en-IN-NeerjaNeural';
      this.lang = 'en-IN';
      this._useStyle = false;
      this.rate = '+16%';         // v73: match Ava's pace (no chat style)
    } else {
      this.voice = 'en-US-AvaMultilingualNeural';
      this.lang = 'en-US';
      this._useStyle = true;
      this.rate = this._baseRate; // back to the lively en-US cadence
    }
  }

  async _ensureToken() {
    if (!SDK) await loadSDK();
    if (!this._tok || Date.now() > this._tok.exp) {
      const t = await getToken();
      this._tok = { token: t.token, region: t.region,
                    exp: Date.now() + 8 * 60_000 };  // azure tokens last 10 min
    }
    return this._tok;
  }

  async startListening() {
    if (this.listening) return;
    const t = await this._ensureToken();
    const cfg = SDK.SpeechConfig.fromAuthorizationToken(t.token, t.region);
    cfg.speechRecognitionLanguage = 'en-US';
    const audio = SDK.AudioConfig.fromDefaultMicrophoneInput();
    this.recognizer = new SDK.SpeechRecognizer(cfg, audio);
    this.recognizer.recognizing = (_s, e) => {
      // v77: ECHO GUARD. While the coach's own TTS is playing the mic
      // hears it and would transcribe her own voice back to herself.
      // Drop any recognition that lands inside the speaking window.
      if (this._sttSuspendedUntil && performance.now() < this._sttSuspendedUntil) return;
      if (this.onpartial) this.onpartial(e.result.text);
    };
    this.recognizer.recognized = (_s, e) => {
      if (e.result.reason !== SDK.ResultReason.RecognizedSpeech) return;
      if (this._sttSuspendedUntil && performance.now() < this._sttSuspendedUntil) return;
      const text = (e.result.text || '').trim();
      if (text && this.onfinal) this.onfinal(text);
    };
    await new Promise((ok, fail) =>
      this.recognizer.startContinuousRecognitionAsync(ok, fail));
    this.listening = true;
  }

  async stopListening() {
    if (!this.listening || !this.recognizer) return;
    await new Promise((ok) =>
      this.recognizer.stopContinuousRecognitionAsync(ok, ok));
    this.recognizer.close();
    this.recognizer = null;
    this.listening = false;
  }

  _wrapSSML(text) {
    // Escape XML-special chars
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const body = esc(text);
    // mstts:express-as makes Ava/Emma sound conversational rather than
    // newsreader. <prosody rate> gives a slightly relaxed cadence so
    // it doesn't feel rushed like the default neural voice.
    // v69: xml:lang follows the selected language; express-as only on
    // en-US voices (hi-IN / en-IN voices reject the 'chat' style).
    const lang = this.lang || 'en-US';
    const open = this._useStyle
      ? `<mstts:express-as style="${this.style}" styledegree="1.2">`
      : '';
    const close = this._useStyle ? `</mstts:express-as>` : '';
    return `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" `
      + `xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${lang}">`
      + `<voice name="${this.voice}">`
      + open
      + `<prosody rate="${this.rate}" pitch="${this.pitch}">`
      + body
      + `</prosody>${close}</voice></speak>`;
  }

  async speak(text, opts = {}) {
    if (!text) return;
    // If the user has barge-in'd recently, drop queued speech entirely
    // for a short cooldown so the coach doesn't immediately resume.
    if (this._silenceUntil && performance.now() < this._silenceUntil) {
      return;
    }
    // Strict FIFO chain — every speak() waits for the previous one to
    // finish. Without this, the agent emitting two assistant_text
    // events back-to-back would create two SpeechSynthesizers writing
    // to the default speaker in parallel → 2-3 voices overlapping.
    // The .catch keeps a rejected speak (e.g. autoplay-blocked
    // greeting before user gesture) from poisoning the chain.
    const prev = this._chain || Promise.resolve();
    const next = prev.catch(() => {}).then(() => this._speakOne(text, opts));
    this._chain = next;
    return next;
  }

  /** Hard-stop any in-flight TTS and discard the FIFO chain.
   *  Called when the user barges in (starts speaking / typing) so the
   *  coach shuts up mid-syllable. Sets a short silence cooldown so any
   *  in-flight server response that already queued speak() doesn't
   *  immediately resume the monologue. */
  cancelSpeak({ silenceMs = 1500 } = {}) {
    this._silenceUntil = performance.now() + silenceMs;
    try {
      if (this._currentSynth) {
        this._currentSynth.close();
        this._currentSynth = null;
      }
    } catch (e) { /* ignore */ }
    try {
      if (this._currentPlayer && this._currentPlayer.pause) {
        this._currentPlayer.pause();
      }
    } catch (e) { /* ignore */ }
    this._currentPlayer = null;
    // Force-resolve the in-flight promise so the chain doesn't deadlock.
    if (this._currentResolve) {
      try { this._currentResolve(null); } catch {}
      this._currentResolve = null;
    }
    // Reset the chain so subsequent speak() doesn't await a dead one.
    this._chain = Promise.resolve();
  }

  async _speakOne(text, { onviseme = null } = {}) {
    if (!text) return;
    // Respect mid-speak cancellation: if cancelSpeak set a recent
    // silenceUntil we skip this utterance entirely.
    if (this._silenceUntil && performance.now() < this._silenceUntil) {
      return;
    }
    // v77: open the echo-guard window up front (rough estimate from
    // text length) so the mic ignores the coach's voice the instant
    // playback begins; refined to the exact audio duration below.
    this._sttSuspendedUntil = performance.now() + Math.max(1500, text.length * 80);
    const t = await this._ensureToken();
    const cfg = SDK.SpeechConfig.fromAuthorizationToken(t.token, t.region);
    cfg.speechSynthesisVoiceName = this.voice;
    cfg.speechSynthesisOutputFormat =
      SDK.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3;
    // v33e: iOS Safari fix. `AudioConfig.fromDefaultSpeakerOutput()`
    // creates an internal AudioContext that the page-level unlock
    // gesture never touches — so synthesis runs but no audio is
    // audible (music keeps playing via our own <audio> element which
    // IS unlocked). `SpeakerAudioDestination` exposes the player so
    // we can call resume() on every speak after the gesture lands.
    let player = null;
    let audio;
    try {
      if (SDK.SpeakerAudioDestination) {
        player = new SDK.SpeakerAudioDestination();
        audio = SDK.AudioConfig.fromSpeakerOutput(player);
      } else {
        audio = SDK.AudioConfig.fromDefaultSpeakerOutput();
      }
    } catch (_e) {
      audio = SDK.AudioConfig.fromDefaultSpeakerOutput();
    }
    const synth = new SDK.SpeechSynthesizer(cfg, audio);
    this._currentSynth = synth;
    this._currentPlayer = player;
    if (onviseme) {
      let _visemeCount = 0;
      synth.visemeReceived = (_s, e) => {
        _visemeCount++;
        if (_visemeCount === 1) console.info('[viseme] first received id=' + e.visemeId);
        onviseme({ id: e.visemeId, offsetMs: e.audioOffset / 10_000 });
      };
      synth.synthesisStarted = (_s, _e) => {
        console.info('[voice] synthesis started');
        // Resume the internal audio element/context. iOS Safari
        // suspends it whenever focus changes; without this the
        // synthesised bytes arrive but never play.
        try { player && player.resume && player.resume(); } catch {}
      };
      synth.synthesisCompleted = (_s, _e) => {
        console.info('[voice] synthesis completed; visemes=' + _visemeCount);
        if (_visemeCount === 0) {
          console.warn('[voice] NO visemes received — lip-sync will be silent');
        }
      };
    }
    const ssml = this._wrapSSML(text);
    // CRITICAL: speakSsmlAsync resolves when SYNTHESIS finishes —
    // not when the audio actually stops playing through the speaker.
    // Without an explicit wait on r.audioDuration, two back-to-back
    // speak() calls overlap audibly even though the FIFO chain looks
    // correct. We wait for the synthesised duration (100-ns ticks)
    // plus a small tail so the next utterance doesn't step on the
    // current one's release.
    const result = await new Promise((ok, fail) =>
      synth.speakSsmlAsync(ssml,
        (r) => { try { synth.close(); } catch {} ok(r); },
        (err) => { try { synth.close(); } catch {} fail(err); }));
    // Clear the live-synth ref; cancelSpeak no longer needs to close it.
    if (this._currentSynth === synth) this._currentSynth = null;
    // If a barge-in landed during synthesis, abort the playback wait
    // immediately rather than letting the speaker finish the buffered
    // audio (which would defeat the whole point of barge-in).
    if (this._silenceUntil && performance.now() < this._silenceUntil) {
      return;
    }
    const ticks = result && result.audioDuration ? result.audioDuration : 0;
    const ms = ticks > 0 ? ticks / 10_000 : 0;
    // v77: refine the echo-guard window to the exact spoken duration
    // (+400 ms tail for the speaker release).
    this._sttSuspendedUntil = performance.now() + ms + 400;
    if (ms > 0) {
      // Wait for the audio to finish playing, but bail early if a
      // barge-in fires mid-utterance.
      const start = performance.now();
      const deadline = start + ms + 120;
      while (performance.now() < deadline) {
        if (this._silenceUntil && performance.now() < this._silenceUntil) {
          return;
        }
        await new Promise((r) => setTimeout(r, 60));
      }
    }
  }
}
