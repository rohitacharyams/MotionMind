// avatar_life.js — procedural "aliveness" on top of a VRM avatar.
//
// Layers (all additive, all run every frame, all CPU-cheap):
//   1. Breathing       — sine on chest scale + tiny spine pitch
//   2. Idle sway       — weight shift + hip micro-bob when NOT dancing
//   3. Eye gaze        — VRM lookAt tracks the camera with random saccades
//   4. Blink           — VRM 'blink' expression, jittered every 3-6 s
//   5. Visemes         — drive vrm.expressionManager from Azure viseme IDs
//   6. Mood            — pick a VRM emotion preset ('happy', 'relaxed', ...)
//   7. Music head bob  — gentle head pitch on detected beat phase
//   8. Smooth start    — ease bone rotations from rest to first-frame
//                        of a clip over ~250 ms so the avatar doesn't snap.
//
// Usage:
//   const life = new AvatarLife(vrm, camera);
//   life.attach(player);                    // share state with MotionPlayer
//   // in render loop:
//   life.update(dt, audioTimeSec?);
//
// While the MotionPlayer is writing bone quaternions, AvatarLife only
// touches: breath additive offsets on chest/spine, head bob, eyes,
// blink, viseme expressions. It NEVER fights the dance — it composes.

import * as THREE from 'three';

// Azure viseme ID (0-21) → VRM mouth blendshape weight map.
// Keys: aa, ih, ou, ee, oh — VRM standard mouth presets.
const VISEME_MAP = {
  0:  { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0    },  // silence
  1:  { aa: 0.7,  ih: 0,    ou: 0,    ee: 0,    oh: 0    },  // æ
  2:  { aa: 1.0,  ih: 0,    ou: 0,    ee: 0,    oh: 0    },  // ɑ
  3:  { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0.9  },  // ɔ
  4:  { aa: 0,    ih: 0,    ou: 0.4,  ee: 0.5,  oh: 0    },  // ɛ ʊ
  5:  { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0.6  },  // ɝ
  6:  { aa: 0,    ih: 0.9,  ou: 0,    ee: 0.3,  oh: 0    },  // j i ɪ
  7:  { aa: 0,    ih: 0,    ou: 1.0,  ee: 0,    oh: 0    },  // w u
  8:  { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 1.0  },  // o
  9:  { aa: 0.6,  ih: 0,    ou: 0.4,  ee: 0,    oh: 0    },  // aʊ
  10: { aa: 0,    ih: 0.4,  ou: 0,    ee: 0,    oh: 0.6  },  // ɔɪ
  11: { aa: 0.5,  ih: 0.5,  ou: 0,    ee: 0,    oh: 0    },  // aɪ
  12: { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0    },  // h
  13: { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0.4  },  // ɹ
  14: { aa: 0,    ih: 0.6,  ou: 0,    ee: 0,    oh: 0    },  // l
  15: { aa: 0,    ih: 0.3,  ou: 0,    ee: 0.2,  oh: 0    },  // s z
  16: { aa: 0,    ih: 0.4,  ou: 0,    ee: 0.1,  oh: 0    },  // ʃ tʃ
  17: { aa: 0,    ih: 0.4,  ou: 0,    ee: 0,    oh: 0    },  // ð
  18: { aa: 0,    ih: 0.3,  ou: 0,    ee: 0.2,  oh: 0    },  // f v
  19: { aa: 0,    ih: 0.4,  ou: 0,    ee: 0,    oh: 0    },  // d t n θ
  20: { aa: 0,    ih: 0.3,  ou: 0,    ee: 0,    oh: 0    },  // k g ŋ
  21: { aa: 0,    ih: 0,    ou: 0,    ee: 0,    oh: 0    },  // p b m (closed)
};

const MOOD_TO_VRM = {
  neutral:   'neutral',
  happy:     'happy',
  relaxed:   'relaxed',
  excited:   'happy',
  focused:   'neutral',
  surprised: 'surprised',
  sad:       'sad',
};
const ALL_MOODS = ['happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral'];

export class AvatarLife {
  constructor(vrm, camera) {
    this.vrm = vrm;
    this.camera = camera;
    this.player = null;

    // Time accumulators
    this.t = 0;
    this.nextBlinkAt = 1.5 + Math.random() * 3;
    this.blinkFor = 0;
    this.nextSaccadeAt = 1.0 + Math.random() * 2;
    this.saccadeOff = new THREE.Vector3();
    this.saccadeTarget = new THREE.Vector3();

    // Breathing config (~16 breaths/min by default)
    this.breathHz = 16 / 60;
    this.breathAmp = 0.045;          // chest scale delta (visible)
    this.breathPitch = 0.045;        // spine pitch in radians (~2.6°)

    // Idle sway config — v108b: the lateral weight-shift + pelvic roll
    // read as an unnatural "rocking left-to-right". User wants it calm:
    // basically just breathing. Keep a whisper of weight shift so she
    // isn't a frozen statue, but no visible body rock.
    this.swayHz = 0.16;              // weight shift period (~6 s, slower)
    this.swayAmp = 0.012;            // metres lateral hip offset (was 0.06)

    // Slow continuous head idle ("looking around with curiosity")
    this.headIdleHz = 0.16;
    this.headIdleYaw = 0.13;         // ~7.4°
    this.headIdlePitch = 0.05;       // ~2.9°

    // Visemes
    this.visemeQueue = [];           // [{id, atMs}], scheduled
    this.visemeStartedAt = 0;        // wall-clock when speak() began
    this.activeViseme = 0;
    this.visemeDecay = 0.85;         // per-frame easing toward target
    this.visemeWeights = { aa:0, ih:0, ou:0, ee:0, oh:0 };

    // Mood
    this.mood = 'relaxed';
    this.moodWeight = 0.4;

    // Ease-in on clip start
    this.easeInDuration = 0.25;      // s
    this.easeInLeft = 0;
    this.easeInRestQuats = null;     // Map<vrmName, Quaternion>

    // Procedural greeting wave — when triggered, the right arm goes
    // up next to the head and the lower arm waves side-to-side for
    // a few seconds. Composed on top of the idle pose.
    this.waveLeft = 0;               // seconds remaining
    this.waveDuration = 2.6;
    this.waveSide = 'right';

    // v33d item 5 — beat-bounce idle. Tiny head/spine vertical pulse
    // synced to BPM when standing around (music plays between
    // clips). Coach.js sets _idleBpm from the active clip's metadata.
    this._idleBpm = 100;
    // v33d item 6 — eyebrow stress on TTS word boundaries. Bumped
    // by pushViseme(); decays exponentially. Drives a small
    // 'surprised' weight so brows lift on stressed syllables.
    this._eyebrowStressT = 0;
    // v33d item 9 — between-clip flourish. Brief shoulder-roll
    // gesture played after a clip ends, before snapping to rest.
    this._flourishLeft = 0;
    this._flourishDuration = 1.4;

    // Speaking mode — when the coach is talking we crank the idle
    // amplitudes (sway, breath, head bob) so she looks like she's
    // actually addressing the user instead of standing frozen. The
    // multiplier blends in/out over speakingBlendT seconds so
    // toggling doesn't pop.
    this.speaking = false;
    this.speakingT = 0;              // 0..1, blended speaking weight
    this.speakingBlendHz = 4;        // ~0.25s in/out
    this.talkPulseT = 0;             // hand/shoulder talk-gesture phase
    // Cache bone refs
    const h = vrm?.humanoid;
    this.bones = {
      hips:    h?.getNormalizedBoneNode('hips')    || null,
      spine:   h?.getNormalizedBoneNode('spine')   || null,
      chest:   h?.getNormalizedBoneNode('chest')   || null,
      neck:    h?.getNormalizedBoneNode('neck')    || null,
      head:    h?.getNormalizedBoneNode('head')    || null,
      leftShoulder:  h?.getNormalizedBoneNode('leftShoulder')  || null,
      rightShoulder: h?.getNormalizedBoneNode('rightShoulder') || null,
      leftUpperArm:  h?.getNormalizedBoneNode('leftUpperArm')  || null,
      rightUpperArm: h?.getNormalizedBoneNode('rightUpperArm') || null,
      leftLowerArm:  h?.getNormalizedBoneNode('leftLowerArm')  || null,
      rightLowerArm: h?.getNormalizedBoneNode('rightLowerArm') || null,
      leftUpperLeg:  h?.getNormalizedBoneNode('leftUpperLeg')  || null,
      rightUpperLeg: h?.getNormalizedBoneNode('rightUpperLeg') || null,
    };
    // Snapshot rest local transforms so additive layers compose cleanly
    this.rest = {};
    for (const [name, b] of Object.entries(this.bones)) {
      if (!b) continue;
      this.rest[name] = {
        pos:   b.position.clone(),
        quat:  b.quaternion.clone(),
        scale: b.scale.clone(),
      };
    }

    // Eye gaze target (a free-floating object the VRM lookAt follows)
    this.gazeTarget = new THREE.Object3D();
    this.gazeTarget.position.copy(camera.position);
    camera.parent ? camera.parent.add(this.gazeTarget)
                  : (vrm.scene.parent || vrm.scene).add(this.gazeTarget);
    if (vrm.lookAt) vrm.lookAt.target = this.gazeTarget;

    // Initial mood
    this.setMood('relaxed');
  }

  attach(player) { this.player = player; }

  // ---- public API ----------------------------------------------------

  setMood(mood) {
    const m = MOOD_TO_VRM[mood] || 'neutral';
    if (!this.vrm?.expressionManager) return;
    for (const e of ALL_MOODS) {
      try { this.vrm.expressionManager.setValue(e, 0); } catch {}
    }
    if (m !== 'neutral') {
      try { this.vrm.expressionManager.setValue(m, this.moodWeight); } catch {}
    }
    this.mood = mood;
  }

  /** Begin a visemes timeline (called when TTS speak() starts).
   *  visemes: array of { id: 0..21, offsetMs: ms-from-speech-start } */
  startVisemes(visemes) {
    this.visemeQueue = visemes.slice().sort((a,b) => a.offsetMs - b.offsetMs);
    this.visemeStartedAt = performance.now();
  }
  pushViseme(v) {
    if (this.visemeStartedAt === 0) this.visemeStartedAt = performance.now();
    this.visemeQueue.push({ id: v.id, offsetMs: v.offsetMs });
    // v33d item 6: each viseme arrival is a rough proxy for a
    // syllable boundary — give the brows a small kick so they
    // animate alongside the speech rhythm.
    this._eyebrowStressT = Math.min(1, this._eyebrowStressT + 0.45);
  }
  stopVisemes() {
    this.visemeQueue = [];
    this.activeViseme = 0;
    this.visemeStartedAt = 0;
  }

  beginEaseIn() {
    // Snapshot the bones the MotionPlayer is about to drive so we can
    // ease from the *current* pose (typically rest) to whatever the
    // first frame demands.
    this.easeInLeft = this.easeInDuration;
  }

  /** Toggle "the coach is currently speaking" body language. When on,
   *  sway / breath / head-bob amplitudes are boosted and a subtle
   *  talk-gesture rocks the shoulders. Blends in/out smoothly. */
  setSpeaking(on) { this.speaking = !!on; }

  // ---- per-frame -----------------------------------------------------

  update(dt) {
    if (!this.vrm) return;
    this.t += dt;
    // Blend speakingT toward 1 (speaking) or 0 (idle).
    const target = this.speaking ? 1 : 0;
    const k = Math.min(1, dt * this.speakingBlendHz);
    this.speakingT += (target - this.speakingT) * k;
    if (this.speaking) this.talkPulseT += dt;
    const dancing = !!(this.player && this.player.playing);

    this._updateBreath(dt, dancing);
    this._updateIdleSway(dt, dancing);
    this._updateBeatBounce(dt, dancing);
    this._updateIdleArms(dt, dancing);
    this._updateEyes(dt);
    this._updateBlink(dt);
    this._updateTalkSmile(dt);
    this._updateEyebrowStress(dt);
    this._updateFlourish(dt);
    this._updateVisemes(dt);
    // headBob is folded into breath/sway so dance keeps its own head
  }

  _updateBreath(dt, dancing) {
    // Always-on breath, smaller amplitude during dance so it doesn't
    // fight choreography. Chest scales slightly, spine pitches slightly.
    // Speaking boost: breath is faster + deeper so the chest visibly
    // moves while the coach talks.
    const speakK = 1 + this.speakingT * 1.4;
    const hz = this.breathHz * (1 + this.speakingT * 0.6);
    const phase = Math.sin(this.t * Math.PI * 2 * hz);
    const amp = (dancing ? this.breathAmp * 0.35 : this.breathAmp) * speakK;
    const pitchAmp = (dancing ? this.breathPitch * 0.25 : this.breathPitch) * speakK;
    if (this.bones.chest && this.rest.chest) {
      const s = 1 + phase * amp;
      this.bones.chest.scale.set(
        this.rest.chest.scale.x * s,
        this.rest.chest.scale.y * s,
        this.rest.chest.scale.z * s,
      );
    }
    if (!dancing && this.bones.spine && this.rest.spine) {
      // Tiny additive pitch (around X)
      const q = this.rest.spine.quat.clone();
      const add = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0), phase * pitchAmp);
      this.bones.spine.quaternion.copy(q.multiply(add));
    }
  }

  _updateIdleSway(dt, dancing) {
    if (dancing) return;            // dance owns hips/legs while playing
    // Speaking boost: amplitudes scale up so the coach visibly
    // address-shifts while she's talking instead of standing still.
    // v33d item 10: also bump the truly-idle baseline so she doesn't
    // look frozen between dances (1.0 → 1.4 idle baseline).
    // v108b: calm idle. Drop the big idle baseline (was 1.4) so the
    // body barely shifts — only breathing should really read. Speaking
    // still adds a touch of life while she talks.
    const swayK = 0.5 + this.speakingT * 0.4;
    const headK = 0.9 + this.speakingT * 0.6;
    const phase = Math.sin(this.t * Math.PI * 2 * this.swayHz) * swayK;
    const phase2 = Math.sin(this.t * Math.PI * 2 * this.swayHz * 1.7) * swayK;
    const phaseH = Math.sin(this.t * Math.PI * 2 * this.headIdleHz) * headK;
    const phaseH2 = Math.sin(this.t * Math.PI * 2 * this.headIdleHz * 1.3
                              + 0.7) * headK;

    if (this.bones.hips && this.rest.hips) {
      this.bones.hips.position.set(
        this.rest.hips.pos.x + phase * this.swayAmp,
        this.rest.hips.pos.y + Math.abs(phase) * -0.008 + phase2 * 0.004,
        this.rest.hips.pos.z + phase2 * 0.012,
      );
      // Pelvic tilt that follows the weight shift — v108b: cut the
      // Z-roll hard (was 0.07) so there's no visible left-right rock.
      const q = this.rest.hips.quat.clone();
      const add = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 0, 1), -phase * 0.012);
      const addY = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), phaseH2 * 0.05);
      this.bones.hips.quaternion.copy(q.multiply(add).multiply(addY));
    }

    if (this.bones.spine && this.rest.spine) {
      // Counter-rotate the spine slightly so the upper body floats more
      const q = this.rest.spine.quat.clone();
      const add = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), -phaseH2 * 0.04);
      this.bones.spine.quaternion.copy(q.multiply(add));
    }

    if (this.bones.neck && this.rest.neck) {
      const q = this.rest.neck.quat.clone();
      const add = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), phaseH * this.headIdleYaw * 0.5);
      this.bones.neck.quaternion.copy(q.multiply(add));
    }

    if (this.bones.head && this.rest.head) {
      const q = this.rest.head.quat.clone();
      // Slow yaw + small pitch — like she's listening / looking around
      const addY = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), phaseH * this.headIdleYaw);
      const addX = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0), phaseH2 * this.headIdlePitch);
      this.bones.head.quaternion.copy(q.multiply(addY).multiply(addX));
    }
  }

  _updateIdleArms(dt, dancing) {
    if (dancing) return;
    // Subtle arm sway — micro motion at breath frequency so arms feel
    // tied to the body, plus a slower lateral drift. Composes onto
    // whatever rest pose applyRestPose() left in place by snapshotting
    // on first use.
    if (!this._restArms) {
      this._restArms = {};
      for (const k of ['leftShoulder','rightShoulder',
                       'leftUpperArm','rightUpperArm',
                       'leftLowerArm','rightLowerArm']) {
        const b = this.bones[k];
        if (b) this._restArms[k] = b.quaternion.clone();
      }
    }
    const br = Math.sin(this.t * Math.PI * 2 * this.breathHz);
    const sl  = Math.sin(this.t * Math.PI * 2 * 0.21);
    const sl2 = Math.sin(this.t * Math.PI * 2 * 0.17 + 1.2);
    // Talking pulses — added on top of the idle terms via this.speakingT.
    const tk  = this.speakingT;
    const tp  = Math.sin(this.talkPulseT * Math.PI * 2 * 0.55);
    const tp2 = Math.sin(this.talkPulseT * Math.PI * 2 * 0.83 + 0.6);
    const tp3 = Math.sin(this.talkPulseT * Math.PI * 2 * 1.10 + 1.3);
    const apply = (name, axis, angle) => {
      const b = this.bones[name];
      const r = this._restArms?.[name];
      if (!b || !r) return;
      const add = new THREE.Quaternion().setFromAxisAngle(axis, angle);
      b.quaternion.copy(r.clone().multiply(add));
    };
    const applyAdd = (name, axis, angle) => {
      // Compose an extra rotation ONTO whatever apply() last wrote.
      const b = this.bones[name];
      if (!b) return;
      const add = new THREE.Quaternion().setFromAxisAngle(axis, angle);
      b.quaternion.multiply(add);
    };
    const Y = new THREE.Vector3(0, 1, 0);
    const Z = new THREE.Vector3(0, 0, 1);
    const X = new THREE.Vector3(1, 0, 0);
    apply('leftShoulder',  Z,  br * 0.025 + sl  * 0.012 + tp  * 0.08 * tk);
    apply('rightShoulder', Z, -br * 0.025 - sl  * 0.012 - tp  * 0.08 * tk);
    apply('leftUpperArm',  Z, -br * 0.020 + sl2 * 0.015 - tp2 * 0.10 * tk);
    apply('rightUpperArm', Z,  br * 0.020 - sl2 * 0.015 + tp2 * 0.10 * tk);
    apply('leftLowerArm',  Y,  br * 0.018 + tp3 * 0.18 * tk);
    apply('rightLowerArm', Y, -br * 0.018 - tp3 * 0.18 * tk);
    if (tk > 0.01) {
      // Forearm lift while explaining — composed onto the Y-axis swing
      // above so it doesn't overwrite it.
      applyAdd('leftLowerArm',  X, -Math.abs(tp2) * 0.22 * tk);
      applyAdd('rightLowerArm', X, -Math.abs(tp3) * 0.22 * tk);
    }

    // Greeting wave — a simple, human-scale right-hand "hi":
    // shoulder lifts slightly, elbow bends, hand waves left-right.
    // No overhead pose, no shoulder forward-tilt — that looked
    // theatrical instead of friendly.
    if (this.waveLeft > 0) {
      this.waveLeft = Math.max(0, this.waveLeft - dt);
      const elapsed = this.waveDuration - this.waveLeft;
      // Ease envelope: 0 → 1 in 0.30s, hold, 1 → 0 in last 0.30s.
      const easeIn  = Math.min(1, elapsed / 0.30);
      const easeOut = Math.min(1, this.waveLeft / 0.30);
      const env = Math.min(easeIn, easeOut);
      // ~1.9 Hz wrist-side oscillation — feels natural, not frantic.
      const wig = Math.sin(elapsed * Math.PI * 2 * 1.9);
      const isLeft = this.waveSide === 'left';
      const up  = isLeft ? 'leftUpperArm'  : 'rightUpperArm';
      const low = isLeft ? 'leftLowerArm'  : 'rightLowerArm';
      // v110b: REAL "hi 👋", tuned LIVE on the avatar (front view) so the
      // hand comes UP beside the head — not out to the side / backward.
      // VRM normalized rig: from T-pose the right arm raises overhead with
      // a POSITIVE local-Z rotation (left arm mirrors with negative Z).
      // Upper arm up-and-out (~60° above horizontal), elbow bent so the
      // forearm is vertical with the hand by the head, then the forearm
      // wiggles on Y to wave side-to-side. These are ABSOLUTE local targets
      // (the old additive apply() calls overwrote each other → arm went
      // backward). Eased rest→target by the envelope so it lifts + settles.
      const sgn = isLeft ? -1 : 1;
      const Eorder = (this.bones[up] && this.bones[up].rotation.order) || 'XYZ';
      // v112: re-tuned LIVE (front view) to a friendly, natural "hi" —
      // upper arm raised up-and-out ~46°, forearm bent so the hand sits
      // BESIDE the head (not straight overhead / out to the side), palm
      // forward, hand rocking side-to-side. Mirrors for the left side.
      const upTarget = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(0, 0, sgn * 0.8, Eorder));
      const lowTarget = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(0, sgn * wig * 0.18, sgn * 1.5, Eorder));
      const ub = this.bones[up], lb = this.bones[low];
      const ur = this._restArms ? this._restArms[up] : null;
      const lr = this._restArms ? this._restArms[low] : null;
      // Ease rest → target by the envelope (slerp), so it lifts in and
      // settles back down smoothly.
      if (ub && ur) ub.quaternion.copy(ur).slerp(upTarget, env);
      if (lb && lr) lb.quaternion.copy(lr).slerp(lowTarget, env);
    }
  }

  /** Trigger a procedural greeting wave on the chosen side. */
  playWave(opts = {}) {
    this.waveSide = opts.side === 'left' ? 'left' : 'right';
    this.waveDuration = Math.max(1.0, +opts.duration || 2.6);
    this.waveLeft = this.waveDuration;
  }

  /** v33d item 9 — between-clip flourish. Quick shoulder-roll +
   *  light arm-cross-and-release sweep. Played after a dance clip
   *  finishes so the avatar doesn't snap-cut back to rest pose.
   *  Composed additively on top of the idle pose. */
  playFlourish(opts = {}) {
    this._flourishDuration = Math.max(0.6, +opts.duration || 1.4);
    this._flourishLeft = this._flourishDuration;
  }

  // v33d item 5 — beat-bounce idle. Subtle vertical head/spine
  // pulse synced to _idleBpm. Skipped while dancing (the clip owns
  // motion). Skipped while flourishing (the flourish owns motion).
  _updateBeatBounce(dt, dancing) {
    if (dancing) return;
    if (this._flourishLeft > 0) return;
    const hz = (this._idleBpm || 100) / 60.0;
    // Half-cycle pulse so the bounce reads as a soft DOWN-and-UP
    // on each beat instead of a sine wobble.
    const phase = Math.abs(Math.sin(this.t * Math.PI * hz));
    // ~6 mm vertical push at the head. Tiny — meant to feel like
    // micro-bobbing to music, not actually dancing.
    const drop = (1 - phase) * 0.006;
    if (this.bones.head && this.rest.head) {
      this.bones.head.position.set(
        this.rest.head.pos.x,
        this.rest.head.pos.y - drop,
        this.rest.head.pos.z,
      );
    }
  }

  // v33d item 6 — eyebrow stress envelope. pushViseme() kicks
  // _eyebrowStressT; here we decay it and write the 'surprised'
  // blendshape so the brows animate in time with speech.
  _updateEyebrowStress(dt) {
    const em = this.vrm?.expressionManager;
    if (!em) return;
    // Exponential decay ~3 Hz so each kick fades over ~300ms.
    const k = Math.exp(-3.0 * dt);
    this._eyebrowStressT *= k;
    if (this._eyebrowStressT < 0.005) {
      this._eyebrowStressT = 0;
    }
    // Only when actually speaking — avoid stray brow lifts at rest.
    const w = this.speakingT * this._eyebrowStressT * 0.22;
    try { em.setValue('surprised', w); } catch {}
  }

  // v33d item 9 — flourish update (shoulder roll + soft arm wash).
  _updateFlourish(dt) {
    if (this._flourishLeft <= 0) return;
    this._flourishLeft = Math.max(0, this._flourishLeft - dt);
    const elapsed = this._flourishDuration - this._flourishLeft;
    const t = Math.min(1, elapsed / this._flourishDuration);
    // Ease envelope: 0→1 in first 25%, 1→0 in last 25%, hold middle.
    const easeIn  = Math.min(1, t * 4);
    const easeOut = Math.min(1, (1 - t) * 4);
    const env = Math.min(easeIn, easeOut);
    const apply = (name, axis, ang) => {
      const b = this.bones[name];
      const r = this.rest[name];
      if (!b || !r) return;
      const q = r.quat.clone();
      const add = new THREE.Quaternion().setFromAxisAngle(axis, ang);
      b.quaternion.copy(q.multiply(add));
    };
    const X = new THREE.Vector3(1, 0, 0);
    const Z = new THREE.Vector3(0, 0, 1);
    // Soft shoulder-roll: both shoulders lift + roll outward briefly.
    apply('leftUpperArm',  Z,  0.22 * env);
    apply('rightUpperArm', Z, -0.22 * env);
    apply('leftUpperArm',  X, -0.18 * env);
    apply('rightUpperArm', X, -0.18 * env);
    // Slight chin-up on the release — "phew, that was fun."
    apply('head', X, -0.12 * env);
  }

  _updateEyes(dt) {
    if (!this.vrm?.lookAt) return;
    // Random saccades — small offset target around the camera
    if (this.t >= this.nextSaccadeAt) {
      this.saccadeTarget.set(
        (Math.random() - 0.5) * 0.4,
        (Math.random() - 0.5) * 0.2,
        0,
      );
      this.nextSaccadeAt = this.t + 1.8 + Math.random() * 2.5;
    }
    // Ease toward target
    this.saccadeOff.lerp(this.saccadeTarget, Math.min(1, dt * 4));
    const camPos = new THREE.Vector3();
    this.camera.getWorldPosition(camPos);
    this.gazeTarget.position.set(
      camPos.x + this.saccadeOff.x,
      camPos.y + this.saccadeOff.y,
      camPos.z + this.saccadeOff.z,
    );
  }

  _updateBlink(dt) {
    const em = this.vrm?.expressionManager;
    if (!em) return;
    if (this.blinkFor > 0) {
      this.blinkFor -= dt;
      // 150 ms closed blink with quick close + quick open
      const phase = 1 - (this.blinkFor / 0.15);     // 0→1 over blink
      const w = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
      try { em.setValue('blink', Math.max(0, Math.min(1, w))); } catch {}
      if (this.blinkFor <= 0) {
        try { em.setValue('blink', 0); } catch {}
        this.nextBlinkAt = this.t + 2.8 + Math.random() * 3.2;
      }
    } else if (this.t >= this.nextBlinkAt) {
      this.blinkFor = 0.15;
    }
  }

  /** v33c (item 4): gentle smile while the coach is speaking. The
   *  expression weight follows speakingT so it eases in/out with
   *  the same blend as the talk-sway. Tiny extra pulse on every
   *  blink so the face doesn't look frozen during long sentences. */
  _updateTalkSmile(dt) {
    const em = this.vrm?.expressionManager;
    if (!em) return;
    // Mood gate: angry/sad/focused suppress the smile (don't grin
    // while supposedly serious); happy/relaxed/surprised allow it.
    const mood = this.mood || 'relaxed';
    if (mood === 'angry' || mood === 'sad' || mood === 'focused') {
      return;
    }
    const base = 0.28 * this.speakingT;
    // pulse: small extra bump while a blink is active so the eyes
    // and mouth move together (looks like a natural micro-smile).
    const pulse = (this.blinkFor > 0) ? 0.10 : 0.0;
    const w = Math.min(0.55, base + pulse);
    try { em.setValue('happy', w); } catch {}
  }

  _updateVisemes(dt) {
    const em = this.vrm?.expressionManager;
    if (!em) return;
    const ease = Math.min(1, dt * 18);   // ~55 ms time-constant
    // v109b: PROCEDURAL MOUTH for Gemini S2S. The viseme stream only
    // exists for Azure TTS (it ships phoneme timings). Gemini live
    // audio has NO viseme data, so in the default voice mode the mouth
    // stayed frozen shut — "no face expressions while she talks". When
    // we're speaking but have no viseme stream, drive a natural mouth
    // flap so she visibly talks. (Two detuned sines = non-robotic.)
    const noVisemeStream =
      this.visemeQueue.length === 0 && this.visemeStartedAt === 0;
    if (this.speaking && noVisemeStream) {
      const t = this.t;
      const env = Math.max(0, 0.45 + 0.55 * Math.sin(t * Math.PI * 2 * 2.7));
      const flap = Math.max(0, Math.sin(t * Math.PI * 2 * 6.3));
      const aa = Math.min(1, flap * env * 1.1);
      const ih = Math.min(1, Math.max(0,
        Math.sin(t * Math.PI * 2 * 7.9 + 1.1)) * 0.35);
      const tgt = { aa, ih, ou: 0, ee: 0, oh: aa * 0.25 };
      for (const k of Object.keys(this.visemeWeights)) {
        this.visemeWeights[k] += ((tgt[k] || 0) - this.visemeWeights[k]) * ease;
        try { em.setValue(k, this.visemeWeights[k]); } catch {}
      }
      return;
    }
    // Advance the active viseme based on offsetMs schedule
    if (this.visemeQueue.length && this.visemeStartedAt > 0) {
      const elapsed = performance.now() - this.visemeStartedAt;
      while (this.visemeQueue.length
             && this.visemeQueue[0].offsetMs <= elapsed) {
        this.activeViseme = this.visemeQueue.shift().id;
      }
    }
    // Decay current weights toward target viseme map
    const target = VISEME_MAP[this.activeViseme] || VISEME_MAP[0];
    // v69: MOUTH GAIN. At full-body camera distance the VRoid mouth is
    // tiny, so the default viseme weights read as "barely moving" — the
    // user couldn't tell she was talking. Push the open-mouth shapes
    // harder (clamped to 1) so the lip-sync is actually visible.
    const GAIN = 1.5;
    for (const k of Object.keys(this.visemeWeights)) {
      const tw = Math.min(1, (target[k] || 0) * GAIN);
      this.visemeWeights[k] += (tw - this.visemeWeights[k]) * ease;
      try { em.setValue(k, this.visemeWeights[k]); } catch {}
    }
    // Auto-clear when no more visemes are queued and elapsed long past
    if (this.visemeQueue.length === 0
        && this.visemeStartedAt > 0
        && performance.now() - this.visemeStartedAt > 30_000) {
      this.stopVisemes();
    }
  }
}
