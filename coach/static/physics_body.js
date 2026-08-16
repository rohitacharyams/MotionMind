// physics_body.js — REAL-PHYSICS WEIGHT LAYER (in-browser, no GPU, no bake)
// ===========================================================================
// The motion player replays recorded joint angles. That is kinematically
// faithful but WEIGHTLESS: a recorded hip curve rides "on rails" with no
// mass, no gravity, no ground reaction — so the avatar reads as a puppet.
//
// This module adds the missing DYNAMICS. It runs a genuine physics
// simulation (Rapier, the same WASM engine games use) of the one thing that
// sells weight: the body's centre of mass falling under gravity and being
// caught by the legs.
//
//   • The PELVIS is a real rigid body with mass, under real gravity.
//   • Each LEG is a spring-damper strut from the pelvis to the *danced*
//     foot position. The struts hold the body up against gravity.
//   • Because it is a mass on springs, it LAGS into fast moves, OVERSHOOTS
//     coming out, SAGS slightly under load and ABSORBS on foot impact —
//     exactly the cues the eye reads as "weight".
//
// It is choreography-preserving: the leg rest-length each frame is the
// CURRENT danced pelvis→foot distance, so intentional squats/level-changes
// are kept; physics only adds the settle/overshoot on top.
//
// Output is the VERTICAL deviation from the kinematic hip (world Y), which
// the player adds to the hip. Vertical only on purpose — world-Y maps
// cleanly to the normalized-rig hip local Y (same axis the existing
// foot-lock already uses), so there is NO coordinate-convention round-trip
// (the bug that sank every offline bake). XZ stays with the existing system.

let _RAPIER = null;
let _rapierPromise = null;

async function loadRapier() {
  if (_RAPIER) return _RAPIER;
  if (!_rapierPromise) {
    _rapierPromise = (async () => {
      const R = await import('https://esm.sh/@dimforge/rapier3d-compat@0.14.0');
      await R.init();
      _RAPIER = R;
      return R;
    })();
  }
  return _rapierPromise;
}

export class WeightSim {
  constructor() {
    this.ready = false;
    this.failed = false;
    this.world = null;
    this.pelvis = null;
    this.R = null;
    this._initStarted = false;
    this._lastY = null;
  }

  /** Lazily boot Rapier + build the world. Safe to call every frame;
   *  it only does work once. Never throws into the hot path. */
  ensureInit(hipY = 0.9) {
    if (this.ready || this.failed || this._initStarted) return;
    this._initStarted = true;
    loadRapier().then((R) => {
      try {
        this.R = R;
        this.world = new R.World({ x: 0, y: -9.81, z: 0 });
        // Floor at y = 0 (matches the player's foot-lock floor plane).
        this.world.createCollider(
          R.ColliderDesc.cuboid(20, 0.1, 20).setTranslation(0, -0.1, 0));
        // Pelvis = a point mass. Rotation is owned by the kinematic pose,
        // so we lock the body's rotation and only simulate translation.
        const bd = R.RigidBodyDesc.dynamic()
          .setTranslation(0, hipY, 0)
          .lockRotations()
          .setLinearDamping(0.6);
        this.pelvis = this.world.createRigidBody(bd);
        this.world.createCollider(
          R.ColliderDesc.ball(0.06).setMass(this._mass()), this.pelvis);
        this._lastY = hipY;
        this.ready = true;
      } catch (e) {
        this.failed = true;
        try { console.warn('[WeightSim] build failed', e); } catch (_) {}
      }
    }).catch((e) => {
      this.failed = true;
      try { console.warn('[WeightSim] Rapier load failed', e); } catch (_) {}
    });
  }

  _mass()   { return (globalThis.__wMass   ?? 45);  }   // kg — upper body + thighs lumped
  _kpLeg()  { return (globalThis.__wKpLeg  ?? 1) * 16000; }  // N/m leg-strut stiffness
  _kdLeg()  { return (globalThis.__wKdLeg  ?? 1) * 220;   }  // N·s/m strut damping
  _kpPose() { return (globalThis.__wKpPose ?? 1) * 1400;  }  // N/m muscle pull to danced level

  /** New clip / teleport: snap the mass to the current hip so it does
   *  not have to fall in from a stale position. */
  reset(hipY) {
    if (!this.ready || !this.pelvis) return;
    this.pelvis.setTranslation({ x: 0, y: hipY, z: 0 }, true);
    this.pelvis.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this._lastY = hipY;
  }

  /** Step one frame and return the physics pelvis world-Y.
   *  @param dt      seconds (clamped internally for stability)
   *  @param kinHipY danced hip world-Y this frame (the target level)
   *  @param legs    [{ footY, L0, planted }, ...] world-Y of each foot,
   *                 danced pelvis→foot length L0, and contact flag.
   *  @returns physics pelvis world-Y (== kinHipY until Rapier finishes booting) */
  step(dt, kinHipY, legs) {
    if (!this.ready) { this.ensureInit(kinHipY); return kinHipY; }
    const p = this.pelvis;
    const pos = p.translation();
    const vel = p.linvel();
    const KP_LEG = this._kpLeg(), KD_LEG = this._kdLeg(), KP_POSE = this._kpPose();

    // Sum vertical forces. Rapier already applies gravity (m·g down).
    let Fy = 0;
    let anyPlanted = false;
    for (const leg of legs) {
      if (!leg.planted) continue;
      anyPlanted = true;
      // Pelvis "should" sit L0 above this planted foot. If it has sunk
      // below that (compressed strut) push up hard; if above, pull down
      // mildly (muscle keeps the foot from floating off the ground).
      const targetY = leg.footY + leg.L0;
      const e = targetY - pos.y;            // >0 -> below target -> push up
      const k = (e > 0) ? KP_LEG : KP_LEG * 0.35;
      Fy += k * e - KD_LEG * vel.y;
    }
    // Soft muscle spring toward the danced hip level so choreographed
    // level-changes (squats, rises) are preserved; physics adds the
    // settle/overshoot around them.
    Fy += KP_POSE * (kinHipY - pos.y);
    // If both feet are airborne (a jump) let gravity own it — only the
    // pose spring + gravity act, so the body actually leaves the floor.

    p.resetForces(true);
    p.addForce({ x: 0, y: Fy, z: 0 }, true);

    // Fixed-step the world. Clamp dt so a long RAF gap can't explode it.
    const h = Math.min(0.05, Math.max(1 / 240, dt));
    this.world.timestep = h;
    this.world.step();

    let y = this.pelvis.translation().y;
    if (!isFinite(y)) { this.reset(kinHipY); y = kinHipY; }
    this._lastY = y;
    return y;
  }
}
