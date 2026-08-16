// motion_player.js — drives a VRM avatar from EITHER:
//   (A) pre-retargeted bone-local quaternions (format: 'vrm-quat'),
//       produced offline by scripts/export_motion_json.py. PREFERRED.
//   (B) raw SMPL 24-joint axis-angle (format: 'smpl-aa'). Fallback,
//       only used while the cache is still being built.
//
// Both code paths share:
//   • Per-bone angular-speed guard (radians/sec between consecutive
//     applied quaternions). If a bone would exceed it, the player
//     freezes on the last safe frame and fires `onflag`.
//   • Hemisphere continuity (q vs -q) — fixes 360° spin artifacts.
//   • Mirror flag.

import * as THREE from 'three';

// ---------- format A: retargeted VRM-bone-local quats ----------------
// Names from export_motion_json.py → VRM normalized humanoid names
const VRM_BONE_MAP = {
  Hips: 'hips',
  Spine: 'spine',
  Spine2: 'chest',
  Neck: 'neck',
  Head: 'head',
  LeftShoulder: 'leftShoulder',
  LeftArm: 'leftUpperArm',
  LeftForeArm: 'leftLowerArm',
  LeftHand: 'leftHand',
  LeftUpLeg: 'leftUpperLeg',
  LeftLeg: 'leftLowerLeg',
  LeftFoot: 'leftFoot',
  RightShoulder: 'rightShoulder',
  RightArm: 'rightUpperArm',
  RightForeArm: 'rightLowerArm',
  RightHand: 'rightHand',
  RightUpLeg: 'rightUpperLeg',
  RightLeg: 'rightLowerLeg',
  RightFoot: 'rightFoot',
};

// ---------- format B: SMPL axis-angle (legacy) ------------------------
export const SMPL_TO_VRM = [
  'hips', 'leftUpperLeg', 'rightUpperLeg', 'spine',
  'leftLowerLeg', 'rightLowerLeg', 'chest', 'leftFoot',
  'rightFoot', 'upperChest', 'leftToes', 'rightToes',
  'neck', 'leftShoulder', 'rightShoulder', 'head',
  'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm',
  'leftHand', 'rightHand', null, null,
];

// ---------- body-part groups -----------------------------------------
// Used by MotionPlayer.isolate() so the coach can drill arms-only,
// legs-only etc. while the rest of the body holds bind-pose.
export const BODY_PART_GROUPS = {
  arms: [
    'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
    'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
  ],
  legs: [
    'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
    'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
  ],
  torso:    ['hips', 'spine', 'chest', 'upperChest'],
  head:     ['neck', 'head'],
  hands:    ['leftHand', 'rightHand'],
  feet:     ['leftFoot', 'rightFoot'],
  left:     ['leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
             'leftUpperLeg', 'leftLowerLeg', 'leftFoot'],
  right:    ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
             'rightUpperLeg', 'rightLowerLeg', 'rightFoot'],
};

// ---------- anatomic joint-angle limits ------------------------------
// Maximum rotation (radians) AWAY FROM BIND-POSE that each bone may
// reach. The mocap source (AIST, CMU) was captured on real humans so
// most frames are well within these caps, but occasional spike frames
// — wrist hyperextension, neck snap, shoulder past 180° — produce the
// "absurd" poses the user keeps seeing. Clamping in-pipeline kills
// those without breaking the dance because the cap sits well past the
// normal range of motion for each joint. Values are deliberately
// generous (dance moves push the envelope) but finite.
const BONE_MAX_ANGLE = {
  hips:           0.55,  // ~31°  pelvis tilt
  spine:          0.70,  // ~40°
  chest:          0.55,  // ~31°
  upperChest:     0.45,  // ~25°
  neck:           0.80,  // ~46°
  head:           1.10,  // ~63°
  leftShoulder:   1.20, rightShoulder:  1.20,  // shrug/roll
  leftUpperArm:   2.80, rightUpperArm:  2.80,  // ~160° raise
  leftLowerArm:   2.60, rightLowerArm:  2.60,  // elbow flex
  leftHand:       1.30, rightHand:      1.30,  // wrist
  leftUpperLeg:   1.90, rightUpperLeg:  1.90,  // ~109° hip flex
  leftLowerLeg:   2.50, rightLowerLeg:  2.50,  // knee
  leftFoot:       0.90, rightFoot:      0.90,
  leftToes:       0.60, rightToes:      0.60,
};

// Lowpass mixing factor for the temporal smoother (1.0 = no smoothing,
// 0 = frozen). 0.65 gives roughly a 5-frame moving-average at 30 fps
// without making the motion feel dragged.
const QUAT_LOWPASS_ALPHA = 0.65;

export class MotionPlayer {
  constructor(vrm, limits) {
    // v34e: expose for live introspection from the page console.
    try { if (typeof window !== 'undefined') window.__player = this; } catch (_) {}
    this.vrm = vrm;
    this.limits = limits || {};
    this.data = null;
    this.format = null;
    this.frame = 0;
    this.speed = 1.0;
    this._mirror = false;
    this.loop = false;
    this.playing = false;
    this.lastFrameIdx = -1;
    this.onflag = null;
    this.onframe = null;
    this.flagged = false;
    this.prevBoneQuats = new Map();
    // Mirror is implemented as a root-level X scale flip on the VRM
    // scene. Doing it per-bone in quaternion space distorts the body
    // because child bones inherit parent orientation; a single root
    // scale gives a clean, correct left-right reflection.
    Object.defineProperty(this, 'mirror', {
      get: () => this._mirror,
      set: (v) => {
        const on = !!v;
        this._mirror = on;
        const root = this.vrm?.scene;
        if (root) {
          root.scale.x = on ? -Math.abs(root.scale.x || 1)
                            :  Math.abs(root.scale.x || 1);
          // Flip winding so lighting stays correct under negative scale.
          root.traverse?.((o) => {
            if (o.isMesh && o.material) {
              const mats = Array.isArray(o.material) ? o.material : [o.material];
              mats.forEach((mat) => { if (mat) mat.side = THREE.DoubleSide; });
            }
          });
        }
      },
      configurable: true,
    });
    // null → no isolation (drive every bone). When set to a Set<string>
    // of VRM bone names, only those bones get driven from the clip; the
    // rest are pinned at bind-pose so the user can see exactly the
    // body part being drilled.
    this._isolated = null;
    this._hipsRestPos = new THREE.Vector3();
    const hips = vrm?.humanoid?.getNormalizedBoneNode('hips');
    if (hips) this._hipsRestPos.copy(hips.position);
    // Snapshot the VRM's bind-pose local quats once. Per-character
    // rigs ship with NON-identity locals on shoulders/hands/legs
    // (esp. VRoid Studio models), so forcing identity in rest mode
    // flares hands outward. We use these snapshots as the "keep as
    // shipped" baseline for everything except arms (which we rotate
    // down ~75°) and hips (slight backward tilt).
    this._bindQuats = new Map();
    if (vrm?.humanoid) {
      for (const vrmName of Object.values(VRM_BONE_MAP)) {
        const b = vrm.humanoid.getNormalizedBoneNode(vrmName);
        if (b) this._bindQuats.set(vrmName, b.quaternion.clone());
      }
    }
  }

  load(data) {
    this.data = data;
    this.format = data?.format
      || (data?.rotations ? 'vrm-quat' : 'smpl-aa');
    this.frame = 0;
    this.lastFrameIdx = -1;
    this.flagged = false;
    // v155: FLOOR MODE. Plank / push-up / sit-up clips (pose_profile:'floor')
    // are performed lying/leaning on the ground, so the standing foot-lock
    // (which only grounds the FEET and lifts the hips) is wrong — it leaves
    // the torso/hands hovering or shoves the body around. In floor mode we
    // instead ground the LOWEST body contact point (hands, forearms, knees,
    // feet, torso) to the floor and skip the airborne/hover heuristics.
    this._floorMode = (data && data.pose_profile === 'floor') || false;
    this.prevBoneQuats.clear();
    this._hipsT0 = null;
    this._trans0 = null;
    // Capture the current bone state so we can SLERP smoothly INTO the
    // first frame of the new clip over ease_in_ms instead of snapping.
    this._easeFromQuats = new Map();
    this._easeFromHipsPos = null;
    if (this.vrm?.humanoid) {
      for (const vrmName of Object.values(VRM_BONE_MAP)) {
        const b = this.vrm.humanoid.getNormalizedBoneNode(vrmName);
        if (b) this._easeFromQuats.set(vrmName, b.quaternion.clone());
      }
      const hips = this.vrm.humanoid.getNormalizedBoneNode('hips');
      if (hips) this._easeFromHipsPos = hips.position.clone();
    }
    // v125: measure the per-VRM foot contact height ONCE, while the VRM is
    // still in its rest pose (first clip) — the shoe sole sits a few cm
    // below the lowest foot bone, so we must ground the bone to THAT height,
    // not world Y=0, or the sole sinks through the floor.
    this._measureContactY();
    this._easeStartedAt = 0;     // performance.now() — set on play()
    // Foot-weight leg-IK latch (per side). null = foot not currently
    // planted. Holds the world XZ the planted foot must hold while in
    // contact. Reset every clip so a new move starts fresh.
    this._footLatch = { left: null, right: null };
    if (this._fw) this._fw._prev = { left: null, right: null };
    // Physics weight sim: re-sync the body mass to the new clip's first
    // hip height so it doesn't have to fall in from a stale position.
    this._weightSimNeedsReset = true;
    // Weight-shift sway: clear the smoothed pelvis offset for the new clip.
    if (this._ws) this._ws.off = { x: 0, z: 0 };
    // A new clip is loaded → we are no longer holding the idle rest pose.
    this._idleHold = false;
    // v33e: per-clip ground anchor. Computed eagerly in play()
    // from the lowest foot/toe bone's world Y on frame 0 so every
    // clip lands the avatar's feet on the floor instead of
    // levitating (CMU clips in particular are recorded at varying
    // global heights). Added to hip.position.y on every frame.
    this._groundOffsetY = 0;
    // v31: bumped from 600 ms → 1200 ms so the SLERP from the
    // previous pose (usually rest-pose for the first clip of a
    // session) into the clip's frame 0 is a clear "she prepares,
    // then begins" rather than an instant snap into mid-dance.
    // v24: dropped to 300 ms. 1.2 s was eating 15-30% of every short
    // AIST clip (4-8 s), making the avatar look like she was thinking
    // before dancing. 300 ms still hides the snap but lets the move
    // start *immediately*.
    this.easeInMs = 300;
  }

  play({ speed = 1.0, mirror = false, loop = 'auto',
         fromFrame = null, toFrame = null } = {}) {
    if (!this.data) return;
    this._idleHold = false;   // playback overrides the idle rest pose
    this.speed = Math.min(1.5, Math.max(0.25, speed));
    this.mirror = !!mirror;
    // v24: auto-loop short clips. Most AIST++ choreographies are
    // 4-8 s — one playthrough at 1× is barely enough for a beginner
    // to register the move. When `loop` is left at its default
    // ('auto'), repeat clips whose duration (post-window) is < 6 s
    // so the move is shown 2-3 times before the avatar resets to
    // idle. Explicit booleans (true/false) from drill controller
    // still win.
    if (loop === 'auto') {
      const _dur = (this.data?.duration_s) ||
                   ((this.data?.frames || this.data?.n_frames || 0) /
                    (this.data?.fps || 60));
      this.loop = (_dur > 0 && _dur < 6.0);
    } else {
      this.loop = !!loop;
    }
    // Optional drill window — frame indices [from, to). Default = whole clip.
    const nFrames = this.data.frames || this.data.n_frames || 0;
    this.windowFrom = (fromFrame == null) ? 0 : Math.max(0, fromFrame | 0);
    this.windowTo   = (toFrame   == null) ? nFrames
                                          : Math.min(nFrames, toFrame | 0);
    if (this.windowTo <= this.windowFrom) this.windowTo = nFrames;
    this.frame = this.windowFrom;
    this.prevBoneQuats.clear();
    this.playing = true;
    this.flagged = false;
    this._softFlagFired = false;
    this._easeStartedAt = performance.now();
    // v33e: ground anchor — measure where the feet land on frame 0
    // with offset=0, then store the negative as a fixed Y shift so
    // the lowest foot/toe sits on the avatar's floor plane.
    // v14: BAKED CORRECTIONS — if the server attached `data.corrections`
    // (from offline analyze_motions.py), use those instead of the
    // per-load runtime heuristics. Heuristics stay as fallback for
    // clips not yet in corrections.json.
    //
    // v14a: SAFETY CLAMP. The offline FK uses the JSON's bone rest
    // and animated rotations directly, but the three-vrm normalized
    // humanoid sometimes applies an internal Z-up→Y-up correction
    // (CMU rigs especially) that my FK doesn't replicate. Result was
    // CMU jog clips reporting foot_p10 ≈ +1.6 m, so the baked
    // ground_offset_y = -1.6 m sank the avatar through the floor.
    // Treat any baked offset whose magnitude looks implausible
    // (> 0.4 m drop or > 0.2 m lift) as untrusted and fall back to
    // runtime FK + the existing ±1.2 m runtime clamp.
    const corr = this.data?.corrections || null;
    this._groundOffsetY = 0;
    // v69: reset the anti-hover state so a new clip starts grounded.
    this._hoverDropY = 0;
    this._airborneT = 0;
    this._soleHist = null;   // v126: reset rolling sole-height window
    if (corr && typeof corr.ground_offset_y === 'number' &&
        corr.ground_offset_y >= -0.40 && corr.ground_offset_y <= 0.20) {
      this._groundOffsetY = corr.ground_offset_y;
    } else {
      this._groundOffsetY = this._computeGroundOffset();
    }
    // v34c: ROOT-MOTION BAKE — replaces the cheap "translate hip by
    // hips_translation and hope feet stay put" with a proper planted-
    // foot anchor. Simulates the clip once, detects which foot is
    // grounded each frame, and computes a per-frame XZ correction so
    // the planted foot stays fixed in world space across its stance.
    // Result: no more in-place "treadmill walking" and no more
    // sliding feet across the floor. Free vertical motion preserved
    // for jumps/squats.
    this._rootBakeXZ = null;
    this._rootBakeMean = null;
    this._rootBakeStart = this.windowFrom || 0;
    // v34d: AUTO-YAW ALIGN. Some CMU subjects were captured facing
    // sideways relative to the global frame, so locomotion clips look
    // like the avatar is moonwalking left/right across the stage
    // instead of toward/away from camera. Detect the dominant XZ
    // travel direction from raw hips_translation in the first ~1.5 s
    // of the playback window and rotate `vrm.scene` once so that
    // direction is aligned with the world ±Z axis. Picks whichever
    // of {0, π} the source angle is closer to (keeps the dancer's
    // chosen facing — we only fix sideways drift, never flip her
    // around if she was already facing camera).
    //
    // Bake runs FIRST in clip-local frame (no yaw applied yet), then
    // we set rotation.y. Because hips.position values fed to the
    // skeleton are in vrm.scene-local space, rotating the parent
    // rotates skeleton + translation + bake correction together —
    // no per-frame matrix math needed.
    if (typeof this._baseYawY !== 'number') {
      this._baseYawY = this.vrm?.scene ? this.vrm.scene.rotation.y : 0;
    }
    if (this.vrm?.scene) this.vrm.scene.rotation.y = this._baseYawY;
    if (corr && typeof corr.yaw_rad === 'number') {
      this._yawCorrection = corr.yaw_rad;
    } else {
      this._yawCorrection = this._computeYawCorrection();
    }
    // v26: prefer the OFFLINE foot-lock bake stored in corrections.
    // The python bake walks the same FK math but is deterministic,
    // not dependent on three-vrm runtime numerical noise, and gets
    // a chance to use proper stance detection (clip-relative Y +
    // velocity, lower-foot tie-break, no static-floor-snap fighting
    // it). Per-clip data: corrections.foot_lock = {
    //   root_dxz: [[dx,dz], ...n_frames],
    //   n, fps, slip_before_m, slip_after_m, contact_pct: {L,R}
    // }. Falls back to runtime bake when offline data missing
    // (e.g. SMPL-fallback clips, v27 will close that gap too).
    const fl = corr && corr.foot_lock;
    if (fl && Array.isArray(fl.root_dxz) && fl.root_dxz.length > 0) {
      this._rootBakeXZ = fl.root_dxz.map(p => ({ x: p[0], z: p[1] }));
      this._rootBakeStart = this.windowFrom || 0;
      this._rootBakeMean = null; // recomputed lazily on first killXZ frame
      try { console.log('[motion_player v26] offline foot-lock',
        'n=', fl.n, 'slip', (fl.slip_before_m*100).toFixed(1),
        'cm ->', (fl.slip_after_m*100).toFixed(1), 'cm'); } catch(_) {}
    } else {
      try { this._bakeRootMotion(); } catch (e) {
        console.warn('[motion_player] root bake failed:', e);
        this._rootBakeXZ = null;
      }
    }
    if (this.vrm?.scene && this._yawCorrection) {
      this.vrm.scene.rotation.y = this._baseYawY + this._yawCorrection;
    }
    // v14c: runtime shoulder-line yaw refinement. The offline pelvis-
    // quat yaw was unreliable (CMU rigs bake a Z-up→Y-up rotation into
    // Hips that three-vrm normalizes away) and travel-based yaw only
    // aligns motion direction, not body facing. After the first frame
    // is applied, measure the world-space shoulder line and rotate the
    // scene so the body actually faces -Z (the camera). Runs on every
    // clip — it's a no-op when the body is already facing camera.
    this._scheduleShoulderYawRefine();
    // v34e: in-place lock — if the clip has tiny vertical hip range
    // (warmups / arm waves / stretches), force per-frame foot-to-floor
    // so the static groundOffset miscalibration can't levitate her.
    let _isInPlace = false;
    if (corr && typeof corr.in_place === 'boolean') {
      _isInPlace = corr.in_place;
    } else {
      _isInPlace = this._detectInPlaceClip();
    }
    this._lockToFloor = _isInPlace;
    // v21 UNIVERSAL FOOT ANCHOR. User saw the avatar floating 20–30 cm
    // off the floor on jogs because the static `_groundOffsetY` only
    // calibrates against the clip's single lowest foot frame; every
    // other frame ends up hovering. Default _lockToFloor=ON for ALL
    // clips so `_applyFootLock` pulls hips down to plant the lowest
    // foot at Y=0 EVERY FRAME. Only verified high jumps are exempt
    // (hip vertical travel > 25 cm AND clip flagged is_jump) so a
    // real hop can still leave the floor for a few frames.
    const _hipRange = corr?.hip_y_range || 0;
    const _isRealJump = !!(corr && corr.is_jump === true && _hipRange > 0.25);
    if (!_isRealJump) this._lockToFloor = true;
    // v15: SUPPRESS HIP BOB. Some CMU rehab clips (subject 105, etc.)
    // were captured as PT exercises — the subject squats 30-40 cm
    // while lifting a knee or doing an arm reach. is_jump=false so
    // it's not a real hop; the vertical drop is a baked rehab artefact
    // we never want to play back as a "warmup". When the offline
    // analyzer flagged the clip (in_place && !is_jump && hip_y_range
    // > 0.20), clamp dy hard so only the upper body's intended motion
    // survives. Real jumps (is_jump=true) are NEVER suppressed.
    this._suppressHipBob = !!(corr && corr.suppress_hip_bob);
    // v15: KILL XZ DRIFT FOR IN-PLACE CLIPS. The baked root-motion
    // correction (this._rootBakeXZ) is correct for locomotion, but
    // for arm warmups / stretches / idle clips it still ends up
    // translating the avatar laterally by a few cm per frame because
    // the source mocap had millimetre wobble in hip XZ. Hard-zero
    // dx/dz when the clip is flagged in_place. Locomotion clips
    // (in_place=false) keep the bake.
    this._killXZ = !!_isInPlace;
    try { console.log('[motion_player v15]',
      'baked=', !!corr,
      'yaw=', this._yawCorrection,
      'baseYawY=', this._baseYawY,
      'sceneRotY=', this.vrm?.scene?.rotation.y,
      'lockToFloor=', this._lockToFloor,
      'killXZ=', this._killXZ,
      'suppressHipBob=', this._suppressHipBob,
      'groundOffsetY=', this._groundOffsetY,
      'isJump=', corr?.is_jump); } catch (_) {}
  }

  /** v14c: after the rig has had one frame applied, measure the
   *  world-space shoulder line and rotate the scene so the avatar
   *  actually faces -Z (camera). Catches CMU rigs whose default
   *  orientation differs from camera-forward despite baseYawY=π.
   *  Right-axis = leftShoulder→rightShoulder. forward = right × up.
   *  Adjustment = atan2(-fx, fz) to bring forward onto +Z (which
   *  three-vrm renders as facing the camera given baseYawY=π). */
  _scheduleShoulderYawRefine() {
    // v19 GUARD: skip shoulder refinement for any clip that carries
    // a real travel-based yaw correction (locomotion). The travel
    // vector is authoritative for "which way is the character
    // walking" — the shoulder line only tells us "which way the
    // chest is currently facing", which on frame 0 of a jog is
    // perpendicular to travel (side-step pose) and would cause the
    // refine to add ~90° and cancel the travel correction, making
    // the avatar moonwalk sideways across the floor (cmu_49_49_08
    // and friends). User complaint: "she is still moving left
    // sideways". Only run the refine for in-place clips where there
    // is no travel signal to fight with.
    if (Math.abs(this._yawCorrection || 0) > 0.05) return;
    if (this._shoulderYawTimer) clearTimeout(this._shoulderYawTimer);
    this._shoulderYawTimer = setTimeout(() => {
      try {
        const vrm = this.vrm;
        if (!vrm?.humanoid || !vrm.scene) return;
        // upperArm gives a much larger XZ signal than shoulder
        // (shoulder bones sit near the spine and project to ~4 cm).
        const ls = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
        const rs = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
        if (!ls || !rs) return;
        vrm.scene.updateMatrixWorld(true);
        const lsp = ls.getWorldPosition(new ls.position.constructor());
        const rsp = rs.getWorldPosition(new rs.position.constructor());
        const rx = rsp.x - lsp.x;
        const rz = rsp.z - lsp.z;
        const fx = rz;
        const fz = -rx;
        const mag = Math.hypot(fx, fz);
        if (mag < 0.03) return;
        const delta = Math.atan2(-fx, fz);
        if (Math.abs(delta) < 0.1) return;
        vrm.scene.rotation.y += delta;
        try { console.log('[motion_player v14c] upperArm-yaw refine delta=', delta.toFixed(3),
          'fwd=', fx.toFixed(3), fz.toFixed(3), 'mag=', mag.toFixed(3)); } catch (_) {}
      } catch (e) {
        try { console.warn('[motion_player v14c] shoulder yaw refine failed', e); } catch (_) {}
      }
    }, 250);
  }

  /** v34e: returns true when the clip has < 15 cm vertical hip
   *  range across the playback window — meaning it's an in-place
   *  clip (arm wave, stretch, idle, head turns) for which we can
   *  safely force the feet to the floor every frame without
   *  clobbering a real jump. */
  _detectInPlaceClip() {
    const hT = this.data?.hips_translation;
    if (!hT || !Array.isArray(hT)) return false;
    const sf = this.windowFrom || 0;
    const ef = Math.min(hT.length, this.windowTo || hT.length);
    let yMin = Infinity, yMax = -Infinity;
    for (let i = sf; i < ef; i++) {
      const t = hT[i]; if (!t) continue;
      const y = t[1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    if (!isFinite(yMin)) return false;
    return (yMax - yMin) < 0.15;
  }

  /** v34m: travel-direction yaw. The actual root cause of "walks
   *  sideways" was that mocap world axes are arbitrary — a CMU jog
   *  recorded along the lab's +X axis appears sideways under our
   *  camera-facing baseYawY. Compute the clip's dominant XZ travel
   *  vector from frame 0 to a sample point ~2 s in (or end of
   *  window), and rotate the scene so that vector points into the
   *  screen (world -Z). Result: jogs/walks travel away from camera
   *  naturally instead of drifting sideways. Returns 0 for in-place
   *  clips (warmups, arm waves, stretches) so they keep facing
   *  camera. */
  _computeYawCorrection() {
    const hT = this.data?.hips_translation;
    if (!hT || !Array.isArray(hT) || hT.length < 4) return 0;
    const sf = this.windowFrom || 0;
    const fps = this.data?.fps || 30;
    const wEnd = Math.min(hT.length, this.windowTo || hT.length);
    const ef = Math.min(wEnd, sf + Math.round(fps * 2));
    if (ef - sf < 2) return 0;
    const a = hT[sf], b = hT[ef - 1];
    if (!a || !b) return 0;
    const dx = b[0] - a[0];
    const dz = b[2] - a[2];
    const len = Math.hypot(dx, dz);
    // Sub-30 cm travel in 2 s → in-place clip; do not rotate.
    if (len < 0.30) return 0;
    // Rotate so the source travel vector (dx, dz) lands on world -Z.
    // Rotation around Y by θ maps (x, z) → (x cosθ + z sinθ,
    // -x sinθ + z cosθ). Solving for resulting x = 0 and resulting
    // z < 0 gives θ = π - atan2(dx, dz).
    return Math.PI - Math.atan2(dx, dz);
  }

  /** v33e: compute per-clip Y offset so feet land on floor.
   *  Silently applies frame 0 (with current offset=0), reads the
   *  lowest foot/toe bone's world Y, returns the negative so adding
   *  it back lifts the avatar to floor level. Called once in play().
   *
   *  v34: sample many frames (not just frame 0) and use the GLOBAL
   *  lowest-foot Y across the clip window. Otherwise clips that
   *  start mid-step (one foot lifted, common in CMU walking/running
   *  captures) calibrate against the only grounded foot, then sink
   *  into the floor when the opposite foot lands. */
  _computeGroundOffset() {
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return 0;
    const nFrames = this.data?.frames || this.data?.n_frames || 0;
    if (nFrames <= 0) return 0;
    const startFrame = this.windowFrom || 0;
    const endFrame   = Math.min(nFrames, this.windowTo || nFrames);
    const span       = Math.max(1, endFrame - startFrame);
    // Sample up to 48 evenly-spaced frames across the playback
    // window. Cheap (one FK per sample) and deterministic.
    const N = Math.min(48, span);
    const probe = new THREE.Vector3();
    const footBones = ['leftToes', 'rightToes', 'leftFoot', 'rightFoot'];
    let minY = Infinity;
    for (let s = 0; s < N; s++) {
      const f = startFrame + Math.floor((s * (span - 1)) / Math.max(1, N - 1));
      try {
        if (this.format === 'vrm-quat') this._applyFrameQuat(f);
        else                            this._applyFrameAA(f);
      } catch (e) { continue; }
      vrm.scene.updateMatrixWorld(true);
      for (const name of footBones) {
        const b = vrm.humanoid.getNormalizedBoneNode(name);
        if (!b) continue;
        b.getWorldPosition(probe);
        if (probe.y < minY) minY = probe.y;
      }
    }
    if (!isFinite(minY)) return 0;
    // Floor plane = avatar root world Y (the parent group anchors
    // the avatar at floor level).
    const rootY = vrm.scene.getWorldPosition(new THREE.Vector3()).y;
    // Clamp: refuse offsets > 1 m so a broken clip can't yeet the
    // avatar through the ceiling.
    const off = rootY - minY;
    if (off > 1.2)  return 1.2;
    if (off < -1.2) return -1.2;
    return off;
  }

  /** v34c: bake per-frame XZ root offset from planted-foot contacts.
   *
   *  Why: CMU/AMASS data was often recorded on a treadmill or in a
   *  small capture volume — the source `hips_translation` barely
   *  moves while feet step in place, so the avatar looks like she
   *  walks on an invisible treadmill. AIST data DOES translate, but
   *  per-frame foot world-position still drifts because the source
   *  retarget didn't know about VRM bone lengths, so feet slide.
   *
   *  Algorithm:
   *    1. Simulate the clip frame-by-frame with foot-lock SUPPRESSED.
   *       Record world XZ of leftFoot and rightFoot every frame.
   *    2. For each frame, classify the planted foot:
   *         - if both feet are near floor (Y < 0.08 m) AND XZ velocity
   *           is low (< 0.8 m/s post-window) → the lower one is planted
   *         - if only one foot meets that test → it's planted
   *         - else → airborne (no anchor this frame)
   *    3. Walk the timeline: when a stance starts, lock the planted
   *       foot's world XZ as the "anchor". Add a running correction
   *       cumX/cumZ so (foot_pos + correction) keeps matching anchor.
   *       Low-pass the per-frame correction delta (ALPHA=0.35) so the
   *       handoff between L→R stances is smooth, not a snap.
   *    4. Boxcar-smooth the resulting correction array (window=4).
   *    5. Store as `this._rootBakeXZ[i] = {x, z}`. `_applyFrameQuat`
   *       adds these to the per-frame hip translation.
   *
   *  Cost: one full per-frame FK pass at load time (~30-150 ms for a
   *  typical 5-30 s clip). Runtime cost is zero — just an array
   *  lookup. */
  _bakeRootMotion() {
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    if (this.format !== 'vrm-quat') return;   // AA path: source has real trans
    const data = this.data;
    const nFrames = data?.frames || data?.n_frames || 0;
    if (nFrames < 4) return;
    const startF = this.windowFrom || 0;
    const endF   = Math.min(nFrames, this.windowTo || nFrames);
    const span   = endF - startF;
    if (span < 4) return;
    const fps = data.fps || 30;
    const dt  = 1 / fps;

    // Pre-pass: snapshot rest, then simulate each frame and record
    // foot world positions. Suppress foot-lock so we read the raw
    // pose driven purely by joint rotations + hips_translation.
    const lfBone = vrm.humanoid.getNormalizedBoneNode('leftFoot');
    const rfBone = vrm.humanoid.getNormalizedBoneNode('rightFoot');
    const hpBone = vrm.humanoid.getNormalizedBoneNode('hips');
    if (!lfBone || !rfBone || !hpBone) return;
    const savedHipsPos = hpBone.position.clone();
    this._suppressFootLock = true;
    // Disable ease-in slerp during bake so we read TRUE clip
    // rotations, not a blend with the pre-play rest pose.
    const savedEaseStart = this._easeStartedAt;
    this._easeStartedAt = 0;

    const lFootXZ = new Float32Array(span * 2);
    const rFootXZ = new Float32Array(span * 2);
    const lFootY  = new Float32Array(span);
    const rFootY  = new Float32Array(span);
    const probe = new THREE.Vector3();
    for (let i = 0; i < span; i++) {
      const f = startF + i;
      try { this._applyFrameQuat(f); } catch (e) { /* ignore */ }
      vrm.scene.updateMatrixWorld(true);
      lfBone.getWorldPosition(probe);
      lFootXZ[i*2]   = probe.x;
      lFootXZ[i*2+1] = probe.z;
      lFootY[i]      = probe.y;
      rfBone.getWorldPosition(probe);
      rFootXZ[i*2]   = probe.x;
      rFootXZ[i*2+1] = probe.z;
      rFootY[i]      = probe.y;
    }
    this._suppressFootLock = false;
    hpBone.position.copy(savedHipsPos);
    this.prevBoneQuats.clear();        // bake polluted the smoother
    this._hipsT0 = null;               // and the per-clip origin
    this._easeStartedAt = savedEaseStart;

    // Velocity per foot (m/s) — central difference where possible.
    const vL = new Float32Array(span);
    const vR = new Float32Array(span);
    for (let i = 0; i < span; i++) {
      const a = Math.max(0, i - 1);
      const b = Math.min(span - 1, i + 1);
      const stride = Math.max(1, b - a) * dt;
      vL[i] = Math.hypot(lFootXZ[b*2]   - lFootXZ[a*2],
                         lFootXZ[b*2+1] - lFootXZ[a*2+1]) / stride;
      vR[i] = Math.hypot(rFootXZ[b*2]   - rFootXZ[a*2],
                         rFootXZ[b*2+1] - rFootXZ[a*2+1]) / stride;
    }

    // Contact classification per frame: 'L' | 'R' | null (airborne).
    // v17: Y_GROUND was an absolute 0.10 m, but the three-vrm
    // normalized ankle bone for our rigs sits at ~0.18-0.22 m above
    // the floor at the planted-foot rest pose. With the absolute
    // threshold, no frame ever counted as "in contact" — the bake
    // produced an all-zero correction and locomotion clips like
    // cmu_02_02_02 had a 52 cm planted-foot slip (browser-recorded).
    // Use a clip-relative window: lowest foot Y across the clip
    // (the foot at its deepest plant) + 5 cm tolerance. Falls back
    // to the old 0.10 m floor if the rig is genuinely close to it.
    const Y_GROUND_REL_TOL = 0.05;
    const lFootMin = Math.min.apply(null, lFootY);
    const rFootMin = Math.min.apply(null, rFootY);
    const Y_GROUND = Math.max(0.10, Math.min(lFootMin, rFootMin) + Y_GROUND_REL_TOL);
    const V_PLANT  = 0.6;        // m/s XZ velocity threshold
    const contact = new Array(span);
    for (let i = 0; i < span; i++) {
      const lDown = lFootY[i] < Y_GROUND;
      const rDown = rFootY[i] < Y_GROUND;
      const lSlow = vL[i] < V_PLANT;
      const rSlow = vR[i] < V_PLANT;
      const lOK = lDown && lSlow;
      const rOK = rDown && rSlow;
      if (lOK && rOK)      contact[i] = (lFootY[i] <= rFootY[i]) ? 'L' : 'R';
      else if (lOK)        contact[i] = 'L';
      else if (rOK)        contact[i] = 'R';
      else                 contact[i] = null;
    }

    // Walk timeline, accumulate XZ correction.
    const ALPHA = 0.35;
    let cumX = 0, cumZ = 0;
    let stance = null;       // 'L' | 'R'
    let anchorX = 0, anchorZ = 0;
    const corrX = new Float32Array(span);
    const corrZ = new Float32Array(span);
    for (let i = 0; i < span; i++) {
      const c = contact[i];
      if (c && c !== stance) {
        // Stance begins (or switches feet). Lock current corrected
        // foot position as the new anchor; the previous correction
        // carries forward, so the handoff is continuous.
        stance = c;
        const fx = (c === 'L') ? lFootXZ[i*2]   : rFootXZ[i*2];
        const fz = (c === 'L') ? lFootXZ[i*2+1] : rFootXZ[i*2+1];
        anchorX = fx + cumX;
        anchorZ = fz + cumZ;
      }
      if (stance && c === stance) {
        const fx = (stance === 'L') ? lFootXZ[i*2]   : rFootXZ[i*2];
        const fz = (stance === 'L') ? lFootXZ[i*2+1] : rFootXZ[i*2+1];
        // Target: anchorX/Z. Current world: fx+cumX, fz+cumZ.
        const dX = anchorX - (fx + cumX);
        const dZ = anchorZ - (fz + cumZ);
        cumX += dX * ALPHA;
        cumZ += dZ * ALPHA;
      } else if (!c) {
        // Airborne: leak correction toward zero very slowly so we
        // don't accumulate drift across long flight phases.
        cumX *= 0.995;
        cumZ *= 0.995;
        stance = null;
      }
      corrX[i] = cumX;
      corrZ[i] = cumZ;
    }
    // Boxcar smoothing (±2 frames) to kill stance-transition kinks.
    const SMOOTH = 2;
    const out = new Array(span);
    for (let i = 0; i < span; i++) {
      let sx = 0, sz = 0, n = 0;
      const lo = Math.max(0, i - SMOOTH);
      const hi = Math.min(span - 1, i + SMOOTH);
      for (let j = lo; j <= hi; j++) {
        sx += corrX[j];
        sz += corrZ[j];
        n++;
      }
      out[i] = { x: sx / n, z: sz / n };
    }
    this._rootBakeXZ = out;
    this._rootBakeStart = startF;
  }
  stop()   { this.playing = false; this._idleHold = true; }
  pause()  { this.playing = false; }
  resume() { this.playing = true; this._idleHold = false; }
  restart() {
    if (!this.data) return;
    const first = this.windowFrom || 0;
    this.frame = first;
    this.lastFrameIdx = -1;
    this.prevBoneQuats.clear();
    this._easeStartedAt = performance.now();
    this._idleHold = false;
    this.playing = true;
    this.flagged = false;
    this.poseFrame(first);
  }

  /** Idle grounding: while holding the rest pose (clip ended/stopped),
   *  playback's per-frame foot-lock no longer runs, so the rest legs +
   *  idle hip height can leave the feet sunk into or floating above the
   *  floor. Shift the hips so the lowest foot kisses world Y = 0. Called
   *  by the render loop AFTER AvatarLife has set the idle hip position. */
  _groundForIdle() {
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    if (!hips) return;
    vrm.scene.updateMatrixWorld(true);
    const probe = new THREE.Vector3();
    let minY = Infinity;
    for (const n of ['leftToes', 'rightToes', 'leftFoot', 'rightFoot']) {
      const b = vrm.humanoid.getNormalizedBoneNode(n);
      if (!b) continue;
      b.getWorldPosition(probe);
      if (probe.y < minY) minY = probe.y;
    }
    if (!isFinite(minY)) return;
    // Move lowest foot to the floor (both lift if sunk, drop if floating).
    // Cap at 0.5 m so a freak pose can't teleport her vertically.
    const shift = Math.max(-0.5, Math.min(0.5, -minY));
    if (Math.abs(shift) > 5e-4) hips.position.y += shift;
  }

  /** Drive only a subset of bones from the clip; pin the rest at
   *  bind-pose. Pass null/empty to clear isolation.
   *
   *  parts may be either a body-part group name ('arms', 'legs',
   *  'torso', 'head', 'hands', 'feet', 'left', 'right'), an array
   *  of group names, or an array of raw VRM bone names. */
  isolate(parts) {
    if (!parts || (Array.isArray(parts) && parts.length === 0)) {
      this._isolated = null;
      return;
    }
    const arr = Array.isArray(parts) ? parts : [parts];
    const bones = new Set();
    for (const p of arr) {
      const group = BODY_PART_GROUPS[p];
      if (group) group.forEach((b) => bones.add(b));
      else bones.add(p);          // treat as raw bone name
    }
    this._isolated = bones;
  }
  unisolate() { this._isolated = null; }

  /** Apply the procedural rest-pose target to a single bone in-place.
   *  Used both by applyRestPose() and by the per-frame loop when a
   *  bone is excluded by isolation — so isolated-out bones hold a
   *  natural standing pose instead of the rigger's T-pose. */
  _setRestPoseBone(boneName, bone) {
    if (!bone) return;
    if (boneName === 'leftUpperArm') {
      // Shoulder abduct ~76° (Z) + small forward swing (X) so the
      // arm hangs *in front of* the body rather than glued to the
      // hip — reads as relaxed standing, not military attention.
      bone.quaternion.set(0.06, 0, 0.62, 0.78).normalize();
      return;
    }
    if (boneName === 'rightUpperArm') {
      bone.quaternion.set(0.06, 0, -0.62, 0.78).normalize();
      return;
    }
    if (boneName === 'leftLowerArm') {
      // ~14° elbow flex so the arm has a natural slight bend.
      bone.quaternion.set(0, 0.12, 0, 0.993).normalize();
      return;
    }
    if (boneName === 'rightLowerArm') {
      bone.quaternion.set(0, -0.12, 0, 0.993).normalize();
      return;
    }
    if (boneName === 'leftUpperLeg' || boneName === 'rightUpperLeg') {
      // ~3° hip flex — kills the locked-knee "plank" feel.
      bone.quaternion.set(0.026, 0, 0, 0.9997).normalize();
      return;
    }
    if (boneName === 'leftLowerLeg' || boneName === 'rightLowerLeg') {
      // ~5° knee flex.
      bone.quaternion.set(0.043, 0, 0, 0.9991).normalize();
      return;
    }
    if (boneName === 'spine') {
      // Tiny anterior pelvic tilt → upright but not stiff.
      bone.quaternion.set(-0.018, 0, 0, 0.9998).normalize();
      return;
    }
    if (boneName === 'hips') {
      bone.quaternion.set(-0.04, 0, 0, 0.9992).normalize();
      return;
    }
    const bq = this._bindQuats?.get(boneName);
    if (bq) bone.quaternion.copy(bq);
  }

  /** Clamp a target quaternion so the rotation away from the bone's
   *  bind-pose does not exceed the anatomic limit for that bone.
   *  Mutates qOut in place. Returns qOut. */
  _clampAnatomic(boneName, qOut) {
    const bind = this._bindQuats?.get(boneName);
    const maxAng = BONE_MAX_ANGLE[boneName];
    if (!bind || !maxAng) return qOut;
    // delta = bind^-1 * qOut  (the rotation away from rest)
    if (!this._tmpInv) {
      this._tmpInv = new THREE.Quaternion();
      this._tmpDelta = new THREE.Quaternion();
      this._tmpIdent = new THREE.Quaternion(0, 0, 0, 1);
    }
    this._tmpInv.copy(bind).invert();
    this._tmpDelta.copy(this._tmpInv).multiply(qOut);
    // Pick the shorter rotation (handle quat double-cover).
    if (this._tmpDelta.w < 0) {
      this._tmpDelta.x = -this._tmpDelta.x;
      this._tmpDelta.y = -this._tmpDelta.y;
      this._tmpDelta.z = -this._tmpDelta.z;
      this._tmpDelta.w = -this._tmpDelta.w;
    }
    const w = Math.min(1, Math.max(-1, this._tmpDelta.w));
    const ang = 2 * Math.acos(w);
    if (ang <= maxAng || !isFinite(ang)) return qOut;
    // SLERP from identity toward delta by (maxAng / ang) → resulting
    // rotation has angle = maxAng but preserves the original axis.
    const t = maxAng / ang;
    this._tmpDelta.copy(this._tmpIdent).slerp(this._tmpDelta, t);
    // qOut = bind * clampedDelta
    qOut.copy(bind).multiply(this._tmpDelta);
    return qOut;
  }

  /** Temporal lowpass: SLERP from prevApplied toward qOut by alpha.
   *  Removes single-frame spikes and high-freq jitter without
   *  noticeably delaying the choreography. Mutates qOut. */
  _lowpassQuat(prev, qOut) {
    if (!prev) return qOut;
    if (!this._tmpTarget) this._tmpTarget = new THREE.Quaternion();
    this._tmpTarget.copy(qOut);
    return qOut.copy(prev).slerp(this._tmpTarget, QUAT_LOWPASS_ALPHA);
  }

  /** Procedural relaxed-standing rest pose. KEEP THIS SIMPLE. Every
   *  asymmetric tweak I tried (contrapposto, palm-in twist, hip drop)
   *  compounded into unnatural poses because three.js applies XYZ Euler
   *  in a specific order on bones whose local axes differ per VRM rig.
   *  Strategy: only the bare minimum — bring the arms straight down
   *  via a pure Z rotation (verified from clip data) and leave every
   *  other bone at identity. AvatarLife adds breath/sway on top. */
  applyRestPose() {
    const vrm = this.vrm;
    if (!vrm?.humanoid) return;
    for (const vrmName of Object.values(VRM_BONE_MAP)) {
      const b = vrm.humanoid.getNormalizedBoneNode(vrmName);
      this._setRestPoseBone(vrmName, b);
    }
    // v33c: relax finger curls. VRM default finger pose on many rigs
    // is a half-fist that makes the avatar look tense / robotic. A
    // soft natural splay reads as alive.
    this._relaxFingers();
    // Snapshot for ease-in next time a clip plays.
    if (!this._easeFromQuats) this._easeFromQuats = new Map();
    for (const vrmName of Object.values(VRM_BONE_MAP)) {
      const b = vrm.humanoid.getNormalizedBoneNode(vrmName);
      if (b) this._easeFromQuats.set(vrmName, b.quaternion.clone());
    }
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    if (hips) this._easeFromHipsPos = hips.position.clone();
  }

  /** Reset every finger bone to a soft natural splay (identity local
   *  rotation = the VRM bind pose). Many VRM rigs ship in a closed
   *  half-fist; that reads as tense and breaks the "alive" feel. */
  _relaxFingers() {
    const vrm = this.vrm;
    if (!vrm?.humanoid) return;
    const sides = ['left', 'right'];
    const digits = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
    const segs = ['Proximal', 'Intermediate', 'Distal'];
    for (const side of sides) {
      for (const d of digits) {
        for (const s of segs) {
          const name = `${side}${d}${s}`;
          const b = vrm.humanoid.getNormalizedBoneNode(name);
          if (!b) continue;
          // Normalised bones bind at identity in three-vrm — identity
          // here = the rig's natural splay.
          b.quaternion.set(0, 0, 0, 1);
        }
      }
    }
  }

  /** Snapshot the current bone quats + hip position as the ease-FROM
   *  pose, so the next frames SLERP from here. Used at the loop seam to
   *  avoid a hard snap between very different end/start poses. */
  _captureEaseFrom() {
    if (!this.vrm?.humanoid) return;
    if (!this._easeFromQuats) this._easeFromQuats = new Map();
    for (const vrmName of Object.values(VRM_BONE_MAP)) {
      const b = this.vrm.humanoid.getNormalizedBoneNode(vrmName);
      if (b) this._easeFromQuats.set(vrmName, b.quaternion.clone());
    }
    const hips = this.vrm.humanoid.getNormalizedBoneNode('hips');
    if (hips) this._easeFromHipsPos = hips.position.clone();
  }

  /** Pose the skeleton at a single frame without starting playback.
   *  Used to settle the avatar into a natural idle stance from a real
   *  clip's frame 0, so it never sits in T-pose. */
  poseFrame(idx) {
    if (!this.data) return;
    const nFrames = this.data.frames || this.data.n_frames || 0;
    if (nFrames <= 0) return;
    const i = Math.max(0, Math.min(nFrames - 1, idx | 0));
    if (this.format === 'vrm-quat') this._applyFrameQuat(i);
    else                            this._applyFrameAA(i);
    this.lastFrameIdx = i;
    this.frame = i;
  }

  update(dt) {
    if (!this.playing || !this.data || this.flagged) return;
    this._lastDt = dt;               // v69: anti-hover clamp reads this
    const fps = this.data.fps || 30;
    const nFrames = this.data.frames || this.data.n_frames || 0;
    if (nFrames <= 0) return;
    this.frame += dt * fps * this.speed;
    const winTo = this.windowTo || nFrames;
    if (this.frame >= winTo - 1) {
      if (this.loop) {
        // LOOP SEAM EASE: many clips have a very different end pose vs
        // start pose (e.g. House Bounce: right forearm differs 50°, hips
        // shifted 11 cm). Jumping straight to frame 0 snaps those bones
        // = a visible jerk + sideways pop every loop. Capture the current
        // (end) pose and restart the ease timer so _applyFrameQuat blends
        // from it into frame 0 over loopEaseMs instead of snapping.
        this._captureEaseFrom();
        this._easeStartedAt = performance.now();
        this.frame = this.windowFrom || 0;
        this.prevBoneQuats.clear();
        this._hipsT0 = null;          // re-anchor hip origin to frame 0
        if (this.onloop) this.onloop();
      } else {
        this.frame = winTo - 1;
        this.playing = false;
        // Clip finished (not looping): legs/arms are frozen on the last
        // frame. Flag idle so the render loop restores the grounded
        // standing pose instead of leaving her tilted / feet floating.
        this._idleHold = true;
        if (this.onend) this.onend();
      }
    }
    const idx = Math.floor(this.frame);
    if (idx === this.lastFrameIdx) return;
    this.lastFrameIdx = idx;
    if (this.format === 'vrm-quat') this._applyFrameQuat(idx);
    else                            this._applyFrameAA(idx);
    if (this.onframe) this.onframe(idx, nFrames);
  }

  // -- Format A: pre-retargeted bone-local quaternions ----------------
  _applyFrameQuat(idx) {
    const vrm = this.vrm;
    if (!vrm?.humanoid) return;
    const rotations = this.data.rotations || {};
    const hipsT = this.data.hips_translation;
    const fps = this.data.fps || 30;
    const dt = 1 / fps;
    const maxAng = this.limits.max_ang_speed || 15.0;
    const q = new THREE.Quaternion();

    // TIER 1 — PURE PLAYBACK (default ON). Play the retargeted motion
    // FAITHFULLY, the way Blender plays a baked clip: no runtime pose
    // "corrections". We proved (native-skeleton render) that the data
    // is faithful and that the band-aids — anatomical clamp, temporal
    // lowpass, foot-IK/foot-weight, weight-shift, self-collision push —
    // are what diverge the browser from Blender (24-35 cm at the foot).
    // Pure mode keeps ONLY: hemisphere continuity + angular-speed guard
    // (safety, not distortion), a one-time ground offset, lift-only
    // foot-lock (so feet don't sink), and the framing clamps. Set
    // window.__coachPureMode=false to restore the old correction stack.
    const pure = (typeof window === 'undefined') || window.__coachPureMode !== false;
    this._pure = pure;

    // Ease-in factor: 0 at the moment play() was called → 1 after
    // easeInMs. While < 1 we SLERP from the captured pre-play pose
    // into the clip's frame using a smoothstep curve.
    const easeT = this._easeStartedAt
      ? Math.min(1, (performance.now() - this._easeStartedAt) / this.easeInMs)
      : 1;
    const easeS = easeT * easeT * (3 - 2 * easeT);

    // ---- hips translation -------------------------------------------
    if (hipsT && Array.isArray(hipsT) && hipsT.length > idx) {
      if (!this._hipsT0) this._hipsT0 = hipsT[0].slice();
      const hips = vrm.humanoid.getNormalizedBoneNode('hips');
      if (hips && this._hipsRestPos) {
        // Root scale handles the X flip; do not negate translation again.
        let dx = (hipsT[idx][0] - this._hipsT0[0]);
        let dy =  hipsT[idx][1] - this._hipsT0[1];
        let dz =  hipsT[idx][2] - this._hipsT0[2];
        // v34c: ADD baked root-motion XZ correction. The bake walked
        // every frame and computed the XZ shift needed to keep the
        // planted foot anchored in world space. Without this, walks
        // look like treadmill walking (feet step but body stays
        // still) and feet slide on the floor between steps.
        if (this._rootBakeXZ) {
          const bi = idx - (this._rootBakeStart || 0);
          if (bi >= 0 && bi < this._rootBakeXZ.length) {
            dx += this._rootBakeXZ[bi].x;
            dz += this._rootBakeXZ[bi].z;
          }
        }
        // v17: in-place clips — discard the raw clip hip translation
        // but KEEP the foot-anchor bake. Earlier v15 zeroed both,
        // which made the planted foot slide as the leg swung in
        // place (treadmill-mocap clips like cmu_02_02_02 had a 52 cm
        // planted-foot slip ratio of 1.0 in browser per-frame QA).
        // We also centre the bake so the avatar stays at origin
        // across the loop rather than drifting with cumulative
        // stance-handoff error.
        if (this._killXZ) {
          dx = 0; dz = 0;
          if (this._rootBakeXZ && this._rootBakeXZ.length) {
            if (this._rootBakeMean == null) {
              let mx = 0, mz = 0;
              for (const p of this._rootBakeXZ) { mx += p.x; mz += p.z; }
              this._rootBakeMean = {
                x: mx / this._rootBakeXZ.length,
                z: mz / this._rootBakeXZ.length,
              };
            }
            const bi = idx - (this._rootBakeStart || 0);
            if (bi >= 0 && bi < this._rootBakeXZ.length) {
              dx = this._rootBakeXZ[bi].x - this._rootBakeMean.x;
              dz = this._rootBakeXZ[bi].z - this._rootBakeMean.z;
            }
          }
        }
        // v15: rehab-clip hip subsidence suppression. See _suppressHipBob
        // initialization above for the why. Clamp ±5 cm.
        if (this._suppressHipBob) {
          if (dy >  0.05) dy =  0.05;
          if (dy < -0.05) dy = -0.05;
        }
        // v34b: invisible boundary disc so locomotion clips (walks,
        // runs) actually translate across the floor instead of
        // gliding in place. The dancer's hip travels up to MAX_R
        // meters from frame-0 origin; beyond that we softly clamp
        // so she always stays in view of the fixed camera. Without
        // this, "show me a walk" looked like she was on an
        // invisible treadmill. AIST cross-stage ±2 m travel is
        // tone-mapped to this disc by simple proportional scaling.
        // v34c: with root bake live, real walks need more room —
        // bumped 1.5 → 3.5 m so a planted-foot bake on a 6-step
        // walk doesn't get clamped back into a tight circle.
        const MAX_R = 3.5;
        const r = Math.hypot(dx, dz);
        if (r > MAX_R) {
          const s = MAX_R / r;
          dx *= s; dz *= s;
        }
        // FLOOR-LOCK: cap vertical drift so the avatar can't levitate.
        // CMU clips in particular are captured with the subject at
        // varying global heights (stage, treadmill, mid-jump) and
        // using their absolute trans.y produces a hovering avatar
        // with feet well above the floor (user screenshots).
        //
        // v34d: was ±0.35 m which crushed real jumps (look-up table
        // says vertical CoM travel for a human hop is 0.4-0.6 m).
        // Bumped to ±0.7 m so hops/skips read as real jumps, with
        // a slightly tighter floor (-0.45 m) so squats can't punch
        // the avatar through the stage.
        const MAX_DOWN = 0.45, MAX_UP = 0.70;
        if (dy >  MAX_UP)   dy =  MAX_UP;
        if (dy < -MAX_DOWN) dy = -MAX_DOWN;
        const tx = this._hipsRestPos.x + dx;
        const ty = this._hipsRestPos.y + dy + (this._groundOffsetY || 0);
        const tz = this._hipsRestPos.z + dz;
        if (easeT < 1 && this._easeFromHipsPos) {
          hips.position.set(
            this._easeFromHipsPos.x + (tx - this._easeFromHipsPos.x) * easeS,
            this._easeFromHipsPos.y + (ty - this._easeFromHipsPos.y) * easeS,
            this._easeFromHipsPos.z + (tz - this._easeFromHipsPos.z) * easeS,
          );
        } else {
          hips.position.set(tx, ty, tz);
        }
      }
    }

    // ---- per-bone rotations -----------------------------------------
    for (const [srcBone, vrmBone] of Object.entries(VRM_BONE_MAP)) {
      const arr = rotations[srcBone];
      if (!arr || !arr[idx]) continue;
      const bone = vrm.humanoid.getNormalizedBoneNode(vrmBone);
      if (!bone) continue;

      // Body-part isolation: bones outside the active set get pinned
      // at a natural standing rest-pose so the user only sees the body
      // part being drilled. (Pinning to raw bind-pose would put the
      // arms in T-pose, which looks broken.)
      if (this._isolated && !this._isolated.has(vrmBone)) {
        this._setRestPoseBone(vrmBone, bone);
        this.prevBoneQuats.set(vrmBone, bone.quaternion.clone());
        continue;
      }

      let [x, y, z, w] = arr[idx];
      // Mirroring is handled by flipping root scale.x; per-bone quat
      // negation here used to distort the body.
      q.set(x, y, z, w);

      // Ease-in: SLERP from pre-play pose into clip quat.
      if (easeT < 1) {
        const from = this._easeFromQuats?.get(vrmBone);
        if (from) {
          const target = new THREE.Quaternion(x, y, z, w);
          q.copy(from).slerp(target, easeS);
        }
      }

      const prev = this.prevBoneQuats.get(vrmBone);
      if (prev) {
        // Hemisphere continuity
        const dot = prev.x*q.x + prev.y*q.y + prev.z*q.z + prev.w*q.w;
        if (dot < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
        // Angular-speed guard. Old behaviour: freeze entirely on a
        // single bad frame. New behaviour: slerp from prev toward
        // target at the maximum allowed angular speed so the motion
        // stays continuous instead of looking dead. We surface the
        // first violation via onflag() for telemetry but never stop
        // playback.
        if (easeT >= 1) {
          let d2 = prev.x*q.x + prev.y*q.y + prev.z*q.z + prev.w*q.w;
          d2 = Math.min(1, Math.max(-1, Math.abs(d2)));
          const ang = 2 * Math.acos(d2);
          if (ang / dt > maxAng) {
            const maxStep = maxAng * dt;
            const t = Math.min(1, maxStep / Math.max(1e-6, ang));
            q.copy(prev).slerp(new THREE.Quaternion(q.x, q.y, q.z, q.w), t);
            this._softFlag({ bone: vrmBone, ang_per_s: ang / dt,
                             frame: idx, format: 'vrm-quat' });
          }
        }
      }
      // Anatomic-limit clamp: cap rotation away from bind-pose at
      // BONE_MAX_ANGLE[vrmBone]. Kills the occasional spike frame
      // where mocap retargeting produced 200° shoulder rolls etc.
      // SKIPPED in pure mode — it clips deep torso bends / big arm
      // moves (up to 30 cm distortion) that Blender shows faithfully.
      if (!pure) this._clampAnatomic(vrmBone, q);
      // Temporal lowpass: only after ease-in is done, to avoid
      // double-smoothing the initial settle. SKIPPED in pure mode
      // (it lags fast moves and softens accents — Blender doesn't).
      if (!pure && easeT >= 1) this._lowpassQuat(prev, q);
      bone.quaternion.copy(q);
      this.prevBoneQuats.set(vrmBone, q.clone());
    }

    // ---- per-frame FOOT LOCK ----------------------------------------
    // After every bone has its rotation for this frame, force the
    // matrix to refresh and measure where the feet ACTUALLY are in
    // world space. If the lowest foot/toe is below the floor plane
    // (world Y = 0), lift the hip by exactly that amount so the foot
    // sits on the plane. Jumps (lowest foot > 0) are left alone so
    // the avatar can leave the floor. This is the "Blender-like"
    // contact behaviour the user expects — feet glued to the floor
    // when grounded, free to lift when jumping. Cost: one extra
    // updateMatrixWorld per frame (~0.1 ms on a desktop GPU).
    this._applyFootLock();
    // SELF-COLLISION stays ON even in pure mode: it ONLY pushes apart
    // overlapping non-adjacent limbs (hands through each other / arm
    // through torso) and was measured to change torso/legs 0 cm. Crossing
    // limbs are a top "looks like AI, not human" tell, so keep it. Toggle
    // off with window.__coachPhysBody=false.
    this._applyPhysicsBody();
    // TIER 1: in pure mode, stop here — the motion is played faithfully
    // (only grounded by lift-only foot-lock above, like Blender). The
    // pose-correcting solvers below are exactly what diverged us from
    // Blender, so they are OFF unless pure mode is explicitly disabled.
    if (!pure) {
      // WEIGHT SHIFT: sway the pelvis over the supporting foot while IK
      // keeps the planted feet on their danced spot.
      this._applyWeightShift();
      // Pin the planted foot's horizontal world position by bending the leg.
      this._applyFootWeight();
      // REAL-PHYSICS weight: body mass settles/overshoots under gravity.
      this._applyWeightSim();
    }
  }

  /** v34b: post-rotation per-frame foot lock. Reads the lowest foot
   *  world Y; if below 0, lifts hips so it kisses the floor. */
  _applyFootLock() {
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    // Suppressed during root-motion bake — the bake reads RAW foot
    // positions and would loop forever if foot-lock kept lifting hips.
    if (this._suppressFootLock) return;
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    if (!hips) return;
    // v155: FLOOR MODE grounding. For plank / push-up / sit-up the contact
    // point is the hands / forearms / knees / feet — NOT just the feet — so
    // we ground the LOWEST of a broad contact set to the floor and skip the
    // standing sole/hover logic entirely. This keeps the whole body resting
    // on the ground instead of hovering or being shoved by the foot-lock.
    if (this._floorMode) {
      vrm.scene.updateMatrixWorld(true);
      const probe = new THREE.Vector3();
      const wy = (name) => {
        const b = vrm.humanoid.getNormalizedBoneNode(name);
        if (!b) return Infinity;
        b.getWorldPosition(probe);
        return probe.y;
      };
      const contacts = ['leftHand', 'rightHand', 'leftLowerArm',
        'rightLowerArm', 'leftFoot', 'rightFoot', 'leftToes', 'rightToes',
        'leftLowerLeg', 'rightLowerLeg', 'leftUpperLeg', 'rightUpperLeg',
        'hips', 'spine', 'chest'];
      let lowest = Infinity;
      for (const c of contacts) { const y = wy(c); if (y < lowest) lowest = y; }
      if (isFinite(lowest)) {
        // The clip resets hips.y every frame, so a DIRECT correction here is
        // self-correcting and stable: shift the whole rig so the lowest
        // contact kisses the floor (small clearance so limbs don't clip
        // through). No airborne/hover heuristics — floor poses don't hop.
        const target = 0.03;               // m — contact clearance
        hips.position.y += (target - lowest);
      }
      return;
    }
    // v69: apply the PERSISTENT anti-hover offset BEFORE measuring, so
    // the accumulated drop actually sticks (the clip resets hips.y every
    // frame, so a per-frame nudge never accumulated — it must be a held
    // offset re-applied each frame and grown/decayed over time).
    if (this._hoverDropY) hips.position.y -= this._hoverDropY;
    vrm.scene.updateMatrixWorld(true);
    const probe = new THREE.Vector3();
    const wy = (name) => {
      const b = vrm.humanoid.getNormalizedBoneNode(name);
      if (!b) return Infinity;
      b.getWorldPosition(probe);
      return probe.y;
    };
    // v125: SOLE-ACCURATE grounding. A single "lowest foot bone" target
    // breaks when the foot isn't flat: on tiptoe the toe is the contact, on
    // a heel strike the ankle is. So we track the TOE and ANKLE bones
    // separately, each against its own rest-contact height (measured in
    // _measureContactY), and the sole's height above the floor is whichever
    // is lowest. This puts the visible shoe sole on the floor in every pose.
    const minToe = Math.min(wy('leftToes'), wy('rightToes'));
    const minAnkle = Math.min(wy('leftFoot'), wy('rightFoot'));
    const tR = (this._toeRestY != null) ? this._toeRestY : 0.045;
    const aR = (this._ankleRestY != null) ? this._ankleRestY : 0.105;
    const soleErr = Math.min(minToe - tR, minAnkle - aR);
    if (!isFinite(soleErr)) return;
    // v125: ROOT FIX for "foot swallowed into the floor" + "floats / never
    // lands". The visible shoe SOLE sits a few cm BELOW the foot bones;
    // grounding a bone to world Y=0 (old behaviour) shoved the sole under
    // the floor, and in pure mode the per-frame drop was skipped entirely so
    // faithful clips stayed floating. Now, EVERY frame regardless of pure
    // mode, we ground the actual sole:
    //   • lift IMMEDIATELY whenever the sole would dip below the floor, and
    //   • ground SUSTAINED floats, with a brief-lift grace so real steps and
    //     hops keep their air time.
    if (soleErr < -0.002) {
      // Sole would dip below the floor — push up so it just kisses it.
      hips.position.y += Math.min(1.0, -soleErr);
    } else if (soleErr > 0) {
      // Sole above the floor: a genuine step/hop OR a non-physical float. A
      // short airborne timer lets brief lifts keep air time while a clip that
      // hangs in the air gets pulled back down to the floor.
      const HOVER = 0.05;          // m above floor still treated as planted
      const HOVER_MAX = 0.30;      // s — longest brief lift before we ground it
      const FLOOR_BAND = 0.02;     // m — settle target for a steady float
      const _dt = this._lastDt || (1 / 60);
      // v126: rolling-window minimum to catch a STEADY float that the
      // airborne timer misses. A real step/hop PLANTS the foot (soleErr dips
      // to ~0) within a ~1.2 s window; a non-physical float (e.g. a stepping
      // clip whose lowest foot hovers ~6 cm but briefly dips to ~3 cm) never
      // plants, so its windowed minimum stays high. When the lowest foot has
      // not come near the floor for the whole window, gently pull the hips
      // down — this converges so the lowest foot kisses the floor without
      // touching clips that genuinely plant (their windowMin ≈ 0 → no pull).
      if (!this._soleHist) this._soleHist = [];
      this._soleHist.push(soleErr);
      const maxN = Math.max(8, Math.round(1.2 / _dt));
      if (this._soleHist.length > maxN) this._soleHist.shift();
      let windowMin = Infinity;
      for (let k = 0; k < this._soleHist.length; k++) {
        if (this._soleHist[k] < windowMin) windowMin = this._soleHist[k];
      }
      if (this._soleHist.length >= maxN && windowMin > FLOOR_BAND) {
        // Steady float — converge the held drop toward planting the lowest
        // foot (~1 s settle). Equilibrium leaves windowMin ≈ FLOOR_BAND.
        this._hoverDropY = (this._hoverDropY || 0) + windowMin * Math.min(1, _dt * 3);
        this._airborneT = 0;
      } else if (soleErr > HOVER) {
        this._airborneT = (this._airborneT || 0) + _dt;
        if (this._airborneT > HOVER_MAX) {
          // Grow the held drop toward grounding the sole (~0.2s settle).
          this._hoverDropY = (this._hoverDropY || 0) + soleErr * Math.min(1, _dt * 10);
        }
      } else {
        // Back near the floor — release the held drop smoothly.
        this._airborneT = 0;
        this._hoverDropY = Math.max(0, (this._hoverDropY || 0) - _dt * 1.5);
      }
    }
    // v198: UNIVERSAL FLOOR GUARD. The foot-lock above only grounds the FEET,
    // so a stretch / lying pose that dips the torso, head or hands toward the
    // ground used to clip THROUGH the floor plane ("why is she inside the
    // plane"). As a final pass, if ANY body contact point would sit below the
    // floor, lift the whole rig so the lowest part just rests on top. It is
    // self-correcting (the clip resets hips.y every frame) and a no-op for
    // normal upright poses (the feet are already the lowest point).
    try {
      vrm.scene.updateMatrixWorld(true);
      const _fp = new THREE.Vector3();
      const _fy = (name) => {
        const b = vrm.humanoid.getNormalizedBoneNode(name);
        if (!b) return Infinity;
        b.getWorldPosition(_fp);
        return _fp.y;
      };
      const _CONTACTS = ['head', 'leftHand', 'rightHand', 'leftLowerArm',
        'rightLowerArm', 'leftUpperArm', 'rightUpperArm', 'chest', 'spine',
        'hips', 'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg',
        'rightLowerLeg', 'leftFoot', 'rightFoot', 'leftToes', 'rightToes'];
      let _low = Infinity;
      for (let i = 0; i < _CONTACTS.length; i++) {
        const y = _fy(_CONTACTS[i]); if (y < _low) _low = y;
      }
      const _CLEAR = 0.03;   // m — keep the lowest body part just above the plane
      if (isFinite(_low) && _low < _CLEAR) hips.position.y += (_CLEAR - _low);
    } catch (e) { /* grounding is best-effort */ }
  }

  // v125: measure the per-VRM rest-contact heights of the TOE and ANKLE
  // bones — their world Y when the avatar stands in its rest pose with the
  // sole on the floor. The shoe sole is a few cm below these bones, so the
  // foot-lock grounds the sole (not a bone) by comparing each bone to its
  // own rest height. Measured ONCE, on the first clip load while the VRM is
  // still in its rest pose.
  _measureContactY() {
    if (this._toeRestY != null) return;
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    vrm.scene.updateMatrixWorld(true);
    const probe = new THREE.Vector3();
    const wy = (name) => {
      const b = vrm.humanoid.getNormalizedBoneNode(name);
      if (!b) return Infinity;
      b.getWorldPosition(probe);
      return probe.y;
    };
    const toe = Math.min(wy('leftToes'), wy('rightToes'));
    const ankle = Math.min(wy('leftFoot'), wy('rightFoot'));
    // Trust only plausible rest readings (bones a few cm above the floor).
    this._toeRestY = (isFinite(toe) && toe > 0.005 && toe < 0.25) ? toe : 0.045;
    this._ankleRestY = (isFinite(ankle) && ankle > 0.02 && ankle < 0.35) ? ankle : 0.105;
    try { console.log('[motion_player v125] toeRestY=', this._toeRestY,
      'ankleRestY=', this._ankleRestY); } catch (_) {}
  }

  // ===================================================================
  //  FOOT WEIGHT  — pins the planted foot so it stops sliding/skating
  // ===================================================================
  //  Vertical contact is handled by _applyFootLock. This handles the
  //  HORIZONTAL half: when a foot is on the floor it must hold its
  //  world XZ instead of skating while the leg swings. We detect the
  //  planted foot per frame, LATCH its world XZ on first contact, and
  //  bend ONLY that leg (knee + hip, clamped CCD) to keep the ankle on
  //  the latch. Releases when the foot lifts.
  //
  //  Why this is safe (unlike the reverted _footAnchor): it never
  //  touches the hips — it only rotates the two leg joints, so the
  //  correction is BOUNDED by leg reach and can never accumulate or
  //  fling the avatar across the floor. The latch is tiny per-side
  //  state that resets the instant the foot leaves the ground, not a
  //  running translation accumulator.
  //
  //  SHIPPED OFF BY DEFAULT (opt-in: window.__coachFootWeight===true).
  //  Browser measurement (3 dance clips, foot-weight ON vs OFF) showed
  //  it reduces AVERAGE planted-foot micro-slide (1.4→0.6 cm/frame) but
  //  INCREASES the worst single-frame pop at step transitions
  //  (16.6→32 cm) at every threshold tried — the leg-IK snaps when a
  //  foot swaps stance. Net-negative on these clips, so it stays opt-in
  //  until a soft-release / real contact solver replaces the hard latch.
  _applyFootWeight() {
    if (typeof window === 'undefined' || window.__coachFootWeight !== true) return;
    if (this._suppressFootLock) return;        // skip during root-motion bake
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    const H = vrm.humanoid;
    if (!this._fw) {
      const g = (n) => H.getNormalizedBoneNode(n);
      this._fw = {
        lUp: g('leftUpperLeg'),  lLo: g('leftLowerLeg'),  lFt: g('leftFoot'),
        rUp: g('rightUpperLeg'), rLo: g('rightLowerLeg'), rFt: g('rightFoot'),
        v:  Array.from({ length: 10 }, () => new THREE.Vector3()),
        sq: Array.from({ length: 6 },  () => new THREE.Quaternion()),
      };
    }
    const fw = this._fw;
    if (!fw.lFt || !fw.rFt || !fw.lLo || !fw.rLo) return;   // rig has no legs
    if (!this._footLatch) this._footLatch = { left: null, right: null };
    vrm.scene.updateMatrixWorld(true);
    const V = fw.v;
    const lP = fw.lFt.getWorldPosition(V[0]);
    const rP = fw.rFt.getWorldPosition(V[1]);
    // A foot counts as planted if it is within 5 cm of the LOWER foot
    // AND moving slowly. The velocity gate is critical: a fast SWING
    // foot can briefly dip near the floor during a dynamic move; without
    // the gate we'd latch it, then snap when it lifts away (a 30 cm pop).
    // Releasing the instant the foot speeds up keeps the release tiny.
    const minY = Math.min(lP.y, rP.y);
    const prm = (typeof window !== 'undefined' && window.__coachFootWeightParams) || {};
    const CONTACT = minY + (prm.band ?? 0.05);
    const prev = fw._prev || (fw._prev = { left: null, right: null });
    const V_PLANT = prm.vplant ?? 0.012;   // m/frame XZ (~0.36 m/s at 30 fps)
    const sides = [
      { name: 'left',  foot: lP, up: fw.lUp, lo: fw.lLo, ft: fw.lFt },
      { name: 'right', foot: rP, up: fw.rUp, lo: fw.rLo, ft: fw.rFt },
    ];
    for (const s of sides) {
      const pv = prev[s.name];
      const speed = pv ? Math.hypot(s.foot.x - pv.x, s.foot.z - pv.z) : 0;
      prev[s.name] = { x: s.foot.x, z: s.foot.z };
      const planted = s.foot.y < CONTACT && speed < V_PLANT;
      if (!planted) { this._footLatch[s.name] = null; continue; }
      let latch = this._footLatch[s.name];
      if (!latch) {
        latch = { x: s.foot.x, z: s.foot.z };   // first contact → latch XZ
        this._footLatch[s.name] = latch;
      }
      // Already within 1 cm of the latch? nothing to do.
      const dx = latch.x - s.foot.x, dz = latch.z - s.foot.z;
      if (dx * dx + dz * dz < 1e-4) continue;
      // Target keeps the current (foot-locked) Y, pins XZ to the latch.
      this._solveLegTo(s, latch.x, s.foot.y, latch.z);
    }
  }

  /** Bend one leg (knee + hip, clamped CCD) so the ankle reaches the
   *  target world position. Touches ONLY that leg's two joints; the
   *  hips and the rest of the body are never moved, so the correction
   *  is bounded by leg reach and can't fling the avatar. */
  _solveLegTo(s, tx, ty, tz) {
    const vrm = this.vrm, fw = this._fw;
    const V = fw.v, SQ = fw.sq, MAXSTEP = 0.20;
    const target = V[2].set(tx, ty, tz);
    // Clamp target to 98% of leg reach so the knee never hyperextends.
    const hipP = s.up.getWorldPosition(V[3]);
    const kneeP = s.lo.getWorldPosition(V[4]);
    const ankP0 = s.ft.getWorldPosition(V[5]);
    const reach = hipP.distanceTo(kneeP) + kneeP.distanceTo(ankP0);
    const toT = V[6].copy(target).sub(hipP);
    if (toT.length() > reach * 0.95) { toT.setLength(reach * 0.95); target.copy(hipP).add(toT); }
    const joints = [s.lo, s.up];               // distal → proximal
    for (let it = 0; it < 2; it++) {
      for (const j of joints) {
        const jp = j.getWorldPosition(V[7]);
        const ap = s.ft.getWorldPosition(V[8]);
        const cur = V[9].copy(ap).sub(jp);
        const des = V[3].copy(target).sub(jp);   // V[3] reused as scratch
        if (cur.lengthSq() < 1e-8 || des.lengthSq() < 1e-8) continue;
        cur.normalize(); des.normalize();
        const qd = SQ[0].setFromUnitVectors(cur, des);
        const ang = 2 * Math.acos(Math.min(1, Math.abs(qd.w)));
        if (ang > MAXSTEP && ang > 1e-6) { SQ[1].identity().slerp(qd, MAXSTEP / ang); qd.copy(SQ[1]); }
        const qw = j.getWorldQuaternion(SQ[2]);
        const qNew = SQ[3].multiplyQuaternions(qd, qw);
        const qp = j.parent ? j.parent.getWorldQuaternion(SQ[4]) : SQ[4].identity();
        const qLocal = SQ[5].copy(qp).invert().multiply(qNew);
        j.quaternion.copy(qLocal);
        vrm.scene.updateMatrixWorld(true);
      }
    }
  }

  // ===================================================================
  //  WEIGHT SHIFT  — pelvis sways over the supporting foot (Lever 1)
  // ===================================================================
  //  Why she reads as a "puppet marching in place": center-lock
  //  (_killXZ) bolts the pelvis to the floor centre so she can't wander,
  //  but a real dancer's pelvis travels OVER the foot that bears weight.
  //  Bolted-to-centre while the legs step = weightless marionette.
  //
  //  This sways the pelvis toward the supporting foot (the lower / more
  //  planted one), then IK-holds each planted foot on its DANCED spot so
  //  the foot does NOT slide — the pelvis moves over a foot that stays
  //  put, exactly like real weight transfer. The offset is a SMOOTHED
  //  pull toward a BOUNDED target (no accumulation), hard-capped, so she
  //  can never fly off (the failure mode of the old root-anchor).
  //  Validated clean across 6 styles → ON by default
  //  (set window.__coachWeightShift=false to disable).
  _applyWeightShift() {
    if (typeof window !== 'undefined' && window.__coachWeightShift === false) return;
    if (this._suppressFootLock) return;        // skip during root-motion bake
    if (!this._killXZ) return;                 // travel clips already shift
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    const H = vrm.humanoid;
    if (!this._fw) {
      const g = (n) => H.getNormalizedBoneNode(n);
      this._fw = {
        lUp: g('leftUpperLeg'),  lLo: g('leftLowerLeg'),  lFt: g('leftFoot'),
        rUp: g('rightUpperLeg'), rLo: g('rightLowerLeg'), rFt: g('rightFoot'),
        v:  Array.from({ length: 10 }, () => new THREE.Vector3()),
        sq: Array.from({ length: 6 },  () => new THREE.Quaternion()),
      };
    }
    const fw = this._fw;
    const hips = H.getNormalizedBoneNode('hips');
    if (!hips || !fw.lFt || !fw.rFt || !fw.lLo || !fw.rLo) return;
    if (!this._ws) this._ws = { off: { x: 0, z: 0 }, v: Array.from({ length: 4 }, () => new THREE.Vector3()) };
    const ws = this._ws, V = ws.v;
    vrm.scene.updateMatrixWorld(true);
    // Danced foot positions BEFORE we move the pelvis (these are the
    // spots the feet must keep — moving the hips would otherwise drag
    // them, so we IK them back below).
    const lp0 = fw.lFt.getWorldPosition(V[0]).clone();
    const rp0 = fw.rFt.getWorldPosition(V[1]).clone();
    // Only sway when she is STANDING on the floor. During jumps, deep
    // crouches and floor-work (breaking) the lowest foot is lifted / the
    // body is low — shifting + IK there causes float and knee-lock, so
    // skip and let the offset decay back to centre.
    const minFootY = Math.min(lp0.y, rp0.y);
    if (minFootY > 0.08) {
      this._ws.off.x *= 0.85; this._ws.off.z *= 0.85;
      if (Math.abs(this._ws.off.x) > 3e-4 || Math.abs(this._ws.off.z) > 3e-4) {
        hips.position.x += this._ws.off.x;
        hips.position.z += this._ws.off.z;
      }
      return;
    }
    // Support weights: the lower foot bears more weight (→ pelvis leans
    // toward it). Linear falloff over `band` above the lowest foot.
    const minY = Math.min(lp0.y, rp0.y);
    const band = 0.10;
    let wL = Math.max(0, 1 - Math.max(0, lp0.y - minY) / band);
    let wR = Math.max(0, 1 - Math.max(0, rp0.y - minY) / band);
    const sum = (wL + wR) || 1; wL /= sum; wR /= sum;
    // Target horizontal offset = a fraction of the way from the foot
    // midpoint toward the weighted support point. Bounded by half the
    // stance width, so a wide stance gives a bigger (but finite) sway.
    const GAIN = (window.__wsGain ?? 0.6);
    const midX = (lp0.x + rp0.x) / 2, midZ = (lp0.z + rp0.z) / 2;
    const supX = lp0.x * wL + rp0.x * wR, supZ = lp0.z * wL + rp0.z * wR;
    const targetX = GAIN * (supX - midX);
    const targetZ = GAIN * (supZ - midZ);
    // Smooth toward the (bounded) target, then hard-cap. Smoothing a
    // bounded target keeps the offset bounded — no runaway accumulation.
    const A = (window.__wsSmooth ?? 0.15);
    ws.off.x += (targetX - ws.off.x) * A;
    ws.off.z += (targetZ - ws.off.z) * A;
    const CAP = (window.__wsCap ?? 0.12);
    ws.off.x = Math.max(-CAP, Math.min(CAP, ws.off.x));
    ws.off.z = Math.max(-CAP, Math.min(CAP, ws.off.z));
    if (Math.abs(ws.off.x) < 3e-4 && Math.abs(ws.off.z) < 3e-4) return;
    // Move the pelvis...
    hips.position.x += ws.off.x;
    hips.position.z += ws.off.z;
    vrm.scene.updateMatrixWorld(true);
    // ...and IK each planted foot back to its danced spot so it does
    // NOT slide (the pelvis travels over a foot that stays put).
    const sides = [
      { name: 'left',  pre: lp0, up: fw.lUp, lo: fw.lLo, ft: fw.lFt, w: wL },
      { name: 'right', pre: rp0, up: fw.rUp, lo: fw.rLo, ft: fw.rFt, w: wR },
    ];
    for (const s of sides) {
      if (s.w < 0.5) continue;                 // only hold the loaded foot
      this._solveLegTo(s, s.pre.x, Math.max(0, s.pre.y), s.pre.z);
    }
    // The horizontal shift + IK can nudge the lowest foot off the floor;
    // re-ground vertically so feet stay planted (foot-lock ran BEFORE us).
    this._applyFootLock();
  }

  // ===================================================================
  //  REAL-PHYSICS WEIGHT  — body mass + gravity + ground reaction
  // ===================================================================
  //  This is the genuine physics layer (Rapier, the WASM engine games
  //  use). The recorded pose is kinematically faithful but weightless;
  //  as spring struts to the danced feet. Because it is a mass on
  //  springs it lags into fast moves, overshoots out of them, and
  //  sags/absorbs on foot impact — the cues the eye reads as WEIGHT.
  //
  //  Flow: read the danced hip + feet, step the sim, get the physics
  //  hip height, move the whole body to it, then IK each planted leg so
  //  its foot stays on the floor — so the KNEES visibly absorb instead
  //  of the foot sinking. Vertical only (world-Y → hip local-Y, no
  //  coordinate round-trip). Opt-in: window.__coachWeightSim===true.
  _applyWeightSim() {
    if (typeof window === 'undefined' || window.__coachWeightSim !== true) return;
    if (this._suppressFootLock) return;        // skip during root-motion bake
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    const H = vrm.humanoid;
    // Lazy-load the physics module (fire-and-forget; no-op until ready).
    if (!this._weightSim && !this._weightSimLoading && !this._weightSimFailed) {
      this._weightSimLoading = true;
      import('./physics_body')
        .then((m) => { this._weightSim = new m.WeightSim(); })
        .catch((e) => { this._weightSimFailed = true;
                        try { console.warn('[weightSim] load failed', e); } catch (_) {} });
    }
    const sim = this._weightSim;
    if (!sim) return;                          // still loading
    // Reuse the foot-weight leg refs / scratch (build if absent).
    if (!this._fw) {
      const g = (n) => H.getNormalizedBoneNode(n);
      this._fw = {
        lUp: g('leftUpperLeg'),  lLo: g('leftLowerLeg'),  lFt: g('leftFoot'),
        rUp: g('rightUpperLeg'), rLo: g('rightLowerLeg'), rFt: g('rightFoot'),
        v:  Array.from({ length: 10 }, () => new THREE.Vector3()),
        sq: Array.from({ length: 6 },  () => new THREE.Quaternion()),
      };
    }
    const fw = this._fw;
    if (!fw.lFt || !fw.rFt || !fw.lLo || !fw.rLo) return;   // rig has no legs
    const hips = H.getNormalizedBoneNode('hips');
    if (!hips) return;
    vrm.scene.updateMatrixWorld(true);
    // Clone so the scratch array is free for _solveLegTo below.
    const hipW = hips.getWorldPosition(fw.v[0]).clone();
    const lF   = fw.lFt.getWorldPosition(fw.v[1]).clone();
    const rF   = fw.rFt.getWorldPosition(fw.v[2]).clone();
    const minY = Math.min(lF.y, rF.y);
    const legs = [
      { footY: lF.y, L0: hipW.distanceTo(lF), planted: lF.y < minY + 0.06 },
      { footY: rF.y, L0: hipW.distanceTo(rF), planted: rF.y < minY + 0.06 },
    ];
    sim.ensureInit(hipW.y);
    if (this._weightSimNeedsReset) { sim.reset(hipW.y); this._weightSimNeedsReset = false; }
    const fps = this.data?.fps || 30;
    const physY = sim.step(1 / fps, hipW.y, legs);
    let dY = physY - hipW.y;
    dY = Math.max(-0.10, Math.min(0.10, dY));   // hard safety clamp
    if (Math.abs(dY) < 5e-4) return;
    // Move the whole body to the physics height...
    hips.position.y += dY;
    vrm.scene.updateMatrixWorld(true);
    // ...then keep each planted foot on the floor by bending its leg, so
    // the knee absorbs the settle instead of the foot punching the floor.
    const sides = [
      { name: 'left',  pre: lF, up: fw.lUp, lo: fw.lLo, ft: fw.lFt, planted: legs[0].planted },
      { name: 'right', pre: rF, up: fw.rUp, lo: fw.rLo, ft: fw.rFt, planted: legs[1].planted },
    ];
    for (const s of sides) {
      if (!s.planted) continue;
      this._solveLegTo(s, s.pre.x, Math.max(0, s.pre.y), s.pre.z);
    }
  }

  // ===================================================================
  //  SELF-COLLISION BODY  — gives limbs VOLUME so they stop overlapping
  // ===================================================================
  //  A puppet is infinitely-thin lines, so hands pass through each
  //  other / the torso (the "superimposed body parts" bug). We wrap a
  //  capsule around each arm segment + a few body capsules, find
  //  non-adjacent overlaps, and bend the offending arm away with a
  //  clamped 2-bone CCD. STATELESS per frame (re-derived from the clip
  //  every frame) so it can never accumulate drift or explode like the
  //  earlier torque/anchor attempts. Disable: window.__coachPhysBody=false.
  _applyPhysicsBody() {
    if (typeof window !== 'undefined' && window.__coachPhysBody === false) return;
    if (this._suppressFootLock) return;        // skip during root-motion bake
    const vrm = this.vrm;
    if (!vrm?.humanoid || !vrm.scene) return;
    const H = vrm.humanoid;
    if (!this._pb) {
      const g = (n) => H.getNormalizedBoneNode(n);
      this._pb = {
        lUp: g('leftUpperArm'),  lLo: g('leftLowerArm'),  lHa: g('leftHand'),
        rUp: g('rightUpperArm'), rLo: g('rightLowerArm'), rHa: g('rightHand'),
        chest: g('chest') || g('upperChest') || g('spine'),
        neck:  g('neck')  || g('head'),
        hips:  g('hips'),  spine: g('spine') || g('chest'),
        head:  g('head'),
        // scratch — allocated once, reused every frame (no GC churn)
        v:  Array.from({ length: 18 }, () => new THREE.Vector3()),
        cs: Array.from({ length: 3 },  () => new THREE.Vector3()),
        sv: Array.from({ length: 6 },  () => new THREE.Vector3()),
        sq: Array.from({ length: 6 },  () => new THREE.Quaternion()),
        pushL: new THREE.Vector3(), pushR: new THREE.Vector3(),
      };
    }
    const pb = this._pb;
    if (!pb.lHa || !pb.rHa || !pb.lLo || !pb.rLo) return;   // rig has no arms
    vrm.scene.updateMatrixWorld(true);
    const V = pb.v;

    // --- world endpoints --------------------------------------------
    const lLoP = pb.lLo.getWorldPosition(V[0]);
    const lHaP = pb.lHa.getWorldPosition(V[1]);
    const rLoP = pb.rLo.getWorldPosition(V[2]);
    const rHaP = pb.rHa.getWorldPosition(V[3]);
    // hand tips — extend past the wrist along the forearm direction so
    // the capsule covers the fingers, not just the wrist joint.
    const lTip = V[4].copy(lHaP).addScaledVector(
      V[8].copy(lHaP).sub(lLoP).normalize(), 0.10);
    const rTip = V[5].copy(rHaP).addScaledVector(
      V[9].copy(rHaP).sub(rLoP).normalize(), 0.10);
    // body capsules (torso / belly / head)
    const chestP = pb.chest ? pb.chest.getWorldPosition(V[6])  : null;
    const neckP  = pb.neck  ? pb.neck.getWorldPosition(V[7])   : null;
    const hipsP  = pb.hips  ? pb.hips.getWorldPosition(V[10])  : null;
    const spineP = pb.spine ? pb.spine.getWorldPosition(V[11]) : null;
    const headP  = pb.head  ? pb.head.getWorldPosition(V[12])  : null;
    const headTip = headP ? V[13].copy(headP).add(V[14].set(0, 0.12, 0)) : null;

    const armSegs = [
      { a: lLoP, b: lHaP, r: 0.050, side: 'left'  },
      { a: lHaP, b: lTip, r: 0.055, side: 'left'  },
      { a: rLoP, b: rHaP, r: 0.050, side: 'right' },
      { a: rHaP, b: rTip, r: 0.055, side: 'right' },
    ];
    const bodySegs = [];
    if (chestP && neckP)  bodySegs.push({ a: chestP, b: neckP,   r: 0.135 });
    if (hipsP && spineP)  bodySegs.push({ a: hipsP,  b: spineP,  r: 0.130 });
    if (headP && headTip) bodySegs.push({ a: headP,  b: headTip, r: 0.090 });

    pb.pushL.set(0, 0, 0);
    pb.pushR.set(0, 0, 0);
    const c1 = V[15], c2 = V[16], nrm = V[17];
    let any = false;
    // SLOP: anti-chatter deadband. Only resolve overlaps DEEPER than this
    // so a limb resting right at the capsule surface doesn't get pushed
    // out one frame and pulled back the next (that toggling is what made
    // the hands/arms visibly vibrate).
    const SLOP = 0.012;   // 1.2 cm
    const consider = (segA, segB, otherIsBody) => {
      const dist = this._closestSegSeg(segA.a, segA.b, segB.a, segB.b, c1, c2);
      const rr = segA.r + segB.r;
      if (dist >= rr - SLOP || dist < 1e-6) return;
      const pen = rr - dist;
      nrm.copy(c1).sub(c2).divideScalar(dist);   // unit push: A away from B
      if (otherIsBody) {
        (segA.side === 'left' ? pb.pushL : pb.pushR).addScaledVector(nrm, pen);
      } else {
        const half = pen * 0.5;
        (segA.side === 'left' ? pb.pushL : pb.pushR).addScaledVector(nrm,  half);
        (segB.side === 'left' ? pb.pushL : pb.pushR).addScaledVector(nrm, -half);
      }
      any = true;
    };

    // arm vs OPPOSITE-side arm (each unordered pair once: left drives)
    for (const A of armSegs) {
      if (A.side !== 'left') continue;
      for (const B of armSegs) { if (B.side !== 'right') continue; consider(A, B, false); }
    }
    // arm vs body (torso / belly / head)
    for (const A of armSegs) for (const B of bodySegs) consider(A, B, true);

    // Anti-chatter correction. The previous code applied the raw per-frame
    // push at full strength and skipped entirely when no overlap was found,
    // so the correction snapped fully ON (collision frame) then fully OFF
    // (clip pulls the limbs back together) — that on/off toggle at the
    // capsule boundary is what made the hands/arms shake. Instead, ramp the
    // push with an EMA and let it DECAY back to zero when the overlap clears
    // so the limb eases out of the body instead of vibrating against it.
    const CAP = 0.12;   // max push per frame (m) — small so it can't snap
    if (!pb.pushLs) { pb.pushLs = new THREE.Vector3(); pb.pushRs = new THREE.Vector3(); }
    if (pb.pushL.length() > CAP) pb.pushL.setLength(CAP);
    if (pb.pushR.length() > CAP) pb.pushR.setLength(CAP);
    const SMOOTH = 0.35;   // 0..1 — lower = smoother but laggier correction
    pb.pushLs.lerp(pb.pushL, SMOOTH);
    pb.pushRs.lerp(pb.pushR, SMOOTH);
    const ACT = 1e-4;      // stop solving once the ramped push has decayed
    if (pb.pushLs.lengthSq() > ACT) this._solveArmTo('left', lHaP, pb.pushLs);
    if (pb.pushRs.lengthSq() > ACT) this._solveArmTo('right', rHaP, pb.pushRs);
  }

  /** Bend one arm (2-bone clamped CCD over elbow + shoulder) so the
   *  wrist moves by `push`. Only touches that arm's two joints — it
   *  can never distort the torso or legs. */
  _solveArmTo(side, wristWorld, push) {
    const pb = this._pb, vrm = this.vrm;
    const lower = (side === 'left') ? pb.lLo : pb.rLo;
    const upper = (side === 'left') ? pb.lUp : pb.rUp;
    const wrist = (side === 'left') ? pb.lHa : pb.rHa;
    if (!lower || !upper || !wrist) return;
    const SV = pb.sv, SQ = pb.sq, MAXSTEP = 0.25;
    const target = SV[0].copy(wristWorld).add(push);
    const joints = [lower, upper];           // distal -> proximal
    for (let it = 0; it < 2; it++) {
      for (const j of joints) {
        const jp = j.getWorldPosition(SV[1]);
        const wp = wrist.getWorldPosition(SV[2]);
        const cur = SV[3].copy(wp).sub(jp);
        const des = SV[4].copy(target).sub(jp);
        if (cur.lengthSq() < 1e-8 || des.lengthSq() < 1e-8) continue;
        cur.normalize(); des.normalize();
        const qd = SQ[0].setFromUnitVectors(cur, des);
        const ang = 2 * Math.acos(Math.min(1, Math.abs(qd.w)));
        if (ang > MAXSTEP && ang > 1e-6) {     // clamp the step
          SQ[1].identity().slerp(qd, MAXSTEP / ang);
          qd.copy(SQ[1]);
        }
        const qw = j.getWorldQuaternion(SQ[2]);
        const qNew = SQ[3].multiplyQuaternions(qd, qw);
        const qp = j.parent ? j.parent.getWorldQuaternion(SQ[4]) : SQ[4].identity();
        const qLocal = SQ[5].copy(qp).invert().multiply(qNew);
        j.quaternion.copy(qLocal);
        vrm.scene.updateMatrixWorld(true);
      }
    }
  }

  /** Closest distance between two segments p1q1 and p2q2; writes the
   *  closest points into c1/c2. Ericson, Real-Time Collision Detection. */
  _closestSegSeg(p1, q1, p2, q2, c1, c2) {
    const S = this._pb.cs;
    const d1 = S[0].copy(q1).sub(p1);
    const d2 = S[1].copy(q2).sub(p2);
    const r  = S[2].copy(p1).sub(p2);
    const a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
    const EPS = 1e-9;
    let s, t;
    if (a <= EPS && e <= EPS) { s = 0; t = 0; }
    else if (a <= EPS) { s = 0; t = Math.min(1, Math.max(0, f / e)); }
    else {
      const c = d1.dot(r);
      if (e <= EPS) { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
      else {
        const b = d1.dot(d2);
        const denom = a * e - b * b;
        s = denom > EPS ? Math.min(1, Math.max(0, (b * f - c * e) / denom)) : 0;
        t = (b * s + f) / e;
        if (t < 0)      { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
        else if (t > 1) { t = 1; s = Math.min(1, Math.max(0, (b - c) / a)); }
      }
    }
    c1.copy(p1).addScaledVector(d1, s);
    c2.copy(p2).addScaledVector(d2, t);
    return c1.distanceTo(c2);
  }

  // -- Format B: raw SMPL axis-angle ---------------------------------
  _hardLimit(j) {
    const k = this.limits.hard_limit_rad?.[String(j)];
    return (typeof k === 'number') ? k : (this.limits.default_hard || 3.14);
  }

  _applyFrameAA(idx) {
    const vrm = this.vrm;
    if (!vrm?.humanoid) return;
    const poses = this.data.poses;
    const trans = this.data.trans;
    if (!poses || !trans) return;
    const base = idx * 24 * 3;
    const dt = 1 / (this.data.fps || 30);
    const maxAng = this.limits.max_ang_speed || 15.0;
    const pure = (typeof window === 'undefined') || window.__coachPureMode !== false;
    this._pure = pure;

    if (!this._trans0) this._trans0 = [trans[0], trans[1], trans[2]];
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    if (hips) {
      const tx = trans[idx*3 + 0] - this._trans0[0];
      let   ty = trans[idx*3 + 1] - this._trans0[1];
      const tz = trans[idx*3 + 2] - this._trans0[2];
      // Floor-lock (mirrors the quat-path clamp).
      if (ty >  0.35) ty =  0.35;
      if (ty < -0.35) ty = -0.35;
      // v15: same in-place + hip-bob suppression as quat path.
      let _tx = tx, _tz = tz, _ty = ty;
      if (this._killXZ) { _tx = 0; _tz = 0; }
      if (this._suppressHipBob) {
        if (_ty >  0.05) _ty =  0.05;
        if (_ty < -0.05) _ty = -0.05;
      }
      hips.position.set(
        this._hipsRestPos.x + _tx,
        this._hipsRestPos.y + _ty + (this._groundOffsetY || 0),
        this._hipsRestPos.z + _tz,
      );
    }

    const q = new THREE.Quaternion();
    const axis = new THREE.Vector3();
    for (let j = 0; j < 24; j++) {
      const boneName = SMPL_TO_VRM[j];
      if (!boneName) continue;
      const bone = vrm.humanoid.getNormalizedBoneNode(boneName);
      if (!bone) continue;

      // Body-part isolation (SMPL-AA path) — same natural rest-pose
      // logic as the quat path so legs/torso don't snap to T-pose
      // when only the arms are being drilled.
      if (this._isolated && !this._isolated.has(boneName)) {
        this._setRestPoseBone(boneName, bone);
        continue;
      }

      let ax = poses[base + j*3 + 0];
      let ay = poses[base + j*3 + 1];
      let az = poses[base + j*3 + 2];
      let mag = Math.hypot(ax, ay, az);
      const lim = this._hardLimit(j);
      if (mag > lim) { const s = lim / mag; ax *= s; ay *= s; az *= s; mag = lim; }
      // Mirroring is handled by flipping root scale.x.

      if (mag < 1e-8) q.set(0, 0, 0, 1);
      else { axis.set(ax/mag, ay/mag, az/mag); q.setFromAxisAngle(axis, mag); }

      const prev = this.prevBoneQuats.get(boneName);
      if (prev) {
        let dot = prev.x*q.x + prev.y*q.y + prev.z*q.z + prev.w*q.w;
        if (dot < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
        let d2 = prev.x*q.x + prev.y*q.y + prev.z*q.z + prev.w*q.w;
        d2 = Math.min(1, Math.max(-1, Math.abs(d2)));
        const ang = 2 * Math.acos(d2);
        if (ang / dt > maxAng) {
          const maxStep = maxAng * dt;
          const t = Math.min(1, maxStep / Math.max(1e-6, ang));
          q.copy(prev).slerp(new THREE.Quaternion(q.x, q.y, q.z, q.w), t);
          this._softFlag({ joint: j, bone: boneName, ang_per_s: ang/dt,
                           frame: idx, format: 'smpl-aa' });
        }
      }
      // Anatomic-limit clamp + temporal lowpass (skipped in pure mode).
      if (!pure) { this._clampAnatomic(boneName, q); this._lowpassQuat(prev, q); }
      bone.quaternion.copy(q);
      this.prevBoneQuats.set(boneName, q.clone());
    }

    // Same per-frame foot lock as the quat path so AA fallback also
    // glues feet to the floor when grounded.
    this._applyFootLock();
    // Self-collision stays on in pure mode (see quat path note).
    this._applyPhysicsBody();
    if (!pure) {
      this._applyWeightShift();
      this._applyFootWeight();
      this._applyWeightSim();
    }
  }

  _raiseFlag(info) {
    this.flagged = true;
    this.playing = false;
    if (this.onflag) this.onflag(info);
  }

  // Soft flag: motion violated angular-speed budget but we clamped it
  // and kept playing. Fire onflag at most once per play() so telemetry
  // still gets a heads-up without spamming the UI or stopping motion.
  _softFlag(info) {
    if (this._softFlagFired) return;
    this._softFlagFired = true;
    if (this.onflag) this.onflag({ ...info, soft: true });
  }
}
