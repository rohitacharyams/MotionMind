"""pd_bake.py — offline MuJoCo PD-tracking physics bake.

Drives the physical humanoid (built by ``humanoid_builder``) to TRACK a
reference warmup/stretch clip under gravity + contact + self-collision,
then records the physically-valid result back into our motion-JSON.

Why this fixes the bugs (no RL needed for slow warmups):
  * self-collision ON  → a hand cannot pass through the head/torso
  * foot contact + friction ON → planted feet stop sliding
  * gravity ON → limbs carry weight (no more "dangling")
The root is PD-held toward the reference trajectory (stiff) so the body
never topples — safe + stable for slow motion. Fast/aerial dance will
need an RL controller (Phase 2, on a CUDA box); this module is Phase 1.

Mapping (clean because rest_local_rotation is identity for all bones):
  child ball-joint target  = clip rotations[B][f]            (VRM-local)
  root world orientation   = R_GLOBAL ⊗ rotations['Hips'][f] (Y-up→Z-up)
  root world position      = R_GLOBAL · hips_translation[f]
Output reverses it: VRM-local[B] = ball qpos; hips = R_GLOBAL⁻¹ applied.

IMPORTANT (Jun 2026): this module now consumes the HIPS-CORRECTED clip
JSON (coach/motion_cache_cmu/*.json after _fix_hips.py). Those clips are
true VRM Y-up and stand upright in three-vrm. The OLD code set R_GLOBAL=
identity claiming the clip was "already Z-up" — that was an artefact of the
broken JSON which baked a ~120° coord rotation into the Hips quaternion.
With the corrected JSON we apply the real Y-up→Z-up rotation here.

Quaternion convention: clip JSON is xyzw; MuJoCo qpos is wxyz.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from coach.physics.humanoid_builder import (  # noqa: E402
    load_skeleton, build_humanoid_xml, PARENT, ORDER)

# Corrected clip data is true VRM Y-up (head at high +Y, feet at low +Y
# — verified by FK: head-minus-feet ≈ [0, +1.15, 0]). MuJoCo is Z-up with
# gravity along −Z, so we rotate the whole body +90° about X (Y→Z) at the
# root. Applied at the freejoint, every child inherits it, so child ball-
# joint targets remain exactly the clip's VRM-local quaternions.
# wxyz quaternion for R_x(+90°) = [cos45, sin45, 0, 0].
R_GLOBAL = np.array([0.70710678, 0.70710678, 0.0, 0.0])   # Y-up -> Z-up


# ── quaternion helpers (all wxyz unless _xyzw) ──────────────────────
def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_conj(a: np.ndarray) -> np.ndarray:
    return np.array([a[0], -a[1], -a[2], -a[3]])


def q_norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def xyzw_to_wxyz(q: List[float]) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def wxyz_to_xyzw(q: np.ndarray) -> List[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return q_mul(q_mul(q, qv), q_conj(q))[1:]


def q_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) → rotation vector (axis*angle), small-angle safe."""
    q = q_norm(q)
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-9 or ang < 1e-9:
        return np.zeros(3)
    axis = q[1:] / s
    return axis * ang


# ── reference clip ──────────────────────────────────────────────────
class Reference:
    def __init__(self, clip_path: str):
        with open(clip_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        self.path = clip_path
        self.fps = float(d.get('fps') or 30.0)
        self.n_frames = int(d.get('n_frames') or 0)
        self.rotations = d['rotations']               # bone -> [n][xyzw]
        self.hips_translation = d['hips_translation']  # [n][xyz]
        self.skeleton = {k: [float(x) for x in v]
                         for k, v in d['rest_local_translation'].items()}
        self.duration_s = float(d.get('duration_s')
                                or self.n_frames / self.fps)
        self._src = d

    def root_pose(self, f: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (root_pos_z_up, root_quat_wxyz_z_up) for frame f."""
        hip_t = np.array(self.hips_translation[f], dtype=float)
        pos = q_rotate(R_GLOBAL, hip_t)
        ref_hips = xyzw_to_wxyz(self.rotations['Hips'][f])
        quat = q_norm(q_mul(R_GLOBAL, ref_hips))
        return pos, quat

    def joint_quat(self, bone: str, f: int) -> np.ndarray:
        return q_norm(xyzw_to_wxyz(self.rotations[bone][f]))


# ── model wrapper ───────────────────────────────────────────────────
class PhysHumanoid:
    def __init__(self, ref: Reference, total_mass: float = 70.0):
        import mujoco
        self.mj = mujoco
        xml, joint_order = build_humanoid_xml(ref.skeleton, total_mass)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.joint_order = joint_order            # ball joints, in order
        self.ref = ref
        # name → joint qpos/qvel/dof addresses
        self.jnt_qpos: Dict[str, int] = {}
        self.jnt_dof: Dict[str, int] = {}
        for b in joint_order:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, b)
            self.jnt_qpos[b] = int(self.model.jnt_qposadr[jid])
            self.jnt_dof[b] = int(self.model.jnt_dofadr[jid])
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root')
        self.root_qpos = int(self.model.jnt_qposadr[rid])
        self.root_dof = int(self.model.jnt_dofadr[rid])
        self.floor_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.sim_dt = float(self.model.opt.timestep)
        # z-lift so the feet rest on the floor. Measure the lowest foot
        # at frame 0 with no lift, then lift so it sits ~2 cm above z=0.
        self.z_lift = 0.0
        self.set_reference_pose(0)
        feet = self.foot_positions()
        min_z = min(p[2] for p in feet.values())
        self.z_lift = float(0.02 - min_z)

    def set_reference_pose(self, f: int, zero_vel: bool = True) -> None:
        d = self.data
        pos, quat = self.ref.root_pose(f)
        pos = pos.copy()
        pos[2] += self.z_lift
        d.qpos[self.root_qpos:self.root_qpos + 3] = pos
        d.qpos[self.root_qpos + 3:self.root_qpos + 7] = quat
        for b in self.joint_order:
            a = self.jnt_qpos[b]
            d.qpos[a:a + 4] = self.ref.joint_quat(b, f)
        if zero_vel:
            d.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    # -- metrics ------------------------------------------------------
    def self_penetration(self) -> Tuple[int, float]:
        """(# self-collision contacts, max penetration depth m).
        Excludes any contact involving the floor."""
        d = self.data
        n = 0
        max_depth = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor_gid or c.geom2 == self.floor_gid:
                continue
            if c.dist < 0:
                n += 1
                max_depth = max(max_depth, -float(c.dist))
        return n, max_depth

    def foot_positions(self) -> Dict[str, np.ndarray]:
        out = {}
        for foot in ('LeftFoot', 'RightFoot'):
            bid = self.mj.mj_name2id(
                self.model, self.mj.mjtObj.mjOBJ_BODY, foot)
            out[foot] = np.array(self.data.xpos[bid], dtype=float)
        return out

    def body_world_pos(self, name: str) -> np.ndarray:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        return np.array(self.data.xpos[bid], dtype=float)


# ── reference forward kinematics (in VRM space) ─────────────────────
def vrm_fk(ref: Reference, f: int) -> Dict[str, np.ndarray]:
    """World position of every bone in VRM Y-up space, for frame f.
    Used only by the correctness gate."""
    world_pos: Dict[str, np.ndarray] = {}
    world_rot: Dict[str, np.ndarray] = {}
    # process parents before children (ORDER is depth-first)
    for b in ORDER:
        off = np.array(ref.skeleton.get(b, [0, 0, 0]), dtype=float)
        q_local = ref.joint_quat(b, f)
        p = PARENT[b]
        if p is None:
            world_pos[b] = np.array(ref.hips_translation[f], dtype=float)
            world_rot[b] = q_local
        else:
            world_pos[b] = world_pos[p] + q_rotate(world_rot[p], off)
            world_rot[b] = q_norm(q_mul(world_rot[p], q_local))
    return world_pos


def validate_fk(ref: Reference, frames: Optional[List[int]] = None
                ) -> Dict[str, float]:
    """Correctness gate: does the MuJoCo reference pose match our VRM FK?
    Compares body world positions (after R_GLOBAL) at several frames.
    Returns {'max_err_m', 'mean_err_m'}. Small (<2 cm) ⇒ mapping correct.
    """
    h = PhysHumanoid(ref)
    if frames is None:
        frames = list(range(0, ref.n_frames,
                            max(1, ref.n_frames // 8)))[:8] or [0]
    max_err = 0.0
    errs: List[float] = []
    for f in frames:
        h.set_reference_pose(f)
        vfk = vrm_fk(ref, f)
        for b in ORDER:
            mj_p = h.body_world_pos(b)
            exp_p = q_rotate(R_GLOBAL, vfk[b]).copy()
            exp_p[2] += h.z_lift          # MuJoCo pose is lifted to floor
            e = float(np.linalg.norm(mj_p - exp_p))
            errs.append(e)
            max_err = max(max_err, e)
    return {'max_err_m': max_err,
            'mean_err_m': float(np.mean(errs)) if errs else 0.0,
            'frames': len(frames)}


# ── PD-tracking bake ────────────────────────────────────────────────
# Gains. Root is stiff (won't topple); limbs moderate (gravity sag +
# contact response still read through). Tuned for stability on CPU
# implicitfast @ 200 Hz.
KP_JOINT = 80.0
KD_JOINT = 6.0
KP_ROOT_POS = 1200.0
KD_ROOT_POS = 120.0
KP_ROOT_ROT = 400.0
KD_ROOT_ROT = 40.0


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = q_norm(a); b = q_norm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b = -b; d = -d
    if d > 0.9995:
        return q_norm(a + t * (b - a))
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * a + (np.sin(t * th) / s) * b


def _pd_control(h: PhysHumanoid, root_pos_t: np.ndarray,
                root_quat_t: np.ndarray,
                joint_t: Dict[str, np.ndarray]) -> None:
    d = h.data
    # Computed-torque control: start from qfrc_bias (gravity + Coriolis +
    # centrifugal) so those are exactly cancelled. Without this, the PD
    # joints behave like soft springs and the upper body SAGS under its
    # own weight (measured: head collapsed 50 cm on a slow warmup). With
    # bias compensation the body holds the reference pose, while contact
    # forces (foot/floor, self-collision) are NOT in qfrc_bias and so
    # still read through and correct the pose. PD then only fights
    # tracking error, not gravity.
    d.qfrc_applied[:] = d.qfrc_bias
    # root translation PD (world frame)
    rp = h.root_dof
    pos = d.qpos[h.root_qpos:h.root_qpos + 3]
    linvel = d.qvel[rp:rp + 3]
    d.qfrc_applied[rp:rp + 3] += (KP_ROOT_POS * (root_pos_t - pos)
                                  - KD_ROOT_POS * linvel)
    # root orientation PD (error in body frame)
    qc = d.qpos[h.root_qpos + 3:h.root_qpos + 7].copy()
    err = q_to_rotvec(q_mul(q_conj(qc), root_quat_t))
    angvel = d.qvel[rp + 3:rp + 6]
    d.qfrc_applied[rp + 3:rp + 6] += (KP_ROOT_ROT * err
                                      - KD_ROOT_ROT * angvel)
    # joints
    for b in h.joint_order:
        a = h.jnt_qpos[b]
        dof = h.jnt_dof[b]
        qc = d.qpos[a:a + 4].copy()
        err = q_to_rotvec(q_mul(q_conj(qc), joint_t[b]))
        omega = d.qvel[dof:dof + 3]
        d.qfrc_applied[dof:dof + 3] += KP_JOINT * err - KD_JOINT * omega


def bake_clip(clip_path: str, out_path: Optional[str] = None,
              total_mass: float = 70.0, verbose: bool = True,
              no_deslide: bool = False
              ) -> Dict[str, object]:
    """Run the PD-tracking bake on one clip. Returns a metrics dict and
    (if out_path) writes the physically-valid *_phys.json.

    no_deslide: when True, SKIP the planted-foot de-slide post-process and
    emit the raw physics root trajectory. Slide is then handled at runtime
    by the player's foot-IK (validated ~1 cm mean). The baked de-slide latch
    can mis-anchor and *add* slide on some clips, so for those it is better
    to let foot-IK do it. Joint angles (the physics pose) are identical
    either way.
    """
    ref = Reference(clip_path)
    h = PhysHumanoid(ref, total_mass)
    mj = h.mj
    sub = max(1, int(round((1.0 / ref.fps) / h.sim_dt)))

    # --- before metrics: penetration in the RAW reference ---
    raw_pen_frames = 0
    raw_pen_max = 0.0
    for f in range(ref.n_frames):
        h.set_reference_pose(f)
        n, dep = h.self_penetration()
        if n > 0:
            raw_pen_frames += 1
            raw_pen_max = max(raw_pen_max, dep)

    # --- run the tracked sim ---
    h.set_reference_pose(0)
    out_rot: Dict[str, List[List[float]]] = {b: [] for b in ref.rotations}
    # World-space root trajectory (MuJoCo Z-up). We defer the VRM
    # conversion until AFTER the de-slide correction so every coordinate
    # operation happens in ONE frame (world), eliminating axis-mix bugs.
    root_pos_world: List[np.ndarray] = []
    root_quat_world: List[np.ndarray] = []
    track_err_deg: List[float] = []
    post_pen_frames = 0
    post_pen_max = 0.0
    foot_slip_total = 0.0
    prev_planted: Dict[str, Optional[np.ndarray]] = {
        'LeftFoot': None, 'RightFoot': None}
    # Record per-frame foot world positions for the planted-foot
    # de-slide post-process (see below).
    feetL_rec: List[np.ndarray] = []
    feetR_rec: List[np.ndarray] = []

    for f in range(ref.n_frames):
        f_next = min(ref.n_frames - 1, f + 1)
        rp0, rq0 = ref.root_pose(f)
        rp1, rq1 = ref.root_pose(f_next)
        rp0 = rp0.copy(); rp0[2] += h.z_lift
        rp1 = rp1.copy(); rp1[2] += h.z_lift
        jt0 = {b: ref.joint_quat(b, f) for b in h.joint_order}
        jt1 = {b: ref.joint_quat(b, f_next) for b in h.joint_order}
        for s in range(sub):
            t = (s + 1) / sub
            rpt = rp0 + (rp1 - rp0) * t
            rqt = _slerp(rq0, rq1, t)
            jtt = {b: _slerp(jt0[b], jt1[b], t) for b in h.joint_order}
            _pd_control(h, rpt, rqt, jtt)
            mj.mj_step(h.model, h.data)
        if not np.all(np.isfinite(h.data.qpos)):
            return {'ok': False, 'reason': 'sim diverged (NaN)',
                    'frame': f, 'clip': os.path.basename(clip_path)}

        # record physically-valid pose for this frame
        mj.mj_forward(h.model, h.data)
        rqmj = h.data.qpos[h.root_qpos + 3:h.root_qpos + 7].copy()
        rpmj = h.data.qpos[h.root_qpos:h.root_qpos + 3].copy()
        # Defer VRM conversion: keep the root in WORLD space for now so
        # the de-slide correction (also world) composes cleanly.
        root_pos_world.append(rpmj)
        root_quat_world.append(rqmj)
        for b in h.joint_order:
            a = h.jnt_qpos[b]
            out_rot[b].append(wxyz_to_xyzw(
                q_norm(h.data.qpos[a:a + 4].copy())))

        # tracking error (deg) vs reference joint targets
        errs = []
        for b in h.joint_order:
            a = h.jnt_qpos[b]
            e = q_to_rotvec(q_mul(q_conj(h.data.qpos[a:a + 4]),
                                  ref.joint_quat(b, f)))
            errs.append(np.degrees(np.linalg.norm(e)))
        track_err_deg.append(float(np.mean(errs)) if errs else 0.0)

        # post penetration
        n, dep = h.self_penetration()
        if n > 0:
            post_pen_frames += 1
            post_pen_max = max(post_pen_max, dep)

        # foot slip: horizontal motion of a foot while it's near floor
        feet = h.foot_positions()
        feetL_rec.append(feet['LeftFoot'].copy())
        feetR_rec.append(feet['RightFoot'].copy())
        for foot, p in feet.items():
            if p[2] < 0.06:           # in contact band
                pv = prev_planted[foot]
                if pv is not None:
                    foot_slip_total += float(np.linalg.norm(p[:2] - pv[:2]))
                prev_planted[foot] = p
            else:
                prev_planted[foot] = None

    # ── PLANTED-FOOT DE-SLIDE (the real slide fix) ───────────────────
    # The stiff root-position PD makes the sim track the ORIGINAL clip's
    # root trajectory — which, if the source mocap drifted, drags the
    # body and forces planted feet to slide. We cancel that here: detect
    # the planted foot each frame and counter-translate the recorded
    # root (out_hips XZ) so that foot holds a fixed world position. Pure
    # root translation — joint angles (the physics pose) are untouched.
    # Result: slide baked OUT of the output, zero runtime cost. Travel
    # is preserved because the anchor re-latches when the foot lifts and
    # re-plants at a new spot.
    fL = np.array(feetL_rec)            # (n,3) world, z-up
    fR = np.array(feetR_rec)
    band = float(min(fL[:, 2].min(), fR[:, 2].min())) + 0.06
    corr = np.zeros(2)                  # world XY correction on the root
    planted: Optional[str] = None
    ref_xy: Optional[np.ndarray] = None
    deslid_slip = 0.0
    prev_corr_foot: Optional[np.ndarray] = None
    for f in range(ref.n_frames):
        if no_deslide:
            break
        lz, rz = fL[f, 2], fR[f, 2]
        lon, ron = lz < band, rz < band
        pick = None
        if lon and ron:
            pick = 'L' if lz <= rz else 'R'
        elif lon:
            pick = 'L'
        elif ron:
            pick = 'R'
        if pick is None:
            planted = None
            prev_corr_foot = None
        else:
            cur = (fL[f, :2] if pick == 'L' else fR[f, :2])
            swapped = (planted != pick)
            if swapped:
                # (re)latch: hold the foot at its current corrected spot
                planted = pick
                ref_xy = cur + corr
            corr = ref_xy - cur
            # generous safety cap (treadmill/in-place clips need large
            # corrections to hold a long single stance — that's correct)
            corr = np.clip(corr, -3.0, 3.0)
            # residual slide = motion of the corrected foot while the
            # SAME foot stays planted (a swap is a step, not a slide).
            cf = cur + corr
            if not swapped and prev_corr_foot is not None:
                deslid_slip += float(np.linalg.norm(cf - prev_corr_foot))
            prev_corr_foot = cf
        # apply the correction to the recorded WORLD root XY (MuJoCo
        # ground plane). All of corr / feet / root are in world space.
        root_pos_world[f][0] += float(corr[0])
        root_pos_world[f][1] += float(corr[1])

    # ── WORLD → VRM conversion (done ONCE, after de-slide) ───────────
    # Undo the floor lift in WORLD Z (where it was added), then rotate
    # the root pose back into VRM Y-up space. Joint quats are bone-local
    # and frame-invariant, so they need no conversion.
    out_hips: List[List[float]] = []
    for f in range(ref.n_frames):
        rp = root_pos_world[f].copy()
        rp[2] -= h.z_lift                      # undo lift in world Z
        vrm_hips_t = q_rotate(q_conj(R_GLOBAL), rp)
        vrm_hips_q = q_norm(q_mul(q_conj(R_GLOBAL), root_quat_world[f]))
        out_hips.append([float(x) for x in vrm_hips_t])
        out_rot['Hips'].append(wxyz_to_xyzw(vrm_hips_q))

    metrics = {
        'ok': True,
        'clip': os.path.basename(clip_path),
        'n_frames': ref.n_frames,
        'fps': ref.fps,
        'substeps_per_frame': sub,
        'raw_self_pen_frames': raw_pen_frames,
        'raw_self_pen_max_cm': round(raw_pen_max * 100, 2),
        'post_self_pen_frames': post_pen_frames,
        'post_self_pen_max_cm': round(post_pen_max * 100, 2),
        'mean_track_err_deg': round(float(np.mean(track_err_deg)), 2),
        'max_track_err_deg': round(float(np.max(track_err_deg)), 2),
        'foot_slip_total_cm': round(foot_slip_total * 100, 2),
        'foot_slide_baked_cm': round(deslid_slip * 100, 2),
        'deslide': (not no_deslide),
    }

    if out_path:
        src = dict(ref._src)
        src['rotations'] = out_rot
        src['hips_translation'] = out_hips
        src['physics'] = {
            'baked': True, 'method': 'mujoco_pd_track_v1',
            'total_mass_kg': total_mass,
            'deslide': (not no_deslide),
            'metrics': {k: metrics[k] for k in (
                'raw_self_pen_frames', 'post_self_pen_frames',
                'mean_track_err_deg', 'foot_slip_total_cm')},
        }
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(src, fp)
        metrics['out'] = out_path

    if verbose:
        print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('--out', default=None)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--mass', type=float, default=70.0)
    ap.add_argument('--no-deslide', action='store_true')
    a = ap.parse_args()
    r = Reference(a.clip)
    vr = validate_fk(r)
    print('FK_GATE', json.dumps(vr))
    if a.validate_only:
        sys.exit(0)
    bake_clip(a.clip, a.out, a.mass, no_deslide=a.no_deslide)

