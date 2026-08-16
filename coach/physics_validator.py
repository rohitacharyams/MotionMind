"""physics_validator.py — defense-in-depth motion safety for the AI Coach.

Three usages:

1. CLI offline scan of the motion DB:
       py -3.12 -m coach.physics_validator scan
   → writes coach/motion_safety.json with pass/fail per clip and reason.

2. Library call from the composer / agent:
       from coach.physics_validator import validate_motion, clamp_pose
       report = validate_motion(poses, trans, fps=30)
       safe_poses = clamp_pose(poses)  # idempotent hard clamp

3. The same `clamp_pose` is shipped to the browser as `motion_player.js`
   so the runtime guard uses identical limits.

The SMPL 24-joint convention (AIST++ schema):
    0  pelvis           12 neck
    1  L_hip            13 L_collar
    2  R_hip            14 R_collar
    3  spine1           15 head
    4  L_knee           16 L_shoulder
    5  R_knee           17 R_shoulder
    6  spine2           18 L_elbow
    7  L_ankle          19 R_elbow
    8  R_ankle          20 L_wrist
    9  spine3           21 R_wrist
   10  L_foot           22 L_hand
   11  R_foot           23 R_hand

Limits are calibrated against the real AIST++ distribution (real human
dancers reach 50 rad/s peak joint speeds, 2000 rad/s² accelerations, and
bent joints can hit π). The validator's "fail" gate is the absurd-pose
guard, NOT a dance-quality gate. Anything tighter belongs in the runtime
browser guard, which can freeze a single bad frame without breaking the
whole clip.

HARD-FAIL (clip refused):
  NaN / inf in poses or trans
  Axis-angle magnitude > π + 0.1  (≈ 3.24 rad)  on any joint
  Joint angular speed > 80 rad/s        (transient explosions)
  Pelvis XY speed > 8 m/s               (teleports)

WARN (still played, picker prefers ok clips):
  Joint angular speed > 50 rad/s
  Joint angular accel > 2500 rad/s²
  Pelvis Z < -2.0 m                     (clearly broken; AIST trans Z is
                                         relative to start pose so going
                                         negative is NORMAL for crouches)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / 'data' / 'motion_db' / 'aistpp_full' / 'motions'
REPORT_PATH = Path(__file__).resolve().parent / 'motion_safety.json'

# ─── limits ────────────────────────────────────────────────────────────
# Hard-fail (absurd-pose) thresholds.
HARD_ANGLE_RAD  = 3.24          # > π+0.1 means data is wrapping garbage
HARD_ANG_SPEED  = 80.0          # rad/s, instantaneous joint explosion
HARD_PELVIS_XY_V = 8.0          # m/s, teleport

# Warn thresholds (clip still plays; picker prefers cleaner clips).
MAX_ANG_SPEED   = 50.0          # rad/s
MAX_ANG_ACCEL   = 2500.0        # rad/s^2
MAX_PELVIS_XY_V = 4.0           # m/s
MAX_PELVIS_Z_V  = 6.0           # m/s
FLOOR_TOLERANCE = -2.0          # pelvis Z (AIST trans is relative to start)

# Per-joint runtime hard clamp (browser uses these to prevent absurd poses).
# These are the *clamp* limits for the runtime guard, not the validator.
HARD_LIMIT_RAD = {
    4: 2.95, 5: 2.95,           # knees  (full flex in real dance hits ~2.8)
    18: 2.95, 19: 2.95,         # elbows
    12: 1.80, 15: 1.80,         # neck, head
    20: 2.20, 21: 2.20,         # wrists
    7: 1.40, 8: 1.40,           # ankles
}
DEFAULT_HARD = 3.14             # everything else: at most a full half-turn


# ─── dataclasses ───────────────────────────────────────────────────────
@dataclass
class Violation:
    kind: str                   # 'angle' | 'angular_speed' | 'angular_accel' |
                                # 'root_speed_xy' | 'root_speed_z' | 'floor' | 'nan'
    joint: Optional[int]
    frame: int
    value: float
    limit: float


@dataclass
class SafetyReport:
    path: str
    frames: int
    fps: int
    passed: bool
    severity: str = 'ok'        # ok | warn | fail
    violations: List[Violation] = field(default_factory=list)
    max_joint_speed_rad_s: float = 0.0
    max_pelvis_speed_m_s: float = 0.0
    min_pelvis_z: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['violations'] = [asdict(v) for v in self.violations]
        return d


# ─── core checks ───────────────────────────────────────────────────────
def _hard_limit(j: int) -> float:
    return HARD_LIMIT_RAD.get(j, DEFAULT_HARD)


def validate_motion(poses: np.ndarray,
                    trans: np.ndarray,
                    fps: int = 30,
                    path: str = '<inline>',
                    max_violations: int = 20) -> SafetyReport:
    """Validate an SMPL motion array.

    Parameters
    ----------
    poses : (T, 24, 3) float, axis-angle per joint
    trans : (T, 3) float, root translation in meters
    fps   : frames per second
    max_violations : stop collecting after this many (still reports counts)
    """
    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(trans, dtype=np.float32).reshape(-1, 3)
    T = poses.shape[0]
    rep = SafetyReport(path=path, frames=T, fps=fps, passed=True)

    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(trans)):
        rep.violations.append(Violation('nan', None, 0, float('nan'), 0.0))
        rep.passed = False
        rep.severity = 'fail'
        return rep

    # angle magnitude per joint per frame (only flag when > hard absurd)
    mag = np.linalg.norm(poses, axis=-1)  # (T, 24)
    bad_angle = np.where(mag > HARD_ANGLE_RAD)
    for f, j in zip(bad_angle[0][:10], bad_angle[1][:10]):
        rep.violations.append(Violation('angle', int(j), int(f),
                                        float(mag[f, j]), HARD_ANGLE_RAD))

    # angular speed (forward difference of axis-angle magnitude is a cheap
    # proxy; for true angular velocity we'd diff the quaternion log, but
    # in practice the magnitude rate-of-change catches all the same
    # offending frames at this scale).
    if T >= 2:
        dt = 1.0 / max(fps, 1)
        d_mag = np.diff(mag, axis=0) / dt          # (T-1, 24) rad/s
        rep.max_joint_speed_rad_s = float(np.max(np.abs(d_mag)))
        # Hard fails first
        bad_hard = np.where(np.abs(d_mag) > HARD_ANG_SPEED)
        for f, j in zip(bad_hard[0][:5], bad_hard[1][:5]):
            rep.violations.append(Violation('angular_speed', int(j),
                                            int(f), float(d_mag[f, j]),
                                            HARD_ANG_SPEED))
        # Soft warns (only count if there were no hard fails on this joint)
        bad_soft = np.where((np.abs(d_mag) > MAX_ANG_SPEED) &
                            (np.abs(d_mag) <= HARD_ANG_SPEED))
        for f, j in zip(bad_soft[0][:3], bad_soft[1][:3]):
            rep.violations.append(Violation('angular_speed_warn', int(j),
                                            int(f), float(d_mag[f, j]),
                                            MAX_ANG_SPEED))

    # angular acceleration (warn-only)
    if T >= 3:
        dt = 1.0 / max(fps, 1)
        d2_mag = np.diff(mag, n=2, axis=0) / (dt * dt)   # (T-2, 24)
        bad = np.where(np.abs(d2_mag) > MAX_ANG_ACCEL)
        for f, j in zip(bad[0][:3], bad[1][:3]):
            rep.violations.append(Violation('angular_accel', int(j),
                                            int(f), float(d2_mag[f, j]),
                                            MAX_ANG_ACCEL))

    # root speed
    if T >= 2:
        dt = 1.0 / max(fps, 1)
        v = np.diff(trans, axis=0) / dt
        v_xy = np.linalg.norm(v[:, :2], axis=1)
        v_z = np.abs(v[:, 2])
        rep.max_pelvis_speed_m_s = float(np.max(v_xy))
        bad_xy_hard = np.where(v_xy > HARD_PELVIS_XY_V)[0]
        for f in bad_xy_hard[:5]:
            rep.violations.append(Violation('root_speed_xy', None, int(f),
                                            float(v_xy[f]), HARD_PELVIS_XY_V))
        bad_xy_soft = np.where((v_xy > MAX_PELVIS_XY_V) & (v_xy <= HARD_PELVIS_XY_V))[0]
        for f in bad_xy_soft[:3]:
            rep.violations.append(Violation('root_speed_xy_warn', None, int(f),
                                            float(v_xy[f]), MAX_PELVIS_XY_V))
        bad_z = np.where(v_z > MAX_PELVIS_Z_V)[0]
        for f in bad_z[:3]:
            rep.violations.append(Violation('root_speed_z', None, int(f),
                                            float(v_z[f]), MAX_PELVIS_Z_V))

    # floor
    rep.min_pelvis_z = float(np.min(trans[:, 2]))
    if rep.min_pelvis_z < FLOOR_TOLERANCE:
        f = int(np.argmin(trans[:, 2]))
        rep.violations.append(Violation('floor', None, f, rep.min_pelvis_z,
                                        FLOOR_TOLERANCE))

    # severity
    hard_kinds = {'nan', 'angle', 'angular_speed', 'root_speed_xy'}
    soft_kinds = {'angular_accel', 'angular_speed_warn',
                  'root_speed_xy_warn', 'root_speed_z', 'floor'}
    has_hard = any(v.kind in hard_kinds for v in rep.violations)
    has_soft = any(v.kind in soft_kinds for v in rep.violations)
    if has_hard:
        rep.passed = False
        rep.severity = 'fail'
    elif has_soft:
        rep.passed = True
        rep.severity = 'warn'
    return rep


# ─── runtime hard clamp (mirrored in motion_player.js) ─────────────────
def clamp_pose(poses: np.ndarray) -> np.ndarray:
    """Idempotent hard clamp on axis-angle magnitudes. Safe to apply
    every frame on top of any pose. Preserves direction, only shrinks
    magnitude when it exceeds the hard joint limit."""
    out = np.asarray(poses, dtype=np.float32).reshape(-1, 24, 3).copy()
    mag = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9  # (T,24,1)
    for j in range(24):
        lim = _hard_limit(j)
        m = mag[:, j, 0]
        over = m > lim
        if np.any(over):
            scale = np.where(over, lim / m, 1.0)
            out[:, j, :] *= scale[:, None]
    return out


def normalize_translation(trans: np.ndarray,
                          floor_percentile: float = 2.0,
                          target_floor_z: float = 0.0,
                          max_xy_radius: float = 2.5,
                          center_xy: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize root translation for studio playback.

    - Floor anchor: shift Z so the low percentile sits on target floor.
    - Center XY: recenter median XY to origin so clips stay framed.
    - Radius cap: if trajectory is huge, uniformly scale XY to max radius.

    Returns normalized trans and a compact stats dict describing changes.
    """
    out = np.asarray(trans, dtype=np.float32).reshape(-1, 3).copy()
    if out.size == 0:
        return out, {'floor_shift_z': 0.0, 'xy_center_shift': 0.0,
                     'xy_scale': 1.0, 'max_xy_radius_before': 0.0,
                     'max_xy_radius_after': 0.0}

    stats: Dict[str, float] = {
        'floor_shift_z': 0.0,
        'xy_center_shift': 0.0,
        'xy_scale': 1.0,
        'max_xy_radius_before': 0.0,
        'max_xy_radius_after': 0.0,
    }

    # Z floor anchor.
    floor_ref = float(np.percentile(out[:, 2], floor_percentile))
    dz = float(target_floor_z - floor_ref)
    out[:, 2] += dz
    stats['floor_shift_z'] = dz

    xy = out[:, :2]
    center = np.zeros(2, dtype=np.float32)
    if center_xy:
        center = np.median(xy, axis=0).astype(np.float32)
        xy = xy - center[None, :]
        stats['xy_center_shift'] = float(np.linalg.norm(center))

    r = np.linalg.norm(xy, axis=1)
    max_before = float(np.max(r)) if r.size else 0.0
    scale = 1.0
    if max_xy_radius > 0 and max_before > max_xy_radius and max_before > 1e-6:
        scale = float(max_xy_radius / max_before)
        xy *= scale

    out[:, :2] = xy
    stats['xy_scale'] = scale
    stats['max_xy_radius_before'] = max_before
    stats['max_xy_radius_after'] = float(np.max(np.linalg.norm(out[:, :2], axis=1)))
    return out, stats


def fix_motion_arrays(poses: np.ndarray,
                      trans: np.ndarray,
                      fps: int = 30,
                      floor_percentile: float = 2.0,
                      target_floor_z: float = 0.0,
                      max_xy_radius: float = 2.5,
                      center_xy: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply deterministic safety fixes to one motion clip.

    This is intentionally conservative: clamp absurd joint magnitudes and
    normalize translation for studio teaching playback.
    """
    fixed_poses = clamp_pose(poses)
    fixed_trans, tr_stats = normalize_translation(
        trans,
        floor_percentile=floor_percentile,
        target_floor_z=target_floor_z,
        max_xy_radius=max_xy_radius,
        center_xy=center_xy,
    )
    tr_stats['fps'] = float(fps)
    return fixed_poses, fixed_trans, tr_stats


def fix_motion_file(path: Path,
                    floor_percentile: float = 2.0,
                    target_floor_z: float = 0.0,
                    max_xy_radius: float = 2.5,
                    center_xy: bool = True,
                    dry_run: bool = False,
                    backup_ext: str = '.bak') -> Tuple[SafetyReport, SafetyReport, Dict[str, float]]:
    """Fix one .pkl file in place and return before/after reports.

    The file format keeps `smpl_trans` in source units and optional
    `smpl_scaling`; we fix in metric space then map back to stored units.
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    original_d = pickle.loads(pickle.dumps(d, protocol=4))

    poses = np.asarray(d['smpl_poses'], dtype=np.float32).reshape(-1, 24, 3)
    trans_raw = np.asarray(d['smpl_trans'], dtype=np.float32).reshape(-1, 3)
    fps = int(d.get('fps', 60))
    if fps == 60:
        poses = poses[::2]
        trans_raw = trans_raw[::2]
        fps = 30

    scaling = float(np.asarray(d.get('smpl_scaling', [1.0])).flatten()[0])
    trans_metric = trans_raw / max(scaling, 1e-6) if scaling != 1.0 else trans_raw.copy()

    before = validate_motion(poses, trans_metric, fps=fps, path=str(path))
    fixed_poses, fixed_trans_metric, fx = fix_motion_arrays(
        poses,
        trans_metric,
        fps=fps,
        floor_percentile=floor_percentile,
        target_floor_z=target_floor_z,
        max_xy_radius=max_xy_radius,
        center_xy=center_xy,
    )
    after = validate_motion(fixed_poses, fixed_trans_metric, fps=fps, path=str(path))

    if not dry_run:
        # Keep the source fps representation (often 60) for compatibility.
        out_poses = fixed_poses
        out_trans_metric = fixed_trans_metric
        if int(d.get('fps', 60)) == 60:
            out_poses = np.repeat(fixed_poses, 2, axis=0)
            out_trans_metric = np.repeat(fixed_trans_metric, 2, axis=0)

        d['smpl_poses'] = out_poses.reshape(-1, 72).astype(np.float32)
        out_trans_raw = out_trans_metric * scaling if scaling != 1.0 else out_trans_metric
        d['smpl_trans'] = out_trans_raw.astype(np.float32)

        if backup_ext:
            b = path.with_suffix(path.suffix + backup_ext)
            if not b.exists():
                with open(b, 'wb') as bf:
                    pickle.dump(original_d, bf, protocol=4)

        with open(path, 'wb') as wf:
            pickle.dump(d, wf, protocol=4)

    return before, after, fx


# ─── JSON export of limits (browser pulls this) ────────────────────────
def export_limits_json() -> dict:
    return {
        'hard_limit_rad':       HARD_LIMIT_RAD,
        'default_hard':         DEFAULT_HARD,
        # The browser runtime guard uses a hard angular-speed cap to
        # freeze on single-frame explosions. Set well above real dance
        # (which peaks ~50 rad/s) so normal hits never trigger.
        'max_ang_speed':        HARD_ANG_SPEED,
        'max_ang_accel':        MAX_ANG_ACCEL,
        'max_pelvis_xy_v':      HARD_PELVIS_XY_V,
        'max_pelvis_z_v':       MAX_PELVIS_Z_V,
        'floor_tolerance':      FLOOR_TOLERANCE,
    }


# ─── CLI ───────────────────────────────────────────────────────────────
def _scan_dir(directory: Path, glob: str = '*.pkl') -> List[SafetyReport]:
    reports: List[SafetyReport] = []
    files = sorted(directory.glob(glob))
    print(f'[scan] {len(files)} files in {directory}')
    t0 = time.time()
    for i, p in enumerate(files):
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            poses = np.asarray(d['smpl_poses']).reshape(-1, 24, 3)
            trans = np.asarray(d['smpl_trans']).reshape(-1, 3)
            # If clip is AIST raw 60fps, decimate to 30
            fps = int(d.get('fps', 60))
            if fps == 60:
                poses = poses[::2]
                trans = trans[::2]
                fps = 30
            scaling = float(np.asarray(d.get('smpl_scaling', [1.0])).flatten()[0])
            if scaling != 1.0:
                trans = trans / max(scaling, 1e-6)
            rep = validate_motion(poses, trans, fps=fps, path=str(p))
        except Exception as e:                                  # noqa: BLE001
            rep = SafetyReport(path=str(p), frames=0, fps=0,
                               passed=False, severity='fail')
            rep.violations.append(Violation('nan', None, 0, 0.0, 0.0))
            print(f'  [{i+1}/{len(files)}] ERR {p.name}: {e!r}')
            reports.append(rep); continue
        tag = {'ok': 'PASS', 'warn': 'WARN', 'fail': 'FAIL'}[rep.severity]
        if rep.severity != 'ok':
            kinds = sorted({v.kind for v in rep.violations})
            print(f'  [{i+1}/{len(files)}] {tag} {p.name}  '
                  f'frames={rep.frames}  v={len(rep.violations)} '
                  f'({", ".join(kinds)})')
        reports.append(rep)
    elapsed = time.time() - t0
    n_pass = sum(1 for r in reports if r.severity == 'ok')
    n_warn = sum(1 for r in reports if r.severity == 'warn')
    n_fail = sum(1 for r in reports if r.severity == 'fail')
    print(f'[scan] done in {elapsed:.1f}s — PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}')
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_scan = sub.add_parser('scan', help='Validate every clip in motion DB.')
    p_scan.add_argument('--dir', default=str(MOTION_DIR))
    p_scan.add_argument('--glob', default='*.pkl')
    p_scan.add_argument('--out', default=str(REPORT_PATH))
    p_scan.add_argument('--fix', action='store_true',
                        help='Apply deterministic fixes in-place before final report.')
    p_scan.add_argument('--dry-run-fix', action='store_true',
                        help='Simulate fixes and include post-fix report, but do not write files.')
    p_scan.add_argument('--floor-percentile', type=float, default=2.0,
                        help='Percentile used as floor reference for Z grounding.')
    p_scan.add_argument('--target-floor-z', type=float, default=0.0,
                        help='Target floor height in meters after normalization.')
    p_scan.add_argument('--max-xy-radius', type=float, default=2.5,
                        help='Cap XY trajectory radius after recentering (meters).')
    p_scan.add_argument('--no-center-xy', action='store_true',
                        help='Disable XY recentering around origin.')
    p_scan.add_argument('--broken-out', default='',
                        help='Optional path to JSON list of clips still warning/failing.')
    sub.add_parser('limits', help='Print current limit values as JSON.')
    args = ap.parse_args()
    if args.cmd == 'limits':
        print(json.dumps(export_limits_json(), indent=2))
        return 0
    if args.cmd == 'scan':
        reports = _scan_dir(Path(args.dir), args.glob)
        fix_log: List[dict] = []
        if args.fix or args.dry_run_fix:
            files = sorted(Path(args.dir).glob(args.glob))
            idx = {Path(r.path).name: r for r in reports}
            for p in files:
                name = p.name
                current = idx.get(name)
                if current is None:
                    continue
                if current.severity == 'ok' and not args.fix and args.dry_run_fix:
                    continue
                try:
                    before, after, fx = fix_motion_file(
                        p,
                        floor_percentile=float(args.floor_percentile),
                        target_floor_z=float(args.target_floor_z),
                        max_xy_radius=float(args.max_xy_radius),
                        center_xy=not bool(args.no_center_xy),
                        dry_run=bool(args.dry_run_fix),
                        backup_ext='.bak' if args.fix and not args.dry_run_fix else '',
                    )
                    idx[name] = after
                    fix_log.append({
                        'file': name,
                        'before': before.severity,
                        'after': after.severity,
                        'floor_shift_z': fx.get('floor_shift_z', 0.0),
                        'xy_center_shift': fx.get('xy_center_shift', 0.0),
                        'xy_scale': fx.get('xy_scale', 1.0),
                    })
                    tag = 'FIX' if args.fix and not args.dry_run_fix else 'SIM'
                    print(f"  [{tag}] {name}: {before.severity} -> {after.severity}")
                except Exception as e:                              # noqa: BLE001
                    fix_log.append({'file': name, 'error': repr(e)})
                    print(f"  [FIX-ERR] {name}: {e!r}")
            reports = [idx[Path(r.path).name] for r in reports]

        broken = [r for r in reports if r.severity in ('warn', 'fail')]
        out = {
            'limits':  export_limits_json(),
            'reports': [r.to_dict() for r in reports],
            'fixes':   fix_log,
            'summary': {
                'total': len(reports),
                'pass':  sum(1 for r in reports if r.severity == 'ok'),
                'warn':  sum(1 for r in reports if r.severity == 'warn'),
                'fail':  sum(1 for r in reports if r.severity == 'fail'),
            },
        }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'[scan] report written → {args.out}')
        if args.broken_out:
            broken_payload = {
                'count': len(broken),
                'clips': [
                    {
                        'path': r.path,
                        'severity': r.severity,
                        'violations': sorted({v.kind for v in r.violations}),
                    }
                    for r in broken
                ],
            }
            Path(args.broken_out).write_text(
                json.dumps(broken_payload, indent=2),
                encoding='utf-8',
            )
            print(f'[scan] broken list written → {args.broken_out}')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
