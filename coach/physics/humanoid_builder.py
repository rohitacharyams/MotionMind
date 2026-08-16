"""humanoid_builder.py — build a MuJoCo humanoid from OUR VRM skeleton.

The coach's retargeted clips use a fixed 19-bone humanoid (see any
``coach/motion_cache_cmu/<id>.json`` → ``rest_local_translation``):

    Hips → Spine → Spine2 → Neck → Head
    Spine2 → {Left,Right}Shoulder → Arm → ForeArm → Hand
    Hips  → {Left,Right}UpLeg → Leg → Foot

This module turns that skeleton (bone offsets in metres, VRM Y-up) into a
physically-grounded MuJoCo model: capsule limbs with anthropometric mass,
collision geometry (so a hand cannot pass through the head), ball joints
with anatomical limits, foot boxes, and a free root.

It is the foundation of the offline PD-tracking *physics bake* — we drive
this body toward each clip's reference pose under gravity + contact, then
record the physically-valid result back into our motion-JSON format.

Design notes
------------
* Masses use Winter's anthropometric fractions of a 70 kg body.
* A bone's capsule spans from its own origin to its *primary* child's
  origin (the limb's "meat"). Leaf bones (Head, Hand, Foot) get fixed
  shapes.
* Self-collision: MuJoCo auto-excludes joint-connected (parent↔child)
  pairs; everything else with matching contype/conaffinity collides — so
  hand↔head and hand↔torso DO collide (the bug we are fixing) while
  forearm↔upperarm do not. A few near-neighbour pairs are excluded
  explicitly to avoid jitter.
* No fingers/toes (our retarget has none) — hands/feet are single bodies.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Parent of each bone (None = root). Matches the retarget skeleton.
PARENT: Dict[str, Optional[str]] = {
    'Hips': None,
    'Spine': 'Hips', 'Spine2': 'Spine', 'Neck': 'Spine2', 'Head': 'Neck',
    'LeftShoulder': 'Spine2', 'LeftArm': 'LeftShoulder',
    'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm',
    'RightShoulder': 'Spine2', 'RightArm': 'RightShoulder',
    'RightForeArm': 'RightArm', 'RightHand': 'RightForeArm',
    'LeftUpLeg': 'Hips', 'LeftLeg': 'LeftUpLeg', 'LeftFoot': 'LeftLeg',
    'RightUpLeg': 'Hips', 'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg',
}

# The capsule for a body points toward this child (its "down the limb"
# direction). Bodies absent here are leaves and get a fixed shape.
PRIMARY_CHILD: Dict[str, str] = {
    'Hips': 'Spine', 'Spine': 'Spine2', 'Spine2': 'Neck', 'Neck': 'Head',
    'LeftShoulder': 'LeftArm', 'LeftArm': 'LeftForeArm',
    'LeftForeArm': 'LeftHand',
    'RightShoulder': 'RightArm', 'RightArm': 'RightForeArm',
    'RightForeArm': 'RightHand',
    'LeftUpLeg': 'LeftLeg', 'LeftLeg': 'LeftFoot',
    'RightUpLeg': 'RightLeg', 'RightLeg': 'RightFoot',
}

# Winter anthropometric mass fraction of total body mass (segment % BW).
MASS_FRAC: Dict[str, float] = {
    'Hips': 0.142,            # pelvis
    'Spine': 0.139,           # lower trunk
    'Spine2': 0.216,          # upper trunk (thorax)
    'Neck': 0.012,
    'Head': 0.069,
    'LeftShoulder': 0.010, 'RightShoulder': 0.010,
    'LeftArm': 0.028, 'RightArm': 0.028,           # upper arm
    'LeftForeArm': 0.016, 'RightForeArm': 0.016,   # forearm
    'LeftHand': 0.006, 'RightHand': 0.006,
    'LeftUpLeg': 0.100, 'RightUpLeg': 0.100,       # thigh
    'LeftLeg': 0.0465, 'RightLeg': 0.0465,         # shank
    'LeftFoot': 0.0145, 'RightFoot': 0.0145,
}

# Capsule radius (m) per bone group — rough limb thickness.
RADIUS: Dict[str, float] = {
    'Hips': 0.11, 'Spine': 0.10, 'Spine2': 0.12, 'Neck': 0.04,
    'LeftShoulder': 0.045, 'RightShoulder': 0.045,
    'LeftArm': 0.045, 'RightArm': 0.045,
    'LeftForeArm': 0.038, 'RightForeArm': 0.038,
    'LeftUpLeg': 0.075, 'RightUpLeg': 0.075,
    'LeftLeg': 0.055, 'RightLeg': 0.055,
}
HEAD_RADIUS = 0.09
HAND_RADIUS = 0.035
FOOT_HALF = (0.045, 0.12, 0.035)   # half-extents box (x,y,z) toes forward

# Default ball-joint angular limit (rad) away from rest. Tighter for the
# spine/neck (they should not flail), looser for shoulders/hips.
JOINT_LIMIT: Dict[str, float] = {
    'Spine': 0.6, 'Spine2': 0.6, 'Neck': 0.8, 'Head': 0.5,
    'LeftShoulder': 0.5, 'RightShoulder': 0.5,
    'LeftArm': 2.3, 'RightArm': 2.3,
    'LeftForeArm': 2.4, 'RightForeArm': 2.4,
    'LeftHand': 1.0, 'RightHand': 1.0,
    'LeftUpLeg': 1.8, 'RightUpLeg': 1.8,
    'LeftLeg': 2.2, 'RightLeg': 2.2,
    'LeftFoot': 0.9, 'RightFoot': 0.9,
}

# Near-neighbour collision pairs to exclude (avoid contact jitter on
# bones that are anatomically close but not parent↔child).
# MuJoCo already auto-excludes immediate parent↔child pairs; these are
# the 2-hop pairs inside the self-collision set that would otherwise
# overlap statically (fat capsules, short neck/spine).
EXCLUDE_PAIRS: List[Tuple[str, str]] = [
    ('Spine2', 'LeftArm'), ('Spine2', 'RightArm'),
    ('Head', 'Spine2'),
]

# SURGICAL self-collision set. Only the bones whose interpenetration is
# the *visible bug* (a raised hand/forearm passing through the head or
# chest, or the two arms crossing). Torso-internal + leg pairs are
# intentionally OUT — their fat capsules overlap statically and aren't
# the bug, so including them only adds false positives + jitter. These
# geoms use collision channel 2 so they collide with EACH OTHER but not
# the floor (channel 1).
SELF_COLLIDE = {
    'Head', 'Spine2',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand',
}

ORDER = [  # depth-first body order (stable joint indexing)
    'Hips', 'Spine', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'RightUpLeg', 'RightLeg', 'RightFoot',
]


def load_skeleton(clip_path: str) -> Dict[str, List[float]]:
    """Read ``rest_local_translation`` (bone offsets) from a clip JSON."""
    with open(clip_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    rlt = d.get('rest_local_translation') or {}
    if not rlt:
        raise ValueError(f'{clip_path} has no rest_local_translation')
    return {k: [float(x) for x in v] for k, v in rlt.items()}


def _children(parent: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {b: [] for b in parent}
    for b, p in parent.items():
        if p is not None:
            out[p].append(b)
    return out


def _capsule_from_to(p_from: np.ndarray, p_to: np.ndarray,
                     radius: float) -> str:
    """MuJoCo capsule geom spanning two local points (fromto)."""
    return (f'<geom type="capsule" fromto="'
            f'{p_from[0]:.4f} {p_from[1]:.4f} {p_from[2]:.4f} '
            f'{p_to[0]:.4f} {p_to[1]:.4f} {p_to[2]:.4f}" '
            f'size="{radius:.4f}"/>')


def build_humanoid_xml(skeleton: Dict[str, List[float]],
                       total_mass: float = 70.0,
                       timestep: float = 0.005) -> Tuple[str, List[str]]:
    """Return (mujoco_xml, joint_order).

    ``joint_order`` lists the bones (excluding Hips root) in the order
    their ball joints appear — the PD bake writes targets in this order.
    """
    children = _children(PARENT)
    off = {b: np.array(skeleton.get(b, [0, 0, 0]), dtype=float)
           for b in PARENT}

    # Root world height: lift so feet are ~ on the floor. Hips offset y is
    # the pelvis height in the rest skeleton (~0.895 above the root frame
    # origin which sits at the feet); we place the free body that high.
    hips_h = float(off['Hips'][1]) if off['Hips'][1] > 0.2 else 0.95

    joint_order: List[str] = []
    lines: List[str] = []

    def emit_body(bone: str, indent: int) -> None:
        pad = '  ' * indent
        o = off[bone]
        if bone == 'Hips':
            # Top-level body carrying the free joint. We keep its rest
            # quat IDENTITY and apply the VRM Y-up→MuJoCo Z-up rotation
            # (R_GLOBAL) when we SET the root qpos in the bake. Because
            # that rotation is applied at the root, every child inherits
            # it, so child ball-joint targets are exactly the clip's
            # VRM-local quaternions (rest_local_rotation is identity).
            lines.append(
                f'{pad}<body name="Hips" pos="0 0 {hips_h:.4f}">')
            lines.append(f'{pad}  <freejoint name="root"/>')
        else:
            pos = f'{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}'
            lines.append(f'{pad}<body name="{bone}" pos="{pos}">')
            lim = JOINT_LIMIT.get(bone, 1.2)
            joint_order.append(bone)
            lines.append(
                f'{pad}  <joint name="{bone}" type="ball" '
                f'damping="2" stiffness="0" '
                f'range="0 {lim:.3f}" limited="true"/>')
        # geom + mass
        frac = MASS_FRAC.get(bone, 0.01)
        mass = max(0.05, frac * total_mass)
        # Collision channels: 2 = self-collision set (limbs that cause
        # the bug — collide with each other, NOT the floor). 0 = no
        # collision. Feet + floor use channel 1 (ground contact only).
        col = 2 if bone in SELF_COLLIDE else 0
        if bone == 'Head':
            lines.append(
                f'{pad}  <geom type="sphere" pos="0 {HEAD_RADIUS:.3f} 0" '
                f'size="{HEAD_RADIUS:.3f}" mass="{mass:.3f}" '
                f'contype="{col}" conaffinity="{col}"/>')
        elif bone in ('LeftHand', 'RightHand'):
            lines.append(
                f'{pad}  <geom type="sphere" size="{HAND_RADIUS:.3f}" '
                f'mass="{mass:.3f}" contype="{col}" conaffinity="{col}"/>')
        elif bone in ('LeftFoot', 'RightFoot'):
            fx, fy, fz = FOOT_HALF
            # foot box extends forward (+y local toe) and sits below ankle
            lines.append(
                f'{pad}  <geom type="box" pos="0 {fy*0.4:.3f} {-fz:.3f}" '
                f'size="{fx:.3f} {fy:.3f} {fz:.3f}" mass="{mass:.3f}" '
                f'contype="1" conaffinity="1" friction="1.2 0.02 0.001"/>')
        else:
            child = PRIMARY_CHILD.get(bone)
            r = RADIUS.get(bone, 0.05)
            if child is not None:
                p_to = off[child]
                # ensure non-zero length
                if np.linalg.norm(p_to) < 1e-3:
                    p_to = np.array([0, -0.08, 0])
                lines.append('  ' * (indent + 1) +
                             _capsule_from_to(np.zeros(3), p_to, r)
                             .replace('/>', f' mass="{mass:.3f}" '
                                      f'contype="{col}" conaffinity="{col}"/>'))
            else:
                lines.append(
                    f'{pad}  <geom type="sphere" size="{r:.3f}" '
                    f'mass="{mass:.3f}" contype="{col}" conaffinity="{col}"/>')
        # recurse children
        for c in children.get(bone, []):
            emit_body(c, indent + 1)
        lines.append(f'{pad}</body>')

    emit_body('Hips', 3)
    body_block = '\n'.join(lines)

    excludes = '\n'.join(
        f'    <exclude body1="{a}" body2="{b}"/>'
        for a, b in EXCLUDE_PAIRS)

    xml = f"""<mujoco model="coach_humanoid">
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="640" offheight="960"/>
  </visual>
  <default>
    <geom condim="3" friction="1 0.05 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <joint armature="0.02"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
          contype="1" conaffinity="1"/>
    <light pos="0 0 4" dir="0 0 -1"/>
{body_block}
  </worldbody>
  <contact>
{excludes}
  </contact>
</mujoco>
"""
    return xml, joint_order


if __name__ == '__main__':
    import sys
    clip = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(os.path.dirname(__file__), '..',
                              'motion_cache_cmu', 'cmu_01_01_01.json'))
    sk = load_skeleton(clip)
    xml, jo = build_humanoid_xml(sk)
    outp = os.path.join(os.path.dirname(__file__), '_humanoid_preview.xml')
    with open(outp, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f'wrote {outp}; joints={len(jo)}: {jo}')
