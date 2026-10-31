"""Fit the layout's depth to a photo of the object's side.

The user's photo is taken from the front, so it pins down where parts are across it but not how deep they sit, how
far they reach forward or back, or how they tilt toward the camera: a ring can lean the wrong way, or reach back
instead of forward, and still cover the front photo. A side photo shows exactly that. Here SAM's outline of the
object in a side photo is matched the way fit.py matches the front: the outer parts, each carrying the parts that sit
on or inside it, are moved in depth, tilted, lifted and stretched front to back, one at a time, keeping each change
that makes the parts' silhouette from that side overlap the outline better without losing the front photo's fit.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from . import pose
from .checks import _bounds, _gap
from .fit import Outline, _aligned, _iou, _projected

# The image edge the object's front points to in the side photo -> where the camera stands (azimuth): from +90 the
# object's front shows on the image's left, from -90 on its right.
AZIMUTH = {"left": 90.0, "right": -90.0}
# A side photo that isn't upright may be held at an angle: its roll and slight elevation are searched, never enough
# to flip it over. An upright one is taken as level.
ROLLS = range(-20, 21, 10)
ELEVATIONS = (-10, 0, 10)
SHIFTS = (-0.3, -0.15, -0.07, 0.0, 0.07, 0.15, 0.3)   # of the object's depth
# Degrees about the object's x axis: coarse first (a ring drafted on its side lies down), then fine.
TILTS = ((-90, -60, -30, -15, 0, 15, 30, 60, 90), (-10, -5, 0, 5, 10))
LIFTS = (-0.08, 0.0, 0.08)                            # of the object's height
STRETCHES = (0.8, 1.0, 1.25)                          # front to back
# Outer parts at least this share of the outer volume are moved; smaller ones ride along with the part they sit on.
MAJOR_SHARE = 0.02
# How far one group may travel in all, so small gains can't add up to a contorted layout.
MAX_SHIFT = 0.35   # of the object's depth
MAX_TILT = 90.0
# Parts that touched before the fit should still touch after it: each pair pulled apart costs up to this much overlap.
CONTACT_GAP_MM = 3.0
CONTACT_COST = 0.05
MIN_GAIN = 0.05
# Box stand-ins can twist into a pose that only roughly matches; a fit is kept only when it matches the side well.
MIN_SIDE = 0.75
MAX_FRONT_LOSS = 0.03
PASSES = 2


def _volume(p: dict[str, Any]) -> float:
    return float(np.prod([max(v, 0.1) for v in p["size_mm"]]))


def groups(parts: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Each major outer part with the parts that ride along with it: those whose centre lies within its box (a little
    enlarged), assigned to the smallest such major part, so buttons stay on their plate and boards in their shell."""
    outer = [p for p in parts if p["kind"] != "internal"]
    total = sum(_volume(p) for p in outer) or 1.0
    major = sorted((p for p in outer if _volume(p) >= MAJOR_SHARE * total), key=_volume, reverse=True)
    out = {p["id"]: [p["id"]] for p in major}
    for p in parts:
        if p["id"] in out:
            continue
        holders = []
        for m in major:
            lo, hi = _bounds(m)
            pad = [0.15 * (hi[i] - lo[i]) for i in range(3)]
            if all(lo[i] - pad[i] <= p["center_mm"][i] <= hi[i] + pad[i] for i in range(3)):
                holders.append(m)
        if holders:
            out[min(holders, key=_volume)["id"]].append(p["id"])
    return out


def _depth_axis(p: dict[str, Any]) -> int:
    """The part's own axis that lies most nearly front to back in the object."""
    frame = np.array(pose.part_rotation(p.get("rotation_deg") or [0, 0, 0]))
    return int(np.argmax(np.abs(frame[2])))


def move(parts: list[dict[str, Any]], members: list[str], *, shift: float = 0.0, tilt: float = 0.0, lift: float = 0.0,
         stretch: float = 1.0) -> list[dict[str, Any]]:
    """A copy of `parts` with one group moved: shifted in depth and height (mm), tilted about the object's x axis
    through its major part's centre, and stretched front to back about that centre."""
    by_id = {p["id"]: p for p in parts}
    pivot = np.array(by_id[members[0]]["center_mm"], float)
    a = math.radians(tilt)
    turn = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    moved = []
    for p in parts:
        if p["id"] not in members:
            moved.append(p)
            continue
        q = copy.deepcopy(p)
        off = np.array(q["center_mm"], float) - pivot
        off[2] *= stretch
        if q["id"] == members[0] and stretch != 1.0:
            axis = _depth_axis(q)
            q["size_mm"][axis] = round(q["size_mm"][axis] * stretch, 1)
        c = pivot + turn @ off + [0.0, lift, shift]
        q["center_mm"] = [round(float(v), 1) for v in c]
        # The part turns are Euler XYZ with x outermost, so a tilt about the object's x axis adds to the first angle.
        if tilt:
            r = list(q.get("rotation_deg") or [0, 0, 0])
            q["rotation_deg"] = [round(r[0] + tilt, 1), r[1], r[2]]
        moved.append(q)
    return moved


def _overlap(parts: list[dict[str, Any]], camera: dict[str, float], outline: Outline) -> float:
    return _iou(_aligned(_projected(parts, camera), outline), outline.mask)


def side_camera(parts: list[dict[str, Any]], outline: Outline, azimuth: float, upright: bool) -> dict[str, float]:
    """The roll and elevation of the side photo that best match the layout as it stands (level when upright)."""
    if upright:
        return {"azimuth_deg": azimuth, "elevation_deg": 0.0, "roll_deg": 0.0}
    views = [{"azimuth_deg": azimuth, "elevation_deg": float(e), "roll_deg": float(r)} for e in ELEVATIONS for r in ROLLS]
    return max(views, key=lambda v: _overlap(parts, v, outline))


def fit_depth(anatomy: dict[str, Any], front: Outline, side: Outline, front_points: str, upright: bool = False) -> dict[str, Any] | None:
    """Fit the layout to the side outline. Mutates `anatomy`; returns a report, or None when the side photo's
    direction is unknown or the fit does not clearly improve the side without losing the front."""
    azimuth = AZIMUTH.get(front_points)
    if azimuth is None:
        return None
    parts = anatomy["parts"]
    front_cam = anatomy["object"].get("camera") or {}
    dims = anatomy["object"]["dimensions_mm"]
    cam = side_camera(parts, side, azimuth, upright)
    side_before, front_before = _overlap(parts, cam, side), _overlap(parts, front_cam, front)

    group_of = {pid: key for key, members in groups(parts).items() for pid in members}
    index = {p["id"]: i for i, p in enumerate(parts)}
    # Touching pairs that a move could separate: in different groups (a group moves as one).
    contacts = [(index[a["id"]], index[b["id"]]) for i, a in enumerate(parts) for b in parts[i + 1:]
                if group_of.get(a["id"], a["id"]) != group_of.get(b["id"], b["id"]) and _gap(a, b) <= CONTACT_GAP_MM]

    def score(ps: list[dict[str, Any]]) -> float:
        apart = sum(min(max(_gap(ps[i], ps[j]) - CONTACT_GAP_MM, 0.0) / 10.0, 1.0) for i, j in contacts)
        return _overlap(ps, cam, side) + _overlap(ps, front_cam, front) - CONTACT_COST * apart

    best, best_score = parts, score(parts)
    travel: dict[str, list[float]] = {}
    for tilts in TILTS[:PASSES]:
        for key, members in groups(best).items():
            done = travel.setdefault(key, [0.0, 0.0])
            for options in ([{"shift": s * dims[2], "tilt": t} for s in SHIFTS for t in tilts],
                            [{"lift": h * dims[1], "stretch": k} for h in LIFTS for k in STRETCHES]):
                pick = None
                for o in options:
                    if (abs(done[0] + o.get("shift", 0.0)) > MAX_SHIFT * dims[2]
                            or abs(done[1] + o.get("tilt", 0.0)) > MAX_TILT):
                        continue
                    trial = move(best, members, **o)
                    s = score(trial)
                    if s > best_score + 1e-3:
                        best, best_score, pick = trial, s, o
                if pick:
                    done[0] += pick.get("shift", 0.0)
                    done[1] += pick.get("tilt", 0.0)
        cam = side_camera(best, side, azimuth, upright)
        best_score = score(best)
    side_after, front_after = _overlap(best, cam, side), _overlap(best, front_cam, front)
    if side_after < max(side_before + MIN_GAIN, MIN_SIDE) or front_after < front_before - MAX_FRONT_LOSS:
        return None
    moved = [p["id"] for p, q in zip(parts, best) if p != q]
    anatomy["parts"] = best
    return {"side_before": side_before, "side_after": side_after, "front_before": front_before,
            "front_after": front_after, "moved": moved, "side_camera": cam}
