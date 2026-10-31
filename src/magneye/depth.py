"""Fit the layout's depth to a photo of the object's side.

The user's photo is taken from the front, so it pins down where parts are across it but not how deep they sit, how
far they reach forward or back, or how they tilt toward the camera: a ring can lean the wrong way, or reach back
instead of forward, and still cover the front photo. A side photo shows exactly that. Here SAM's outline of the
object in a side photo is matched the way fit.py matches the front: the outer parts, each carrying the parts that sit
on or inside it, are moved in depth, tilted, lifted and stretched front to back, one at a time, keeping each change
that makes the parts' silhouette from that side overlap the outline better without losing the front photo's fit.

Once parts are modeled, the silhouettes are drawn from their real shapes (a ring as a ring, with its hole) rather
than their boxes, which can twist into a pose that only roughly matches: each part's surface is sampled densely once,
and every trial places the samples, projects them and marks the grid cells they land in.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

import numpy as np

from . import model as M
from . import pose
from .checks import _bounds, _gap
from .mates import round_axis
from .fit import GRID, MAX_AZIMUTH, Outline, _aligned, _iou, _projected

# The image edge the object's front points to in the side photo -> where the camera stands (azimuth): from +90 the
# object's front shows on the image's left, from -90 on its right.
AZIMUTH = {"left": 90.0, "right": -90.0}
# A side photo that isn't upright may be held at an angle: its roll and slight elevation are searched, never enough
# to flip it over. An upright one is taken as level.
ROLLS = (-10, 0, 10)
ELEVATIONS = (-10, 0, 10)
# Moves are measured against the object's largest size: its drafted depth is the very thing in doubt.
SHIFTS = ((-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4), (-0.05, -0.025, 0.0, 0.025, 0.05))
# Degrees about the object's x axis: coarse first (a ring drafted on its side lies down), then fine.
TILTS = ((-90, -60, -30, -15, 0, 15, 30, 60, 90), (-10, -5, 0, 5, 10))
# Up and down only a little: the front photo already placed parts' heights (SAM saw them there), which a side outline
# matched without regard to scale can't check; just enough to stay seated after a tilt.
LIFTS = ((-0.1, -0.05, 0.0, 0.05, 0.1), (-0.025, 0.0, 0.025))
LIFT_COST = 0.005   # overlap a millimetre of height change must earn
STRETCHES = (0.8, 1.0, 1.25)                          # front to back
# Outer parts at least this share of the outer volume are moved; smaller ones ride along with the part they sit on.
MAJOR_SHARE = 0.08
# The biggest part that isn't round is the body the front photo placed: the anchor the rest are fitted around. It may
# lean a little and move in depth, but not up or down.
ANCHOR_TILT = 15.0
# How far one group may travel in all, so small gains can't add up to a contorted layout.
MAX_SHIFT = 0.45   # of the object's largest size
# Round parts (a ring, a dial) may turn right over: which way they face is what a front photo can't tell. Other parts
# only lean.
MAX_TILT = 90.0
MAX_LEAN = 30.0
ROUND = ("cylinder", "disc", "ring", "cone", "capsule", "coil", "gear", "spiral")
# Parts that touched before the fit should still touch after it: each pair pulled apart costs up to this much overlap.
CONTACT_GAP_MM = 3.0
CONTACT_COST = 0.15
MIN_GAIN = 0.05
# A fit is kept only when it matches the side well. Real shapes match a correct layout less tightly than boxes can
# be twisted to (a photo is never a pure profile), so they get their own bar.
MIN_SIDE = 0.75
MIN_SIDE_SHAPES = 0.55
MAX_FRONT_LOSS = 0.03
PASSES = 2


def _volume(p: dict[str, Any], solids: dict[str, Any] | None = None) -> float:
    """A part's volume: its model's when built (a ring is mostly the hole in its box), else its box's."""
    if solids and p["id"] in solids:
        b = solids[p["id"]].bounding_box()
        built = np.prod([max(b[3 + i] - b[i], 1e-6) for i in range(3)])
        return float(solids[p["id"]].volume() * np.prod([max(v, 0.1) for v in p["size_mm"]]) / built)
    return float(np.prod([max(v, 0.1) for v in p["size_mm"]]))


# A part drafted in pieces shares its id's stem: ir_led_flex_a and ir_led_flex_b, speaker_left and speaker_right.
_PIECE = re.compile(r"[_-]?(a|b|c|d|l|r|left|right|top|bottom|upper|lower|inner|outer|\d+)$")


def _stem(pid: str) -> str:
    return _PIECE.sub("", pid)


def _coaxial(ring: dict[str, Any], p: dict[str, Any]) -> bool:
    """A similar round part around the same centre (a shield or diffuser inside the ring), whatever the ring's turn."""
    outer = max(ring["size_mm"]) / 2
    return (p["shape"] in ROUND and math.dist(p["center_mm"], ring["center_mm"]) < 0.2 * outer
            and max(p["size_mm"]) >= 1.2 * outer)


def groups(parts: list[dict[str, Any]], solids: dict[str, Any] | None = None,
           connections: list[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """Each major outer part with the parts that ride along with it, so buttons stay on their plate, boards in their
    shell and covers on it. A round major part (a ring) carries only what belongs to it: parts connected to it, the
    other pieces of those (a flex circuit drafted in halves), and a similar ring around its centre (a shield). Its box
    is mostly the hole, and its drafted turn may be the very thing that is wrong, so neither says what's in it. Everything else rides with the body part
    whose box (a little enlarged) holds its centre, nearest centre first, or else the nearest body part. The body
    (the biggest major part that isn't round) comes first."""
    outer = [p for p in parts if p["kind"] != "internal"]
    size = {p["id"]: _volume(p, solids) for p in outer}
    total = sum(size.values()) or 1.0
    major = sorted((p for p in outer if size[p["id"]] >= MAJOR_SHARE * total), key=lambda p: (p["shape"] not in ROUND, size[p["id"]]), reverse=True)
    out = {p["id"]: [p["id"]] for p in major}
    linked = {frozenset((c.get("from"), c.get("to"))) for c in connections or []}
    bodies = [m for m in major if m["shape"] not in ROUND] or major
    rings = [m for m in major if m["shape"] in ROUND]
    belongs = {r["id"]: {p["id"] for p in parts if frozenset((p["id"], r["id"])) in linked} for r in rings}
    for r in rings:
        stems = {_stem(pid) for pid in belongs[r["id"]]}
        belongs[r["id"]] |= {p["id"] for p in parts if _stem(p["id"]) in stems or _coaxial(r, p)}
    for p in parts:
        if p["id"] in out:
            continue
        ring = next((r for r in rings if p["id"] in belongs[r["id"]]), None)
        if ring:
            out[ring["id"]].append(p["id"])
            continue
        holders = []
        for m in bodies:
            lo, hi = _bounds(m)
            pad = [0.15 * (hi[i] - lo[i]) for i in range(3)]
            if all(lo[i] - pad[i] <= p["center_mm"][i] <= hi[i] + pad[i] for i in range(3)):
                holders.append(m)
        if holders:
            out[min(holders, key=lambda m: math.dist(m["center_mm"], p["center_mm"]))["id"]].append(p["id"])
        elif bodies:
            out[min(bodies, key=lambda m: _gap(p, m))["id"]].append(p["id"])
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


# Sample spacing as a share of a grid cell, and a cap per part so a dense mesh can't slow every trial.
SAMPLE_SPACING = 0.6
MAX_SAMPLES = 12_000


class Shapes:
    """The outer parts' real surfaces as sample points in each built solid's own frame (boxes for parts without a
    model), placed through the same transform the viewer uses, so a moved or turned part moves its samples too."""

    def __init__(self, parts: list[dict[str, Any]], solids: dict[str, Any], extent: float):
        spacing = extent * 1.5 / GRID * SAMPLE_SPACING
        rng = np.random.default_rng(0)
        self.solids: dict[str, Any] = {}
        self.samples: dict[str, np.ndarray] = {}
        for p in parts:
            if p["kind"] == "internal":
                continue
            solid = solids.get(p["id"]) or M.Manifold.cube(p["size_mm"], True)
            v, tri = M.arrays(solid)
            corners = v[tri].astype(float)
            area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
            n = int(min(MAX_SAMPLES, max(area.sum() / spacing ** 2, 200)))
            pick = rng.choice(len(tri), size=n, p=area / area.sum()) if area.sum() > 0 else np.zeros(n, int)
            a, b = rng.random((2, n, 1))
            flip = (a + b) > 1
            a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
            c = corners[pick]
            self.samples[p["id"]] = np.vstack([c[:, 0] + a * (c[:, 1] - c[:, 0]) + b * (c[:, 2] - c[:, 0]), v.astype(float)])
            self.solids[p["id"]] = solid

    def projected(self, parts: list[dict[str, Any]], camera: dict[str, float]) -> np.ndarray:
        rot = np.array(pose.rotation({"object": {"camera": camera}}))
        out = []
        for p in parts:
            pts = self.samples.get(p["id"])
            if pts is None:
                continue
            place = M.placement(p, self.solids[p["id"]])
            out.append((pts @ place[:, :3].T + place[:, 3]) @ rot[:2].T)
        return np.concatenate(out) if out else np.zeros((0, 2))

    def mask(self, parts: list[dict[str, Any]], camera: dict[str, float], outline: Outline) -> np.ndarray:
        """The silhouette scaled and centred onto SAM's box, as _aligned does for boxes."""
        pts = self.projected(parts, camera)
        grid = np.zeros((GRID, GRID), bool)
        if not len(pts):
            return grid
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        bw, bh = outline.box[2] - outline.box[0], outline.box[3] - outline.box[1]
        k = math.sqrt((bw * bh) / (span[0] * span[1]))
        mid = (lo + hi) / 2
        bx, by = (outline.box[0] + outline.box[2]) / 2, (outline.box[1] + outline.box[3]) / 2
        u = ((bx + (pts[:, 0] - mid[0]) * k - outline.x0) / outline.cell).astype(int)
        v = ((by - (pts[:, 1] - mid[1]) * k - outline.y0) / outline.cell).astype(int)
        keep = (u >= 0) & (u < GRID) & (v >= 0) & (v < GRID)
        grid[v[keep], u[keep]] = True
        # Close the pinholes between samples (one cell), keeping real holes such as a ring's.
        grown = grid.copy()
        grown[1:] |= grid[:-1]; grown[:-1] |= grid[1:]; grown[:, 1:] |= grid[:, :-1]; grown[:, :-1] |= grid[:, 1:]
        shrunk = grown.copy()
        shrunk[1:] &= grown[:-1]; shrunk[:-1] &= grown[1:]; shrunk[:, 1:] &= grown[:, :-1]; shrunk[:, :-1] &= grown[:, 1:]
        return shrunk | grid


def _overlap(parts: list[dict[str, Any]], camera: dict[str, float], outline: Outline, shapes: Shapes | None = None) -> float:
    if shapes is not None:
        return _iou(shapes.mask(parts, camera, outline), outline.mask)
    return _iou(_aligned(_projected(parts, camera), outline), outline.mask)


def side_camera(parts: list[dict[str, Any]], outline: Outline, azimuth: float, upright: bool,
                shapes: Shapes | None = None) -> dict[str, float]:
    """The roll and elevation of the side photo that best match the layout as it stands (level when upright)."""
    if upright:
        return {"azimuth_deg": azimuth, "elevation_deg": 0.0, "roll_deg": 0.0}
    views = [{"azimuth_deg": azimuth, "elevation_deg": float(e), "roll_deg": float(r)} for e in ELEVATIONS for r in ROLLS]
    return max(views, key=lambda v: _overlap(parts, v, outline, shapes))


# The front photo's viewpoint was estimated against the drafted layout; as parts tilt it is searched again nearby
# (a flat ring seen from higher up looks like a tilted ring seen from lower down).
FRONT_AZIMUTHS = (-10, -5, 0, 5, 10)
FRONT_ELEVATIONS = (-10, 0, 10, 20)
FRONT_ROLLS = (-10, 0, 10)


def front_camera(parts: list[dict[str, Any]], outline: Outline, around: dict[str, float], shapes: Shapes | None) -> dict[str, float]:
    a0, e0, r0 = (float(around.get(k) or 0) for k in ("azimuth_deg", "elevation_deg", "roll_deg"))
    a0 = min(max(a0, -MAX_AZIMUTH), MAX_AZIMUTH)
    views = [{"azimuth_deg": a0 + a, "elevation_deg": float(min(max(e0 + e, -30), 80)), "roll_deg": r0 + r}
             for a in FRONT_AZIMUTHS for e in FRONT_ELEVATIONS for r in FRONT_ROLLS if abs(a0 + a) <= MAX_AZIMUTH]
    return max(views, key=lambda v: _overlap(parts, v, outline, shapes))


def fit_depth(anatomy: dict[str, Any], front: Outline, side: Outline, front_points: str, upright: bool = False,
              solids: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Fit the layout to the side outline. Mutates `anatomy`; returns a report, or None when the side photo's
    direction is unknown or the fit does not clearly improve the side without losing the front. With `solids` (the
    built models by part id) silhouettes come from the real shapes, and modeled parts are never stretched."""
    azimuth = AZIMUTH.get(front_points)
    if azimuth is None:
        return None
    parts = anatomy["parts"]
    first_cam = front_cam = dict(anatomy["object"].get("camera") or {})
    dims = anatomy["object"]["dimensions_mm"]
    shapes = Shapes(parts, solids, max(dims)) if solids is not None else None
    cam = side_camera(parts, side, azimuth, upright, shapes)
    side_before, front_before = _overlap(parts, cam, side, shapes), _overlap(parts, front_cam, front, shapes)

    connections = anatomy.get("connections") or []
    members_of = groups(parts, solids, connections)
    index = {p["id"]: i for i, p in enumerate(parts)}
    # Groups that touch before the fit (any of their parts within the gap) should still touch after it; a group
    # moves as one, so each pair is judged by its closest parts.
    keys = list(members_of)
    def group_gap(ps: list[dict[str, Any]], a: str, b: str) -> float:
        return min(_gap(ps[index[x]], ps[index[y]]) for x in members_of[a] for y in members_of[b])
    contacts = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:] if group_gap(parts, a, b) <= CONTACT_GAP_MM]

    # The front photo can't judge depth or tilt, and its viewpoint was estimated for the drafted layout, so the search
    # goes by the side photo alone; the front's viewpoint is refitted at the end, where the gate makes sure the front
    # didn't suffer.
    def score(ps: list[dict[str, Any]]) -> float:
        apart = sum(min(max(group_gap(ps, a, b) - CONTACT_GAP_MM, 0.0) / 10.0, 1.0) for a, b in contacts)
        return _overlap(ps, cam, side, shapes) - CONTACT_COST * apart

    best, best_score = parts, score(parts)
    by_id = {p["id"]: p for p in parts}
    travel: dict[str, list[float]] = {}
    reach = max(dims)
    for tilts, shifts, lifts in zip(TILTS, SHIFTS, LIFTS):
        # The other groups are fitted around the body first, then the body itself, so it doesn't bend to make up for
        # a part that is still wrong.
        ordered = list(groups(best, solids, connections).items())
        for n, (key, members) in sorted(enumerate(ordered), key=lambda e: e[0] == 0):
            done = travel.setdefault(key, [0.0, 0.0])
            anchor = n == 0
            lean = ANCHOR_TILT if anchor else MAX_TILT if by_id[key]["shape"] in ROUND else MAX_LEAN
            for options in ([{"shift": s * reach, "tilt": t, "lift": h * reach} for s in shifts for t in tilts for h in lifts],
                            [{"stretch": k} for k in STRETCHES]):
                # Every option starts from the same layout, and only the best is applied, so moves can't chain past
                # the travel limits.
                base, pick, found, found_score = best, None, best, best_score
                for o in options:
                    if (abs(done[0] + o.get("shift", 0.0)) > MAX_SHIFT * reach
                            or abs(done[1] + o.get("tilt", 0.0)) > lean
                            or (anchor and o.get("lift"))
                            or (o.get("stretch", 1.0) != 1.0 and solids and members[0] in solids)):
                        continue
                    trial = move(base, members, **o)
                    s = score(trial) - LIFT_COST * abs(o.get("lift", 0.0))
                    if s > found_score + 1e-3:
                        found, found_score, pick = trial, s, o
                best, best_score = found, score(found)
                if pick:
                    done[0] += pick.get("shift", 0.0)
                    done[1] += pick.get("tilt", 0.0)
        cam = side_camera(best, side, azimuth, upright, shapes)
        best_score = score(best)
    front_cam = front_camera(best, front, first_cam, shapes)
    side_after, front_after = _overlap(best, cam, side, shapes), _overlap(best, front_cam, front, shapes)
    if side_after < max(side_before + MIN_GAIN, MIN_SIDE_SHAPES if shapes else MIN_SIDE) or front_after < front_before - MAX_FRONT_LOSS:
        return None
    moved = [p["id"] for p, q in zip(parts, best) if p != q]
    anatomy["parts"] = best
    anatomy["object"]["camera"] = {k: round(float(v), 1) for k, v in front_cam.items()}
    return {"side_before": side_before, "side_after": side_after, "front_before": front_before,
            "front_after": front_after, "moved": moved, "side_camera": cam, "camera": anatomy["object"]["camera"]}
