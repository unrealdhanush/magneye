"""Mates: how parts fit together.

Instead of every part being placed on its own (and drifting), the modeler says how one part attaches to another
and this places it exactly, the way things are assembled: a cell inside its holder, a button on its switch, a screw
coaxial with its hole, two gears meshed. Only the moving part (`part`) moves; `to` stays. Parts are measured by
their boxes as turned by their rotation (see checks._bounds).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from . import pose
from .checks import _bounds

TYPES = ("inside", "on", "coaxial", "touch", "mesh")
AXES = {"x": 0, "y": 1, "z": 2}
DOC = """mate_parts(mates): fit parts together the way they really assemble; each mate moves `part` onto `to`, which stays put:
- {"part": a, "to": b, "type": "inside"}: a sits within b (a cell in its holder, LEDs inside the ring, a board inside the shell); "at": [dx, dy, dz] offsets it from b's centre.
- {"part": a, "to": b, "type": "on", "face": "+y"}: a sits against that face of b (a button on its switch, the switch on the board, a cover on the housing); a keeps its place across the face, kept within it, unless "at" gives an offset from b's centre.
- {"part": a, "to": b, "type": "coaxial", "axis": "z"}: a shares b's axis (a screw in its hole, a shaft through a gear, a cap on a stick); a keeps its position along the axis.
- {"part": a, "to": b, "type": "touch"}: a moves the least distance that makes it meet b (a cable reaching its connector).
- {"part": a, "to": b, "type": "mesh", "axis": "z", "direction": "+x"}: two gears turning about that axis, a placed beside b in that direction so their teeth engage."""


class MateError(ValueError):
    pass


def _face(face: Any) -> tuple[int, int]:
    if not (isinstance(face, str) and len(face) == 2 and face[0] in "+-" and face[1] in AXES):
        raise MateError("face/direction is one of +x, -x, +y, -y, +z, -z")
    return AXES[face[1]], 1 if face[0] == "+" else -1


def _axis(axis: Any) -> int:
    if axis not in AXES:
        raise MateError("axis is x, y or z")
    return AXES[axis]


def apply(parts: dict[str, dict[str, Any]], mate: dict[str, Any]) -> str:
    """Move mate["part"] so the mate holds; returns what happened, or raises MateError with the reason."""
    a, b = parts.get(mate.get("part", "")), parts.get(mate.get("to", ""))
    if a is None or b is None or a is b:
        raise MateError("part and to must be two different existing part ids")
    kind = mate.get("type")
    if kind not in TYPES:
        raise MateError(f"type is one of {', '.join(TYPES)}")
    (alo, ahi), (blo, bhi) = _bounds(a), _bounds(b)
    asize = [ahi[i] - alo[i] for i in range(3)]
    acen, bcen = list(a["center_mm"]), list(b["center_mm"])
    at = mate.get("at") or [0, 0, 0]
    if not (isinstance(at, list) and len(at) == 3 and all(isinstance(v, (int, float)) for v in at)):
        raise MateError("at is [dx, dy, dz]")
    target = list(acen)

    def within(i: int, want: float) -> float:
        """Keep a's extent inside b's along axis i (centred if a is the larger)."""
        lo, hi = blo[i] + asize[i] / 2, bhi[i] - asize[i] / 2
        return (blo[i] + bhi[i]) / 2 if lo > hi else min(max(want, lo), hi)

    if kind == "inside":
        target = [within(i, bcen[i] + at[i]) for i in range(3)]
    elif kind == "on":
        i, sign = _face(mate.get("face"))
        for j in range(3):
            across = bcen[j] + at[j] if "at" in mate else acen[j]
            target[j] = (bhi[j] + asize[j] / 2 if sign > 0 else blo[j] - asize[j] / 2) if j == i else within(j, across)
    elif kind == "coaxial":
        i = _axis(mate.get("axis"))
        target = [acen[j] if j == i else bcen[j] + at[j] for j in range(3)]
    elif kind == "touch":
        gaps = [max(0.0, blo[i] - ahi[i]) - max(0.0, alo[i] - bhi[i]) for i in range(3)]
        i = max(range(3), key=lambda j: abs(gaps[j]))
        target[i] = acen[i] + gaps[i]
    else:  # mesh
        i = _axis(mate.get("axis"))
        d, sign = _face(mate.get("direction"))
        if d == i:
            raise MateError("mesh: direction must lie across the gears' axis")
        radius = lambda lo, hi: max(hi[j] - lo[j] for j in range(3) if j != i) / 2
        # Tips overlap by a few percent so the teeth engage rather than just touching.
        reach = (radius(alo, ahi) + radius(blo, bhi)) * 0.94
        target = list(bcen)
        target[d] = bcen[d] + sign * reach
    target = [round(v, 1) for v in target]
    moved = sum((x - y) ** 2 for x, y in zip(target, acen)) ** 0.5
    a["center_mm"] = target
    return f"{a['id']} {kind} {b['id']}: moved {moved:.0f} mm"


ROUND = ("cylinder", "disc", "ring", "cone", "capsule", "coil", "gear", "spiral")
FLAT = ("disc", "ring", "gear", "spiral")
# Round things that roll across a face rather than face out of it (a scroll wheel lies in its well on its side).
ROLLS = re.compile(r"wheel|roller|drum|reel|spool|pulley|crown|axle|hinge", re.I)
FACING_TOLERANCE_DEG = 20


def round_axis(shape: str, size: list[float]) -> int:
    """The axis a round shape turns about, as the viewer reads it from size_mm (see web/src/blueprint/draw.ts)."""
    if shape in FLAT:
        return size.index(min(size))
    score = [abs(size[i] - sum(size[j] for j in range(3) if j != i) / 2) - abs(size[(i + 1) % 3] - size[(i + 2) % 3]) for i in range(3)]
    return score.index(max(score))


def _distinct_axis(p: dict[str, Any]) -> int:
    """The axis a part points along: a round part's own axis, or the box side that differs most from the other two
    (a switch's height, a module's shaft)."""
    return round_axis(p["shape"] if p["shape"] in ROUND else "cylinder", p["size_mm"])


def _quarter(k: int, deg: float) -> np.ndarray:
    """A turn about own axis k."""
    c, s = round(np.cos(np.radians(deg))), round(np.sin(np.radians(deg)))
    m = np.eye(3)
    i, j = (k + 1) % 3, (k + 2) % 3
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def _point_out(p: dict[str, Any], axis: int, plate: dict[str, Any], t: int, out: np.ndarray) -> bool:
    """Point the part's own `axis` along `out` (the plate's normal, outward) if it is more than the tolerance off:
    the plate's turn plus a quarter or half turn that carries the axis onto the plate's normal. Only the turn changes,
    never the size, so a built model turns as a whole. Records "faces" (the own axis that points out). True if turned."""
    own = np.array(pose.part_rotation(p.get("rotation_deg") or [0, 0, 0]))
    along = float(own[:, axis] @ out)
    if abs(along) >= np.cos(np.radians(FACING_TOLERANCE_DEG)):
        p["faces"] = ("+" if along > 0 else "-") + "xyz"[axis]
        return False
    frame = np.array(pose.part_rotation(plate.get("rotation_deg") or [0, 0, 0]))
    target = np.zeros(3)
    target[t] = 1.0 if float(frame[:, t] @ out) > 0 else -1.0
    # A turn (in the plate's own frame) taking the part's axis onto the plate's outward thin axis.
    turns = [np.eye(3)] + [_quarter(k, d) for k in range(3) for d in (90, -90, 180)]
    turn = next(q for q in turns if np.allclose(q[:, axis], target))
    p["rotation_deg"] = pose.euler_xyz((frame @ turn).tolist())
    p["faces"] = "+" + "xyz"[axis]
    return True


def face_out(anatomy: dict[str, Any]) -> list[str]:
    """Turn controls that sit on a plate (buttons, caps, sticks, dials, lenses) so they face out of it, and the
    mechanisms joined to them behind it (a thumbstick's module, a button's switch) so they point the same way, as the
    real ones do. Each part handled gets "faces": the own axis that points out (its cap, shaft or face goes that way).
    Parts are only turned, never resized, so modeled ones turn as a whole; things that roll across a face are left
    alone. Returns what changed."""
    parts = anatomy["parts"]
    by_id = {p["id"]: p for p in parts}
    # Thin outer parts are faces things mount on; round thin ones (a button, a ring) are the things mounted.
    plates = [p for p in parts if p["kind"] != "internal" and p["shape"] not in ROUND
              and sorted(p["size_mm"])[0] <= 0.35 * sorted(p["size_mm"])[1]]

    def mount(p: dict[str, Any], behind: float = 0.0) -> tuple[dict[str, Any], int, np.ndarray] | None:
        """The plate `p` sits on (or up to `behind` mm beneath): the plate, its thin axis, and its outward normal."""
        best = None
        for q in plates:
            # A part mounts on a face bigger than itself (a ring around a faceplate isn't mounted on it).
            if q is p or max(p["size_mm"]) > max(q["size_mm"]):
                continue
            frame = np.array(pose.part_rotation(q.get("rotation_deg") or [0, 0, 0]))
            local = frame.T @ (np.array(p["center_mm"], float) - np.array(q["center_mm"], float))
            t = int(np.argmin(q["size_mm"]))
            # On the plate: near its face, and within its outline across the face.
            if abs(local[t]) > max(p["size_mm"]) + q["size_mm"][t] + behind:
                continue
            if any(abs(local[i]) > q["size_mm"][i] / 2 * 1.1 for i in range(3) if i != t):
                continue
            # Out is the plate's side away from the object's centre (a part a little behind the plate's mid-plane
            # still faces out); a plate through the centre goes by the side the part is on.
            away = float(frame[:, t] @ np.array(q["center_mm"], float))
            side = away if abs(away) > 1.0 else float(local[t])
            if best is None or abs(local[t]) < best[0]:
                best = (abs(local[t]), q, t, frame[:, t] * (1.0 if side >= 0 else -1.0))
        return best and best[1:]

    turned, controls = [], {}
    for p in parts:
        if (p["kind"] != "exterior" or p["shape"] not in ROUND or p in plates
                or ROLLS.search(f"{p['id']} {p.get('name', '')}")):
            continue
        found = mount(p)
        if found is None:
            continue
        q, t, out = found
        controls[p["id"]] = (q, t, out)
        if _point_out(p, round_axis(p["shape"], p["size_mm"]), q, t, out):
            turned.append(f"{p['id']} now faces out of {q['id']}")
    # The mechanism a control works sits behind the face, pointing the same way (its outward side is the control's):
    # a part joined to it mechanically, or an internal part right beneath it that shares its name (thumbstick_cap and
    # thumbstick_module) or is joined to it at all.
    linked = {frozenset((c.get("from"), c.get("to"))): c.get("kind") for c in anatomy.get("connections", [])}
    for a, (q, t, out) in controls.items():
        control = by_id[a]
        for m in parts:
            # A mechanism is about its control's size; a board under every button belongs to none of them.
            if m["id"] in controls or m in plates or m is control or max(m["size_mm"]) > MECHANISM_SCALE * max(control["size_mm"]):
                continue
            kind = linked.get(frozenset((a, m["id"])))
            if kind != "mechanical":
                if m["kind"] != "internal" or not (kind or _shares_name(a, m["id"])) or not _beneath(m, control, out):
                    continue
            if mount(m, behind=max(control["size_mm"])) is None:
                continue
            axis = _distinct_axis(m)
            moved = _point_out(m, axis, q, t, out) | _seat_behind(m, axis, control, out)
            if moved:
                turned.append(f"{m['id']} now points out of {q['id']} with {a}")
    return turned


SEAT_OVERLAP = 0.3
MECHANISM_SCALE = 2.5


def _seat_behind(m: dict[str, Any], axis: int, control: dict[str, Any], out: np.ndarray) -> bool:
    """Stack the mechanism straight behind its control along the face's normal (the control stays where the photo
    showed it), overlapping a little so a shaft reaches into its cap. True if it moved."""
    c_axis = round_axis(control["shape"], control["size_mm"]) if control["shape"] in ROUND else int(np.argmin(control["size_mm"]))
    hc, hm = control["size_mm"][c_axis], m["size_mm"][axis]
    gap = hc / 2 + hm / 2 - SEAT_OVERLAP * min(hc, hm)
    target = [round(float(v), 1) for v in np.array(control["center_mm"], float) - out * gap]
    if math.dist(target, m["center_mm"]) < 1.0:
        return False
    m["center_mm"] = target
    return True


def _shares_name(a: str, b: str) -> bool:
    words = lambda s: {w for w in re.split(r"[_\W]+", s.lower()) if len(w) > 2}
    return bool(words(a) & words(b))


def _beneath(m: dict[str, Any], control: dict[str, Any], out: np.ndarray) -> bool:
    """Whether `m` sits behind `control` within its footprint: under it along the face's normal, not off to a side."""
    d = np.array(m["center_mm"], float) - np.array(control["center_mm"], float)
    depth_ = float(d @ out)
    across = float(np.linalg.norm(d - depth_ * out))
    return depth_ <= 3 and -depth_ <= max(control["size_mm"]) + max(m["size_mm"]) and across <= max(control["size_mm"]) / 2 + 3
