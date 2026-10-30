"""Mates: how parts fit together.

Instead of every part being placed on its own (and drifting), the modeler says how one part attaches to another
and this places it exactly, the way things are assembled: a cell inside its holder, a button on its switch, a screw
coaxial with its hole, two gears meshed. Only the moving part (`part`) moves; `to` stays. Parts are measured by
their boxes as turned by their rotation (see checks._bounds).
"""

from __future__ import annotations

from typing import Any

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
