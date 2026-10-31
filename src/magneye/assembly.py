"""Parts are solids that fit together like building blocks: no two may share space.

Where one part holds another (a shell around a board, a plate under a button, a ring around its LEDs, a caddy around
its cell), the holder gets a pocket the shape of the part it holds, with a little clearance: the part sits in a
groove rather than fusing into it. The pocket is recorded as a cut on the holder ("cuts": ids of the parts it holds)
and carved whenever the holder's model is built, so it follows the part if either moves. Where neither holds the other
(two boards, a cell and a motor), the smaller one is pushed out the shortest way until the two just touch.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from manifold3d import Manifold

from . import model as M

CLEARANCE_MM = 0.4
# Overlaps smaller than this share of the smaller part, or this volume, are contact, not fusion.
MIN_SHARE = 0.02
MIN_VOLUME_MM3 = 8.0
PUSH_ROUNDS = 8
GAP_MM = 0.3
# What a part is, by the last word of its id or name: these hold other parts, even inside the object.
HOLDER_WORDS = {"shell", "housing", "case", "casing", "body", "plate", "faceplate", "cover", "frame", "chassis",
                "bracket", "caddy", "holder", "tray", "cradle", "socket", "bay", "mount", "enclosure", "bezel",
                "handle", "base", "dock", "lid", "door", "panel", "deck"}
# An outer part holds an inner one this many times smaller than itself (a shell around a board), not a neighbour.
OUTER_HOLDS = 3.0


def _solid(part: dict[str, Any]) -> Manifold:
    """The part's own solid, as built (its model, or a box of its size)."""
    if part.get("model"):
        try:
            return M.build(part["model"])
        except M.RecipeError:
            pass
    return Manifold.cube(part["size_mm"], True)


def _placed(part: dict[str, Any], solid: Manifold) -> Manifold:
    return solid.transform(M.placement(part, solid))


def _grown(solid: Manifold, by: float) -> Manifold:
    """The solid enlarged by `by` mm on every side: itself merged with copies shifted that far along each axis (a
    true offset outward for any shape, holes included, unlike scaling, which moves a ring's inner edge the wrong way)."""
    out = solid
    for axis in range(3):
        for sign in (1, -1):
            step = [0.0, 0.0, 0.0]
            step[axis] = sign * by
            out = out + solid.translate(step)
    return out


def holder(a: dict[str, Any], b: dict[str, Any], va: float, vb: float) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(holder, held) when one of two overlapping parts holds the other, else None (they are peers). A part whose
    name says it holds things (a shell, plate, caddy...) holds one that doesn't; between two such, the bigger holds
    the smaller. Otherwise an outer part holds an inner one much smaller than itself (a shell around a board)."""
    named = lambda p: _kind_word(p) in HOLDER_WORDS
    if named(a) != named(b):
        return (a, b) if named(a) else (b, a)
    if named(a):
        return (a, b) if va >= vb else (b, a)
    big, small, vbig, vsmall = (a, b, va, vb) if va >= vb else (b, a, vb, va)
    if big["kind"] != "internal" and small["kind"] == "internal" and vbig >= OUTER_HOLDS * vsmall:
        return big, small
    return None


def _kind_word(p: dict[str, Any]) -> str:
    """What the part is: the last word of its id (tracking_ring_shell is a shell, ring_ribbon_cable a cable)."""
    words = [w for w in re.split(r"[_\W]+", p["id"].lower()) if w and not w.isdigit() and len(w) > 1]
    return words[-1] if words else ""


def _inside(point: list[float], solid: Manifold) -> bool:
    b = solid.bounding_box()
    return all(b[i] <= point[i] <= b[3 + i] for i in range(3))


def _push(mover: dict[str, Any], placed_mover: Manifold, other: Manifold) -> list[float]:
    """The shortest move (along one axis) that takes `mover` out of `other`, plus a hair."""
    inter = placed_mover ^ other
    if inter.is_empty():
        return [0.0, 0.0, 0.0]
    b = inter.bounding_box()
    depth = [b[3 + i] - b[i] for i in range(3)]
    axis = int(np.argmin(depth))
    mb, ob = placed_mover.bounding_box(), other.bounding_box()
    mid = lambda bb: (bb[axis] + bb[3 + axis]) / 2
    sign = 1.0 if mid(mb) >= mid(ob) else -1.0
    move = [0.0, 0.0, 0.0]
    move[axis] = sign * (depth[axis] + GAP_MM)
    return move


def resolve(anatomy: dict[str, Any]) -> list[str]:
    """Make the parts fit together without sharing space: holders get pockets for what they hold (recorded in
    "cuts"), peers are pushed apart. Mutates `anatomy`; returns what changed."""
    parts = anatomy["parts"]
    own = {p["id"]: _solid(p) for p in parts}
    changes: list[str] = []
    for p in parts:
        p.pop("cuts", None)
    for _ in range(PUSH_ROUNDS):
        placed = {p["id"]: _placed(p, own[p["id"]]) for p in parts}
        volume = {pid: s.volume() for pid, s in placed.items()}
        moved = False
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                sa, sb = placed[a["id"]], placed[b["id"]]
                ba, bb = sa.bounding_box(), sb.bounding_box()
                if any(ba[3 + k] <= bb[k] or bb[3 + k] <= ba[k] for k in range(3)):
                    continue
                shared = (sa ^ sb).volume()
                if shared < max(MIN_VOLUME_MM3, MIN_SHARE * min(volume[a["id"]], volume[b["id"]])):
                    continue
                pair = holder(a, b, volume[a["id"]], volume[b["id"]])
                # A holder holds what sits inside it; anything else it overlaps is a neighbour to move aside. Two
                # outer parts, both placed by the photo, never move: the bigger takes a groove for the smaller.
                if pair and not _inside(pair[1]["center_mm"], placed[pair[0]["id"]]):
                    pair = None
                if not pair and a["kind"] != "internal" and b["kind"] != "internal":
                    pair = (a, b) if volume[a["id"]] >= volume[b["id"]] else (b, a)
                if pair:
                    host, held = pair
                    cuts = host.setdefault("cuts", [])
                    if held["id"] not in cuts:
                        cuts.append(held["id"])
                        changes.append(f"{host['id']} holds {held['id']} in a pocket")
                    continue
                # The photo placed the outer parts, so an inner one gives way; between two of a kind, the smaller.
                if (a["kind"] == "internal") != (b["kind"] == "internal"):
                    mover = a if a["kind"] == "internal" else b
                else:
                    mover = a if volume[a["id"]] <= volume[b["id"]] else b
                other = b if mover is a else a
                step = _push(mover, placed[mover["id"]], placed[other["id"]])
                if any(step):
                    mover["center_mm"] = [round(c + d, 1) for c, d in zip(mover["center_mm"], step)]
                    placed[mover["id"]] = _placed(mover, own[mover["id"]])
                    changes.append(f"{mover['id']} moved {max(map(abs, step)):.0f} mm out of {other['id']}")
                    moved = True
        if not moved:
            break
    # Whatever pushing couldn't separate (parts packed on every side) is settled with a groove in the bigger part,
    # so in the end no two parts share space.
    placed = {p["id"]: _placed(p, own[p["id"]]) for p in parts}
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if b["id"] in a.get("cuts", []) or a["id"] in b.get("cuts", []):
                continue
            sa, sb = placed[a["id"]], placed[b["id"]]
            if (sa ^ sb).volume() < MIN_VOLUME_MM3:
                continue
            host, held = (a, b) if sa.volume() >= sb.volume() else (b, a)
            host.setdefault("cuts", []).append(held["id"])
            changes.append(f"{host['id']} holds {held['id']} in a pocket")
    return list(dict.fromkeys(changes))


def build_part(part: dict[str, Any], parts: dict[str, dict[str, Any]]) -> tuple[Manifold, Manifold]:
    """(solid with its pockets carved, solid as built). Both are in the part's own recipe frame; the uncut one gives
    the bounds the viewer scales by, so a pocket that opens through a face doesn't resize the part."""
    solid = M.build(part["model"]) if part.get("model") else Manifold.cube(part["size_mm"], True)
    cut = solid
    if part.get("cuts"):
        place = M.placement(part, solid)
        a, t = place[:, :3], place[:, 3]
        inv = np.linalg.inv(a)
        back = np.hstack([inv, (-inv @ t).reshape(3, 1)])
        for gid in part["cuts"]:
            guest = parts.get(gid)
            if guest is None:
                continue
            try:
                pocket = _grown(_placed(guest, _solid(guest)), CLEARANCE_MM).transform(back)
                carved = cut - pocket
            except Exception:  # a degenerate pocket is skipped rather than failing the part
                continue
            if not carved.is_empty() and carved.volume() > 0.05 * solid.volume():
                cut = carved
    return cut, solid
