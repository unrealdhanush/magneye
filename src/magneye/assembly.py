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
from . import pose

CLEARANCE_MM = 0.4
# Overlaps smaller than this share of the smaller part, or this volume, are contact, not fusion.
MIN_SHARE = 0.02
MIN_VOLUME_MM3 = 8.0
PUSH_ROUNDS = 8
# A part is pushed at most this far at once, and this far in all; a collision that needs more is settled with a
# groove, so a deep overlap can't shove a neighbour across the object.
MAX_PUSH_MM = 8.0
MAX_TRAVEL_MM = 15.0
GAP_MM = 0.3
# What a part is, by the last word of its id or name: these hold other parts, even inside the object.
HOLDER_WORDS = {"shell", "housing", "case", "casing", "body", "plate", "faceplate", "cover", "frame", "chassis",
                "bracket", "caddy", "holder", "tray", "cradle", "socket", "bay", "mount", "enclosure", "bezel",
                "handle", "base", "dock", "lid", "door", "panel", "deck"}
# An outer part holds an inner one this many times smaller than itself (a shell around a board), not a neighbour.
OUTER_HOLDS = 3.0
# An inner part may poke out of the object by this share of its volume (a connector at a port); beyond it, it is
# nudged inward and, if that isn't enough, slimmed along the sides that stick out (a board drafted wider than its grip).
MAX_OUTSIDE = 0.01
# Inner parts keep this much wall between them and the outside, or the pocket carved for them opens through it.
WALL_MM = 1.5
NUDGE_MM = 1.0
MAX_NUDGE_MM = 8.0
SLIM_STEP = 0.94
MIN_SCALE = 0.6
CONTAIN_STEPS = 30


def _solid(part: dict[str, Any]) -> Manifold:
    """The part's own solid as the viewer draws it: its model, else its components, else its plain shape."""
    if part.get("model"):
        try:
            return M.build(part["model"])
        except M.RecipeError:
            pass
    comps = [c for c in part.get("components") or [] if len(c.get("size_mm") or []) == 3 and len(c.get("offset_mm") or []) == 3]
    if comps:
        out = None
        for c in comps:
            turn = np.array(pose.part_rotation(c.get("rotation_deg") or [0, 0, 0]))
            piece = M.primitive(c.get("shape", "box"), c["size_mm"]).transform(np.hstack([turn, np.array(c["offset_mm"], float).reshape(3, 1)]))
            out = piece if out is None else out + piece
        if out is not None and not out.is_empty():
            return out
    return M.primitive(part.get("shape", "box"), part["size_mm"])


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


ROUND = ("cylinder", "disc", "ring", "cone", "capsule", "coil", "gear", "spiral", "torus")
# Outer skins: thin covering parts. They hold the controls set into them, never inner parts, which belong behind.
SKIN_WORDS = {"plate", "faceplate", "cover", "panel", "lid", "door", "deck", "bezel"}


def holder(a: dict[str, Any], b: dict[str, Any], va: float, vb: float) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(holder, held) when one of two overlapping parts holds the other, else None (they are peers). A part whose
    name says it holds things (a shell, plate, caddy...) holds one that doesn't; between two such, the bigger holds
    the smaller. Otherwise an outer part holds an inner one much smaller than itself (a shell around a board)."""
    if (_skin(a) and b["kind"] == "internal") or (_skin(b) and a["kind"] == "internal"):
        return None
    named = lambda p: _kind_word(p) in HOLDER_WORDS
    if named(a) != named(b):
        return (a, b) if named(a) else (b, a)
    if named(a):
        return (a, b) if va >= vb else (b, a)
    big, small, vbig, vsmall = (a, b, va, vb) if va >= vb else (b, a, vb, va)
    if big["kind"] != "internal" and small["kind"] == "internal" and vbig >= OUTER_HOLDS * vsmall:
        return big, small
    return None


def _skin(p: dict[str, Any]) -> bool:
    """An outer skin: named as one, or an outer part that is thin and flat (a ring is thin too, but it's a band)."""
    return _kind_word(p) in SKIN_WORDS or (p["kind"] != "internal" and p["shape"] not in ROUND
                                           and sorted(p["size_mm"])[0] <= 0.35 * sorted(p["size_mm"])[1])


def _kind_word(p: dict[str, Any]) -> str:
    """What the part is: the last word of its id (tracking_ring_shell is a shell, ring_ribbon_cable a cable)."""
    words = [w for w in re.split(r"[_\W]+", p["id"].lower()) if w and not w.isdigit() and len(w) > 1]
    return words[-1] if words else ""


def _inside(point: list[float], solid: Manifold) -> bool:
    b = solid.bounding_box()
    return all(b[i] <= point[i] <= b[3 + i] for i in range(3))


def _normal(p: dict[str, Any]) -> np.ndarray:
    """A part's thinnest own axis in the object frame: the way a plate or board faces."""
    frame = np.array(pose.part_rotation(p.get("rotation_deg") or [0, 0, 0]))
    return frame[:, int(np.argmin(p["size_mm"]))]


def _pushes(mover: dict[str, Any], other: dict[str, Any], placed_mover: Manifold, placed_other: Manifold,
            inward: bool) -> list[list[float]]:
    """The moves that take the mover clear of the other part, shortest first, each found by sliding the real solid
    (boxes overstate a tilted plate's depth): along the object's axes and along either part's facing direction (a
    board slides off a plate along their normals). `inward` keeps only moves toward the object's centre, so an inner
    part caught in an outer skin goes behind it, never out through it."""
    directions = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)]
    directions += [n * s for n in (_normal(mover), _normal(other)) for s in (1, -1)]
    centre = np.array(mover["center_mm"], float)
    moves = []
    for d in directions:
        if inward and float(d @ -centre) < 0:
            continue
        def clear(dist: float) -> bool:
            return (placed_mover.translate((d * dist).tolist()) ^ placed_other).volume() < MIN_VOLUME_MM3
        if not clear(MAX_PUSH_MM):
            continue
        lo, hi = 0.0, MAX_PUSH_MM
        for _ in range(7):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if clear(mid) else (mid, hi)
        moves.append([round(float(v), 2) for v in d * (hi + GAP_MM)])
    return sorted(moves, key=lambda s: float(np.linalg.norm(s)))


def _container(mover: dict[str, Any], parts: list[dict[str, Any]], placed: dict[str, Manifold]) -> Manifold | None:
    """What a part must stay inside when it's pushed: the outer part around it (its holder), if any."""
    around = [placed[p["id"]] for p in parts if p is not mover and p["kind"] != "internal"
              and _inside(mover["center_mm"], placed[p["id"]])]
    return max(around, key=lambda s: s.volume()) if around else None


def _band(p: dict[str, Any]) -> bool:
    """A ring, halo or loop: its hull would fill the hole in the middle, which is open air."""
    return p["shape"] in ("ring", "torus") or _kind_word(p) in ("ring", "halo", "loop", "hoop")


# Turns a part's own axis 0, 1 or 2 onto z (proper rotations, so nothing is mirrored).
_ONTO_Z = {0: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float), 1: np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float),
           2: np.eye(3)}


def _filled(p: dict[str, Any], solid: Manifold) -> Manifold:
    """The space a placed outer part closes in: its hull (a shell's hull fills its hollow). A band's hull would also
    fill the hole in its middle, so a band is its outline seen along its thin axis, extruded back through its
    thickness: a ring's annulus, which holds what runs inside its tube but not what sits in the hole."""
    if not _band(p):
        return solid.hull()
    own = _solid(p)
    turn = _ONTO_Z[int(np.argmin(p["size_mm"]))]
    flat = own.transform(np.hstack([turn, np.zeros((3, 1))]))
    b = flat.bounding_box()
    slab = Manifold.extrude(flat.project(), b[5] - b[2]).translate([0, 0, b[2]])
    band = slab.transform(np.hstack([turn.T, np.zeros((3, 1))])) ^ own.hull()
    return band.transform(M.placement(p, own))


def _envelope(parts: list[dict[str, Any]], placed: dict[str, Manifold]) -> Manifold | None:
    """The space the outer parts close in: the hull of each (a shell's hull fills its hollow), but a ring only as
    itself. An inner part outside it shows through the outside of the model."""
    hulls = []
    for p in parts:
        if p["kind"] != "exterior":
            continue
        hulls.append(_filled(p, placed[p["id"]]))
    if not hulls:
        return None
    out = hulls[0]
    for h in hulls[1:]:
        out = out + h
    return out


def _contain(parts: list[dict[str, Any]], own: dict[str, Manifold], pinned: frozenset[str] | set[str],
             drafted: dict[str, tuple[list[float], list[float]]]) -> list[str]:
    """Bring inner parts that stick out of the object back inside it: the move or slimming that leaves least of the
    part outside, a step at a time, preferring moves, until at most MAX_OUTSIDE is out. Moves and slimming are bounded
    from the part as `drafted` (centre, size). Pinned parts stay as placed."""
    placed = {p["id"]: _placed(p, own[p["id"]]) for p in parts}
    env = _envelope(parts, placed)
    if env is None:
        return []
    changes = []
    for p in parts:
        if p["kind"] != "internal" or p["id"] in pinned:
            continue
        verts = np.asarray(own[p["id"]].hull().to_mesh().vert_properties)[:, :3]
        def outside(centre: list[float], size: list[float]) -> float:
            # The part's hull, grown by the wall: a hull of points is far quicker than booleans on the real solid.
            place = M.placement({**p, "center_mm": centre, "size_mm": size}, own[p["id"]])
            at = verts @ place[:, :3].T + place[:, 3]
            s = Manifold.hull_points(np.vstack([at + d for d in np.vstack([np.eye(3), -np.eye(3)]) * WALL_MM]))
            return (s - env).volume() / max(s.volume(), 1e-6)
        start_c, start_s = drafted[p["id"]]
        centre, size = list(p["center_mm"]), list(p["size_mm"])
        first_c, first_s = centre, size
        share = before = outside(centre, size)
        for _ in range(CONTAIN_STEPS):
            if share <= MAX_OUTSIDE:
                break
            tries = []
            for axis in range(3):
                for sign in (1, -1):
                    c = list(centre)
                    c[axis] = round(c[axis] + sign * NUDGE_MM, 1)
                    if float(np.linalg.norm(np.subtract(c, start_c))) <= MAX_NUDGE_MM:
                        tries.append((c, size))
                if size[axis] * SLIM_STEP >= MIN_SCALE * start_s[axis]:
                    s = list(size)
                    s[axis] = round(s[axis] * SLIM_STEP, 1)
                    tries.append((centre, s))
            # A move keeps the part as drafted, so a slimming has to do clearly better to win.
            scored = [(outside(*t), t) for t in tries]
            best, (c, s) = min(scored, key=lambda x: x[0] + (0.01 if x[1][1] is not size else 0))
            if best >= share - 1e-3:
                break
            centre, size, share = c, s, best
        if share < before - 1e-3:
            p["center_mm"], p["size_mm"] = centre, size
            moved = float(np.linalg.norm(np.subtract(centre, first_c)))
            changes.append(f"{p['id']} brought inside the object ({100 * before:.0f}% to {100 * share:.0f}% outside"
                           + (f"; moved {moved:.0f} mm" if moved >= 0.5 else "")
                           + (f"; slimmed to {' x '.join(f'{v:g}' for v in size)} mm" if size != first_s else "") + ")")
    return changes


def _host_recipe(host: dict[str, Any]) -> dict[str, Any]:
    """A host part's form as a recipe: its model, or the plain shape the viewer draws for it."""
    if host.get("model"):
        return host["model"]
    size, shape = host["size_mm"], host.get("shape")
    if shape == "rounded_box":
        return {"op": "box", "size": size, "radius": round(0.15 * min(size), 2)}
    if shape in ("sphere", "capsule"):
        return {"op": "ellipsoid", "size": size}
    return {"op": "box", "size": size}


def _affine(m: np.ndarray) -> np.ndarray:
    return np.vstack([m, [0, 0, 0, 1]])


def mould_recipe(part: dict[str, Any], host: dict[str, Any], thickness: float, region: dict[str, Any]) -> dict[str, Any]:
    """The recipe for a part cut from the outside of its host: the host's outer `thickness` mm, within `region` (a
    node in the part's own frame). A cover, door, faceplate or grip pad made this way follows every curve of the body
    it sits on, and the body takes a matching pocket when the parts are fitted, so it sits flush."""
    recipe = _host_recipe(host)
    own = M.build(recipe)
    place = M.placement(host, own)
    turn = np.array(pose.part_rotation(part.get("rotation_deg") or [0, 0, 0]))
    centre = np.array(part["center_mm"], float)
    into_part = _affine(np.hstack([turn.T, (-turn.T @ centre).reshape(3, 1)]))
    # The host shrunk by the thickness about the middle of its placed box: what stays the host's own inside.
    b = own.transform(place).bounding_box()
    mid = np.array([(b[i] + b[3 + i]) / 2 for i in range(3)])
    span = np.array([max(b[3 + i] - b[i], 1e-6) for i in range(3)])
    k = np.maximum(span - 2 * thickness, 0.2 * span) / span
    shrink = _affine(np.hstack([np.diag(k), (mid - k * mid).reshape(3, 1)]))
    outer = (into_part @ _affine(place))[:3].round(5).tolist()
    inner = (into_part @ shrink @ _affine(place))[:3].round(5).tolist()
    return {"op": "difference", "of": [
        {"op": "intersection", "of": [{"op": "transform", "matrix": outer, "of": recipe}, region]},
        {"op": "transform", "matrix": inner, "of": recipe}]}


def apply_mould(part: dict[str, Any], by_id: dict[str, dict[str, Any]], spec: dict[str, Any] | None = None) -> str | None:
    """(Re)build a moulded part from its spec ({"on", "thickness", "region"}) against where its host is now. Returns
    an error to report, or None."""
    spec = spec or part.get("mould")
    if not spec:
        return None
    host = by_id.get(str(spec.get("on")))
    if host is None or host is part:
        return f"mould: no part {spec.get('on')!r} to mould onto"
    try:
        thickness = float(spec.get("thickness") or 2.0)
    except (TypeError, ValueError):
        return "mould: thickness must be a number"
    region = spec.get("region") if isinstance(spec.get("region"), dict) else {"op": "box", "size": part["size_mm"]}
    recipe = mould_recipe(part, host, max(thickness, 0.4), region)
    try:
        solid = M.build(recipe)
    except M.RecipeError as e:
        return f"mould: {e}; move the region onto {host['id']}'s surface"
    b = solid.bounding_box()
    part["mould"] = {"on": host["id"], "thickness": thickness, "region": region}
    part["model"] = recipe
    part["size_mm"] = [round(b[3 + i] - b[i], 2) for i in range(3)]
    return None


def resolve(anatomy: dict[str, Any], pinned: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Make the parts fit together without sharing space: holders get pockets for what they hold (recorded in
    "cuts"), peers are pushed apart. `pinned` parts (the ones the user just placed) never move: what collides with
    them gives way, or the two get a groove. Mutates `anatomy`; returns what changed."""
    parts = anatomy["parts"]
    by_id = {p["id"]: p for p in parts}
    # Moulded parts follow their host wherever it is now.
    for p in parts:
        if p.get("mould"):
            apply_mould(p, by_id)
    own = {p["id"]: _solid(p) for p in parts}
    changes: list[str] = []
    for p in parts:
        p.pop("cuts", None)
    # Inner parts are brought inside the object first, so the pushes start from a layout that fits in it.
    drafted = {p["id"]: (list(p["center_mm"]), list(p["size_mm"])) for p in parts}
    changes += _contain(parts, own, pinned, drafted)
    start = {p["id"]: list(p["center_mm"]) for p in parts}
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
                # A part the user just placed stays put: an inner neighbour gives way instead, or else a groove.
                if mover["id"] in pinned:
                    mover, other = other, mover
                    if mover["id"] in pinned or mover["kind"] != "internal":
                        continue
                # The shortest push that keeps an inner part inside the part around it (a cell stays in its bay);
                # if none does, the two are left for a groove.
                box = _container(mover, parts, placed) if mover["kind"] == "internal" else None
                def allowed(s: list[float]) -> bool:
                    to = [c + d for c, d in zip(mover["center_mm"], s)]
                    return (float(np.linalg.norm(s)) <= MAX_PUSH_MM + GAP_MM and float(np.linalg.norm(np.subtract(to, start[mover["id"]]))) <= MAX_TRAVEL_MM
                            and (box is None or _inside(to, box)))
                inward = mover["kind"] == "internal" and other["kind"] != "internal"
                step = next((s for s in _pushes(mover, other, placed[mover["id"]], placed[other["id"]], inward) if allowed(s)), None)
                if step:
                    mover["center_mm"] = [round(c + d, 1) for c, d in zip(mover["center_mm"], step)]
                    placed[mover["id"]] = _placed(mover, own[mover["id"]])
                    changes.append(f"{mover['id']} moved {float(np.linalg.norm(step)):.0f} mm out of {other['id']}")
                    moved = True
        if not moved:
            break
    # A push can nudge a part back out through a wall; it is brought in again before the grooves are cut.
    changes += _contain(parts, own, pinned, drafted)
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
            # The rest fit around what the user just placed: an unpinned part takes the groove for a pinned one. An
            # outer skin is never cut for an inner part (its surface would open); the inner part takes the groove.
            if (a["id"] in pinned) != (b["id"] in pinned):
                host, held = (b, a) if a["id"] in pinned else (a, b)
            elif _skin(a) and b["kind"] == "internal":
                host, held = b, a
            elif _skin(b) and a["kind"] == "internal":
                host, held = a, b
            else:
                host, held = (a, b) if sa.volume() >= sb.volume() else (b, a)
            host.setdefault("cuts", []).append(held["id"])
            changes.append(f"{host['id']} holds {held['id']} in a pocket")
    return list(dict.fromkeys(changes))


def _under_skin(solid: Manifold, part: dict[str, Any] | None = None) -> Manifold:
    """The inside of an outer part: what it closes in (see _filled) shrunk by WALL_MM on every side (scaling about
    its centre, close to a true inward offset for these rounded shapes)."""
    h = _filled(part, solid) if part else solid.hull()
    b = h.bounding_box()
    centre = [(b[i] + b[3 + i]) / 2 for i in range(3)]
    span = [max(b[3 + i] - b[i], 1e-6) for i in range(3)]
    factor = [max(v - 2 * WALL_MM, 0.5 * v) / v for v in span]
    return h.translate([-c for c in centre]).scale(factor).translate(centre)


def interior(parts: list[dict[str, Any]]) -> Manifold | None:
    """The space under the object's skin: where inner parts may show, so none pokes out through an outer surface."""
    inside = [_under_skin(_placed(p, _solid(p)), p) for p in parts if p["kind"] == "exterior"]
    if not inside:
        return None
    out = inside[0]
    for h in inside[1:]:
        out = out + h
    return out


def build_part(part: dict[str, Any], parts: dict[str, dict[str, Any]],
               inside: Manifold | None = None) -> tuple[Manifold, Manifold]:
    """(solid with its pockets carved, solid as built). Both are in the part's own recipe frame; the uncut one gives
    the bounds the viewer scales by, so a pocket that opens through a face doesn't resize the part. With `inside`
    (the object's interior), an inner part is trimmed to it and an outer part's pockets for inner parts stop under
    its skin, so nothing inside shows through the outside."""
    solid = M.build(part["model"]) if part.get("model") else _solid(part)
    cut = solid
    place = M.placement(part, solid)
    a, t = place[:, :3], place[:, 3]
    inv = np.linalg.inv(a)
    back = np.hstack([inv, (-inv @ t).reshape(3, 1)])
    skin = _under_skin(solid.transform(place)).transform(back) if part["kind"] == "exterior" and inside else None
    for gid in part.get("cuts") or []:
        guest = parts.get(gid)
        if guest is None:
            continue
        try:
            pocket = _grown(_placed(guest, _solid(guest)), CLEARANCE_MM).transform(back)
            if skin is not None and guest["kind"] == "internal":
                pocket = pocket ^ skin
            carved = cut - pocket
        except Exception:  # a degenerate pocket is skipped rather than failing the part
            continue
        if not carved.is_empty() and carved.volume() > 0.05 * solid.volume():
            cut = carved
    if inside is not None and part["kind"] == "internal":
        # Trimmed only when it is mostly inside: a part drafted to sit out in the open stays whole.
        trimmed = cut ^ inside.transform(back)
        if not trimmed.is_empty() and trimmed.volume() >= 0.6 * cut.volume():
            cut = trimmed
    return cut, solid
