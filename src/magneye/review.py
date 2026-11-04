"""Reviewer: Muse looks at the finished model exactly as the viewer will show it (pockets carved, parts fitted
together) from every side, next to the real photos, and fixes its own design.

Software can say where the model is wrong (an inner part showing through, a part trimmed to fit, a part floating
free); only the model can say why and what the real object does instead (the shell is too narrow, the housing under
the faceplate is missing, the board was drafted too big). So the renders mark those flaws for it, and it edits the
design until the renders look like the real thing.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import numpy as np
from manifold3d import Manifold
from PIL import Image, ImageDraw

from . import anatomy as A
from . import assembly, meta, pose
from . import model as M
from .agent import Emit, Tool, ToolResult, items, one_at_a_time, params, run_loop

TILE = 360
FLAW = np.array([255, 40, 200], np.float32)
# An inner part covering less than this share of the object's outline in every view doesn't count as showing.
MIN_SHOWING = 0.004
# A part the fitting trimmed by more than this share was drafted bigger than the space for it.
MIN_TRIMMED = 0.08
# A part SAM saw in the photo that shows less than this share of itself from the photo's viewpoint is hidden behind
# another part in the model: the parts are in the wrong order front to back.
MIN_IN_VIEW = 0.35
# An outer part the photo doesn't show that the model shows this openly (this share of itself, this share of the
# view) from the photo's viewpoint may be drafted on the wrong side.
SHOWN_IN_VIEW = 0.6
MIN_SHOWN = 0.015
PART_DEFAULTS: dict[str, Any] = {
    "name": "New part", "kind": "exterior", "function": "", "material": "plastic", "shape": "rounded_box",
    "size_mm": [10, 10, 10], "center_mm": [0, 0, 0], "rotation_deg": [0, 0, 0], "layer": 0, "sam_prompt": None,
    "confidence": 0.6, "basis": "inferred",
}

REVIEW_SYSTEM = f"""You are Magneye's reviewer. The 3D model of the object in the photo is built; before the user sees it, look at it the way they will (assembled, from every side) and fix what makes it unlike the real object.

The renders show the finished model: outer parts in their colours, and every internal part that can be seen from outside painted magenta and labelled. In the real object nothing inside shows: every magenta patch is a flaw. Look just as hard for what the checks can't measure: an outer part poking out through another (a handle's end showing through a faceplate), a cover or panel standing off the body instead of flush in it, a trigger or button floating free, shapes and proportions unlike the photo. Fix each at its cause, using what you know about the real object (the research, the teardown and its photos):
- The shell is too small or the wrong shape for what it holds: make it the real size (place_parts size_mm) or give it the real form (model_parts).
- The real object has a housing, cover or body section the model lacks (a top housing under a faceplate, a back cover, the body joining a handle to a head): add it with add_part, sized and placed to close the gap, with a recipe in the same call when its shape matters.
- The internal part is drafted too big or in the wrong place: give it its real size and position.
A part listed as trimmed was cut down to fit inside the shell: the same causes apply. A part listed as floating doesn't touch the body: in the real object every outer part is joined to it (a ring is moulded to the head, a button sits in its plate, the head runs into the grip), so close the gap the way the real object does: move it, extend it, or reshape the part between. A part listed as hidden is in plain sight in the photo but covered by another part from the photo's viewpoint: the two are in the wrong order front to back (a ring drafted in front of the buttons it arcs behind), so move or turn the one that is wrong until the photo's view of the model shows what the photo shows. Parts listed as reaching past the object make the model bigger than the real one, whose size was measured from the photo: shorten or move them back inside it. You also get how far each control (a button, stick, dial or key) stands out of the body under it: make it stand out as far as the real one does, set into its surface rather than sitting on top of it. Also fix outer parts that float apart from the body or stick out where the real object is smooth, and compare the whole model with the photos: at a glance it should read as the same object, with the same proportions and the same smooth, moulded forms. Give every outer part you add a recipe in its real shape (a loft through its cross-sections, or a blend, so it is rounded, tapers and joins its neighbours without steps); without one it is a plain rounded box. Reshape blocky outer parts the same way with model_parts.

Positions and sizes are millimetres in the object's own frame (x its right, y up, z out of its front); a part's size_mm is in its own frame before its rotation_deg. After your edits the parts are fitted together again (what you placed stays put, the rest gives way) and you get the new renders.

Tools:
- place_parts(updates): change center_mm, size_mm and rotation_deg of parts.
- model_parts(models): give parts a new geometry recipe.
- add_part(part, recipe): add a missing part, with an optional recipe.
- remove_part(id): remove a part that doesn't exist in the real object.
- finish(summary): when the renders look like the real object, or nothing more can be fixed.

{M.OPS_DOC}

Work in at most three turns: write down every flaw you see and its cause, fix them all in one turn, check the new renders, and finish."""


def assembled(anatomy: dict[str, Any]) -> tuple[dict[str, Manifold], list[str]]:
    """Every part as the viewer shows it, placed in the object (pockets carved, inner parts trimmed to the inside),
    and which parts the trimming cut down by much."""
    by_id = {p["id"]: p for p in anatomy["parts"]}
    inside = assembly.interior(anatomy["parts"])
    out, trimmed = {}, []
    for p in anatomy["parts"]:
        try:
            # The viewer draws a part with no recipe and no pockets as its plain primitive, untrimmed.
            cut, frame = (assembly.build_part(p, by_id, inside) if p.get("model") or p.get("cuts")
                          else (assembly._solid(p),) * 2)
        except Exception:  # a recipe that no longer builds is shown as its plain shape
            cut = frame = M.primitive(p.get("shape", "box"), p["size_mm"])
        out[p["id"]] = cut.transform(M.placement(p, frame))
        if p["kind"] == "internal" and inside is not None:
            whole = assembly._placed(p, frame).volume()
            if whole > 0 and 1 - out[p["id"]].volume() / whole > MIN_TRIMMED and not p.get("cuts"):
                trimmed.append(f"{p['id']} was trimmed by {100 * (1 - out[p['id']].volume() / whole):.0f}% to fit inside the shell")
    return out, trimmed


def _draw(anatomy: dict[str, Any], solids: dict[str, Manifold], view: tuple[float, float, float],
          size: int = TILE, look_for: frozenset[str] = frozenset()
          ) -> tuple[Image.Image, dict[str, int], int, dict[str, tuple[float, float]], dict[str, tuple[float, str, int]]]:
    """A shaded render with internal parts that show painted magenta: (image, visible pixels per internal part,
    pixels of the whole object, where each showing part's pixels centre, and for each part in `look_for` the share
    of its outline in view, the part covering most of the rest, and its pixels in view)."""
    camera = dict(zip(("azimuth_deg", "elevation_deg", "roll_deg"), view))
    rot = np.array(pose.rotation({"object": {"camera": camera}}), dtype=np.float32)
    tris, cols, owners = [], [], []
    order = [p for p in anatomy["parts"] if p["id"] in solids]
    # Small triangles, so drawing them back to front gets the depth right: a board made of two big triangles
    # would otherwise sort behind the shell it pokes out of.
    span = max((max(b[3 + i] - b[i] for i in range(3)) for b in (solids[p["id"]].bounding_box() for p in order)), default=1.0)
    for i, p in enumerate(order):
        v, t = M.arrays(solids[p["id"]].refine_to_length(span / 25))
        if not len(t):
            continue
        tris.append((v @ rot.T)[t])
        colour = FLAW if p["kind"] == "internal" else np.array(M.COLOUR.get(p["material"], M.COLOUR["other"]), np.float32)
        cols.append(np.repeat(colour[None], len(t), axis=0))
        owners.append(np.full(len(t), i + 1))
    im = Image.new("RGB", (size, size), (21, 24, 31))
    if not tris:
        return im, {}, 0, {}, {}
    tri, col, own = np.concatenate(tris), np.concatenate(cols), np.concatenate(owners)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    keep = normal[:, 2] > 0
    tri, col, own, normal = tri[keep], col[keep], own[keep], normal[keep]
    light = np.array([0.35, 0.55, 0.75], np.float32)
    light /= np.linalg.norm(light)
    col = np.clip(col * (0.35 + 0.65 * np.clip(normal @ light, 0, 1))[:, None], 0, 255)
    lo, hi = tri[..., :2].reshape(-1, 2).min(axis=0), tri[..., :2].reshape(-1, 2).max(axis=0)
    k = (size - 40) / max(float((hi - lo).max()), 1e-6)
    xy = np.empty_like(tri[..., :2])
    xy[..., 0] = 20 + (tri[..., 0] - lo[0]) * k + (size - 40 - (hi[0] - lo[0]) * k) / 2
    xy[..., 1] = 20 + (hi[1] - tri[..., 1]) * k + (size - 40 - (hi[1] - lo[1]) * k) / 2
    ids = Image.new("I", (size, size), 0)
    draw, mark = ImageDraw.Draw(im), ImageDraw.Draw(ids)
    for i in np.argsort(tri[:, :, 2].mean(axis=1)):
        poly = [tuple(p) for p in xy[i]]
        draw.polygon(poly, fill=tuple(int(c) for c in col[i]))
        mark.polygon(poly, fill=int(own[i]))
    grid = np.asarray(ids)
    total = int((grid > 0).sum())
    showing, where = {}, {}
    for i, p in enumerate(order):
        if p["kind"] != "internal":
            continue
        ys, xs = np.nonzero(grid == i + 1)
        if len(xs):
            showing[p["id"]] = len(xs)
            where[p["id"]] = (float(xs.mean()), float(ys.mean()))
    cover = {}
    for i, p in enumerate(order):
        if p["id"] not in look_for:
            continue
        alone = Image.new("1", (size, size), 0)
        pen = ImageDraw.Draw(alone)
        for t in xy[own == i + 1]:
            pen.polygon([tuple(q) for q in t], fill=1)
        outline = np.asarray(alone, bool)
        if outline.sum() < 20:
            continue
        under = grid[outline]
        others = under[(under != i + 1) & (under > 0)]
        top = order[int(np.bincount(others).argmax()) - 1]["id"] if len(others) else ""
        cover[p["id"]] = (float((under == i + 1).mean()), top, int((under == i + 1).sum()))
    return im, showing, total, where, cover


TOUCH_MM = 1.5


def _surface_points(solid: Manifold, rng: np.random.Generator, most: int = 3000) -> np.ndarray:
    v, _ = M.arrays(solid.refine_to_length(2.0))
    return v if len(v) <= most else v[rng.choice(len(v), most, replace=False)]


def _floating(anatomy: dict[str, Any], solids: dict[str, Manifold]) -> list[str]:
    """Outer parts that don't touch the main body (the largest group of outer parts that touch each other), with how
    far they are from it: a ring hanging in the air, a button hovering over its plate, a head apart from its grip."""
    outer = [p for p in anatomy["parts"] if p["kind"] == "exterior" and p["id"] in solids]
    if len(outer) < 2:
        return []
    rng = np.random.default_rng(0)
    pts = {p["id"]: _surface_points(solids[p["id"]], rng) for p in outer}
    boxes = {p["id"]: np.array(solids[p["id"]].bounding_box()) for p in outer}

    def gap(a: str, b: str, reach: float) -> float:
        ba, bb = boxes[a], boxes[b]
        apart = np.maximum(0, np.maximum(ba[:3] - bb[3:], bb[:3] - ba[3:]))
        if np.linalg.norm(apart) > reach:
            return float("inf")
        best = float("inf")
        for chunk in np.array_split(pts[a], max(1, len(pts[a]) // 500)):
            d = np.sqrt(((chunk[:, None, :] - pts[b][None, :, :]) ** 2).sum(-1)).min()
            best = min(best, float(d))
        return best
    ids = [p["id"] for p in outer]
    group = {i: i for i in ids}

    def root(i: str) -> str:
        while group[i] != i:
            i = group[i]
        return i
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            if root(a) != root(b) and gap(a, b, TOUCH_MM) <= TOUCH_MM:
                group[root(a)] = root(b)
    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(root(i), []).append(i)
    body = max(members.values(), key=lambda g: sum(solids[i].volume() for i in g))
    out = []
    for g in members.values():
        if g is body:
            continue
        near = min(((gap(a, b, 60), a, b) for a in g for b in body), key=lambda x: x[0])
        what = " and ".join(g[:3]) + (" (together)" if len(g) > 1 else "")
        out.append(f"{what} floats {near[0]:.0f} mm from the body (nearest: {near[1]} to {near[2]})" if near[0] < 60
                   else f"{what} is far from the body")
    return out


def _proud(anatomy: dict[str, Any], solids: dict[str, Manifold]) -> list[str]:
    """How far each control (an outer part facing out of a surface: a button, a stick, a dial) stands out of the body
    beneath it, along the way it faces: a button drafted 10 mm in front of its plate reads as a knob on a stalk."""
    controls = [p for p in anatomy["parts"] if p["kind"] == "exterior" and p.get("faces") and p["id"] in solids]
    body = [p["id"] for p in anatomy["parts"] if p["kind"] == "exterior" and not p.get("faces") and p["id"] in solids]
    if not controls or not body:
        return []
    rng = np.random.default_rng(0)
    ground = np.vstack([_surface_points(solids[i], rng) for i in body])
    out = []
    for p in controls:
        f = p["faces"]
        k = "xyz".index(f[1])
        turn = np.array(pose.part_rotation(p.get("rotation_deg") or [0, 0, 0]))
        d = turn[:, k] * (1 if f[0] == "+" else -1)
        reach = 0.5 * min(p["size_mm"][i] for i in range(3) if i != k)
        c = np.array(p["center_mm"], float)
        own = _surface_points(solids[p["id"]], rng)
        top = float(((own - c) @ d).max())
        # The body under the control: its surface within the control's footprint around the axis it faces along.
        rel = ground - c
        along = rel @ d
        across = np.linalg.norm(rel - along[:, None] * d, axis=1)
        under = along[across <= reach]
        if not len(under):
            continue
        rise = top - float(under.max())
        if rise > 0:
            out.append(f"{p['id']} stands {rise:.0f} mm out of the body")
    return out


# Outer parts reaching this far past the object's measured extent (and at least OVERSIZE_MM) make it the wrong size.
OVERSIZE = 0.06
OVERSIZE_MM = 4.0


def _oversize(anatomy: dict[str, Any], solids: dict[str, Manifold]) -> list[str]:
    """Outer parts that reach past the object's measured size (dimensions_mm, centred on its origin), per side: a
    grip drawn longer than the real one, a head poking up out of its ring."""
    dims = anatomy["object"].get("dimensions_mm")
    if not dims:
        return []
    boxes = {p["id"]: solids[p["id"]].bounding_box() for p in anatomy["parts"] if p["kind"] == "exterior" and p["id"] in solids}
    out = []
    for k, (low, high) in enumerate((("left", "right"), ("bottom", "top"), ("back", "front"))):
        limit = dims[k] / 2 + max(OVERSIZE * dims[k], OVERSIZE_MM)
        for side, sign in ((low, -1), (high, 1)):
            over = sorted(((sign * b[k + (3 if sign > 0 else 0)] - dims[k] / 2, pid) for pid, b in boxes.items()
                           if sign * b[k + (3 if sign > 0 else 0)] > limit), reverse=True)
            if over:
                out.append(f"{', '.join(f'{pid} ({mm:.0f} mm)' for mm, pid in over[:3])} reach past the object's {side}: "
                           f"it is {dims[k]:.0f} mm {('wide', 'tall', 'deep')[k]}")
    return out


def _clashes(anatomy: dict[str, Any]) -> list[str]:
    """Outer parts cut for other outer parts that are not much smaller than them: likely one passing through the other
    rather than a control set into a surface."""
    by_id = {p["id"]: p for p in anatomy["parts"]}
    out = []
    for host in anatomy["parts"]:
        for gid in host.get("cuts") or []:
            guest = by_id.get(gid)
            if guest and host["kind"] != "internal" and guest["kind"] != "internal" and max(guest["size_mm"]) >= 0.4 * max(host["size_mm"]):
                out.append(f"{gid} runs into {host['id']}")
    return out


def views(anatomy: dict[str, Any]) -> list[tuple[str, tuple[float, float, float]]]:
    cam = anatomy["object"].get("camera") or {}
    az, el = float(cam.get("azimuth_deg") or 0), float(cam.get("elevation_deg") or 0)
    return [("the front, as photographed", (az, el, float(cam.get("roll_deg") or 0))), ("the back", (az + 180, 0, 0)),
            ("its left side", (-90, 0, 0)), ("its right side", (90, 0, 0)), ("behind on its left", (-135, 25, 0)),
            ("behind on its right", (135, 25, 0)), ("above", (0, 89, 0)), ("below", (0, -89, 0))]


def sheet(anatomy: dict[str, Any], photo: bytes, box: list[float], side: dict[str, Any] | None = None
          ) -> tuple[bytes, list[str], list[str]]:
    """The review sheet (the photo, then the model from eight sides with what shows through marked), the flaws, and
    measurements for Muse to judge against the real object (how far the controls stand out)."""
    solids, trimmed = assembled(anatomy)
    tiles, seen = [], {}
    im = Image.open(io.BytesIO(photo)).convert("RGB")
    pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
    crop = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
    crop.thumbnail((TILE, TILE))
    tiles.append(("The photo", crop))
    if side is not None:
        ref = Image.open(io.BytesIO(side["jpeg"])).convert("RGB")
        ref.thumbnail((TILE, TILE))
        tiles.append((f"Real photo: {side.get('view', 'side').replace('_', ' ')}", ref))
    in_photo = frozenset(p["id"] for p in anatomy["parts"] if p["kind"] == "exterior" and p.get("sam_prompt"))
    # Parts with no SAM prompt aren't in the photo (those seen only in reference views have none either).
    unseen = frozenset(p["id"] for p in anatomy["parts"] if p["kind"] == "exterior" and not p.get("sam_prompt"))
    hidden, shown = [], []
    for n, (name, view) in enumerate(views(anatomy)):
        tile, showing, total, where, cover = _draw(anatomy, solids, view, look_for=in_photo | unseen if n == 0 else frozenset())
        shown += [f"{pid} ({100 * px / max(total, 1):.0f}% of the view)" for pid, (share, _, px) in cover.items()
                  if pid in unseen and share >= SHOWN_IN_VIEW and px / max(total, 1) >= MIN_SHOWN]
        hidden += [f"{pid} is in plain sight in the photo but hidden behind {by} from the photo's viewpoint (only "
                   f"{100 * share:.0f}% of it shows): one of them is on the wrong side of the other"
                   for pid, (share, by, _) in cover.items() if pid in in_photo and share < MIN_IN_VIEW and by]
        d = ImageDraw.Draw(tile)
        for pid, n in showing.items():
            share = n / max(total, 1)
            if share >= MIN_SHOWING:
                seen.setdefault(pid, []).append((name, share))
                d.text(where[pid], pid, fill=(255, 255, 255))
        tiles.append((f"Model from {name}", tile))
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (TILE + 12), rows * (TILE + 28)), (12, 14, 18))
    draw = ImageDraw.Draw(out)
    for i, (label, tile) in enumerate(tiles):
        x, y = (i % cols) * (TILE + 12), (i // cols) * (TILE + 28)
        out.paste(tile, (x + (TILE - tile.width) // 2, y + 24 + (TILE - tile.height) // 2))
        draw.text((x + 4, y + 6), label, fill=(230, 236, 245))
    flaws = [f"{pid} shows through from " + ", ".join(f"{n} ({100 * s:.1f}% of the view)" for n, s in where_)
             for pid, where_ in seen.items()] + hidden + _oversize(anatomy, solids) + trimmed + _floating(anatomy, solids)
    notes = _proud(anatomy, solids)
    if shown:
        notes.append("In plain view from the photo's viewpoint although the photo doesn't show them: " + ", ".join(shown)
                     + ". If the real part is on the far side (a trigger under the head, a cover on the back), move it there")
    return M.jpeg(out), flaws, notes


async def review(*, anatomy: dict[str, Any], photo: bytes, box: list[float], emit: Emit, model_id: str,
                 side: dict[str, Any] | None = None, inside: list[dict[str, Any]] | None = None) -> list[str]:
    """Let Muse fix its own design from renders of the finished model. Mutates `anatomy`; returns the flaws left."""
    parts = {p["id"]: p for p in anatomy["parts"]}
    state: dict[str, Any] = {}

    def look() -> tuple[list[dict[str, Any]], list[str]]:
        image, flaws, notes = sheet(anatomy, photo, box, side)
        state["flaws"] = flaws
        text = ("Flaws: " + "; ".join(flaws)) if flaws else "No internal part shows from outside and none was trimmed."
        clashes = _clashes(anatomy)
        if clashes:
            text += (" Outer parts that run into each other (one is cut where the other passes through it; fine for a "
                     "button set into a plate, wrong where a body pokes out through a cover): " + "; ".join(clashes))
        if notes:
            text += " To judge against the real object: " + "; ".join(notes) + "."
        return [meta.image_part(image), {"type": "text", "text": text}], flaws

    async def refit(edited: set[str]) -> list[dict[str, Any]]:
        # Geometry work runs off the event loop, so the stream's keep-alives and health checks still get through.
        await asyncio.to_thread(assembly.resolve, anatomy, edited)
        await emit({"type": "anatomy", "anatomy": A.sanitize(json.loads(json.dumps(anatomy)))})
        attach, flaws = await asyncio.to_thread(look)
        return [{"type": "text", "text": f"The model after your edits ({len(flaws)} flaws left):"}, *attach]

    async def place_parts(a: dict[str, Any]) -> ToolResult:
        done = set()
        for u in items(a.get("updates")):
            p = parts.get(u.get("id", ""))
            fields = {k: [float(x) for x in u[k]] for k in ("center_mm", "size_mm", "rotation_deg")
                      if isinstance(u.get(k), list) and len(u[k]) == 3 and all(isinstance(x, (int, float)) for x in u[k])}
            if p is None or not fields or ("size_mm" in fields and min(fields["size_mm"]) <= 0):
                continue
            p.update(fields)
            done.add(p["id"])
        if not done:
            return ToolResult("No valid updates.", "No parts changed", ok=False)
        return ToolResult("Changed; the new renders are attached.", f"Reworked {len(done)} parts", attach=await refit(done))

    def _recipe(p: dict[str, Any], recipe: Any) -> str | None:
        if isinstance(recipe, dict) and recipe.get("op") == "mould":
            return assembly.apply_mould(p, parts, recipe)
        p.pop("mould", None)
        try:
            solid = M.build(recipe)
        except M.RecipeError as e:
            return str(e)
        b = solid.bounding_box()
        p["model"], p["size_mm"] = recipe, [round(b[3 + i] - b[i], 1) for i in range(3)]
        return None

    async def model_parts(a: dict[str, Any]) -> ToolResult:
        done, errors = set(), {}
        for item in items(a.get("models")):
            p = parts.get(item.get("id", ""))
            if p is None:
                errors[item.get("id", "")] = "unknown part id"
                continue
            err = _recipe(p, item.get("recipe"))
            if err:
                errors[p["id"]] = err
            else:
                done.add(p["id"])
        if not done:
            return ToolResult(json.dumps({"errors": errors}), "No recipes built", ok=False)
        return ToolResult(json.dumps({"built": sorted(done), "errors": errors}), f"Remodeled {len(done)} parts", attach=await refit(done))

    async def add_part(a: dict[str, Any]) -> ToolResult:
        p = {**PART_DEFAULTS, **(a.get("part") or {})}
        if not p.get("id") or p["id"] in parts:
            return ToolResult("A new part needs a fresh snake_case id.", "add_part rejected", ok=False)
        # An outer part needs a recipe to take the object's colour and a moulded look: a rounded box at the least.
        recipe = a.get("recipe") or ({"op": "box", "size": p["size_mm"], "radius": round(0.2 * min(p["size_mm"]), 1)}
                                     if p["kind"] != "internal" else None)
        err = _recipe(p, recipe) if recipe else None
        anatomy["parts"].append(p)
        parts[p["id"]] = p
        return ToolResult("Added" + (f", but its recipe failed ({err}) so it is a box" if err else "") + "; renders attached.",
                          f"Added {p.get('name') or p['id']}", attach=await refit({p["id"]}))

    async def remove_part(a: dict[str, Any]) -> ToolResult:
        pid = a.get("id", "")
        if pid not in parts:
            return ToolResult(f"No part {pid!r}.", "remove_part: unknown part", ok=False)
        parts.pop(pid)
        anatomy["parts"] = [p for p in anatomy["parts"] if p["id"] != pid]
        A.sanitize(anatomy)
        return ToolResult("Removed; renders attached.", f"Removed {pid}", attach=await refit(set()))

    async def finish(a: dict[str, Any]) -> ToolResult:
        return ToolResult("ok", str(a.get("summary") or "Reviewed"), stop=True)

    vec = {"type": "array", "items": {"type": "number"}}
    part_fields = A.ANATOMY_SCHEMA["properties"]["parts"]["items"]["properties"]
    tools = [
        Tool("place_parts", "Move, resize or turn parts.", params({"updates": {"type": "array", "items": params(
            {"id": {"type": "string"}, "center_mm": vec, "size_mm": vec, "rotation_deg": vec, "reason": {"type": "string"}},
            ["id", "reason"])}}), place_parts),
        Tool("model_parts", "Give parts new geometry recipes.", params({"models": {"type": "array", "items": params(
            {"id": {"type": "string"}, "recipe": {"type": "object"}})}}), model_parts),
        Tool("add_part", "Add a part the model is missing, with an optional geometry recipe.",
             params({"part": {"type": "object", "properties": part_fields}, "recipe": {"type": "object"}}, ["part"]), add_part),
        Tool("remove_part", "Remove a part.", params({"id": {"type": "string"}}), remove_part),
        Tool("finish", "Finish the review.", params({"summary": {"type": "string"}}), finish),
    ]
    attach, flaws = await asyncio.to_thread(look)
    await emit({"type": "step", "agent": "builder", "tool": "review", "status": "done",
                "text": f"Rendered the finished model from eight sides: {len(flaws)} flaws" if flaws else
                        "Rendered the finished model from eight sides: nothing inside shows"})
    # Even with nothing showing through, the model can see what no check measures: parts poking through each other,
    # slabs standing off the body, shapes unlike the photo.
    brief = [{k: p[k] for k in ("id", "name", "kind", "shape", "size_mm", "center_mm", "rotation_deg") if k in p} for p in anatomy["parts"]]
    research = (anatomy.get("research") or {}).get("notes", "")
    photos = [x for p in inside or [] for x in ({"type": "text", "text": f"Inside view, iFixit “{p['guide']}” step {p['step']}: {p['shows']}"},
                                                 meta.image_part(p["jpeg"]))]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": [meta.image_part(photo), *photos, {"type": "text", "text":
            f"Object: {anatomy['object']['name']}, {anatomy['object']['dimensions_mm']} mm.\nParts: {json.dumps(brief)}\n"
            + (f"\nResearch notes:\n{research[:6000]}\n" if research else "") + "\nThe finished model:"}, *attach]},
    ]
    await run_loop(agent="builder", model=model_id, messages=messages, tools=one_at_a_time(tools), emit=emit, max_turns=5,
                   reasoning_effort="medium", until="finish", max_tokens=24000)
    return state.get("flaws", [])
