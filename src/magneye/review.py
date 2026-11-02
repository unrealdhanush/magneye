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
A part listed as trimmed was cut down to fit inside the shell: the same causes apply. Also fix outer parts that float apart from the body or stick out where the real object is smooth, and compare the whole model with the photos: at a glance it should read as the same object, with the same proportions and the same smooth, moulded forms. Give every outer part you add a recipe in its real shape (a loft through its cross-sections, or a blend, so it is rounded, tapers and joins its neighbours without steps); without one it is a plain rounded box. Reshape blocky outer parts the same way with model_parts.

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
          size: int = TILE) -> tuple[Image.Image, dict[str, int], int, dict[str, tuple[float, float]]]:
    """A shaded render with internal parts that show painted magenta: (image, visible pixels per internal part,
    pixels of the whole object, where each showing part's pixels centre)."""
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
        return im, {}, 0, {}
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
    return im, showing, total, where


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
          ) -> tuple[bytes, list[str]]:
    """The review sheet (the photo, then the model from eight sides with what shows through marked) and the flaws."""
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
    for name, view in views(anatomy):
        tile, showing, total, where = _draw(anatomy, solids, view)
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
             for pid, where_ in seen.items()] + trimmed
    return M.jpeg(out), flaws


async def review(*, anatomy: dict[str, Any], photo: bytes, box: list[float], emit: Emit, model_id: str,
                 side: dict[str, Any] | None = None, inside: list[dict[str, Any]] | None = None) -> list[str]:
    """Let Muse fix its own design from renders of the finished model. Mutates `anatomy`; returns the flaws left."""
    parts = {p["id"]: p for p in anatomy["parts"]}
    state: dict[str, Any] = {}

    def look() -> tuple[list[dict[str, Any]], list[str]]:
        image, flaws = sheet(anatomy, photo, box, side)
        state["flaws"] = flaws
        text = ("Flaws: " + "; ".join(flaws)) if flaws else "No internal part shows from outside and none was trimmed."
        clashes = _clashes(anatomy)
        if clashes:
            text += (" Outer parts that run into each other (one is cut where the other passes through it; fine for a "
                     "button set into a plate, wrong where a body pokes out through a cover): " + "; ".join(clashes))
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
