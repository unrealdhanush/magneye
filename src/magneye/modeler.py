"""Modeler agent: gives the verified parts a real 3D form with geometry recipes, and checks its work by rendering its
models from the photo's viewpoint next to the photo itself."""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image, ImageDraw

from . import fit, mates, meta
from .checks import geometry_issues
from . import model as M
from .agent import Emit, Tool, ToolResult, params, run_loop

MODELER_SYSTEM = f"""You are Magneye's modeler. The anatomy of the object in the photo is verified, but its parts are still plain primitives. Give the parts that define how the object looks and works a real 3D form, as geometry recipes.

What to model, in this order: every exterior part (its rounded edges and corners, and the hollows, openings and cut-outs it really has: wells, ports, grilles, holes, buttons, seams), accessories, then the main internal mechanisms (gears with teeth, springs, motors, lenses, speaker drivers, cells). Skip parts under about 3 mm and anything that really is a plain box or cylinder already.

Sculpt, don't stack: where the real object is one moulded body (a mouse shell, a drill's grip running into its motor housing, a bottle, a controller's grip and ring), model that body as one smooth continuous shape: tubes and tori along its centre lines, hulls of spheres for bulges, then shell it and cut its openings. Rings and halos are tori, not polygons. Seams belong where the real part splits, not between your primitives.

{M.OPS_DOC}

A recipe is in millimetres around the part's own centre, in the part's own frame: the frame its size_mm is measured in, which its rotation_deg then turns into place in the object (x its right, y up, z out of its front). So build the part as it is before that turn, at its real size: the built size becomes the part's size_mm. Keep each recipe under 40 nodes.

Tools:
- model_parts(models): build recipes for several parts in one call; each reports its built size against the part's and any error to fix.
- view_models(): see your work: the photo, your parts drawn over the photo exactly where the camera would see them, and the internals on their own; when a real photo of another side exists, also that photo next to your parts seen from the same side.
- place_parts(updates): fix where parts are and how big they are (center_mm, size_mm, rotation_deg), in the object's own frame. Use it when the overlay shows parts offset, too small or large, turned the wrong way, or not nested the way they really are (LEDs inside the ring they light, a cell inside its holder, a board inside its shell).
- set_camera(azimuth_deg, elevation_deg, roll_deg): correct the photo's viewpoint if the whole model sits turned against the photo.
- {mates.DOC}
- finish(summary).

The overlay is the test: when you are done, the parts should cover the object in the photo, each where it really is, and fit together the way they are assembled (use mate_parts rather than nudging positions by hand).

The main photo can't show depth, so a layout can match it and still be wrong from the side. When the side view is there, it is the second test: compare the silhouettes part by part (how long and thick each part is front to back, its tilt, where it meets its neighbours) and fix every mismatch with place_parts or model_parts, even when the main overlay already looks right. A body the real object has as one piece (a handle running up into a head or ring) must read as one piece from the side too: no gaps, no floating slabs.

Work in at most three turns:
1. model_parts for every part you will model, and view_models in the same turn.
2. Fix what the overlay and the side view show: first write down each mismatch you see in either, then set_camera if everything is turned, place_parts for parts in the wrong place, size or tilt, model_parts for shapes that are wrong; then view_models again in the same turn.
3. Final touches if needed, and finish."""


def _count(n: int, one: str) -> str:
    return f"{n} {one}{'s' * (n != 1)}"


# Where the camera stands for a reference photo of each side (azimuth, elevation), in the object's frame.
SIDE_VIEWS = {"left": (-90.0, 0.0), "right": (90.0, 0.0), "back": (180.0, 0.0), "top": (0.0, 90.0)}


def side_reference(views: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The reference photo that best shows what the main photo can't: a side, then the back, then the top."""
    order = list(SIDE_VIEWS)
    found = [v for v in views or [] if v.get("view") in SIDE_VIEWS and v.get("jpeg")]
    return min(found, key=lambda v: (order.index(v["view"]), not v.get("upright"))) if found else None


def _side_panels(view: dict[str, Any], anatomy: dict[str, Any], solids: dict[str, Any]) -> bytes:
    """A reference photo of another side next to the models seen from that side, so depth, tilt and how parts
    join (which the main photo can't show) can be checked against the real object."""
    size = 420
    ref = Image.open(io.BytesIO(view["jpeg"])).convert("RGB")
    ref.thumbnail((size, size))
    az, el = SIDE_VIEWS[view["view"]]
    tiles = [(f"Reference photo: {view['view']} side", ref), (f"Your models from the {view['view']} side", M.render(anatomy, solids, size=size, view=(az, el, 0.0)))]
    sheet = Image.new("RGB", (size * 2 + 16, size + 28), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for i, (label, tile) in enumerate(tiles):
        x = i * (size + 16)
        sheet.paste(tile, (x + (size - tile.width) // 2, 28 + (size - tile.height) // 2))
        draw.text((x + 4, 6), label, fill=(230, 236, 245))
    return M.jpeg(sheet)


def _panels(photo: bytes, box: list[float], anatomy: dict[str, Any], solids: dict[str, Any]) -> bytes:
    """The photo of the object, the models drawn over it where the camera would see them, and the internals from
    a three-quarter view."""
    size = 420
    im = Image.open(io.BytesIO(photo)).convert("RGB")
    pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
    region = (max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))
    crop, over = im.crop(region), M.overlay(anatomy, solids, im, box).crop(region)
    crop.thumbnail((size, size))
    over.thumbnail((size, size))
    cam = anatomy["object"].get("camera") or {}
    three_quarter = (float(cam.get("azimuth_deg") or 0) + 35, float(cam.get("elevation_deg") or 0) + 20, 0.0)
    tiles = [("Photo", crop), ("Your models over the photo", over),
             ("Internals", M.render(anatomy, solids, internals=True, size=size, view=three_quarter))]
    sheet = Image.new("RGB", (size * 3 + 32, size + 28), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for i, (label, tile) in enumerate(tiles):
        x = i * (size + 16)
        sheet.paste(tile, (x + (size - tile.width) // 2, 28 + (size - tile.height) // 2))
        draw.text((x + 4, 6), label, fill=(230, 236, 245))
    return M.jpeg(sheet)


async def model(*, anatomy: dict[str, Any], photo: bytes, box: list[float], emit: Emit, model_id: str,
                outline: fit.Outline | None = None, printed: list[dict[str, Any]] | None = None,
                views: list[dict[str, Any]] | None = None) -> int:
    """Run the modeler on `anatomy` in place (recipes go on parts as "model"). Returns how many parts it modeled."""
    parts = {p["id"]: p for p in anatomy["parts"]}
    solids: dict[str, Any] = {}
    side = side_reference(views)
    # The best-scoring layout seen so far: a late move that loses overlap with SAM's outline is undone at the end.
    best: dict[str, Any] = {}

    def layout() -> dict[str, Any]:
        return {"camera": dict(anatomy["object"].get("camera") or {}),
                "parts": {pid: (list(p["center_mm"]), p.get("rotation_deg")) for pid, p in parts.items()}}

    def score() -> float | None:
        if outline is None:
            return None
        s = fit.coverage(anatomy, outline)
        if s > best.get("score", -1.0):
            best.update(score=s, layout=layout())
        return s

    async def model_parts(a: dict[str, Any]) -> ToolResult:
        report, built = [], []
        for item in a.get("models", []):
            pid, recipe = item.get("id", ""), item.get("recipe")
            part = parts.get(pid)
            if part is None:
                report.append({"id": pid, "error": "unknown part id"})
                continue
            try:
                solid = M.build(recipe)
            except M.RecipeError as e:
                report.append({"id": pid, "error": str(e)})
                continue
            b = solid.bounding_box()
            size = [round(b[3 + i] - b[i], 1) for i in range(3)]
            entry: dict[str, Any] = {"id": pid, "ok": True, "built_size_mm": size, "was_size_mm": part["size_mm"]}
            # The recipe is the truth about the part's form: its built size becomes the part's size, so nothing is
            # stretched. A big change usually means the recipe is turned against the part's own frame.
            off = [i for i in range(3) if part["size_mm"][i] > 0 and not 0.75 < size[i] / part["size_mm"][i] < 1.33]
            if off:
                entry["note"] = (f"built size differs from the part's along {', '.join('xyz'[i] for i in off)}; the part now takes "
                                 "the built size; if that is wrong, the recipe is probably turned against the part's own frame")
            report.append(entry)
            part["model"] = recipe
            part["size_mm"] = size
            solids[pid] = solid
            built.append(pid)
        if built:
            await emit({"type": "anatomy", "anatomy": anatomy})
        errors = sum(1 for r in report if "error" in r)
        summary = f"Built {_count(len(built), 'part')} with Manifold" + (f"; {errors} to fix" if errors else "")
        return ToolResult(json.dumps(report), summary, ok=bool(built) or not report)

    async def view_models(_: dict[str, Any]) -> ToolResult:
        sheet = _panels(photo, box, anatomy, solids)
        now = score()
        score_text = f" Overlap with SAM's outline of the object: {now:.0%} (higher is better)." if now is not None else ""
        gaps = [i for i in geometry_issues(anatomy) if "apart; move them to touch" in i]
        joins = (" Joined parts that don't touch yet: " + " ".join(gaps)) if gaps else ""
        attach = [meta.image_part(sheet), {"type": "text", "text": "Left: the photo. Middle: your parts drawn over the photo where the camera would see them (unmodeled parts as boxes); they should cover the object, each in its real place. Right: the internals."}]
        if side:
            attach += [meta.image_part(_side_panels(side, anatomy, solids)), {"type": "text", "text":
                       f"A real photo of the {side['view']} side ({side.get('note', '')}; it may be turned or tilted, so compare shapes, "
                       f"not orientation) next to your outer parts seen from that side. The main photo can't show depth: check here "
                       f"that each part's tilt, depth and how it joins its neighbours match the real object, and fix what doesn't."}]
        return ToolResult(f"The renders are attached in the next message.{score_text}{joins}",
                          f"Drew {_count(len(solids), 'model')} over the photo{f' · {now:.0%} overlap' if now is not None else ''}"
                          + (f" and from the {side['view']} side" if side else ""), attach=attach)

    async def place_parts(a: dict[str, Any]) -> ToolResult:
        done, unknown = [], []
        for u in a.get("updates", []):
            part = parts.get(u.get("id", ""))
            fields = {k: u[k] for k in ("center_mm", "size_mm", "rotation_deg")
                      if isinstance(u.get(k), list) and len(u[k]) == 3 and all(isinstance(x, (int, float)) for x in u[k])}
            if part is None or not fields or ("size_mm" in fields and min(fields["size_mm"]) <= 0):
                unknown.append(u.get("id"))
                continue
            part.update(fields)
            done.append(part["id"])
        if done:
            await emit({"type": "anatomy", "anatomy": anatomy})
        return ToolResult(json.dumps({"placed": done, "rejected": unknown}),
                          f"Moved or resized {_count(len(done), 'part')}" if done else "No parts placed", ok=bool(done))

    async def mate_parts(a: dict[str, Any]) -> ToolResult:
        done, failed = [], []
        for m in a.get("mates", []):
            try:
                done.append(mates.apply(parts, m))
                anatomy.setdefault("mates", []).append({k: m[k] for k in ("part", "to", "type", "face", "axis", "direction", "at") if k in m})
            except mates.MateError as e:
                failed.append({"mate": m, "error": str(e)})
        if done:
            await emit({"type": "anatomy", "anatomy": anatomy})
        return ToolResult(json.dumps({"done": done, "failed": failed}),
                          f"Fitted {_count(len(done), 'part')} together" + (f"; {len(failed)} to fix" if failed else ""), ok=bool(done))

    async def set_camera(a: dict[str, Any]) -> ToolResult:
        try:
            cam = {k: float(a[k]) for k in ("azimuth_deg", "elevation_deg", "roll_deg")}
        except (KeyError, TypeError, ValueError):
            return ToolResult("Give azimuth_deg, elevation_deg and roll_deg as numbers.", "set_camera rejected", ok=False)
        before = score()
        anatomy["object"]["camera"] = cam
        await emit({"type": "anatomy", "anatomy": anatomy})
        after = score()
        change = f" Overlap with SAM's outline: {before:.0%} -> {after:.0%}." if before is not None and after is not None else ""
        return ToolResult("ok." + change, f"Viewpoint now {cam['azimuth_deg']:g}° around, {cam['elevation_deg']:g}° up, {cam['roll_deg']:g}° rolled")

    async def finish(a: dict[str, Any]) -> ToolResult:
        return ToolResult("ok", str(a.get("summary", "Modeled")), stop=True)

    recipe_list = {"type": "array", "items": params({"id": {"type": "string"}, "recipe": {"type": "object"}})}
    tools = [
        Tool("model_parts", "Build geometry recipes for several parts.", params({"models": recipe_list}), model_parts),
        Tool("view_models", "Draw your models over the photo, and the internals.", params({}), view_models),
        Tool("place_parts", "Move, resize or turn parts (object frame, mm and degrees).",
             params({"updates": {"type": "array", "items": params({"id": {"type": "string"}, "center_mm": {"type": "array", "items": {"type": "number"}},
                                                                    "size_mm": {"type": "array", "items": {"type": "number"}},
                                                                    "rotation_deg": {"type": "array", "items": {"type": "number"}},
                                                                    "reason": {"type": "string"}}, ["id", "reason"])}}), place_parts),
        Tool("mate_parts", "Fit parts together: inside, on a face, coaxial, touching, or meshed gears. Moves `part` onto `to`.",
             params({"mates": {"type": "array", "items": params({
                 "part": {"type": "string"}, "to": {"type": "string"}, "type": {"type": "string", "enum": list(mates.TYPES)},
                 "face": {"type": "string"}, "axis": {"type": "string"}, "direction": {"type": "string"},
                 "at": {"type": "array", "items": {"type": "number"}}}, ["part", "to", "type"])}}), mate_parts),
        Tool("set_camera", "Correct where the photo was taken from.",
             params({"azimuth_deg": {"type": "number"}, "elevation_deg": {"type": "number"}, "roll_deg": {"type": "number"}}), set_camera),
        Tool("finish", "Finish modeling.", params({"summary": {"type": "string"}}), finish),
    ]
    brief = [{k: p.get(k) for k in ("id", "name", "kind", "function", "material", "shape", "size_mm", "center_mm", "layer")} for p in anatomy["parts"]]
    # Printable models made for the product show its real shell shape (a stand's cradle, a cover's inside).
    shapes = [part for r in printed or [] for part in (
        meta.image_part(r["drawing"]),
        {"type": "text", "text": f"Printable model \"{r['title']}\" ({r['relation'].replace('_', ' ')}), {' x '.join(f'{v:g}' for v in r['model_size_mm'])} mm: {r['note']}"})]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODELER_SYSTEM},
        {"role": "user", "content": [meta.image_part(photo), *shapes, {"type": "text", "text":
            f"Object: {anatomy['object']['name']} ({anatomy['object'].get('category', '')}), {anatomy['object']['dimensions_mm']} mm.\n"
            f"Parts:\n{json.dumps(brief)}\n\nModel them."
            + ("\n\nThe printable models above were made by people who own this product; match the outer shell to them." if shapes else "")}]},
    ]
    score()
    await run_loop(agent="builder", model=model_id, messages=messages, tools=tools, emit=emit, max_turns=5,
                   reasoning_effort="low", until="finish", max_tokens=24000)
    final = fit.coverage(anatomy, outline) if outline else None
    if final is not None and best and final < best["score"] - 0.03:
        anatomy["object"]["camera"] = best["layout"]["camera"]
        for pid, (center, rotation) in best["layout"]["parts"].items():
            if pid in parts:
                parts[pid]["center_mm"] = center
                if rotation is not None:
                    parts[pid]["rotation_deg"] = rotation
        await emit({"type": "anatomy", "anatomy": anatomy})
        await emit({"type": "step", "agent": "builder", "tool": "fit_view", "status": "done",
                    "text": f"Restored the best-fitting layout: overlap {final:.0%} → {fit.coverage(anatomy, outline):.0%}"})
    return len(solids)


