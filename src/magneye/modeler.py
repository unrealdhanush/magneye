"""Modeler agent: gives the verified parts a real 3D form with geometry recipes, and checks its work by rendering its
models from the photo's viewpoint next to the photo itself."""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image, ImageDraw

from . import meta
from . import model as M
from .agent import Emit, Tool, ToolResult, params, run_loop

MODELER_SYSTEM = f"""You are Magneye's modeler. The anatomy of the object in the photo is verified, but its parts are still plain primitives. Give the parts that define how the object looks and works a real 3D form, as geometry recipes.

What to model, in this order: every exterior part (its rounded edges and corners, and the hollows, openings and cut-outs it really has: wells, ports, grilles, holes, buttons, seams), accessories, then the main internal mechanisms (gears with teeth, springs, motors, lenses, speaker drivers, cells). Skip parts under about 3 mm and anything that really is a plain box or cylinder already.

{M.OPS_DOC}

A recipe is in millimetres around the part's own centre, in the object's axes (x its right, y up, z out of its front). The built solid is fitted to the part's size_mm wherever it is shown, so get the proportions and features right; the size and place stay as they are. Keep each recipe under 40 nodes.

Tools:
- model_parts(models): build recipes for several parts in one call; each reports its built size against the part's and any error to fix.
- view_models(): see your work: the photo, your parts rendered from the photo's viewpoint, and the internals on their own.
- finish(summary).

Work in at most two turns:
1. model_parts for every part you will model, and view_models in the same turn.
2. Fix errors and anything the render shows is off with one model_parts call for just those parts, and call finish in that same turn."""


def _count(n: int, one: str) -> str:
    return f"{n} {one}{'s' * (n != 1)}"


def _panels(photo: bytes, box: list[float], anatomy: dict[str, Any], solids: dict[str, Any]) -> bytes:
    """The photo of the object beside the models seen the same way, and the internals from a three-quarter view."""
    size = 420
    im = Image.open(io.BytesIO(photo)).convert("RGB")
    pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
    crop = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
    crop.thumbnail((size, size))
    cam = anatomy["object"].get("camera") or {}
    three_quarter = (float(cam.get("azimuth_deg") or 0) + 35, float(cam.get("elevation_deg") or 0) + 20, 0.0)
    tiles = [("Photo", crop), ("Your models, as photographed", M.render(anatomy, solids, size=size)),
             ("Internals", M.render(anatomy, solids, internals=True, size=size, view=three_quarter))]
    sheet = Image.new("RGB", (size * 3 + 32, size + 28), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for i, (label, tile) in enumerate(tiles):
        x = i * (size + 16)
        sheet.paste(tile, (x + (size - tile.width) // 2, 28 + (size - tile.height) // 2))
        draw.text((x + 4, 6), label, fill=(230, 236, 245))
    return M.jpeg(sheet)


async def model(*, anatomy: dict[str, Any], photo: bytes, box: list[float], emit: Emit, model_id: str) -> int:
    """Run the modeler on `anatomy` in place (recipes go on parts as "model"). Returns how many parts it modeled."""
    parts = {p["id"]: p for p in anatomy["parts"]}
    solids: dict[str, Any] = {}

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
            entry: dict[str, Any] = {"id": pid, "ok": True, "built_size_mm": size, "part_size_mm": part["size_mm"]}
            # It is stretched to the part's size anyway; a big mismatch means the proportions are probably wrong.
            off = [i for i in range(3) if part["size_mm"][i] > 0 and not 0.75 < size[i] / part["size_mm"][i] < 1.33]
            if off:
                entry["note"] = f"proportions differ from the part along {', '.join('xyz'[i] for i in off)}; it will be stretched to fit"
            report.append(entry)
            part["model"] = recipe
            solids[pid] = solid
            built.append(pid)
        if built:
            await emit({"type": "anatomy", "anatomy": anatomy})
        errors = sum(1 for r in report if "error" in r)
        summary = f"Built {_count(len(built), 'part')} with Manifold" + (f"; {errors} to fix" if errors else "")
        return ToolResult(json.dumps(report), summary, ok=bool(built) or not report)

    async def view_models(_: dict[str, Any]) -> ToolResult:
        sheet = _panels(photo, box, anatomy, solids)
        return ToolResult("The renders are attached in the next message.", f"Rendered {_count(len(solids), 'model')} beside the photo",
                          attach=[meta.image_part(sheet), {"type": "text", "text": "Left: the photo. Middle: your models from the photo's viewpoint (unmodeled parts as boxes). Right: the internals."}])

    async def finish(a: dict[str, Any]) -> ToolResult:
        return ToolResult("ok", str(a.get("summary", "Modeled")), stop=True)

    recipe_list = {"type": "array", "items": params({"id": {"type": "string"}, "recipe": {"type": "object"}})}
    tools = [
        Tool("model_parts", "Build geometry recipes for several parts.", params({"models": recipe_list}), model_parts),
        Tool("view_models", "Render your models beside the photo.", params({}), view_models),
        Tool("finish", "Finish modeling.", params({"summary": {"type": "string"}}), finish),
    ]
    brief = [{k: p.get(k) for k in ("id", "name", "kind", "function", "material", "shape", "size_mm", "center_mm", "layer")} for p in anatomy["parts"]]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODELER_SYSTEM},
        {"role": "user", "content": [meta.image_part(photo), {"type": "text", "text":
            f"Object: {anatomy['object']['name']} ({anatomy['object'].get('category', '')}), {anatomy['object']['dimensions_mm']} mm.\n"
            f"Parts:\n{json.dumps(brief)}\n\nModel them."}]},
    ]
    await run_loop(agent="builder", model=model_id, messages=messages, tools=tools, emit=emit, max_turns=4,
                   reasoning_effort="low", until="finish", max_tokens=24000)
    return len(solids)


