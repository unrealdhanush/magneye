"""Guide agent: answers prompts by driving the 3D viewer, looking closer at the photo, and asking before it blames."""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

from . import anatomy as A
from . import meta
from .agent import Emit, Tool, ToolResult, params, run_loop
from .ground import SegmentCache

PART_SCHEMA = A.ANATOMY_SCHEMA["properties"]["parts"]["items"]
PART_FIELDS = PART_SCHEMA["properties"]
# Filled in for anything a new part leaves out, so the viewer and checks always have a complete part.
PART_DEFAULTS: dict[str, Any] = {
    "name": "New part", "kind": "internal", "function": "", "material": "plastic", "shape": "box", "size_mm": [10, 10, 10],
    "center_mm": [0, 0, 0], "rotation_deg": [0, 0, 0], "layer": 1, "sam_prompt": None, "confidence": 0.5, "basis": "inferred",
}

GUIDE_SYSTEM = """You are Magneye's guide. The user is looking at an interactive 3D exploded view of the object in their photo. Its anatomy (parts, connections, scenarios, failures) is in the first message; part ids there are the only ones that exist.

You act through the viewer, then explain in words:
- highlight(part_ids): point at parts while you talk about them.
- play_scenario(scenario_id) / create_scenario(scenario): show how something works as a chain of steps through parts. Create one only when none fits.
- show_failure(failure_id) / create_failure(failure): when you are confident about a fault, show it: the parts in the way lift off and the culprit pulses red.
- mark_failures(failure_ids): for "where does it usually fail", "weak points", "common problems": mark every relevant failure at once; each culprit glows red with a number while the rest of the object fades back. create_failure first for any common failure that is missing, then mark them all in one call.
- set_explode(amount): 0 assembled, 1 fully apart.
- look_closer(part_id): crop the photo around a visible part for a closer look (visible wear, damage, debris).
- ask_user(question): ask ONE short question when the answer would change your diagnosis. Your turn ends; their reply comes next.
- edit_parts(updates) / add_part(part) / remove_part(id): change the model itself when the user asks to (add, remove, resize, move or correct a part). Positions and sizes are millimetres in the object's own frame (x its right, y up, z out of its front); keep new parts inside the object and give them a material and a short function. Highlight what you changed, then say what you did.

For symptoms: if more than one failure plausibly fits, ask a single discriminating question first (e.g. direction, timing, what they already tried), then diagnose. Never ask more than two questions in a conversation. For "how does it work" prompts, play or create a scenario. Always finish with a short spoken-style answer (at most 70 words) unless you just asked a question. Be honest about what is inferred."""


def _crop(image: bytes, box: list[float], pad: float = 0.35) -> bytes:
    im = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = box[2] - box[0], box[3] - box[1]
    region = (max(0, box[0] - w * pad), max(0, box[1] - h * pad), min(im.width, box[2] + w * pad), min(im.height, box[3] + h * pad))
    crop = im.crop(tuple(round(v) for v in region))
    crop.thumbnail((768, 768))
    out = io.BytesIO()
    crop.save(out, "JPEG", quality=90)
    return out.getvalue()


def opening_messages(anatomy: dict[str, Any], image: bytes) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": GUIDE_SYSTEM},
        {"role": "user", "content": [meta.image_part(image), {"type": "text", "text": f"Anatomy JSON:\n{json.dumps(anatomy)}"}]},
    ]


async def respond(
    *, anatomy: dict[str, Any], messages: list[dict[str, Any]], prompt: str, image: bytes, cache: SegmentCache,
    emit: Emit, model: str, ground: dict[str, Any] | None,
) -> None:
    """Run one user turn. Extends `messages` and mutates `anatomy` when the agent creates scenarios/failures."""
    ids = {p["id"] for p in anatomy["parts"]}
    by_id = {p["id"]: p for p in anatomy["parts"]}

    async def act(action: dict[str, Any], summary: str) -> ToolResult:
        await emit({"type": "action", "action": action})
        return ToolResult("ok", summary)

    async def highlight(a: dict[str, Any]) -> ToolResult:
        keep = [pid for pid in a.get("part_ids", []) if pid in ids]
        if not keep:
            return ToolResult("None of those part ids exist.", "highlight: no valid parts", ok=False)
        return await act({"type": "highlight", "part_ids": keep}, "Highlighted " + ", ".join(by_id[p]["name"] for p in keep))

    async def play_scenario(a: dict[str, Any]) -> ToolResult:
        sc = next((s for s in anatomy["scenarios"] if s["id"] == a.get("scenario_id")), None)
        if sc is None:
            return ToolResult("No such scenario id.", "play_scenario: unknown id", ok=False)
        return await act({"type": "play_scenario", "scenario": sc}, f"Playing “{sc['title']}”")

    async def create_scenario(a: dict[str, Any]) -> ToolResult:
        sc = A.sanitize({"parts": anatomy["parts"], "scenarios": [a.get("scenario") or {}]})["scenarios"] if a.get("scenario", {}).get("steps") else []
        if not sc:
            return ToolResult("Scenario needs steps that use existing part ids.", "create_scenario rejected", ok=False)
        anatomy["scenarios"].append(sc[0])
        return await act({"type": "play_scenario", "scenario": sc[0], "created": True}, f"Created and playing “{sc[0]['title']}”")

    async def show_failure(a: dict[str, Any]) -> ToolResult:
        fl = next((f for f in anatomy["failures"] if f["id"] == a.get("failure_id")), None)
        if fl is None:
            return ToolResult("No such failure id.", "show_failure: unknown id", ok=False)
        return await act({"type": "show_failure", "failure": fl}, f"Diagnosed: {by_id[fl['culprit_part_id']]['name']}")

    async def mark_failures(a: dict[str, Any]) -> ToolResult:
        wanted = a.get("failure_ids") or [f["id"] for f in anatomy["failures"]]
        found = [f for fid in wanted for f in anatomy["failures"] if f["id"] == fid]
        if not found:
            return ToolResult("None of those failure ids exist.", "mark_failures: unknown ids", ok=False)
        names = ", ".join(by_id[f["culprit_part_id"]]["name"] for f in found)
        return await act({"type": "mark_failures", "failures": found}, f"Marked {len(found)} weak points in red: {names}")

    async def create_failure(a: dict[str, Any]) -> ToolResult:
        f = a.get("failure") or {}
        fl = A.sanitize({"parts": anatomy["parts"], "failures": [f]})["failures"] if "culprit_part_id" in f and "access_path" in f else []
        if not fl:
            return ToolResult("Failure needs an existing culprit_part_id and access_path.", "create_failure rejected", ok=False)
        anatomy["failures"].append(fl[0])
        return await act({"type": "show_failure", "failure": fl[0], "created": True}, f"Diagnosed: {by_id[fl[0]['culprit_part_id']]['name']}")

    async def set_explode(a: dict[str, Any]) -> ToolResult:
        amount = min(1.0, max(0.0, float(a.get("amount", 0.5))))
        return await act({"type": "explode", "amount": amount}, f"Explode {round(amount * 100)}%")

    async def look_closer(a: dict[str, Any]) -> ToolResult:
        part = by_id.get(a.get("part_id", ""))
        if part is None:
            return ToolResult("Unknown part id.", "look_closer: unknown part", ok=False)
        box = (ground or {}).get("parts", {}).get(part["id"], {}).get("box")
        if box is None and part.get("sam_prompt"):
            matches = await cache.get(part["sam_prompt"])
            if not isinstance(matches, Exception) and matches:
                box = matches[0]["box"]
        if box is None:
            return ToolResult(f"{part['name']} is not visible in the photo.", f"{part['name']} isn't visible in the photo", ok=False)
        await emit({"type": "action", "action": {"type": "highlight", "part_ids": [part["id"]]}})
        return ToolResult("A close-up crop is attached in the next message.", f"Looked closer at {part['name']}",
                          attach=[meta.image_part(_crop(image, box)), {"type": "text", "text": f"Close-up of {part['name']} from the photo."}])

    async def ask_user(a: dict[str, Any]) -> ToolResult:
        q = str(a.get("question", "")).strip()
        await emit({"type": "question", "text": q, "options": a.get("options") or []})
        return ToolResult("Question shown. Wait for the user's reply.", "Asked you a question", stop=True)

    async def changed(part_ids: list[str]) -> None:
        anatomy["revision"] = anatomy.get("revision", 0) + 1
        await emit({"type": "anatomy", "anatomy": anatomy})
        await emit({"type": "action", "action": {"type": "highlight", "part_ids": part_ids}})

    async def edit_parts(a: dict[str, Any]) -> ToolResult:
        done, unknown = [], []
        for u in a.get("updates", []):
            p = by_id.get(u.get("id", ""))
            if p is None:
                unknown.append(u.get("id"))
                continue
            p.update({k: v for k, v in u.items() if k in PART_FIELDS and k != "id"}, edited=True)
            done.append(p["id"])
        if done:
            await changed(done)
        return ToolResult(json.dumps({"updated": done, "unknown_ids": unknown}),
                          f"Changed {', '.join(by_id[i]['name'] for i in done)}" if done else "No parts changed", ok=bool(done))

    async def add_part(a: dict[str, Any]) -> ToolResult:
        p = {**PART_DEFAULTS, **(a.get("part") or {}), "edited": True}
        if not p.get("id") or p["id"] in by_id:
            return ToolResult("A new part needs a fresh snake_case id.", "add_part rejected", ok=False)
        anatomy["parts"].append(p)
        by_id[p["id"]] = p
        ids.add(p["id"])
        await changed([p["id"]])
        return ToolResult("ok", f"Added {p.get('name') or p['id']}")

    async def remove_part(a: dict[str, Any]) -> ToolResult:
        pid = a.get("id", "")
        if pid not in by_id:
            return ToolResult(f"No part {pid!r}.", "remove_part: unknown part", ok=False)
        name = by_id.pop(pid)["name"]
        ids.discard(pid)
        anatomy["parts"] = [p for p in anatomy["parts"] if p["id"] != pid]
        A.sanitize(anatomy)
        await changed([])
        return ToolResult("ok", f"Removed {name}")

    editable = {k: v for k, v in PART_FIELDS.items() if k != "id"}
    scenario_schema, failure_schema = A.SCENARIO, A.FAILURE
    tools = [
        Tool("highlight", "Highlight parts in the viewer.", params({"part_ids": {"type": "array", "items": {"type": "string"}}}), highlight),
        Tool("play_scenario", "Play an existing scenario.", params({"scenario_id": {"type": "string"}}), play_scenario),
        Tool("create_scenario", "Create a new scenario from existing parts and play it.", params({"scenario": scenario_schema}), create_scenario),
        Tool("show_failure", "Show an existing failure: lift off the access path and pulse the culprit.", params({"failure_id": {"type": "string"}}), show_failure),
        Tool("create_failure", "Create a new failure from existing parts and show it.", params({"failure": failure_schema}), create_failure),
        Tool("mark_failures", "Mark several failures' culprits in red at once (common failure points).",
             params({"failure_ids": {"type": "array", "items": {"type": "string"}}}), mark_failures),
        Tool("set_explode", "Set how far apart the parts are (0 to 1).", params({"amount": {"type": "number"}}), set_explode),
        Tool("look_closer", "Crop the photo around a visible part to inspect it.", params({"part_id": {"type": "string"}}), look_closer),
        Tool("edit_parts", "Change fields of existing parts (name, size_mm, center_mm, material, shape, function, ...).",
             params({"updates": {"type": "array", "items": params({"id": {"type": "string"}, **editable}, ["id"])}}), edit_parts),
        Tool("add_part", "Add a new part to the model.", params({"part": PART_SCHEMA}), add_part),
        Tool("remove_part", "Remove a part from the model.", params({"id": {"type": "string"}}), remove_part),
        Tool("ask_user", "Ask the user one short question; ends your turn.",
             params({"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, ["question"]), ask_user),
    ]
    messages.append({"role": "user", "content": prompt})
    text = await run_loop(agent="guide", model=model, messages=messages, tools=tools, emit=emit, max_turns=6,
                          reasoning_effort="low")
    if text:
        await emit({"type": "answer", "text": text})
