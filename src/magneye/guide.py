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

GUIDE_SYSTEM = """You are Magneye's guide. The user is looking at an interactive 3D exploded view of the object in their photo. Its anatomy (parts, connections, scenarios, failures) is in the first message; part ids there are the only ones that exist.

You act through the viewer, then explain in words:
- highlight(part_ids): point at parts while you talk about them.
- play_scenario(scenario_id) / create_scenario(scenario): show how something works as a chain of steps through parts. Create one only when none fits.
- show_failure(failure_id) / create_failure(failure): when you are confident about a fault, show it: the parts in the way lift off and the culprit pulses red.
- set_explode(amount): 0 assembled, 1 fully apart.
- look_closer(part_id): crop the photo around a visible part for a closer look (visible wear, damage, debris).
- ask_user(question): ask ONE short question when the answer would change your diagnosis. Your turn ends; their reply comes next.

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

    scenario_schema, failure_schema = A.SCENARIO, A.FAILURE
    tools = [
        Tool("highlight", "Highlight parts in the viewer.", params({"part_ids": {"type": "array", "items": {"type": "string"}}}), highlight),
        Tool("play_scenario", "Play an existing scenario.", params({"scenario_id": {"type": "string"}}), play_scenario),
        Tool("create_scenario", "Create a new scenario from existing parts and play it.", params({"scenario": scenario_schema}), create_scenario),
        Tool("show_failure", "Show an existing failure: lift off the access path and pulse the culprit.", params({"failure_id": {"type": "string"}}), show_failure),
        Tool("create_failure", "Create a new failure from existing parts and show it.", params({"failure": failure_schema}), create_failure),
        Tool("set_explode", "Set how far apart the parts are (0 to 1).", params({"amount": {"type": "number"}}), set_explode),
        Tool("look_closer", "Crop the photo around a visible part to inspect it.", params({"part_id": {"type": "string"}}), look_closer),
        Tool("ask_user", "Ask the user one short question; ends your turn.",
             params({"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, ["question"]), ask_user),
    ]
    messages.append({"role": "user", "content": prompt})
    text = await run_loop(agent="guide", model=model, messages=messages, tools=tools, emit=emit, max_turns=6,
                          reasoning_effort="low")
    if text:
        await emit({"type": "answer", "text": text})
