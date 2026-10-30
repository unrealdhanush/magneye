"""Builder agent: draft the anatomy with Muse Wonder, then verify and revise it against the photo."""

from __future__ import annotations

import copy
import json
from typing import Any

from . import anatomy as A
from . import meta
from .agent import Emit, Tool, ToolResult, params, run_loop
from .checks import geometry_issues, layout_overlay
from .ground import SegmentCache, object_frame, part_matches

PART_SCHEMA = A.ANATOMY_SCHEMA["properties"]["parts"]["items"]

BUILDER_SYSTEM = """You are Magneye's builder. You drafted the anatomy of the object in the photo; now verify it before anyone sees it, and fix what is wrong.

Tools:
- find_in_photo(phrase): SAM 3.1 segments a noun phrase in the photo and reports matches on the object, as centre/size in the object's millimetre frame (x right, y up, origin at the object's centre). Use it to confirm parts you marked visible, find their real positions, and discover visible parts you missed. "No matches" means the part is probably not visible: set its sam_prompt to null, or remove it if it was exterior and imaginary.
- check_geometry(): lists physical problems (parts outside the object, internal parts in the same space, access paths that dig outward).
- view_layout(): shows every part's front-view footprint drawn over the photo (exterior cyan, internal orange). Use it to judge proportions and placement.
- update_part / add_part / remove_part: edit the anatomy. Keep ids stable; give a short reason.
- finish(summary): call when the anatomy is consistent with the photo and check_geometry reports nothing serious.

Work efficiently: check the visible parts first (several find_in_photo calls in one turn is fine), then geometry, then look at the layout once. Do not invent precision you cannot see; hidden internals stay inferred. Finish within a few turns."""


def _brief(anatomy: dict[str, Any]) -> str:
    """Anatomy without scenarios/failures detail: what the builder needs to reason about layout."""
    return json.dumps({"object": anatomy["object"], "parts": anatomy["parts"],
                       "failures": [{k: f[k] for k in ("id", "culprit_part_id", "access_path")} for f in anatomy["failures"]]})


async def build(
    *, image: bytes, prompt: str, emit: Emit, model: str, reasoning: str, cache: SegmentCache, max_turns: int = 8,
) -> dict[str, Any]:
    await emit({"type": "step", "agent": "builder", "tool": "draft", "status": "running"})
    draft = A.sanitize(await meta.chat_json(
        model=model, system=A.ANATOMY_SYSTEM, user_text=A.anatomy_user_text(prompt), image_jpeg=image,
        schema_name="anatomy", schema=A.ANATOMY_SCHEMA, reasoning_effort=reasoning, max_tokens=24000,
    ))
    await emit({"type": "step", "agent": "builder", "tool": "draft", "status": "done",
                "text": f"Drafted {draft['object']['name']} with {len(draft['parts'])} parts"})
    await emit({"type": "anatomy", "anatomy": draft})

    found = await object_frame(cache, draft)
    if found is None:
        await emit({"type": "step", "agent": "builder", "tool": "find_in_photo", "status": "error",
                    "text": f"SAM 3.1 couldn't find “{draft['object']['sam_prompt']}”; keeping the unverified draft"})
        return draft
    frame, target = found
    work = copy.deepcopy(draft)
    parts = {p["id"]: p for p in work["parts"]}

    async def changed() -> None:
        await emit({"type": "anatomy", "anatomy": A.sanitize(copy.deepcopy(work))})

    async def find_in_photo(a: dict[str, Any]) -> ToolResult:
        phrase = str(a.get("phrase", "")).strip()
        matches = await cache.get(phrase)
        if isinstance(matches, Exception):
            return ToolResult(f"SAM error: {matches}", f"SAM failed on “{phrase}”", ok=False)
        on_object = part_matches(frame, target, matches)[:6]
        found_mm = [{**frame.box_to_mm(m["box"]), "confidence": round(m["confidence"] or 0, 2)} for m in on_object]
        summary = f"SAM 3.1: {len(found_mm)} × “{phrase}”" if found_mm else f"SAM 3.1: no “{phrase}” on the object"
        return ToolResult(json.dumps({"phrase": phrase, "matches": found_mm}), summary)

    async def check_geometry(_: dict[str, Any]) -> ToolResult:
        issues = geometry_issues(work)
        return ToolResult(json.dumps({"issues": issues}),
                          f"{len(issues)} geometry issue{'s' * (len(issues) != 1)}" if issues else "Geometry checks pass")

    async def view_layout(_: dict[str, Any]) -> ToolResult:
        overlay = layout_overlay(image, frame, work)
        return ToolResult("The layout overlay is attached in the next message.", f"Drew {len(work['parts'])} parts over the photo",
                          attach=[meta.image_part(overlay), {"type": "text", "text": "Your current layout over the photo."}])

    async def update_part(a: dict[str, Any]) -> ToolResult:
        p = parts.get(a.get("id", ""))
        if p is None:
            return ToolResult(f"No part {a.get('id')!r}.", f"update_part: unknown {a.get('id')}", ok=False)
        fields = {k: v for k, v in a.items() if k in PART_SCHEMA["properties"] and k != "id"}
        p.update(fields)
        await changed()
        return ToolResult("ok", f"Updated {p['id']} ({', '.join(fields) or 'nothing'}): {a.get('reason', '')}".rstrip(": "))

    async def add_part(a: dict[str, Any]) -> ToolResult:
        p = a.get("part") or {}
        missing = [k for k in PART_SCHEMA["required"] if k not in p]
        if missing or p.get("id") in parts:
            return ToolResult(f"Invalid part: missing {missing} or duplicate id.", "add_part rejected", ok=False)
        work["parts"].append(p)
        parts[p["id"]] = p
        await changed()
        return ToolResult("ok", f"Added {p['id']}: {a.get('reason', '')}".rstrip(": "))

    async def remove_part(a: dict[str, Any]) -> ToolResult:
        pid = a.get("id", "")
        if pid not in parts:
            return ToolResult(f"No part {pid!r}.", f"remove_part: unknown {pid}", ok=False)
        work["parts"] = [p for p in work["parts"] if p["id"] != pid]
        del parts[pid]
        await changed()
        return ToolResult("ok", f"Removed {pid}: {a.get('reason', '')}".rstrip(": "))

    async def finish(a: dict[str, Any]) -> ToolResult:
        return ToolResult("ok", str(a.get("summary", "Verified")), stop=True)

    editable = {k: v for k, v in PART_SCHEMA["properties"].items() if k != "id"}
    tools = [
        Tool("find_in_photo", "Segment a short noun phrase in the photo with SAM 3.1; returns matches on the object in mm.",
             params({"phrase": {"type": "string"}}), find_in_photo),
        Tool("check_geometry", "List physical plausibility problems in the current anatomy.", params({}), check_geometry),
        Tool("view_layout", "See the current part layout drawn over the photo.", params({}), view_layout),
        Tool("update_part", "Change fields of one part.",
             params({"id": {"type": "string"}, "reason": {"type": "string"}, **editable}, ["id", "reason"]), update_part),
        Tool("add_part", "Add a part you missed.", params({"part": PART_SCHEMA, "reason": {"type": "string"}}), add_part),
        Tool("remove_part", "Remove a part that does not exist.", params({"id": {"type": "string"}, "reason": {"type": "string"}}), remove_part),
        Tool("finish", "Finish verification.", params({"summary": {"type": "string"}}), finish),
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": [meta.image_part(image), {"type": "text", "text": f"Your draft:\n{_brief(work)}\n\nVerify it."}]},
    ]
    try:
        await run_loop(agent="builder", model=model, messages=messages, tools=tools, emit=emit,
                       max_turns=max_turns, reasoning_effort=reasoning)
    except meta.MetaAPIError as e:
        await emit({"type": "step", "agent": "builder", "tool": "verify", "status": "error",
                    "text": f"Verification stopped ({e}); keeping the latest anatomy"})
    return A.sanitize(work)
