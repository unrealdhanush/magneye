"""Builder agent: draft the anatomy with Muse Wonder, then verify and revise it against the photo."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from . import anatomy as A
from . import meta, views
from .agent import Emit, Tool, ToolResult, params, run_loop
from .checks import geometry_issues, layout_overlay
from .ground import SegmentCache, object_frame, part_matches

PART_SCHEMA = A.ANATOMY_SCHEMA["properties"]["parts"]["items"]

BUILDER_SYSTEM = """You are Magneye's builder. You drafted the anatomy of the object in the photo; now verify it before anyone sees it, and fix what is wrong.

Tools:
- find_in_photo(phrase): SAM 3.1 segments a noun phrase in the photo and reports matches on the object, as centre/size in the object's millimetre frame (x right, y up, origin at the object's centre). Use it to confirm parts you marked visible, find their real positions, and discover visible parts you missed. "No matches" means the part is probably not visible: set its sam_prompt to null, or remove it if it was exterior and imaginary. SAM wants plain generic nouns ("button", "strap", "lens"); brand names and labels match nothing, so retry a miss with a simpler noun before concluding. SAM finds whatever best fits the phrase, so a match far from where you expect is probably a different part: compare positions before trusting it.
- check_geometry(): lists physical problems (parts outside the object, internal parts in the same space, access paths that dig outward).
- view_layout(): shows every part's front-view footprint drawn over the photo (exterior cyan, internal orange). Use it to judge proportions and placement.
- move_parts(dx_mm, dy_mm, dz_mm, ids): shift a group of parts together (default: every part except accessories). When SAM shows the whole body sits off where you drafted it, one move_parts call fixes every part at once; do not rewrite each centre with update_parts.
- update_parts / add_part / remove_part: edit the anatomy. Put every part change into ONE update_parts call. Keep ids stable; give a short reason per part.
- finish(summary): call when the anatomy is consistent with the photo and check_geometry reports nothing serious.

Work in at most three turns:
1. In one turn, call find_in_photo for every part you marked visible, plus check_geometry and view_layout.
2. Retry any misses with simpler nouns, if needed, and call move_parts if the whole body is offset, in one turn.
3. Apply the remaining fixes in one update_parts call (plus any add_part/remove_part), using positions after any move, and call finish in that same turn.
Do not invent precision you cannot see; hidden internals stay inferred."""


def _brief(anatomy: dict[str, Any]) -> str:
    """Anatomy without scenarios/failures detail: what the builder needs to reason about layout."""
    return json.dumps({"object": anatomy["object"], "parts": anatomy["parts"],
                       "failures": [{k: f[k] for k in ("id", "culprit_part_id", "access_path")} for f in anatomy["failures"]]})


async def _research(*, image: bytes, prompt: str, emit: Emit, model: str) -> dict[str, Any] | None:
    """Search the web for teardowns and specs of the object in the photo; None if research is unavailable."""
    await emit({"type": "step", "agent": "builder", "tool": "research", "status": "running"})
    try:
        found = await meta.research(model=model, image_jpeg=image, instructions=A.RESEARCH_SYSTEM,
                                    question=A.research_question(prompt))
    except meta.MetaAPIError as e:
        await emit({"type": "step", "agent": "builder", "tool": "research", "status": "error",
                    "text": f"Web research unavailable ({e}); drafting from the model's own knowledge"})
        return None
    hosts = list(dict.fromkeys(_host(s["url"]) for s in found["sources"]))
    await emit({"type": "step", "agent": "builder", "tool": "research", "status": "done",
                "text": f"{_count(len(found['searches']), 'search', 'searches')}, {_count(len(found['sources']), 'source', 'sources')} cited"
                        + (f" ({', '.join(hosts[:4])})" if hosts else "")})
    return found if found["text"] else None


async def _views(*, image: bytes, own: list[bytes], query: str | None, sources: list[dict[str, str]], emit: Emit,
                 model: str) -> list[dict[str, Any]]:
    """Vetted photos of the same product from other angles, the user's own extra photos first; [] when none turn up."""
    if not query and not sources and not own:
        return []
    await emit({"type": "step", "agent": "builder", "tool": "reference_views", "status": "running"})
    try:
        found = await views.find(image=image, query=query, sources=sources, model=model, own=own)
    except meta.MetaAPIError as e:
        await emit({"type": "step", "agent": "builder", "tool": "reference_views", "status": "error",
                    "text": f"Reference photos unavailable ({e}); building from the one photo"})
        return []
    picked = found["views"]
    where = list(dict.fromkeys({"commons": "Wikimedia Commons", "user": "your photos"}.get(v["origin"]) or _host(v["page_url"])
                               for v in picked))
    await emit({"type": "step", "agent": "builder", "tool": "reference_views", "status": "done",
                "text": f"{_count(len(picked), 'reference view', 'reference views')} of {found['considered']} photos checked: "
                        f"{', '.join(v['view'].replace('_', ' ') for v in picked)} ({', '.join(where)})" if picked
                        else f"No usable reference views among {_count(found['considered'], 'photo', 'photos')}"})
    return picked


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0].removeprefix("www.")


async def build(
    *, image: bytes, prompt: str, emit: Emit, model: str, reasoning: str, cache: SegmentCache, max_turns: int = 6,
    own: list[bytes] | None = None,
) -> dict[str, Any]:
    research = await _research(image=image, prompt=prompt, emit=emit, model=model)
    query = None
    if research:
        research["text"], query = A.photo_query(research["text"])
    picked = await _views(image=image, own=own or [], query=query, sources=research["sources"] if research else [],
                          emit=emit, model=model)
    await emit({"type": "step", "agent": "builder", "tool": "draft", "status": "running"})
    draft = A.sanitize(await meta.chat_json(
        model=model, system=A.ANATOMY_SYSTEM, user_text=A.anatomy_user_text(prompt, research and research["text"], picked),
        image_jpeg=image, more_images=[v["jpeg"] for v in picked], schema_name="anatomy", schema=A.ANATOMY_SCHEMA,
        reasoning_effort=reasoning, max_tokens=24000,
    ))
    if research:
        # Kept with the anatomy so the guide can answer from the same references and the UI can cite them.
        draft["research"] = {"notes": research["text"][:6000], "sources": research["sources"][:12], "searches": research["searches"]}
    if picked:
        draft["references"] = [views.cite(v) for v in picked]
    # The side profile only needs SAM on a reference photo, so it runs while the draft is verified.
    side = views.profile_view(picked)
    profile = asyncio.create_task(views.side_profile(side, draft["object"]["sam_prompt"])) if side else None
    await emit({"type": "step", "agent": "builder", "tool": "draft", "status": "done",
                "text": f"Drafted {draft['object']['name']} with {len(draft['parts'])} parts"})
    await emit({"type": "anatomy", "anatomy": draft})

    found = await object_frame(cache, draft)
    if found is None:
        await emit({"type": "step", "agent": "builder", "tool": "find_in_photo", "status": "error",
                    "text": f"SAM 3.1 couldn't find “{draft['object']['sam_prompt']}”; keeping the unverified draft"})
        return await _with_profile(draft, profile, emit)
    await emit({"type": "step", "agent": "builder", "tool": "find_in_photo", "status": "done",
                "text": f"SAM 3.1 found the {draft['object']['sam_prompt']}"})
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

    async def move_parts(a: dict[str, Any]) -> ToolResult:
        try:
            d = [float(a.get(k) or 0) for k in ("dx_mm", "dy_mm", "dz_mm")]
        except (TypeError, ValueError):
            return ToolResult("dx_mm, dy_mm and dz_mm must be numbers.", "move_parts rejected", ok=False)
        ids = set(a.get("ids") or [p["id"] for p in work["parts"] if p["kind"] != "accessory"]) & set(parts)
        for pid in ids:
            parts[pid]["center_mm"] = [round(c + v, 1) for c, v in zip(parts[pid]["center_mm"], d)]
        if ids:
            await changed()
        return ToolResult(json.dumps({"moved": sorted(ids)}),
                          f"Moved {_count(len(ids), 'part', 'parts')} by ({', '.join(f'{v:+g}' for v in d)}) mm", ok=bool(ids))

    async def update_parts(a: dict[str, Any]) -> ToolResult:
        done, unknown = [], []
        for u in a.get("updates", []):
            p = parts.get(u.get("id", ""))
            if p is None:
                unknown.append(u.get("id"))
                continue
            p.update({k: v for k, v in u.items() if k in PART_SCHEMA["properties"] and k != "id"})
            done.append(p["id"])
        if done:
            await changed()
        reasons = "; ".join(f"{u['id']}: {u.get('reason', '')}" for u in a.get("updates", []) if u.get("id") in done)
        return ToolResult(json.dumps({"updated": done, "unknown_ids": unknown}),
                          f"Revised {len(done)} part{'s' * (len(done) != 1)} — {reasons}" if done else "No parts updated", ok=bool(done))

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
        Tool("move_parts", "Shift a group of parts by the same offset (default: all parts except accessories).",
             params({"dx_mm": {"type": "number"}, "dy_mm": {"type": "number"}, "dz_mm": {"type": "number"},
                     "ids": {"type": "array", "items": {"type": "string"}}}, ["dx_mm", "dy_mm", "dz_mm"]), move_parts),
        Tool("update_parts", "Change fields of several parts in one call.",
             params({"updates": {"type": "array", "items": params({"id": {"type": "string"}, "reason": {"type": "string"}, **editable}, ["id", "reason"])}}),
             update_parts),
        Tool("add_part", "Add a part you missed.", params({"part": PART_SCHEMA, "reason": {"type": "string"}}), add_part),
        Tool("remove_part", "Remove a part that does not exist.", params({"id": {"type": "string"}, "reason": {"type": "string"}}), remove_part),
        Tool("finish", "Finish verification.", params({"summary": {"type": "string"}}), finish),
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": [meta.image_part(image), {"type": "text", "text": f"Your draft:\n{_brief(work)}\n\nVerify it."}]},
    ]
    try:
        # The draft needs deep reasoning; the verify turns are mostly bookkeeping on tool results.
        await run_loop(agent="builder", model=model, messages=messages, tools=tools, emit=emit,
                       max_turns=max_turns, reasoning_effort="low", until="finish")
    except meta.MetaAPIError as e:
        await emit({"type": "step", "agent": "builder", "tool": "verify", "status": "error",
                    "text": f"Verification stopped ({e}); keeping the latest anatomy"})
    return await _with_profile(A.sanitize(work), profile, emit)


async def _with_profile(anatomy: dict[str, Any], task: asyncio.Task | None, emit: Emit) -> dict[str, Any]:
    if task is None:
        return anatomy
    try:
        found = await task
    except Exception as e:  # the profile is a refinement; never fail the build over it
        found, err = None, str(e)
    else:
        err = "no outline"
    if found:
        anatomy["profile"] = found
        await emit({"type": "step", "agent": "builder", "tool": "side_profile", "status": "done",
                    "text": f"SAM 3.1 traced the side profile in the {found['view']} view"})
    else:
        await emit({"type": "step", "agent": "builder", "tool": "side_profile", "status": "error",
                    "text": f"No side profile ({err}); the shell keeps a straight depth"})
    return anatomy
