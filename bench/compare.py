"""Benchmark Magneye's agents on Meta Muse Wonder vs Claude Opus 5.5 with everything else identical.

Both runs use the same photo, prompts, SAM 3.1 calls, checks and agent code; only the reasoning model behind the
builder and guide agents changes. Each run is saved as a normal session so it can be opened in the viewer.

    uv run --group bench python bench/compare.py PHOTO [--prompt "..."] [--models wonder,opus] [--with earlier.json]

Needs MODEL_API_KEY (Meta, also used for SAM 3.1) and Anthropic credentials (ANTHROPIC_API_KEY) in the env or .env.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from magneye import guide, meta  # noqa: E402
from magneye.app import DATA, _normalize_image  # noqa: E402
from magneye.builder import build  # noqa: E402
from magneye.checks import geometry_issues  # noqa: E402
from magneye.ground import SegmentCache, ground  # noqa: E402

GUIDE_PROMPTS = [
    "My character keeps walking forward even when I'm not touching the stick",
    "Something's off with this controller in games lately",
    "How does pulling the trigger reach the game?",
]


def _provider(name: str) -> tuple[str, Any]:
    """(model id, usage log) after pointing magneye.meta's chat calls at the chosen provider."""
    if name == "wonder":
        return "muse-wonder-1.0", meta.USAGE
    import claude_adapter

    meta.chat_json, meta.chat = claude_adapter.chat_json, claude_adapter.chat
    return claude_adapter.MODEL, claude_adapter.USAGE


async def run(name: str, photo: bytes, size: tuple[int, int], prompt: str) -> dict[str, Any]:
    model, usage = _provider(name)
    start_usage = len(usage)
    events: list[dict[str, Any]] = []

    async def emit(e: dict[str, Any]) -> None:
        events.append({**e, "t": round(time.time() - t0, 1)})

    t0 = time.time()
    cache = SegmentCache(photo)
    anatomy = await build(image=photo, prompt=prompt, emit=emit, model=model, reasoning="medium", cache=cache)
    t_build = time.time() - t0
    draft_t = next((e["t"] for e in events if e.get("tool") == "draft" and e.get("status") == "done"), None)
    g = await ground(photo, size, anatomy, cache)
    mask = g.pop("object_mask_png")

    guide_runs = []
    messages = guide.opening_messages(anatomy, photo)
    for q in GUIDE_PROMPTS:
        g_events: list[dict[str, Any]] = []

        async def g_emit(e: dict[str, Any]) -> None:
            g_events.append(e)

        t1 = time.time()
        try:
            await guide.respond(anatomy=anatomy, messages=messages, prompt=q, image=photo, cache=cache, emit=g_emit,
                                model=model, ground=g)
            err = None
        except Exception as e:  # keep benchmarking the other prompts
            err = str(e)
        guide_runs.append({
            "prompt": q, "seconds": round(time.time() - t1, 1), "error": err,
            "actions": [e["action"]["type"] + ":" + str(e["action"].get("part_ids") or (e["action"].get("failure") or {}).get("culprit_part_id")
                        or (e["action"].get("scenario") or {}).get("title") or e["action"].get("amount")) for e in g_events if e["type"] == "action"],
            "question": next((e["text"] for e in g_events if e["type"] == "question"), None),
            "answer": next((e["text"] for e in g_events if e["type"] == "answer"), None),
        })

    sid = f"bench{name}{uuid.uuid4().hex[:8]}"
    d = DATA / "sessions" / sid
    d.mkdir(parents=True)
    (d / "image.jpg").write_bytes(photo)
    if mask:
        (d / "mask.png").write_bytes(mask)
    (d / "anatomy.json").write_text(json.dumps(anatomy, indent=1))
    (d / "ground.json").write_text(json.dumps(g, indent=1))
    (d / "session.json").write_text(json.dumps({"prompt": prompt, "size": size, "image_sha": hashlib.sha256(photo).hexdigest()}))
    (d / "trace.jsonl").write_text("".join(json.dumps({k: v for k, v in e.items() if k != "t"}) + "\n"
                                           for e in events if e["type"] in ("step", "thought")))

    calls = usage[start_usage:]
    steps = [e for e in events if e["type"] == "step" and e.get("status") != "running"]
    return {
        "provider": name, "model": model, "session": sid,
        "object": anatomy["object"]["name"], "confidence": anatomy["object"]["confidence"],
        "parts": len(anatomy["parts"]),
        "parts_with_components": sum(1 for p in anatomy["parts"] if p.get("components")),
        "scenarios": len(anatomy["scenarios"]), "failures": len(anatomy["failures"]),
        "sam_checks": sum(1 for e in steps if e["tool"] == "find_in_photo"),
        "edits": sum(1 for e in steps if e["tool"] in ("move_parts", "update_parts", "add_part", "remove_part")),
        "finished": any(e["tool"] == "finish" for e in steps),
        "object_grounded": g["object"] is not None, "parts_grounded": len(g["parts"]),
        "geometry_issues_left": len(geometry_issues(anatomy)),
        "draft_seconds": draft_t, "build_seconds": round(t_build, 1),
        "input_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
        "served_by": sorted({c["model"] for c in calls}),
        "guide": guide_runs,
    }


def table(results: list[dict[str, Any]]) -> str:
    rows = [("", *(r["model"] for r in results))]
    for key in ["object", "confidence", "parts", "parts_with_components", "scenarios", "failures", "sam_checks", "edits",
                "finished", "object_grounded", "parts_grounded", "geometry_issues_left", "draft_seconds", "build_seconds",
                "input_tokens", "output_tokens", "served_by"]:
        rows.append((key, *(str(r[key]) for r in results)))
    for i, q in enumerate(GUIDE_PROMPTS):
        rows.append((f"guide {i + 1}: {q[:38]}…", *(
            f"{g['seconds']}s · {'asked' if g['question'] else ','.join(g['actions']) or 'answered'}" for g in (r["guide"][i] for r in results))))
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--prompt", default="How does this work?")
    ap.add_argument("--models", default="wonder,opus")
    ap.add_argument("--with", dest="previous", help="an earlier results JSON to include in the comparison table")
    args = ap.parse_args()
    photo, size = _normalize_image(Path(args.photo).read_bytes())
    results = json.loads(Path(args.previous).read_text()) if args.previous else []
    for name in args.models.split(","):
        print(f"→ {name} …", flush=True)
        results.append(await run(name, photo, size, args.prompt))
    out = DATA / "bench" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(table(results))
    print(f"\nFull results: {out}")
    for r in results:
        print(f"{r['model']}: http://localhost:8787/#/s/{r['session']}")


if __name__ == "__main__":
    asyncio.run(main())
