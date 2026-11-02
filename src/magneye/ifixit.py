"""iFixit as tools Muse can use: device pages, teardowns and repair guides step by step, the photos of each step,
and the Q&A forum, so the model reads how the real object comes apart instead of guessing its insides.

A teardown researcher runs these tools on its own while the web research runs, and hands the draft a dossier (the
parts in assembly order, how they are held, the disassembly order, what fails) plus the photos of the inside it
found most telling. The guide gets the same tools to answer repair questions from the real steps.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image, ImageOps

from . import meta
from .agent import Emit, Tool, ToolResult, params, run_loop

API = "https://www.ifixit.com/api/2.0"
HEADERS = {"User-Agent": "Magneye/0.1 (https://github.com/unrealdhanush/magneye)"}
TIMEOUT = httpx.Timeout(12.0, connect=5.0)
GUIDE_CHARS = 9000
PAGE_CHARS = 2500
ANSWER_CHARS = 700
MAX_PHOTOS = 6
MAX_KEPT = 4
PHOTO_PX = 768

# Wiki markup: [link|url|text], [url|text|new_window=true], ''bold'', ~~struck~~, ++underlined++.
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_STRUCK = re.compile(r"~~.*?~~")
_EMPHASIS = re.compile(r"'{2,3}|\+\+|\*{2,3}")


def clean(text: str | None) -> str:
    """Plain text from iFixit's wiki markup: links keep their words, struck-through jokes go."""
    def bracket(m: re.Match[str]) -> str:
        words = [w for w in m.group(1).split("|") if w and w not in ("link", "guide", "new_window=true")
                 and not w.startswith(("http", "/")) and "=" not in w and not w.isdigit()]
        return " ".join(words)
    text = _STRUCK.sub("", text or "")
    return re.sub(r"[ \t]+", " ", _EMPHASIS.sub("", _BRACKET.sub(bracket, text))).strip()


async def _get(client: httpx.AsyncClient, path: str, **query: Any) -> Any:
    r = await client.get(f"{API}/{path}", params=query or None)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


async def search(client: httpx.AsyncClient, query: str) -> dict[str, Any]:
    """Devices, guides and Q&A threads matching the query, grouped. iFixit matches every word, so a query that finds
    nothing is retried without its last words ("Oculus Touch controller teardown", then "Oculus Touch controller"...)."""
    words = query.split()
    data: dict[str, Any] = {}
    n = 0
    for n in range(len(words), 0, -1):
        data = await _get(client, f"search/{quote(' '.join(words[:n]))}", limit=20) or {}
        if data.get("results") or n <= 2:
            break
    out: dict[str, Any] = {"devices": [], "guides": [], "questions": []}
    for r in data.get("results") or []:
        kind = r.get("dataType")
        if kind == "wiki" and r.get("namespace", "CATEGORY") == "CATEGORY":
            out["devices"].append({"title": r.get("title"), "summary": clean(r.get("summary") or r.get("text"))[:200]})
        elif kind == "guide":
            out["guides"].append({"guideid": r.get("guideid"), "type": r.get("type"), "title": r.get("title")})
        elif kind == "question":
            out["questions"].append({"postid": r.get("postid"), "title": r.get("title"), "answers": r.get("answer_count")})
    if n < len(words):
        out["matched"] = " ".join(words[:n])
    return out


async def device(client: httpx.AsyncClient, title: str) -> dict[str, Any] | None:
    """A device page: what it is, its guides, and the models above and below it (siblings share guides)."""
    w = await _get(client, f"wikis/CATEGORY/{quote(title.replace(' ', '_'))}")
    if not w:
        return None
    titles = lambda xs: [x.get("title") for x in xs or [] if isinstance(x, dict)]
    return {"title": w.get("title"), "url": w.get("url") or f"https://www.ifixit.com/Device/{quote(title.replace(' ', '_'))}",
            "about": clean(w.get("contents_raw") or w.get("description"))[:PAGE_CHARS],
            "guides": [{"guideid": g.get("guideid"), "type": g.get("type"), "title": g.get("title")} for g in w.get("guides") or []][:20],
            "parent": (titles(w.get("ancestors")) or [None])[0], "models": titles(w.get("children"))[:12],
            "related": titles(w.get("related_wikis"))[:8]}


async def guide(client: httpx.AsyncClient, guideid: int) -> dict[str, Any] | None:
    g = await _get(client, f"guides/{int(guideid)}")
    if not g:
        return None
    steps = []
    for n, s in enumerate(g.get("steps") or [], 1):
        media = s.get("media") or {}
        photos = [m for m in media.get("data") or []] if media.get("type") == "image" else []
        steps.append({"n": n, "title": clean(s.get("title")),
                      "lines": [(int(line.get("level") or 0), line.get("bullet") or "black", clean(line.get("text_raw")))
                                for line in s.get("lines") or [] if clean(line.get("text_raw"))],
                      "photos": [m.get("medium") or m.get("large") or m.get("standard") for m in photos if isinstance(m, dict)],
                      "video": media.get("type") == "video"})
    return {"guideid": g.get("guideid"), "title": g.get("title"), "type": g.get("type"), "device": g.get("category"),
            "url": g.get("url"), "difficulty": g.get("difficulty"), "time": g.get("time_required"),
            "tools": [t.get("text") for t in g.get("tools") or [] if t.get("text")],
            "parts": [p.get("text") for p in g.get("parts") or [] if p.get("text")],
            "intro": clean(g.get("introduction_raw"))[:800], "steps": steps, "conclusion": clean(g.get("conclusion_raw"))[:600]}


def guide_text(g: dict[str, Any], limit: int = GUIDE_CHARS) -> str:
    """A guide as the model reads it: every step's lines (indented by level, with the colour of the marker in that
    step's photos), and how many photos each step has."""
    head = [f"{g['title']} ({g['type']}, device: {g['device']}, difficulty: {g['difficulty'] or 'unknown'}) {g['url']}"]
    if g["tools"]:
        head.append("Tools: " + ", ".join(g["tools"]))
    if g["parts"]:
        head.append("Parts sold for it: " + ", ".join(g["parts"]))
    if g["intro"]:
        head.append("Intro: " + g["intro"])
    body = []
    for s in g["steps"]:
        media = f"{len(s['photos'])} photo{'s' * (len(s['photos']) != 1)}" if s["photos"] else ("video" if s["video"] else "no photo")
        body.append(f"Step {s['n']}{' ' + s['title'] if s['title'] else ''} ({media}):")
        body += ["  " * (level + 1) + ("- " if bullet in ("black", "") else f"- [{bullet}] ") + text
                 for level, bullet, text in s["lines"]]
    if g["conclusion"]:
        body.append("Conclusion: " + g["conclusion"])
    text = "\n".join(head + body)
    return text if len(text) <= limit else text[:limit] + f"\n[... cut at {limit} characters; {len(g['steps'])} steps in all]"


async def answers(client: httpx.AsyncClient, query: str, limit: int = 3) -> list[dict[str, Any]]:
    """The most answered Q&A threads for the query, each with its accepted or best answer."""
    words, found = query.split(), []
    # Questions only, retried with fewer words: a symptom is phrased many ways.
    for n in range(len(words), 0, -1):
        data = await _get(client, f"search/{quote(' '.join(words[:n]))}", filter="question", limit=20) or {}
        found = [q for q in data.get("results") or [] if q.get("answer_count")]
        if found or n <= 2:
            break
    found = sorted(found, key=lambda q: -q["answer_count"])[:limit]
    out = []
    for q in await asyncio.gather(*(_get(client, f"answers/{q['postid']}") for q in found)):
        if not q:
            continue
        replies = q.get("answers") or []
        best = next((a for a in replies if a.get("postid") == q.get("accepted_answerid")), None) or \
            max(replies, key=lambda a: a.get("score") or 0, default=None)
        out.append({"question": q.get("title"), "url": q.get("url"), "asked": clean(q.get("raw_text") or q.get("text"))[:ANSWER_CHARS],
                    "answer": clean((best or {}).get("raw_text") or (best or {}).get("text"))[:ANSWER_CHARS],
                    "accepted": bool(best and best.get("postid") == q.get("accepted_answerid"))})
    return out


async def photo(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        r = await client.get(url)
        r.raise_for_status()
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
    except Exception:
        return None
    im.thumbnail((PHOTO_PX, PHOTO_PX))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()


@dataclass
class Session:
    """What the tools have read in one run: guides (so photos can be asked for by step), the pages cited, and the
    photos the model chose to keep."""

    client: httpx.AsyncClient
    guides: dict[int, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    seen: dict[str, tuple[bytes, dict[str, Any]]] = field(default_factory=dict)
    kept: list[dict[str, Any]] = field(default_factory=list)

    async def guide(self, gid: int) -> dict[str, Any] | None:
        if gid not in self.guides:
            g = await guide(self.client, gid)
            if g is None:
                return None
            self.guides[gid] = g
            self.sources[g["url"]] = f"iFixit: {g['title']}"
        return self.guides[gid]


TOOLS_DOC = """iFixit (teardowns, repair guides, device pages and Q&A for most electronics and many tools and appliances):
- ifixit_search(query): devices, guides (with guideid and type: teardown, replacement, disassembly...) and Q&A threads. Search the product's plain name ("Oculus Touch", "Canon EOS 600D", "Dyson V8"); try its siblings and family names when the exact model has nothing.
- ifixit_device(title): a device page: what it is, its guides, the models it belongs to and the ones under it (siblings often share a body and guides).
- ifixit_guide(guideid): every step of a guide in order. Lines tagged [red], [orange], [yellow]... point at the marker of that colour drawn in that step's photos.
- ifixit_photos(guideid, steps): look at the photos of up to six steps at once. Teardowns show the inside layer by layer, often ending with every part laid out; repair guides show how one part is reached.
- ifixit_answers(query): the forum's best-answered questions on a symptom or fault, with their answers. Keep it to two or three words, the product family and the symptom ("Quest controller drift", "Kindle won't charge"); check each thread is about this product."""


def tools(s: Session) -> list[Tool]:
    """The iFixit tools, sharing what they read through `s`."""

    async def ifixit_search(a: dict[str, Any]) -> ToolResult:
        q = str(a.get("query") or "").strip()
        found = await search(s.client, q)
        n = sum(len(found[k]) for k in ("devices", "guides", "questions"))
        return ToolResult(json.dumps(found) if n else f"Nothing on iFixit for {q!r}; try a shorter or family name.",
                          f"iFixit: {n} results for “{found.get('matched', q)}”" if n else f"iFixit: nothing for “{q}”", ok=bool(n))

    async def ifixit_device(a: dict[str, Any]) -> ToolResult:
        d = await device(s.client, str(a.get("title") or ""))
        if d is None:
            return ToolResult("No such device page; use a title exactly as ifixit_search listed it.", "iFixit: no such device", ok=False)
        s.sources[d["url"]] = f"iFixit: {d['title']}"
        return ToolResult(json.dumps(d), f"Read the iFixit page for {d['title']} ({len(d['guides'])} guides)")

    async def ifixit_guide(a: dict[str, Any]) -> ToolResult:
        g = await s.guide(int(a.get("guideid") or 0))
        if g is None:
            return ToolResult("No such guide id.", "iFixit: no such guide", ok=False)
        return ToolResult(guide_text(g), f"Read “{g['title']}” ({len(g['steps'])} steps)")

    async def ifixit_photos(a: dict[str, Any]) -> ToolResult:
        g = await s.guide(int(a.get("guideid") or 0))
        if g is None:
            return ToolResult("No such guide id.", "iFixit: no such guide", ok=False)
        wanted = [(st, k, url) for n in a.get("steps") or [] for st in g["steps"] if st["n"] == n
                  for k, url in enumerate(st["photos"][:3], 1)][:MAX_PHOTOS]
        if not wanted:
            return ToolResult("Those steps have no photos; ifixit_guide lists how many each step has.", "iFixit: no photos there", ok=False)
        got = await asyncio.gather(*(photo(s.client, url) for _, _, url in wanted))
        attach, ids = [], []
        for (st, k, url), jpeg in zip(wanted, got):
            if jpeg is None:
                continue
            pid = f"{g['guideid']}-{st['n']}-{k}"
            s.seen[pid] = (jpeg, {"guide": g["title"], "url": g["url"], "step": st["n"]})
            ids.append(pid)
            attach += [{"type": "text", "text": f"Photo {pid}: “{g['title']}”, step {st['n']}, photo {k}"}, meta.image_part(jpeg)]
        return ToolResult(f"Photos {', '.join(ids)} are attached in the next message." if ids else "The photos didn't load.",
                          f"Looked at {len(ids)} photos from “{g['title']}”", ok=bool(ids), attach=attach)

    async def ifixit_answers(a: dict[str, Any]) -> ToolResult:
        q = str(a.get("query") or "").strip()
        found = await answers(s.client, q)
        for f in found:
            s.sources[f["url"]] = f"iFixit Answers: {f['question']}"
        return ToolResult(json.dumps(found) if found else "No answered questions match; try fewer words.",
                          f"iFixit Answers: {len(found)} threads on “{q}”", ok=bool(found))

    return [
        Tool("ifixit_search", "Search iFixit for devices, guides and Q&A threads.", params({"query": {"type": "string"}}), ifixit_search),
        Tool("ifixit_device", "Read an iFixit device page.", params({"title": {"type": "string"}}), ifixit_device),
        Tool("ifixit_guide", "Read every step of an iFixit guide.", params({"guideid": {"type": "integer"}}), ifixit_guide),
        Tool("ifixit_photos", "Look at the photos of some steps of an iFixit guide (up to six photos).",
             params({"guideid": {"type": "integer"}, "steps": {"type": "array", "items": {"type": "integer"}}}), ifixit_photos),
        Tool("ifixit_answers", "Search iFixit Answers for a fault or symptom and read the best answers.",
             params({"query": {"type": "string"}}), ifixit_answers),
    ]


RESEARCH_SYSTEM = f"""You are Magneye's teardown researcher. Magneye builds an interactive 3D model of the object in the photo, inside and out, from the evidence it can find. Your evidence is iFixit, where people have taken this kind of object apart step by step and photographed every layer.

{TOOLS_DOC}
- keep_photos(photos): choose up to {MAX_KEPT} photos you have looked at for the 3D modeller, each with what it shows (which parts, seen from which side, which layer of the teardown). Prefer photos that show where the inner parts sit in the shell and against each other, and a layout of all the parts.
- finish(dossier): hand over your findings.

Identify the exact product in the photo first. Then find its teardown (or, if there is none, its disassembly and replacement guides, or the teardown of the closest sibling that shares the body; say which). Read the guides, look at the photos that open the shell, reveal each layer and lay the parts out, and read the Q&A on what fails.

The dossier is what a modeller builds the inside from, so be concrete:
1. Sources: which guides and pages you used, and whether they cover this exact model or a sibling.
2. Internal parts in assembly order from the front (the side facing the camera in the photo) to the back. For each: what it is, where it sits (against which shell wall, under which control, beside what), approximate size in mm (scale it against things of known size in the photos: an AA cell is 50.5 x 14.5 mm, a coin cell 20 mm, a USB-C port 9 mm wide, screws, and the device's overall size), shape, colour and material; chips by part number.
3. How each part is held and joined: screw types, clips, glue, magnets, springs, solder, connectors, and every cable from where to where.
4. Disassembly order exactly as the guides do it, outermost first: this is the path to each part.
5. What fails and why, from repair guides and answers, with the part at fault.
Write only what the sources show or clearly imply, and say when a size is your estimate. Under 900 words.

Work fast, at most six turns: search and read the device page or guide list in the first turn; read the one or two best guides and look at their key photos next (several tool calls per turn); keep photos and finish last."""


async def research(*, image: bytes, prompt: str, emit: Emit, model: str) -> dict[str, Any] | None:
    """The teardown dossier for the object in the photo: {"text", "sources", "photos"}, or None when iFixit has
    nothing on it or the search fails."""
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        s = Session(client)
        out: dict[str, Any] = {}

        async def keep_photos(a: dict[str, Any]) -> ToolResult:
            for item in a.get("photos") or []:
                pid = str(item.get("id") or "")
                if pid in s.seen and len(s.kept) < MAX_KEPT and all(k["id"] != pid for k in s.kept):
                    jpeg, where = s.seen[pid]
                    s.kept.append({"id": pid, "jpeg": jpeg, "shows": str(item.get("shows") or ""), **where})
            return ToolResult(f"Kept {[k['id'] for k in s.kept]}.", f"Kept {len(s.kept)} photos of the inside for the modeller")

        async def finish(a: dict[str, Any]) -> ToolResult:
            out["text"] = str(a.get("dossier") or "").strip()
            return ToolResult("ok", "Wrote the teardown dossier", stop=True)

        hand_over = [
            Tool("keep_photos", "Keep photos you have seen for the 3D modeller, with what each shows.",
                 params({"photos": {"type": "array", "items": params({"id": {"type": "string"}, "shows": {"type": "string"}})}}), keep_photos),
            Tool("finish", "Hand over the teardown dossier.", params({"dossier": {"type": "string"}}), finish),
        ]
        messages = [{"role": "system", "content": RESEARCH_SYSTEM},
                    {"role": "user", "content": [meta.image_part(image), {"type": "text", "text":
                        f"User prompt: {prompt.strip() or 'Show me how this works.'}\n\nFind how this object is built inside on iFixit."}]}]
        try:
            await run_loop(agent="builder", model=model, messages=messages, tools=tools(s) + hand_over, emit=emit, max_turns=8,
                           reasoning_effort="low", until="finish", max_tokens=8000)
            if "text" not in out and s.sources:
                # Out of turns while still reading: what it has read is worth handing over.
                messages.append({"role": "user", "content": "Stop researching now: keep the best photos you have seen and write the dossier with finish."})
                await run_loop(agent="builder", model=model, messages=messages, tools=hand_over, emit=emit, max_turns=2,
                               reasoning_effort="low", until="finish", max_tokens=8000)
        except (meta.MetaAPIError, httpx.HTTPError) as e:
            await emit({"type": "step", "agent": "builder", "tool": "ifixit", "status": "error", "text": f"iFixit research stopped ({e})"})
        if not out.get("text") or not s.sources:
            return None
        return {"text": out["text"], "sources": [{"url": u, "title": t} for u, t in s.sources.items()], "photos": s.kept}
