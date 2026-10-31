"""Reference views: photos of the same product from other angles, found once research has identified it.

The user's photo stays the front view and defines the coordinate frame. Other angles (side, back, top) let the
draft see depth, the back and the side profile instead of guessing them. Candidates come from Wikimedia Commons
(freely licensed, with author and licence, so the UI can show them) and from the photos on the pages the research
cited (used only as model input and cited by link). Muse Wonder vets them in one call: does it have the
same body shape, which side faces the camera, is the whole object in view.
"""

from __future__ import annotations

import asyncio
import html
import io
import ipaddress
import re
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
from PIL import Image, ImageOps

from . import meta

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# iFixit's repair guides open with photos of the whole device from several sides, for most electronics.
IFIXIT_API = "https://www.ifixit.com/api/2.0"
IFIXIT_GUIDES = 4
IFIXIT_STEPS = 2
MAX_IFIXIT = 6
HEADERS = {"User-Agent": "Magneye/0.1 (https://github.com/unrealdhanush/magneye)"}
FETCH_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
PAGE_LIMIT = 600_000
IMAGE_LIMIT = 6_000_000
MAX_CANDIDATES = 12
MAX_COMMONS = 6
PER_PAGE = 4
MAX_VIEWS = 3
MAX_TOTAL = 6
# One vetting call sees at most this many photos (the user's own plus found ones), at least MIN_FOUND found ones.
MAX_VETTED = 14
MIN_FOUND = 4
VIEWS = ["front", "back", "left", "right", "top", "bottom", "three_quarter", "detail", "other"]
# Views that add something the front photo cannot show, best first.
USEFUL = ["left", "right", "back", "top", "bottom", "three_quarter"]

VET_SYSTEM = """You vet reference photos for Magneye, which builds 3D models of objects. Image 0 is the user's photo; it defines the front view. Images 1 to N are candidate photos: some the user took, some found online.

For each candidate:
- same_shape: true if it shows an object with the same body shape as image 0: the same model, or a variant that differs only in colour, material, dial or finish. A different generation, size or redesigned body is false.
- view: which side of the object faces the camera, relative to image 0. "front" = the side facing the camera in image 0; "left"/"right" = the side that is on the left/right in image 0; "back", "top", "bottom"; "three_quarter" for an angled view showing two sides; "detail" for a close-up of one feature; "other" for anything else (packaging, people using it, no object at all). A photo of a few units side by side (a comparison) still gets the view they are shown from.
- clean: the whole object is in view, not cut off and not mostly hidden by hands or other things.
- upright: the object is the same way up as in image 0 (its top at the top), not lying down or turned.
- note: what it shows that image 0 does not, in under 12 words.
Output only JSON matching the schema."""

VET_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "same_shape": {"type": "boolean"},
            "view": {"type": "string", "enum": VIEWS},
            "clean": {"type": "boolean"},
            "upright": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["index", "same_shape", "view", "clean", "upright", "note"],
    }}},
    "required": ["candidates"],
}

_OG_TAG = re.compile(r"<meta\b[^>]*(?:property|name)\s*=\s*[\"'](?:og:image(?::secure_url)?|twitter:image)[\"'][^>]*>", re.I)
_CONTENT = re.compile(r"content\s*=\s*[\"']([^\"']+)", re.I)
_IMG_SRC = re.compile(r"<img\b[^>]*?\b(?:data-src|src)\s*=\s*[\"']([^\"']+\.(?:jpe?g|png|webp)(?:\?[^\"']*)?)[\"']", re.I)
_NOT_PHOTO = re.compile(r"logo|icon|sprite|avatar|badge|flag|payment|banner|placeholder|\.svg", re.I)
_TAGS = re.compile(r"<[^>]+>")


def _public(url: str) -> bool:
    """Only plain http(s) URLs on public hosts; found URLs never reach this machine's own network."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname or u.hostname == "localhost" or u.hostname.endswith(".local"):
        return False
    try:
        return ipaddress.ip_address(u.hostname).is_global
    except ValueError:
        return True


async def _fetch(client: httpx.AsyncClient, url: str, limit: int) -> tuple[str, bytes] | None:
    """(content type, body) of a public URL, or None if it fails or is larger than `limit` bytes."""
    if not _public(url):
        return None
    try:
        async with client.stream("GET", url) as r:
            if r.status_code != 200 or not _public(str(r.url)):
                return None
            body = b""
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) > limit:
                    return None
            return r.headers.get("content-type", "").split(";")[0].strip(), body
    except (httpx.HTTPError, ValueError):
        return None


def _text(field: dict[str, Any] | None) -> str:
    return html.unescape(_TAGS.sub("", (field or {}).get("value", ""))).strip()


async def _commons(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrnamespace": 6,
              "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 640}
    try:
        r = await client.get(COMMONS_API, params=params)
        pages = (r.json().get("query") or {}).get("pages") or {}
    except (httpx.HTTPError, ValueError):
        return []
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
            continue
        meta_ = info.get("extmetadata") or {}
        out.append({"origin": "commons", "title": p["title"].removeprefix("File:").rsplit(".", 1)[0],
                    "page_url": info.get("descriptionurl"), "image_url": info.get("thumburl") or info.get("url"),
                    "license": _text(meta_.get("LicenseShortName")), "author": _text(meta_.get("Artist"))[:120]})
    return out


async def _ifixit(client: httpx.AsyncClient, queries: list[str]) -> list[dict[str, Any]]:
    """Photos from the opening steps of iFixit guides for the product (teardowns and repairs)."""
    async def search(q: str) -> list[dict[str, Any]]:
        try:
            r = await client.get(f"{IFIXIT_API}/search/{quote(q)}", params={"filter": "guide", "limit": IFIXIT_GUIDES})
            return [g for g in (r.json().get("results") or []) if isinstance(g, dict) and g.get("guideid")]
        except (httpx.HTTPError, ValueError):
            return []
    found = [g for hits in await asyncio.gather(*(search(q) for q in queries)) for g in hits]
    # Teardowns first: they open with the whole device from several sides; repair guides open on one detail.
    found.sort(key=lambda g: g.get("type") != "teardown")
    guides: dict[int, str] = {}
    for g in found:
        guides.setdefault(g["guideid"], g.get("url", ""))
    per_guide = []
    for gid in list(guides)[:IFIXIT_GUIDES]:
        try:
            g = (await client.get(f"{IFIXIT_API}/guides/{gid}")).json()
        except (httpx.HTTPError, ValueError):
            continue
        photos = []
        for step in (g.get("steps") or [])[:IFIXIT_STEPS] if isinstance(g, dict) else []:
            media = step.get("media") if isinstance(step, dict) else None
            # A step holds a list of images, or a video (whose data is a dict), which is skipped.
            data = media.get("data") if isinstance(media, dict) and media.get("type") == "image" else None
            photos += [{"origin": "web", "title": g.get("title", "iFixit guide"), "page_url": g.get("url") or guides[gid],
                        "image_url": m.get("large") or m.get("medium")} for m in data or []
                       if isinstance(m, dict) and (m.get("large") or m.get("medium"))]
        per_guide.append(photos)
    # Taken guide by guide in turn, so one long guide can't crowd out a teardown.
    return [c for rank in zip_longest(*per_guide) for c in rank if c][:MAX_IFIXIT]


async def _page_images(client: httpx.AsyncClient, source: dict[str, str]) -> list[dict[str, Any]]:
    """Photos on a cited page: the preview image it declares (og:image), then the first few in its gallery."""
    got = await _fetch(client, source["url"], PAGE_LIMIT)
    if not got or "html" not in got[0]:
        return []
    page = got[1].decode("utf-8", "ignore")
    tag = _OG_TAG.search(page)
    content = tag and _CONTENT.search(tag.group(0))
    urls = [content.group(1)] if content else []
    urls += [u for u in _IMG_SRC.findall(page) if not _NOT_PHOTO.search(u)]
    found = list(dict.fromkeys(urljoin(source["url"], html.unescape(u)) for u in urls))[:PER_PAGE]
    return [{"origin": "web", "title": source["title"], "page_url": source["url"], "image_url": u} for u in found]


def _normalize(raw: bytes) -> bytes | None:
    """Small upright RGB JPEG, or None for anything that is not a usable photo (icons, logos, broken files)."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        return None
    if min(im.size) < 240:
        return None
    im.thumbnail((640, 640))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()


async def _download(client: httpx.AsyncClient, cand: dict[str, Any]) -> dict[str, Any] | None:
    got = await _fetch(client, cand["image_url"], IMAGE_LIMIT)
    jpeg = got and got[0].startswith("image/") and _normalize(got[1])
    return {**cand, "jpeg": jpeg} if jpeg else None


async def candidates(query: str | None, sources: list[dict[str, str]], names: list[str] | None = None) -> list[dict[str, Any]]:
    """Downloaded candidate photos: Commons first (clean licensing), then iFixit guide photos (whole-device shots
    from several sides), then photos from the pages research cited, taken round-robin so one big gallery cannot
    crowd out the other pages. `names` are other short names for the product, tried on iFixit after the query."""
    searches = list(dict.fromkeys([q for q in [query, *(names or [])] if q]))[:4]
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        commons, fixit, *pages = await asyncio.gather(_commons(client, query) if query else asyncio.sleep(0, []),
                                                      _ifixit(client, searches) if searches else asyncio.sleep(0, []),
                                                      *(_page_images(client, s) for s in sources[:8]))
        web = [c for rank in zip_longest(*pages) for c in rank if c]
        listed = [*commons[:MAX_COMMONS], *fixit, *web]
        unique = list({c["image_url"]: c for c in reversed(listed)}.values())[::-1]
        got = await asyncio.gather(*(_download(client, c) for c in unique[:MAX_CANDIDATES + 6]))
    return [c for c in got if c][:MAX_CANDIDATES]


def pick(cands: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The user's own photos first, each one, in the order they were added; then up to MAX_VIEWS distinct useful views
    found online with the same body shape, for sides the user did not cover (clean shots and Commons photos win
    ties). At most MAX_TOTAL in all."""
    own: dict[int, dict[str, Any]] = {}
    best: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    for v in verdicts:
        i, view = v.get("index", 0) - 1, v.get("view")
        if not (0 <= i < len(cands)):
            continue
        entry = {**cands[i], "view": view, "note": v.get("note", ""), "clean": bool(v.get("clean")), "upright": bool(v.get("upright"))}
        if cands[i]["origin"] == "user":
            # Taken on purpose, so kept unless it shows something else; a close-up of one feature counts.
            if view != "other" and (v.get("same_shape") or view == "detail"):
                own[i] = entry
            continue
        if not v.get("same_shape") or view not in USEFUL:
            continue
        rank = (int(bool(v.get("clean"))), int(cands[i]["origin"] == "commons"))
        if view not in best or rank > best[view][0]:
            best[view] = (rank, entry)
    covered = {c["view"] for c in own.values()}
    found = [c for _, c in sorted(best.values(), key=lambda rv: (-rv[0][0], USEFUL.index(rv[1]["view"]))) if c["view"] not in covered]
    return ([own[i] for i in sorted(own)] + found[:MAX_VIEWS])[:MAX_TOTAL]


async def find(*, image: bytes, query: str | None, sources: list[dict[str, str]], model: str,
               own: Sequence[bytes] = (), names: list[str] | None = None) -> dict[str, Any]:
    """Vetted reference views: {"views": [...with jpeg...], "considered": n}. More photos the user took (`own`)
    are vetted first and come before any photo found online."""
    mine = [{"origin": "user", "own": n, "title": f"Your photo {n + 2}", "page_url": None, "image_url": None, "jpeg": jpeg}
            for n, jpeg in enumerate(_normalize(raw) for raw in own) if jpeg]
    found = await candidates(query, sources, names) if query or sources else []
    # The user's photos come first; found ones fill the rest of one vetting call, with a few always allowed in.
    cands = mine + found[:max(MIN_FOUND, MAX_VETTED - len(mine))]
    if not cands:
        return {"views": [], "considered": 0}
    try:
        verdict = await meta.chat_json(
            model=model, system=VET_SYSTEM, image_jpeg=image, more_images=[c["jpeg"] for c in cands],
            user_text=f"Vet candidates 1 to {len(cands)}.", schema_name="vetting", schema=VET_SCHEMA,
            reasoning_effort="low", max_tokens=8000,
        )
    except meta.MetaAPIError:
        if not mine:
            raise
        # The user took these on purpose, so they still go to the draft unvetted; found photos are dropped.
        unvetted = [{**c, "view": "other_angle", "note": "", "clean": False, "upright": False} for c in mine]
        return {"views": unvetted[:MAX_TOTAL], "considered": len(cands), "unvetted": True}
    return {"views": pick(cands, verdict.get("candidates") or []), "considered": len(cands)}


DIRECTION_SYSTEM = """Image 0 shows an object from its front: the side facing the camera there, with the features you can see (its buttons, controls, screen, dial, lens or face). Image 1 shows the same kind of object. First decide whether image 1 shows it from one side: its front turned toward the left or right edge of the image, so you see its profile (the object may lie down). If image 1 instead looks mostly at its front, back, top or bottom, answer "not_side". Otherwise work out where those front features are in image 1: name two or three features you can see and which edge of the image each is near or faces, then decide which edge of image 1 the object's front is on. Output only JSON."""
DIRECTION_SCHEMA = {"type": "object", "properties": {"features": {"type": "string"},
                                                     "front_edge": {"type": "string", "enum": ["left", "right", "not_side", "unclear"]}},
                    "required": ["features", "front_edge"]}
DIRECTION_VOTES = 5


async def front_edge(image: bytes, side: bytes, model: str) -> str:
    """Whether a photo shows the object's side and which edge its front is on: "left", "right", or "unclear" (not a
    side view, or no agreement). Mirroring is the one thing a side photo's outline can't tell, and one answer is wrong
    often enough to matter, so several are asked at once; answers that don't name what they saw are dropped, and only
    a clear majority counts."""
    async def ask() -> dict[str, Any]:
        return await meta.chat_json(model=model, system=DIRECTION_SYSTEM, image_jpeg=image, more_images=[side],
                                    user_text="Which edge of image 1 is the object's front on?", schema_name="side_direction",
                                    schema=DIRECTION_SCHEMA, reasoning_effort="medium", max_tokens=12000)
    answers = await asyncio.gather(*(ask() for _ in range(DIRECTION_VOTES)), return_exceptions=True)
    votes = [a["front_edge"] for a in answers if isinstance(a, dict) and len(str(a.get("features", ""))) >= 20]
    left, right = votes.count("left"), votes.count("right")
    # Most answers must see a side at all; then one direction must win clearly among them.
    if left + right < 2 or 2 * (left + right) < len(votes):
        return "unclear"
    if max(left, right) >= 2 and min(left, right) * 2 < max(left, right):
        return "left" if left > right else "right"
    return "unclear"


PROFILE_ROWS = 48


def profile_view(picked: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A side view that shows the whole object the right way up, so its outline is the object's side profile."""
    return next((v for v in picked if v["view"] in ("left", "right") and v.get("clean") and v.get("upright")), None)


async def side_profile(view: dict[str, Any], phrase: str) -> dict[str, Any] | None:
    """The object's depth at each height, from SAM 3.1's outline of it in a side view.

    Rows run top to bottom; each is the [back, front] extent as fractions of the object's depth (-0.5 to 0.5, front
    toward the camera of the user's photo). Seen from the right, the front is on the image's left; from the left,
    on its right.
    """
    matches = await meta.segment(view["jpeg"], phrase)
    if not matches:
        return None
    m = max(matches, key=lambda m: (m["confidence"] or 0) * m["coverage"] * m["mask_size"][0] * m["mask_size"][1])
    w, h = m["mask_size"]
    raster = bytes(m["raster"])
    rows: list[list[float] | None] = []
    for k in range(PROFILE_ROWS):
        lo, hi = w, -1
        for r in range(k * h // PROFILE_ROWS, max((k + 1) * h // PROFILE_ROWS, k * h // PROFILE_ROWS + 1)):
            line = raster[r * w:(r + 1) * w]
            first = line.find(1)
            if first >= 0:
                lo, hi = min(lo, first), max(hi, line.rfind(1))
        if hi < 0:
            rows.append(None)
            continue
        u0, u1 = lo / w, (hi + 1) / w
        rows.append([round(0.5 - u1, 3), round(0.5 - u0, 3)] if view["view"] == "right" else [round(u0 - 0.5, 3), round(u1 - 0.5, 3)])
    filled = [r for r in rows if r]
    if len(filled) < PROFILE_ROWS // 2:
        return None
    # Rows the outline misses (thin tips, holes) borrow their nearest filled neighbour.
    for i, r in enumerate(rows):
        if r is None:
            rows[i] = min(((abs(i - j), rows[j]) for j in range(PROFILE_ROWS) if rows[j]), key=lambda d: d[0])[1]
    return {"view": view["view"], "page_url": view["page_url"], "rows": rows}


def cite(view: dict[str, Any]) -> dict[str, Any]:
    """What the anatomy keeps for the UI: Commons photos can be shown with their licence; web photos are only linked;
    the user's own photo is served from their session."""
    extra = {"commons": ["image_url", "license", "author"], "user": ["own"]}.get(view["origin"], [])
    keep = ["origin", "view", "title", "page_url", "note", *extra]
    return {k: view.get(k) for k in keep}
