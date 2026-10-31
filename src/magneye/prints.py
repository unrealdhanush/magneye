"""3D-printing references: printable models made for the product, found on Printables.

People who own the object model it or things that fit it exactly: replicas and scans, replacement shells, and
accessories (a grip cover traces its shell, a stand holds its exact outline). Their geometry gives real measurements
and a shape to sculpt from. Only Creative Commons models are downloaded (their licence allows reuse with credit);
every model used is cited with its licence and author. Thingiverse and others refuse downloads without an account,
so this sticks to Printables' public API.
"""

from __future__ import annotations

import asyncio
import io
import re
import struct
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw

from . import meta

API = "https://api.printables.com/graphql/"
HEADERS = {"User-Agent": "Magneye/0.1 (https://github.com/unrealdhanush/magneye)", "Content-Type": "application/json"}
TIMEOUT = httpx.Timeout(20.0, connect=5.0)
SEARCH = "query($q: String!){ searchPrints2(query: $q, limit: 8) { items { id name } } }"
DETAIL = """query($id: ID!){ print(id: $id){ id name slug license{ abbreviation name } user{ publicUsername handle }
  stls{ id name fileSize } } }"""
LINK = """mutation($id: ID!, $printId: ID!){ getDownloadLink(id: $id, printId: $printId, fileType: stl, source: model_detail){
  ok output{ link } } }"""
MAX_MODELS = 6
MAX_FILE_BYTES = 15_000_000
FILLED_TRIANGLES = 80_000
RELATIONS = ["replica", "fits_over", "fits_into", "accessory", "unrelated"]

VET_SYSTEM = """You vet 3D-printable models for Magneye, which builds 3D models of the object in the user's photo. Image 0 is the photo. Each later image shows one printable model drawn from three sides (front, top, side) with its size in millimetres.

For each model say:
- relation: "replica" (a model or scan of the object itself or its housing), "fits_over" (made to wrap its outside: a cover, grip, skin; its inner shape traces the object's shell), "fits_into" (the object sits in it: a stand, dock, holster; its cradle traces the object's outline), "accessory" (attaches but tells little about its shape), or "unrelated".
- same_product: true if it is made for this exact product (the same generation and model), false for a related one.
- object_size_mm: the object's own width, height and depth (x its right, y up, z out of its front) as this model implies, or null.
- note: what this model shows about the object's shape, in under 20 words.
Output only JSON matching the schema."""

VET_SCHEMA = {
    "type": "object",
    "properties": {"models": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "relation": {"type": "string", "enum": RELATIONS},
            "same_product": {"type": "boolean"},
            "object_size_mm": {"type": ["array", "null"], "items": {"type": "number"}},
            "note": {"type": "string"},
        },
        "required": ["index", "relation", "same_product", "object_size_mm", "note"],
    }}},
    "required": ["models"],
}


def parse_stl(data: bytes) -> np.ndarray:
    """The vertices (N x 3, mm) of a binary or ASCII STL."""
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return rec["v"].reshape(-1, 3).astype(np.float32)
    text = data.decode("ascii", "ignore")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not nums:
        raise ValueError("not an STL")
    return np.array(nums, dtype=np.float32)


def drawing(v: np.ndarray, size: int = 220) -> Image.Image:
    """Three silhouettes (front x/y, top x/z, side z/y) with the overall size, for the vetting model to look at.
    Models up to FILLED_TRIANGLES are drawn as filled triangles; denser scans as a plot of their vertices."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = hi - lo
    sheet = Image.new("RGB", (size * 3 + 32, size + 34), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    k = (size - 16) / max(float(span.max()), 1e-6)
    for i, (u, w, label) in enumerate([(0, 1, "front"), (0, 2, "top"), (2, 1, "side")]):
        xs = (v[:, u] - lo[u]) * k + 8
        ys = size - 8 - (v[:, w] - lo[w]) * k
        tile = Image.new("L", (size, size), 255)
        if len(v) // 3 <= FILLED_TRIANGLES:
            # Shaded, painted back to front, so cradles, holes and cut-outs read as depth.
            d = 3 - u - w
            pos = np.stack([xs, ys, v[:, d]], axis=1).reshape(-1, 3, 3)
            normal = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
            shade = (60 + 170 * np.abs(normal @ np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]))).astype(int)
            pen = ImageDraw.Draw(tile)
            for j in np.argsort(pos[:, :, 2].mean(axis=1)):
                pen.polygon([tuple(q) for q in pos[j, :, :2]], fill=int(shade[j]))
        else:
            grid = np.full((size, size), 255, np.uint8)
            grid[ys.astype(int).clip(0, size - 1), xs.astype(int).clip(0, size - 1)] = 0
            tile = Image.fromarray(grid)
        sheet.paste(tile.convert("RGB"), (i * (size + 16), 24))
        draw.text((i * (size + 16) + 4, 6), label, fill=(40, 40, 40))
    draw.text((4, size + 22), f"{span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f} mm", fill=(40, 40, 40))
    return sheet


async def _gql(client: httpx.AsyncClient, query: str, variables: dict[str, Any]) -> dict[str, Any] | None:
    try:
        r = await client.post(API, json={"query": query, "variables": variables})
        return (r.json() or {}).get("data")
    except (httpx.HTTPError, ValueError):
        return None


async def _model(client: httpx.AsyncClient, pid: str) -> dict[str, Any] | None:
    """A Creative Commons model with its first one or two STL files downloaded and parsed."""
    data = await _gql(client, DETAIL, {"id": pid})
    p = (data or {}).get("print")
    if not p or not (p.get("license") or {}).get("abbreviation", "").startswith("CC"):
        return None
    files = [f for f in p.get("stls") or [] if 0 < (f.get("fileSize") or 0) <= MAX_FILE_BYTES][:2]
    meshes = []
    for f in files:
        link = ((await _gql(client, LINK, {"id": f["id"], "printId": pid}) or {}).get("getDownloadLink") or {}).get("output") or {}
        if not link.get("link"):
            continue
        try:
            r = await client.get(link["link"])
            meshes.append((f["name"], parse_stl(r.content)))
        except (httpx.HTTPError, ValueError):
            continue
    if not meshes:
        return None
    user = p.get("user") or {}
    return {"id": pid, "title": p["name"], "url": f"https://www.printables.com/model/{pid}-{p.get('slug', '')}",
            "license": p["license"]["abbreviation"], "author": user.get("publicUsername") or user.get("handle") or "",
            "files": [n for n, _ in meshes], "vertices": meshes[0][1]}


async def search(queries: list[str]) -> list[dict[str, Any]]:
    """Printable models for the product: searched by each short query, then fetched (Creative Commons only)."""
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        found = await asyncio.gather(*(_gql(client, SEARCH, {"q": q}) for q in queries))
        ids = list(dict.fromkeys(i["id"] for d in found for i in ((d or {}).get("searchPrints2") or {}).get("items") or []))
        models = await asyncio.gather(*(_model(client, pid) for pid in ids[:MAX_MODELS * 2]))
    return [m for m in models if m][:MAX_MODELS]


async def find(*, image: bytes, queries: list[str], model: str) -> dict[str, Any]:
    """Vetted printable references: {"refs": [...with drawing jpeg...], "considered": n}."""
    found = await search(queries)
    if not found:
        return {"refs": [], "considered": 0}
    drawings = []
    for m in found:
        out = io.BytesIO()
        drawing(m["vertices"]).save(out, "JPEG", quality=85)
        drawings.append(out.getvalue())
    verdict = await meta.chat_json(
        model=model, system=VET_SYSTEM, image_jpeg=image, more_images=drawings,
        user_text="Models in order: " + "; ".join(f"{i + 1}. {m['title']} ({', '.join(m['files'])})" for i, m in enumerate(found)),
        schema_name="print_vetting", schema=VET_SCHEMA, reasoning_effort="low", max_tokens=6000,
    )
    refs = []
    for v in verdict.get("models") or []:
        i = v.get("index", 0) - 1
        if not (0 <= i < len(found)) or v.get("relation") in (None, "unrelated", "accessory"):
            continue
        m = found[i]
        lo, hi = m["vertices"].min(axis=0), m["vertices"].max(axis=0)
        refs.append({**{k: m[k] for k in ("id", "title", "url", "license", "author", "files")},
                     "relation": v["relation"], "same_product": bool(v.get("same_product")),
                     "object_size_mm": v.get("object_size_mm"), "note": v.get("note", ""),
                     "model_size_mm": [round(float(x), 1) for x in hi - lo], "drawing": drawings[i]})
    # Exact-product references first; replicas before things that fit around or under the object.
    refs.sort(key=lambda r: (not r["same_product"], RELATIONS.index(r["relation"])))
    return {"refs": refs[:3], "considered": len(found)}


def cite(ref: dict[str, Any]) -> dict[str, Any]:
    return {k: ref[k] for k in ("title", "url", "license", "author", "files", "relation", "same_product", "object_size_mm", "note", "model_size_mm")}
