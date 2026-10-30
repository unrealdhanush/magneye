"""Magneye HTTP API. The browser orchestrates the stages so each one can show progress."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import anatomy as A
from . import guide, meta
from . import model as model_kit
from .agent import Emit
from .builder import build
from .clean import clear_hands
from .ground import SegmentCache, ground

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MAGNEYE_DATA", ROOT / ".data"))
WEB_DIST = ROOT / "web" / "dist"
MODEL = os.environ.get("MAGNEYE_MODEL", "muse-wonder-1.0")
REASONING = os.environ.get("MAGNEYE_REASONING", "medium")
MAX_SIDE = 1600
MAX_EXTRAS = 5
# A public deployment spends the operator's Meta API budget, so each visitor gets this many agent runs (full builds
# and guide questions) per hour; cached builds are free. 0 turns the limit off.
RUNS_PER_HOUR = int(os.environ.get("MAGNEYE_RUNS_PER_HOUR", "30"))
_runs: dict[str, list[float]] = {}


def _spend(request: Request) -> None:
    if RUNS_PER_HOUR <= 0:
        return
    forwarded = request.headers.get("fly-client-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    who = forwarded or (request.client.host if request.client else "unknown")
    now = time.time()
    recent = [t for t in _runs.get(who, []) if now - t < 3600]
    if len(recent) >= RUNS_PER_HOUR:
        raise HTTPException(429, "That's the limit of builds and questions for this hour. Try again a little later.")
    _runs[who] = [*recent, now]

app = FastAPI(title="Magneye")


def _session_dir(sid: str) -> Path:
    if not sid.isalnum():
        raise HTTPException(400, "bad session id")
    d = DATA / "sessions" / sid
    if not d.is_dir():
        raise HTTPException(404, "session not found")
    return d


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1))


def _normalize_image(raw: bytes) -> tuple[bytes, tuple[int, int]]:
    """Apply EXIF rotation, flatten alpha onto white, cap the long side, re-encode as JPEG."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception as e:
        raise HTTPException(400, f"not an image: {e}") from e
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue(), im.size


def _api_error(e: Exception) -> HTTPException:
    return HTTPException(502, str(e)) if isinstance(e, meta.MetaAPIError) else HTTPException(500, repr(e))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL, "has_key": bool(os.environ.get("MODEL_API_KEY")),
            "sam3d": bool(os.environ.get("SAM3D_URL"))}


@app.post("/api/sessions")
async def create_session(image: UploadFile = File(...), prompt: str = Form(""),
                         extras: list[UploadFile] = File(default=[])) -> dict[str, Any]:
    """A new session from the user's main photo (the front view), up to MAX_EXTRAS more photos of the object from
    other angles or close up, and a prompt."""
    if len(extras) > MAX_EXTRAS:
        raise HTTPException(400, f"at most {MAX_EXTRAS} extra photos")
    jpeg, size = _normalize_image(await image.read())
    more = [_normalize_image(await f.read())[0] for f in extras]
    sid = uuid.uuid4().hex[:16]
    d = DATA / "sessions" / sid
    d.mkdir(parents=True)
    (d / "image.jpg").write_bytes(jpeg)
    for n, extra in enumerate(more, 1):
        (d / f"extra-{n}.jpg").write_bytes(extra)
    session = {"prompt": prompt, "size": size, "image_sha": hashlib.sha256(jpeg).hexdigest(),
               "extra_shas": [hashlib.sha256(extra).hexdigest() for extra in more]}
    _write_json(d / "session.json", session)
    return {"id": sid, "image_url": f"/api/sessions/{sid}/image.jpg", "size": size, "prompt": prompt}


@app.get("/api/sessions/{sid}/image.jpg")
def session_image(sid: str) -> FileResponse:
    return FileResponse(_session_dir(sid) / "image.jpg", media_type="image/jpeg")


def _extra_photos(d: Path) -> list[Path]:
    # Sessions from before multiple photos kept a single "side.jpg".
    found = sorted(d.glob("extra-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    return found or [p for p in [d / "side.jpg"] if p.exists()]


@app.get("/api/sessions/{sid}/extra/{n}.jpg")
def session_extra_image(sid: str, n: int) -> FileResponse:
    photos = _extra_photos(_session_dir(sid))
    if not 1 <= n <= len(photos):
        raise HTTPException(404, "no such photo")
    return FileResponse(photos[n - 1], media_type="image/jpeg")


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    trace = d / "trace.jsonl"
    # A cleaned photo gets its own URL, so a browser that cached the photo as taken fetches the new one.
    image_url = f"/api/sessions/{sid}/image.jpg" + ("?v=cleaned" if s.get("cleaned") else "")
    return {"id": sid, **s, "image_url": image_url,
            "anatomy": _read_json(d / "anatomy.json"), "ground": _read_json(d / "ground.json"),
            "has_mesh": (d / "shell.glb").exists(),
            "original_url": f"/api/sessions/{sid}/original.jpg" if (d / "original.jpg").exists() else None,
            "trace": [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []}


def _stream(d: Path, work: Callable[[Emit], Awaitable[None]]) -> StreamingResponse:
    """Run an agent and stream its events as server-sent events, recording the trace for reloads."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        queue.put_nowait(event)
        if event["type"] in TRACED:
            with (d / "trace.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")

    async def runner() -> None:
        try:
            await work(emit)
        except Exception as e:
            await emit({"type": "error", "message": str(e) if isinstance(e, meta.MetaAPIError) else repr(e)})
        finally:
            queue.put_nowait(None)

    async def events():
        task = asyncio.create_task(runner())
        while (event := await queue.get()) is not None:
            yield f"data: {json.dumps(event)}\n\n"
        await task

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


TRACED = {"user", "step", "thought", "question", "answer", "error"}


@app.get("/api/sessions/{sid}/original.jpg")
def session_original_image(sid: str) -> FileResponse:
    path = _session_dir(sid) / "original.jpg"
    if not path.exists():
        raise HTTPException(404, "the photo was used as taken")
    return FileResponse(path, media_type="image/jpeg")


async def _clear_one(photo: Path, keep_as: Path) -> bool | None:
    """Remove a hand covering the object from one photo, in place, keeping the original as `keep_as`. The cleaned copy
    is cached by the original's hash, so the same photo is never edited twice. True if cleaned, False if nothing
    covered the object, None if the check failed."""
    original = photo.read_bytes()
    cached = DATA / "cache" / f"{hashlib.sha256(original).hexdigest()[:24]}.clean.jpg"
    if cached.exists():
        cleaned = cached.read_bytes()
    else:
        with Image.open(io.BytesIO(original)) as im:
            size = im.size
        try:
            found = await clear_hands(original, size)
        except meta.MetaAPIError:
            return None
        if not found:
            return False
        cleaned = found["image"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(cleaned)
    keep_as.write_bytes(original)
    photo.write_bytes(_normalize_image(cleaned)[0])
    return True


async def _clear_photos(d: Path, s: dict[str, Any], emit: Emit) -> None:
    """Clear hands off the main photo and every extra photo, all at once, before anything reads them. Decided once per
    session; the originals are kept (original.jpg, original-extra-N.jpg). Sessions built before this step are left
    as they are."""
    if "cleaned" in s or (d / "anatomy.json").exists():
        return
    photos = [(d / "image.jpg", d / "original.jpg")] + [(p, d / f"original-{p.name}") for p in _extra_photos(d)]
    await emit({"type": "step", "agent": "builder", "tool": "clean_photo", "status": "running"})
    results = await asyncio.gather(*(_clear_one(photo, keep) for photo, keep in photos))
    cleaned, failed = sum(r is True for r in results), sum(r is None for r in results)
    s["cleaned"] = {"by": "muse-image-1.0", "removed": "hand"} if results[0] else None
    s["extras_cleaned"] = [n for n, r in enumerate(results[1:], 1) if r]
    if results[0]:
        with Image.open(d / "image.jpg") as im:
            s["size"] = list(im.size)
    _write_json(d / "session.json", s)
    if len(photos) == 1:
        text = ("Muse Image removed the hand covering the object and restored the part it hid" if cleaned
                else "Nothing covering the object")
    else:
        text = (f"Muse Image removed the hand from {cleaned} of {len(photos)} photos and restored what it hid" if cleaned
                else "Nothing covering the object in any photo")
    if failed:
        text += f"; couldn't check {failed} (used as taken)"
    await emit({"type": "step", "agent": "builder", "tool": "clean_photo", "status": "error" if failed == len(photos) else "done",
                "text": text})


@app.post("/api/sessions/{sid}/build")
async def build_session(sid: str, request: Request) -> StreamingResponse:
    """Builder agent: Muse Wonder drafts the anatomy, then verifies it with SAM 3.1, geometry checks and a
    look at its own layout; finally SAM 3.1 grounds the visible parts for the viewer."""
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    extra = "".join(s.get("extra_shas") or [s.get("side_sha", "")])
    key = f"{s['image_sha'][:24]}-{hashlib.sha256((s['prompt'] + extra).encode()).hexdigest()[:12]}.json"
    cached = DATA / "cache" / key
    if not (d / "anatomy.json").exists() and not cached.exists():
        _spend(request)

    async def work(emit: Emit) -> None:
        await _clear_photos(d, s, emit)
        image = (d / "image.jpg").read_bytes()
        own = [p.read_bytes() for p in _extra_photos(d)]
        cache = SegmentCache(image)
        anatomy = _read_json(d / "anatomy.json")
        # Same photos + same prompt -> reuse the earlier grounded anatomy (keeps demos fast and cheap).
        if anatomy is None:
            if cached.exists():
                anatomy = _read_json(cached)
                await emit({"type": "step", "agent": "builder", "tool": "cache", "status": "done",
                            "text": "Reused the verified anatomy for this photo and prompt"})
            else:
                anatomy = await build(image=image, prompt=s["prompt"], emit=emit, model=MODEL, reasoning=REASONING,
                                      cache=cache, own=own)
            # The user's own photos are served from this session (a cached anatomy may come from another).
            for ref in anatomy.get("references", []):
                if ref["origin"] == "user":
                    ref["image_url"] = f"/api/sessions/{sid}/extra/{ref.get('own', 0) + 1}.jpg"
            _write_json(d / "anatomy.json", anatomy)
        g = _read_json(d / "ground.json")
        if g is None:
            await emit({"type": "step", "agent": "builder", "tool": "ground", "status": "running"})
            g = await ground(image, tuple(s["size"]), anatomy, cache)
            mask = g.pop("object_mask_png")
            if mask:
                (d / "mask.png").write_bytes(mask)
            _write_json(d / "anatomy.json", anatomy)
            _write_json(d / "ground.json", g)
            if g["object"] and not cached.exists():
                # Only anatomy that SAM could ground is worth reusing.
                cached.parent.mkdir(parents=True, exist_ok=True)
                _write_json(cached, anatomy)
            await emit({"type": "step", "agent": "builder", "tool": "ground", "status": "done" if g["object"] else "error",
                        "text": f"SAM 3.1 grounded {len(g['parts'])} visible parts" if g["object"]
                        else "Object not found in photo; showing the inferred layout"})
        await emit({"type": "anatomy", "anatomy": anatomy})
        await emit({"type": "ground", "ground": g})
        await emit({"type": "done"})

    return _stream(d, work)


class Ask(BaseModel):
    prompt: str


@app.get("/api/sessions/{sid}/models")
def session_models(sid: str) -> dict[str, Any]:
    """Meshes of the parts that have geometry recipes, built by Manifold and cached until a recipe changes."""
    d = _session_dir(sid)
    anatomy = _read_json(d / "anatomy.json") or {"parts": []}
    recipes = {p["id"]: p["model"] for p in anatomy["parts"] if p.get("model")}
    key = hashlib.sha256(json.dumps(recipes, sort_keys=True).encode()).hexdigest()[:16]
    cached = _read_json(d / "models.json")
    if cached and cached.get("key") == key:
        return cached
    meshes: dict[str, Any] = {}
    for pid, recipe in recipes.items():
        try:
            meshes[pid] = model_kit.encode(model_kit.build(recipe))
        except model_kit.RecipeError:
            continue  # an edited part whose recipe no longer builds is drawn as its primitive
    out = {"key": key, "parts": meshes}
    _write_json(d / "models.json", out)
    return out


REQUIRED_PART_KEYS = {"id", "name", "kind", "material", "shape", "size_mm", "center_mm", "rotation_deg", "layer"}


@app.put("/api/sessions/{sid}/anatomy")
def save_anatomy(sid: str, body: dict[str, Any]) -> dict[str, Any]:
    """Save the user's edits from the blueprint. References to removed parts are dropped (scenario steps, failures,
    connections, grounding), and the revision moves on so the guide hears about the change."""
    d = _session_dir(sid)
    current = _read_json(d / "anatomy.json")
    if current is None:
        raise HTTPException(409, "build the anatomy first")
    parts = body.get("parts")
    if not isinstance(parts, list) or not all(isinstance(p, dict) and REQUIRED_PART_KEYS <= p.keys() for p in parts):
        raise HTTPException(400, f"every part needs {sorted(REQUIRED_PART_KEYS)}")
    if len({p["id"] for p in parts}) != len(parts):
        raise HTTPException(400, "part ids must be unique")
    anatomy = A.sanitize({**current, **{k: v for k, v in body.items() if k in ("parts", "scenarios", "failures", "connections")}})
    anatomy["revision"] = current.get("revision", 0) + 1
    _write_json(d / "anatomy.json", anatomy)
    g = _read_json(d / "ground.json")
    if g:
        ids = {p["id"] for p in parts}
        g["parts"] = {k: v for k, v in g["parts"].items() if k in ids}
        _write_json(d / "ground.json", g)
    return anatomy


@app.post("/api/sessions/{sid}/ask")
async def ask(sid: str, body: Ask, request: Request) -> StreamingResponse:
    """Guide agent: answers by driving the viewer, looking closer at the photo, or asking a question first."""
    d = _session_dir(sid)
    anatomy = _read_json(d / "anatomy.json")
    if anatomy is None:
        raise HTTPException(409, "build the anatomy first")
    _spend(request)
    image = (d / "image.jpg").read_bytes()
    messages = _read_json(d / "guide.json") or guide.opening_messages(anatomy, image)

    # When the model was edited since the guide last spoke, tell it what the parts are now.
    told = (_read_json(d / "guide_revision.json") or {}).get("revision", 0)
    prompt = body.prompt
    if anatomy.get("revision", 0) != told:
        current = [{"id": p["id"], "name": p["name"], "center_mm": p["center_mm"], "size_mm": p["size_mm"]} for p in anatomy["parts"]]
        prompt = f"(The model was edited since you last looked. Current parts: {json.dumps(current)})\n\n{body.prompt}"

    async def work(emit: Emit) -> None:
        await emit({"type": "user", "text": body.prompt})
        try:
            await guide.respond(anatomy=anatomy, messages=messages, prompt=prompt, image=image,
                                cache=SegmentCache(image), emit=emit, model=MODEL, ground=_read_json(d / "ground.json"))
        finally:
            # Scenarios/failures the guide created become part of the object for the rest of the session.
            _write_json(d / "anatomy.json", anatomy)
            _write_json(d / "guide.json", messages)
            _write_json(d / "guide_revision.json", {"revision": anatomy.get("revision", 0)})
        await emit({"type": "anatomy", "anatomy": anatomy})
        await emit({"type": "done"})

    return _stream(d, work)


@app.post("/api/sessions/{sid}/reconstruct")
async def reconstruct(sid: str) -> dict[str, Any]:
    """SAM 3D Objects turns the photo + object mask into a 3D shell (GLB), on a GPU endpoint we deploy to Modal."""
    d = _session_dir(sid)
    url = os.environ.get("SAM3D_URL", "").strip()
    if not url:
        raise HTTPException(501, "SAM3D_URL is not configured; the viewer keeps the mask-extruded shell")
    if not (d / "mask.png").exists():
        raise HTTPException(409, "ground the object first")
    files = {"image": ("image.jpg", (d / "image.jpg").read_bytes(), "image/jpeg"),
             "mask": ("mask.png", (d / "mask.png").read_bytes(), "image/png")}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        r = await client.post(url, files=files)
    if r.status_code >= 400:
        raise HTTPException(502, f"sam3d {r.status_code}: {r.text[:300]}")
    (d / "shell.glb").write_bytes(r.content)
    return {"mesh_url": f"/api/sessions/{sid}/shell.glb"}


@app.get("/api/sessions/{sid}/shell.glb")
def shell(sid: str) -> FileResponse:
    path = _session_dir(sid) / "shell.glb"
    if not path.exists():
        raise HTTPException(404, "no mesh yet")
    return FileResponse(path, media_type="model/gltf-binary")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict[str, str]:
    """Muse Voice Transcribe for spoken prompts (the browser sends 16 kHz mono WAV)."""
    try:
        return {"text": await meta.transcribe(await audio.read())}
    except Exception as e:
        raise _api_error(e) from e


if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("magneye.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8787)), reload=False)
