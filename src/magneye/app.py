"""Magneye HTTP API. The browser orchestrates the stages so each one can show progress."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import guide, meta
from .agent import Emit
from .builder import build
from .ground import SegmentCache, ground

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MAGNEYE_DATA", ROOT / ".data"))
WEB_DIST = ROOT / "web" / "dist"
MODEL = os.environ.get("MAGNEYE_MODEL", "muse-wonder-1.0")
REASONING = os.environ.get("MAGNEYE_REASONING", "medium")
MAX_SIDE = 1600

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
async def create_session(image: UploadFile = File(...), prompt: str = Form("")) -> dict[str, Any]:
    jpeg, size = _normalize_image(await image.read())
    sid = uuid.uuid4().hex[:16]
    d = DATA / "sessions" / sid
    d.mkdir(parents=True)
    (d / "image.jpg").write_bytes(jpeg)
    _write_json(d / "session.json", {"prompt": prompt, "size": size, "image_sha": hashlib.sha256(jpeg).hexdigest()})
    return {"id": sid, "image_url": f"/api/sessions/{sid}/image.jpg", "size": size, "prompt": prompt}


@app.get("/api/sessions/{sid}/image.jpg")
def session_image(sid: str) -> FileResponse:
    return FileResponse(_session_dir(sid) / "image.jpg", media_type="image/jpeg")


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    trace = d / "trace.jsonl"
    return {"id": sid, "image_url": f"/api/sessions/{sid}/image.jpg", **s,
            "anatomy": _read_json(d / "anatomy.json"), "ground": _read_json(d / "ground.json"),
            "has_mesh": (d / "shell.glb").exists(),
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


@app.post("/api/sessions/{sid}/build")
async def build_session(sid: str) -> StreamingResponse:
    """Builder agent: Muse Wonder drafts the anatomy, then verifies it with SAM 3.1, geometry checks and a
    look at its own layout; finally SAM 3.1 grounds the visible parts for the viewer."""
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    image = (d / "image.jpg").read_bytes()

    async def work(emit: Emit) -> None:
        cache = SegmentCache(image)
        anatomy = _read_json(d / "anatomy.json")
        # Same photo + same prompt -> reuse the earlier grounded anatomy (keeps demos fast and cheap).
        key = f"{s['image_sha'][:24]}-{hashlib.sha256(s['prompt'].encode()).hexdigest()[:12]}.json"
        cached = DATA / "cache" / key
        if anatomy is None:
            if cached.exists():
                anatomy = _read_json(cached)
                await emit({"type": "step", "agent": "builder", "tool": "cache", "status": "done",
                            "text": "Reused the verified anatomy for this photo and prompt"})
            else:
                anatomy = await build(image=image, prompt=s["prompt"], emit=emit, model=MODEL, reasoning=REASONING, cache=cache)
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


@app.post("/api/sessions/{sid}/ask")
async def ask(sid: str, body: Ask) -> StreamingResponse:
    """Guide agent: answers by driving the viewer, looking closer at the photo, or asking a question first."""
    d = _session_dir(sid)
    anatomy = _read_json(d / "anatomy.json")
    if anatomy is None:
        raise HTTPException(409, "build the anatomy first")
    image = (d / "image.jpg").read_bytes()
    messages = _read_json(d / "guide.json") or guide.opening_messages(anatomy, image)

    async def work(emit: Emit) -> None:
        await emit({"type": "user", "text": body.prompt})
        try:
            await guide.respond(anatomy=anatomy, messages=messages, prompt=body.prompt, image=image,
                                cache=SegmentCache(image), emit=emit, model=MODEL, ground=_read_json(d / "ground.json"))
        finally:
            # Scenarios/failures the guide created become part of the object for the rest of the session.
            _write_json(d / "anatomy.json", anatomy)
            _write_json(d / "guide.json", messages)
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
