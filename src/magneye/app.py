"""Magneye HTTP API. The browser orchestrates the stages so each one can show progress."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import anatomy as A
from . import meta
from .ground import ground

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
    _write_json(d / "session.json", {"prompt": prompt, "size": size, "image_sha": hashlib.sha256(jpeg).hexdigest(),
                                     "history": []})
    return {"id": sid, "image_url": f"/api/sessions/{sid}/image.jpg", "size": size, "prompt": prompt}


@app.get("/api/sessions/{sid}/image.jpg")
def session_image(sid: str) -> FileResponse:
    return FileResponse(_session_dir(sid) / "image.jpg", media_type="image/jpeg")


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    return {"id": sid, "image_url": f"/api/sessions/{sid}/image.jpg", **s,
            "anatomy": _read_json(d / "anatomy.json"), "ground": _read_json(d / "ground.json"),
            "has_mesh": (d / "shell.glb").exists()}


@app.post("/api/sessions/{sid}/anatomy")
async def build_anatomy(sid: str) -> dict[str, Any]:
    """Muse Wonder looks at the photo and infers the object's parts, scenarios and failure modes."""
    d = _session_dir(sid)
    s = _read_json(d / "session.json")
    # Same photo + same prompt -> reuse the earlier inference (keeps demos fast and cheap).
    cache = DATA / "cache" / f"{s['image_sha'][:24]}-{hashlib.sha256(s['prompt'].encode()).hexdigest()[:12]}.json"
    if cache.exists():
        result = _read_json(cache)
    else:
        try:
            result = await meta.chat_json(
                model=MODEL, system=A.ANATOMY_SYSTEM, user_text=A.anatomy_user_text(s["prompt"]),
                image_jpeg=(d / "image.jpg").read_bytes(), schema_name="anatomy", schema=A.ANATOMY_SCHEMA,
                reasoning_effort=REASONING, max_tokens=24000,
            )
        except Exception as e:
            raise _api_error(e) from e
        result = A.sanitize(result)
        cache.parent.mkdir(parents=True, exist_ok=True)
        _write_json(cache, result)
    _write_json(d / "anatomy.json", result)
    return result


@app.post("/api/sessions/{sid}/ground")
async def ground_anatomy(sid: str) -> dict[str, Any]:
    """SAM 3.1 finds the object and its visible parts in the photo; visible parts snap to their real positions."""
    d = _session_dir(sid)
    anatomy = _read_json(d / "anatomy.json")
    if anatomy is None:
        raise HTTPException(409, "build the anatomy first")
    s = _read_json(d / "session.json")
    try:
        g = await ground((d / "image.jpg").read_bytes(), tuple(s["size"]), anatomy)
    except Exception as e:
        raise _api_error(e) from e
    mask = g.pop("object_mask_png")
    if mask:
        (d / "mask.png").write_bytes(mask)
    _write_json(d / "anatomy.json", anatomy)
    _write_json(d / "ground.json", g)
    return {"anatomy": anatomy, "ground": g}


class Ask(BaseModel):
    prompt: str


@app.post("/api/sessions/{sid}/ask")
async def ask(sid: str, body: Ask) -> dict[str, Any]:
    """Muse Wonder answers a prompt grounded in the anatomy and picks what the viewer should play."""
    d = _session_dir(sid)
    anatomy = _read_json(d / "anatomy.json")
    if anatomy is None:
        raise HTTPException(409, "build the anatomy first")
    s = _read_json(d / "session.json")
    try:
        answer = await meta.chat_json(
            model=MODEL, system=A.ANSWER_SYSTEM,
            user_text=A.answer_user_text(json.dumps(anatomy), body.prompt, s["history"]),
            image_jpeg=None, schema_name="answer", schema=A.ANSWER_SCHEMA, reasoning_effort="low", max_tokens=6000,
        )
    except Exception as e:
        raise _api_error(e) from e
    answer = A.sanitize_answer(answer, anatomy)
    # New scenarios/failures become part of the object's anatomy for the rest of the session.
    if answer.get("new_scenario"):
        anatomy["scenarios"].append(answer["new_scenario"])
        answer["scenario_id"] = answer["new_scenario"]["id"]
    if answer.get("new_failure"):
        anatomy["failures"].append(answer["new_failure"])
        answer["failure_id"] = answer["new_failure"]["id"]
    _write_json(d / "anatomy.json", anatomy)
    s["history"].append(body.prompt)
    _write_json(d / "session.json", s)
    return answer


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
