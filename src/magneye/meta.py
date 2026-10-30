"""Thin async client for the Meta Model API (https://api.meta.ai/v1)."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    decode_mask_to_svg_path,
    image_segmentation_format,
)

BASE_URL = "https://api.meta.ai/v1"
TIMEOUT = httpx.Timeout(300.0, connect=15.0)


class MetaAPIError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise MetaAPIError("MODEL_API_KEY is not set. Copy .env.example to .env and add your Meta Model API key.")
    return key


def _headers(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_key()}", **extra}


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


async def chat_json(
    *,
    model: str,
    system: str,
    user_text: str,
    image_jpeg: bytes | None,
    schema_name: str,
    schema: dict[str, Any],
    reasoning_effort: str = "medium",
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """One Chat Completions call that must return JSON matching `schema`."""
    content: list[dict[str, Any]] = []
    if image_jpeg is not None:
        content.append({"type": "image_url", "image_url": {"url": data_url(image_jpeg), "detail": "high"}})
    content.append({"type": "text", "text": user_text})
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": False},
        },
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{BASE_URL}/chat/completions", headers=_headers(), json=body)
    if r.status_code >= 400:
        raise MetaAPIError(f"chat/completions {r.status_code}: {r.text[:500]}")
    message = r.json()["choices"][0]["message"]
    text = message.get("content") or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise MetaAPIError(f"{model} did not return valid JSON: {text[:300]}") from e


def image_part(jpeg: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url(jpeg), "detail": "high"}}


async def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    reasoning_effort: str = "medium",
    max_tokens: int = 8000,
) -> dict[str, Any]:
    """One Chat Completions turn with tools. Returns the assistant message (content and/or tool_calls)."""
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{BASE_URL}/chat/completions", headers=_headers(), json=body)
    if r.status_code >= 400:
        raise MetaAPIError(f"chat/completions {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]


async def segment(image_jpeg: bytes, phrase: str) -> list[dict[str, Any]]:
    """SAM 3.1 text-prompted segmentation of one noun phrase.

    Returns one dict per match: box in source pixels, confidence, and the mask
    as an SVG path (box-local pixel coordinates), its coverage fraction and the
    raw 0/1 raster (row-major, mask_size) for callers that need pixels.
    """
    body = {
        "model": "sam-3.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": data_url(image_jpeg)},
                ],
            }
        ],
        "metadata": {"include_confidence": "true"},
        "stream": True,
    }
    parser = image_segmentation_format().create_parser()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/responses", headers=_headers(Accept="text/event-stream"), json=body
        ) as r:
            if r.status_code >= 400:
                raise MetaAPIError(f"sam-3.1 {r.status_code}: {(await r.aread()).decode()[:500]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                if event.get("type") == "response.output_text.delta":
                    parser.push(event.get("delta", ""), emit=False)
                elif event.get("type") in ("response.failed", "error"):
                    raise MetaAPIError(f"sam-3.1 failed: {payload[:500]}")
    result = parser.finish(CompletedOutcome()).result

    boxes = {rec.object_id: rec for rec in result.records if isinstance(rec, SegmentationBoxRecord)}
    matches = []
    for rec in result.records:
        if not isinstance(rec, SegmentationMaskRecord):
            continue
        raster = decode_mask_to_raster(rec.mask)
        box = boxes.get(rec.object_id)
        b = box or rec.bounds
        matches.append(
            {
                "object_id": rec.object_id,
                "box": [b.left, b.top, b.right, b.bottom],
                "confidence": rec.confidence if rec.confidence is not None else getattr(box, "confidence", None),
                "mask_bounds": [rec.bounds.left, rec.bounds.top, rec.bounds.right, rec.bounds.bottom],
                "mask_size": [rec.mask.width, rec.mask.height],
                "mask_svg": decode_mask_to_svg_path(rec.mask),
                "coverage": sum(raster) / max(1, len(raster)),
                "raster": raster,
            }
        )
    matches.sort(key=lambda m: -(m["confidence"] or 0))
    return matches


async def transcribe(wav: bytes) -> str:
    """Muse Voice Transcribe on a 16/24 kHz mono 16-bit WAV clip."""
    request = json.dumps({"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"})
    files = {
        "request": (None, request, "application/json"),
        "audio": ("clip.wav", wav, "audio/wav"),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{BASE_URL}/asr/transcribe", headers=_headers(Accept="application/json"), files=files)
    if r.status_code >= 400:
        raise MetaAPIError(f"asr/transcribe {r.status_code}: {r.text[:500]}")
    return r.json().get("transcript", "").strip()
