"""Pin Muse Wonder's inferred layout to the real photo with SAM 3.1 masks."""

from __future__ import annotations

import asyncio
import io
from typing import Any

from PIL import Image

from . import meta

MAX_PHRASES = 10


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _inside(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


async def _safe_segment(image: bytes, phrase: str) -> list[dict[str, Any]] | Exception:
    try:
        return await meta.segment(image, phrase)
    except Exception as e:  # one failed phrase should not sink the whole grounding pass
        return e


async def ground(image: bytes, size: tuple[int, int], anatomy: dict[str, Any]) -> dict[str, Any]:
    """Segment the object and its visible parts, then move visible parts to where SAM found them.

    Mutates `anatomy` in place (center/size of grounded parts) and returns the masks for the viewer,
    plus the object's full-frame mask as PNG bytes under "object_mask_png" (None if not found).
    """
    obj = anatomy["object"]
    visible = [p for p in anatomy["parts"] if p.get("sam_prompt")]
    phrases = list(dict.fromkeys([obj["sam_prompt"], *(p["sam_prompt"] for p in visible)]))[:MAX_PHRASES]
    results = dict(zip(phrases, await asyncio.gather(*(_safe_segment(image, ph) for ph in phrases))))

    errors = {ph: str(r) for ph, r in results.items() if isinstance(r, Exception)}
    object_matches = results.get(obj["sam_prompt"])
    if isinstance(object_matches, Exception) or not object_matches:
        return {"image_size": size, "object": None, "parts": {}, "errors": errors, "object_mask_png": None}

    # The biggest confident match is the object itself (smaller matches are usually reflections or parts).
    target = max(object_matches, key=lambda m: (m["confidence"] or 0.5) * m["coverage"] * _area(m["box"]))
    box = target["box"]
    ow, oh = box[2] - box[0], box[3] - box[1]
    dw, dh, _ = obj["dimensions_mm"]

    def to_px(center_mm: list[float]) -> tuple[float, float]:
        return box[0] + (center_mm[0] / dw + 0.5) * ow, box[1] + (0.5 - center_mm[1] / dh) * oh

    grounded: dict[str, Any] = {}
    taken: set[tuple[str, str]] = set()
    for part in visible:
        matches = results.get(part["sam_prompt"])
        if isinstance(matches, Exception) or not matches:
            continue
        expected = to_px(part["center_mm"])
        candidates = [
            m for m in matches
            if _inside(_center(m["box"]), box) and (part["sam_prompt"], m["object_id"]) not in taken
            and m["box"] != target["box"]
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda m: _dist(_center(m["box"]), expected))
        taken.add((part["sam_prompt"], best["object_id"]))
        cx, cy = _center(best["box"])
        part["center_mm"][0] = round(((cx - box[0]) / ow - 0.5) * dw, 1)
        part["center_mm"][1] = round((0.5 - (cy - box[1]) / oh) * dh, 1)
        bw = (best["box"][2] - best["box"][0]) / ow * dw
        bh = (best["box"][3] - best["box"][1]) / oh * dh
        # Only trust SAM's extent when it is in the same ballpark as Wonder's estimate (rotated parts skew boxes).
        if 0.33 < bw / max(part["size_mm"][0], 0.1) < 3 and 0.33 < bh / max(part["size_mm"][1], 0.1) < 3:
            part["size_mm"][0], part["size_mm"][1] = round(bw, 1), round(bh, 1)
        part["grounded"] = True
        grounded[part["id"]] = {"box": best["box"], "confidence": best["confidence"], "mask_svg": best["mask_svg"],
                                "mask_bounds": best["mask_bounds"], "mask_size": best["mask_size"]}

    return {
        "image_size": size,
        "object": {k: target[k] for k in ("box", "confidence", "mask_svg", "mask_bounds", "mask_size")},
        "parts": grounded,
        "errors": errors,
        "object_mask_png": _full_frame_png(target, size),
    }


def _full_frame_png(match: dict[str, Any], size: tuple[int, int]) -> bytes:
    """Place a box-local SAM raster into a full-frame 8-bit mask image."""
    w, h = match["mask_size"]
    local = Image.frombytes("L", (w, h), bytes(255 if v else 0 for v in match["raster"]))
    left, top, right, bottom = match["mask_bounds"]
    span = (max(1, round(right - left)), max(1, round(bottom - top)))
    if span != (w, h):
        local = local.resize(span, Image.NEAREST)
    frame = Image.new("L", size, 0)
    frame.paste(local, (round(left), round(top)))
    out = io.BytesIO()
    frame.save(out, "PNG")
    return out.getvalue()


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
