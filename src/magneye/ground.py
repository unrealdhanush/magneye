"""Pin Muse Wonder's inferred layout to the real photo with SAM 3.1 masks."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from . import meta

MAX_PHRASES = 10

Matches = list[dict[str, Any]]


class SegmentCache:
    """SAM 3.1 results per phrase for one photo, shared by the builder agent and final grounding."""

    def __init__(self, image: bytes) -> None:
        self.image = image
        self._results: dict[str, Matches | Exception] = {}

    async def get(self, phrase: str) -> Matches | Exception:
        key = phrase.strip().lower()
        if key not in self._results:
            try:
                self._results[key] = await meta.segment(self.image, phrase)
            except Exception as e:  # one failed phrase should not sink a whole grounding pass
                self._results[key] = e
        return self._results[key]


@dataclass
class Frame:
    """Maps between photo pixels and the object's millimetre frame (photo = front view, +y up)."""

    box: list[float]
    dims: list[float]

    def to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        ow, oh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        return self.box[0] + (x_mm / self.dims[0] + 0.5) * ow, self.box[1] + (0.5 - y_mm / self.dims[1]) * oh

    def to_mm(self, px: float, py: float) -> tuple[float, float]:
        ow, oh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        return ((px - self.box[0]) / ow - 0.5) * self.dims[0], (0.5 - (py - self.box[1]) / oh) * self.dims[1]

    def box_to_mm(self, box: list[float]) -> dict[str, list[float]]:
        """A pixel box as centre and size in millimetres (x, y only)."""
        cx, cy = self.to_mm((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        ow, oh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        w = (box[2] - box[0]) / ow * self.dims[0]
        h = (box[3] - box[1]) / oh * self.dims[1]
        return {"center_mm": [round(cx, 1), round(cy, 1)], "size_mm": [round(w, 1), round(h, 1)]}


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _inside(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def pick_object(matches: Matches) -> dict[str, Any]:
    """The biggest confident match is the object itself (smaller matches are usually reflections or parts)."""
    return max(matches, key=lambda m: (m["confidence"] or 0.5) * m["coverage"] * _area(m["box"]))


async def object_frame(cache: SegmentCache, anatomy: dict[str, Any]) -> tuple[Frame, dict[str, Any]] | None:
    matches = await cache.get(anatomy["object"]["sam_prompt"])
    if isinstance(matches, Exception) or not matches:
        return None
    target = pick_object(matches)
    return Frame(target["box"], anatomy["object"]["dimensions_mm"]), target


def part_matches(frame: Frame, target: dict[str, Any], matches: Matches) -> Matches:
    """Matches whose centre lies on the object, excluding the object's own mask."""
    return [m for m in matches if _inside(_center(m["box"]), frame.box) and m["box"] != target["box"]]


async def ground(image: bytes, size: tuple[int, int], anatomy: dict[str, Any],
                 cache: SegmentCache | None = None) -> dict[str, Any]:
    """Segment the object and its visible parts, then move visible parts to where SAM found them.

    Mutates `anatomy` in place (center/size of grounded parts) and returns the masks for the viewer,
    plus the object's full-frame mask as PNG bytes under "object_mask_png" (None if not found).
    """
    cache = cache or SegmentCache(image)
    obj = anatomy["object"]
    visible = [p for p in anatomy["parts"] if p.get("sam_prompt")]
    phrases = list(dict.fromkeys([obj["sam_prompt"], *(p["sam_prompt"] for p in visible)]))[:MAX_PHRASES]
    results = dict(zip(phrases, await asyncio.gather(*(cache.get(ph) for ph in phrases))))

    errors = {ph: str(r) for ph, r in results.items() if isinstance(r, Exception)}
    found = await object_frame(cache, anatomy)
    if found is None:
        return {"image_size": size, "object": None, "parts": {}, "errors": errors, "object_mask_png": None}
    frame, target = found

    grounded: dict[str, Any] = {}
    taken: set[tuple[str, str]] = set()
    for part in visible:
        matches = results.get(part["sam_prompt"])
        if isinstance(matches, Exception) or not matches:
            continue
        expected = frame.to_px(part["center_mm"][0], part["center_mm"][1])
        candidates = [m for m in part_matches(frame, target, matches) if (part["sam_prompt"], m["object_id"]) not in taken]
        if not candidates:
            continue
        best = min(candidates, key=lambda m: _dist(_center(m["box"]), expected))
        taken.add((part["sam_prompt"], best["object_id"]))
        mm = frame.box_to_mm(best["box"])
        part["center_mm"][0], part["center_mm"][1] = mm["center_mm"]
        bw, bh = mm["size_mm"]
        # Only trust SAM's extent when it is in the same ballpark as Wonder's estimate (rotated parts skew boxes).
        if 0.33 < bw / max(part["size_mm"][0], 0.1) < 3 and 0.33 < bh / max(part["size_mm"][1], 0.1) < 3:
            part["size_mm"][0], part["size_mm"][1] = bw, bh
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
