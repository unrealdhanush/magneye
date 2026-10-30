"""Deterministic checks the builder agent runs against its own anatomy draft."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from .ground import Frame


def _bounds(p: dict[str, Any]) -> tuple[list[float], list[float]]:
    lo = [c - s / 2 for c, s in zip(p["center_mm"], p["size_mm"])]
    hi = [c + s / 2 for c, s in zip(p["center_mm"], p["size_mm"])]
    return lo, hi


def _volume(p: dict[str, Any]) -> float:
    x, y, z = (max(v, 0.1) for v in p["size_mm"])
    return x * y * z


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    (alo, ahi), (blo, bhi) = _bounds(a), _bounds(b)
    v = 1.0
    for i in range(3):
        v *= max(0.0, min(ahi[i], bhi[i]) - max(alo[i], blo[i]))
    return v


def geometry_issues(anatomy: dict[str, Any]) -> list[str]:
    """Physical-plausibility problems: parts outside the object, internal parts sharing space,
    and disassembly paths that dig outward."""
    issues: list[str] = []
    dims = anatomy["object"]["dimensions_mm"]
    parts = anatomy["parts"]
    by_id = {p["id"]: p for p in parts}
    for p in parts:
        lo, hi = _bounds(p)
        for i, axis in enumerate("xyz"):
            limit = dims[i] / 2 * 1.1
            if lo[i] < -limit or hi[i] > limit:
                issues.append(f"{p['id']} sticks out of the object along {axis} "
                              f"(spans {lo[i]:.0f}..{hi[i]:.0f} mm, object is ±{dims[i] / 2:.0f} mm).")
                break
    internal = [p for p in parts if p["kind"] == "internal"]
    for i, a in enumerate(internal):
        for b in internal[i + 1:]:
            shared = _overlap(a, b) / min(_volume(a), _volume(b))
            if shared > 0.5:
                issues.append(f"{a['id']} and {b['id']} occupy the same space ({shared:.0%} of the smaller one).")
    for f in anatomy.get("failures", []):
        culprit = by_id.get(f["culprit_part_id"])
        if not culprit:
            continue
        for pid in f["access_path"]:
            if pid in by_id and by_id[pid]["layer"] > culprit["layer"]:
                issues.append(f"Failure {f['id']}: access path removes {pid} (layer {by_id[pid]['layer']}) "
                              f"to reach {culprit['id']} (layer {culprit['layer']}), which is shallower.")
    return issues


def layout_overlay(photo_jpeg: bytes, frame: Frame, anatomy: dict[str, Any]) -> bytes:
    """Draw every part's front-view footprint over the photo so the model can see its own layout."""
    im = Image.open(io.BytesIO(photo_jpeg)).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle(frame.box, outline=(255, 255, 255, 255), width=3)
    for p in sorted(anatomy["parts"], key=lambda q: -q["layer"]):
        (x0, y0, _), (x1, y1, _) = _bounds(p)
        (px0, py0), (px1, py1) = frame.to_px(x0, y1), frame.to_px(x1, y0)
        colour = (64, 200, 224) if p["kind"] == "exterior" else (255, 159, 10)
        draw.rectangle([px0, py0, px1, py1], outline=colour + (255,), fill=colour + (40,), width=2)
        draw.text((px0 + 3, py0 + 2), p["id"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()
