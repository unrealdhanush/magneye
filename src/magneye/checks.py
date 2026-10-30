"""Deterministic checks the builder agent runs against its own anatomy draft."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from . import pose
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
    for p in parts:
        if p["shape"] in ("disc", "ring", "gear", "spiral"):
            a, b = sorted(p["size_mm"])[1:]
            if a > 0 and b / a > 1.3:
                issues.append(f"{p['id']} is a round {p['shape']} but its diameters differ ({a:.0f} vs {b:.0f} mm); "
                              f"a SAM box may have caught neighbouring parts.")
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


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull (monotone chain), so a box seen from an angle is drawn as its true outline."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(pts[::-1])


def layout_overlay(photo_jpeg: bytes, frame: Frame, anatomy: dict[str, Any]) -> bytes:
    """Draw every part's box as the camera sees it over the photo, so the model can see its own layout."""
    im = Image.open(io.BytesIO(photo_jpeg)).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle(frame.box, outline=(255, 255, 255, 255), width=3)
    for p in sorted(anatomy["parts"], key=lambda q: -q["layer"]):
        outline = [frame.to_px(x, y) for x, y in _hull(pose.box_corners_in_photo(anatomy, p["center_mm"], p["size_mm"]))]
        colour = (64, 200, 224) if p["kind"] == "exterior" else (255, 159, 10)
        draw.polygon(outline, outline=colour + (255,), fill=colour + (40,), width=2)
        left, top = min(x for x, _ in outline), min(y for _, y in outline)
        draw.text((left + 3, top + 2), p["id"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()
