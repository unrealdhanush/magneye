"""Fit the model to the photo with numbers, not guesses.

Muse Wonder estimates where the photo was taken from and lays the parts out in its head; nothing ties the parts SAM
can't see to the photo. Here the viewpoint is searched instead: for each candidate (azimuth, elevation, roll) the
parts are projected into the photo plane, scaled and centred onto SAM's outline of the object, and scored by how
well the two silhouettes overlap (intersection over union). Ties go to the angle nearest Wonder's estimate. The
best viewpoint is applied and the layout is rescaled and recentred so the parts fill the object as SAM sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose
from .checks import _hull

GRID = 96
# Search ranges (degrees). Elevation is kept to photos taken from roughly level to straight down or slightly below.
COARSE = {"azimuth": range(-180, 180, 30), "elevation": range(-30, 91, 15), "roll": range(-180, 180, 15)}
# How much one degree away from Wonder's estimate costs, in overlap: a mild preference that only breaks near-ties.
PRIOR_PER_DEG = 0.0006
MIN_GAIN = 0.03
# A layout that already covers the object this well is left as it is.
GOOD_ENOUGH = 0.8


@dataclass
class Outline:
    """SAM's mask of the whole object on a small grid over the object's box (with a margin), in photo pixels."""
    mask: np.ndarray
    x0: float
    y0: float
    cell: float
    box: list[float]

    @classmethod
    def from_match(cls, match: dict[str, Any]) -> Outline:
        box = match["box"]
        w, h = box[2] - box[0], box[3] - box[1]
        side = max(w, h) * 1.5
        x0, y0 = (box[0] + box[2]) / 2 - side / 2, (box[1] + box[3]) / 2 - side / 2
        cell = side / GRID
        mw, mh = match["mask_size"]
        ml, mt, mr, mb = match["mask_bounds"]
        raster = np.frombuffer(bytes(match["raster"]), dtype=np.uint8).reshape(mh, mw)
        cx = x0 + (np.arange(GRID) + 0.5) * cell
        cy = y0 + (np.arange(GRID) + 0.5) * cell
        u, v = np.meshgrid(np.floor((cx - ml) / max(mr - ml, 1e-6) * mw).astype(int),
                           np.floor((cy - mt) / max(mb - mt, 1e-6) * mh).astype(int))
        inside = (u >= 0) & (u < mw) & (v >= 0) & (v < mh)
        mask = np.zeros((GRID, GRID), bool)
        mask[inside] = raster[v[inside], u[inside]] > 0
        return cls(mask, x0, y0, cell, box)


def _corners(p: dict[str, Any]) -> np.ndarray:
    return np.array(pose.box_corners(p["center_mm"], p["size_mm"], p.get("rotation_deg")), float)


def _projected(parts: list[dict[str, Any]], camera: dict[str, float]) -> list[np.ndarray]:
    """Each part's box corners in the photo plane (x right, y up; mm) from the given viewpoint."""
    rot = np.array(pose.rotation({"object": {"camera": camera}}))
    return [(_corners(p) @ rot.T)[:, :2] for p in parts]


def _raster(polys_px: list[list[tuple[float, float]]]) -> np.ndarray:
    im = Image.new("1", (GRID, GRID), 0)
    draw = ImageDraw.Draw(im)
    for poly in polys_px:
        if len(poly) >= 3:
            draw.polygon(poly, fill=1)
    return np.array(im, bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _aligned(projected: list[np.ndarray], outline: Outline) -> np.ndarray:
    """The parts' silhouette with their projected bounds scaled and centred onto SAM's box."""
    pts = np.concatenate(projected)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    bw, bh = outline.box[2] - outline.box[0], outline.box[3] - outline.box[1]
    k = math.sqrt((bw * bh) / (span[0] * span[1]))
    mid = (lo + hi) / 2
    bx, by = (outline.box[0] + outline.box[2]) / 2, (outline.box[1] + outline.box[3]) / 2
    to_grid = lambda xy: [((bx + (x - mid[0]) * k - outline.x0) / outline.cell, (by - (y - mid[1]) * k - outline.y0) / outline.cell) for x, y in xy]
    return _raster([to_grid(_hull([tuple(p) for p in pr])) for pr in projected])


def coverage(anatomy: dict[str, Any], outline: Outline) -> float:
    """How well the parts, as currently placed and seen through the current viewpoint, cover SAM's outline: the
    overlap the viewer and the modeler's overlay show (no refitting)."""
    projected = _projected(anatomy["parts"], anatomy["object"].get("camera") or {})
    span = pose.photo_extent(anatomy)
    ow, oh = outline.box[2] - outline.box[0], outline.box[3] - outline.box[1]
    to_grid = lambda xy: [((outline.box[0] + (x / span[0] + 0.5) * ow - outline.x0) / outline.cell,
                           (outline.box[1] + (0.5 - y / span[1]) * oh - outline.y0) / outline.cell) for x, y in xy]
    return _iou(_raster([to_grid(_hull([tuple(p) for p in pr])) for pr in projected]), outline.mask)


def _angle_gap(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(abs((x - y + 180) % 360 - 180) for x, y in zip(a, b))


def search(anatomy: dict[str, Any], outline: Outline) -> tuple[tuple[float, float, float], float]:
    """The viewpoint whose silhouette, fitted onto SAM's box, best overlaps SAM's outline (coarse, then fine)."""
    cam = anatomy["object"].get("camera") or {}
    guess = (float(cam.get("azimuth_deg") or 0), float(cam.get("elevation_deg") or 0), float(cam.get("roll_deg") or 0))
    parts = anatomy["parts"]

    def score(view: tuple[float, float, float]) -> tuple[float, float]:
        projected = _projected(parts, dict(zip(("azimuth_deg", "elevation_deg", "roll_deg"), view)))
        iou = _iou(_aligned(projected, outline), outline.mask)
        return iou - PRIOR_PER_DEG * _angle_gap(view, guess), iou

    candidates = [guess] + [(a, e, r) for a in COARSE["azimuth"] for e in COARSE["elevation"] for r in COARSE["roll"]]
    best = max(candidates, key=lambda v: score(v)[0])
    for step in (10, 5):
        around = [(best[0] + da, best[1] + de, best[2] + dr) for da in (-step, 0, step) for de in (-step, 0, step) for dr in (-step, 0, step)]
        best = max(around, key=lambda v: score(v)[0])
    return best, score(best)[1]


def fit(anatomy: dict[str, Any], outline: Outline) -> dict[str, Any] | None:
    """Apply the best viewpoint and fill the object with the layout. Mutates `anatomy`; returns a report, or None
    when the fit would not clearly improve on what is there."""
    before = coverage(anatomy, outline)
    if before >= GOOD_ENOUGH:
        return None
    view, _ = search(anatomy, outline)
    old = dict(anatomy["object"].get("camera") or {})
    anatomy["object"]["camera"] = dict(zip(("azimuth_deg", "elevation_deg", "roll_deg"), (round(v, 1) for v in view)))
    parts = anatomy["parts"]
    # Rescale the layout about its own centre so its projected bounds fill the object's projected extent, then
    # centre it on the object (photo plane), keeping depth.
    camera = anatomy["object"]["camera"]
    bounds = lambda: (lambda pts: (pts.min(axis=0), pts.max(axis=0)))(np.concatenate(_projected(parts, camera)))
    lo, hi = bounds()
    span = pose.photo_extent(anatomy)
    factor = float(np.clip(math.sqrt((span[0] * span[1]) / max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-6)), 0.6, 1.7))
    centre = np.mean([p["center_mm"] for p in parts], axis=0)
    saved = [(list(p["center_mm"]), list(p["size_mm"])) for p in parts]
    for p in parts:
        p["center_mm"] = [round(float(c + (v - c) * factor), 1) for v, c in zip(p["center_mm"], centre)]
        p["size_mm"] = [round(float(v * factor), 1) for v in p["size_mm"]]
    lo, hi = bounds()
    mid = (lo + hi) / 2
    shift = pose.shift_in_photo(anatomy, [-float(mid[0]), -float(mid[1]), 0.0])
    for p in parts:
        p["center_mm"] = [round(v + d, 1) for v, d in zip(p["center_mm"], shift)]
    after = coverage(anatomy, outline)
    if after < before + MIN_GAIN:
        anatomy["object"]["camera"] = old
        for p, (c, s) in zip(parts, saved):
            p["center_mm"], p["size_mm"] = c, s
        return None
    return {"before": before, "after": after, "camera": anatomy["object"]["camera"], "previous": old, "scale": factor}
