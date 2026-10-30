"""Where the photo was taken from.

Parts live in the object's own frame: x to its right, y up, z out of its front, as it sits in normal use. The photo
may show it from any angle, so Muse Wonder estimates the camera's azimuth (around the object, positive toward its
right side), elevation (above it, positive looking down) and roll (the photo turned, positive counterclockwise).
These helpers carry positions between the object frame and the photo plane, treating the photo as orthographic:
x right and y up in the photo, z toward the camera. A straight-on photo has every angle at 0 and the two frames
coincide.
"""

from __future__ import annotations

import math
from typing import Any

Vec = list[float]
Matrix = list[list[float]]

# Angles below this count as straight-on: sizes read off the photo can be trusted and the frames nearly coincide.
FRONTAL_DEG = 12


def angles(anatomy: dict[str, Any]) -> tuple[float, float, float]:
    cam = anatomy["object"].get("camera") or {}
    return tuple(float(cam.get(k) or 0) for k in ("azimuth_deg", "elevation_deg", "roll_deg"))  # type: ignore[return-value]


def _mul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation(anatomy: dict[str, Any]) -> Matrix:
    """Object frame -> photo frame: turn the object so the camera's side faces +z, tip its top toward the camera by
    the elevation, then apply the roll."""
    a, e, r = (math.radians(v) for v in angles(anatomy))
    ry = [[math.cos(-a), 0, math.sin(-a)], [0, 1, 0], [-math.sin(-a), 0, math.cos(-a)]]
    rx = [[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]]
    rz = [[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]]
    return _mul(rz, _mul(rx, ry))


def apply(m: Matrix, v: Vec) -> Vec:
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def transpose(m: Matrix) -> Matrix:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def frontal(anatomy: dict[str, Any]) -> bool:
    return all(abs(v) < FRONTAL_DEG for v in angles(anatomy))


def extent(m: Matrix, size: Vec) -> Vec:
    """How far an axis-aligned box of `size` (object frame) spans along the photo's x, y and z."""
    return [sum(abs(m[i][j]) * size[j] for j in range(3)) for i in range(3)]


def photo_extent(anatomy: dict[str, Any]) -> Vec:
    """The object's span in the photo plane (x, y) and toward the camera (z), in millimetres."""
    return extent(rotation(anatomy), anatomy["object"]["dimensions_mm"])


def in_photo(anatomy: dict[str, Any], point: Vec) -> Vec:
    """An object-frame point in the photo frame."""
    return apply(rotation(anatomy), point)


def place_in_photo(anatomy: dict[str, Any], point: Vec, x: float, y: float) -> Vec:
    """Move an object-frame point so it shows at (x, y) in the photo plane, keeping its depth toward the camera."""
    m = rotation(anatomy)
    c = apply(m, point)
    return [round(v, 1) for v in apply(transpose(m), [x, y, c[2]])]


def shift_in_photo(anatomy: dict[str, Any], delta: Vec) -> Vec:
    """A photo-frame shift (right, up, toward the camera) as an object-frame shift."""
    return apply(transpose(rotation(anatomy)), delta)


def box_corners(center: Vec, size: Vec, rotation_deg: Vec | None = None) -> list[Vec]:
    """The eight corners of a part's box in the object frame: sized in its own frame, turned by its rotation."""
    turn = part_rotation(rotation_deg or [0, 0, 0])
    out = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                r = apply(turn, [sx * size[0], sy * size[1], sz * size[2]])
                out.append([center[i] + r[i] for i in range(3)])
    return out


def box_corners_in_photo(anatomy: dict[str, Any], center: Vec, size: Vec, rotation_deg: Vec | None = None) -> list[tuple[float, float]]:
    """The photo-plane (x, y) of a part box's eight corners."""
    m = rotation(anatomy)
    return [(c[0], c[1]) for c in (apply(m, k) for k in box_corners(center, size, rotation_deg))]


def part_rotation(rotation_deg: Vec) -> Matrix:
    """A part's own frame -> the object's frame, exactly as the viewer turns it (three.js Euler XYZ: Rx · Ry · Rz)."""
    x, y, z = (math.radians(float(v or 0)) for v in (list(rotation_deg) + [0, 0, 0])[:3])
    rx = [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    ry = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    rz = [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
    return _mul(rx, _mul(ry, rz))
