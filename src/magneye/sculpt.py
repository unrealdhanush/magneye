"""Free-form solids for recipes: lofts through cross-sections and smooth blends of primitives.

Boxes, cylinders and hulls make parts that read as blocks. Real moulded bodies change cross-section as they go (a
grip that flares into a head) and melt into each other with fillets. A loft sweeps a rounded section along a curved
path, changing its width, depth and squareness section by section; a blend merges primitives through a signed
distance field with a smooth minimum, so they join like clay instead of meeting at a crease.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from manifold3d import Manifold, Mesh

RING = 48
PER_SPAN = 10
CAP_RINGS = 6
# A blend is sampled on a grid this many cells across its longest side.
BLEND_CELLS = 56


class SculptError(ValueError):
    pass


def _catmull(points: np.ndarray, per: int) -> tuple[np.ndarray, np.ndarray]:
    """Points along a smooth curve through `points`, and for each the fractional section index it lies at."""
    if len(points) == 2:
        t = np.linspace(0, 1, per + 1)
        return points[0] + (points[1] - points[0]) * t[:, None], t
    ext = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out, where = [], []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = ext[i:i + 4]
        for t in np.linspace(0, 1, per, endpoint=i == len(points) - 2):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            where.append(i + t)
    return np.array(out), np.array(where)


def _section(w: float, d: float, square: float) -> np.ndarray:
    """A superellipse w wide and d deep: an ellipse at square 0, a rounded rectangle toward square 1."""
    n = 2 + 8 * square
    a = np.linspace(0, 2 * math.pi, RING, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    return np.stack([w / 2 * np.sign(c) * np.abs(c) ** (2 / n), d / 2 * np.sign(s) * np.abs(s) ** (2 / n)], axis=1)


def loft(sections: list[dict[str, Any]], caps: str = "round") -> Manifold:
    """A body through cross-sections: each {"at": [x, y, z], "w", "d", "square": 0..1}. The path runs smoothly
    through the centres; width, depth and squareness change smoothly between sections. The section's width lies
    across the path, as near the object's x axis as the path allows."""
    centres = np.array([s["at"] for s in sections], float)
    dims = np.array([[s["w"], s["d"], s["square"]] for s in sections], float)
    path, where = _catmull(centres, PER_SPAN)
    i = np.clip(np.floor(where).astype(int), 0, len(sections) - 2)
    f = where - i
    f = f * f * (3 - 2 * f)
    size = dims[i] * (1 - f)[:, None] + dims[i + 1] * f[:, None]
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    # Frames carried along the path (parallel transport), so the section doesn't spin as the path bends.
    ref = np.array([1.0, 0, 0]) if abs(tangent[0][0]) < 0.9 else np.array([0, 0, 1.0])
    x = ref - tangent[0] * (ref @ tangent[0])
    frames = []
    for t in tangent:
        x = x - t * (x @ t)
        x /= np.linalg.norm(x) + 1e-12
        frames.append((x.copy(), np.cross(t, x)))
    rings = [path[k] + sec[:, :1] * frames[k][0] + sec[:, 1:] * frames[k][1]
             for k, sec in enumerate(_section(*size[k]) for k in range(len(path)))]
    # Round caps close each end with shrinking rings, like the end of a capsule; flat caps close it at once.
    def cap(k: int, sign: float) -> list[np.ndarray]:
        if caps != "round":
            return []
        # Inset like a fillet of radius r: a wide end stays flat in the middle with rounded edges, not a pyramid.
        w, d, square = size[k]
        r = 0.5 * min(w, d)
        out = []
        for j in range(1, CAP_RINGS):
            a = math.pi / 2 * j / CAP_RINGS
            inset = 2 * r * (1 - math.cos(a))
            sec = _section(max(w - inset, 1e-3), max(d - inset, 1e-3), square)
            out.append(path[k] + sign * tangent[k] * r * math.sin(a) + sec[:, :1] * frames[k][0] + sec[:, 1:] * frames[k][1])
        return out
    start, end = cap(0, -1.0)[::-1], cap(len(path) - 1, 1.0)
    rings = start + rings + end
    tip0 = path[0] - (tangent[0] * 0.5 * min(size[0][:2]) if caps == "round" else 0)
    tip1 = path[-1] + (tangent[-1] * 0.5 * min(size[-1][:2]) if caps == "round" else 0)
    verts = np.vstack([*rings, tip0, tip1]).astype(np.float32)
    n, m = len(rings), RING
    tris = []
    for r in range(n - 1):
        for j in range(m):
            a, b = r * m + j, r * m + (j + 1) % m
            c, d = a + m, b + m
            tris += [[a, b, d], [a, d, c]]
    t0, t1 = n * m, n * m + 1
    tris += [[t0, (j + 1) % m, j] for j in range(m)]
    tris += [[t1, (n - 1) * m + j, (n - 1) * m + (j + 1) % m] for j in range(m)]
    tris = np.array(tris, np.uint32)
    solid = Manifold(Mesh(vert_properties=verts, tri_verts=tris))
    if solid.is_empty() or solid.volume() < 0:
        solid = Manifold(Mesh(vert_properties=verts, tri_verts=tris[:, ::-1].copy()))
    if solid.is_empty():
        raise SculptError("loft: the sections fold through each other; space them out or make them smaller")
    return solid


def _rotation(deg: list[float]) -> np.ndarray:
    """The matrix Manifold.rotate applies: about x, then y, then z."""
    x, y, z = (math.radians(v) for v in deg)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _along(p: np.ndarray, axis: str) -> np.ndarray:
    """Coordinates with the given axis last, so round shapes can be written along z."""
    return {"x": p[:, [1, 2, 0]], "y": p[:, [0, 2, 1]], "z": p}.get(axis, p)


def distance(node: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """The signed distance (negative inside) to a primitive node, for many points at once."""
    op = node.get("op")
    turn = _rotation(node.get("rotate") or [0, 0, 0])
    at = np.array(node.get("at") or [0, 0, 0], float)
    axis = str(node.get("axis", "z"))

    def local(p: np.ndarray) -> np.ndarray:
        return (p - at) @ turn

    if op == "sphere":
        r = float(node["d"]) / 2
        return lambda p: np.linalg.norm(local(p), axis=1) - r
    if op == "ellipsoid":
        r = np.array(node["size"], float) / 2

        def ellipsoid(p: np.ndarray) -> np.ndarray:
            q = local(p)
            k0 = np.linalg.norm(q / r, axis=1)
            k1 = np.linalg.norm(q / (r * r), axis=1)
            return k0 * (k0 - 1) / np.maximum(k1, 1e-9)
        return ellipsoid
    if op == "box":
        half = np.array(node["size"], float) / 2
        rad = min(float(node.get("radius") or 0), float(half.min()) - 1e-3)

        def box(p: np.ndarray) -> np.ndarray:
            q = np.abs(local(p)) - (half - rad)
            return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - rad
        return box
    if op == "cylinder":
        r, h = float(node["d"]) / 2, float(node["h"]) / 2

        def cylinder(p: np.ndarray) -> np.ndarray:
            q = _along(local(p), axis)
            d = np.stack([np.linalg.norm(q[:, :2], axis=1) - r, np.abs(q[:, 2]) - h], axis=1)
            return np.minimum(d.max(axis=1), 0) + np.linalg.norm(np.maximum(d, 0), axis=1)
        return cylinder
    if op == "torus":
        big, small = float(node["d"]) / 2 - float(node["w"]) / 2, float(node["w"]) / 2

        def torus(p: np.ndarray) -> np.ndarray:
            q = _along(local(p), axis)
            return np.sqrt((np.linalg.norm(q[:, :2], axis=1) - big) ** 2 + q[:, 2] ** 2) - small
        return torus
    if op == "tube":
        pts = np.array(node["points"], float)
        ds = node["d"] if isinstance(node["d"], list) else [node["d"]] * len(pts)
        rs = np.array(ds, float) / 2

        def tube(p: np.ndarray) -> np.ndarray:
            q, best = local(p), None
            for a, b, ra, rb in zip(pts, pts[1:], rs, rs[1:]):
                ba = b - a
                h = np.clip(((q - a) @ ba) / max(float(ba @ ba), 1e-9), 0, 1)
                d = np.linalg.norm(q - a - h[:, None] * ba, axis=1) - (ra + (rb - ra) * h)
                best = d if best is None else np.minimum(best, d)
            return best
        return tube
    raise SculptError(f"blend: {op!r} can't be blended; use sphere, ellipsoid, box, cylinder, torus or tube")


def blend(nodes: list[dict[str, Any]], k: float, bounds: list[float]) -> Manifold:
    """The primitives melted together: a smooth minimum of their distances, with fillets about k mm wide. `bounds`
    is the box (min xyz, max xyz) around them."""
    fields = [distance(n) for n in nodes]
    lo, hi = np.array(bounds[:3]) - k, np.array(bounds[3:]) + k
    step = float((hi - lo).max()) / BLEND_CELLS
    axes = [np.arange(lo[i] - step, hi[i] + 2 * step, step) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d = fields[0](grid)
    for f in fields[1:]:
        e = f(grid)
        h = np.maximum(k - np.abs(d - e), 0) / max(k, 1e-9)
        d = np.minimum(d, e) - h * h * k / 4
    field = d.reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    origin, shape = np.array([a[0] for a in axes]), np.array(field.shape) - 1

    def sample(x: float, y: float, z: float) -> float:
        # Trilinear lookup in the sampled field (level_set asks point by point); positive inside, as it expects.
        g = np.clip((np.array([x, y, z]) - origin) / step, 0, shape - 1e-6)
        i, f = g.astype(int), g - g.astype(int)
        c = field[i[0]:i[0] + 2, i[1]:i[1] + 2, i[2]:i[2] + 2]
        c = c[0] * (1 - f[0]) + c[1] * f[0]
        c = c[0] * (1 - f[1]) + c[1] * f[1]
        return -float(c[0] * (1 - f[2]) + c[1] * f[2])

    solid = Manifold.level_set(sample, [*lo, *hi], step)
    if solid.is_empty():
        raise SculptError("blend: the shapes vanished; check their sizes")
    return solid
