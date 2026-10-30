"""True 3D parts from geometry recipes.

Muse Wonder describes a part as a small tree of solid operations; Manifold (the geometry kernel behind OpenSCAD)
builds it. A recipe is only data, never code, in millimetres around the part's centre and in the object's own axes
(x its right, y up, z out of its front). The built solid is fitted to the part's size_mm wherever it is shown, so a
recipe sets the part's form and features while the anatomy keeps its size and place.

    {"op": "difference", "of": [
        {"op": "box", "size": [60, 45, 22], "radius": 6},
        {"op": "cylinder", "d": 14, "h": 30, "axis": "y", "at": [-15, 10, 0]}]}
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from manifold3d import CrossSection, Manifold
from PIL import Image, ImageDraw

from . import pose

MAX_NODES = 48
MAX_DEPTH = 8
MAX_COPIES = 24
SEGMENTS = 40

OPS_DOC = """Recipe nodes (JSON). Every node may also have "at": [x, y, z] (move it) and "rotate": [rx, ry, rz] degrees (turn it about its own centre first).
- {"op": "box", "size": [x, y, z], "radius": r}: a box centred on its origin; radius rounds every edge and corner (0 for sharp).
- {"op": "cylinder", "d": d, "h": h, "axis": "x"|"y"|"z", "d2": top_d}: a cylinder along the axis (default z); d2 makes it a cone or frustum.
- {"op": "sphere", "d": d}
- {"op": "gear", "d": d, "h": h, "teeth": n, "axis": ...}: a spur gear.
- {"op": "extrude", "polygon": [[a, b], ...], "h": h, "axis": ...}: a 2D outline (in the plane across the axis) pushed out h, centred.
- {"op": "revolve", "profile": [[r, t], ...], "axis": ...}: a profile of radius r against position t along the axis, spun round it (knobs, lenses, bottles, cups).
- {"op": "union" | "difference" | "intersection", "of": [nodes]}: difference keeps the first and cuts all the others out of it.
- {"op": "shell", "of": node, "wall": t, "open": "+y" | "-y" | "+x" | "-x" | "+z" | "-z"}: hollow the node to a wall of thickness t, optionally open on one side (a case, a cup, a lid).
- {"op": "hull", "of": [nodes]}: the convex hull around them (smooth organic forms from spheres).
- {"op": "array", "of": node, "count": n, "step": [dx, dy, dz]}: n copies in a row (vents, keys, cells).
- {"op": "radial", "of": node, "count": n, "axis": ...}: n copies spun evenly round the axis (bolts, spokes, blades)."""

# Base colours per material for the self-check render (the viewer has its own palette).
COLOUR = {
    "plastic": (199, 201, 209), "metal": (168, 176, 189), "pcb": (31, 143, 90), "battery": (255, 159, 10),
    "rubber": (90, 90, 96), "glass": (100, 210, 255), "magnet": (94, 92, 230), "fabric": (191, 90, 242),
    "fluid": (10, 132, 255), "organic": (255, 100, 130), "ceramic": (242, 242, 247), "other": (142, 142, 147),
}


class RecipeError(ValueError):
    pass


def _num(node: dict[str, Any], key: str, default: float | None = None, lo: float = 0.01) -> float:
    v = node.get(key, default)
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RecipeError(f"{node.get('op')}: {key} must be a number") from None
    if not math.isfinite(f) or f < lo:
        raise RecipeError(f"{node.get('op')}: {key} must be at least {lo}")
    return f


def _vec(node: dict[str, Any], key: str, default: list[float] | None = None) -> list[float]:
    v = node.get(key, default)
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)):
        raise RecipeError(f"{node.get('op')}: {key} must be [x, y, z]")
    return [float(x) for x in v]


def _to_axis(m: Manifold, axis: str) -> Manifold:
    """Solids are built along z; turn one to run along the given axis."""
    if axis == "x":
        return m.rotate([0, 90, 0])
    if axis == "y":
        return m.rotate([-90, 0, 0])
    if axis != "z":
        raise RecipeError(f"axis must be x, y or z, not {axis!r}")
    return m


def _polygon(points: Any, what: str) -> list[list[float]]:
    if not (isinstance(points, list) and len(points) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points)):
        raise RecipeError(f"{what} needs at least three [a, b] points")
    return [[float(p[0]), float(p[1])] for p in points]


def _gear(d: float, teeth: int) -> list[list[float]]:
    r_out, r_in = d / 2, d / 2 * 0.84
    step = 2 * math.pi / teeth
    pts = []
    for i in range(teeth):
        a = i * step
        for r, t in ((r_in, 0.0), (r_out, 0.15), (r_out, 0.45), (r_in, 0.6)):
            pts.append([r * math.cos(a + t * step), r * math.sin(a + t * step)])
    return pts


class _Builder:
    def __init__(self) -> None:
        self.nodes = 0

    def build(self, node: Any, depth: int = 0) -> Manifold:
        if not isinstance(node, dict) or "op" not in node:
            raise RecipeError("every node is an object with an op")
        self.nodes += 1
        if self.nodes > MAX_NODES or depth > MAX_DEPTH:
            raise RecipeError(f"recipes are limited to {MAX_NODES} nodes and {MAX_DEPTH} levels")
        m = self._shape(node, depth)
        if "rotate" in node:
            m = m.rotate(_vec(node, "rotate"))
        if "at" in node:
            m = m.translate(_vec(node, "at"))
        return m

    def _children(self, node: dict[str, Any], depth: int, least: int) -> list[Manifold]:
        of = node.get("of")
        if not isinstance(of, list) or len(of) < least:
            raise RecipeError(f"{node['op']} needs \"of\": a list of at least {least} nodes")
        return [self.build(c, depth + 1) for c in of]

    def _shape(self, node: dict[str, Any], depth: int) -> Manifold:
        op = node["op"]
        axis = str(node.get("axis", "z"))
        if op == "box":
            size = _vec(node, "size")
            if min(size) <= 0:
                raise RecipeError("box: size must be positive")
            r = min(float(node.get("radius") or 0), min(size) / 2 - 1e-3)
            if r <= 0.05:
                return Manifold.cube(size, True)
            # Rounded box: the hull of a sphere at each inner corner.
            corners = [[sx * (size[0] / 2 - r), sy * (size[1] / 2 - r), sz * (size[2] / 2 - r)]
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            return Manifold.batch_hull([Manifold.sphere(r, 24).translate(c) for c in corners])
        if op == "cylinder":
            d, h = _num(node, "d"), _num(node, "h")
            d2 = _num(node, "d2", d, lo=0.0)
            return _to_axis(Manifold.cylinder(h, d / 2, d2 / 2, SEGMENTS, True), axis)
        if op == "sphere":
            return Manifold.sphere(_num(node, "d") / 2, SEGMENTS)
        if op == "gear":
            d, h = _num(node, "d"), _num(node, "h")
            teeth = int(min(max(_num(node, "teeth", 18, lo=4), 4), 120))
            return _to_axis(CrossSection([_gear(d, teeth)]).extrude(h).translate([0, 0, -h / 2]), axis)
        if op == "extrude":
            h = _num(node, "h")
            return _to_axis(CrossSection([_polygon(node.get("polygon"), "extrude")]).extrude(h).translate([0, 0, -h / 2]), axis)
        if op == "revolve":
            profile = _polygon(node.get("profile"), "revolve")
            if any(r < 0 for r, _ in profile):
                raise RecipeError("revolve: radii must be zero or more")
            # Manifold spins the (r, t) outline round its t axis and stands the result along z.
            solid = CrossSection([profile]).revolve(SEGMENTS)
            return _to_axis(solid, axis)
        if op in ("union", "difference", "intersection"):
            parts = self._children(node, depth, 1 if op == "union" else 2)
            out = parts[0]
            for p in parts[1:]:
                out = out + p if op == "union" else out - p if op == "difference" else out ^ p
            return out
        if op == "hull":
            return Manifold.batch_hull(self._children(node, depth, 1))
        if op == "shell":
            inner_of = node.get("of")
            outer = self.build(inner_of, depth + 1)
            wall = _num(node, "wall")
            lo, hi = np.array(outer.bounding_box()[:3]), np.array(outer.bounding_box()[3:])
            size, centre = hi - lo, (hi + lo) / 2
            if np.any(size <= 2 * wall):
                raise RecipeError("shell: wall is thicker than the shape")
            inner = outer.translate(list(-centre)).scale(list((size - 2 * wall) / size)).translate(list(centre))
            side = node.get("open")
            if side:
                if side not in ("+x", "-x", "+y", "-y", "+z", "-z"):
                    raise RecipeError("shell: open is one of +x, -x, +y, -y, +z, -z")
                i, sign = "xyz".index(side[1]), 1 if side[0] == "+" else -1
                # Extend the cavity out through that side.
                cut_size = list(size - 2 * wall)
                cut_size[i] = size[i]
                offset = [0.0, 0.0, 0.0]
                offset[i] = sign * size[i] / 2
                inner = inner + Manifold.cube(cut_size, True).translate(list(centre + np.array(offset)))
            return outer - inner
        if op == "array":
            item = self.build(node.get("of"), depth + 1)
            count = int(min(_num(node, "count", 2, lo=1), MAX_COPIES))
            step = _vec(node, "step")
            out = item
            for i in range(1, count):
                out = out + item.translate([s * i for s in step])
            return out.translate([-s * (count - 1) / 2 for s in step])
        if op == "radial":
            item = self.build(node.get("of"), depth + 1)
            count = int(min(_num(node, "count", 2, lo=1), MAX_COPIES))
            spin = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(axis)
            if spin is None:
                raise RecipeError("radial: axis must be x, y or z")
            out = item
            for i in range(1, count):
                out = out + item.rotate([v * 360 * i / count for v in spin])
            return out
        raise RecipeError(f"unknown op {op!r}")


def build(recipe: Any) -> Manifold:
    """The solid a recipe describes; raises RecipeError with a reason the model can act on."""
    try:
        solid = _Builder().build(recipe)
    except RecipeError:
        raise
    except Exception as e:  # Manifold rejects degenerate input with assorted errors
        raise RecipeError(f"could not build: {e}") from None
    if solid.is_empty() or solid.volume() <= 0:
        raise RecipeError("the recipe builds an empty solid (did a difference cut everything away?)")
    return solid


def arrays(solid: Manifold) -> tuple[np.ndarray, np.ndarray]:
    """(vertices N x 3, triangles M x 3) of a solid."""
    mesh = solid.to_mesh()
    return np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3], np.asarray(mesh.tri_verts, dtype=np.uint32)


# Turns that lay each blueprint view's axes onto the drawing plane: front (x, y), top (x, -z), right (-z, y).
VIEW_TURNS = {"front": [0, 0, 0], "top": [90, 0, 0], "right": [0, -90, 0]}


def outlines(solid: Manifold) -> dict[str, list[list[list[float]]]]:
    """The solid's silhouette in each blueprint view, as polygons in that view's (u, v) millimetres; through-holes
    come out as inner contours."""
    out = {}
    for view, turn in VIEW_TURNS.items():
        polys = solid.rotate(turn).project().simplify(0.15).to_polygons()
        out[view] = [[[round(float(x), 2), round(float(y), 2)] for x, y in poly] for poly in polys]
    return out


def encode(solid: Manifold) -> dict[str, Any]:
    """A solid for the viewer: base64 float32 positions and uint32 indices, its bounds in millimetres, and its
    silhouettes for the blueprint."""
    v, t = arrays(solid)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return {"positions": base64.b64encode(v.tobytes()).decode(), "indices": base64.b64encode(t.tobytes()).decode(),
            "min": [round(float(x), 3) for x in lo], "max": [round(float(x), 3) for x in hi], "outlines": outlines(solid)}


def fitted(part: dict[str, Any], v: np.ndarray) -> np.ndarray:
    """A part's mesh scaled to its size_mm and moved to its centre, as the viewer shows it."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.where(hi - lo > 1e-6, hi - lo, 1)
    return (v - (lo + hi) / 2) * (np.array(part["size_mm"], dtype=np.float32) / span) + np.array(part["center_mm"], dtype=np.float32)


def _box(part: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return arrays(Manifold.cube(part["size_mm"], True).translate(part["center_mm"]))


def _shaded(anatomy: dict[str, Any], solids: dict[str, Manifold], internals: bool,
            view: tuple[float, float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    """Camera-facing triangles of the parts in the view's frame (x right, y up, z toward the camera; mm), with
    their shaded colours. Modeled parts as built, the rest as their boxes; `internals` leaves outer parts out."""
    camera = dict(zip(("azimuth_deg", "elevation_deg", "roll_deg"), view)) if view else anatomy["object"].get("camera") or {}
    rot = np.array(pose.rotation({"object": {"camera": camera}}), dtype=np.float32)
    tris, colours = [], []
    for p in anatomy["parts"]:
        if internals and p["kind"] != "internal":
            continue
        if p["id"] in solids:
            v, t = arrays(solids[p["id"]])
            v = fitted(p, v)
        else:
            v, t = _box(p)
        tris.append((v @ rot.T)[t])
        colours.append(np.repeat(np.array([COLOUR.get(p["material"], COLOUR["other"])], dtype=np.float32), len(t), axis=0))
    if not tris:
        return np.zeros((0, 3, 3), np.float32), np.zeros((0, 3), np.float32)
    tri, col = np.concatenate(tris), np.concatenate(colours)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    keep = normal[:, 2] > 0
    light = np.array([0.35, 0.55, 0.75], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normal[keep] @ light, 0, 1)
    return tri[keep], np.clip(col[keep] * shade[:, None], 0, 255)


def _paint(draw: ImageDraw.ImageDraw, tri: np.ndarray, col: np.ndarray, xy: np.ndarray, alpha: int | None = None) -> None:
    """Painter's algorithm: farthest triangles first."""
    for i in np.argsort(tri[:, :, 2].mean(axis=1)):
        c = tuple(int(x) for x in col[i])
        draw.polygon([tuple(p) for p in xy[i]], fill=c + ((alpha,) if alpha is not None else ()))


def render(anatomy: dict[str, Any], solids: dict[str, Manifold], *, internals: bool = False, size: int = 560,
           view: tuple[float, float, float] | None = None) -> Image.Image:
    """A flat-shaded orthographic render of the parts, fitted to a square image, from the photo's viewpoint unless
    `view` (azimuth, elevation, roll) is given."""
    im = Image.new("RGB", (size, size), (21, 24, 31))
    tri, col = _shaded(anatomy, solids, internals, view)
    if not len(tri):
        return im
    lo, hi = tri[..., :2].reshape(-1, 2).min(axis=0), tri[..., :2].reshape(-1, 2).max(axis=0)
    k = (size - 40) / max(float((hi - lo).max()), 1e-6)
    xy = np.empty_like(tri[..., :2])
    xy[..., 0] = 20 + (tri[..., 0] - lo[0]) * k + (size - 40 - (hi[0] - lo[0]) * k) / 2
    xy[..., 1] = 20 + (hi[1] - tri[..., 1]) * k + (size - 40 - (hi[1] - lo[1]) * k) / 2
    _paint(ImageDraw.Draw(im), tri, col, xy)
    return im


def overlay(anatomy: dict[str, Any], solids: dict[str, Manifold], photo: Image.Image, box: list[float]) -> Image.Image:
    """The parts drawn over the photo where the camera would see them: through the photo's viewpoint and the
    object's box in the photo, the same mapping SAM grounding uses. Misplaced or missized parts show as offsets."""
    tri, col = _shaded(anatomy, solids, False, None)
    span = pose.photo_extent(anatomy)
    ow, oh = box[2] - box[0], box[3] - box[1]
    xy = np.empty_like(tri[..., :2])
    xy[..., 0] = box[0] + (tri[..., 0] / span[0] + 0.5) * ow
    xy[..., 1] = box[1] + (0.5 - tri[..., 1] / span[1]) * oh
    layer = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    _paint(ImageDraw.Draw(layer), tri, col, xy, alpha=150)
    return Image.alpha_composite(photo.convert("RGBA"), layer).convert("RGB")


def jpeg(im: Image.Image) -> bytes:
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()
