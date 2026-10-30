"""Geometry recipes: what Muse Wonder writes, built by Manifold."""

import numpy as np
import pytest

from magneye import model as M


def _bounds(solid):
    b = solid.bounding_box()
    return np.round(np.array(b[3:]) - np.array(b[:3]), 1)


def test_rounded_hollow_case_with_wells():
    case = {"op": "difference", "of": [
        {"op": "shell", "of": {"op": "box", "size": [60, 45, 22], "radius": 6}, "wall": 2, "open": "+y"},
        {"op": "cylinder", "d": 10, "h": 30, "axis": "y", "at": [-15, 10, 0]}]}
    solid = M.build(case)
    full = M.build({"op": "box", "size": [60, 45, 22], "radius": 6})
    assert list(_bounds(full)) == [60, 45, 22]
    # Opening the top removes its flat face; the walls end inside the rounded edge, a little lower.
    width, height, depth = _bounds(solid)
    assert width == 60 and depth == 22 and 40 < height < 45
    # Hollow and open: far less material than the solid block.
    assert solid.volume() < 0.4 * full.volume()


def test_axes_turn_round_shapes():
    assert list(_bounds(M.build({"op": "cylinder", "d": 10, "h": 40, "axis": "x"}))) == [40, 10, 10]
    assert list(_bounds(M.build({"op": "cylinder", "d": 10, "h": 40, "axis": "y"}))) == [10, 40, 10]
    knob = M.build({"op": "revolve", "profile": [[0, 0], [8, 0], [8, 4], [5, 10], [0, 10]], "axis": "y"})
    assert list(_bounds(knob)) == [16, 10, 16]
    gear = M.build({"op": "gear", "d": 20, "h": 3, "teeth": 18})
    assert abs(_bounds(gear)[0] - 20) < 0.5 and _bounds(gear)[2] == 3


def test_patterns_and_transforms():
    vents = M.build({"op": "array", "of": {"op": "box", "size": [2, 10, 2]}, "count": 5, "step": [4, 0, 0]})
    assert list(_bounds(vents)) == [18, 10, 2]
    spokes = M.build({"op": "radial", "of": {"op": "box", "size": [20, 2, 2], "at": [10, 0, 0]}, "count": 4, "axis": "z"})
    assert list(_bounds(spokes)) == [40, 40, 2]
    moved = M.build({"op": "sphere", "d": 10, "at": [5, 0, 0]})
    assert round(moved.bounding_box()[3], 1) == 10


@pytest.mark.parametrize("bad, reason", [
    ({"op": "teleport"}, "unknown op"),
    ({"op": "box", "size": [1, 2]}, "size must be"),
    ({"op": "difference", "of": [{"op": "box", "size": [5, 5, 5]}, {"op": "box", "size": [9, 9, 9]}]}, "empty solid"),
    ({"op": "cylinder", "d": 4, "h": 4, "axis": "w"}, "axis must be"),
    ({"op": "shell", "of": {"op": "box", "size": [4, 4, 4]}, "wall": 3}, "thicker"),
])
def test_bad_recipes_say_why(bad, reason):
    with pytest.raises(M.RecipeError, match=reason):
        M.build(bad)


def test_render_and_encode():
    anatomy = {"object": {"camera": {"azimuth_deg": 30, "elevation_deg": 25, "roll_deg": 0}},
               "parts": [{"id": "case", "kind": "exterior", "material": "plastic", "size_mm": [60, 45, 22], "center_mm": [0, 0, 0]},
                         {"id": "cell", "kind": "internal", "material": "battery", "size_mm": [30, 10, 10], "center_mm": [0, -5, 0]}]}
    solids = {"case": M.build({"op": "box", "size": [60, 45, 22], "radius": 6})}
    im = M.render(anatomy, solids)
    assert im.size == (560, 560) and len(set(im.get_flattened_data())) > 20
    assert M.render(anatomy, solids, internals=True).getpixel((280, 280)) != (21, 24, 31)
    enc = M.encode(solids["case"])
    assert enc["max"] == [30.0, 22.5, 11.0] and len(enc["positions"]) > 100
