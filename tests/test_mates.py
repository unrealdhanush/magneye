"""Mates: placing parts by how they attach."""

import pytest

from magneye import mates


def _p(id, center, size, rotation=(0, 0, 0)):
    return {"id": id, "center_mm": list(center), "size_mm": list(size), "rotation_deg": list(rotation)}


def test_inside_on_coaxial_touch_and_mesh():
    parts = {p["id"]: p for p in [
        _p("holder", (0, 0, 0), (20, 60, 20)), _p("cell", (40, 30, 5), (14, 50, 14)),
        _p("board", (0, 0, 0), (40, 2, 30)), _p("switch", (5, 9, 3), (6, 3, 6)),
        _p("hole", (10, 0, 0), (4, 4, 10)), _p("screw", (7, 3, 20), (3, 3, 12)),
        _p("plug", (0, 0, 0), (8, 8, 8)), _p("cable", (0, 30, 0), (2, 10, 2)),
        _p("big_gear", (0, 0, 0), (40, 40, 3)), _p("small_gear", (90, 4, 7), (20, 20, 3)),
    ]}
    mates.apply(parts, {"part": "cell", "to": "holder", "type": "inside"})
    assert parts["cell"]["center_mm"] == [0, 0, 0]
    mates.apply(parts, {"part": "switch", "to": "board", "type": "on", "face": "+y"})
    # Sits on top of the board (1 + 3/2) and keeps its place across it.
    assert parts["switch"]["center_mm"] == [5, 2.5, 3]
    mates.apply(parts, {"part": "screw", "to": "hole", "type": "coaxial", "axis": "z"})
    assert parts["screw"]["center_mm"] == [10, 0, 20]
    mates.apply(parts, {"part": "cable", "to": "plug", "type": "touch"})
    assert parts["cable"]["center_mm"] == [0, 9, 0]
    mates.apply(parts, {"part": "small_gear", "to": "big_gear", "type": "mesh", "axis": "z", "direction": "+x"})
    # Radii 20 + 10, overlapping a little so the teeth engage; on the big gear's plane.
    assert parts["small_gear"]["center_mm"] == [28.2, 0, 0]


def test_mates_respect_rotation_and_report_mistakes():
    # A rod 40 long in its own x, stood upright, fits inside a tall sleeve.
    parts = {p["id"]: p for p in [_p("sleeve", (0, 0, 0), (10, 50, 10)), _p("rod", (30, 0, 0), (40, 4, 4), (0, 0, 90))]}
    mates.apply(parts, {"part": "rod", "to": "sleeve", "type": "inside"})
    assert parts["rod"]["center_mm"] == [0, 0, 0]
    for bad, reason in [({"part": "rod", "to": "nope", "type": "inside"}, "existing"), ({"part": "rod", "to": "sleeve", "type": "glue"}, "type is"),
                        ({"part": "rod", "to": "sleeve", "type": "on", "face": "up"}, "face"),
                        ({"part": "rod", "to": "sleeve", "type": "mesh", "axis": "z", "direction": "+z"}, "across")]:
        with pytest.raises(mates.MateError, match=reason):
            mates.apply(parts, bad)
