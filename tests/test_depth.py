import copy

import numpy as np

from magneye import depth, mates
from magneye.fit import GRID, Outline, _aligned, _projected


def _part(pid, center, size, kind="exterior", shape="rounded_box", rotation=(0, 0, 0)):
    return {"id": pid, "name": pid, "kind": kind, "shape": shape, "center_mm": list(center), "size_mm": list(size),
            "rotation_deg": list(rotation), "layer": 1}


def _points_out(part, axis, plate):
    """Whether the part's own `axis` lies along the plate's normal, pointing away from the object's centre."""
    from magneye import pose
    own = np.array(pose.part_rotation(part["rotation_deg"]))[:, axis]
    normal = np.array(pose.part_rotation(plate["rotation_deg"]))[:, int(np.argmin(plate["size_mm"]))]
    normal *= np.sign(normal @ np.array(plate["center_mm"], float)) or 1.0
    return float(own @ normal) > 0.99


def _outline(parts, camera):
    """The outline a perfect segmentation of `parts` seen from `camera` would give."""
    empty = Outline(np.zeros((GRID, GRID), bool), -25.0, -25.0, 150 / GRID, [0, 0, 100, 100])
    return Outline(_aligned(_projected(parts, camera), empty), -25.0, -25.0, 150 / GRID, [0, 0, 100, 100])


def test_depth_fit_brings_a_ring_forward_to_match_the_side_photo():
    # A handle with a flat ring on top reaching forward (+z) over it, as the side photo shows.
    truth = [_part("handle", (0, -30, 0), (36, 80, 36)), _part("ring", (0, 18, 28), (90, 16, 90), shape="ring")]
    front_cam = {"azimuth_deg": 0.0, "elevation_deg": 25.0, "roll_deg": 0.0}
    side_cam = {"azimuth_deg": depth.AZIMUTH["left"], "elevation_deg": 0.0, "roll_deg": 0.0}
    front, side = _outline(truth, front_cam), _outline(truth, side_cam)
    # The draft put the ring reaching backwards instead.
    drafted = copy.deepcopy(truth)
    drafted[1]["center_mm"] = [0, 18, -28]
    anatomy = {"object": {"dimensions_mm": [90, 110, 110], "camera": front_cam}, "parts": drafted, "connections": []}
    report = depth.fit_depth(anatomy, front, side, "left", upright=True)
    assert report and report["side_after"] > report["side_before"] + 0.1
    ring = next(p for p in anatomy["parts"] if p["id"] == "ring")
    assert ring["center_mm"][2] > 0
    # An unknown direction is never guessed.
    assert depth.fit_depth(copy.deepcopy(anatomy), front, side, "unclear") is None


def test_round_parts_on_a_plate_face_out_of_it_but_wheels_do_not():
    plate = _part("face_plate", (0, 20, 10), (48, 52, 6), rotation=(-25, 0, 10))
    cap = _part("thumbstick_cap", (0, 30, 14), (18, 10, 18), shape="disc", rotation=(-25, 0, 0))
    wheel = _part("scroll_wheel", (0, 10, 14), (6, 18, 18), shape="disc")
    anatomy = {"parts": [plate, cap, wheel]}
    assert mates.face_out(anatomy) == ["thumbstick_cap now faces out of face_plate"]
    # Turned, not resized: its own round axis (y) now points out of the plate, away from the object's centre.
    assert cap["size_mm"] == [18, 10, 18] and cap["faces"] == "+y"
    assert _points_out(cap, 1, plate)
    assert wheel["rotation_deg"] == [0, 0, 0]


def test_a_part_bigger_than_the_plate_is_not_mounted_on_it():
    plate = _part("faceplate", (0, 20, 10), (48, 52, 6))
    ring = _part("tracking_ring", (0, 30, 12), (96, 18, 70), shape="ring")
    assert mates.face_out({"parts": [plate, ring]}) == []


def test_a_controls_mechanism_points_out_of_the_face_with_it():
    plate = _part("face_plate", (0, 20, 10), (48, 52, 6), rotation=(-25, 0, 0))
    cap = _part("thumbstick_cap", (0, 28, 16), (18, 18, 10), shape="disc", rotation=(-25, 0, 0))
    # Drafted with its shaft (the odd side out, y) pointing up instead of out through the plate.
    module = _part("thumbstick_module", (0, 26, 4), (17, 14, 17), kind="internal", shape="box", rotation=(-25, 0, 0))
    anatomy = {"parts": [plate, cap, module], "connections": [{"from": "thumbstick_cap", "to": "thumbstick_module", "kind": "mechanical"}]}
    assert mates.face_out(anatomy) == ["thumbstick_module now points out of face_plate with thumbstick_cap"]
    assert module["size_mm"] == [17, 14, 17] and module["faces"] == "+y" and cap["faces"] == "+z"
    assert _points_out(module, 1, plate)
