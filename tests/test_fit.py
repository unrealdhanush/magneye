"""Fitting the viewpoint and layout to SAM's outline of the object."""

import copy

from magneye import fit, pose


def _outline(box, size):
    w, h = size
    return fit.Outline.from_match({"box": box, "mask_bounds": box, "mask_size": [w, h], "raster": bytes([1]) * (w * h)})


ANATOMY = {
    "object": {"dimensions_mm": [30, 90, 20], "camera": {"azimuth_deg": 0, "elevation_deg": 0, "roll_deg": 90}},
    "parts": [
        {"id": "body", "kind": "exterior", "center_mm": [0, 0, 0], "size_mm": [24, 70, 16]},
        {"id": "cap", "kind": "exterior", "center_mm": [0, 38, 0], "size_mm": [14, 10, 14]},
    ],
}


def test_a_wrong_roll_is_found_and_the_layout_fills_the_object():
    # SAM sees a tall, narrow object; Wonder guessed the object was rolled onto its side.
    anatomy = copy.deepcopy(ANATOMY)
    outline = _outline([100, 100, 130, 190], (30, 90))
    before = fit.coverage(anatomy, outline)
    report = fit.fit(anatomy, outline)
    assert report and report["after"] > before + 0.2 and report["after"] > 0.7
    roll = anatomy["object"]["camera"]["roll_deg"]
    assert abs((roll + 180) % 180) < 15 or abs((roll + 180) % 180 - 180) < 15
    # The parts were grown to fill the object's stated size.
    assert anatomy["parts"][0]["size_mm"][1] > 70


def test_a_good_fit_is_left_alone():
    anatomy = copy.deepcopy(ANATOMY)
    anatomy["object"]["camera"]["roll_deg"] = 0
    anatomy["parts"][0]["size_mm"] = [30, 80, 20]
    anatomy["parts"][1]["center_mm"] = [0, 40, 0]
    outline = _outline([100, 100, 130, 190], (30, 90))
    assert fit.coverage(anatomy, outline) > 0.8
    assert fit.fit(anatomy, outline) is None and anatomy["object"]["camera"]["roll_deg"] == 0
    assert pose.frontal(anatomy)
