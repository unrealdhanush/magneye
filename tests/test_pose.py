"""The photo's viewpoint: carrying positions between the object's own frame and the photo plane."""

import asyncio
import io
import math

from PIL import Image

from magneye import ground as G
from magneye import pose
from magneye.checks import layout_overlay


def _anatomy(az=0.0, el=0.0, roll=0.0, dims=(60, 50, 20)):
    return {"object": {"sam_prompt": "case", "dimensions_mm": list(dims),
                       "camera": {"azimuth_deg": az, "elevation_deg": el, "roll_deg": roll}}, "parts": []}


def _close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_straight_on_photo_is_the_object_frame():
    a = _anatomy()
    assert pose.frontal(a) and _close(pose.in_photo(a, [3, 4, 5]), [3, 4, 5])
    assert _close(pose.photo_extent(a), [60, 50, 20])


def test_camera_angles_turn_the_object_the_right_way():
    # From its right side, the object's right (+x) faces the camera.
    assert _close(pose.in_photo(_anatomy(az=90), [1, 0, 0]), [0, 0, 1])
    # From above, its top leans toward the camera and its front drops down the photo.
    up, front = pose.in_photo(_anatomy(el=90), [0, 1, 0]), pose.in_photo(_anatomy(el=90), [0, 0, 1])
    assert _close(up, [0, 0, 1]) and _close(front, [0, -1, 0])
    # Rolled 90 degrees counterclockwise, its up points to the photo's left.
    assert _close(pose.in_photo(_anatomy(roll=90), [0, 1, 0]), [-1, 0, 0])


def test_place_in_photo_keeps_depth_and_lands_where_asked():
    a = _anatomy(az=30, el=40, roll=10)
    start = [5.0, 8.0, -3.0]
    moved = pose.place_in_photo(a, start, 12.0, -7.0)
    seen, before = pose.in_photo(a, moved), pose.in_photo(a, start)
    assert abs(seen[0] - 12) < 0.2 and abs(seen[1] + 7) < 0.2 and abs(seen[2] - before[2]) < 0.2
    # A photo-plane shift is the same move expressed in the object frame.
    shift = pose.shift_in_photo(a, [10, 0, 0])
    assert abs(math.hypot(*shift) - 10) < 1e-6 and _close(pose.in_photo(a, shift), [10, 0, 0])


def test_ground_snaps_through_the_viewpoint_and_leaves_sizes_alone(monkeypatch):
    # Seen from 45 degrees above: the object spans 60 wide and 50*cos45 + 20*sin45 tall in the photo.
    anatomy = _anatomy(el=45)
    lid = {"id": "lid", "name": "lid", "kind": "exterior", "shape": "box", "size_mm": [50, 6, 18], "center_mm": [0, 22, 0],
           "rotation_deg": [0, 0, 0], "layer": 0, "sam_prompt": "lid"}
    anatomy["parts"] = [lid]
    span_y = pose.photo_extent(anatomy)[1]

    def match(box):
        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        return {"object_id": "1", "box": box, "confidence": 0.9, "mask_bounds": box, "mask_size": [w, h],
                "mask_svg": "M0 0Z", "coverage": 1.0, "raster": bytes([1]) * (w * h)}

    async def fake_segment(_image, phrase):
        # Object box 100..400 x 100..400 px; the lid's box is centred 30 px above the object's centre.
        return [match([100, 100, 400, 400])] if phrase == "case" else [match([150, 205, 350, 235])]

    monkeypatch.setattr(G.meta, "segment", fake_segment)
    g = asyncio.run(G.ground(b"", (500, 500), anatomy))
    assert "lid" in g["parts"] and lid["size_mm"] == [50, 6, 18]
    seen = pose.in_photo(anatomy, lid["center_mm"])
    assert abs(seen[0]) < 0.2 and abs(seen[1] - 30 / 300 * span_y) < 0.3


def test_layout_overlay_draws_angled_boxes():
    out = io.BytesIO()
    Image.new("RGB", (400, 400), (90, 90, 90)).save(out, "JPEG")
    anatomy = _anatomy(az=35, el=40)
    anatomy["parts"] = [{"id": "body", "kind": "exterior", "layer": 0, "center_mm": [0, 0, 0], "size_mm": [60, 50, 20]}]
    frame = G.Frame([50, 50, 350, 350], pose.photo_extent(anatomy))
    assert Image.open(io.BytesIO(layout_overlay(out.getvalue(), frame, anatomy))).size == (400, 400)
