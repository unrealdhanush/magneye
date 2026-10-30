import asyncio

from magneye import anatomy as A
from magneye import ground as G


def _part(id, center=(0, 0, 0), size=(10, 10, 10), sam=None):
    return {"id": id, "name": id, "kind": "internal", "function": "", "material": "plastic", "shape": "box",
            "size_mm": list(size), "center_mm": list(center), "rotation_deg": [0, 0, 0], "layer": 1,
            "sam_prompt": sam, "confidence": 0.9}


def test_sanitize_drops_dangling_references():
    anatomy = {
        "parts": [_part("a"), _part("b")],
        "connections": [{"from": "a", "to": "b", "kind": "signal", "label": ""}, {"from": "a", "to": "ghost", "kind": "signal", "label": ""}],
        "scenarios": [
            {"id": "s1", "title": "", "summary": "", "steps": [{"part_id": "a", "effect": "glow", "label": "", "duration_ms": 500},
                                                               {"part_id": "ghost", "effect": "glow", "label": "", "duration_ms": 500}]},
            {"id": "s2", "title": "", "summary": "", "steps": [{"part_id": "ghost", "effect": "glow", "label": "", "duration_ms": 500}]},
        ],
        "failures": [
            {"id": "f1", "symptom": "", "culprit_part_id": "b", "mechanism": "", "likelihood": "common", "checks": [], "fix": "", "access_path": ["ghost", "a"]},
            {"id": "f2", "symptom": "", "culprit_part_id": "ghost", "mechanism": "", "likelihood": "rare", "checks": [], "fix": "", "access_path": []},
        ],
    }
    out = A.sanitize(anatomy)
    assert [c["to"] for c in out["connections"]] == ["b"]
    assert [s["id"] for s in out["scenarios"]] == ["s1"] and len(out["scenarios"][0]["steps"]) == 1
    assert [f["id"] for f in out["failures"]] == ["f1"] and out["failures"][0]["access_path"] == ["a"]


def test_ground_snaps_visible_part_to_sam_box(monkeypatch):
    # Object spans pixels 100..300 x 100..500 and is 100 x 200 mm, so 2 px per mm.
    anatomy = {"object": {"sam_prompt": "controller", "dimensions_mm": [100, 200, 50]},
               "parts": [_part("stick", center=(0, 0, 20), size=(12, 12, 8), sam="thumbstick")]}

    def match(box, conf=0.9):
        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        return {"object_id": "1", "box": box, "confidence": conf, "mask_bounds": box, "mask_size": [w, h],
                "mask_svg": "M0 0Z", "coverage": 1.0, "raster": bytes([1]) * (w * h)}

    async def fake_segment(_image, phrase):
        return [match([100, 100, 300, 500])] if phrase == "controller" else [match([220, 140, 244, 164])]

    monkeypatch.setattr(G.meta, "segment", fake_segment)
    g = asyncio.run(G.ground(b"", (400, 600), anatomy))
    stick = anatomy["parts"][0]
    # Box centre (232, 152) -> x = (132/200 - 0.5) * 100 = 16 mm, y = (0.5 - 52/400) * 200 = 74 mm.
    assert stick["center_mm"][:2] == [16.0, 74.0] and stick["center_mm"][2] == 20
    assert stick["size_mm"][:2] == [12.0, 12.0] and stick["grounded"] is True
    assert "stick" in g["parts"] and g["object_mask_png"].startswith(b"\x89PNG")


def test_sanitize_drops_quarter_turns_on_round_shapes():
    plate = {**_part("plate", size=(30, 30, 1.5)), "shape": "disc", "rotation_deg": [90, 0, 0],
             "components": [{"shape": "gear", "size_mm": [8, 8, 1], "offset_mm": [0, 0, 0], "rotation_deg": [0, -90, 0]}]}
    pusher = {**_part("pusher", size=(6, 12, 6)), "shape": "cylinder", "rotation_deg": [30, 0, 0]}
    board = {**_part("board", size=(30, 2, 50)), "rotation_deg": [90, 0, 0]}
    A.sanitize({"parts": [plate, pusher, board]})
    # Round shapes face along their size, so a quarter turn would stand them on edge; real tilts and boxes keep theirs.
    assert plate["rotation_deg"] == [0, 0, 0] and plate["components"][0]["rotation_deg"] == [0, 0, 0]
    assert pusher["rotation_deg"] == [30, 0, 0] and board["rotation_deg"] == [90, 0, 0]


def test_geometry_flags_oval_round_parts():
    from magneye.checks import geometry_issues
    bezel = {**_part("bezel", size=(42, 90, 3)), "shape": "ring", "kind": "exterior"}
    dial = {**_part("dial", size=(31, 31, 1)), "shape": "disc", "kind": "exterior"}
    issues = geometry_issues({"object": {"dimensions_mm": [50, 200, 15]}, "parts": [bezel, dial], "failures": []})
    assert any(i.startswith("bezel is a round ring but its diameters differ (42 vs 90 mm)") for i in issues)
    assert not any(i.startswith("dial") for i in issues)


def test_ground_ignores_oval_boxes_for_round_parts(monkeypatch):
    anatomy = {"object": {"sam_prompt": "watch", "dimensions_mm": [100, 200, 50]},
               "parts": [{**_part("bezel", center=(0, 20, 5), size=(40, 40, 3), sam="bezel"), "shape": "ring"}]}

    def match(box):
        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        return {"object_id": "1", "box": box, "confidence": 0.9, "mask_bounds": box, "mask_size": [w, h],
                "mask_svg": "M0 0Z", "coverage": 1.0, "raster": bytes([1]) * (w * h)}

    async def fake_segment(_image, phrase):
        # The "bezel" box runs down into the bracelet: 80 x 180 px is 40 x 90 mm.
        return [match([100, 100, 300, 500])] if phrase == "watch" else [match([160, 180, 240, 360])]

    monkeypatch.setattr(G.meta, "segment", fake_segment)
    g = asyncio.run(G.ground(b"", (400, 600), anatomy))
    bezel = anatomy["parts"][0]
    assert "bezel" not in g["parts"] and not bezel.get("grounded") and bezel["size_mm"] == [40, 40, 3]
