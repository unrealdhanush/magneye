"""Agent loops driven by a scripted fake model, so tool wiring is tested without API calls."""

import asyncio
import copy
import io
import json

from PIL import Image

from magneye import builder, guide, meta
from magneye.checks import geometry_issues
from magneye.ground import SegmentCache


def _photo() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (400, 600), (200, 200, 200)).save(out, "JPEG")
    return out.getvalue()


def _part(id, kind="internal", center=(0, 0, 0), size=(10, 10, 10), layer=1, sam=None):
    return {"id": id, "name": id.replace("_", " "), "kind": kind, "function": "", "material": "plastic", "shape": "box",
            "size_mm": list(size), "center_mm": list(center), "rotation_deg": [0, 0, 0], "layer": layer,
            "sam_prompt": sam, "confidence": 0.8}


ANATOMY = {
    "object": {"name": "Controller", "maker": None, "model": None, "category": "controller", "summary": "",
               "dimensions_mm": [100, 200, 50], "confidence": 0.9, "evidence": [], "sam_prompt": "controller"},
    "parts": [_part("shell", "exterior", layer=0), _part("stick", "exterior", center=(0, 0, 20), sam="thumbstick", layer=0),
              _part("pots", center=(0, 0, 10), layer=1), _part("battery", center=(0, -50, 0), size=(14, 50, 14), layer=2),
              _part("ghost_button", "exterior", center=(30, 60, 25), sam="red button", layer=0)],
    "connections": [],
    "scenarios": [{"id": "tilt", "title": "Tilt the stick", "summary": "",
                   "steps": [{"part_id": "stick", "effect": "press", "label": "tilt", "duration_ms": 500}]}],
    "failures": [{"id": "drift", "symptom": "drift", "culprit_part_id": "pots", "mechanism": "", "likelihood": "common",
                  "checks": [], "fix": "", "access_path": ["shell"]},
                 {"id": "dirt", "symptom": "sticky", "culprit_part_id": "stick", "mechanism": "", "likelihood": "occasional",
                  "checks": [], "fix": "", "access_path": []}],
    "caveats": [],
}


def _match(box, conf=0.9):
    w, h = int(box[2] - box[0]), int(box[3] - box[1])
    return {"object_id": str(box[0]), "box": box, "confidence": conf, "mask_bounds": box, "mask_size": [w, h],
            "mask_svg": "M0 0Z", "coverage": 1.0, "raster": bytes([1]) * (w * h)}


async def _fake_segment(_image, phrase):
    return {"controller": [_match([100, 100, 300, 500])], "thumbstick": [_match([180, 260, 220, 300])]}.get(phrase, [])


def _call(name, args, i=0):
    return {"id": f"c{name}{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Script:
    """Replays assistant turns in order and records what the model was sent."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    async def __call__(self, *, messages, tools, **_):
        self.seen.append((copy.deepcopy(messages), [t["function"]["name"] for t in tools]))
        return self.turns.pop(0)


def _collect():
    events = []

    async def emit(e):
        events.append(e)
    return events, emit


def test_builder_verifies_and_edits(monkeypatch):
    async def fake_draft(**_):
        return copy.deepcopy(ANATOMY)

    script = Script([
        {"content": "", "tool_calls": [_call("find_in_photo", {"phrase": "thumbstick"}), _call("find_in_photo", {"phrase": "red button"}, 1),
                                       _call("check_geometry", {}), _call("view_layout", {})]},
        {"content": "", "tool_calls": [_call("remove_part", {"id": "ghost_button", "reason": "SAM found no red button"}),
                                       _call("update_part", {"id": "stick", "reason": "SAM position", "center_mm": [0, 20, 20]}, 1)]},
        {"content": "", "tool_calls": [_call("finish", {"summary": "Verified against the photo"})]},
    ])
    monkeypatch.setattr(meta, "chat_json", fake_draft)
    monkeypatch.setattr(meta, "chat", script)
    monkeypatch.setattr(meta, "segment", _fake_segment)
    events, emit = _collect()
    photo = _photo()

    result = asyncio.run(builder.build(image=photo, prompt="", emit=emit, model="m", reasoning="low", cache=SegmentCache(photo)))

    assert [p["id"] for p in result["parts"]] == ["shell", "stick", "pots", "battery"]
    assert next(p for p in result["parts"] if p["id"] == "stick")["center_mm"] == [0, 20, 20]
    texts = [e.get("text", "") for e in events if e["type"] == "step"]
    assert "SAM 3.1: 1 × “thumbstick”" in texts and "SAM 3.1: no “red button” on the object" in texts
    assert texts[-1] == "Verified against the photo"
    # The layout overlay reached the model as an image right after the tool results.
    second_turn_messages = script.seen[1][0]
    assert second_turn_messages[-1]["role"] == "user" and second_turn_messages[-1]["content"][0]["type"] == "image_url"
    # SAM's thumbstick box (centre 200,280 on a 200x400 px object) is reported in the object's mm frame.
    sam_reply = json.loads(next(m["content"] for m in second_turn_messages if m.get("tool_call_id") == "cfind_in_photo0"))
    assert sam_reply["matches"][0]["center_mm"] == [0.0, 10.0]


def test_guide_asks_then_diagnoses(monkeypatch):
    monkeypatch.setattr(meta, "segment", _fake_segment)
    anatomy = copy.deepcopy(ANATOMY)
    photo = _photo()
    messages = guide.opening_messages(anatomy, photo)

    first = Script([{"content": "", "tool_calls": [_call("highlight", {"part_ids": ["stick", "nope"]}),
                                                   _call("ask_user", {"question": "Does it drift with nothing touching it?"}, 1)]}])
    monkeypatch.setattr(meta, "chat", first)
    events, emit = _collect()
    asyncio.run(guide.respond(anatomy=anatomy, messages=messages, prompt="my character keeps walking", image=photo,
                              cache=SegmentCache(photo), emit=emit, model="m", ground=None))
    assert {"type": "action", "action": {"type": "highlight", "part_ids": ["stick"]}} in events
    assert any(e["type"] == "question" for e in events) and not any(e["type"] == "answer" for e in events)

    second = Script([
        {"content": "", "tool_calls": [_call("look_closer", {"part_id": "stick"}), _call("show_failure", {"failure_id": "drift"}, 1)]},
        {"content": "That's classic drift: worn potentiometer tracks report an off-centre stick at rest."},
    ])
    monkeypatch.setattr(meta, "chat", second)
    events, emit = _collect()
    asyncio.run(guide.respond(anatomy=anatomy, messages=messages, prompt="yes, even on the table", image=photo,
                              cache=SegmentCache(photo), emit=emit, model="m", ground=None))
    shown = next(e["action"] for e in events if e["type"] == "action" and e["action"]["type"] == "show_failure")
    assert shown["failure"]["culprit_part_id"] == "pots"
    assert events[-1] == {"type": "answer", "text": "That's classic drift: worn potentiometer tracks report an off-centre stick at rest."}
    # The resumed conversation kept the first turn, its question, and the user's reply.
    sent = second.seen[0][0]
    assert [m["role"] for m in sent[:2]] == ["system", "user"]
    assert any(m.get("content") == "yes, even on the table" for m in sent)


def test_geometry_flags_outside_overlap_and_outward_access():
    a = copy.deepcopy(ANATOMY)
    a["parts"].append(_part("pots_twin", center=(0, 0, 10), layer=1))
    a["parts"].append(_part("antenna", center=(0, 120, 0), layer=1))
    a["failures"][0]["access_path"] = ["battery"]
    issues = geometry_issues(a)
    assert any("antenna sticks out" in i for i in issues)
    assert any("pots and pots_twin occupy the same space" in i for i in issues)
    assert any("access path removes battery" in i for i in issues)
