"""Agent loops driven by a scripted fake model, so tool wiring is tested without API calls."""

import asyncio
import copy
import io
import json

from PIL import Image

from magneye import builder, fit, guide, meta, modeler, views
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
                                       _call("update_parts", {"updates": [{"id": "stick", "reason": "SAM position", "center_mm": [0, 20, 20]}]}, 1)]},
        {"content": "", "tool_calls": [_call("finish", {"summary": "Verified against the photo"})]},
    ])
    async def fake_research(**_):
        return {"text": "Teardown notes: potentiometer stick module under the faceplate.\n\nPhoto search: Acme controller",
                "sources": [{"url": "https://www.ifixit.com/Teardown/x", "title": "Teardown"}], "searches": ["controller teardown"]}

    async def fake_views(*, query, **_):
        assert query == "Acme controller"
        return {"views": [{"origin": "commons", "view": "left", "title": "Acme side", "page_url": "https://commons.wikimedia.org/wiki/File:A.jpg",
                           "image_url": "https://upload.wikimedia.org/a.jpg", "license": "CC0", "author": "Someone", "note": "grip depth",
                           "jpeg": photo}], "considered": 4}

    monkeypatch.setattr(meta, "research", fake_research)
    monkeypatch.setattr(views, "find", fake_views)
    monkeypatch.setattr(meta, "chat_json", fake_draft)
    monkeypatch.setattr(meta, "chat", script)
    monkeypatch.setattr(meta, "segment", _fake_segment)
    monkeypatch.setattr(modeler, "model", _no_modeling)
    monkeypatch.setattr(fit, "fit", lambda *_a, **_k: None)
    events, emit = _collect()
    photo = _photo()

    result = asyncio.run(builder.build(image=photo, prompt="", emit=emit, model="m", reasoning="low", cache=SegmentCache(photo)))

    assert [p["id"] for p in result["parts"]] == ["shell", "stick", "pots", "battery"]
    assert result["research"]["sources"][0]["url"] == "https://www.ifixit.com/Teardown/x"
    assert "Photo search" not in result["research"]["notes"]
    assert result["references"] == [{"origin": "commons", "view": "left", "title": "Acme side", "note": "grip depth",
                                     "page_url": "https://commons.wikimedia.org/wiki/File:A.jpg",
                                     "image_url": "https://upload.wikimedia.org/a.jpg", "license": "CC0", "author": "Someone"}]
    assert any(e.get("tool") == "reference_views" and e.get("text", "").startswith("1 reference view of 4 photos checked: left") for e in events)
    assert any(e.get("tool") == "research" and e.get("text", "").startswith("1 search, 1 source cited (ifixit.com)") for e in events)
    assert next(p for p in result["parts"] if p["id"] == "stick")["center_mm"] == [0, 20, 20]
    texts = [e.get("text", "") for e in events if e["type"] == "step"]
    assert "SAM 3.1: 1 × “thumbstick”" in texts and "SAM 3.1: no “red button” on the object" in texts
    assert "Verified against the photo" in texts
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


def test_reference_views_pick_distinct_useful_views():
    cands = [{"origin": "web", "page_url": "https://a"}, {"origin": "commons", "page_url": "https://b"},
             {"origin": "commons", "page_url": "https://c"}, {"origin": "web", "page_url": "https://d"}]
    verdicts = [{"index": 1, "same_shape": True, "view": "left", "clean": True, "note": ""},
                {"index": 2, "same_shape": True, "view": "left", "clean": True, "note": ""},
                {"index": 3, "same_shape": True, "view": "front", "clean": True, "note": ""},
                {"index": 4, "same_shape": False, "view": "back", "clean": True, "note": ""},
                {"index": 9, "same_shape": True, "view": "top", "clean": True, "note": ""}]
    picked = views.pick(cands, verdicts)
    # One per view, Commons wins the tie, the front view adds nothing, other products and bad indexes are dropped.
    assert [(v["view"], v["page_url"]) for v in picked] == [("left", "https://b")]
    assert views.cite({**picked[0], "jpeg": b"x"}).get("jpeg") is None
    assert "image_url" not in views.cite({"origin": "web", "view": "back", "image_url": "https://x/y.jpg", "page_url": "https://x"})


def test_reference_views_only_fetch_public_urls():
    assert views._public("https://upload.wikimedia.org/a.jpg")
    for url in ["http://localhost:8787/api", "http://127.0.0.1/", "http://192.168.1.4/x.jpg", "file:///etc/passwd", "http://printer.local/"]:
        assert not views._public(url)


def test_loop_nudges_once_when_the_model_narrates_instead_of_finishing(monkeypatch):
    from magneye.agent import Tool, ToolResult, params, run_loop

    async def finish(_):
        return ToolResult("ok", "Done", stop=True)

    script = Script([{"content": "Now I'll apply the fixes.", "tool_calls": []},
                     {"content": "", "tool_calls": [_call("finish", {})]}])
    monkeypatch.setattr(meta, "chat", script)
    events, emit = _collect()
    messages = [{"role": "user", "content": "go"}]
    asyncio.run(run_loop(agent="builder", model="m", messages=messages, tools=[Tool("finish", "", params({}), finish)],
                         emit=emit, until="finish"))
    assert len(script.seen) == 2 and "call finish" in script.seen[1][0][-1]["content"]
    assert events[-1]["tool"] == "finish"


def test_builder_moves_the_body_in_one_call(monkeypatch):
    async def fake_draft(**_):
        draft = copy.deepcopy(ANATOMY)
        draft["parts"].append(_part("strap", "accessory", center=(0, -80, 0), size=(20, 40, 4), layer=0))
        return draft

    script = Script([{"content": "", "tool_calls": [_call("move_parts", {"dx_mm": 0, "dy_mm": 17, "dz_mm": 0})]},
                     {"content": "", "tool_calls": [_call("finish", {"summary": "ok"})]}])
    monkeypatch.setattr(meta, "research", _no_research)
    monkeypatch.setattr(meta, "chat_json", fake_draft)
    monkeypatch.setattr(meta, "chat", script)
    monkeypatch.setattr(meta, "segment", _fake_segment)
    monkeypatch.setattr(modeler, "model", _no_modeling)
    monkeypatch.setattr(fit, "fit", lambda *_a, **_k: None)
    events, emit = _collect()
    photo = _photo()
    result = asyncio.run(builder.build(image=photo, prompt="", emit=emit, model="m", reasoning="low", cache=SegmentCache(photo)))
    centers = {p["id"]: p["center_mm"] for p in result["parts"]}
    # Every body part moved up 17 mm; the accessory stayed where it was.
    assert centers["battery"] == [0, -33, 0] and centers["stick"] == [0, 17, 20] and centers["strap"] == [0, -80, 0]
    assert any(e.get("text") == "Moved 5 parts by (+0, +17, +0) mm" for e in events)


async def _no_research(**_):
    raise meta.MetaAPIError("offline")


def test_side_profile_maps_outline_rows_to_depth(monkeypatch):
    # A 10 x 48 px side-view mask: the top half is 5 px deep at the image's left, the bottom half fills the width.
    w, h = 10, 48
    raster = bytes(1 if (c < 5 or r >= 24) else 0 for r in range(h) for c in range(w))

    async def fake_segment(_image, _phrase):
        return [{"confidence": 0.9, "coverage": 0.75, "mask_size": [w, h], "raster": raster}]

    monkeypatch.setattr(meta, "segment", fake_segment)
    view = {"view": "right", "page_url": "https://x", "jpeg": b"", "clean": True, "upright": True}
    assert views.profile_view([{**view, "view": "back"}, view]) is view
    prof = asyncio.run(views.side_profile(view, "controller"))
    # Seen from the right, the image's left is the front: the thin top sits at the front half of the depth.
    assert prof["rows"][0] == [0.0, 0.5] and prof["rows"][-1] == [-0.5, 0.5] and len(prof["rows"]) == 48
    mirrored = asyncio.run(views.side_profile({**view, "view": "left"}, "controller"))
    assert mirrored["rows"][0] == [-0.5, 0.0]


def test_users_own_photos_are_kept_in_order_before_found_ones(monkeypatch):
    seen = {}

    async def fake_vet(*, more_images, **_):
        seen["n"] = len(more_images)
        return {"candidates": [
            {"index": 1, "same_shape": True, "view": "left", "clean": True, "upright": True, "note": "grip side"},
            {"index": 2, "same_shape": False, "view": "detail", "clean": False, "upright": True, "note": "battery door"},
            {"index": 3, "same_shape": False, "view": "other", "clean": False, "upright": True, "note": "a cat"},
        ]}

    monkeypatch.setattr(meta, "chat_json", fake_vet)
    # No product name and no cited pages (research failed): the user's own photos are still vetted, with no web lookups.
    found = asyncio.run(views.find(image=_photo(), query=None, sources=[], model="m", own=[_photo(), _photo(), _photo()]))
    assert seen["n"] == 3 and found["considered"] == 3
    # Each photo counts (a close-up too); one that shows something else is dropped.
    assert [(v["own"], v["view"]) for v in found["views"]] == [(0, "left"), (1, "detail")]
    assert views.profile_view(found["views"]) is found["views"][0]
    assert views.cite(found["views"][0]) == {"origin": "user", "view": "left", "title": "Your photo 2", "page_url": None,
                                             "note": "grip side", "own": 0}


def test_found_views_skip_sides_the_user_already_covered():
    cands = [{"origin": "user", "own": 0, "page_url": None}, {"origin": "commons", "page_url": "https://a"},
             {"origin": "web", "page_url": "https://b"}]
    verdicts = [{"index": 1, "same_shape": True, "view": "left", "clean": True, "note": ""},
                {"index": 2, "same_shape": True, "view": "left", "clean": True, "note": ""},
                {"index": 3, "same_shape": True, "view": "back", "clean": True, "note": ""}]
    assert [(v["origin"], v["view"]) for v in views.pick(cands, verdicts)] == [("user", "left"), ("web", "back")]

def test_vetting_failure_still_keeps_the_users_own_photos(monkeypatch):
    async def broken_vet(**_):
        raise meta.MetaAPIError("muse-wonder-1.0 did not return valid JSON")

    monkeypatch.setattr(meta, "chat_json", broken_vet)
    found = asyncio.run(views.find(image=_photo(), query=None, sources=[], model="m", own=[_photo(), _photo()]))
    assert found["unvetted"] and [(v["origin"], v["own"]) for v in found["views"]] == [("user", 0), ("user", 1)]
    assert views.profile_view(found["views"]) is None
    # With nothing of the user's to fall back on, the failure is reported as before.
    try:
        asyncio.run(views.find(image=_photo(), query=None, sources=[], model="m", own=[]))
    except meta.MetaAPIError:
        raise AssertionError("no candidates means no vetting call at all")


def test_guide_edits_the_model_when_asked(monkeypatch):
    anatomy = copy.deepcopy(ANATOMY)
    photo = _photo()
    script = Script([
        {"content": "", "tool_calls": [
            _call("add_part", {"part": {"id": "magnet", "name": "Magnet", "material": "magnet", "center_mm": [0, -20, 0], "size_mm": [8, 8, 3]}}),
            _call("edit_parts", {"updates": [{"id": "battery", "size_mm": [14, 40, 14]}]}, 1),
            _call("remove_part", {"id": "pots"}, 2)]},
        {"content": "Added a magnet, shortened the battery and removed the pots."},
    ])
    monkeypatch.setattr(meta, "chat", script)
    events, emit = _collect()
    asyncio.run(guide.respond(anatomy=anatomy, messages=guide.opening_messages(anatomy, photo), prompt="add a magnet...",
                              image=photo, cache=SegmentCache(photo), emit=emit, model="m", ground=None))
    by_id = {p["id"]: p for p in anatomy["parts"]}
    assert by_id["magnet"]["edited"] and by_id["magnet"]["shape"] == "box" and by_id["magnet"]["kind"] == "internal"
    assert by_id["battery"]["size_mm"] == [14, 40, 14] and by_id["battery"]["edited"]
    # Removing the culprit of a failure drops that failure too.
    assert "pots" not in by_id and all(f["culprit_part_id"] != "pots" for f in anatomy["failures"])
    assert anatomy["revision"] == 3 and any(e["type"] == "anatomy" for e in events)


async def _no_modeling(**_):
    return 0


def test_modeler_builds_checks_and_fixes(monkeypatch):
    anatomy = copy.deepcopy(ANATOMY)
    case = {"op": "shell", "of": {"op": "box", "size": [100, 200, 50], "radius": 8}, "wall": 3, "open": "+z"}
    script = Script([
        {"content": "", "tool_calls": [
            _call("model_parts", {"models": [{"id": "shell", "recipe": case},
                                             {"id": "stick", "recipe": {"op": "cylinder", "d": 10, "h": 8, "axis": "w"}}]}),
            _call("view_models", {}, 1)]},
        {"content": "", "tool_calls": [
            _call("model_parts", {"models": [{"id": "stick", "recipe": {"op": "cylinder", "d": 10, "h": 8, "axis": "z"}}]}),
            _call("finish", {"summary": "Modeled the shell and stick"}, 1)]},
    ])
    monkeypatch.setattr(meta, "chat", script)
    events, emit = _collect()
    n = asyncio.run(modeler.model(anatomy=anatomy, photo=_photo(), box=[100, 100, 300, 500], emit=emit, model_id="m"))
    by_id = {p["id"]: p for p in anatomy["parts"]}
    assert n == 2 and by_id["shell"]["model"] == case and by_id["stick"]["model"]["axis"] == "z"
    # The broken recipe was reported back with a reason, and the render reached the model as an image.
    first_report = json.loads(next(m["content"] for m in script.seen[1][0] if m.get("role") == "tool"))
    assert any("axis must be" in r.get("error", "") for r in first_report)
    assert script.seen[1][0][-1]["content"][0]["type"] == "image_url"
    assert any(e.get("text") == "Built 1 part with Manifold; 1 to fix" for e in events)


def test_modeler_can_move_parts_and_fix_the_viewpoint(monkeypatch):
    anatomy = copy.deepcopy(ANATOMY)
    script = Script([
        {"content": "", "tool_calls": [
            _call("place_parts", {"updates": [{"id": "battery", "center_mm": [0, -40, 0], "size_mm": [14, 60, 14], "reason": "fill the grip"},
                                              {"id": "battery", "size_mm": [0, 1, 1], "reason": "bad"}]}),
            _call("set_camera", {"azimuth_deg": 10, "elevation_deg": 30, "roll_deg": 0}, 1),
            _call("finish", {"summary": "ok"}, 2)]},
    ])
    monkeypatch.setattr(meta, "chat", script)
    events, emit = _collect()
    asyncio.run(modeler.model(anatomy=anatomy, photo=_photo(), box=[100, 100, 300, 500], emit=emit, model_id="m"))
    battery = next(p for p in anatomy["parts"] if p["id"] == "battery")
    assert battery["center_mm"] == [0, -40, 0] and battery["size_mm"] == [14, 60, 14]
    assert anatomy["object"]["camera"] == {"azimuth_deg": 10.0, "elevation_deg": 30.0, "roll_deg": 0.0}
