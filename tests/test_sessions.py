"""Creating a session with a main photo and extra photos, and serving them back."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from magneye import app as A


def _jpeg(colour) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (640, 480), colour).save(out, "JPEG")
    return out.getvalue()


def test_session_keeps_extra_photos_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "DATA", tmp_path)
    client = TestClient(A.app)
    files = [("image", ("main.jpg", _jpeg((200, 0, 0)), "image/jpeg"))] + [
        ("extras", (f"e{i}.jpg", _jpeg((0, 60 * i, 0)), "image/jpeg")) for i in range(1, 4)]
    sid = client.post("/api/sessions", data={"prompt": "how?"}, files=files).json()["id"]
    session = client.get(f"/api/sessions/{sid}").json()
    assert len(session["extra_shas"]) == 3
    # Served in the order they were added.
    greens = [Image.open(io.BytesIO(client.get(f"/api/sessions/{sid}/extra/{n}.jpg").content)).getpixel((5, 5))[1] for n in (1, 2, 3)]
    assert greens == sorted(greens) and greens[0] < greens[-1]
    assert client.get(f"/api/sessions/{sid}/extra/4.jpg").status_code == 404


def test_session_rejects_too_many_extra_photos(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "DATA", tmp_path)
    client = TestClient(A.app)
    files = [("image", ("main.jpg", _jpeg((1, 1, 1)), "image/jpeg"))] + [
        ("extras", (f"e{i}.jpg", _jpeg((i, i, i)), "image/jpeg")) for i in range(A.MAX_EXTRAS + 1)]
    assert client.post("/api/sessions", files=files).status_code == 400


def test_saving_edits_cleans_references(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(A, "DATA", tmp_path)
    d = tmp_path / "sessions" / "s1"
    d.mkdir(parents=True)
    part = {"name": "x", "kind": "internal", "function": "", "material": "plastic", "shape": "box", "size_mm": [1, 1, 1],
            "center_mm": [0, 0, 0], "rotation_deg": [0, 0, 0], "layer": 1}
    anatomy = {"object": {"name": "o", "dimensions_mm": [10, 10, 10]}, "parts": [{**part, "id": "a"}, {**part, "id": "b"}],
               "connections": [{"from": "a", "to": "b", "kind": "signal", "label": ""}], "scenarios": [],
               "failures": [{"id": "f", "culprit_part_id": "b", "access_path": ["a"]}]}
    (d / "anatomy.json").write_text(json.dumps(anatomy))
    (d / "ground.json").write_text(json.dumps({"parts": {"a": {}, "b": {}}, "object": None}))
    client = TestClient(A.app)

    saved = client.put("/api/sessions/s1/anatomy", json={**anatomy, "parts": [{**part, "id": "a", "edited": True}]}).json()
    assert [p["id"] for p in saved["parts"]] == ["a"] and saved["failures"] == [] and saved["connections"] == []
    assert saved["revision"] == 1 and json.loads((d / "ground.json").read_text())["parts"] == {"a": {}}
    assert client.put("/api/sessions/s1/anatomy", json={"parts": [{"id": "a"}]}).status_code == 400
    assert client.put("/api/sessions/s1/anatomy", json={"parts": [{**part, "id": "a"}, {**part, "id": "a"}]}).status_code == 400


def test_runs_per_hour_limit_spares_cached_and_built_sessions(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(A, "DATA", tmp_path)
    monkeypatch.setattr(A, "RUNS_PER_HOUR", 1)
    monkeypatch.setattr(A, "_runs", {})
    client = TestClient(A.app)
    d = tmp_path / "sessions" / "s1"
    d.mkdir(parents=True)
    (d / "anatomy.json").write_text(json.dumps({"object": {}, "parts": [], "connections": [], "scenarios": [], "failures": []}))
    (d / "session.json").write_text(json.dumps({"prompt": "", "size": [1, 1], "image_sha": "ab" * 32}))
    (d / "image.jpg").write_bytes(_jpeg((1, 2, 3)))
    # Opening a session that is already built spends nothing; each visitor has their own hourly budget.
    assert client.post("/api/sessions/s1/build").status_code == 200
    req = type("R", (), {"headers": {}, "client": type("C", (), {"host": "1.2.3.4"})()})()
    A._spend(req)
    try:
        A._spend(req)
        raise AssertionError("second run in the hour should be refused")
    except A.HTTPException as e:
        assert e.status_code == 429
    other = type("R", (), {"headers": {"x-forwarded-for": "5.6.7.8, 10.0.0.1"}, "client": None})()
    A._spend(other)  # a different visitor has their own budget
