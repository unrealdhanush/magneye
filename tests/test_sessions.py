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
