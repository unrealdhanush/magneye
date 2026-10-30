"""Clearing a hand that covers the object, before the build reads the photo."""

import asyncio
import io

from PIL import Image

from magneye import app as A
from magneye import clean, meta


def _jpeg(colour, size=(640, 480)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, "JPEG")
    return out.getvalue()


def _match(box, conf=0.9):
    return {"box": box, "confidence": conf}


def _session(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "DATA", tmp_path)
    d = tmp_path / "sessions" / "s1"
    d.mkdir(parents=True)
    photo = _jpeg((200, 200, 200))
    (d / "image.jpg").write_bytes(photo)
    s = {"prompt": "", "size": [640, 480], "image_sha": "ab" * 32}
    return d, s, photo


def _run(d, s):
    events = []

    async def emit(e):
        events.append(e)
    asyncio.run(A._clear_photos(d, s, emit))
    return (d / "image.jpg").read_bytes(), events


def test_hand_is_removed_once_and_the_original_kept(tmp_path, monkeypatch):
    d, s, photo = _session(tmp_path, monkeypatch)
    edits = []

    async def fake_segment(_image, phrase):
        return [_match([300, 300, 600, 480])] if phrase == "hand" else []

    async def fake_edit(jpeg, prompt, size=None):
        edits.append(prompt)
        return _jpeg((10, 120, 10), (1024, 768))

    monkeypatch.setattr(meta, "segment", fake_segment)
    monkeypatch.setattr(meta, "edit_image", fake_edit)
    image, events = _run(d, s)
    assert len(edits) == 1 and "hand" in edits[0]
    assert (d / "original.jpg").read_bytes() == photo and (d / "image.jpg").read_bytes() == image
    assert Image.open(io.BytesIO(image)).getpixel((5, 5))[1] > 100
    assert s["cleaned"]["by"] == "muse-image-1.0" and s["size"] == [1024, 768]
    assert events[-1]["text"].startswith("Muse Image removed the hand")

    # Another session with the same photo reuses the cleaned copy instead of editing again.
    d2 = tmp_path / "sessions" / "s2"
    d2.mkdir()
    (d2 / "image.jpg").write_bytes(photo)
    s2 = {"prompt": "", "size": [640, 480], "image_sha": "ab" * 32}
    _run(d2, s2)
    assert len(edits) == 1 and s2["cleaned"] and (d2 / "original.jpg").exists()


def test_photo_without_a_hand_is_used_as_taken(tmp_path, monkeypatch):
    d, s, photo = _session(tmp_path, monkeypatch)

    async def fake_segment(_image, phrase):
        # A small hand far away in the background does not count.
        return [_match([0, 0, 20, 20])]

    async def no_edit(*_a, **_k):
        raise AssertionError("Muse Image should not run")

    monkeypatch.setattr(meta, "segment", fake_segment)
    monkeypatch.setattr(meta, "edit_image", no_edit)
    image, events = _run(d, s)
    assert image == photo and s["cleaned"] is None and not (d / "original.jpg").exists()
    assert events[-1]["text"] == "Nothing covering the object"


def test_built_sessions_are_left_alone(tmp_path, monkeypatch):
    d, s, photo = _session(tmp_path, monkeypatch)
    (d / "anatomy.json").write_text("{}")

    async def boom(*_a, **_k):
        raise AssertionError("no SAM call for a session that is already built")

    monkeypatch.setattr(meta, "segment", boom)
    image, events = _run(d, s)
    assert image == photo and events == []


def test_hand_threshold():
    assert clean.MIN_HAND_AREA * 640 * 480 > 20 * 20


def test_extra_photos_are_cleared_too(tmp_path, monkeypatch):
    d, s, photo = _session(tmp_path, monkeypatch)
    held, free = _jpeg((90, 90, 90)), _jpeg((30, 30, 30))
    (d / "extra-1.jpg").write_bytes(held)
    (d / "extra-2.jpg").write_bytes(free)

    async def fake_segment(image, phrase):
        # Only the first extra photo has a hand on the object.
        return [_match([300, 300, 600, 480])] if image == held else []

    async def fake_edit(jpeg, prompt, size=None):
        return _jpeg((10, 120, 10))

    monkeypatch.setattr(meta, "segment", fake_segment)
    monkeypatch.setattr(meta, "edit_image", fake_edit)
    image, events = _run(d, s)
    assert image == photo and s["cleaned"] is None and s["extras_cleaned"] == [1]
    assert (d / "original-extra-1.jpg").read_bytes() == held and (d / "extra-2.jpg").read_bytes() == free
    assert not (d / "original-extra-2.jpg").exists()
    # Kept originals are not mistaken for extra photos.
    assert [p.name for p in A._extra_photos(d)] == ["extra-1.jpg", "extra-2.jpg"]
    assert events[-1]["text"] == "Muse Image removed the hand from 1 of 3 photos and restored what it hid"
