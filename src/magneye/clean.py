"""Clear what covers the object before anything reads the photo.

People hold things up to the camera, so a hand often hides part of the object: SAM's outline then has a bite out of
it and the 3D shell inherits the gap. SAM 3.1 looks for a hand; if one is there, Muse Image removes it and restores
the part it hid, keeping the rest of the photo as it was. The restored area is generated, not observed, so callers
keep the original and say the photo was cleaned.
"""

from __future__ import annotations

from typing import Any

from . import meta

# A hand smaller than this share of the photo is in the background, not holding the object.
MIN_HAND_AREA = 0.01
MIN_CONFIDENCE = 0.5

PROMPT = ("Remove the hand and fingers holding the object. Restore every part of the object they covered so it is "
          "complete, matching its shape, colour, material and lighting. Keep everything else exactly as it is: the same "
          "framing, camera angle, background, and the object's position and size. Do not add, move or restyle anything.")


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


async def clear_hands(jpeg: bytes, size: tuple[int, int]) -> dict[str, Any] | None:
    """{"image": edited JPEG bytes, "hands": n} when a hand was found and removed, else None."""
    hands = [m for m in await meta.segment(jpeg, "hand")
             if (m["confidence"] or 0) >= MIN_CONFIDENCE and _area(m["box"]) >= MIN_HAND_AREA * size[0] * size[1]]
    if not hands:
        return None
    return {"image": await meta.edit_image(jpeg, PROMPT, size=size), "hands": len(hands)}
