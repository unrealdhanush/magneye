"""Prompt and JSON schema for Muse Wonder's first draft of an object's anatomy, plus reference sanitizing."""

from __future__ import annotations

import re
from typing import Any

MATERIALS = ["plastic", "metal", "pcb", "battery", "rubber", "glass", "magnet", "fabric", "fluid", "organic", "ceramic", "other"]
SHAPES = ["box", "rounded_box", "cylinder", "sphere", "capsule", "disc", "ring", "cone", "plate", "coil", "gear", "spiral"]
BASIS = ["seen", "documented", "inferred"]
LINKS = ["mechanical", "electrical", "signal", "fluid", "optical", "thermal", "magnetic", "air"]
EFFECTS = ["press", "rotate", "slide", "vibrate", "glow", "flow", "heat", "light"]

_vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_str = {"type": "string"}
_ids = {"type": "array", "items": _str}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


MAX_COMPONENTS = 5

# A part drawn as several primitives (e.g. a thumbstick module = housing + shaft + two potentiometer drums).
COMPONENT = _obj(
    {
        "shape": {"type": "string", "enum": SHAPES},
        "material": {"type": "string", "enum": MATERIALS},
        "size_mm": _vec3,
        "offset_mm": _vec3,
        "rotation_deg": _vec3,
    }
)

SCENARIO = _obj(
    {
        "id": _str,
        "title": _str,
        "summary": _str,
        "steps": {
            "type": "array",
            "items": _obj(
                {
                    "part_id": _str,
                    "effect": {"type": "string", "enum": EFFECTS},
                    "label": _str,
                    "duration_ms": {"type": "integer"},
                }
            ),
        },
    }
)

FAILURE = _obj(
    {
        "id": _str,
        "symptom": _str,
        "culprit_part_id": _str,
        "mechanism": _str,
        "likelihood": {"type": "string", "enum": ["common", "occasional", "rare"]},
        "checks": _ids,
        "fix": _str,
        "access_path": _ids,
    }
)

ANATOMY_SCHEMA = _obj(
    {
        "object": _obj(
            {
                "name": _str,
                "maker": {"type": ["string", "null"]},
                "model": {"type": ["string", "null"]},
                "category": _str,
                "summary": _str,
                "dimensions_mm": _vec3,
                "confidence": {"type": "number"},
                "evidence": _ids,
                "sam_prompt": _str,
            }
        ),
        "parts": {
            "type": "array",
            "items": _obj(
                {
                    "id": _str,
                    "name": _str,
                    "kind": {"type": "string", "enum": ["exterior", "internal", "accessory"]},
                    "function": _str,
                    "material": {"type": "string", "enum": MATERIALS},
                    "shape": {"type": "string", "enum": SHAPES},
                    "size_mm": _vec3,
                    "center_mm": _vec3,
                    "rotation_deg": _vec3,
                    "layer": {"type": "integer"},
                    "sam_prompt": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "basis": {"type": "string", "enum": BASIS},
                    "components": {"type": "array", "items": COMPONENT, "maxItems": MAX_COMPONENTS},
                }
            ),
        },
        "connections": {
            "type": "array",
            "items": _obj({"from": _str, "to": _str, "kind": {"type": "string", "enum": LINKS}, "label": _str}),
        },
        "scenarios": {"type": "array", "items": SCENARIO},
        "failures": {"type": "array", "items": FAILURE},
        "caveats": _ids,
    }
)

ANATOMY_SYSTEM = f"""You are Magneye, an engineer who can see inside things. Given one photo of an object and the user's prompt, identify the object and reconstruct its internal anatomy as structured data a 3D viewer will render.

Coordinate frame (millimetres, origin at the object's centre): treat the photo as the front view. +x points to the image's right, +y to the image's top, +z out of the image toward the camera. dimensions_mm is the object's full extent along x, y, z. Every part's center_mm and size_mm are in this frame and must fit inside the object's extent.

Orientation: size_mm is always the part's extent along the object's x, y and z axes as it sits in the object, never in the part's own frame. Round shapes orient themselves from it: a disc, ring, gear or spiral faces along its thinnest dimension, and a cylinder, cone, capsule or coil runs along the dimension that differs from the other two. So a watch dial facing the camera is a disc of [30, 30, 1] and a crown sticking out to the right is a cylinder of [6, 4, 4], both with rotation_deg [0, 0, 0]. rotation_deg (XYZ Euler) is only for tilts off those axes, such as a pusher angled at 30 degrees; never use it to stand a shape up or turn it to face the camera. Components follow the same rules.

Parts:
- 12 to 28 parts. Include the outer housing pieces as kind "exterior", plus every functionally important internal component as kind "internal" (boards, sensors, actuators, batteries, springs, mechanisms, fasteners that matter for disassembly).
- Attachments outside the main body that are not part of its mechanism (watch strap or bracelet, clasp, cable, stand, lanyard) are kind "accessory", layer 0. Keep them to 1 to 3 parts; the viewer frames the main body and moves accessories aside when it explodes.
- id is snake_case and unique. shape is one of {SHAPES}; pick the closest primitive and size it realistically. material is one of {MATERIALS}.
- layer is depth from the outside: 0 is the outer shell, then increasing as you go inward. It drives the exploded view and the disassembly order.
- sam_prompt: for parts visible in this photo, the plain generic noun an open-vocabulary segmenter would use to find it: 1 to 3 words, concrete, no brand or product names, no letters or labels, no jargon (good: "thumbstick", "button", "strap", "lens"; bad: "Oculus menu button", "battery cover R", "triangular faceplate"). null for parts hidden in this photo.
- object.sam_prompt: the generic everyday noun for the whole object, 1 or 2 words, never a brand (good: "game controller", "camera", "kettle"; bad: "Oculus Touch VR controller").
- components: how the part looks, as 2 to 5 primitives, so the 3D view reads as the real thing rather than one block. offset_mm is each primitive's centre relative to the part's centre, in the same axes; together they should fill the part's size_mm. Examples: thumbstick module = metal housing box + cylinder shaft + two small cylinder potentiometer drums on adjacent sides; circuit board = thin pcb plate + a few small box chips + a connector; battery = cylinder body + small metal disc terminal; trigger = curved capsule blade + small cylinder pivot. Use [] for parts that really are one simple shape (a cover, a ring, a button cap).
- basis: where your knowledge of this part comes from. "seen" = visible in the photo or a reference view; "documented" = described in the reference research below (teardown, service notes, specs); "inferred" = neither, reasoned from how such objects are usually built.
- Shapes "gear" (a toothed wheel; size_mm is diameter, diameter, thickness) and "spiral" (a flat spiral spring such as a hairspring or mainspring) exist for mechanisms; use them instead of discs where they apply.
- confidence (0 to 1) is how sure you are this part exists as described in this specific model. Use well-known teardown knowledge when you recognise the exact product; otherwise infer from the category and say so in caveats.

Connections link parts that physically or electrically interact (kind is one of {LINKS}).

Scenarios (2 to 4): how the object works, as ordered chains of steps through parts, e.g. "press the trigger" -> trigger pivots -> sensor reads -> board processes -> radio transmits. Each step names a part_id, an effect from {EFFECTS}, a short label, and a duration_ms (300 to 1500). Make one scenario answer the user's prompt when it asks how something works.

Failures (2 to 5): realistic failure modes for this exact kind of object. culprit_part_id is the failing part, mechanism explains the physics in one or two sentences, checks are simple things the owner can try first, fix is the repair, and access_path is the ordered list of part_ids to remove to reach the culprit, outermost first.

Be honest: do not invent precise specs you do not know. Put uncertainty in caveats. Output only JSON matching the schema."""

RESEARCH_SYSTEM = """You research real objects for Magneye, which builds interactive 3D exploded views. Identify the exact object in the photo (maker, model, generation or reference number when visible), then search the web for its teardowns, exploded views, service or repair guides, and official specifications.

Write a blueprint brief a 3D modeller could build from:
1. Identification and how sure you are.
2. Overall dimensions (and which part of the object they cover).
3. Internal components, in assembly order from the front (the side facing the camera) to the back, each with approximate size in mm, material, and where it sits.
4. How the main mechanisms work, step by step.
5. Disassembly order, and common failures with their root causes.

Prefer teardowns and service documentation over listings and marketing. If documentation exists only for a closely related model, use it and say which. Mark anything you could not find a source for as unverified. Keep it under 900 words.

Budget: run at most 6 searches, the most targeted first (teardown or exploded view, then specifications, then common failures).

End with a "Sources" section: a numbered list of the pages you actually used, one per line as "N. Title — URL".

Then a last line "Photo search: <maker and model in 2 to 5 words>", the query that would find photos of this exact product (for example "Rolex Daytona" or "Oculus Touch controller")."""


def research_question(prompt: str) -> str:
    return f"User prompt: {prompt.strip() or 'Show me how this works.'}\n\nResearch the object in this photo and write the blueprint brief."


def anatomy_user_text(prompt: str, brief: str | None = None, views: list[dict[str, Any]] | None = None) -> str:
    reference = (f"\n\nReference research from the web (teardowns, service notes, specs). Prefer it over memory for internal "
                 f"layout, sizes and failure modes, and mark parts it describes as basis \"documented\":\n<research>\n{brief}\n</research>"
                 if brief else "")
    if views:
        listed = "; ".join(f"image {i + 2} = {v['view'].replace('_', ' ')} view ({v['note']})" for i, v in enumerate(views))
        reference += (f"\n\nThe first image is the user's photo: it is the front view and defines the coordinate frame. The others are "
                      f"reference photos of the same product from other angles: {listed}. Use them for depth (z), the side profile, "
                      f"the back and parts the front photo hides; parts visible in them are basis \"seen\" but get sam_prompt null "
                      f"unless they are also visible in the user's photo.")
    return f"User prompt: {prompt.strip() or 'Show me how this works.'}{reference}\n\nReconstruct this object's anatomy."


def photo_query(brief: str) -> tuple[str, str | None]:
    """Split the research brief's closing "Photo search:" line off: (brief, query or None)."""
    m = re.search(r"^\W*Photo search:\W*(.+?)\W*$", brief, re.M | re.I)
    if not m:
        return brief, None
    return (brief[:m.start()] + brief[m.end():]).strip(), m.group(1).strip() or None


ROUND = {"cylinder", "disc", "ring", "cone", "capsule", "coil", "gear", "spiral"}


def _quarter_turns_only(rotation: list[float]) -> bool:
    return any(rotation) and all(abs(a - round(a / 90) * 90) < 1e-6 for a in rotation)


def sanitize(anatomy: dict[str, Any]) -> dict[str, Any]:
    """Drop references to parts that do not exist, so the viewer never receives dangling ids. Round shapes orient
    themselves from size_mm, so a whole quarter-turn rotation on one only turns it a second time; drop it."""
    for p in anatomy.get("parts", []):
        for shape in [p, *(p.get("components") or [])]:
            if shape.get("shape") in ROUND and _quarter_turns_only(shape.get("rotation_deg") or []):
                shape["rotation_deg"] = [0, 0, 0]
    ids = {p["id"] for p in anatomy.get("parts", [])}
    anatomy["connections"] = [c for c in anatomy.get("connections", []) if c["from"] in ids and c["to"] in ids]
    for s in anatomy.get("scenarios", []):
        s["steps"] = [st for st in s["steps"] if st["part_id"] in ids]
    anatomy["scenarios"] = [s for s in anatomy.get("scenarios", []) if s["steps"]]
    for f in anatomy.get("failures", []):
        f["access_path"] = [pid for pid in f["access_path"] if pid in ids]
    anatomy["failures"] = [f for f in anatomy.get("failures", []) if f["culprit_part_id"] in ids]
    return anatomy
