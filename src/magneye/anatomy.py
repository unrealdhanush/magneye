"""Prompt and JSON schema for Muse Wonder's first draft of an object's anatomy, plus reference sanitizing."""

from __future__ import annotations

from typing import Any

MATERIALS = ["plastic", "metal", "pcb", "battery", "rubber", "glass", "magnet", "fabric", "fluid", "organic", "ceramic", "other"]
SHAPES = ["box", "rounded_box", "cylinder", "sphere", "capsule", "disc", "ring", "cone", "plate", "coil"]
LINKS = ["mechanical", "electrical", "signal", "fluid", "optical", "thermal", "magnetic", "air"]
EFFECTS = ["press", "rotate", "slide", "vibrate", "glow", "flow", "heat", "light"]

_vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_str = {"type": "string"}
_ids = {"type": "array", "items": _str}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


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
                    "kind": {"type": "string", "enum": ["exterior", "internal"]},
                    "function": _str,
                    "material": {"type": "string", "enum": MATERIALS},
                    "shape": {"type": "string", "enum": SHAPES},
                    "size_mm": _vec3,
                    "center_mm": _vec3,
                    "rotation_deg": _vec3,
                    "layer": {"type": "integer"},
                    "sam_prompt": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
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

Coordinate frame (millimetres, origin at the object's centre): treat the photo as the front view. +x points to the image's right, +y to the image's top, +z out of the image toward the camera. dimensions_mm is the object's full extent along x, y, z. Every part's center_mm and size_mm are in this frame and must fit inside the object's extent. rotation_deg is XYZ Euler rotation of the part's own shape.

Parts:
- 12 to 28 parts. Include the outer housing pieces as kind "exterior", plus every functionally important internal component as kind "internal" (boards, sensors, actuators, batteries, springs, mechanisms, fasteners that matter for disassembly).
- id is snake_case and unique. shape is one of {SHAPES}; pick the closest primitive and size it realistically. material is one of {MATERIALS}.
- layer is depth from the outside: 0 is the outer shell, then increasing as you go inward. It drives the exploded view and the disassembly order.
- sam_prompt: for parts visible in this photo, a short noun phrase (2 to 4 words) an open-vocabulary segmenter can find in the image, e.g. "thumbstick", "trigger button". null for parts hidden in this photo.
- object.sam_prompt: a short noun phrase that finds the whole object in the photo.
- confidence (0 to 1) is how sure you are this part exists as described in this specific model. Use well-known teardown knowledge when you recognise the exact product; otherwise infer from the category and say so in caveats.

Connections link parts that physically or electrically interact (kind is one of {LINKS}).

Scenarios (2 to 4): how the object works, as ordered chains of steps through parts, e.g. "press the trigger" -> trigger pivots -> sensor reads -> board processes -> radio transmits. Each step names a part_id, an effect from {EFFECTS}, a short label, and a duration_ms (300 to 1500). Make one scenario answer the user's prompt when it asks how something works.

Failures (2 to 5): realistic failure modes for this exact kind of object. culprit_part_id is the failing part, mechanism explains the physics in one or two sentences, checks are simple things the owner can try first, fix is the repair, and access_path is the ordered list of part_ids to remove to reach the culprit, outermost first.

Be honest: do not invent precise specs you do not know. Put uncertainty in caveats. Output only JSON matching the schema."""

def anatomy_user_text(prompt: str) -> str:
    return f"User prompt: {prompt.strip() or 'Show me how this works.'}\n\nReconstruct this object's anatomy."


def sanitize(anatomy: dict[str, Any]) -> dict[str, Any]:
    """Drop references to parts that do not exist, so the viewer never receives dangling ids."""
    ids = {p["id"] for p in anatomy.get("parts", [])}
    anatomy["connections"] = [c for c in anatomy.get("connections", []) if c["from"] in ids and c["to"] in ids]
    for s in anatomy.get("scenarios", []):
        s["steps"] = [st for st in s["steps"] if st["part_id"] in ids]
    anatomy["scenarios"] = [s for s in anatomy.get("scenarios", []) if s["steps"]]
    for f in anatomy.get("failures", []):
        f["access_path"] = [pid for pid in f["access_path"] if pid in ids]
    anatomy["failures"] = [f for f in anatomy.get("failures", []) if f["culprit_part_id"] in ids]
    return anatomy
