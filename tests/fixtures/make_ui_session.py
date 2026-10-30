"""Create a stand-in session so the UI can be exercised without Meta API calls.

The anatomy below is hand-written test data for a generic handheld controller, used only to
check rendering, playback and layout. Real sessions get their anatomy from Muse Wonder.

    uv run python tests/fixtures/make_ui_session.py   # prints the session id
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SID = "uifixture0000001"
W, H = 1200, 900
# Object box in the photo and its real-world size.
BOX = (420, 120, 780, 780)
DIMS = [72.0, 132.0, 64.0]


def outline() -> list[tuple[float, float]]:
    """A ring on top of a grip, in image pixels."""
    cx, cy, r = 600, 300, 175
    top = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in [math.pi * i / 24 for i in range(-2, 27)]]
    return top + [(690, 470), (705, 780), (495, 780), (510, 470)]


def make() -> str:
    d = ROOT / ".data" / "sessions" / SID
    d.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), (236, 238, 242))
    draw = ImageDraw.Draw(img)
    pts = outline()
    draw.polygon(pts, fill=(40, 42, 48))
    draw.ellipse((470, 170, 730, 430), fill=(236, 238, 242))
    draw.ellipse((560, 520, 620, 580), fill=(25, 25, 28))  # thumbstick
    draw.ellipse((620, 610, 652, 642), fill=(70, 72, 80))  # button A
    draw.ellipse((560, 640, 592, 672), fill=(70, 72, 80))  # button B
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    (d / "image.jpg").write_bytes(buf.getvalue())

    def part(id, name, kind, material, shape, size, center, layer, fn, sam=None, rot=(0, 0, 0), conf=0.8):
        return {"id": id, "name": name, "kind": kind, "function": fn, "material": material, "shape": shape,
                "size_mm": list(size), "center_mm": list(center), "rotation_deg": list(rot), "layer": layer,
                "sam_prompt": sam, "confidence": conf}

    parts = [
        part("tracking_ring", "Tracking ring", "exterior", "plastic", "ring", (70, 70, 18), (0, 38, 0), 0, "Holds IR LEDs the headset cameras track.", "ring"),
        part("front_shell", "Front grip shell", "exterior", "plastic", "rounded_box", (40, 60, 10), (0, -30, 24), 0, "Outer housing facing the user."),
        part("back_shell", "Back grip shell", "exterior", "plastic", "rounded_box", (40, 60, 10), (0, -30, -24), 0, "Rear housing with the battery door."),
        part("thumbstick_cap", "Thumbstick cap", "exterior", "rubber", "disc", (12, 4, 12), (0, -12, 30), 0, "Where your thumb rests.", "thumbstick", rot=(90, 0, 0)),
        part("button_a", "A button", "exterior", "plastic", "cylinder", (6, 4, 6), (7, -26, 30), 0, "Face button.", "button", rot=(90, 0, 0)),
        part("trigger", "Index trigger", "exterior", "plastic", "capsule", (14, 30, 10), (0, -18, -30), 0, "Read by a hall sensor as you pull."),
        part("ir_leds", "IR LED array", "internal", "pcb", "ring", (66, 66, 4), (0, 38, 0), 1, "Infrared LEDs blinking in a pattern for tracking."),
        part("stick_module", "Thumbstick module", "internal", "metal", "box", (16, 10, 16), (0, -12, 18), 1, "Two potentiometers read the stick's X and Y."),
        part("main_pcb", "Main board", "internal", "pcb", "plate", (30, 50, 2), (0, -32, 8), 2, "Microcontroller, radio and sensor inputs."),
        part("imu", "IMU", "internal", "pcb", "box", (4, 4, 1), (4, -40, 10), 2, "Accelerometer + gyroscope for motion."),
        part("radio", "Radio antenna", "internal", "metal", "plate", (10, 20, 1), (-10, -20, 8), 2, "Sends inputs to the headset."),
        part("haptic", "Haptic actuator", "internal", "magnet", "cylinder", (12, 16, 12), (0, -52, 4), 3, "Voice-coil motor that buzzes the palm."),
        part("battery", "AA battery", "internal", "battery", "cylinder", (14, 50, 14), (0, -38, -12), 3, "Powers the controller."),
        part("battery_spring", "Battery spring", "internal", "metal", "coil", (8, 8, 8), (0, -66, -12), 3, "Keeps contact with the battery's negative end."),
    ]
    anatomy = {
        "object": {"name": "Handheld VR controller (UI fixture)", "maker": None, "model": None, "category": "VR controller",
                   "summary": "Hand-written stand-in data for checking the interface.", "dimensions_mm": DIMS,
                   "confidence": 0.5, "evidence": ["Fixture data, not a model output"], "sam_prompt": "controller"},
        "parts": parts,
        "connections": [{"from": "stick_module", "to": "main_pcb", "kind": "electrical", "label": "X/Y voltages"}],
        "scenarios": [{
            "id": "stick_push", "title": "Push the thumbstick", "summary": "Stick tilt becomes a movement signal in the headset.",
            "steps": [
                {"part_id": "thumbstick_cap", "effect": "press", "label": "Thumb tilts the cap", "duration_ms": 900},
                {"part_id": "stick_module", "effect": "rotate", "label": "Pots change resistance", "duration_ms": 1100},
                {"part_id": "main_pcb", "effect": "glow", "label": "MCU samples X/Y", "duration_ms": 900},
                {"part_id": "radio", "effect": "flow", "label": "Radio sends packet", "duration_ms": 900},
            ]}],
        "failures": [{
            "id": "drift", "symptom": "Character moves without touching the stick", "culprit_part_id": "stick_module",
            "mechanism": "Worn carbon tracks inside the potentiometers report an off-centre value at rest.",
            "likelihood": "common", "checks": ["Recalibrate the stick", "Blow out dust under the cap"],
            "fix": "Replace the thumbstick module.", "access_path": ["back_shell", "battery", "front_shell"]}],
        "caveats": ["Fixture data for UI testing."],
    }
    (d / "anatomy.json").write_text(json.dumps(anatomy, indent=1))

    path = "M " + " L ".join(f"{x - BOX[0]:.1f} {y - BOX[1]:.1f}" for x, y in pts) + " Z"
    mw, mh = BOX[2] - BOX[0], BOX[3] - BOX[1]
    obj = {"box": list(BOX), "confidence": 0.95, "mask_svg": path, "mask_bounds": list(BOX), "mask_size": [mw, mh]}
    stick = {"box": [560, 520, 620, 580], "confidence": 0.9, "mask_bounds": [560, 520, 620, 580], "mask_size": [60, 60],
             "mask_svg": "M 30 0 L 60 30 L 30 60 L 0 30 Z"}
    ground = {"image_size": [W, H], "object": obj, "parts": {"thumbstick_cap": stick}, "errors": {}}
    (d / "ground.json").write_text(json.dumps(ground, indent=1))
    (d / "session.json").write_text(json.dumps({"prompt": "", "size": [W, H], "image_sha": "fixture", "history": []}))
    return SID


if __name__ == "__main__":
    sys.stdout.write(make() + "\n")
