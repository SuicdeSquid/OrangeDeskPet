from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "transparent"
OUT = ROOT / "final"
OUT.mkdir(exist_ok=True)

CANVAS = (192, 208)
MAX_CONTENT = (184, 200)
DIRECTIONS = [
    "center", "up", "up-right", "right", "down-right",
    "down", "down-left", "left", "up-left",
]


def orange_median(image):
    arr = np.asarray(image).astype(np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255.0
    mask = (
        (alpha > 0.7)
        & (rgb[..., 0] > 85)
        & (rgb[..., 0] > rgb[..., 1] * 1.08)
        & (rgb[..., 1] > rgb[..., 2] * 1.12)
    )
    return np.median(rgb[mask], axis=0)


def normalize_orange(image, target):
    arr = np.asarray(image).astype(np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255.0
    mask = (
        (alpha > 0.7)
        & (rgb[..., 0] > 85)
        & (rgb[..., 0] > rgb[..., 1] * 1.08)
        & (rgb[..., 1] > rgb[..., 2] * 1.12)
    )
    current = np.median(rgb[mask], axis=0)
    ratio = np.clip(target / np.maximum(current, 1), 0.86, 1.14)
    corrected = np.clip(rgb * ratio, 0, 255)
    saturation_hint = np.clip((rgb[..., 0] - rgb[..., 2]) / 90.0, 0, 1)
    weight = (saturation_hint * alpha * mask * 0.68)[..., None]
    arr[..., :3] = rgb * (1 - weight) + corrected * weight
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


sources = {
    direction: Image.open(SRC / f"gaze-{direction}.png").convert("RGBA")
    for direction in DIRECTIONS
}

# One shared crop and scale keeps the body anchored while head and tail react.
bboxes = []
for image in sources.values():
    bbox = image.getchannel("A").point(lambda x: 255 if x > 12 else 0).getbbox()
    if not bbox:
        raise ValueError("A gaze source contains no visible pixels")
    bboxes.append(bbox)

shared_bbox = (
    min(b[0] for b in bboxes),
    min(b[1] for b in bboxes),
    max(b[2] for b in bboxes),
    max(b[3] for b in bboxes),
)
crop_width = shared_bbox[2] - shared_bbox[0]
crop_height = shared_bbox[3] - shared_bbox[1]
scale = min(MAX_CONTENT[0] / crop_width, MAX_CONTENT[1] / crop_height)
size = (round(crop_width * scale), round(crop_height * scale))

target_orange = orange_median(sources["center"])
prepared = {}
for direction, image in sources.items():
    corrected = normalize_orange(image, target_orange)
    crop = corrected.crop(shared_bbox).resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    canvas.alpha_composite(crop, ((CANVAS[0] - size[0]) // 2, (CANVAS[1] - size[1]) // 2))
    prepared[direction] = canvas
    canvas.save(OUT / f"gaze-{direction}.png")

# Preview only: runtime selects by pointer zone instead of playing this sequence.
preview_order = [
    "center", "right", "up-right", "up", "up-left", "left",
    "down-left", "down", "down-right", "right", "center",
]
preview = []
for direction in preview_order:
    background = Image.new("RGB", CANVAS, (239, 238, 234))
    frame = prepared[direction]
    background.paste(frame, mask=frame.getchannel("A"))
    preview.append(background)
preview[0].save(
    OUT / "gaze-follow-preview.gif",
    save_all=True,
    append_images=preview[1:],
    duration=[520, 260, 260, 260, 260, 260, 260, 260, 260, 260, 520],
    loop=0,
    disposal=2,
)

contact_grid = [
    ["up-left", "up", "up-right"],
    ["left", "center", "right"],
    ["down-left", "down", "down-right"],
]
contact = Image.new("RGB", (CANVAS[0] * 3, CANVAS[1] * 3), (239, 238, 234))
draw = ImageDraw.Draw(contact)
for row, directions in enumerate(contact_grid):
    for column, direction in enumerate(directions):
        x, y = column * CANVAS[0], row * CANVAS[1]
        frame = prepared[direction]
        contact.paste(frame, (x, y), frame)
        draw.text((x + 8, y + 8), direction, fill=(88, 75, 63))
contact.save(OUT / "gaze-follow-contact-sheet.png")

config = {
    "schemaVersion": 1,
    "id": "gaze_follow_front",
    "displayName": "正面坐姿视线跟随",
    "version": "1.0.0",
    "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
    "anchor": {"x": 0.5, "y": 0.971},
    "view": "front",
    "kind": "pointer-reactive-set",
    "trackingRadiusPxAt100Scale": 420,
    "centerDeadZonePxAt100Scale": 52,
    "switchDebounceMs": 80,
    "returnToCenterDelayMs": 180,
    "directionSectors": 8,
    "sectorWidthDegrees": 45,
    "sectorHysteresisDegrees": 7,
    "directionFrames": {
        "up": "gaze-up.png",
        "up-right": "gaze-up-right.png",
        "right": "gaze-right.png",
        "down-right": "gaze-down-right.png",
        "down": "gaze-down.png",
        "down-left": "gaze-down-left.png",
        "left": "gaze-left.png",
        "up-left": "gaze-up-left.png"
    },
    "centerFrame": "gaze-center.png",
    "selectionRule": "center inside dead-zone; outside it, convert atan2(dy, dx) to the nearest 45-degree sector with 7-degree hysteresis",
    "notes": "Head and tail join the reaction; do not blend with eye-only drafts."
}
(OUT / "gaze-map.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

print(f"Built {len(DIRECTIONS)} pointer-reactive gaze frames in {OUT}")
