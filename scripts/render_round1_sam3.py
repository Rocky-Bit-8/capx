"""Render SAM3 text-segmentation diagnostics from a raw MuJoCo RGB frame."""

from __future__ import annotations

import json
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import requests


INPUT_DIR = Path("sam_test/round1_scene")
OUTPUT_DIR = Path("sam_test/round1_sam3")
PROMPTS = (
    "extruded handle of the brown square nut",
    "white hollow center of the brown square nut",
    "brown square block",
    "center hole of the brown square nut",
    "peg of the brown square block",
    "peg on the brown square block",
)
COLORS = ((255, 64, 64), (64, 160, 255), (80, 220, 120))


def slug(text: str) -> str:
    return text.replace(" ", "_")


def render(image: np.ndarray, results: list[dict], prompt: str) -> None:
    stem = slug(prompt)
    metadata: list[dict] = []
    summary = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(summary)
    for index, result in enumerate(results[:3]):
        mask = np.asarray(result["mask"], dtype=bool)
        color = np.asarray(COLORS[index], dtype=np.uint8)
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
        box = [float(value) for value in result["box"]]
        score = float(result["score"])
        individual = Image.fromarray(overlay)
        individual_draw = ImageDraw.Draw(individual)
        individual_draw.rectangle(box, outline=tuple(color.tolist()), width=3)
        individual_draw.text((box[0] + 2, max(0, box[1] - 14)), f"#{index + 1} {score:.4f}", fill=tuple(color.tolist()))
        individual.save(OUTPUT_DIR / f"{stem}_top{index + 1}_overlay.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUTPUT_DIR / f"{stem}_top{index + 1}_mask.png")
        draw.rectangle(box, outline=tuple(color.tolist()), width=3)
        draw.text((box[0] + 2, max(0, box[1] - 14)), f"#{index + 1} {score:.4f}", fill=tuple(color.tolist()))
        metadata.append({"rank": index + 1, "score": score, "box": box, "pixels": int(mask.sum())})
    summary.save(OUTPUT_DIR / f"{stem}_summary.png")
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps({"prompt": prompt, "matches": metadata}, indent=2))
    print(f"{prompt}: {len(results)} results; saved top {len(metadata)}")


def segment(image: Image.Image, prompt: str) -> list[dict]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = requests.post(
        "http://127.0.0.1:8114/segment",
        json={
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text_prompt": prompt,
        },
        timeout=180,
    )
    response.raise_for_status()
    results = response.json()["results"]
    decoded = []
    for result in results:
        shape = tuple(result["shape"])
        mask = np.frombuffer(base64.b64decode(result["mask_base64"]), dtype=np.uint8)
        decoded.append({**result, "mask": mask.reshape(shape).astype(bool)})
    return decoded


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT_DIR / "rgb.png").convert("RGB")
    image_np = np.asarray(image)
    for prompt in PROMPTS:
        render(image_np, segment(image, prompt), prompt)


if __name__ == "__main__":
    main()
