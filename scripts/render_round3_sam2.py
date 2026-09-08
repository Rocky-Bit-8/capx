"""Run local Meta SAM2 and select the best local, non-background mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


INPUT_DIR = Path("sam_test/round1_sam2")
OUTPUT_DIR = Path("sam_test/round3_sam2")
CONFIG = "configs/sam2/sam2_hiera_l.yaml"
CHECKPOINT = Path("sam2_hiera_large.pt")
PROMPTS = (
    "extruded_handle_of_the_brown_square_nut",
    "white_hollow_center_of_the_brown_square_nut",
    "brown_square_block",
    "center_hole_of_the_brown_square_nut",
    "peg_of_the_brown_square_block",
    "peg_on_the_brown_square_block",
)
MAX_LOCAL_AREA_FRACTION = 0.20


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ``segmentation_image.jpg`` is a black-background debug artifact. Each
    # round1 overlay contains the real RGB frame with only its own mask drawn.
    # A pixel-wise median removes those differing mask overlays.
    reference_frames = [
        np.asarray(Image.open(INPUT_DIR / f"{prompt}_sam2_overlay.jpg").convert("RGB"))
        for prompt in PROMPTS
    ]
    image = np.median(np.stack(reference_frames), axis=0).astype(np.uint8)
    Image.fromarray(image).save(OUTPUT_DIR / "reference_rgb_reconstructed.png")
    height, width = image.shape[:2]
    max_area = int(height * width * MAX_LOCAL_AREA_FRACTION)
    predictor = SAM2ImagePredictor(build_sam2(CONFIG, str(CHECKPOINT), device="cuda"))
    # SAM2's preprocessing can mutate its NumPy input; keep the visualization
    # source separate so overlays are always drawn on the original RGB image.
    predictor.set_image(image.copy())

    for prompt in PROMPTS:
        seed_mask = np.asarray(Image.open(INPUT_DIR / f"{prompt}_sam2_mask.png").convert("L")) > 0
        ys, xs = np.nonzero(seed_mask)
        point = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        areas = masks.reshape(len(masks), -1).sum(axis=1)
        valid = np.flatnonzero((areas > 0) & (areas <= max_area))
        selected = int(valid[np.argmax(scores[valid])]) if len(valid) else int(np.argmin(areas))
        mask = masks[selected].astype(bool)

        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 64, 64])).astype(np.uint8)
        output = Image.fromarray(overlay)
        draw = ImageDraw.Draw(output)
        x, y = point[0]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 0), width=2)
        draw.text((x + 7, y - 12), f"selected #{selected + 1}: {scores[selected]:.4f}", fill=(255, 255, 0))
        output.save(OUTPUT_DIR / f"{prompt}_selected_overlay.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUTPUT_DIR / f"{prompt}_selected_mask.png")
        details = {
            "prompt": prompt.replace("_", " "),
            "point_xy": point[0].tolist(),
            "max_local_area": max_area,
            "selected_index": selected,
            "selected_score": float(scores[selected]),
            "selected_pixels": int(areas[selected]),
            "candidates": [
                {"index": int(i), "score": float(scores[i]), "pixels": int(areas[i])}
                for i in range(len(masks))
            ],
        }
        (OUTPUT_DIR / f"{prompt}.json").write_text(json.dumps(details, indent=2))
        print(f"{prompt}: selected #{selected + 1}, score={scores[selected]:.4f}, pixels={areas[selected]}")


if __name__ == "__main__":
    main()
