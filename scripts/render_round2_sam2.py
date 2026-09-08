"""Re-run local Meta SAM2 with round1 mask-centroid point prompts.

Requires ``PYTHONPATH`` to include the Meta SAM2 source tree and its retained
Hydra dependency. The model checkpoint is loaded only from this workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


INPUT_DIR = Path("sam_test/round1_sam2")
OUTPUT_DIR = Path("sam_test/round2_sam2")
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
COLORS = ((255, 64, 64), (64, 160, 255), (80, 220, 120))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ``segmentation_image.jpg`` is a black-background debug artifact, not the
    # camera RGB frame. Reconstruct the scene RGB from the round1 overlays;
    # their masks differ, so a per-pixel median removes the colored overlays.
    reference_frames = [
        np.asarray(Image.open(INPUT_DIR / f"{prompt}_sam2_overlay.jpg").convert("RGB"))
        for prompt in PROMPTS
    ]
    image = np.median(np.stack(reference_frames), axis=0).astype(np.uint8)
    Image.fromarray(image).save(OUTPUT_DIR / "reference_rgb_reconstructed.png")
    model = build_sam2(CONFIG, str(CHECKPOINT), device="cuda")
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)

    for prompt in PROMPTS:
        round1_mask = np.asarray(Image.open(INPUT_DIR / f"{prompt}_sam2_mask.png").convert("L")) > 0
        ys, xs = np.nonzero(round1_mask)
        if len(xs) == 0:
            print(f"{prompt}: skipped empty round1 mask")
            continue
        point = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        order = np.argsort(-scores)
        summary = Image.fromarray(image.copy())
        summary_draw = ImageDraw.Draw(summary)
        metadata = {"prompt": prompt.replace("_", " "), "point_xy": point[0].tolist(), "matches": []}
        for rank, index in enumerate(order[:3], start=1):
            mask = masks[index].astype(bool)
            color = np.asarray(COLORS[rank - 1], dtype=np.uint8)
            overlay = image.copy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
            overlay_image = Image.fromarray(overlay)
            draw = ImageDraw.Draw(overlay_image)
            x, y = point[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=tuple(color.tolist()), width=2)
            draw.text((x + 7, y - 12), f"#{rank} {scores[index]:.4f}", fill=tuple(color.tolist()))
            overlay_image.save(OUTPUT_DIR / f"{prompt}_top{rank}_overlay.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(OUTPUT_DIR / f"{prompt}_top{rank}_mask.png")
            summary_draw.text((x + 7, y + 4 + 13 * rank), f"#{rank} {scores[index]:.4f}", fill=tuple(color.tolist()))
            metadata["matches"].append({"rank": rank, "score": float(scores[index]), "pixels": int(mask.sum())})
        summary_draw.ellipse((point[0, 0] - 5, point[0, 1] - 5, point[0, 0] + 5, point[0, 1] + 5), outline=(255, 255, 0), width=2)
        summary.save(OUTPUT_DIR / f"{prompt}_summary.png")
        (OUTPUT_DIR / f"{prompt}.json").write_text(json.dumps(metadata, indent=2))
        print(f"{prompt}: point={point[0].round(1).tolist()}, scores={scores[order[:3]].round(4).tolist()}")


if __name__ == "__main__":
    main()
