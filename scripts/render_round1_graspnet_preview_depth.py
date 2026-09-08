"""Render Contact-GraspNet candidates from raw MuJoCo metric depth."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


INPUT_DIR = Path("sam_test/round1_scene")
SAM3_DIR = Path("sam_test/round1_sam3")
OUTPUT_DIR = Path("sam_test/round1_graspnet")
PROMPT = "extruded_handle_of_the_brown_square_nut"


def encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def project(point: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float] | None:
    if point[2] <= 1e-5:
        return None
    uv = intrinsics @ point
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def run_case(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    label: str,
    *,
    filter_grasps: bool,
) -> tuple[int, int]:
    response = requests.post(
        "http://127.0.0.1:8115/plan",
        json={
            "depth_base64": encode_array(depth),
            "cam_K_base64": encode_array(intrinsics),
            "segmap_base64": encode_array(mask.astype(np.int32)),
            "segmap_id": 1,
            "local_regions": True,
            "filter_grasps": filter_grasps,
            "skip_border_objects": False,
            "z_range": [0.2, 2.0],
            "forward_passes": 1,
            "max_retries": 1,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    grasps = np.load(io.BytesIO(base64.b64decode(data["grasps_base64"])))
    scores = np.load(io.BytesIO(base64.b64decode(data["scores_base64"]))).reshape(-1)
    order = np.argsort(-scores)[:10] if len(scores) else np.array([], dtype=int)
    overlay = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(overlay)
    mask_overlay = rgb.copy()
    mask_overlay[mask] = (0.45 * mask_overlay[mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(mask_overlay).save(OUTPUT_DIR / f"{label}_input_mask.png")
    results = []
    colors = [(0, 255, 0), (255, 220, 0), (0, 180, 255), (255, 80, 255)]
    for rank, index in enumerate(order, start=1):
        transform = grasps[index]
        center = transform[:3, 3]
        point = project(center, intrinsics)
        if point is None:
            continue
        color = colors[(rank - 1) % len(colors)]
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
        draw.text((x + 7, y - 7), f"{rank}:{scores[index]:.3f}", fill=color)
        for axis in range(3):
            endpoint = project(center + 0.04 * transform[:3, axis], intrinsics)
            if endpoint is not None:
                draw.line((x, y, endpoint[0], endpoint[1]), fill=color, width=2)
        results.append({"rank": rank, "score": float(scores[index]), "translation_camera": center.tolist()})
    overlay.save(OUTPUT_DIR / f"{label}_graspnet_candidates.png")
    (OUTPUT_DIR / f"{label}_candidates.json").write_text(json.dumps(results, indent=2))
    return len(grasps), len(results)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(INPUT_DIR / "rgb.png").convert("RGB"))
    depth = np.load(INPUT_DIR / "depth_m.npy").astype(np.float32)
    intrinsics = np.load(INPUT_DIR / "intrinsics.npy").astype(np.float32)
    sam3_mask = np.asarray(Image.open(SAM3_DIR / f"{PROMPT}_top1_mask.png").convert("L")) > 0
    strict_counts = run_case(
        rgb, depth, intrinsics, sam3_mask, "handle_sam3_top1_strict", filter_grasps=True
    )
    relaxed_counts = run_case(
        rgb, depth, intrinsics, sam3_mask, "handle_sam3_top1_unfiltered", filter_grasps=False
    )
    Image.fromarray((depth * 1000).astype(np.uint16)).save(OUTPUT_DIR / "depth_mm.png")
    (OUTPUT_DIR / "README.txt").write_text(
        "Input is a raw MuJoCo observation captured in sam_test/round1_scene.\n"
        "depth_m.npy is metric depth in meters and intrinsics.npy is the camera K.\n"
        "The mask is SAM3's top text-segmentation result, not a SAM2 artifact.\n"
        "strict uses the runtime filter_grasps=True setting; unfiltered is a\n"
        "diagnostic only, used to identify candidates removed by that filter.\n"
    )
    print(f"strict candidates={strict_counts[0]}, visualized={strict_counts[1]}")
    print(f"unfiltered candidates={relaxed_counts[0]}, visualized={relaxed_counts[1]}")


if __name__ == "__main__":
    main()
