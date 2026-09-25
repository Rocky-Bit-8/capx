#!/usr/bin/env python3
"""Capture a Nut Assembly scene and validate SAM3 + Contact-GraspNet."""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "output_nut"
SAM3_URL = "http://127.0.0.1:8114"
GRASPNET_URL = "http://127.0.0.1:8115"


def _next_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_ROOT / f"test{index}").exists():
        index += 1
    path = OUTPUT_ROOT / f"test{index}"
    path.mkdir()
    return path


def _service_up(url: str) -> bool:
    try:
        requests.get(url, timeout=1.5)
        return True
    except requests.RequestException:
        return False


def _start_service(module: str, port: int, log_path: Path) -> subprocess.Popen[Any] | None:
    if _service_up(f"http://127.0.0.1:{port}"):
        print(f"Using existing service on port {port}")
        return None
    log = log_path.open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", module, "--device", "cuda", "--port", str(port), "--host", "127.0.0.1"],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{module} exited early; see {log_path}")
        if _service_up(f"http://127.0.0.1:{port}"):
            print(f"Started {module} on port {port}")
            return process
        time.sleep(2)
    process.terminate()
    raise TimeoutError(f"Timed out waiting for {module} on port {port}; see {log_path}")


def _save_top_grasps(path: Path, grasps: np.ndarray, scores: np.ndarray, contacts: np.ndarray) -> None:
    grasps = np.asarray(grasps)
    scores = np.asarray(scores).reshape(-1)
    contacts = np.asarray(contacts)
    count = min(20, len(grasps), len(scores))
    order = np.argsort(scores)[::-1][:count]
    all_rows = []
    for index in np.argsort(scores)[::-1]:
        row = {
            "source_index": int(index),
            "score": float(scores[index]),
            "pose_camera": np.asarray(grasps[index]).tolist(),
        }
        if len(contacts) > index:
            row["contact_point_camera"] = np.asarray(contacts[index]).tolist()
        all_rows.append(row)
    rows = []
    for rank, index in enumerate(order, start=1):
        row = {
            "rank": rank,
            "source_index": int(index),
            "score": float(scores[index]),
            "pose_camera": np.asarray(grasps[index]).tolist(),
        }
        if len(contacts) > index:
            row["contact_point_camera"] = np.asarray(contacts[index]).tolist()
        rows.append(row)
    (path / "top20_grasps.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (path / "all_grasps.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(path / "top20_grasps.npy", np.asarray(grasps)[order])
    np.save(path / "top20_scores.npy", scores[order])
    np.save(path / "all_grasps.npy", np.asarray(grasps))
    np.save(path / "all_scores.npy", scores)
    np.save(path / "all_contact_points.npy", contacts)


def _save_visualizations(
    path: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    grasps: np.ndarray,
    scores: np.ndarray,
    contacts: np.ndarray,
) -> None:
    """Save lightweight RGB/depth/mask and grasp overlays without matplotlib."""
    base = Image.fromarray(rgb).convert("RGBA")
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[mask] = (30, 220, 80, 115)
    overlay = Image.alpha_composite(base, Image.fromarray(mask_rgba, mode="RGBA"))
    Image.alpha_composite(overlay, Image.fromarray(np.zeros_like(mask_rgba), mode="RGBA")).convert("RGB").save(
        path / "sam3_overlay.png"
    )

    valid = np.isfinite(depth) & (depth > 0)
    depth_view = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        depth_view[..., 0] = (255 * scaled).astype(np.uint8)
        depth_view[..., 1] = (255 * (1 - np.abs(scaled - 0.5) * 2)).astype(np.uint8)
        depth_view[..., 2] = (255 * (1 - scaled)).astype(np.uint8)
    Image.fromarray(depth_view).save(path / "depth_colormap.png")

    # Draw metric-depth contours over the RGB frame for geometric inspection.
    # Invalid pixels are masked so they do not create artificial contour lines.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        contour_depth = np.asarray(depth, dtype=float).copy()
        contour_depth[~valid] = np.nan
        contour_depth *= 1000.0  # draw and label contours in millimeters
        figure, axis = plt.subplots(figsize=(8, 8), dpi=128)
        axis.imshow(rgb)
        finite_values = contour_depth[np.isfinite(contour_depth)]
        if finite_values.size:
            # One contour per millimeter, matching birdview_depth_mm.npy.
            low_mm = int(np.floor(np.percentile(finite_values, 1)))
            high_mm = int(np.ceil(np.percentile(finite_values, 99)))
            levels = np.arange(low_mm, high_mm + 1, dtype=float)
            if levels.size >= 2:
                contours = axis.contour(contour_depth, levels=levels, cmap="turbo", linewidths=0.8)
                # Label every 10 mm to keep dense 1 mm contours readable.
                label_levels = levels[(levels.astype(int) - low_mm) % 10 == 0]
                if label_levels.size:
                    axis.clabel(contours, levels=label_levels, inline=True, fontsize=6, fmt="%.0f mm")
                figure.colorbar(contours, ax=axis, fraction=0.046, pad=0.04, label="Depth (mm)")
        axis.set_title("Metric depth contours (1 mm spacing)")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(path / "depth_contours.png", bbox_inches="tight")
        plt.close(figure)
    except Exception as exc:
        (path / "depth_contours_error.txt").write_text(str(exc), encoding="utf-8")

    K = np.asarray(intrinsics, dtype=float)
    contacts = np.asarray(contacts).reshape(-1, 3) if np.asarray(contacts).size else np.empty((0, 3))
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(scores)[::-1][: min(20, len(contacts), len(scores))]

    def draw_contacts(image: Image.Image, indices: np.ndarray, annotate: bool) -> None:
        draw = ImageDraw.Draw(image)
        for rank, index in enumerate(indices, start=1):
            point = contacts[index]
            if not np.all(np.isfinite(point)) or point[2] <= 1e-6:
                continue
            u = int(round(K[0, 0] * point[0] / point[2] + K[0, 2]))
            v = int(round(K[1, 1] * point[1] / point[2] + K[1, 2]))
            if not (0 <= u < image.width and 0 <= v < image.height):
                continue
            if annotate:
                radius = 7 if rank == 1 else 5
                color = (255, 220, 30) if rank == 1 else (30, 220, 255)
                draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=2)
                draw.text((u + radius + 2, v - radius), str(rank), fill=color)
            else:
                draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(255, 80, 40), outline=(255, 230, 80))

    image = Image.fromarray(rgb).convert("RGB")
    draw_contacts(image, order, annotate=True)
    image.save(path / "top20_grasps_overlay.png")

    all_image = Image.fromarray(rgb).convert("RGB")
    draw_contacts(all_image, np.argsort(scores)[::-1], annotate=False)
    all_image.save(path / "all_grasps_overlay.png")


def _numpy_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_numpy(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)))


def _isolate_table_objects(
    depth: np.ndarray, margin_fraction: float = 0.08, object_height_mm: float = 3.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the table and objects above it, excluding far background/robot edges."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    h, w = depth.shape
    y0, y1 = int(h * margin_fraction), int(h * (1.0 - margin_fraction))
    x0, x1 = int(w * margin_fraction), int(w * (1.0 - margin_fraction))
    roi = np.zeros_like(valid)
    roi[y0:y1, x0:x1] = True
    # The table is the dominant quantized depth mode in this camera.
    quantized_mm = np.rint(depth[valid] * 1000.0).astype(np.int32)
    values, counts = np.unique(quantized_mm, return_counts=True)
    table_depth_m = float(values[np.argmax(counts)]) / 1000.0
    object_mask = valid & roi & (depth < table_depth_m - object_height_mm / 1000.0)
    filtered = np.where(valid & roi & (depth <= table_depth_m + 0.002), depth, 0.0).astype(np.float32)
    info = {
        "table_depth_m": table_depth_m,
        "table_depth_mm": int(round(table_depth_m * 1000.0)),
        "table_roi_xyxy": [x0, y0, x1, y1],
        "object_height_threshold_mm": float(object_height_mm),
        "object_pixels": int(object_mask.sum()),
    }
    return filtered, info


def _apply_depth_roi(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Black out pixels removed from the metric depth ROI."""
    isolated = np.asarray(rgb).copy()
    isolated[np.asarray(depth) <= 0] = 0
    return isolated


def _segment_sam3(rgb: np.ndarray, prompt: str) -> list[dict[str, Any]]:
    image_buffer = io.BytesIO()
    Image.fromarray(rgb).save(image_buffer, format="PNG")
    response = requests.post(
        f"{SAM3_URL}/segment",
        json={
            "image_base64": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            "text_prompt": prompt,
        },
        timeout=180,
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("results", []):
        mask_bytes = base64.b64decode(item["mask_base64"])
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(tuple(item["shape"])).astype(bool)
        results.append({**item, "mask": mask})
    return results


def _plan_grasps(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    segmap: np.ndarray,
    max_retries: int,
    filter_grasps: bool = True,
    local_regions: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "depth_base64": _numpy_base64(depth),
        "cam_K_base64": _numpy_base64(intrinsics),
        "segmap_base64": _numpy_base64(segmap),
        "segmap_id": 1,
        "local_regions": local_regions,
        "filter_grasps": filter_grasps,
        "skip_border_objects": False,
        "z_range": [0.2, 2.0],
        "forward_passes": 1,
        "max_retries": max_retries,
    }
    response = requests.post(f"{GRASPNET_URL}/plan", json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return (
        _decode_numpy(data["grasps_base64"]),
        _decode_numpy(data["scores_base64"]),
        _decode_numpy(data["contact_pts_base64"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="brown square nut",
        help="SAM3 text prompt used to build the GraspNet segment.",
    )
    parser.add_argument("--scene", choices=("nut", "cube"), default="nut")
    parser.add_argument("--output-root", type=Path, default=None)
    # Keep an extra retry slot: the service discards results when its retry
    # counter reaches max_retries, including a result from the final retry.
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--filter-grasps",
        action="store_true",
        help="Enable Contact-GraspNet filtering. By default all raw candidates are kept.",
    )
    parser.add_argument(
        "--global-regions",
        action="store_true",
        help="Run inference on the full scene instead of the SAM3 local region.",
    )
    args = parser.parse_args()

    global OUTPUT_ROOT
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root
    elif args.scene == "cube":
        OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "output_cube"
    if args.scene == "cube" and args.prompt == "brown square nut":
        args.prompt = "cube"

    output_dir = _next_output_dir()
    started: list[subprocess.Popen[Any]] = []
    try:
        started_service = _start_service(
            "capx.serving.launch_sam3_server", 8114, output_dir / "sam3.log"
        )
        if started_service is not None:
            started.append(started_service)
        started_service = _start_service(
            "capx.serving.launch_contact_graspnet_server", 8115, output_dir / "graspnet.log"
        )
        if started_service is not None:
            started.append(started_service)

        # Import after services are started so the script remains lightweight at startup.
        if args.scene == "cube":
            from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel
            env = FrankaRobosuiteCubesLowLevel(enable_render=True, viser_debug=False, max_steps=2000)
        else:
            from capx.envs.simulators.robosuite_nut_assembly import FrankaRobosuiteNutAssemblyVisual
            env = FrankaRobosuiteNutAssemblyVisual(enable_render=True, viser_debug=False, max_steps=2000)
        try:
            # Robosuite camera observations are not valid until the task has
            # been reset and its settling steps have completed.
            reset_result = env.reset()
            obs = env.get_observation()
            camera = obs["robot0_robotview"]
            rgb = np.asarray(camera["images"]["rgb"])[..., :3].astype(np.uint8)
            if rgb.size == 0 or int(rgb.max()) == 0:
                raise RuntimeError("Rendered RGB frame is all black; check MUJOCO_GL and environment reset")
            depth = np.asarray(camera["images"]["depth"])
            if depth.ndim == 3:
                depth = depth[..., 0]
            raw_depth = depth.astype(np.float32, copy=True)
            if args.scene == "nut":
                depth, depth_info = _isolate_table_objects(depth)
                # Keep the model input metric, but quantize it explicitly to
                # millimeter resolution for deterministic point-cloud input.
                depth = np.where(depth > 0, np.rint(depth * 1000.0) / 1000.0, 0.0).astype(np.float32)
                raw_rgb = rgb.copy()
                rgb = _apply_depth_roi(rgb, depth)
                Image.fromarray(raw_rgb).save(output_dir / "birdview_rgb_raw.png")
            else:
                depth_info = {"mode": "raw_metric_depth"}
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
            Image.fromarray(rgb).save(output_dir / "birdview_rgb.png")
            np.save(output_dir / "birdview_depth_raw.npy", raw_depth)
            np.save(output_dir / "birdview_depth.npy", depth)
            np.save(output_dir / "birdview_depth_mm.npy", np.rint(depth * 1000.0).astype(np.uint16))
            np.save(output_dir / "intrinsics.npy", intrinsics)

            sam_results = _segment_sam3(rgb, args.prompt)
            if not sam_results:
                raise RuntimeError(f"SAM3 returned no detections for prompt: {args.prompt!r}")
            selected = max(sam_results, key=lambda item: float(item.get("score", 0.0)))
            mask = np.asarray(selected["mask"], dtype=bool)
            if mask.shape != depth.shape:
                raise RuntimeError(f"SAM3 mask shape {mask.shape} != depth shape {depth.shape}")
            segmap = mask.astype(np.int32)
            Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / "sam3_mask.png")

            valid_depth = np.isfinite(depth) & (depth > 0)
            segment_depth = valid_depth & mask
            metadata = {
                "prompt": args.prompt,
                "scene": args.scene,
                "depth_processing": depth_info,
                "depth_units": {
                    "graspnet_input": "meters (float32)",
                    "saved_depth_mm": "millimeters (uint16)",
                    "saved_depth_raw": "meters (float32)",
                },
                "reset_info": reset_result[1] if isinstance(reset_result, tuple) and len(reset_result) > 1 else {},
                "sam_score": float(selected["score"]),
                "mask_pixels": int(mask.sum()),
                "valid_depth_pixels": int(valid_depth.sum()),
                "segment_valid_depth_pixels": int(segment_depth.sum()),
                "max_retries": int(args.max_retries),
                "filter_grasps": args.filter_grasps,
                "local_regions": not args.global_regions,
            }
            if segment_depth.any():
                metadata["segment_depth_min"] = float(depth[segment_depth].min())
                metadata["segment_depth_max"] = float(depth[segment_depth].max())

            grasps, scores, contacts = _plan_grasps(
                depth,
                intrinsics,
                segmap,
                args.max_retries,
                filter_grasps=args.filter_grasps,
                local_regions=not args.global_regions,
            )
            grasps = np.asarray(grasps)
            scores = np.asarray(scores)
            contacts = np.asarray(contacts)
            metadata.update(
                {
                    "grasps_shape": list(grasps.shape),
                    "scores_shape": list(scores.shape),
                    "contacts_shape": list(contacts.shape),
                    "candidate_count": int(len(grasps)),
                }
            )
            _save_top_grasps(output_dir, grasps, scores, contacts)
            _save_visualizations(output_dir, rgb, depth, mask, intrinsics, grasps, scores, contacts)
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if len(grasps) == 0:
                print("GraspNet returned no grasp candidates; see metadata.json and graspnet.log.", file=sys.stderr)
            else:
                print(f"GraspNet returned {len(grasps)} candidates; saving all candidates and top 20.")
            print(f"Saved top-20 grasp results to {output_dir}")
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    finally:
        for process in started:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
