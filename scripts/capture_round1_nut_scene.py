"""Capture one unmodified MuJoCo Nut Assembly camera observation for diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from capx.envs.simulators.robosuite_nut_assembly import FrankaRobosuiteNutAssemblyVisual


OUTPUT_DIR = Path("sam_test/round1_scene")
CAMERA_KEY = "robot0_robotview"
SEED = 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = FrankaRobosuiteNutAssemblyVisual(seed=SEED, enable_render=True)
    try:
        observation, _ = env.reset(seed=SEED)
        camera = observation[CAMERA_KEY]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float32).squeeze(-1)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)

        Image.fromarray(rgb).save(OUTPUT_DIR / "rgb.png")
        np.save(OUTPUT_DIR / "depth_m.npy", depth)
        np.save(OUTPUT_DIR / "intrinsics.npy", intrinsics)
        (OUTPUT_DIR / "metadata.json").write_text(
            json.dumps(
                {
                    "source": "MuJoCo NutAssemblySquare raw observation",
                    "seed": SEED,
                    "camera_key": CAMERA_KEY,
                    "rgb_shape": list(rgb.shape),
                    "depth_shape": list(depth.shape),
                    "depth_unit": "meters",
                    "depth_min_m": float(np.nanmin(depth)),
                    "depth_max_m": float(np.nanmax(depth)),
                    "intrinsics": intrinsics.tolist(),
                },
                indent=2,
            )
        )
        print(f"saved raw RGB: {OUTPUT_DIR / 'rgb.png'}")
        print(f"saved metric depth: {OUTPUT_DIR / 'depth_m.npy'}")
        print(f"depth range: {depth.min():.4f}..{depth.max():.4f} m")
    finally:
        env.robosuite_env.close()


if __name__ == "__main__":
    main()
