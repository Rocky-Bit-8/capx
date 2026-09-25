"""Nut Assembly visual API with an optional Molmo semantic grasp point."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from capx.integrations.franka.nut_assembly_visual import (
    FrankaControlNutAssemblyVisualApi,
)
from capx.integrations.vision.molmo import init_molmo
from capx.utils.camera_utils import obs_get_rgb


class FrankaControlNutAssemblyMolmoApi(FrankaControlNutAssemblyVisualApi):
    """Original Nut Assembly pipeline plus Molmo-guided handle point selection."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.molmo_point_fn = init_molmo()

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["point_prompt_molmo"] = self.point_prompt_molmo
        return fns

    def point_prompt_molmo(self, text_prompt: str) -> dict[str, tuple[int | None, int | None]]:
        """Return Molmo's pixel point for the current robot-view image."""
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        obs = self._env.get_observation()
        rgb_images = obs_get_rgb(obs)
        if not rgb_images:
            raise RuntimeError("No RGB image in Nut Assembly observation")
        image = Image.fromarray(next(iter(rgb_images.values()))).convert("RGB")
        result = self.molmo_point_fn(image, objects=[text_prompt])
        self._log_step_update(text=f"Molmo point for '{text_prompt}': {result}")
        return result

    def _segment_object_from_language(self, image: Image.Image, object_name: str):
        """Keep the original mask/geometry selection and improve handle pixels with Molmo."""
        mask, point_xy, scores = super()._segment_object_from_language(image, object_name)
        if mask is None or "handle" not in object_name.lower():
            return mask, point_xy, scores

        try:
            result = self.molmo_point_fn(image, objects=[object_name])
            candidate = result.get(object_name)
            if candidate is None or candidate[0] is None or candidate[1] is None:
                return mask, point_xy, scores
            x, y = (int(round(candidate[0])), int(round(candidate[1])))
            height, width = mask.shape
            if not (0 <= x < width and 0 <= y < height):
                return mask, point_xy, scores

            # Molmo can point at an edge pixel. Use the nearest mask pixel so
            # robust depth sampling remains on the segmented nut surface.
            if not mask[y, x]:
                pixels = np.argwhere(mask)
                nearest = pixels[np.argmin((pixels[:, 1] - x) ** 2 + (pixels[:, 0] - y) ** 2)]
                y, x = int(nearest[0]), int(nearest[1])
            self._log_step_update(text=f"Using Molmo-guided handle point [{x}, {y}].")
            return mask, (x, y), scores
        except Exception as exc:  # noqa: BLE001
            print(f"Molmo handle point unavailable; keeping Nut geometry point: {exc}")
            return mask, point_xy, scores

__all__ = ["FrankaControlNutAssemblyMolmoApi"]
