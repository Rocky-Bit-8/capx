"""Backward-compatible Nut Assembly Molmo API wrapper.

Molmo-backed segmentation and grasping belong to
``FrankaControlNutAssemblyVisualApi``.  This class is retained only for older
configurations that expose an explicit ``point_prompt_molmo`` helper.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from capx.integrations.franka.nut_assembly_visual import (
    FrankaControlNutAssemblyVisualApi,
)
from capx.utils.camera_utils import obs_get_rgb


class FrankaControlNutAssemblyMolmoApi(FrankaControlNutAssemblyVisualApi):
    """Original Nut Assembly pipeline plus Molmo-guided handle point selection."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)

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

__all__ = ["FrankaControlNutAssemblyMolmoApi"]
