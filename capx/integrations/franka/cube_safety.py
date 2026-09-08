"""Cube-task control API with an automatic vertical clearance maneuver."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.franka.control import FrankaControlApi


class FrankaCubeSafetyApi(FrankaControlApi):
    """Franka control API for cube lifting, stacking, and restacking tasks.

    Every Cartesian command first raises the end effector by a small vertical
    distance. This keeps lateral or joint-space motion away from cubes on the
    table, including before returning to the home joint configuration.
    """

    VERTICAL_CLEARANCE_METERS = 0.08

    def __init__(self, env: BaseEnv, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)

    def _move_to_clearance(self) -> None:
        """Raise the current TCP by 8cm along the base-frame vertical axis."""
        obs = self._env.get_observation()
        cartesian = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64)
        current_pos = cartesian[:3].copy()
        current_quat = cartesian[3:7].copy()
        clearance_pos = current_pos.copy()
        clearance_pos[2] += self.VERTICAL_CLEARANCE_METERS
        self._log_step(
            "cube_clearance",
            "Raising end effector vertically by "
            f"{self.VERTICAL_CLEARANCE_METERS:.3f} m …",
        )
        super().goto_pose(clearance_pos, current_quat)
        self._log_step_update(text="Safe clearance reached.")

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Move to a pose after automatically clearing the cube workspace."""
        self._move_to_clearance()
        super().goto_pose(position, quaternion_wxyz, z_approach=z_approach)

    def home_pose(self) -> None:
        """Clear the cube workspace before moving to the home joint pose."""
        self._move_to_clearance()
        super().home_pose()


__all__ = ["FrankaCubeSafetyApi"]
