"""Molmo-guided Nut Assembly API with a constrained top-down grasp pose."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.integrations.franka.nut_assembly_visual import (
    FrankaControlNutAssemblyVisualApi,
)
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import deproject_pixel_to_camera


class FrankaControlNutAssemblyGuideApi(FrankaControlNutAssemblyVisualApi):
    """Nut Assembly control using Molmo XY and a world-vertical tool Z axis.

    Motion, local IK, gripper control, and verification are inherited from the
    Visual API. Only object/grasp pose estimation is replaced: the Molmo point
    selects the target pixel, while the target orientation is fixed top-down.
    Returned poses are in the robot base frame expected by local IK.
    """

    # A 180-degree rotation about world Y maps tool-local +Z to world -Z.
    _TOP_DOWN_WORLD_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    def functions(self) -> dict[str, Any]:
        return super().functions()

    def _world_pose_to_base(
        self, position_world: np.ndarray, quat_world_wxyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the desired world pose to the input frame of local IK."""
        base_pose = np.asarray(self._env.base_link_wxyz_xyz, dtype=np.float64).reshape(7)
        base_rot = SciRotation.from_quat(
            [base_pose[1], base_pose[2], base_pose[3], base_pose[0]]
        )
        world_rot = SciRotation.from_quat(
            [
                quat_world_wxyz[1],
                quat_world_wxyz[2],
                quat_world_wxyz[3],
                quat_world_wxyz[0],
            ]
        )
        position_base = base_rot.inv().apply(
            np.asarray(position_world, dtype=np.float64).reshape(3) - base_pose[4:]
        )
        quat_base_xyzw = (base_rot.inv() * world_rot).as_quat()
        quat_base_wxyz = np.array(
            [quat_base_xyzw[3], *quat_base_xyzw[:3]], dtype=np.float64
        )
        return position_base, quat_base_wxyz

    @staticmethod
    def _depth_at_molmo_point(
        depth: np.ndarray, point_xy: tuple[int, int], mask: np.ndarray
    ) -> float:
        """Read depth at the Molmo pixel, with a mask-local invalid-depth fallback."""
        x = int(np.clip(point_xy[0], 0, depth.shape[1] - 1))
        y = int(np.clip(point_xy[1], 0, depth.shape[0] - 1))
        value = float(depth[y, x])
        if np.isfinite(value) and value > 0.0:
            return value

        valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            raise ValueError("No valid depth at the Molmo point or inside its SAM mask")
        nearest = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        return float(depth[ys[nearest], xs[nearest]])

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return Molmo-point XYZ and a world-top-down orientation in base frame."""
        obs = self._env.get_observation()
        rgb_images = obs_get_rgb(obs)
        if not rgb_images:
            raise ValueError("No RGB image in observation")
        rgb = list(rgb_images.values())[0]
        image = Image.fromarray(rgb).convert("RGB")

        mask, point_xy, _ = self._segment_object_from_language(image, object_name)
        if point_xy is None or mask is None:
            return None, None

        camera = obs[self.camera_name]
        depth = np.asarray(camera["images"]["depth"][:, :, 0], dtype=np.float64)
        if mask.shape != depth.shape:
            raise ValueError(
                f"SAM3 mask shape {mask.shape} does not match depth shape {depth.shape}"
            )
        point_depth = self._depth_at_molmo_point(depth, point_xy, mask)
        camera_point = deproject_pixel_to_camera(
            point_xy, point_depth, camera["intrinsics"]
        )
        camera_to_world = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=camera["pose"][3:]),
            translation=camera["pose"][:3],
        )
        world_point = (
            camera_to_world @ vtf.SE3.from_translation(camera_point)
        ).wxyz_xyz[-3:]
        position_base, quat_base = self._world_pose_to_base(
            world_point, self._TOP_DOWN_WORLD_QUAT_WXYZ
        )

        safe_name = object_name.replace(" ", "_")
        self._save_verify_image(
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).convert("RGB"),
            f"sam3_mask_{safe_name}",
        )
        overlay = np.asarray(image).copy()
        overlay[mask] = (
            0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)
        ImageDraw.Draw(overlay_image).ellipse(
            [point_xy[0] - 6, point_xy[1] - 6, point_xy[0] + 6, point_xy[1] + 6],
            outline=(255, 255, 0),
            width=2,
        )
        self._save_verify_image(overlay_image, f"molmo_point_{safe_name}")

        if self._env.viser_server is not None:
            self._env.viser_server.scene.add_frame(
                f"guide/{safe_name}_top_down",
                position=world_point,
                wxyz=self._TOP_DOWN_WORLD_QUAT_WXYZ,
                axes_length=0.05,
                axes_radius=0.005,
            )
        print(f"Guide pose world XYZ for {object_name}: {world_point}")
        print(f"Guide pose base WXYZ for {object_name}: {quat_base}")
        return position_base, quat_base

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Use the same Molmo-guided top-down pose for grasping."""
        return self.get_object_pose(object_name)


__all__ = ["FrankaControlNutAssemblyGuideApi"]
