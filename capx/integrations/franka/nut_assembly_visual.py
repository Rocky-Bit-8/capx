import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.motion.pyroki import init_pyroki_remote
from capx.integrations.vision.sam3 import init_sam3
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_to_pointcloud,
    depth_to_rgb,
)


# ------------------------------- Control API ------------------------------
class FrankaControlNutAssemblyVisualApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - goto_home_joint_position() -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)
    _SAM3_PROMPT_FALLBACKS = {
        "extruded handle of the brown square nut": (
            "brown square nut handle",
            "handle",
        ),
        "white hollow center of the brown square nut": (
            "white center hole",
            "center hole",
            "brown square nut",
        ),
    }

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
        # self._TCP_OFFSET = _TCP_OFFSET
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # Nut Assembly uses SAM3's text segmentation directly.  This avoids a
        # separate Molmo GPU service merely to obtain a point prompt.
        self.sam3_segment_fn = init_sam3()
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki_remote()
        self.cfg: np.ndarray | None = None
        self.camera_name = "robot0_robotview"

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "goto_home_joint_position": self.goto_home_joint_position,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.
        Raises:
            RuntimeError: If SAM3 cannot provide a valid point for the
                requested object.  Continuing with a ``None`` pose makes the
                generated task code fail later with an unrelated NumPy error.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """

        # deproject the point to the world coordinate, use oriented bounding box to get the rotation, and return the position of the point and the rotation of the oriented bounding box
        self._log_step(
            "Nut Pose Query",
            f"Capturing camera observation for '{object_name}' ...",
        )
        start_time = time.time()
        obs = self._env.get_observation()
        print(f"get observation in {time.time() - start_time} seconds")

        fixed_rotation = None
        if all(i in object_name for i in ["square", "block"]):
            fixed_rotation = obs["nut_poses"]["square_peg"][3:]

        rgb_imgs = obs_get_rgb(obs)
        assert len(rgb_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rgb_imgs.values())[0]
        pil_rgb = Image.fromarray(rgb).convert("RGB")
        self._log_step_update(images=rgb)

        mask_bool, point_px, sam_scores = self._segment_object_from_language(pil_rgb, object_name)
        if point_px is None:
            self._log_step_update(text="SAM3 did not return a usable object mask.")
            raise RuntimeError(
                f"SAM3 could not segment '{object_name}' in the Nut Assembly camera image, "
                "including its Nut-specific fallback prompts."
            )

        depth = obs[self.camera_name]["images"]["depth"][:, :, 0]
        if mask_bool.shape != depth.shape:
            raise ValueError(
                f"SAM3 mask shape {mask_bool.shape} does not match depth shape {depth.shape}"
            )

        if self._env.viser_debug:
            depth_img = depth_to_rgb(depth)
            Image.fromarray(depth_img).save("depth_image.jpg")

            mask_overlay = rgb.copy()
            mask_overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
            overlay_img = Image.fromarray(mask_overlay)
            draw = ImageDraw.Draw(overlay_img)
            radius = 6
            draw.ellipse(
                [
                    point_px[0] - radius,
                    point_px[1] - radius,
                    point_px[0] + radius,
                    point_px[1] + radius,
                ],
                outline=(255, 255, 0),
                width=2,
            )
            overlay_path = pathlib.Path(f"{object_name.replace(' ', '_')}_sam3_overlay.jpg")
            overlay_img.save(overlay_path)
            print(f"SAM3 mask scores for {object_name}: {sam_scores}")

            mask_binary_path = pathlib.Path(f"{object_name.replace(' ', '_')}_sam3_mask.png")
            Image.fromarray(mask_bool.astype(np.uint8) * 255).save(mask_binary_path)

        self._log_step("Nut 3D Pose", f"Deprojecting SAM3 mask for '{object_name}' ...")

        # A SAM mask can include a few table pixels at its edge.  The former
        # minimum-depth rule made one such pixel move the 3D target by cm.
        # Estimate depth from valid mask pixels nearest the selected semantic
        # point instead, which also works when the point is the center hole.
        median_depth = self._robust_mask_depth(depth, mask_bool, point_px)
        camera_point = deproject_pixel_to_camera(
            point_px, median_depth, obs[self.camera_name]["intrinsics"]
        )
        camera_tf = vtf.SE3.from_translation(camera_point)
        world_point = (
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(wxyz=obs[self.camera_name]["pose"][3:]),
                translation=obs[self.camera_name]["pose"][:3],
            )
            @ camera_tf
        )

        valid_mask = mask_bool & np.isfinite(depth) & (depth > 0.015) & (depth < 20.0)
        points = depth_to_pointcloud(
            depth, obs[self.camera_name]["intrinsics"], filter_invalid=False
        )[valid_mask.ravel()]
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)

        obb = o3d_points.get_oriented_bounding_box()

        # Exposing these to the low level environment for viser
        if self._env.viser_debug:
            self._env.cube_center = obb.center
            self._env.cube_rot = obb.R

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs[self.camera_name]["pose"][3:]),
            translation=obs[self.camera_name]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        obb_tf_world = cam_extr_tf @ obb_tf

        # if z axis isn't pointing down, flip the z axis
        R = obb_tf_world.rotation().as_matrix()
        z_axis_world = R[:, 2]
        if z_axis_world[2] > 0:
            obb_tf_world = obb_tf_world @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_matrix(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            )

        if fixed_rotation is None:
            fixed_rotation = obb_tf_world.wxyz_xyz[:4]

        if self._env.viser_server is not None:
            self._env.viser_server.scene.add_frame(
                f"{self.camera_name}/{object_name}_frame",
                position=camera_tf.wxyz_xyz[-3:],
                wxyz=camera_tf.wxyz_xyz[:4],
                axes_length=0.05,
                axes_radius=0.005,
            )

            self._env.viser_server.scene.add_frame(
                f"sam3_point_{object_name}",
                position=world_point.wxyz_xyz[-3:],
                wxyz=fixed_rotation,
                axes_length=0.05,
                axes_radius=0.005,
            )
        print(f"get_object_pose in {time.time() - start_time} seconds")
        self._log_step_update(
            text=(
                "Pose estimated at "
                f"[{world_point.wxyz_xyz[-3]:.3f}, {world_point.wxyz_xyz[-2]:.3f}, "
                f"{world_point.wxyz_xyz[-1]:.3f}] m."
            )
        )
        return world_point.wxyz_xyz[-3:], fixed_rotation

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        # Reuse the SAM3-derived object pose as the grasp pose.
        return self.get_object_pose(object_name)

    # def goto_pose(
    #     self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    # ) -> None:
    #     """Solve IK for requested pose (with optional approach offset) and move joints smoothly.

    #     Args:
    #         position: (3,) XYZ in meters.
    #         quaternion_wxyz: (4,) WXYZ unit quaternion.
    #         z_approach: Optional approach distance along tool Z (meters). When non-zero the
    #             motion first reaches position + z_approach in tool Z before descending.
    #     """

    #     pos = np.asarray(position, dtype=np.float64).reshape(3)
    #     quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    #     quat_xyzw = np.array(
    #         [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    #     )
    #     rot = SciRotation.from_quat(quat_xyzw)
    #     offset_pos = pos + rot.apply(self._TCP_OFFSET)

    #     targets: list[tuple[np.ndarray, np.ndarray]] = []
    #     if z_approach != 0.0:
    #         z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
    #         targets.append((z_offset_pos, quat_wxyz))
    #         if self._env.viser_debug:
    #             self._env.mjcf_ee_frame_handle.position = z_offset_pos
    #             self._env.mjcf_ee_frame_handle.wxyz = quat_wxyz

    #             z_offset_gripper_pos = pos + rot.apply(np.array([0, 0, -z_approach]))
    #             self._env.mjcf_gripper_frame_handle.position = z_offset_gripper_pos
    #             self._env.mjcf_gripper_frame_handle.wxyz = quat_wxyz

    #     targets.append((offset_pos, quat_wxyz))
    #     if self._env.viser_debug:
    #         self._env.mjcf_ee_frame_handle.position = offset_pos
    #         self._env.mjcf_ee_frame_handle.wxyz = quat_wxyz

    #         self._env.mjcf_gripper_frame_handle.position = pos
    #         self._env.mjcf_gripper_frame_handle.wxyz = quat_wxyz

    #     seed = self.cfg
    #     for target_position, target_quat in targets:
    #         ik_solution = self._solve_ik_with_seed(target_position, target_quat, seed)
    #         self.cfg = ik_solution
    #         joints = self._extract_arm_joints(ik_solution)
    #         self._env.move_to_joints_blocking(joints)
    #         seed = ik_solution

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        if position is None or quaternion_wxyz is None:
            raise RuntimeError(
                "Cannot move to a perception result without a valid pose. "
                "Ensure the Nut Assembly SAM3 service on port 8114 is running, then query the object again."
            )

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        self._log_step(
            "Nut IK Motion",
            f"Solving IK for [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m ...",
        )
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (
            z_approach != 0.0
        ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)
            self._log_step_update(text="Approach pose reached; solving final target ...")

        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self._log_step_update(text="Target pose reached.")

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._log_step("open_gripper", "Opening gripper ...")
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()
        self._log_step_update(text="Gripper opened.")

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._log_step("close_gripper", "Closing gripper ...")
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()
        self._log_step_update(text="Gripper closed.")

    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        self._log_step("goto_home_joint_position", "Returning arm to home joint configuration ...")
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self.cfg = None
        self._log_step_update(text="Home configuration reached.")

    def _solve_ik_with_seed(
        self, target_position: np.ndarray, target_quat: np.ndarray, seed: np.ndarray | None
    ) -> np.ndarray:
        """Solve IK using PyRoKI with an optional previous solution as the initial guess."""
        solution = self._pks.solve_ik(
            robot=self._robot,
            target_link_name=self._target_link_name,
            target_position=target_position,
            target_wxyz=target_quat,
            initial_cfg=seed,
        )
        return np.asarray(solution, dtype=np.float64)

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray, tuple[int, int], list[float]]:
        """Return a Nut-specific SAM3 mask and a semantically useful pixel."""
        prompts = (object_name, *self._SAM3_PROMPT_FALLBACKS.get(object_name.lower(), ()))
        self._log_step("SAM3 Segmentation", f"Segmenting '{object_name}' with SAM3 ...", images=image)
        results = []
        selected_prompt = object_name
        for prompt in prompts:
            results = self.sam3_segment_fn(image, prompt)
            if results:
                selected_prompt = prompt
                break
        if not results:
            self._log_step_update(text="No masks returned for the requested object.")
            return None, None, None
        if selected_prompt != object_name:
            print(f"SAM3 prompt fallback: '{object_name}' -> '{selected_prompt}'")

        is_nut_part = "square nut" in object_name.lower() or "hollow center" in object_name.lower()
        nut_mask, hole_mask = self._select_square_nut_mask(results) if is_nut_part else (None, None)
        if is_nut_part and nut_mask is None:
            # Part-level language grounding frequently returns the entire nut,
            # but not consistently.  Ask for the stable outer object and use
            # its ring geometry rather than treating a hole as an object.
            outer_results = self.sam3_segment_fn(image, "brown square nut")
            nut_mask, hole_mask = self._select_square_nut_mask(outer_results)
            if nut_mask is not None:
                results = outer_results
                selected_prompt = "brown square nut"
                print(f"SAM3 geometric fallback: '{object_name}' -> '{selected_prompt}'")

        if nut_mask is not None:
            mask_bool = nut_mask
            if "handle" in object_name.lower():
                point_xy = self._handle_grasp_pixel(mask_bool, hole_mask)
            elif "hollow center" in object_name.lower() or "center hole" in object_name.lower():
                point_xy = self._mask_center_pixel(hole_mask)
            else:
                point_xy = self._mask_center_pixel(mask_bool)
        else:
            mask_bool = self._as_mask(results[0]["mask"])
            if mask_bool is None:
                return None, None, None
            point_xy = self._mask_center_pixel(mask_bool)

        scores = [float(result["score"]) for result in results]
        if self._webui_enabled:
            overlay = np.asarray(image).copy()
            overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
            self._log_step_update(
                text=f"Returned {len(results)} mask(s), best score: {scores[0]:.3f}.",
                images=overlay,
            )
        else:
            self._log_step_update(text=f"Returned {len(results)} mask(s), best score: {scores[0]:.3f}.")
        return mask_bool, point_xy, scores

    @staticmethod
    def _as_mask(mask: Any) -> np.ndarray | None:
        mask_bool = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_bool.ndim != 2 or not np.any(mask_bool):
            return None
        return mask_bool

    @classmethod
    def _select_square_nut_mask(
        cls, results: list[dict[str, Any]]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Find the compact ring-shaped square-nut proposal among SAM3 masks."""
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for result in results:
            mask = cls._as_mask(result.get("mask"))
            if mask is None:
                continue
            ys, xs = np.nonzero(mask)
            height, width = ys.ptp() + 1, xs.ptp() + 1
            if mask.sum() < 400 or min(height, width) < 25 or max(height, width) > 130:
                continue
            filled = ndimage.binary_fill_holes(mask)
            holes, count = ndimage.label(filled & ~mask)
            if count == 0:
                continue
            sizes = np.bincount(holes.ravel())
            sizes[0] = 0
            hole = holes == int(np.argmax(sizes))
            hole_area = int(hole.sum())
            if hole_area < 60 or hole_area > int(mask.sum() * 0.7):
                continue
            aspect = max(height, width) / min(height, width)
            # The score is only a tie-breaker.  A compact ring with a sizeable
            # enclosed hole is the task-specific identity cue for this nut.
            geometry_score = 3.0 - abs(aspect - 1.0) + min(hole_area / mask.sum(), 0.35)
            score = geometry_score + 0.05 * float(result.get("score", 0.0))
            if best is None or score > best[0]:
                best = (score, mask, hole)
        return (best[1], best[2]) if best is not None else (None, None)

    @staticmethod
    def _mask_center_pixel(mask: np.ndarray) -> tuple[int, int]:
        pixels = np.argwhere(mask)
        if pixels.size == 0:
            raise ValueError("Cannot choose a pixel from an empty mask")
        center_yx = np.median(pixels, axis=0)
        point_yx = pixels[np.argmin(np.sum((pixels - center_yx) ** 2, axis=1))]
        return int(point_yx[1]), int(point_yx[0])

    @classmethod
    def _handle_grasp_pixel(cls, nut_mask: np.ndarray, hole_mask: np.ndarray) -> tuple[int, int]:
        """Place the grasp point inside the protruding handle, not nut center."""
        hole_pixels = np.argwhere(hole_mask)
        if hole_pixels.size == 0:
            return cls._mask_center_pixel(nut_mask)
        hole_center = np.median(hole_pixels, axis=0)
        nut_pixels = np.argwhere(nut_mask)
        offsets = nut_pixels - hole_center
        distances = np.linalg.norm(offsets, axis=1)
        direction = offsets[np.argmax(distances)]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return cls._mask_center_pixel(nut_mask)
        direction /= direction_norm
        projections = offsets @ direction
        # 72% lies inside the handle rather than on its fragile outer edge.
        target_projection = 0.72 * float(np.max(projections))
        lateral = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
        cost = (projections - target_projection) ** 2 + 2.0 * lateral**2
        point_yx = nut_pixels[np.argmin(cost)]
        return int(point_yx[1]), int(point_yx[0])

    @staticmethod
    def _robust_mask_depth(
        depth: np.ndarray, mask: np.ndarray, point_xy: tuple[int, int]) -> float:
        valid = mask & np.isfinite(depth) & (depth > 0.015) & (depth < 20.0)
        pixels = np.argwhere(valid)
        if pixels.size == 0:
            raise RuntimeError("SAM3 mask contains no valid depth pixels")
        point_yx = np.array([point_xy[1], point_xy[0]])
        distances = np.sum((pixels - point_yx) ** 2, axis=1)
        nearby = pixels[np.argsort(distances)[: min(96, len(pixels))]]
        values = depth[nearby[:, 0], nearby[:, 1]]
        return float(np.median(values))

    @staticmethod
    def _extract_arm_joints(cfg: np.ndarray) -> np.ndarray:
        """PyRoKI returns actuated joints including gripper; strip to Panda arm joints."""
        return np.asarray(cfg[:-1], dtype=np.float64).reshape(7)


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), lab, fill=(255, 0, 0))
    if scores is not None:
        for b, score in zip(boxes, scores, strict=False):
            x1, y1, x2, y2 = b
            draw.text((x1 + 100, max(0, y1 - 12)), f"{score:.2f}", fill=(255, 0, 0))
    return img
