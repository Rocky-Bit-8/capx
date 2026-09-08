from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.tasks.franka.franka_pick_place import FrankaPickPlaceCodeEnv
from capx.integrations.base_api import register_api
from capx.integrations.franka.control import FrankaControlApi


T = TypeVar("T")


class FrankaExpertApi(FrankaControlApi):
    """Franka control API with recovery helpers for SAM3 object lookup."""

    _SAM3_RETRY_COUNT = 3
    _SAM3_MIN_SCORE = 0.4
    _SAM3_NO_DETECTION_MESSAGE = "No SAM3 detections"
    _SAM3_LOW_SCORE_MESSAGE = "SAM3 mask score below safe threshold"
    _DEFAULT_MOVE_AWAY_DISTANCE = 0.15

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._safe_lookup_active = False

    def _segment_sam3_with_aliases(
        self, rgb: np.ndarray, object_name: str
    ) -> tuple[list[dict[str, Any]], str]:
        results, prompt_used = super()._segment_sam3_with_aliases(rgb, object_name)
        if self._safe_lookup_active and self.use_sam3 and results:
            best_score = max(float(result.get("score", 0.0)) for result in results)
            if best_score < self._SAM3_MIN_SCORE:
                raise ValueError(
                    f"{self._SAM3_LOW_SCORE_MESSAGE}: "
                    f"{best_score:.3f} < {self._SAM3_MIN_SCORE:.3f}"
                )
        return results, prompt_used

    def functions(self) -> dict[str, Any]:
        functions = super().functions()
        functions.update(
            {
                "get_object_pose_safe": self.get_object_pose_safe,
                "sample_grasp_pose_safe": self.sample_grasp_pose_safe,
                "sample_grasp_pose_overlook": self.sample_grasp_pose_overlook,
                "sample_grasp_pose_with_graspnet": self.sample_grasp_pose_with_graspnet,
                "sample_grasp_pose_multi_angle": self.sample_grasp_pose_multi_angle,
                "goto_pose_overlook": self.goto_pose_overlook,
                "move_away": self.move_away,
            }
        )
        return functions

    def move_away(self, distance: float = _DEFAULT_MOVE_AWAY_DISTANCE) -> None:
        """Move the end effector upward while preserving its pose and gripper state.

        Args:
            distance: Upward displacement in meters. Defaults to 0.15 m.
        """
        distance = float(distance)
        if distance <= 0.0:
            raise ValueError("distance must be positive")

        observation = self._env.get_observation()
        cartesian_pose = observation.get("robot_cartesian_pos")
        if cartesian_pose is None:
            raise RuntimeError(
                "The environment observation does not provide robot_cartesian_pos"
            )

        cartesian_pose = np.asarray(cartesian_pose, dtype=np.float64).reshape(-1)
        if cartesian_pose.size < 7:
            raise RuntimeError(
                "robot_cartesian_pos must contain position and quaternion"
            )

        position = cartesian_pose[:3].copy()
        quaternion_wxyz = cartesian_pose[3:7].copy()
        position[2] += distance
        self._log_step(
            "move_away",
            f"Moving the end effector upward by {distance:.3f} m ...",
        )
        self.goto_pose(position, quaternion_wxyz)

    def _run_object_lookup_safe(
        self, object_name: str, lookup: Callable[[], T]
    ) -> T:
        """Retry a failed SAM3 lookup, move upward, then try once more."""
        self._safe_lookup_active = True
        try:
            for retry_index in range(self._SAM3_RETRY_COUNT + 1):
                try:
                    return lookup()
                except ValueError as error:
                    error_text = str(error)
                    if (
                        self._SAM3_NO_DETECTION_MESSAGE not in error_text
                        and self._SAM3_LOW_SCORE_MESSAGE not in error_text
                    ):
                        raise
                    if retry_index == self._SAM3_RETRY_COUNT:
                        break

                    self._log_step(
                        "safe_object_lookup",
                        f"SAM3 did not find a sufficiently confident '{object_name}'. "
                        f"Retrying ({retry_index + 1}/{self._SAM3_RETRY_COUNT}) ...",
                    )

            self._log_step(
                "safe_object_lookup",
                f"SAM3 still cannot find a sufficiently confident '{object_name}'. "
                "Moving the arm upward before the final lookup ...",
            )
            self.move_away()
            return lookup()
        finally:
            self._safe_lookup_active = False

    def get_object_pose_safe(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get an object pose using a safer perception mode.

        This mode reduces failures to obtain a reliable object position when
        the robot arm or gripper occludes the object.
        """
        return self._run_object_lookup_safe(
            object_name,
            lambda: super(FrankaExpertApi, self).get_object_pose(
                object_name, return_bbox_extent=return_bbox_extent
            ),
        )

    def sample_grasp_pose_safe(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose using a safer perception mode.

        This mode reduces failures to obtain a reliable object position when
        the robot arm or gripper occludes the object.
        """
        return self._run_object_lookup_safe(
            object_name,
            lambda: super(FrankaExpertApi, self).sample_grasp_pose(object_name),
        )

    def sample_grasp_pose_overlook(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a recommended, stable top-down grasp for simple scenes.

        The grasp position is the perceived geometric center. The gripper is
        kept vertical and its in-plane orientation is selected to maximize the
        tangent contact length along an object edge. This is recommended for
        simple scenes with a clear view of the target object.
        """
        try:
            center, object_quaternion_wxyz, bbox_extent = self.get_object_pose_safe(
                object_name, return_bbox_extent=True
            )
        except (RuntimeError, ValueError) as error:
            # Perception itself failed, so the parent remains a last-resort plan.
            message = (
                f"Overlook grasp for '{object_name}' fell back to the parent "
                f"grasp plan because centered perception failed: {error}"
            )
            print(message)
            self._log_step("sample_grasp_pose_overlook", message)
            return super().sample_grasp_pose(object_name)

        position = np.asarray(center, dtype=np.float64).reshape(3).copy()
        # This established control-frame orientation is vertical and downward.
        downward_rotation = SciRotation.from_quat([0.0, 1.0, 0.0, 0.0])
        gripper_yaw = 0.0
        try:
            if bbox_extent is None:
                raise ValueError("no bounding-box extent was returned")
            extent = np.asarray(bbox_extent, dtype=np.float64).reshape(3)
            object_quaternion_wxyz = np.asarray(
                object_quaternion_wxyz, dtype=np.float64
            ).reshape(4)
            if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
                raise ValueError("bounding-box extent is invalid")
            if not np.all(np.isfinite(object_quaternion_wxyz)):
                raise ValueError("bounding-box quaternion is invalid")
            object_rotation = SciRotation.from_quat(
                object_quaternion_wxyz[[1, 2, 3, 0]]
            )
            object_axes = object_rotation.as_matrix()

            # A finger can make broad, face-to-face contact when it is tangent
            # to an OBB edge.  The projected edge length is therefore its
            # contact-span score; the jaw closes perpendicular to that edge.
            projected_axes = object_axes[:2, :]
            tangent_lengths = extent * np.linalg.norm(projected_axes, axis=0)
            best_axis = int(np.argmax(tangent_lengths))
            best_length = float(tangent_lengths[best_axis])
            if not np.isfinite(best_length) or best_length <= 1e-6:
                raise ValueError("no usable horizontal bounding-box edge")

            # A square has two equally good side-contact directions. Choose one
            # of its near-maximal OBB edges rather than treating symmetry as an
            # invalid grasp. This keeps the contact-length objective valid for
            # cubes while avoiding a deterministic bias toward either edge.
            tie_tolerance = max(0.002, best_length * 0.05)
            candidate_axes = np.flatnonzero(
                tangent_lengths >= best_length - tie_tolerance
            )
            if candidate_axes.size == 0:
                candidate_axes = np.array([best_axis])
            selected_axis = int(np.random.choice(candidate_axes))
            tangent_axis = projected_axes[:, selected_axis]
            tangent_yaw = float(np.arctan2(tangent_axis[1], tangent_axis[0]))
            # In the downward Franka hand frame, local X is the horizontal
            # tangent of each finger pad and local Y is the opening/closing
            # axis. Align X with the selected object edge so that Y closes
            # across it and the pads contact along its maximal length.
            gripper_yaw = tangent_yaw

            grasp_rotation = SciRotation.from_euler("z", gripper_yaw) * downward_rotation
            grasp_quaternion_xyzw = grasp_rotation.as_quat()
            grasp_quaternion_wxyz = grasp_quaternion_xyzw[[3, 0, 1, 2]]
        except (ValueError, np.linalg.LinAlgError) as error:
            message = (
                f"Overlook grasp for '{object_name}' used the fixed downward "
                f"orientation because OBB contact optimization failed: {error}"
            )
            print(message)
            self._log_step("sample_grasp_pose_overlook", message)
            grasp_quaternion_wxyz = downward_rotation.as_quat()[[3, 0, 1, 2]]

        self._log_step(
            "sample_grasp_pose_overlook",
            f"Using centered top-down grasp for '{object_name}' with gripper yaw "
            f"{np.degrees(gripper_yaw):.1f} degrees and maximal tangent contact.",
        )
        return position, grasp_quaternion_wxyz

    def sample_grasp_pose_with_graspnet(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a Contact-GraspNet grasp pose through the parent API.

        This retains the parent API's general-purpose grasp selection behavior.
        """
        return super().sample_grasp_pose(object_name)

    def sample_grasp_pose_multi_angle(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose through the current multi-angle API baseline.

        Multi-angle selection is not implemented yet; this currently delegates
        to the parent general-purpose grasp sampler.
        """
        return super().sample_grasp_pose(object_name)

    def goto_pose_overlook(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Move to an overlook pose while keeping the gripper vertically downward.

        Only the input pose's XY-plane yaw is retained. Any pitch or roll is
        discarded before calling the parent IK solver, so this is suitable for
        top-down grasping and placement.
        """
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(quat_wxyz)) or np.linalg.norm(quat_wxyz) <= 1e-8:
            raise ValueError("quaternion_wxyz must be a finite non-zero quaternion")

        input_rotation = SciRotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
        input_x_axis = input_rotation.as_matrix()[:2, 0]
        if np.linalg.norm(input_x_axis) <= 1e-8:
            yaw = 0.0
        else:
            yaw = float(np.arctan2(input_x_axis[1], input_x_axis[0]))
        downward_rotation = SciRotation.from_quat([0.0, 1.0, 0.0, 0.0])
        vertical_rotation = SciRotation.from_euler("z", yaw) * downward_rotation
        vertical_quaternion_wxyz = vertical_rotation.as_quat()[[3, 0, 1, 2]]

        self._log_step(
            "goto_pose_overlook",
            f"Moving to vertically downward overlook pose (yaw={np.degrees(yaw):.1f} deg) ...",
        )
        super().goto_pose(position, vertical_quaternion_wxyz, z_approach=z_approach)


class FrankaExpertPickPlaceCodeEnv(FrankaPickPlaceCodeEnv):
    """Cube-stack environment whose import activates ``FrankaExpertApi``."""


# The expert module is imported by the Hydra target in the matching YAML.
# Registering here avoids changes to the global integrations registry.
register_api("FrankaExpertApi", lambda env: FrankaExpertApi(env, use_sam3=True))


__all__ = ["FrankaExpertApi", "FrankaExpertPickPlaceCodeEnv"]
