from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.motion.pyroki import init_pyroki_remote


# ------------------------------- Control API ------------------------------
class FrankaControlNutAssemblyPrivilegedApi(ApiBase):
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

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        # self._TCP_OFFSET = _TCP_OFFSET
        self.ik_solve_fn = init_pyroki_remote()
        self.cfg: np.ndarray | None = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "goto_home_joint_position": self.goto_home_joint_position,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_nut_handle_to_center_offset(self, object_name: str) -> np.ndarray:
        """Get the offset of the nut handle from the nut center.

        Args:
            object_name: The name of the object to get the offset of.

        Returns:
            offset: (3,) XYZ translation offset in meters, nut handle in the frame of the nut center
        """
        self._log_step("Nut Handle Offset", "Reading nut handle-to-center offset ...")
        obs = self._env.get_observation()
        offset = -obs["nut_poses"]["nut_handle_to_center_offset"]
        self._log_step_update(text=f"Offset: {np.array2string(offset, precision=3)} m")
        return offset

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        self._log_step("Nut Pose Query", f"Reading privileged pose for '{object_name}' ...")
        obs = self._env.get_observation()

        if all(i in object_name for i in ["square", "nut", "handle"]):
            pose = obs["nut_poses"]["square_nut_handle"]
        elif all(i in object_name for i in ["square", "nut"]):
            pose = obs["nut_poses"]["square_nut"]
        elif any(i in object_name for i in ["block", "peg"]):
            pose = obs["nut_poses"]["square_peg"]
        else:
            raise ValueError(f"Invalid object name: {object_name}")
        self._log_step_update(text=f"Pose: {np.array2string(pose[:3], precision=3)} m")
        return pose[:3], pose[3:]

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        self._log_step("Nut Grasp Pose", f"Reading privileged grasp pose for '{object_name}' ...")
        obs = self._env.get_observation()

        if all(i in object_name for i in ["square", "nut", "handle"]):
            pose = obs["nut_poses"]["square_nut_handle"]
        elif any(i in object_name for i in ["peg", "block"]):
            pose = obs["nut_poses"]["square_peg"]
        else:
            raise ValueError(f"Invalid object name: {object_name}")
        self._log_step_update(text=f"Grasp pose: {np.array2string(pose[:3], precision=3)} m")
        return pose[:3], pose[3:]

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

    @staticmethod
    def _extract_arm_joints(cfg: np.ndarray) -> np.ndarray:
        """PyRoKI returns actuated joints including gripper; strip to Panda arm joints."""
        return np.asarray(cfg[:-1], dtype=np.float64).reshape(7)
