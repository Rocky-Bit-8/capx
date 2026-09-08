from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from capx.utils.serve_utils import post_with_retries

DEFAULT_URL = "http://127.0.0.1:8117"


def init_curobo(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Return an IK solver callable that forwards requests to a cuRobo server.

    Same interface as ``init_pyroki()`` — returns a function with signature::

        ik_solve_fn(target_pose_wxyz_xyz, prev_cfg=None) -> joint_positions

    Args:
        server_url: Base URL of the cuRobo FastAPI server.

    Returns:
        IK solver function.
    """
    server_url = server_url.rstrip("/")

    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        pose = np.asarray(target_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        seed = None
        if prev_cfg is not None:
            seed_array = np.asarray(prev_cfg, dtype=np.float64).reshape(-1)
            if seed_array.size < 7:
                raise ValueError(f"cuRobo seed has {seed_array.size} values; expected at least 7")
            seed = seed_array[:7]
        payload = {
            "target_pose_wxyz_xyz": pose.tolist(),
            "prev_cfg": seed.tolist() if seed is not None else None,
        }
        data = post_with_retries(f"{server_url}/ik", payload)
        if not bool(data.get("success", False)):
            raise RuntimeError(
                "cuRobo could not solve IK for target "
                f"{np.array2string(pose, precision=4)}"
            )
        joints = np.asarray(data["joint_positions"], dtype=np.float64).reshape(-1)
        if joints.size != 7:
            raise ValueError(f"cuRobo returned {joints.size} joints; expected 7")
        # Match the legacy PyRoKi control shape: seven arm joints plus the
        # gripper placeholder consumed by the existing Franka APIs.
        return np.concatenate([joints, np.array([0.0], dtype=np.float64)])

    return ik_solve_fn


def init_curobo_trajopt(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a trajectory-optimisation callable that forwards to a cuRobo server.

    Same interface as ``init_pyroki_trajopt()`` — returns a function with signature::

        trajopt_plan_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz) -> waypoints

    Args:
        server_url: Base URL of the cuRobo FastAPI server.

    Returns:
        Trajectory planning function.
    """
    server_url = server_url.rstrip("/")

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        payload = {
            "start_pose_wxyz_xyz": start_pose_wxyz_xyz.tolist(),
            "end_pose_wxyz_xyz": end_pose_wxyz_xyz.tolist(),
        }
        data = post_with_retries(f"{server_url}/plan", payload)
        return np.asarray(data["waypoints"], dtype=np.float32)

    return trajopt_plan_fn
