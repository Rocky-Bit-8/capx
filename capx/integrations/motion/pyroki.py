from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRotation

from capx.utils.serve_utils import post_with_retries


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _solve_local_ik(
    env,
    target_pose_wxyz_xyz: np.ndarray,
    prev_cfg: np.ndarray | None = None,
    *,
    arm_index: int = 0,
    max_nfev: int = 300,
) -> np.ndarray:
    sim = getattr(getattr(env, "robosuite_env", None), "sim", None)
    if sim is None:
        handle = getattr(env, "handle", None)
        if handle is not None:
            sim = getattr(handle, "env", None)
            if sim is not None:
                sim = getattr(sim, "sim", None)
    if sim is None:
        sim = getattr(getattr(env, "handle", None), "sim", None)
    if sim is None and hasattr(env, "sim"):
        sim = env.sim
    if sim is None:
        raise AttributeError("Could not resolve a MuJoCo sim from env")

    if arm_index not in (0, 1):
        raise ValueError(f"Unsupported arm index: {arm_index}")

    body_id = getattr(env, f"gripper_link_idx_{arm_index}", None)
    if body_id is None and arm_index == 0:
        body_id = getattr(env, "gripper_link_idx", None)
    if body_id is None:
        body_id = getattr(env, "gripper_eef_body_id", None)
    if body_id is None:
        raise AttributeError("Could not resolve gripper link id from env")
    target_pose_wxyz_xyz = np.asarray(target_pose_wxyz_xyz, dtype=np.float64).reshape(7)
    target_wxyz = target_pose_wxyz_xyz[:4]
    target_xyz = target_pose_wxyz_xyz[4:]

    base_pose = getattr(env, f"base_link_wxyz_xyz_{arm_index}", None)
    if base_pose is None and arm_index == 0:
        base_pose = getattr(env, "base_link_wxyz_xyz", None)
    if base_pose is None:
        raise AttributeError(f"Could not resolve base pose for arm {arm_index}")
    base_pose = np.asarray(base_pose, dtype=np.float64).reshape(7)
    base_rot = SciRotation.from_quat(_quat_wxyz_to_xyzw(base_pose[:4]))
    target_rot_base = SciRotation.from_quat(_quat_wxyz_to_xyzw(target_wxyz))
    target_xyz = base_pose[4:] + base_rot.apply(target_xyz)
    target_wxyz = _quat_xyzw_to_wxyz((base_rot * target_rot_base).as_quat())

    joint_names = [f"robot{arm_index}_joint{i}" for i in range(1, 8)]
    try:
        qpos_indices = np.array(
            [int(sim.model.get_joint_qpos_addr(name)) for name in joint_names], dtype=np.intp
        )
        joint_ids = np.array(
            [int(sim.model.joint_name2id(name)) for name in joint_names], dtype=np.intp
        )
    except Exception:
        if arm_index != 0:
            raise
        qpos_indices = np.arange(7, dtype=np.intp)
        joint_ids = np.arange(7, dtype=np.intp)

    current_qpos = np.array(sim.data.qpos[qpos_indices], dtype=np.float64)

    lower = np.array(sim.model.jnt_range[joint_ids, 0], dtype=np.float64)
    upper = np.array(sim.model.jnt_range[joint_ids, 1], dtype=np.float64)
    # relax a bit to reduce optimizer failures on exact boundaries
    lower = lower + 1e-3
    upper = upper - 1e-3

    if np.any(lower >= upper):
        raise ValueError(
            f"Invalid joint bounds for arm {arm_index}: lower={lower}, upper={upper}"
        )
    margin = np.maximum(1e-8, (upper - lower) * 1e-8)

    def clamp_seed(seed: np.ndarray) -> np.ndarray:
        seed = np.asarray(seed, dtype=np.float64).reshape(-1)
        if seed.size != 7:
            raise ValueError(f"IK seed has shape {seed.shape}; expected seven arm joints.")
        seed = np.nan_to_num(seed, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(seed, lower + margin, upper - margin)

    saved_qpos = np.array(sim.data.qpos, dtype=np.float64)
    saved_qvel = np.array(sim.data.qvel, dtype=np.float64)

    target_xyzw = _quat_wxyz_to_xyzw(target_wxyz)

    def residual(q: np.ndarray) -> np.ndarray:
        sim.data.qpos[qpos_indices] = q
        sim.data.qvel[:] = 0.0
        sim.forward()
        cur_xyz = np.array(sim.data.xpos[body_id], dtype=np.float64)
        cur_wxyz = np.array(sim.data.xquat[body_id], dtype=np.float64)
        cur_xyzw = _quat_wxyz_to_xyzw(cur_wxyz)
        pos_err = cur_xyz - target_xyz
        rel = SciRotation.from_quat(target_xyzw) * SciRotation.from_quat(cur_xyzw).inv()
        ori_err = rel.as_rotvec()
        return np.concatenate([5.0 * pos_err, 1.5 * ori_err])

    try:
        # A single warm start is unreliable for arbitrary Contact-GraspNet
        # orientations: least_squares may stop at a bad local minimum while
        # returning a perfectly valid-looking joint vector.  Solve from the
        # measured state, the previous target, home, centre of limits, and
        # deterministic whole-workspace samples, then retain the closest FK
        # result.  This is deliberately collision-agnostic IK; trajectory
        # collision avoidance belongs to the planner that consumes this pose.
        seeds = [current_qpos]
        if prev_cfg is not None:
            prev_cfg = np.asarray(prev_cfg, dtype=np.float64).reshape(-1)
            if prev_cfg.size >= 7:
                seeds.append(prev_cfg[:7])
        home = getattr(env, "home_joint_position", None)
        if home is not None:
            seeds.append(np.asarray(home, dtype=np.float64).reshape(7))
        seeds.append((lower + upper) / 2.0)
        rng = np.random.default_rng(0)
        seeds.extend(rng.uniform(lower, upper, size=(12, 7)))

        best_q: np.ndarray | None = None
        best_cost = np.inf
        best_pos_error = np.inf
        best_ori_error = np.inf
        successful_solves = 0
        seen: list[np.ndarray] = []
        for raw_seed in seeds:
            seed = clamp_seed(raw_seed)
            if any(np.allclose(seed, previous, rtol=0.0, atol=1e-6) for previous in seen):
                continue
            seen.append(seed)
            result = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
                x_scale="jac",
            )
            q = np.asarray(result.x, dtype=np.float64)
            err = residual(q)
            cost = float(np.dot(err, err))
            pos_error = float(np.linalg.norm(err[:3] / 5.0))
            ori_error = float(np.linalg.norm(err[3:] / 1.5))
            if result.success:
                successful_solves += 1
            if cost < best_cost:
                best_q = q
                best_cost = cost
                best_pos_error = pos_error
                best_ori_error = ori_error

        if best_q is None:
            raise RuntimeError("Local IK did not produce a candidate solution.")
        print(
            "Local IK best of "
            f"{len(seen)} seeds ({successful_solves} converged): "
            f"position_error={best_pos_error:.4f} m, "
            f"orientation_error={np.degrees(best_ori_error):.2f} deg"
        )
        q = best_q
    finally:
        sim.data.qpos[:] = saved_qpos
        sim.data.qvel[:] = saved_qvel
        sim.forward()

    return np.concatenate([q, np.array([0.0], dtype=np.float64)])


def init_pyroki_local(env, *, arm_index: int = 0, max_nfev: int = 300):
    """Create an IK solver bound explicitly to a local MuJoCo environment."""
    if env is None:
        raise ValueError("init_pyroki_local requires a MuJoCo-backed env")

    def ik_solve_fn(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None) -> np.ndarray:
        return _solve_local_ik(
            env,
            target_pose_wxyz_xyz,
            prev_cfg=prev_cfg,
            arm_index=arm_index,
            max_nfev=max_nfev,
        )

    return ik_solve_fn


def init_pyroki_remote(server_url: str = "http://127.0.0.1:8116"):
    """Create an IK solver that explicitly calls the real-robot PyRoKi server."""
    server_url = server_url.rstrip("/")

    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        pose = np.asarray(target_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        payload = {
            "target_pose_wxyz_xyz": pose.tolist(),
            "prev_cfg": (
                np.asarray(prev_cfg, dtype=np.float64).reshape(-1).tolist()
                if prev_cfg is not None
                else None
            ),
        }
        data = post_with_retries(f"{server_url}/ik", payload)
        joints = np.asarray(data["joint_positions"], dtype=np.float64).reshape(-1)
        if joints.size != 7:
            raise ValueError(f"PyRoKi server returned {joints.size} joints; expected 7")
        # Preserve the existing control API cfg shape: 7 arm joints + gripper slot.
        return np.concatenate([joints, np.array([0.0], dtype=np.float64)])

    return ik_solve_fn


def init_pyroki(
    server_url: str = "http://127.0.0.1:8116",
    robot_urdf_or_name: str = "panda_description",
    target_link_name: str = "panda_hand",
    env=None,
    arm_index: int = 0,
    use_remote: bool | None = None,
):
    """Compatibility wrapper; new code should choose local or remote explicitly."""
    del robot_urdf_or_name, target_link_name
    if use_remote is True:
        return init_pyroki_remote(server_url)
    if use_remote is False:
        return init_pyroki_local(env, arm_index=arm_index)
    if env is not None:
        return init_pyroki_local(env, arm_index=arm_index)
    raise RuntimeError(
        "Choose an IK backend explicitly: init_pyroki_local(env) or "
        "init_pyroki_remote(server_url)"
    )


def init_pyroki_trajopt_local(env):
    """Create a trajectory helper explicitly bound to local MuJoCo."""
    if env is None:
        raise ValueError("init_pyroki_trajopt_local requires a MuJoCo-backed env")

    def trajopt_plan_fn(start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _solve_local_ik(env, start_pose_wxyz_xyz),
                _solve_local_ik(env, end_pose_wxyz_xyz),
            ],
            axis=0,
        )

    return trajopt_plan_fn


def init_pyroki_trajopt(
    server_url: str = "http://127.0.0.1:8116",
    robot_urdf_or_name: str = "panda_description",
    target_link_name: str = "panda_hand",
    env=None,
):
    """Compatibility wrapper for the local trajectory helper."""
    del server_url, robot_urdf_or_name, target_link_name
    return init_pyroki_trajopt_local(env)
