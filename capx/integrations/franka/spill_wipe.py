import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki_local
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.franka.common import (
    apply_tcp_offset,
    build_segmentation_map_from_sam2,
    compute_bbox_indices,
    draw_boxes,
    save_segmentation_debug,
    select_instance_from_box,
)
from capx.integrations.vision.sam3 import init_sam3, visualize_sam3_results
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)


# ------------------------------- Control API ------------------------------
class FrankaControlSpillWipeApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] = [0.0, 0.0, -0.107],
        use_sam3: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        print("init franka control api")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        # Pyroki solves against the active MuJoCo scene in-process. The old
        # 8116 HTTP server cannot provide this task-specific simulator state.
        self.ik_solve_fn = init_pyroki_local(self._env)
        self.cfg = None
        self.use_sam3 = use_sam3
        self.debug = debug
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            print("init sam3 seg fn")
        else:
            self.owl_vit_det_fn = init_owlvit(device="cuda")
            print("init owlvit det fn")
            self.sam2_seg_fn = init_sam2()
            print("init sam2 seg fn")

    def functions(self) -> dict[str, Any]:
        fns = {
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
        }
        return fns

    def _get_segmentation_map(
        self, obs: dict[str, Any], rgb: np.ndarray, box: list[float] = None
    ) -> np.ndarray:
        return build_segmentation_map_from_sam2(
            self.sam2_seg_fn, rgb, obs["robot0_robotview"]["images"], box=box
        )

    def _save_segmentation_debug(self, segmentation: np.ndarray, path: pathlib.Path) -> None:
        save_segmentation_debug(segmentation, path)

    def _compute_bbox_indices(
        self, box: list[float], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        return compute_bbox_indices(box, shape)

    def _select_instance_from_box(
        self, segmentation: np.ndarray, box: list[float]
    ) -> tuple[int, np.ndarray]:
        return select_instance_from_box(segmentation, box)

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent:  Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        start_time = time.time()
        obs = self._env.get_observation()

        rbg_imgs = obs_get_rgb(obs)
        assert len(rbg_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rbg_imgs.values())[0]

        depth = obs["robot0_robotview"]["images"]["depth"]

        # Debug image saves TODO: Remove this eventually, or add a debug mode branch
        # save depth image with colormap
        depth_img = depth_to_rgb(depth[:, :, 0])
        depth_img_out = Image.fromarray(depth_img)
        depth_img_out.save("depth_image.jpg")

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        if self.use_sam3:
            results = self.sam3_seg_fn(rgb, text_prompt=object_name)
            if len(results) == 0:
                raise ValueError("No sam3 detections")
            scores = [result["score"] for result in results]

            box = results[np.argmax(scores)]["box"]
            mask = results[np.argmax(scores)]["mask"]

            if self.debug:
                visualize_sam3_results(
                    Image.fromarray(rgb),
                    object_name,
                    results,
                    output_dir=pathlib.Path("."),
                    show=False,
                )
            idxs = np.where(mask.flatten() & binary_map_nan_is_zero.flatten().astype(bool))
        else:
            dets = self.owl_vit_det_fn(rgb, texts=[[object_name]])

            if len(dets) == 0:
                raise ValueError("No detections; environment constraints or model mismatch")

            boxes = [d["box"] for d in dets]
            labels = [d["label"] for d in dets]
            scores = [d["score"] for d in dets]

            box = boxes[np.argmax(scores)]

            if self.debug:
                img_out = _draw_boxes(
                    rgb, [box], [labels[np.argmax(scores)]], scores=[scores[np.argmax(scores)]]
                )
                out_file = pathlib.Path("owlvit_det.jpg")
                img_out.save(out_file)
                assert out_file.exists() and out_file.stat().st_size > 0

            # save segmentation image
            segmentation = self._get_segmentation_map(obs, rgb, box=box)
            if self.debug:
                self._save_segmentation_debug(segmentation, pathlib.Path("segmentation_image.jpg"))

            queried_instance_idx, seg_crop = self._select_instance_from_box(segmentation, box)
            if self.debug:
                self._save_segmentation_debug(seg_crop, pathlib.Path("seg_crop_image.jpg"))

            # idxs = np.where(segmentation.flatten() == queried_instance_idx) # Old assumes there are no Nans in the depth map (happens in real ZED returns)
            idxs = np.where(
                segmentation.flatten()[binary_map_nan_is_zero.flatten().astype(bool)]
                == queried_instance_idx
            )

        # points = depth_to_pointcloud(depth[:, :, 0], obs["robot0_robotview"]["intrinsics"])[idxs]
        points, color = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs["robot0_robotview"]["intrinsics"]
        )

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points[idxs])

        obb = o3d_points.get_oriented_bounding_box()

        # Exposing these to the low level environment for viser

        self._env.cube_points = points[idxs]
        self._env.cube_color = color[idxs]

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["robot0_robotview"]["pose"][3:]),
            translation=obs["robot0_robotview"]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        obb_tf_world = cam_extr_tf @ obb_tf

        self._env.cube_center = obb_tf_world.translation()
        self._env.cube_rot = obb_tf_world.rotation().as_matrix()

        x1, y1, x2, y2 = box

        # Camera intrinsics and extrinsics
        K = obs["robot0_robotview"]["intrinsics"]  # (3,3)
        pose_mat = obs["robot0_robotview"]["pose_mat"]  # (4,4)

        # Camera pose in world: world to camera
        R_cw = pose_mat[:3, :3]
        t_cw = pose_mat[:3, 3]

        # Function to backproject a pixel to world point on table plane z=0
        def pixel_to_world(u, v):
            # Pixel direction in camera coordinates
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            dir_cam /= np.linalg.norm(dir_cam)
            # Direction in world coordinates
            dir_world = R_cw @ dir_cam
            # Solve for t such that (t_cw + t*dir_world)[2] = 0
            t = -t_cw[2] / dir_world[2]
            point_world = t_cw + t * dir_world
            return point_world

        # Compute world coordinates of bounding box corners
        p1 = pixel_to_world(x1, y1)
        p2 = pixel_to_world(x2, y1)
        p3 = pixel_to_world(x2, y2)
        p4 = pixel_to_world(x1, y2)

        # Determine min/max x and y based on the bounding box corners
        xs = np.array([p1[0], p2[0], p3[0], p4[0]])
        ys = np.array([p1[1], p2[1], p3[1], p4[1]])
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        extent = np.array([xmax - xmin, ymax - ymin, 0.001])

        print(f"get_object_pose in {time.time() - start_time} seconds")
        if return_bbox_extent:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], extent
        else:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], None

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

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

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint where you want to evaluate progress and potentially regenerate the remaining code.

        Args:
            None
        """
        return None


# --------------------------- Whole-table wipe API --------------------------
class FrankaControlSpillWipeTableApi(FrankaControlSpillWipeApi):
    """Spill-wipe control API extended with a whole-table wiping skill.

    Functions:
      - get_object_pose(object_name: str, return_bbox_extent: bool = False)
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray) -> None
      - wipe_table_surface(...) -> dict[str, Any]
    """

    # Downward-facing (sponge down) orientation used for every wiping motion.
    _WIPE_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["wipe_table_surface"] = self.wipe_table_surface
        return fns

    # ------------------------- Internal helpers ----------------------------
    def _depth_points_in_base_frame(
        self, camera_name: str = "robot0_robotview", subsample_factor: int = 4
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Deproject the current depth image into the robot base frame.

        The base frame is the same frame used by goto_pose() and
        get_object_pose(), so the returned points can be commanded directly.

        Args:
            camera_name: Camera entry in the observation dictionary.
            subsample_factor: Stride used when deprojecting the depth image.

        Returns:
            points_base: (N, 3) XYZ points in meters, in the robot base frame.
            ee_pos: (3,) current end-effector XYZ in the robot base frame, or
                None when the observation does not expose it.
        """
        obs = self._env.get_observation()
        cam = obs.get(camera_name)
        if cam is None or "depth" not in cam.get("images", {}):
            raise ValueError(f"No depth image available for camera '{camera_name}'.")

        depth = np.asarray(cam["images"]["depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        intrinsics = np.asarray(cam["intrinsics"], dtype=np.float64)
        pose_mat = np.asarray(cam["pose_mat"], dtype=np.float64)

        points_cam = depth_to_pointcloud(depth, intrinsics, subsample_factor=subsample_factor)
        if points_cam.size == 0:
            raise ValueError("Depth image produced no valid 3D points; cannot locate the table.")
        points_base = points_cam @ pose_mat[:3, :3].T + pose_mat[:3, 3]

        ee_pos = None
        cartesian = obs.get("robot_cartesian_pos")
        if cartesian is not None:
            arr = np.asarray(cartesian, dtype=np.float64).reshape(-1)
            if arr.size >= 3 and bool(np.all(np.isfinite(arr[:3]))):
                ee_pos = arr[:3]
        return points_base, ee_pos

    def _estimate_table_surface(
        self,
        points_base: np.ndarray,
        ee_pos: np.ndarray | None,
        min_points_fraction: float = 0.05,
        bin_width: float = 0.002,
    ) -> float:
        """Estimate the table surface height (z in the base frame) from depth.

        The table is the largest horizontal surface in the scene, so its height
        is the dominant peak of the z histogram. Everything at or above the end
        effector (arm links / sponge) is excluded, which also removes lower
        horizontal surfaces such as the floor.

        Args:
            points_base: (N, 3) XYZ points in the robot base frame.
            ee_pos: (3,) end-effector position, or None if unavailable.
            min_points_fraction: Minimum share of points a height bin must hold
                to be considered a surface candidate.
            bin_width: Histogram bin width in meters.

        Returns:
            table_z: Table surface height in meters, in the robot base frame.
        """
        zs = points_base[:, 2]
        z_min, z_max = float(np.min(zs)), float(np.max(zs))
        n_bins = max(int(np.ceil((z_max - z_min) / bin_width)), 1)
        counts, edges = np.histogram(zs, bins=n_bins, range=(z_min, z_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Smooth over ~1 cm so a slightly tilted/noisy surface stays one peak.
        smoothed = np.convolve(counts, np.ones(5, dtype=np.float64), mode="same")

        threshold = max(50.0, min_points_fraction * float(len(zs)))
        candidates = smoothed >= threshold
        if ee_pos is not None:
            candidates &= centers <= (float(ee_pos[2]) - 0.02)
        if not bool(np.any(candidates)):
            raise RuntimeError(
                "Could not find a dominant horizontal surface below the end effector. "
                "Make sure the table is visible in the depth image."
            )
        # Highest candidate = table top (floor/lower surfaces sit below it).
        peak_z = float(centers[int(np.max(np.nonzero(candidates)[0]))])
        near_peak = np.abs(zs - peak_z) < 0.01
        return float(np.median(zs[near_peak]))

    # --------------------------- Public skill ------------------------------
    def wipe_table_surface(
        self,
        step_size: float = 0.05,
        lift_height: float = 0.08,
        margin: float = 0.02,
        press_depth: float = 0.005,
        max_reach: float = 0.65,
        max_waypoints: int = 120,
        camera_name: str = "robot0_robotview",
    ) -> dict[str, Any]:
        """Wipe the whole table surface when the spill is not fully cleaned up.

        Call this only after a normal wiping attempt when the table still looks
        dirty: it is a brute-force cleanup, not a replacement for wiping the
        spill itself.

        The table height is measured from the current depth image and then held
        constant: every wiping waypoint uses that same z coordinate, while x and
        y sweep a serpentine (back-and-forth) grid over the visible table
        surface. The sponge keeps a downward-facing orientation (0, 0, 1, 0)
        wxyz for the whole motion, and the arm lifts between rows so it never
        drags across the table while repositioning.

        Args:
            step_size: Spacing in meters between wiping waypoints.
            lift_height: Height in meters above the wipe height used when moving
                between rows.
            margin: Inset in meters applied to the table bounds so the sponge
                stays away from the table edges.
            press_depth: How far in meters below the measured surface height to
                press, so the sponge keeps contact despite depth noise.
            max_reach: Maximum horizontal distance in meters from the robot base
                for a waypoint; farther waypoints are skipped as unreachable.
            max_waypoints: Safety cap on the number of wiping waypoints; the
                step size is enlarged automatically when the grid is larger.
            camera_name: Camera entry used for the depth measurement.

        Returns:
            A dict with the measured table height ("table_surface_z"), the
            wiping height ("wipe_z"), the swept bounds ("x_range", "y_range"),
            the executed waypoints ("waypoints", shape (M, 2) in meters), and
            counts of skipped/failed points ("num_skipped_unreachable",
            "num_failed_moves").
        """
        points_base, ee_pos = self._depth_points_in_base_frame(camera_name)
        table_z = self._estimate_table_surface(points_base, ee_pos)

        # Points that lie on the measured table plane define the sweep bounds.
        on_table = points_base[np.abs(points_base[:, 2] - table_z) < 0.01][:, :2]
        if on_table.shape[0] < 10:
            raise RuntimeError(
                "Too few depth points lie on the estimated table plane; cannot wipe the table."
            )
        xmin, ymin = np.min(on_table, axis=0)
        xmax, ymax = np.max(on_table, axis=0)
        xmin += margin
        xmax -= margin
        ymin += margin
        ymax -= margin
        if (xmax - xmin) < 1e-3 or (ymax - ymin) < 1e-3:
            raise RuntimeError(
                f"Table region is too small to wipe after applying margin={margin} m: "
                f"x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}]"
            )

        n_x = max(int(np.ceil((xmax - xmin) / step_size)) + 1, 2)
        n_y = max(int(np.ceil((ymax - ymin) / step_size)) + 1, 2)
        if n_x * n_y > max_waypoints:
            scale = float(np.sqrt((n_x * n_y) / max_waypoints))
            n_x = max(int(np.floor(n_x / scale)), 2)
            n_y = max(int(np.floor(n_y / scale)), 2)
        xs = np.linspace(xmin, xmax, n_x)
        ys = np.linspace(ymin, ymax, n_y)

        wipe_z = table_z - abs(press_depth)
        lift_z = wipe_z + abs(lift_height)
        quat = self._WIPE_QUAT_WXYZ

        print(
            f"[wipe_table_surface] table_z={table_z:.4f} m, wipe_z={wipe_z:.4f} m, "
            f"x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}], "
            f"grid={n_x}x{n_y}, step={step_size} m"
        )

        waypoints: list[list[float]] = []
        skipped = 0
        failed = 0

        for row_idx, y in enumerate(ys):
            row_xs = xs if row_idx % 2 == 0 else xs[::-1]
            # Split the row into contiguous reachable runs so the arm never
            # drags across an unreachable gap at wiping height.
            runs: list[list[float]] = []
            run: list[float] = []
            for x in row_xs:
                if float(np.hypot(x, y)) <= max_reach:
                    run.append(float(x))
                else:
                    skipped += 1
                    if len(run) >= 2:
                        runs.append(run)
                    run = []
            if len(run) >= 2:
                runs.append(run)
            else:
                skipped += len(run)

            for run_xs in runs:
                try:
                    self.goto_pose(np.array([run_xs[0], y, lift_z]), quat)
                    self.goto_pose(np.array([run_xs[0], y, wipe_z]), quat)
                    for x in run_xs[1:]:
                        self.goto_pose(np.array([x, y, wipe_z]), quat)
                    self.goto_pose(np.array([run_xs[-1], y, lift_z]), quat)
                except Exception as exc:  # keep cleaning; report failures
                    failed += 1
                    print(f"[wipe_table_surface] skipping row y={y:.3f}: {exc!r}")
                    continue
                waypoints.extend([[float(x), float(y)] for x in run_xs])

        summary: dict[str, Any] = {
            "table_surface_z": float(table_z),
            "wipe_z": float(wipe_z),
            "x_range": [float(xmin), float(xmax)],
            "y_range": [float(ymin), float(ymax)],
            "waypoints": np.asarray(waypoints, dtype=np.float64).reshape(-1, 2),
            "num_waypoints": len(waypoints),
            "num_skipped_unreachable": int(skipped),
            "num_failed_moves": int(failed),
        }
        print(
            f"[wipe_table_surface] done: {len(waypoints)} waypoints wiped, "
            f"{skipped} skipped as unreachable, {failed} rows failed"
        )
        return summary


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    return draw_boxes(rgb, boxes, labels, scores)
