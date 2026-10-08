"""Reusable Franka verification artifacts and diagnostics API."""

from __future__ import annotations

import csv
import os
import pathlib
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from capx.integrations.base_api import ApiBase


class FrankaVerifyRecorder:
    """Best-effort recorder for images, IK vectors, and simulator EE poses."""

    def __init__(self, output_dir: str | os.PathLike[str] | None = None) -> None:
        self._verify_dir: pathlib.Path | None = None
        self._verify_image_index = 0
        self._verify_motion_index = 0
        self.set_output_dir(output_dir)

    def set_output_dir(self, output_dir: str | os.PathLike[str] | None) -> None:
        self._verify_dir = pathlib.Path(output_dir) if output_dir else None
        if self._verify_dir is not None:
            self._verify_dir.mkdir(parents=True, exist_ok=True)

    def path(self) -> pathlib.Path:
        if self._verify_dir is None:
            self._verify_dir = pathlib.Path("/home/rocky/Code/cap-x-main/bin/verify")
            self._verify_dir.mkdir(parents=True, exist_ok=True)
        return self._verify_dir

    def save_image(self, image: Image.Image, stem: str) -> None:
        try:
            image.save(self.path() / f"{self._verify_image_index:04d}_{stem}.png")
            self._verify_image_index += 1
        except Exception as exc:  # diagnostics must not affect control
            print(f"[verify] Could not save image: {exc}")

    def record_ik(
        self,
        target: np.ndarray,
        actual_pose: tuple[np.ndarray, np.ndarray] | None,
        returned_joints: np.ndarray,
        *,
        phase: str = "final",
        target_frame: str = "world",
        transformed_target: np.ndarray | None = None,
        target_quat: np.ndarray | None = None,
        transformed_target_quat: np.ndarray | None = None,
    ) -> None:
        if actual_pose is None:
            return
        try:
            input_target = np.asarray(target, dtype=np.float64).reshape(3)
            target = (
                np.asarray(transformed_target, dtype=np.float64).reshape(3)
                if transformed_target is not None
                else input_target
            )
            actual, actual_quat = actual_pose
            actual = np.asarray(actual, dtype=np.float64).reshape(3)
            actual_quat = np.asarray(actual_quat, dtype=np.float64).reshape(4)
            error = actual - target
            row: dict[str, Any] = {
                # Position columns are intentionally interleaved target/actual.
                "world_target_x": target[0], "actual_x": actual[0],
                "world_target_y": target[1], "actual_y": actual[1],
                "world_target_z": target[2], "actual_z": actual[2],
                "target_direction_x": "", "target_direction_y": "",
                "target_direction_z": "", "actual_direction_x": "",
                "actual_direction_y": "", "actual_direction_z": "",
                "abs_position_error": float(np.linalg.norm(error)),
                "abs_orientation_error_deg": "",
            }
            target_quat_world = (
                None
                if transformed_target_quat is None
                else np.asarray(transformed_target_quat, dtype=np.float64).reshape(4)
            )
            if target_quat_world is not None:
                target_rotation = Rotation.from_quat(
                    [
                        target_quat_world[1], target_quat_world[2],
                        target_quat_world[3], target_quat_world[0],
                    ]
                )
                actual_rotation = Rotation.from_quat(
                    [actual_quat[1], actual_quat[2], actual_quat[3], actual_quat[0]]
                )
                # Direction vector means the world direction of the pose's
                # local +Z axis, i.e. the third column of its rotation matrix.
                target_direction = target_rotation.apply([0.0, 0.0, 1.0])
                actual_direction = actual_rotation.apply([0.0, 0.0, 1.0])
                quat_error = Rotation.from_quat(
                    [actual_quat[1], actual_quat[2], actual_quat[3], actual_quat[0]]
                ) * target_rotation.inv()
                row.update(
                    {
                        "target_direction_x": target_direction[0],
                        "target_direction_y": target_direction[1],
                        "target_direction_z": target_direction[2],
                        "actual_direction_x": actual_direction[0],
                        "actual_direction_y": actual_direction[1],
                        "actual_direction_z": actual_direction[2],
                        "abs_orientation_error_deg": float(
                            np.degrees(quat_error.magnitude())
                        ),
                    }
                )
            else:
                for name in (
                    "target_direction_x", "target_direction_y", "target_direction_z",
                    "actual_direction_x", "actual_direction_y", "actual_direction_z",
                ):
                    row[name] = ""
            path = self.path() / "goto_pose_vectors.csv"
            fieldnames = list(row)
            old_rows: list[dict[str, str]] = []
            rewrite = not path.exists()
            if path.exists():
                with path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames != fieldnames:
                        old_rows = list(reader)
                        rewrite = True
            with path.open("w" if rewrite else "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if rewrite:
                    writer.writeheader()
                    for old in old_rows:
                        writer.writerow({key: old.get(key, "") for key in fieldnames})
                writer.writerow({
                    key: f"{value:.9f}" if isinstance(value, np.floating) else value
                    for key, value in row.items()
                })
            self._verify_motion_index += 1
        except Exception as exc:  # diagnostics must not affect control
            print(f"[verify] Could not record IK vector: {exc}")


class FrankaVerifyApi(ApiBase):
    """Optional verification helpers reusable by Franka task APIs."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.verify_recorder = FrankaVerifyRecorder()

    def _target_in_world(self, target: np.ndarray, target_frame: str) -> np.ndarray:
        """Convert a local-IK base-frame target to simulator world coordinates."""
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if target_frame == "world":
            return target
        if target_frame != "base":
            raise ValueError(f"Unsupported verification target frame: {target_frame}")
        base_pose = getattr(self._env, "base_link_wxyz_xyz", None)
        if base_pose is None:
            raise AttributeError("Environment has no base_link_wxyz_xyz for base-frame verification")
        base_pose = np.asarray(base_pose, dtype=np.float64).reshape(7)
        base_rot = Rotation.from_quat(
            [base_pose[1], base_pose[2], base_pose[3], base_pose[0]]
        )
        return base_pose[4:] + base_rot.apply(target)

    def _quat_in_world(self, quat: np.ndarray, target_frame: str) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if target_frame == "world":
            return quat
        base_pose = np.asarray(self._env.base_link_wxyz_xyz, dtype=np.float64).reshape(7)
        base_rot = Rotation.from_quat([base_pose[1], base_pose[2], base_pose[3], base_pose[0]])
        target_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        world_xyzw = (base_rot * target_rot).as_quat()
        return np.array([world_xyzw[3], *world_xyzw[:3]], dtype=np.float64)

    def set_verify_output_dir(self, output_dir: str | os.PathLike[str] | None) -> None:
        """Set the directory used for verification artifacts."""
        self.verify_recorder.set_output_dir(output_dir)

    def verify_save_image(self, image: Image.Image, name: str = "image") -> None:
        """Save an RGB/PIL verification image under the configured verify directory."""
        self.verify_recorder.save_image(image, name)

    def verify_record_ik(
        self,
        target: np.ndarray,
        actual_pose: tuple[np.ndarray, np.ndarray] | None,
        returned_joints: np.ndarray,
        phase: str = "final",
        target_frame: str = "world",
        target_quat: np.ndarray | None = None,
    ) -> None:
        """Append target, actual pose, error, and returned joints to the IK CSV."""
        try:
            transformed = self._target_in_world(target, target_frame)
        except Exception as exc:
            print(f"[verify] Could not transform target to world coordinates: {exc}")
            transformed = np.asarray(target, dtype=np.float64).reshape(3)
        transformed_quat = None
        if target_quat is not None:
            try:
                transformed_quat = self._quat_in_world(target_quat, target_frame)
            except Exception as exc:
                print(f"[verify] Could not transform target quaternion: {exc}")
                transformed_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
        self.verify_recorder.record_ik(
            target,
            actual_pose,
            returned_joints,
            phase=phase,
            target_frame=target_frame,
            transformed_target=transformed,
            target_quat=target_quat,
            transformed_target_quat=transformed_quat,
        )

    def functions(self) -> dict[str, Any]:
        return {
            "set_verify_output_dir": self.set_verify_output_dir,
            "verify_save_image": self.verify_save_image,
            "verify_record_ik": self.verify_record_ik,
        }
