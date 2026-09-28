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
            input_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float64).reshape(4)
            target_quat_world = None if transformed_target_quat is None else np.asarray(transformed_target_quat, dtype=np.float64).reshape(4)
            returned_joints = np.asarray(returned_joints, dtype=np.float64).reshape(7)
            error = actual - target
            row: dict[str, Any] = {
                "target_x": target[0], "target_y": target[1], "target_z": target[2],
                "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
                "input_target_x": input_target[0], "input_target_y": input_target[1],
                "input_target_z": input_target[2],
                "target_frame": "world", "input_target_frame": target_frame,
                "call_index": self._verify_motion_index, "phase": phase,
                "error_x": error[0], "error_y": error[1], "error_z": error[2],
                "error_abs_x": abs(error[0]), "error_abs_y": abs(error[1]),
                "error_abs_z": abs(error[2]), "position_error_m": float(np.linalg.norm(error)),
                "actual_qw": actual_quat[0], "actual_qx": actual_quat[1],
                "actual_qy": actual_quat[2], "actual_qz": actual_quat[3],
            }
            if input_quat is not None and target_quat_world is not None:
                quat_error = Rotation.from_quat(
                    [actual_quat[1], actual_quat[2], actual_quat[3], actual_quat[0]]
                ) * Rotation.from_quat(
                    [target_quat_world[1], target_quat_world[2], target_quat_world[3], target_quat_world[0]]
                ).inv()
                quat_error_xyzw = quat_error.as_quat()
                quat_error_wxyz = np.array([quat_error_xyzw[3], *quat_error_xyzw[:3]])
                row.update({f"target_q{axis}": input_quat[i] for i, axis in enumerate("wxyz")})
                row.update({f"target_world_q{axis}": target_quat_world[i] for i, axis in enumerate("wxyz")})
                row.update({f"quat_error_q{axis}": quat_error_wxyz[i] for i, axis in enumerate("wxyz")})
                # Axis-angle rotation vector in world coordinates (radians).
                # This is the 3-D vector representation of the target
                # orientation: vector direction is the rotation axis and its
                # magnitude is the rotation angle.
                target_world_rotvec = Rotation.from_quat(
                    [
                        target_quat_world[1],
                        target_quat_world[2],
                        target_quat_world[3],
                        target_quat_world[0],
                    ]
                ).as_rotvec()
                row.update(
                    {
                        "target_world_rotvec_x": target_world_rotvec[0],
                        "target_world_rotvec_y": target_world_rotvec[1],
                        "target_world_rotvec_z": target_world_rotvec[2],
                    }
                )
                row["orientation_error_rad"] = float(quat_error.magnitude())
                row["orientation_error_deg"] = float(np.degrees(quat_error.magnitude()))
            row.update({f"returned_joint_{i}": returned_joints[i] for i in range(7)})
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
