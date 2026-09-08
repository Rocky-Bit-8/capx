"""Probe cuRobo IK with fixed, local-frame target poses.

The cuRobo server must already be running on 127.0.0.1:8117.
"""

from __future__ import annotations

import argparse
import sys

import requests


# [qw, qx, qy, qz, x, y, z], positions are in the Panda base frame.
TARGETS = [
    ("center_down", [0.0, 1.0, 0.0, 0.0, 0.78, 0.00, 0.25]),
    ("left_down", [0.0, 1.0, 0.0, 0.0, 0.70, -0.15, 0.25]),
    ("right_down", [0.0, 1.0, 0.0, 0.0, 0.70, 0.15, 0.25]),
    ("front_down", [0.0, 1.0, 0.0, 0.0, 0.60, 0.00, 0.20]),
    ("back_down", [0.0, 1.0, 0.0, 0.0, 0.90, 0.00, 0.25]),
    ("center_high", [0.0, 1.0, 0.0, 0.0, 0.78, 0.00, 0.45]),
    ("center_identity", [1.0, 0.0, 0.0, 0.0, 0.78, 0.00, 0.35]),
    ("left_identity", [1.0, 0.0, 0.0, 0.0, 0.70, -0.15, 0.35]),
    ("right_identity", [1.0, 0.0, 0.0, 0.0, 0.70, 0.15, 0.35]),
    ("center_tilt_x", [0.7071, 0.7071, 0.0, 0.0, 0.78, 0.00, 0.30]),
    ("center_tilt_y", [0.7071, 0.0, 0.7071, 0.0, 0.78, 0.00, 0.30]),
    ("center_tilt_z", [0.7071, 0.0, 0.0, 0.7071, 0.78, 0.00, 0.30]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8117")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        health.raise_for_status()
        print(f"server: {health.json()}")
    except requests.RequestException as exc:
        print(f"cuRobo server unavailable at {base_url}: {exc}", file=sys.stderr)
        return 2

    successes = 0
    for index, (name, pose) in enumerate(TARGETS, start=1):
        try:
            response = requests.post(
                f"{base_url}/ik",
                json={"target_pose_wxyz_xyz": pose, "prev_cfg": None},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            success = bool(data.get("success", False))
            successes += int(success)
            joints = data.get("joint_positions", [])
            print(
                f"{index:02d} {name:18s} success={success!s:5s} "
                f"joints={len(joints)} target_xyz={pose[4:]}"
            )
        except requests.RequestException as exc:
            print(f"{index:02d} {name:18s} request_failed={exc}")

    total = len(TARGETS)
    print(f"success_rate={successes}/{total} ({100.0 * successes / total:.1f}%)")
    return 0 if successes == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
