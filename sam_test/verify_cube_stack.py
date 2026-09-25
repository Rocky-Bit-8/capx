#!/usr/bin/env python3
"""Capture the Robosuite cube-stack scene and validate SAM3 + Contact-GraspNet."""

from __future__ import annotations

import sys

from verify_nut_assembly import main


if __name__ == "__main__":
    # Reuse the identical capture, planning, serialization, and visualization
    # pipeline while selecting the cube-stack scene and output directory.
    if "--scene" not in sys.argv:
        sys.argv[1:1] = ["--scene", "cube"]
    raise SystemExit(main())
