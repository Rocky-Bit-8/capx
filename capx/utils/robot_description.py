"""Local robot-description loaders used by simulation and visualization."""

from __future__ import annotations

from pathlib import Path

import yourdfpy


def load_local_panda_description() -> yourdfpy.URDF:
    """Load the checked-in Panda URDF without accessing the network."""
    asset_dir = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "robosuite"
        / "robosuite"
        / "models"
        / "assets"
        / "bullet_data"
        / "panda_description"
    )
    urdf_path = asset_dir / "urdf" / "panda_arm_hand.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Checked-in Panda URDF not found: {urdf_path}")
    return yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(asset_dir))
