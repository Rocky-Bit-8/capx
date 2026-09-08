import numpy as np
import pytest
import viser.transforms as vtf

from capx.integrations.franka.nut_assembly_visual import (
    FrankaControlNutAssemblyVisualApi,
)
from capx.integrations.motion.curobo import init_curobo


def test_refine_square_nut_handle_mask_extracts_narrow_end() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[15:55, 20:60] = True
    mask[27:43, 32:48] = False
    mask[55:75, 33:47] = True

    refined = FrankaControlNutAssemblyVisualApi._refine_square_nut_handle_mask(mask)

    ys, xs = np.nonzero(refined)
    assert len(xs) < np.count_nonzero(mask) // 2
    assert np.median(ys) > 60
    assert 33 <= np.median(xs) <= 46


def test_refine_square_nut_handle_mask_keeps_handle_only_mask() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 15:25] = True

    refined = FrankaControlNutAssemblyVisualApi._refine_square_nut_handle_mask(mask)

    np.testing.assert_array_equal(refined, mask)


def test_fallback_handle_grasp_deprojects_surface_to_handle_center() -> None:
    api = object.__new__(FrankaControlNutAssemblyVisualApi)
    depth = np.ones((3, 3), dtype=np.float32)
    intrinsics = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    obs = {
        "robot0_robotview": {
            "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        }
    }

    position, quaternion = api._fallback_handle_grasp(obs, depth, intrinsics, mask)

    np.testing.assert_allclose(position, [0.0, 0.0, 0.99])
    np.testing.assert_array_equal(quaternion, [0.0, 0.0, 1.0, 0.0])


def test_api_pose_to_panda_hand_pose_accounts_for_robosuite_eef_offset() -> None:
    api = object.__new__(FrankaControlNutAssemblyVisualApi)
    api_position = np.array([0.42, -0.11, 0.23], dtype=np.float64)
    api_quaternion = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)

    hand_position, hand_quaternion = api._api_pose_to_panda_hand_pose(
        api_position, api_quaternion
    )
    hand = vtf.SE3.from_rotation_and_translation(
        rotation=vtf.SO3(wxyz=hand_quaternion), translation=hand_position
    )
    reconstructed_api = hand @ api._PANDA_HAND_TO_ROBOSUITE_EEF @ api._API_FRAME_FLIP

    np.testing.assert_allclose(reconstructed_api.translation(), api_position, atol=1e-8)
    assert (
        FrankaControlNutAssemblyVisualApi._orientation_error_deg(
            reconstructed_api.rotation().wxyz, api_quaternion
        )
        < 1e-6
    )


def test_curobo_client_rejects_unsolved_ik(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "capx.integrations.motion.curobo.post_with_retries",
        lambda *_args, **_kwargs: {"success": False, "joint_positions": [0.0] * 7},
    )

    with pytest.raises(RuntimeError, match="could not solve IK"):
        init_curobo("http://unused")(
            np.array([1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.3])
        )
