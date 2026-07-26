from types import SimpleNamespace

import pytest
import torch

from models.visual_world_model import VWorldModel


class DummyEncoder(torch.nn.Module):
    emb_dim = 2
    name = "dummy"

    def agg(self, tokens):
        return tokens.mean(dim=1)


def model_with_penalty(token):
    return VWorldModel(
        image_size=8,
        num_hist=2,
        num_pred=1,
        encoder=DummyEncoder(),
        proprio_encoder=None,
        action_encoder=None,
        decoder=None,
        predictor=None,
        straighten=token,
    )


def penalties(points, *, beta=1.0):
    features = torch.tensor(points, dtype=torch.float64).unsqueeze(0)
    model = SimpleNamespace()
    return VWorldModel.trajectory_penalties(model, features, beta=beta)


def test_constant_velocity_has_zero_penalty():
    result = penalties([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert all(value.item() == pytest.approx(0.0) for value in result.values())


def test_right_angle_at_constant_speed_is_direction_only():
    result = penalties([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert result["R0"].item() == pytest.approx(1.0)
    assert result["R1"].item() == pytest.approx(0.0)
    assert result["R2"].item() == pytest.approx(2.0)
    assert result["R3"].item() == pytest.approx(1.0)
    assert result["R4"].item() == pytest.approx(2.0)


def test_speed_doubling_is_symmetric_ratio_penalty():
    result = penalties([[0.0], [1.0], [3.0]], beta=2.0)
    assert result["R0"].item() == pytest.approx(0.0)
    assert result["R1"].item() == pytest.approx(0.5)
    assert result["R2"].item() == pytest.approx(0.5)
    assert result["R3"].item() == pytest.approx(1.0)
    assert result["R4"].item() == pytest.approx(1.0)


def test_r2_identity_and_scale_invariance():
    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.5, 2.0], [3.0, 2.5]]],
        dtype=torch.float64,
    )
    model = SimpleNamespace()
    base = VWorldModel.trajectory_penalties(model, points, beta=0.25)
    scaled = VWorldModel.trajectory_penalties(model, points * 7.0, beta=0.25)
    assert base["R2"].item() == pytest.approx(
        (base["R1"] + 2.0 * base["R0"]).item()
    )
    for name in ("R0", "R1", "R2", "R3"):
        assert scaled[name].item() == pytest.approx(base[name].item())
    assert scaled["R4"].item() == pytest.approx(49.0 * base["R4"].item())


def test_stationary_steps_are_ignored_without_nan():
    result = penalties([[1.0], [1.0], [1.0]])
    assert all(torch.isfinite(value) and value.item() == 0.0 for value in result.values())


@pytest.mark.parametrize(
    ("token", "mode", "scale", "beta"),
    [
        ("aggr1_1e-1", "r1", 0.1, 1.0),
        ("aggr2_1e-1", "r2", 0.1, 1.0),
        ("aggr3b0.25_1e-1", "r3", 0.1, 0.25),
        ("aggr4_1e-1", "r4", 0.1, 1.0),
    ],
)
def test_penalty_tokens_configure_aggregated_modes(token, mode, scale, beta):
    model = model_with_penalty(token)
    assert model.trajectory_penalty
    assert model.trajectory_penalty_aggregate
    assert model.trajectory_penalty_mode == mode
    assert model.trajectory_penalty_scale == pytest.approx(scale)
    assert model.trajectory_penalty_beta == pytest.approx(beta)


def test_aggregate_mode_pools_patches_before_velocity():
    model = model_with_penalty("aggr1_1e-1")
    features = torch.tensor(
        [[[[0.0], [2.0]], [[1.0], [3.0]], [[3.0], [5.0]]]],
        dtype=torch.float64,
    )
    result = model.trajectory_penalties(features, aggregate=True)
    assert result["R0"].item() == pytest.approx(0.0)
    assert result["R1"].item() == pytest.approx(0.5)
