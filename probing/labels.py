"""Ground-truth probe targets from UMaze / PointMaze state tensors."""

from __future__ import annotations

import torch


def state_tensor_to_probe_labels(
    states: torch.Tensor,
    *,
    min_speed: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """
    Build probe targets from raw simulator state (not normalized proprio).

    Args:
        states: ``(T, 4)`` or ``(B, T, 4)`` with columns
            ``[qpos_x, qpos_y, qvel_x, qvel_y]``.
        min_speed: speeds below this get direction set to NaN (excluded in training).

    Returns:
        Dict of float tensors with trailing shape ``(T,)`` or ``(B, T)``:
        ``position_x``, ``position_y``, ``speed``, ``direction_cos``, ``direction_sin``.
    """
    if states.ndim == 2:
        states = states.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    pos = states[..., :2]
    vel = states[..., 2:4]
    speed = vel.norm(dim=-1)

    direction_cos = vel[..., 0] / (speed + 1e-8)
    direction_sin = vel[..., 1] / (speed + 1e-8)
    low_speed = speed < min_speed
    direction_cos = direction_cos.masked_fill(low_speed, float("nan"))
    direction_sin = direction_sin.masked_fill(low_speed, float("nan"))

    labels = {
        "position_x": pos[..., 0],
        "position_y": pos[..., 1],
        "speed": speed,
        "direction_cos": direction_cos,
        "direction_sin": direction_sin,
    }
    if squeeze:
        labels = {k: v.squeeze(0) for k, v in labels.items()}
    return labels


def labels_to_matrix(labels: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    """Stack selected label keys into ``(N, len(keys))``."""
    return torch.stack([labels[k] for k in keys], dim=-1)


def valid_direction_mask(labels: dict[str, torch.Tensor]) -> torch.Tensor:
    """Boolean mask for frames with reliable heading (non-NaN cos/sin)."""
    return torch.isfinite(labels["direction_cos"]) & torch.isfinite(
        labels["direction_sin"]
    )
