from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WallDatasetConfig:
    action_angle_noise: float = 0.2
    action_step_mean: float = 1.0
    action_step_std: float = 0.4
    action_lower_bd: float = 0.2
    action_upper_bd: float = 1.8
    batch_size: int = 64
    device: str = "cuda"
    dot_std: float = 1.7
    border_wall_loc: int = 5
    fix_wall_batch_k: int | None = None
    fix_wall: bool = True
    fix_door_location: int = 30
    fix_wall_location: int = 32
    exclude_wall_train: str = ""
    exclude_door_train: str = ""
    only_wall_val: str = ""
    only_door_val: str = ""
    wall_padding: int = 20
    door_padding: int = 10
    wall_width: int = 6
    door_space: int = 4
    num_train_layouts: int = -1
    img_size: int = 65
    max_step: int = 1
    n_steps: int = 17
    n_steps_reduce_factor: int = 1
    size: int = 20000
    val_size: int = 10000
    train: bool = True
    repeat_actions: int = 1


class WallDataset:
    """Compatibility placeholder for legacy wall environment imports."""

    pass
