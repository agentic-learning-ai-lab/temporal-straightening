from __future__ import annotations

from collections.abc import Iterable

from .wall import WallDatasetConfig


def _parse_int_filter(value: str | Iterable[int] | None) -> set[int]:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        return {int(part) for part in value.split(",") if part.strip()}
    return {int(part) for part in value}


def _candidate_values(fixed: bool, fixed_value: int, lower: int, upper: int) -> list[int]:
    if fixed:
        return [int(fixed_value)]
    return list(range(int(lower), int(upper) + 1))


def generate_wall_layouts(config: WallDatasetConfig):
    """Enumerate wall/door layouts expected by DotWall.

    The released wrapper references this helper but the data package is absent in
    the checkout. The fixed-layout default matches `wall_single`; the generalized
    branch keeps the wrapper usable for randomized layouts.
    """

    half_width = config.wall_width // 2
    wall_min = config.border_wall_loc + half_width + 1
    wall_max = config.img_size - config.border_wall_loc - half_width - 1
    door_min = config.border_wall_loc + config.door_padding
    door_max = config.img_size - config.border_wall_loc - config.door_padding

    wall_values = _candidate_values(
        config.fix_wall, config.fix_wall_location, wall_min, wall_max
    )
    door_values = _candidate_values(
        True, config.fix_door_location, door_min, door_max
    )

    excluded_walls = _parse_int_filter(config.exclude_wall_train)
    excluded_doors = _parse_int_filter(config.exclude_door_train)
    only_walls = _parse_int_filter(config.only_wall_val)
    only_doors = _parse_int_filter(config.only_door_val)

    layouts = {}
    other_layouts = {}
    for wall_pos in wall_values:
        if wall_pos in excluded_walls:
            continue
        if only_walls and wall_pos not in only_walls:
            continue
        for door_pos in door_values:
            if door_pos in excluded_doors:
                continue
            if only_doors and door_pos not in only_doors:
                continue
            key = f"wall{wall_pos}_door{door_pos}"
            layouts[key] = {"wall_pos": int(wall_pos), "door_pos": int(door_pos)}

    if config.num_train_layouts is not None and config.num_train_layouts > 0:
        selected = list(layouts.items())[: config.num_train_layouts]
        held_out = list(layouts.items())[config.num_train_layouts :]
        layouts = dict(selected)
        other_layouts = dict(held_out)

    if not layouts:
        key = f"wall{config.fix_wall_location}_door{config.fix_door_location}"
        layouts[key] = {
            "wall_pos": int(config.fix_wall_location),
            "door_pos": int(config.fix_door_location),
        }
    return layouts, other_layouts

