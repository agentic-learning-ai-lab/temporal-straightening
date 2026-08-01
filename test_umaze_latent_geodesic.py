import numpy as np

from scripts.evaluate_umaze_latent_geodesic import (
    astar_step_counts,
    build_grid_states,
    latent_distances_to_goal,
    make_occupancy_grid,
    nearest_valid_node,
)


def test_umaze_grid_is_connected_and_goal_has_zero_distance():
    _, valid = make_occupancy_grid(spacing=0.2)
    goal = nearest_valid_node((1.0, 1.0), valid)
    steps = astar_step_counts(valid, goal)

    assert set(steps) == set(valid)
    assert steps[goal] == 0
    assert max(steps.values()) > 0


def test_astar_routes_around_u_shape_instead_of_through_wall():
    states, goal = build_grid_states(0.5, 3.1, 0.1, (1.0, 1.0))
    right_bottom = min(
        states,
        key=lambda state: (state.x - 3.0) ** 2 + (state.y - 1.0) ** 2,
    )
    direct_manhattan = abs(right_bottom.ix - goal[0]) + abs(right_bottom.iy - goal[1])

    assert right_bottom.astar_steps > direct_manhattan


def test_latent_distance_uses_requested_goal_embedding():
    states, goal = build_grid_states(0.5, 3.1, 0.2, (1.0, 1.0))
    embeddings = np.arange(len(states) * 3, dtype=np.float64).reshape(len(states), 3)
    distances = latent_distances_to_goal(embeddings, states, goal)
    goal_index = next(
        i for i, state in enumerate(states) if (state.ix, state.iy) == goal
    )

    assert distances[goal_index] == 0.0
    assert np.all(distances >= 0.0)
