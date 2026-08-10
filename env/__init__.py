from gym.envs.registration import register

# mujoco_py compiles a C extension on first import and fails on machines
# without GL headers. Wall/PushT need neither mujoco nor the maze specs, so
# degrade to skipping the point_maze registrations instead of crashing.
try:
    from .pointmaze import U_MAZE, MEDIUM_MAZE
except Exception as _pointmaze_import_error:
    import warnings

    U_MAZE = MEDIUM_MAZE = None
    warnings.warn(
        "point_maze environments unavailable "
        f"({type(_pointmaze_import_error).__name__}: {_pointmaze_import_error}); "
        "skipping their gym registration."
    )
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
if U_MAZE is not None:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )

    register(
        id="point_maze_medium",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=600,
        kwargs={
            "maze_spec": MEDIUM_MAZE,
            "reward_type": "sparse",
            "reset_target": False,
            "ref_min_score": 13.13,
            "ref_max_score": 277.39,
            "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5",
        },
    )

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)