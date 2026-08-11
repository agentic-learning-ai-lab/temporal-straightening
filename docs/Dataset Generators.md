# Dataset Generators

**Sources:** `generate_point_maze_medium.py`, `generate_wall_dataset.py` — both are reproduction-study additions (Layer 2).

The paper used DINO-WM's published datasets (OSF download). The reproduction couldn't (or chose not to) rely on those for Medium and Wall, so it wrote generators that produce data in the **exact same on-disk format** the loaders in [Datasets and Data Loading](Datasets%20and%20Data%20Loading.md) expect:

```
states.pth        (N, T, state_dim)   float64
actions.pth       (N, T, action_dim)  float64
seq_lengths.pth   (N,)                int64
obses/episode_XXX.pth   (T, 224, 224, 3) uint8   — rendered frames
```

## `generate_point_maze_medium.py`

Builds the Medium maze via `PointMazeWrapper(MEDIUM_MAZE)` and rolls episodes with either a random policy or the **waypoint controller** (the scripted value-iteration expert from `env/pointmaze/` — see [Environments](Environments.md)).

The docstring is a small masterclass in reproduction hygiene: the authors *verified the format against the published UMaze dataset by replaying its recorded actions through the environment and confirming zero state error*. That established the exact alignment convention — `actions[t]` is computed at `states[t]`, and only the first T−1 actions are ever executed — before generating new data. It checkpoints progress periodically (`save_checkpoint`) so a long render job can resume.

Rendering is headless MuJoCo via EGL, which is why [Setup and Dependencies](Setup%20and%20Dependencies.md) fusses about NVIDIA EGL libraries.

## `generate_wall_dataset.py`

Same job for the Wall environment ("a wall_single dataset compatible with `conf/env/wall.yaml`"). Its distinguishing feature is **shape canonicalization** (`_visual_to_uint8_hwc`, `_canonical_visual`): the Wall env emitted visual observations with inconsistent shapes/layouts (CHW vs HWC, varying sizes), which crashed training. The generator forces every frame to a fixed-size uint8 height-width-channel array. This plus the `WallDataset` import shim were the two "code compatibility fixes" the Wall section of [The Reproduction Study](The%20Reproduction%20Study.md) mentions.

## Why 50 × 100 random episodes matters for interpretation

The reproduction's runs used **50 random-policy episodes of 100 frames** — tiny compared to the paper's datasets (thousands of episodes). Random policies also cover the state space thinly (a random walk rarely traverses a maze). Both caveats should be front of mind when reading the absolute success numbers (0.14/0.22 on Medium, 0.02 on Wall) in [Evidence Packs](Evidence%20Packs.md): the reproduction compares variants *against each other* under identical small-data conditions; it does not reproduce the paper's absolute performance.
