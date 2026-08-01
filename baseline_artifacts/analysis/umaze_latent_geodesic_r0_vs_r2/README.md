# UMaze latent-geodesic correlation: R0 vs R2

## Executive summary

This experiment tests a proposed middle link in our hypothesis:

> Steadier latent pace → more honest latent distances → better planning.

In plain English: if the representation moves through latent space at a more consistent speed, perhaps the distance between two latent points will become a better estimate of how far the agent must actually travel.

For this test, **R0 performed better than R2**. R0's latent distance to the goal followed the true maze distance extremely closely (Spearman ρ = 0.9560). R2 also had a strong relationship, but it was weaker (ρ = 0.8565). The paired difference was −0.0995, and its 95% confidence interval did not include zero.

The main conclusion is:

> R2's previously observed UMaze planning behavior is not explained by a better global ordering of states according to direct Euclidean latent distance. Under this diagnostic, R0 represents distance-to-goal more faithfully.

This does **not** prove that R2 is useless, or that its speed term has no effect. It means that one specific proposed mechanism—improving global Euclidean latent distance—was not supported by this experiment.

## Question being tested

The practical question is whether a model's latent distance to the goal agrees with the actual number of moves required to reach that goal while respecting the maze walls.

For every valid UMaze grid state, we compare two quantities:

1. **True A* step-count:** the shortest number of legal grid moves from that state to the goal.
2. **Latent distance:** the Euclidean distance between the model's representation of that state and its representation of the goal.

A useful latent map should generally assign larger distances to states that are truly farther from the goal. Importantly, this test is about ordering states correctly. It does not require one latent unit to equal exactly one physical step.

## Conditions

- **R0 direction-only:** applies the direction/curvature penalty without explicitly penalizing changes in latent speed.
- **R2 full penalty:** combines direction and speed consistency. The R2 coefficient was 0.05, matching R0's effective direction weight.

The same rendered observations were passed through both checkpoints. Agent velocity was fixed at zero, and the planning-matched visual representation was used (alpha = 0).

## How the ground-truth distance is computed

The UMaze is represented as an occupancy grid. Free cells are traversable and wall cells are blocked. For each valid starting cell, the analysis computes a shortest path to the canonical goal using four-neighbor movement: up, down, left, or right.

The resulting A* step-count is our ground-truth maze distance. Here, “ground truth” does not mean an unknowable perfect description of the entire environment. It means the correct shortest-path distance under the grid model used by this diagnostic.

Why A* distance matters: two states can be physically close in a straight line but separated by a wall. Euclidean state distance may call them close, while A* correctly counts the detour around the wall.

## What Spearman correlation means

Spearman's rank correlation, ρ, asks whether the model orders states from near-to-far in the same way as A*.

- **ρ = 1:** perfect ordering. Every state ranked farther away by A* is also ranked farther away in latent space.
- **ρ = 0:** no consistent relationship between the rankings.
- **ρ = −1:** perfectly reversed ordering.

A high Spearman value does not mean latent distances have the correct numerical scale. For example, the model could multiply every latent distance by ten and preserve the same Spearman correlation. It measures ordering, not calibration.

## Results

| Condition | Spearman ρ | 95% bootstrap CI | p-value |
|---|---:|---:|---:|
| R0 direction-only | **0.9560** | [0.9401, 0.9679] | 7.01e−251 |
| R2 full penalty | **0.8565** | [0.8147, 0.8918] | 3.02e−136 |

Paired difference, ρ(R2) − ρ(R0): **−0.0995**, with a paired-state bootstrap 95% confidence interval of **[−0.1298, −0.0749]**.

The confidence interval is entirely below zero. This indicates that R2's lower correlation is a consistent difference across the sampled states, rather than an apparent difference easily explained by resampling noise.

## Plain-English interpretation

Both models learned meaningful maze geometry. R2's correlation of 0.8565 is still strong: states that are farther away in the maze usually remain farther away in its latent space.

However, R0 is substantially more faithful. Its 0.9560 correlation means its latent distance almost perfectly preserves the true near-to-far ordering of grid states. R2 makes more ranking mistakes—for example, it is more likely to represent a state requiring a longer detour as closer than another state with a shorter true path.

Therefore, **R0 is better for this particular notion of an “honest” latent map**: direct latent Euclidean distance to the goal more accurately reflects shortest-path distance.

## What this result supports

- R0 produces a highly path-aware global distance-to-goal representation on the tested UMaze grid.
- R2 still contains strong maze-distance information, but less than R0 under direct Euclidean readout.
- The difference between the checkpoints is statistically well separated in this state set.
- The proposed “R2 improves planning because Euclidean latent distance becomes more globally honest” explanation is not supported here.

## What this result does not prove

This experiment does not show that R2 is always worse, nor does it directly measure planning success. A representation can help a planner even if direct Euclidean distance is not its best global distance metric.

Possible alternatives include:

1. **Local versus global geometry.** R2 may regularize the size of consecutive latent steps along trajectories while making direct long-range distances less faithful.
2. **Latent shortcuts.** Euclidean distance draws a straight line through latent space. That line may cross regions corresponding to walls or unreachable states, just as a straight line in the physical maze can cross a wall.
3. **Optimization benefits.** A smoother or steadier trajectory could make the planner's optimization easier without improving global distance-to-goal correlation.
4. **A tradeoff introduced by the speed term.** Enforcing similar latent step sizes may sacrifice some global spatial ordering while improving another property used during planning.
5. **Rank versus calibration.** This test measures whether states are ordered correctly, not whether changes in latent distance correspond proportionally to changes in physical distance.

## Limitations

- The comparison uses one R0 checkpoint and one R2 checkpoint. Multiple training seeds are needed to separate a penalty effect from checkpoint-to-checkpoint variation.
- The analysis uses the canonical goal. Repeating it across many goals would test whether the conclusion generalizes across the maze.
- A* operates on a discretized four-neighbor grid, while the original environment has continuous states and dynamics.
- Agent velocity is fixed at zero, so the test isolates visual position geometry but does not evaluate velocity-dependent representations.
- Spearman correlation ignores absolute scale and local smoothness.
- Direct Euclidean latent distance may be the wrong global readout if the learned manifold bends around obstacles.

## Recommended next experiments

### 1. Test the speed claim directly

Along held-out trajectories, compare the length of each latent step with the corresponding physical displacement. Measure:

- variation in latent step length;
- correlation between latent step length and physical displacement;
- speed-ratio error between consecutive steps.

This determines whether R2 actually produces a steadier latent pace, rather than inferring that property from planning results.

### 2. Compare latent graph-geodesic distance with A*

Build a graph connecting nearby valid states in latent space, then compute shortest-path distance over that graph. This prevents a direct Euclidean line from taking an unrealistic shortcut across a wall or a gap in the learned manifold.

If R2 improves local geometry but not global Euclidean geometry, graph-geodesic distance could reveal that advantage.

### 3. Break results down by region and distance

Report correlations separately for:

- states near versus far from the goal;
- states on opposite sides of the central wall;
- states whose optimal path requires a major turn or detour.

This can reveal where R2 loses ordering accuracy.

### 4. Repeat across goals and seeds

Use several goal locations and multiple independently trained checkpoints. Report the distribution of R2 − R0 differences, not only one paired comparison.

### 5. Connect geometry to actual planning

For each start state, compare latent-distance error with planning success, final state distance, and proprioceptive distance. This tests whether the geometric differences observed here actually predict downstream behavior.

## Reproduction and implementation

The complete analysis implementation is:

- **scripts/evaluate_umaze_latent_geodesic.py**

Important functions include:

- **astar_step_counts()**: computes shortest four-neighbor step-counts to the goal;
- **build_grid_states()**: creates the valid UMaze state grid;
- **latent_distances_to_goal()**: calculates checkpoint latent distances;
- **bootstrap_spearman_ci()**: estimates uncertainty for each correlation;
- **paired_bootstrap_delta_ci()**: compares R2 and R0 on the same states;
- **write_markdown()**: writes this report.

The end-to-end training pipeline invokes the analysis from:

- **scripts/run_umaze_q1_retrain.sh**

Analysis settings recorded in **results.json** include 5,000 bootstrap samples and seed 20260723.

## Artifacts

- **grid_latent_distances.csv**: per-state A* distance and R0/R2 latent-distance measurements
- **results.json**: exact statistics, confidence intervals, checkpoints, seed, and run settings
- **latent_vs_astar_scatter.png**: latent distance plotted against true A* distance
- **distance_fields.png**: true and learned distance maps over the maze
- **analysis.log**: execution log

## Bottom line

R2 did not make direct global latent distances more faithful to true UMaze path distance. R0 was clearly better under this metric. The most useful next step is to test whether R2 improves **local latent pacing** or **latent graph-geodesic distance**, which would provide a different mechanism for any observed planning benefit.
