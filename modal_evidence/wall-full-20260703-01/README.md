# Wall Full Runs

This folder contains the local evidence pulled from Modal for the Wall environment retry.

Completed runs:

- `wall-full-20260703-01`: authors-style DINO patch temporal straightening with `cos1e-1`.
- `wall-dino-20260703-01`: DINO patch baseline with straightening disabled.

Both runs used 50 random-policy episodes, 100 frames per episode, 20 training epochs, and gradient-descent planning with 50 evaluations and goal horizon 25.

Main result: both Wall variants reached planner success rate 0.02 and mean state distance 33.4643. DINO patch without straightening trained to a lower validation loss, but did not improve the planner outcome.

The Wall planner path produced logs and `plan_targets.pkl` on the Modal volume, but did not render image/contact-sheet media. Only compact summaries and logs were pulled into this PR.

See `wall_results.json` for the consolidated result and `pulled/` for raw Modal summaries and logs.
