# Evidence Packs

**Source:** `modal_evidence/` — the reproduction's receipts: everything pulled off the Modal volume that was small enough to commit. Big artifacts (checkpoints, `plan_targets.pkl`) deliberately stayed on the volume.

Each pack is named `<experiment>-<YYYYMMDD>-<seq>/` and contains its own `README.md` (with results tables and sometimes Modal dashboard links), JSON summaries, raw logs, and — where the planner rendered media — contact-sheet PNGs (image grids of decoded reconstructions/rollouts; see [Predictor and Decoders](Predictor%20and%20Decoders.md) for why decodes exist).

## The four packs

### `medium-full-20260702-01` — Experiment 1 (baseline reproduction)

The three-way Medium comparison (authors' straightening vs DINO patch vs DINO CLS).
- `combined_results.json` — the headline success/loss table
- `latent_analysis.json` — geometry diagnostics
- `logs/<variant>/train.log`, `plan_gd_fixed.log` — full training/planning stdout; the `_fixed` suffix and `plans_fixed/` folder mean planning was **re-run after a planner bug-fix**, superseding the original plans
- `epoch20_validation_contact_sheet.png`, `epoch20_rollout_train0_contact_sheet.png` — visual sanity checks
- `local_artifact_manifest.json`, `archives/…manifest.json` — inventories of what exists where

### `medium-ablations-20260702-01` — Experiment 2 (frozen-DINO speed ablation)

The ablation that *couldn't* work, and the evidence proving it: `combined_medium_results.json`, per-variant `extracted/runs/<run-id>/` trees (note the checkpoint folder names — they're the self-documenting Hydra paths from [Hydra Configs](Hydra%20Configs.md)), and crucially `image_evidence_summary.json`, which documents that the patch-variant contact sheets are **byte-identical** across loss settings — the smoking gun that frozen features ignore representation losses ([The Reproduction Study](The%20Reproduction%20Study.md)).

### `medium-adapter-ablations-20260702-01` — Experiment 3 (the informative one)

Four `dino_channel` adapter variants (none / cos / speed / cos+speed).
- `adapter_results.json` (+ `partial_adapter_results.json` from before the spend-limit interruption was resolved)
- `pulled/medium-adapter-<variant>-summary_plans_fixed.json` and `-latent_analysis.json` — per-variant raw pulls
- `raw_volume/` — planner logs straight off the volume
- Its README records the study's caveats: the speed-only variant's diagnostics used epoch 18 / 20 rollouts after the 50-rollout job stalled.

### `wall-full-20260703-01` — Experiment 4 (Wall)

`wall_results.json` plus `pulled/` logs for both Wall runs (`wall-full-…` = straightened, `wall-dino-…` = plain), including the dataset-generation log. No images — the Wall planner path rendered no media, so evidence is numeric/log-only. Both variants: 0.02 success, identical distances.

## How to audit a claim

Every number in `report.md` / `research_note.tex` traces to a JSON here; the report's "Evidence Index" section is the map. Chain: **claim (report) → summary JSON (pack root) → raw pull (`pulled/`, `raw_volume/`, `logs/`) → run ID → Modal dashboard**. The run IDs double as [Modal Runner](Modal%20Runner.md) arguments, so any pack can in principle be regenerated with the commands listed at the end of `report.md`.
