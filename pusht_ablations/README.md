# PushT trajectory-penalty ablation

Speed-sensitive latent trajectory penalties on PushT, matching the wall/umaze
ablation protocol (`origin/agent/umaze-trajectory-penalty-ablation`).

## Conditions

| Folder | Token | Objective |
|---|---|---|
| `r0_direction_only/` | `aggcos1e-1` | 0.1 × (1 − cos θ) |
| `r1_speed_only/` | `aggr1_1e-1` | 0.1 × R1 |
| `r2_full_matched/` | `aggr2_5e-2` | 0.05 × R2 |
| `r3_beta1/` | `aggr3b1_1e-1` | 0.1 × (R0 + R1) |

All conditions use `dino_channel` (14×14×8), 2 training epochs, open-loop GD
planning with `objective.alpha=1`.

## Layout (same style as `pusht_reproduction/`)

```
pusht_ablations/
├── RESULTS.txt              # summary table (regenerate with make_results.py)
├── make_results.py          # parse plan_logs → RESULTS.txt + per-condition txt
├── r0_direction_only/
│   ├── plan_logs/           # gd_seed{100,200,300}_chunk{00..40}.log
│   └── r0_direction_only_results_3seed.txt
├── r1_speed_only/plan_logs/
├── r2_full_matched/plan_logs/
└── r3_beta1/plan_logs/
```

## R0 provenance

R0 is the straightening baseline (`aggcos1e-1`). Its `plan_logs/` were copied
from `pusht_reproduction/straightening/` — same model, same planning protocol.
R1–R3 are trained separately via `run_pusht_ablation_a100.sh`.

## Regenerate summaries

```bash
python pusht_ablations/make_results.py
```

## Run new conditions

```bash
bash run_pusht_ablation_a100.sh train
bash run_pusht_ablation_a100.sh plan
python pusht_ablations/make_results.py
```
