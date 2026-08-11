# Hydra Configs

**Source:** `conf/` — the wiring diagram for every experiment.

The repo uses [Hydra](https://hydra.cc): experiments are described by **composing YAML fragments**, and any value can be overridden from the command line. When you read a training command like

```bash
python train.py --config-name train.yaml env=point_maze encoder=dino_channel training.straighten=aggcos1e-1
```

each `key=value` swaps in a different YAML fragment or overrides a leaf value. Code never hardcodes *which* encoder/planner/env to use — YAML files carry a `_target_:` line naming the Python class, and Hydra instantiates it. **The config tree mirrors the code tree almost one-to-one**, which makes it a great index of the system:

```
conf/
├── train.yaml            ← training entry point (defaults, lrs, straighten flag)
├── plan_gd.yaml          ← the three planning entry points
├── plan_cem.yaml            (see Planning - plan.py)
├── plan_gd_mpc.yaml
├── env/                  ← which world + dataset loader        → [Environments](Environments.md)
│   point_maze / point_maze_medium / pusht / wall / rope / granular / deformable_env
├── encoder/              ← the model's eyes                    → [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md)
│   dino / dino_cls / dino_channel / dino_global / r3m /
│   resnet / scratch_resnet / scratch_resnet_spatial / dummy
├── predictor/vit.yaml    ← the imagination                     → [Predictor and Decoders](Predictor%20and%20Decoders.md)
├── decoder/              ← vqvae / transposed_conv             → [Predictor and Decoders](Predictor%20and%20Decoders.md)
├── planner/              ← gd / cem / mpc_gd / mpc_cem         → [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md)
├── action_encoder/       ← proprio / dummy
└── proprio_encoder/      ← proprio / dummy
```

## `train.yaml` — the values that define the paper's setup

- `frameskip: 5`, `num_hist: 3`, `num_pred: 1` — the temporal window everything assumes
- `img_size: 224` — fixes the 14×14 DINO patch grid
- `training.straighten: False` — the string DSL slot ([The Straightening Loss](The%20Straightening%20Loss.md))
- `training.stop_grad: True` — the anti-collapse default ([The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md))
- per-module learning rates; `epochs: 20`; bf16 mixed precision
- `plan_settings` — the (disabled-by-default) mid-training planning evals

## The self-documenting checkpoint path

The `hydra.run.dir` template builds the output directory name out of the experiment's own parameters:

```
<env>_<straighten>_agg32_proj<projector>_dim<dim>_hw<hw>_sg<stop_grad>_lr<lr>
e.g.  medium_cos1e-1+aggspeed1e-1_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
```

Every checkpoint folder name is a fingerprint of its config — you can reconstruct what a run was from `ls` alone (you'll see these names inside [Evidence Packs](Evidence%20Packs.md)). The custom OmegaConf resolvers in `custom_resolvers.py` (`replace_slash`, `replace_substring`) exist purely to make these strings path-safe.

`models/encoder/r3m/cfgs/` is a separate mini-Hydra-tree vendored with R3M — ignore it.
