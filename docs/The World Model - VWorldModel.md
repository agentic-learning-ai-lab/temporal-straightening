# The World Model — VWorldModel

**Source:** `models/visual_world_model.py` (493 lines) — *the most important file in the repo.*

`VWorldModel` is the hub every other module plugs into. It owns five sub-networks and defines the training loss and the imagination rollout.

## The five sub-networks

| Slot | What it does | Default | Note |
|---|---|---|---|
| `encoder` | image → latent tokens | frozen DINOv2 | [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md) |
| `proprio_encoder` | robot's own state → small embedding | sin/cos MLP | [Predictor and Decoders](Predictor%20and%20Decoders.md) |
| `action_encoder` | action → small embedding | same | |
| `predictor` | past tokens → next tokens | ViT | [Predictor and Decoders](Predictor%20and%20Decoders.md) |
| `decoder` | tokens → image (for humans) | VQ-VAE | trained on **detached** latents |

## How a state is represented

`encode()` produces, per frame, a grid of visual tokens (e.g. 14×14 = 196 for DINO patches) and then **staples the action and proprioception on**. With the default `concat_dim=1`, the action/proprio embeddings are tiled across all 196 tokens and concatenated along the feature dimension — every token knows what the robot did. (`concat_dim=0` instead appends them as two extra tokens.)

This is why helper slicers exist and show up everywhere:
- `separate_emb(z)` → splits back into (visual, proprio) and action parts
- `visual_only(z)` → strips action+proprio dims (used by the straightening losses)
- `replace_actions_from_z(z, act)` → overwrites the action dims — **this is the planner's steering wheel**: during imagination, candidate actions are injected into the latent this way.

## The forward pass (training)

For a window of 4 frames (`num_hist=3` history + `num_pred=1` future):

1. Encode all 4 frames → `z`.
2. `z_src` = frames 0–2, `z_tgt` = frames 1–3.
3. Predictor maps `z_src` → `z_pred` (its prediction of frames 1–3).
4. **Main loss**: MSE between `z_pred` and `z_tgt`, excluding the action dims (predicting your own future actions would be cheating — actions are inputs, not observations).
5. **Anti-collapse**: `z_tgt` is `.detach()`-ed by default (`stop_grad=True`). Without this, the encoder could minimize prediction error by encoding *everything to the same point* — perfect predictability, zero information. Stop-grad breaks that shortcut, the same trick BYOL/SimSiam use. VICReg regularization (`vcreg*`) is the supported alternative.
6. **Optional regularizers**: [The Straightening Loss](The%20Straightening%20Loss.md) (curvature and/or speed constancy) on `visual_only(z)`.
7. **Decoder losses**: reconstruction of both raw latents and predicted latents, computed on detached tensors so decoding quality never distorts the representation. It's a window, not a hand on the wheel.

Returns `(z_pred, visual_pred, visual_reconstructed, loss, loss_components)` — the components dict is what shows up as separate curves in the training logs.

## `rollout()` — the imagination engine

Used by every planner. Given initial observations and a full action sequence:

```
z ← encode(obs_0, first actions)
repeat:
    z_next ← predictor(last num_hist frames of z)     # dream one step
    inject the next planned action into z_next        # replace_actions_from_z
    append z_next
```

No simulator is touched. The output latent trajectory is fully differentiable with respect to the injected actions — that's the property [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md) exploit.

## The loss-flag parser

`training.straighten` arrives as a string like `"cos1e-1+aggspeed1e-1"` and is tokenized in `__init__` into `(curvature_mode, straighten_scale)` and `(speed_constancy_mode, speed_constancy_scale)`. It also accepts the reproduction study's R0--R4 trajectory-penalty tokens, including pooled forms such as `aggr2_5e-2` and the tunable R3 form `aggr3b1_1e-1`. Full DSL table in [The Straightening Loss](The%20Straightening%20Loss.md). The `speed`/`aggspeed` and R0--R4 branches are additions to this otherwise paper-original file.
