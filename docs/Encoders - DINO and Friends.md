# Encoders — DINO and Friends

**Sources:** `models/dino.py`, `models/encoder/` (vit.py, resnet.py, r3m/), configs in `conf/encoder/`

The encoder is the model's eyes: image in, latent tokens out. The paper's experiments are largely *encoder comparisons*, so this family tree is the key to reading every results table.

## DinoV2Encoder (`models/dino.py`) — the main character

Wraps a **frozen** DINOv2 ViT from torch.hub. DINOv2 is Meta's self-supervised vision transformer — its features are excellent at "what is where" without any task-specific training. Two output flavors, chosen by `feature_key`:

- **`x_norm_patchtokens`** → a 14×14 grid of 384-dim vectors, one per image patch. Spatially detailed. This is "DINO patch" in the tables (config `dino.yaml`).
- **`x_norm_clstoken`** → a single 384-dim summary vector for the whole image. This is "DINO CLS" (config `dino_cls.yaml`). Fun fact from [The Reproduction Study](The%20Reproduction%20Study.md): this simplest option planned best on Medium.

### Projectors — the trainable adapters

Because the backbone is frozen, straightening can only reshape latent geometry if *something* trainable sits between DINO and the measured latents. Two options in `dino.py`:

- **`ChannelProjector`** (config `dino_channel.yaml`): 1×1 convolutions over the patch grid — remixes the 384 channels per patch but keeps the 14×14 spatial layout. Think "trainable color filter for feature channels." This is the **adapter** in the reproduction's decisive ablation.
- **`GlobalProjector`** (config `dino_global.yaml`): a small conv pyramid that downsamples 14×14 → pooled → a single global vector (or a 2×2 grid). Trades spatial detail for a compact planning-friendly summary.

### The `agg()` pooling head

For `aggcos`/`aggspeed` losses ([The Straightening Loss](The%20Straightening%20Loss.md)), the encoder must pool its patch grid into one vector per frame. `agg_type` picks how: `mean` (average patches), `flatten` (concatenate all — huge), or `mlp` (flatten, then a 3-layer MLP + LayerNorm, configurable via `agg_out_dim`/`agg_mlp_hidden_dim`).

## The supporting cast

- **`models/encoder/resnet.py`** — ResNet-18-style CNNs trained *from scratch* (no pretraining): `resnet18` produces one global vector (config `scratch_resnet.yaml`); `ResNetSpatial` keeps a spatial feature map, mimicking patch tokens (config `scratch_resnet_spatial.yaml`). These test whether straightening helps when the encoder is fully trainable end-to-end.
- **`models/encoder/vit.py`** — a from-scratch ViT encoder alternative.
- **`models/encoder/r3m/`** — Meta's **R3M** (robot-pretrained ResNet features), vendored wholesale with its own utils/configs. Only touched via `conf/encoder/r3m.yaml`; not worth a deep dive.
- **`models/dummy.py`** — `DummyModel`/`DummyRepeatActionEncoder`: identity/no-op stand-ins so configs can "turn off" a slot (e.g. `action_encoder=dummy`) without special-casing the code.

## Why the frozen/trainable distinction runs the whole story

| Config | Backbone | Trainable path to latents? | Can straightening reshape geometry? |
|---|---|---|---|
| `dino` | frozen | none | **No** — loss is measured but impotent |
| `dino_cls` | frozen | none | **No** |
| `dino_channel` | frozen | ChannelProjector (+agg) | **Yes** — the adapter ablation |
| `dino_global` | frozen | GlobalProjector | Yes |
| `scratch_resnet(_spatial)` | trained | everything | Yes |

The reproduction's frozen-DINO ablation produced *bit-identical latent diagnostics* across all loss settings — the table above explains why, and why the adapter ablation was needed. See [The Reproduction Study](The%20Reproduction%20Study.md).
