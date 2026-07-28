# Predictor and Decoders

**Sources:** `models/vit.py`, `models/vqvae.py`, `models/decoder/transposed_conv.py`, `models/proprio.py`, `models/dummy.py`

## The predictor: `ViTPredictor` (`models/vit.py`)

The imagination itself — the network that answers *"given the last 3 mental snapshots (with actions stapled on), what does the next snapshot look like?"*

It's a standard transformer with one crucial modification: a **block-causal attention mask** (`generate_mask_matrix`). The input is the token grids of `num_hist` frames laid end-to-end (3 × 196 tokens). The mask lets every token attend to *all tokens of its own frame and earlier frames*, but never to future frames. Like a flip-book reader who can study pages 1–3 in full but must guess page 4 — within a page, everything is visible; across pages, time flows one way.

The output is the same shape as the input: the token grid shifted one frame into the future. `VWorldModel.predict()` handles the reshaping (`b t p d ↔ b (t·p) d`) around it.

Supporting classes (`Attention`, `FeedForward`, `Transformer`) are textbook ViT blocks with the mask injected.

## The decoders: for humans, not for the model

Latents are for planning; decoders exist so *people* can inspect them. They're trained on **detached** latents ([The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md)) — pure spectators. If the decoder can reconstruct a legible image from `z`, the latent demonstrably retains the scene; if predicted-latent decodes look right, the predictor is dreaming coherently. The contact-sheet PNGs in [Evidence Packs](Evidence%20Packs.md) are exactly these decodes.

- **`models/vqvae.py`** — the default (config `decoder/vqvae.yaml`). A Vector-Quantized VAE: latents pass through a `Quantize` layer that snaps each vector to the nearest entry of a learned codebook (like reducing a photo to a fixed palette), then a conv decoder paints the image. The "vq loss" / `diff` terms in training logs are the codebook commitment costs. `ProjectorDecoder` adapts mismatched projector dims back into the decoder's expected shape.
- **`models/decoder/transposed_conv.py`** — a simpler alternative (config `decoder/transposed_conv.yaml`): straight transposed-convolution upsampling, no codebook, optional distribution-style output helpers.

## The small embeddings: `models/proprio.py`

`ProprioceptiveEmbedding` maps low-dimensional vectors (joint positions, actions) into `proprio_emb_dim`/`action_emb_dim`-sized embeddings (default 10), with optional 1-D sinusoidal position codes. Both the `proprio_encoder` and `action_encoder` slots use this class (config `proprio_encoder/proprio.yaml`, `action_encoder/proprio.yaml`). These embeddings are what `VWorldModel.encode()` tiles onto the visual tokens.

`models/dummy.py` provides no-op stand-ins for any of these slots (see [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md)).

## Sizing note

`conf/train.yaml` fixes `img_size=224`; with DINO's patch size 14 that yields the 14×14=196-token grid everything else assumes (e.g. the hardcoded `196` in the encoder's MLP agg head). The VQ-VAE's stride stack likewise assumes 224 — hence the `decoder_scale = 16` resize logic in `VWorldModel.__init__`.
