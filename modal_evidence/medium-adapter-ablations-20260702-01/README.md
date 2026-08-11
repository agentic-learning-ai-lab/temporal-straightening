# Medium Adapter Ablations

These runs were launched to address the caveat in the frozen-DINO speed ablation. The variants use `encoder=dino_channel`, so the DINO backbone stays frozen but the projected latent path includes trainable projector and aggregation parameters.

## Completed retry

After the Modal spend limit was resolved, all four adapter variants completed planner evaluation and latent diagnostics. The consolidated evidence is in `adapter_results.json`, with raw pulled Modal summaries in `pulled/`.

Main indication:

- Cosine straightening improved directional straightness versus no straightening, but increased speed variation.
- Speed-only improved speed regularity, but did not improve directional straightness.
- Cosine plus speed gave the best curvature, path ratio, and mean state distance, but not the best planner success rate.

The speed-only run has a caveat: the planner used the latest available checkpoint and the latent diagnostic resolved it to epoch 18 with 20 analyzed rollouts. The other adapter diagnostics used epoch 20 and 50 rollouts.

## Original interruption evidence

The first attempt was incomplete. Modal rejected new app creation with:

```text
App creation failed: workspace billing cycle spend limit reached
```

The raw logs recovered from the Modal volume are in `raw_volume/`. The parsed partial summary is `partial_adapter_results.json` and should be treated only as interruption evidence.
