# Temporal Straightening Reproduction Report

Date: 2026-07-03

## Executive Summary

I retried the Modal pipeline after the spend-limit interruption and completed the requested end-to-end runs for Medium and Wall. The reproduced Medium baseline shows authors-style DINO patch temporal straightening and DINO patch without straightening both at 0.14 planner success, while DINO CLS reaches 0.22 success in this setup. The frozen-DINO speed ablation supports the critique but cannot prove a fix because the measured latent path is frozen.

The follow-up trainable DINO-channel adapter ablation addresses that caveat. It shows the expected tradeoff: cosine straightening improves directional straightness and path ratio but increases latent speed variation; speed-only improves speed regularity but does not make the path directionally straighter. Combining cosine and speed gives the best curvature, path ratio, and mean state distance, but not the best planner success rate.

For Wall, both authors-style DINO patch straightening and DINO patch without straightening completed end to end. DINO trained to a lower validation loss, but both variants produced the same planner success rate, 0.02, and the same final distance metrics.

## Methodology

The core claim being tested is that a useful latent encoding should make gradient-descent planning easy by making trajectories as straight as possible in latent space. The original temporal straightening loss minimizes cosine distance between consecutive latent velocity vectors:

```text
1 - cos(z[t+1] - z[t], z[t+2] - z[t+1])
```

That objective aligns directions but normalizes away speed. The ablation therefore adds speed-constancy penalties on consecutive latent step magnitudes. This targets the gap between the paper's constant-latent-speed assumption and the fact that the cosine loss itself does not enforce constant speed.

All completed Medium runs used 50 generated random-policy episodes, 100 frames per episode, 20 training epochs, and gradient-descent planning with 50 evaluations and goal horizon 25. Wall used the same episode count, episode length, epoch count, and planner settings after adding the missing Wall dataset support and fixing generated image shape consistency.

## Medium Baseline

| Variant | Encoder | Loss setting | Epoch | Train loss | Val loss | Planner success | Mean state dist |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authors approach | DINO patch | `cos1e-1` | 20 | 0.2552 | 0.2385 | 0.14 | 4.4943 |
| DINO patch | DINO patch | `False` | 20 | 0.1195 | 0.1025 | 0.14 | 4.4943 |
| DINO CLS | DINO CLS | `False` | 20 | 0.0785 | 0.0691 | 0.22 | 3.5456 |

Finding: DINO CLS was strongest in this reproduction. Authors-style patch cosine straightening did not improve planner success over the frozen DINO patch baseline.

## Frozen-DINO Speed Ablation

| Variant | Encoder | Loss setting | Epoch | Train loss | Val loss | Planner success | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Patch speed | DINO patch | `aggspeed1e-1` | 20 | 0.1268 | 0.1094 | 0.14 | No planner/latent change versus frozen patch baseline |
| Patch cosine + speed | DINO patch | `cos1e-1+aggspeed1e-1` | 20 | 0.2625 | 0.2455 | 0.14 | Same frozen patch geometry as baseline |
| CLS speed | DINO CLS | `speed1e-1` | 20 | 0.0860 | 0.0754 | 0.22 | Same CLS geometry as CLS baseline |

The frozen patch family had identical latent diagnostics across authors-style cosine, no straightening, speed-only, and cosine-plus-speed variants: mean cosine curvature 1.3402, path-to-endpoint ratio 16.0212, latent speed CV 0.3973, and relative speed jump 0.3791. The CLS family similarly matched between no-straightening and speed settings, with mean cosine curvature 1.3533 and latent speed CV 0.3377.

Interpretation: the speed-loss critique is valid, but this first ablation is limited. With frozen DINO features, the speed term can change model training losses but not the measured frozen encoder geometry used in the latent diagnostics.

## Adapter Ablation

The adapter ablation uses `encoder=dino_channel`, where the DINO backbone stays frozen but the measured projection path includes trainable channel projection and aggregation parameters. This makes it a more meaningful test of whether speed constancy can affect the latent geometry used for planning.

| Variant | Loss setting | Epoch | Train loss | Val loss | Planner success | Mean state dist |
|---|---:|---:|---:|---:|---:|---:|
| `dino_channel_no_straightening` | `False` | 20 | 0.1064 | 0.0957 | 0.10 | 3.5815 |
| `dino_channel_cos_straightened` | `cos1e-1` | 20 | 0.2104 | 0.1900 | 0.12 | 3.3451 |
| `dino_channel_speed_constancy` | `speed1e-1` | latest checkpoint | 0.0939 | 0.0742 | 0.14 | 3.4117 |
| `dino_channel_cos_plus_speed` | `cos1e-1+speed1e-1` | 20 | 0.1856 | 0.1680 | 0.10 | 3.1176 |

| Variant | Rollouts | Curvature | Path/end | Speed CV | Speed p95/p05 | Speed jump |
|---|---:|---:|---:|---:|---:|---:|
| No straightening | 50 | 1.3060 | 16.5666 | 0.4577 | 9.8059 | 0.4501 |
| Cosine | 50 | 1.0271 | 12.3965 | 0.5772 | 12.3654 | 0.6053 |
| Speed-only | 20 | 1.3431 | 17.4157 | 0.4038 | 9.5826 | 0.3865 |
| Cosine + speed | 50 | 0.9806 | 11.9015 | 0.4914 | 10.4932 | 0.4911 |

Interpretation:

- Cosine straightening improves directional straightness versus no straightening, lowering curvature from 1.3060 to 1.0271 and path ratio from 16.5666 to 12.3965.
- Cosine also worsens speed variation, increasing speed CV from 0.4577 to 0.5772 and relative speed jump from 0.4501 to 0.6053.
- Speed-only improves speed regularity, with the lowest speed CV and relative speed jump, but it does not improve curvature or path ratio.
- Cosine plus speed gives the best curvature, best path ratio, and best mean state distance, but planner success is not higher than the other adapter variants.

Caveat: the speed-only adapter diagnostic used 20 rollouts and an epoch-18 checkpoint after the 50-rollout diagnostic stalled. The other adapter diagnostics used 50 rollouts and epoch 20.

## Wall Results

Wall initially failed from the checkout because the legacy Wall import expected `WallDataset`, and the generated visual observations had inconsistent shapes. The retry succeeded after adding a Wall compatibility placeholder and canonicalizing generated Wall images to fixed-size HWC tensors.

| Variant | Encoder | Loss setting | Epoch | Train loss | Val loss | Planner success | Mean state dist |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authors approach | DINO patch | `cos1e-1` | 20 | 0.2824 | 0.2715 | 0.02 | 33.4643 |
| DINO patch | DINO patch | `False` | 20 | 0.1385 | 0.1209 | 0.02 | 33.4643 |

Both Wall variants also matched on mean visual distance 1.1894, mean proprio distance 21.4001, mean visual embedding divergence 1621.2992, and mean proprio embedding divergence 37.3018.

Interpretation: DINO patch without straightening trained better on reconstruction/prediction loss, but it did not translate into better Wall planning under this gradient-descent planner setup.

## Image Evidence

The Medium contact sheets align with the frozen-encoder findings. Patch speed and patch cosine-plus-speed validation and rollout contact sheets are byte-identical. CLS contact sheets differ from patch contact sheets. That matches the numerical result: unchanged frozen measured latent path gives unchanged image evidence, while changing encoder family changes the images and planner metrics.

The Wall planner path did not render image or contact-sheet media. Wall evidence consists of summaries and logs; `plan_targets.pkl` remains on the Modal volume and is intentionally not staged into the PR.

## Evidence Index

- Medium completed summary: `modal_evidence/medium-full-20260702-01/combined_results.json`
- Medium latent diagnostics: `modal_evidence/medium-full-20260702-01/latent_analysis.json`
- Frozen speed ablation summary: `modal_evidence/medium-ablations-20260702-01/combined_medium_results.json`
- Image evidence summary: `modal_evidence/medium-ablations-20260702-01/image_evidence_summary.json`
- Adapter completed summary: `modal_evidence/medium-adapter-ablations-20260702-01/adapter_results.json`
- Adapter pulled raw summaries: `modal_evidence/medium-adapter-ablations-20260702-01/pulled/`
- Wall completed summary: `modal_evidence/wall-full-20260703-01/wall_results.json`
- Wall pulled raw summaries/logs: `modal_evidence/wall-full-20260703-01/pulled/`
- Contact sheets:
  - `modal_evidence/medium-full-20260702-01/epoch20_validation_contact_sheet.png`
  - `modal_evidence/medium-full-20260702-01/epoch20_rollout_train0_contact_sheet.png`
  - `modal_evidence/medium-ablations-20260702-01/ablation_epoch20_validation_contact_sheet.png`
  - `modal_evidence/medium-ablations-20260702-01/ablation_epoch20_train0_rollout_contact_sheet.png`

Large checkpoint archives, full extracted artifacts, and planner target pickles remain local or on the Modal volume but are not suitable for normal GitHub PR upload.

## Reproduction Commands

Medium adapter planner retries:

```bash
modal run modal_medium_runner.py --action plan-existing --run-id medium-adapter-none-20260702-01 --epochs 20 --plan-n-evals 50 --plan-goal-h 25 --include-adapter-ablations --variant-name dino_channel_no_straightening
modal run modal_medium_runner.py --action plan-existing --run-id medium-adapter-cos-20260702-01 --epochs 20 --plan-n-evals 50 --plan-goal-h 25 --include-adapter-ablations --variant-name dino_channel_cos_straightened
modal run modal_medium_runner.py --action plan-existing --run-id medium-adapter-speed-20260702-01 --epochs 20 --plan-n-evals 50 --plan-goal-h 25 --include-adapter-ablations --variant-name dino_channel_speed_constancy
modal run modal_medium_runner.py --action plan-existing --run-id medium-adapter-cos-speed-20260702-01 --epochs 20 --plan-n-evals 50 --plan-goal-h 25 --include-adapter-ablations --variant-name dino_channel_cos_plus_speed
```

Medium adapter latent diagnostics:

```bash
modal run modal_medium_runner.py --action analyze-latents --run-id medium-adapter-none-20260702-01 --epochs 20 --environment medium --include-adapter-ablations --variant-name dino_channel_no_straightening --max-rollouts 50
modal run modal_medium_runner.py --action analyze-latents --run-id medium-adapter-cos-20260702-01 --epochs 20 --environment medium --include-adapter-ablations --variant-name dino_channel_cos_straightened --max-rollouts 50
modal run modal_medium_runner.py --action analyze-latents --run-id medium-adapter-speed-20260702-01 --epochs 20 --environment medium --include-adapter-ablations --variant-name dino_channel_speed_constancy --max-rollouts 20
modal run modal_medium_runner.py --action analyze-latents --run-id medium-adapter-cos-speed-20260702-01 --epochs 20 --environment medium --include-adapter-ablations --variant-name dino_channel_cos_plus_speed --max-rollouts 50
```

Wall full runs:

```bash
modal run modal_medium_runner.py --action run --environment wall --run-id wall-full-20260703-01 --epochs 20 --n-episodes 50 --episode-length 100 --batch-size 32 --num-workers 0 --plan-n-evals 50 --plan-goal-h 25 --variant-name authors_dino_patch_straightened
modal run modal_medium_runner.py --action run --environment wall --run-id wall-dino-20260703-01 --epochs 20 --n-episodes 50 --episode-length 100 --batch-size 32 --num-workers 0 --plan-n-evals 50 --plan-goal-h 25 --variant-name dino_patch_no_straightening
```
