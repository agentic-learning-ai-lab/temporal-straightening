# Medium Temporal Straightening Results

This local evidence pack combines the existing Medium baseline/DINO run with three speed-constancy ablations run on Modal.

| variant | encoder | loss | train | val | success | state_dist | curvature | speed_cv | p95/p05 | rel_jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| authors_dino_patch_straightened | dino_patch | cos1e-1 | 0.2552 | 0.2385 | 0.14 | 4.4943 | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| dino_patch_no_straightening | dino_patch | False | 0.1195 | 0.1025 | 0.14 | 4.4943 | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| dino_cls_no_straightening | dino_cls | False | 0.0785 | 0.0691 | 0.22 | 3.5456 | 1.3533 | 0.3377 | 3.9247 | 0.3636 |
| dino_patch_speed_constancy | dino_patch | aggspeed1e-1 | 0.1268 | 0.1094 | 0.14 | 4.4943 | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| dino_patch_cos_plus_speed | dino_patch | cos1e-1+aggspeed1e-1 | 0.2625 | 0.2455 | 0.14 | 4.4943 | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| dino_cls_speed_constancy | dino_cls | speed1e-1 | 0.0860 | 0.0754 | 0.22 | 3.5456 | 1.3533 | 0.3377 | 3.9247 | 0.3636 |

## Findings

- The authors-style cosine straightening baseline and DINO patch no-straightening both reached 0.14 planner success on Medium.
- DINO CLS no-straightening reached 0.22 planner success and lower mean state distance, even with slightly worse cosine curvature than patch DINO.
- Patch-token speed constancy and cosine+speed ablations also reached 0.14 success; CLS speed constancy reached 0.22 success.
- Latent speed diagnostics did not move within an encoder family because these runs use frozen DINO features. The speed loss changed the training objective/loss, not the frozen visual encoder geometry measured by `latent_analysis.json`.
- The key evidence for the critique is still visible: patch-DINO has lower cosine curvature than CLS, but worse speed variation and worse planning success.

## Image Evidence

- Extracted media files: 2520
- Validation contact sheet: `/Users/utsavsharma/Documents/GitHub/temporal-straightening/modal_evidence/medium-ablations-20260702-01/ablation_epoch20_validation_contact_sheet.png`
- Train rollout contact sheet: `/Users/utsavsharma/Documents/GitHub/temporal-straightening/modal_evidence/medium-ablations-20260702-01/ablation_epoch20_train0_rollout_contact_sheet.png`
- The images use the same Medium layout, rollouts, and evaluation seeds, so they naturally look similar. They are qualitative reconstruction/rollout checks, not the primary evidence for latent planning quality.
- Representative patch speed and patch cosine+speed images are byte-identical; representative CLS images differ pixel-wise but preserve the same task setup. Details are in `image_evidence_summary.json`.

## Artifact Layout

- `combined_medium_results.json`: merged metrics table.
- `image_evidence_summary.json`: image counts, contact-sheet paths, hashes, and pixel diffs.
- `archives/`: downloaded Modal evidence tarballs and manifests.
- `extracted/runs/<run_id>/`: extracted run summaries, latent diagnostics, logs, planner outputs, validation images, and rollout plots.
- Epoch `.pth` checkpoint binaries were skipped in local archives to avoid multi-GB transfers; the manifest lists them and they remain on Modal volume `temporal-straightening-medium`.
