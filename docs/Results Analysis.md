# Results Analysis

**Sources:** the raw result files under `modal_evidence/` — `*/combined_results.json`, `medium-adapter-ablations-20260702-01/adapter_results.json`, `wall-full-20260703-01/wall_results.json`, the per-variant `latent_analysis.json`, the `plans_fixed/*/logs.json` planner logs, and `medium-ablations-20260702-01/image_evidence_summary.json`.

This is an **independent read of the numbers**, going a layer deeper than the study's own write-up in [The Reproduction Study](The%20Reproduction%20Study.md). The short version: the headline "straightening makes no difference" results on Medium and Wall are **measurement degeneracies, not genuine null effects** — the frozen-encoder setup provably cannot detect any effect. The only experiment with real discriminative power (the trainable adapter) confirms the [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md) diagnosis but shows latent geometry does **not** predict planning success at this scale.

## The four experiments at a glance

| Pack | Environment | Tests | Discriminative power |
|---|---|---|---|
| `medium-full` | PointMaze-Medium | authors' straightening vs DINO-patch vs DINO-CLS | ⚠️ degenerate |
| `medium-ablations` | Medium | speed loss on **frozen** DINO | ⚠️ structurally inert |
| `medium-adapter-ablations` | Medium | speed loss on **trainable** adapter | ✅ the only real signal |
| `wall-full` | Wall | straightening vs baseline | ⚠️ degenerate |

## Headline numbers

**Medium baseline (frozen DINO):**

| Variant | Straighten | Val loss | Success | State dist |
|---|---|---|---|---|
| Authors (DINO patch) | `cos1e-1` | 0.2385 | 0.14 | 4.4943 |
| DINO patch | `False` | 0.1025 | 0.14 | 4.4943 |
| DINO CLS | `False` | 0.0691 | **0.22** | **3.5456** |

**Adapter ablation (trainable DINO-channel) — the informative one:**

| Variant | Success | State dist | Curvature ↓ | Path/end ↓ | Speed CV ↓ | Speed jump ↓ |
|---|---|---|---|---|---|---|
| None | 0.10 | 3.582 | 1.306 | 16.57 | 0.458 | 0.450 |
| Cosine | 0.12 | 3.345 | **1.027** | 12.40 | 0.577 ⚠ | 0.605 ⚠ |
| Speed | **0.14** | 3.412 | 1.343 | 17.42 | **0.404** | **0.386** |
| Cosine+speed | 0.10 | **3.118** | **0.981** | **11.90** | 0.491 | 0.491 |

## The core finding: most "no difference" results are degenerate

The report describes the Medium baseline as showing *equal planner success* for straightened vs unstraightened. Reading the raw files shows it's stronger and more troubling than "equal":

- The planner logs `plans_fixed/authors_dino_patch_straightened/logs.json` and `plans_fixed/dino_patch_no_straightening/logs.json` are **byte-identical** — success 0.14, state dist 4.494283676147461, visual dist, proprio dist, and even `mean_div_visual_emb` 778.6959228515625, matching to **13 significant figures**.
- **Wall** is identical too: both variants report state dist 33.464290618896484 to full float precision.
- In the frozen ablation, the epoch-20 contact sheets for the speed and cosine+speed patch variants share an **identical SHA-256** (`2b90becb…`) — cryptographically identical images. Only the CLS-family sheet differs.

These are genuinely different trained models (train loss 0.2552 vs 0.1195), yet they produce float-for-float and pixel-for-pixel identical planning outcomes. Independent gradient-descent planning runs cannot coincidentally match to 13 digits.

**Mechanism:** with a fully frozen DINO encoder and stop-grad, the straightening loss changes the *predictor's* training but not the frozen encoder geometry the planner's objective is measured in. In this setup the planner outcome is dominated by the frozen encoder and is **insensitive** to everything straightening touched (see [The Straightening Loss](The%20Straightening%20Loss.md) on why a frozen encoder has no trainable path to reshape latent geometry, and [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md) for the frozen/trainable table).

**Implication:** the Medium-baseline and Wall "no difference" results carry **zero information** about whether straightening helps. They aren't evidence against the paper — they're evidence the experiment couldn't detect an effect either way. The study's own framing correctly calls this out for the *frozen ablation*, but carries the same "equal success" language into the *baseline comparison*, where it reads like a finding and is actually a measurement artifact.

## What survives scrutiny: the adapter ablation

The instant there's a **trainable** path (the `dino_channel` adapter), the numbers finally vary per variant — confirming the degeneracy above was structural, not coincidental. Here the story is clean and honest:

- **Cosine straightens direction but wrecks pacing.** Curvature 1.306 → 1.027 (better), but speed CV 0.458 → 0.577 and speed jump 0.450 → 0.605 (worse). This is the [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md) made concrete: the cosine loss buys straightness by spending speed regularity.
- **Speed-only is the mirror image.** Best pacing (CV 0.404, jump 0.386), zero straightening benefit (curvature 1.343).
- **Cosine+speed gets the best geometry on both axes** (curvature 0.981, path/end 11.90, state dist 3.118).
- **But planner success does not follow geometry.** Speed-only "wins" success at 0.14; best-geometry cosine+speed ties for last at 0.10.

Caveat baked into the data: the speed-only row used **epoch 18 / 20 rollouts**, while the others used **epoch 20 / 50 rollouts** — so its apparent "win" rests on slightly different footing (noted in the pack's own caveats).

## Secondary observations

- **DINO CLS quietly beat everything** — 0.22 success vs 0.14, best state distance (3.55), no straightening at all. The simplest representation won. This is arguably the most robust single result in the set and cuts against the premise that patch-level straightening is the path forward.
- **Patch latents are far harder to plan in.** `mean_div_visual_emb` is 778.7 for DINO patch vs 36.9 for CLS — a 20× gap in how much the planner's imagined visual embeddings diverge from what's achieved. CLS gives the planner a much better-behaved target space.
- **The ground-truth state-speed CV is constant** (0.5634) across adapter variants, as it should be — it's a property of the environment's trajectories, not the model, and serves as a sanity anchor.

## Statistical reality check

Every success number here comes from **50 planning evaluations, one seed, 20 training epochs, 50 random-policy episodes**. The adapter success spread is 0.10–0.14 — a difference of **2 out of 50 rollouts**, comfortably inside sampling noise. The baseline spread of 0.14–0.22 is 4 rollouts. None of these differences are statistically powered; treat every ranking as a hint, not a result. See [Dataset Generators](Dataset%20Generators.md) on why the small random-policy dataset limits absolute performance.

## Bottom line

1. **Straightening's headline benefit did not reproduce** — but the frozen-encoder experiments were structurally *incapable* of measuring it (identical-to-13-digits outputs). Medium-baseline and Wall are inconclusive-by-construction.
2. **The critique's diagnosis is confirmed** by the one experiment with measurement power: cosine straightening genuinely trades away speed regularity.
3. **The proposed fix is unproven**: better latent geometry didn't translate into better planning — though at 1 seed and 1–2-rollout differences, nothing here is powered to show it either way.
4. **DINO CLS (no straightening) is the strongest result**, which is itself a notable finding.

**What would make this conclusive:** multiple seeds (error bars on the 0.10–0.22 numbers), a larger and/or expert-policy dataset, and running the entire comparison on **trainable encoders only** — since the frozen setup provably cannot see the effect. Full artifact index in [Evidence Packs](Evidence%20Packs.md); planner mechanics in [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md).
