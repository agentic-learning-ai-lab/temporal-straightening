# The Reproduction Study

**Sources:** `report.md` (the full report), `research_note.tex` / `.pdf` (the same study as a LaTeX note, credited to "Codex experiment report"), dated 2026-07-03.

An independent re-run of the paper's core claim, executed on Modal cloud GPUs via [Modal Runner](Modal%20Runner.md), with all artifacts archived in [Evidence Packs](Evidence%20Packs.md). It asks two questions:

1. **Does temporal straightening actually improve gradient-descent planning?** (reproduction)
2. **Does the cosine loss even enforce what the theory assumes?** (critique — see [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md))

Scale caveat up front: 50 random-policy episodes × 100 frames, 20 epochs, 50 planning evaluations, one seed — a small-budget study designed to compare variants fairly, not to match the paper's absolute numbers ([Dataset Generators](Dataset%20Generators.md)).

## Experiment 1 — Medium baseline

Three variants on PointMaze-Medium, all with gradient-descent planning (`goal_H=25`):

| Variant | Encoder | Straighten | Val loss | Success |
|---|---|---|---|---|
| Authors' approach | DINO patch | `cos1e-1` | 0.2385 | **0.14** |
| No straightening | DINO patch | `False` | 0.1025 | **0.14** |
| DINO CLS | DINO CLS | `False` | 0.0691 | **0.22** |

Two uncomfortable findings: straightening **did not beat** the plain baseline, and the simplest representation (one CLS vector per frame — see [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md)) beat both patch variants. (Note the val losses aren't comparable across rows — the straightened run's loss includes the extra regularizer term.)

## Experiment 2 — Frozen-DINO speed ablation (the flawed one)

Added `speed`/`aggspeed` losses to the frozen-DINO variants. Result: planner success and *all latent diagnostics identical to the baselines, to four decimals* — even the contact-sheet PNGs were **byte-identical**. This isn't a null result about speed constancy; it's a demonstration that with a fully frozen encoder there are no trainable parameters between pixels and the measured latents, so *no* representation loss can change the measured geometry ([The Straightening Loss](The%20Straightening%20Loss.md) explains the mechanics). The ablation validated the critique's logic but couldn't test the fix.

## Experiment 3 — Adapter ablation (the informative one)

Switch to `encoder=dino_channel`: frozen DINO backbone + **trainable ChannelProjector adapter**, so losses can actually sculpt the measured latents. Four loss settings:

| Variant | Success | State dist | Curvature ↓ | Speed CV ↓ | Speed jump ↓ |
|---|---|---|---|---|---|
| No straightening | 0.10 | 3.58 | 1.306 | 0.458 | 0.450 |
| Cosine only | 0.12 | 3.35 | **1.027** | 0.577 ⚠ | 0.605 ⚠ |
| Speed only | **0.14** | 3.41 | 1.343 | **0.404** | **0.387** |
| Cosine + speed | 0.10 | **3.12** | **0.981** | 0.491 | 0.491 |

The clean double dissociation: **cosine straightens direction but makes pacing *worse*** (⚠); **speed evens pacing but doesn't straighten**; combined gets the best geometry — *and yet planner success doesn't follow the geometry*. Speed-only won on success, best-geometry tied for last. (Caveat: speed-only diagnostics used 20 rollouts / epoch-18 checkpoint vs 50 / epoch-20 for the others.)

## Experiment 4 — Wall

Required two code fixes first (a `WallDataset` import shim and image-shape canonicalization — see [Dataset Generators](Dataset%20Generators.md)). Then: authors' straightening and plain DINO patch both scored **0.02 success** with *identical* distance metrics — at this data scale, nothing could really plan on Wall, so the comparison is uninformative beyond "no difference detected."

## What to take away

- The critique's **diagnosis is confirmed**: cosine straightening trades speed regularity away for directional straightness — the constant-speed assumption in the theory is genuinely unenforced ([The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md)).
- The proposed **fix is unproven**: better latent geometry did not translate into better planning in these runs.
- Straightening's headline benefit **did not reproduce** at this scale — but the scale is far below the paper's, so this is a flag, not a refutation. The note itself says more seeds and larger datasets are needed.
- Methodologically, this study is a nice example of ablation hygiene: noticing that Experiment 2 *couldn't* work, explaining why, and designing Experiment 3 to close the loophole.
