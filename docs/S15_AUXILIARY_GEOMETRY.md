# S15 — Auxiliary geometry supervision: semantics as a training scaffold only

Experiment **X0013**, sealed with `record_hash` recorded in `labstore/X/X0013.json`, **160/160 cells
COMPLETED**, 0 INVALID, 0 replaced. Preregistration
`tools/s15/s15-preregistration.json` hash `sha256:224c1280aad1db5387d6af9d39797d1567489ef7b6b57e2f2785ed3c1d5f8a41`;
auxiliary target manifest `sha256:382c8487…`; weight calibration `sha256:1a7effa4…`.
Issue [Mnehmos/chess-vision-studio-lab#42](https://github.com/Mnehmos/chess-vision-studio-lab/issues/42),
slice of #18.

**Answer.** Yes — named deterministic geometry is useful as **training-side supervision** even when
the deployed evaluator never sees it. The same RAW-768 → score network, trained on the same rows
with the same optimizer and update count, is **strictly better on the frozen exam at H4, H16 and
H32** when it additionally predicts the registry's geometry facts through training-only heads, and
the benefit **grows with width** (d = −0.0014 → −0.0016 → −0.0022). At H1 the contrast is
unresolved. The auxiliary heads are discarded at inference: the deployed model is byte-for-byte the
same input surface and parameter count as the control's.

## The contract (one variable changes)

Teacher and exam are S13/S14's CVS contract unchanged: teacher **CVS-4k over D0036** (TargetSpec
hash asserted equal to X0008's sealed 4k arm), exam **E0004** (CVS-DEEP 400k, the 200 held-out
identities), `MAX_UPDATES 760 × BATCH 256 = 194,560` presentations for every cell, cold starts, same
optimizer/LR/init, paired seeds 0–19. Both arms receive **RAW-768 only**; the treatment arm's
difference is entirely in the loss.

## The auxiliary targets (registry-pinned, position-only)

`tools/s15/s15-aux-manifest.json` names every supervised quantity: **21 families × 2 kinds** —
21 continuous *value* targets (the family's White-POV signed delta ÷ its own bucket-3 threshold,
sign-flipped for black to move, i.e. exactly S14's GEO columns) and 21 × 4 *bucket* logits (the
registry's 0–3 magnitude, cross-entropy). Registry hash `sha256:a91c902a…`, the same vocabulary S14
froze. Missingness: none. Never depends on teacher search, the target score, outcome, best move or
eval split membership — it is a function of the position alone.

## The loss, and how its weight was chosen (training-dynamics calibration)

`total = score + AUX_WEIGHT × (mean value MSE over families + mean 4-class CE over families)`, each
family averaged so no family dominates by column count. `AUX_WEIGHT` was chosen **before any
outcome** by a preregistered rule that reads only training-row gradients: at initialisation the
auxiliary loss puts **3.037×** the score loss's trunk gradient at weight 1, so
**AUX_WEIGHT = 0.0823074** gives the declared 25 % ratio (`s15-aux-weight-calibration.json`). Fixed
for every run; no per-seed or per-width tuning.

## The result

`d = test_loss(AUX-GEO) − test_loss(RAW-SCORE)`, paired by seed, separate one-sided 95 % exact-t
bounds (df 19); negative = the auxiliary arm is better.

| width | RAW-SCORE | AUX-GEO | d | bounds | verdict |
|---|---:|---:|---:|---|---|
| H1 | 0.016271 | 0.016386 | +0.000114 | [−0.001262, +0.001490] | unresolved |
| H4 | 0.016261 | **0.014825** | **−0.001436** | [−0.002197, −0.000674] | aux better |
| H16 | 0.017991 | **0.016354** | **−0.001637** | [−0.002648, −0.000625] | aux better |
| H32 | 0.019040 | **0.016794** | **−0.002246** | [−0.003272, −0.001221] | aux better |

Secondary metrics at H32: cp_mae 128.1 (aux) vs 138.9 (control); sign agreement 0.905 vs 0.891.
The auxiliary arm also **fits the training rows less hard while generalising better**: train-split
loss 0.00751 vs 0.00505, so its train-vs-exam distribution gap is *smaller* (−0.0093 vs −0.0140).
There is no validation split for D0036 (fractions 1.0/0.0/0.0), so that gap is a distribution gap,
not a classic overfit gap.

**The effect is not a small-model scaffold.** The benefit grows monotonically with width, the
opposite of "geometry substitutes for capacity" — it looks like a better-learned *representation*
at every size above the smallest, consistent with S14's finding that explicit geometry carries
information RAW does not recover efficiently.

## Accounting, honestly split

| | RAW-SCORE | AUX-GEO |
|---|---:|---:|
| persistent inference parameters (H32) | 24,641 | **24,641 (identical)** |
| training-only auxiliary parameters (H32) | 0 | **3,465** |
| mean training wall time (H32) | 5.66 s | 12.97 s (2.3×) |
| inference throughput (H32, batched numpy) | 450,838 pos/s | 505,048 pos/s |

The inference difference is measurement noise between two identical deployments: `predict_cp` never
reads the auxiliary arrays, the input surface is RAW-768 in both arms, and no geometry is available
at inference. What the treatment arm really costs is **training**: 3,465 extra parameters at H32
(14 % of the deployed model) and 2.3× the wall time — a cost paid once, at training time.

## Auxiliary prediction quality by family (H16, 4,000 training rows)

The heads genuinely learn the registry — exactly where the registry is unambiguous and weakly where
it is not: `OPEN_CENTER_KING` bucket accuracy 0.998 (value MSE 0.003), `HANGING_MATERIAL` 0.52,
`MOBILITY_KNIGHT` 0.44, `ISOLATED_PAWN` 0.44, `MOBILITY_ROOK` 0.39, and `PASSED_PAWN` has the
largest value error (MSE 0.51). Per the preregistration: **better auxiliary prediction is not
evidence of better chess judgment** — the score exam is the evidence, and it improved.

## The family ablation did NOT run (the frozen rule's outcome)

The preregistered trigger was: *run the one-family-at-a-time ablation iff the **H=1** contrast's
one-sided upper bound is below zero* (the smallest width, where a scaffolding effect should appear
first). At H1 the interval is [−0.001262, +0.001490] — unresolved — so the trigger did **not**
fire, no ablation matrix was frozen, and `family_ablation: {fired: false}` is recorded in the
sealed result. The third question ("which families contribute signal?") is therefore **open** in
S15. It is answerable by a successor that fixes its sentinel width from *its own* prior evidence
(H4/H16/H32 are legitimate choices for a new preregistration, since these effects are already
public) — the S11-style targeted follow-up, not a post-hoc narrowing of this study's rule.

## Non-claims

* Static-loss results only; no search, playing-strength or game claims.
* No teacher-authority change; no GEO/HYBRID inputs at inference; no active learning; no
  combinatorial ablations.
* One population (18,953 positions), one exam, one optimizer, 20 seeds, one machine; python-measured
  throughput.
* The load-bearing claim is narrow: *geometry as auxiliary supervision improves this RAW evaluator
  on this exam at these widths*; it is not a claim that the heads learned chess concepts.

## Evidence bundle

| artifact | path |
|---|---|
| preregistration + hash | `tools/s15/s15-preregistration.json` / `.hash` |
| auxiliary target manifest + hash | `tools/s15/s15-aux-manifest.json` / `.hash` |
| weight calibration (train-only) + hash | `tools/s15/s15-aux-weight-calibration.json` / `.hash` |
| driver (`manifest` \| `preregister` \| `experiment` \| `run` \| `ablation`) | `tools/s15/s15_run.py` |
| analysis + seal | `tools/s15/s15_analyse.py` |
| state / result / summary | `tools/s15/s15-state.json`, `s15-result.json`, `s15-result-summary.json` |
| auxiliary supervision (lab) | `cvslab/nnue.py` (AUX modes, heads, loss), `cvslab/families.py` (AUX/AUX_WEIGHT switches, parameter split) |
| tests | `tests/test_nnue_aux.py` (7 cases) |
| sealed lattice | `labstore/X/X0013.json` (`SEALED`, 160/160) |
