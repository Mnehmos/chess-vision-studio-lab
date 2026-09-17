# S15 — Auxiliary geometry supervision: semantics as a training scaffold only

**Two experiments, one finding — and the finding is the weaker, cleaner one.** `X0013` measured the
comparison first, but its arms did not share a minibatch stream: one generator served the persistent
weights, the auxiliary-head weights and the sampler, so the auxiliary draws advanced the RNG before
the trainer shuffled. `X0013` is preserved, sealed and **superseded for causal claims**
(`tools/s15/s15-x0013-supersession.json`, `sha256:ce040de7…`). **`X0014`** is the replacement under
explicitly split streams (`SeedSequence(seed).spawn(3)` → persistent init, auxiliary init, sampling),
sealed with 160/160 cells COMPLETED, 0 INVALID. Preregistration v2
`sha256:ddb971cd…`; auxiliary manifest `sha256:382c8487…`; weight calibration `sha256:1a7effa4…`.
Issue [Mnehmos/chess-vision-studio-lab#42](https://github.com/Mnehmos/chess-vision-studio-lab/issues/42).

**Answer.** Geometry as **training-only** supervision makes the same RAW-768 evaluator better — it
is strictly better on the frozen exam at the two larger widths under identical example order
(−0.0018 at H16 and H32, bounds excluding zero) — but the effect is **narrower than the first run
suggested**: with the stochastic path controlled, H4's −0.0014 collapses to an unresolved −0.0008,
and H1 is unresolved. All four contrasts point the same way, so the direction is robust even where
the intervals cross zero. The deployed evaluator remains byte-for-byte identical in input surface
and parameter count; only training changes.

## The confound that forced a replacement (recorded, not patched over)

One `np.random.default_rng(seed)` fed `RawNnue.initialize` (persistent weights, then training-only
head weights) and was then handed to `nnue.train` for shuffling. Paired arms therefore started from
identical persistent weights but saw **different minibatch orders** — "score-only loss + order A"
versus "auxiliary loss + order B". Since the effects are 0.001–0.002, that was a real confound.

The repair, frozen before the replacement ran: three independent streams per seed
(`run_rng_streams`), persistent draws that never depend on whether auxiliary heads exist
(`initialize_for_run`), and a single sampler `nnue.batch_stream` shared by training and its
regression test — so the tested sampler *is* the training sampler.
`tests/test_nnue_aux.py::test_paired_aux_and_control_arms_share_init_and_batch_order` proves
identical persistent weights **and** identical first 20 minibatches across paired arms;
`test_the_sampler_stream_is_independent_of_other_draws` proves initialisation cannot move the
sampler.

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

## The result (X0014, split streams — authoritative)

`d = test_loss(AUX-GEO) − test_loss(RAW-SCORE)`, paired by seed under **identical example order**,
separate one-sided 95 % exact-t bounds (df 19); negative = the auxiliary arm is better.

| width | RAW-SCORE | AUX-GEO | d | bounds | verdict |
|---|---:|---:|---:|---|---|
| H1 | 0.015989 | 0.015828 | −0.000162 | [−0.000332, +0.000009] | unresolved |
| H4 | 0.016280 | 0.015519 | −0.000761 | [−0.001609, +0.000087] | unresolved |
| H16 | 0.018735 | **0.016961** | **−0.001774** | [−0.002901, −0.000648] | aux better |
| H32 | 0.019586 | **0.017806** | **−0.001780** | [−0.002820, −0.000741] | aux better |

For comparison, X0013 (shared stream, superseded) measured −0.0014 (H4, strict), −0.0016 (H16) and
−0.0022 (H32). Under the clean streams the effect **keeps its sign at every width** and its
significance at the two larger ones; H4's original strictness was partly the stochastic path.

Secondary metrics at H32: cp_mae 132.9 (aux) vs 140.1 (control). The auxiliary arm still **fits the
training rows less hard while generalising better**: train loss 0.0076 vs 0.0050, gap −0.0102 vs
−0.0145 (a distribution gap — D0036 has no validation split, fractions 1.0/0.0/0.0).

**Still not a small-model scaffold**: the two resolved widths are the larger ones, consistent with
S14's finding that explicit geometry carries information RAW does not recover efficiently, and with
the reading that semantics shape the representation rather than substituting for capacity.

## Accounting, honestly split

| | RAW-SCORE | AUX-GEO |
|---|---:|---:|
| persistent inference parameters (H32) | 24,641 | **24,641 (identical)** |
| training-only auxiliary parameters (H32) | 0 | **3,465** |
| mean training wall time (H32) | 5.39 s | 11.10 s (2.1×) |
| inference throughput (H32, batched numpy) | as measured in `s15-result-v2.json` | same deployment |

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
first). Under X0014, H1's interval is [−0.000332, +0.000009] — unresolved by 9e-6 — so the trigger
again did **not** fire, and `family_ablation: {fired: false}` is recorded in the sealed result. The third question ("which families contribute signal?") is therefore **open** in
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
| X0013 supersession record | `tools/s15/s15-x0013-supersession.json` / `.hash` |
| preregistration v2 (split streams) | `tools/s15/s15-preregistration-v2.json` / `.hash` |
| replacement driver | `tools/s15/s15_replace.py` |
| sealed lattices | `labstore/X/X0014.json` (`SEALED`, 160/160 — the finding), `X0013` (sealed, superseded) |
