# S16 — Teacher × representation: the minimal factorial matrix (sealed)

Experiment **X0015**, sealed at `record_hash sha256:a9ff1ef21477b2508f747b92389684e9563ec6d4bef70ac13cdec193ea7af98b`,
**160/160 cells COMPLETED**, 0 INVALID, 0 replaced. Preregistration
`tools/s16/s16-preregistration.json` hash `sha256:c82bb6d1…`. Issue
[Mnehmos/chess-vision-studio-lab#43](https://github.com/Mnehmos/chess-vision-studio-lab/issues/43),
the factorial capstone of the static phase (#18/#20).

**Answer.** Both main effects reproduce inside one controlled matrix, and the interaction is
negative everywhere but only partly resolved:

> **All four interaction estimates are negative; one is statistically resolved and three are
> unresolved. HYBRID consistently attenuates the Stockfish penalty relative to RAW, but the
> strength of that interaction is only partly resolved.**

The teacher ordering is uniform — the cheap CVS-4k teacher wins in every representation and at
every capacity on both exams — and the representation crossover reproduces S14 (HYBRID worse at
H=1, better at H=4). Because the same teacher that costs 8× more is *also worse in every tested
cell*, the factorial cannot identify whether representation substitutes for **beneficial** teacher
compute; what it does show is an economically attractive **dominance result**: CVS-4k + HYBRID beats
SF-32k + RAW on both exams at the informative capacity.

## The minimal matrix (selected from prior evidence, before any S16 outcome)

| factor | chosen | rule and evidence |
|---|---|---|
| teachers | **CVS-4k** (D0036, 2.0×, spec `6b8bd7dc…`) and **SF-32k cold** (D0044, 16.1×, spec `28d9613d…`) | exactly the two authorities S13 established, at S13's frozen contracts; no recalibration |
| representations | **RAW** and **HYBRID** | GEO is **excluded**: S14 measured it strictly worse than RAW at every matched budget, so it cannot inform an interaction HYBRID already answers |
| capacities | **H=1** (S14's sentinel) and **H=4** (S14's knee) | two scales only; exact parameter counts reported per cell (RAW 771 / HYBRID 813 at H=1; 3,081 / 3,249 at H=4) |
| auxiliary arm | **excluded** | S15's supported effect lives at H16/H32, outside these capacities (H1/H4 were unresolved there); adding it would move the capacity axis |

4 arms × 2 capacities × 20 seeds = **160 cells**. Every cell shares the same 18,953 positions
(D0036 and D0044 carry an identical `record_ids_hash`, `sha256:7d5a8a78…`), the same optimizer and
194,560 student presentations, cold starts, paired seeds, and the **dual frozen exams**:
CVS-DEEP (`E0004`, 400k CVS nodes) and SF-DEEP (`E0006`/`D0043`, cold Stockfish at 4M/label),
never pooled.

## Main effects (paired by seed, one-sided 95 % exact-t bounds, df 19)

`teacher effect` = loss(SF) − loss(CVS): **positive means the Stockfish-trained cell is worse**.
`representation effect` = loss(HYBRID) − loss(RAW): **negative means HYBRID is better**.

| capacity | exam | teacher eff. on RAW | teacher eff. on HYBRID | repr. eff. under CVS | repr. eff. under SF |
|---|---|---|---|---|---|
| H1 | CVS-DEEP | +0.002312 (worse) | +0.001959 (worse) | +0.002881 (HYBRID worse) | +0.002528 (worse) |
| H1 | SF-DEEP | +0.002299 (worse) | +0.001739 (worse) | +0.003492 (worse) | +0.002932 (worse) |
| H4 | CVS-DEEP | +0.002851 (worse) | +0.002001 (worse) | **−0.002984 (HYBRID better)** | **−0.003834 (better)** |
| H4 | SF-DEEP | +0.002585 (worse) | +0.001454 (worse) | **−0.001752 (better)** | **−0.002884 (better)** |

Both main effects reproduce the prior studies inside a factorial: CVS-4k beats SF-32k-cold in every
cell on both exams (S13's finding), and HYBRID's crossover is capacity-dependent — worse at H=1,
better at H=4 (S14's finding, including its non-monotonicity).

## Difference-in-differences interaction

`(SF − CVS | HYBRID) − (SF − CVS | RAW)`, paired by seed; the symmetric construction
`(HYBRID − RAW | SF) − (HYBRID − RAW | CVS)` gives the identical number by construction
(checked in the artifact).

| capacity | exam | interaction | bounds | verdict |
|---|---|---|---|---|
| H1 | CVS-DEEP | −0.000353 | [−0.000935, +0.000229] | unresolved |
| H1 | SF-DEEP | **−0.000560** | [−0.001062, −0.000058] | **negative (resolved)** |
| H4 | CVS-DEEP | −0.000850 | [−0.002065, +0.000366] | unresolved |
| H4 | SF-DEEP | −0.001131 | [−0.002375, +0.000113] | unresolved |

**All four point estimates are negative and none is positive**; one interval is strictly resolved.
The label is therefore `NEGATIVE_INTERACTION_PARTLY_RESOLVED`, and the defensible reading is exactly
that sentence: HYBRID consistently attenuates the Stockfish penalty relative to RAW, with the
strength of the interaction only partly resolved.

Two cautions that belong with the numbers, not in a footnote:

* the interaction is **not** an order of magnitude smaller than the main effects — at H4 on the SF
  exam the interaction is −0.001131 while the teacher effect on HYBRID is +0.001454, i.e. the same
  general scale — so "near additivity" would be an overstatement;
* three intervals crossing zero do **not** establish independence. Unresolved is unresolved:
  the data are consistent with a small interaction in either direction, and this matrix cannot
  distinguish "independent" from "weakly complement-leaning".

## Dominance: does the cheap teacher with structure beat the expensive teacher without it?

This comparison is a **dominance result, not an identified substitution effect**: the more expensive
teacher (SF-32k) is *also the worse teacher* in every tested cell, so "structure substitutes for
teacher compute" cannot be concluded from it — that would require a more expensive teacher that is
actually better, which this program's frozen contracts do not provide. At the informative capacity
(H=4):

| cell | teacher spend | exam loss (CVS-DEEP) | exam loss (SF-DEEP) |
|---|---:|---:|---:|
| **CVS-4k / HYBRID** | **2.0×** | **0.013296** | **0.025726** |
| SF-32k / HYBRID | 16.1× | 0.015297 | 0.027180 |
| CVS-4k / RAW | 2.0× | 0.016280 | 0.027478 |
| SF-32k / RAW | 16.1× | 0.019131 | 0.030063 |

The cheap teacher with HYBRID **beats the 8×-pricier teacher with RAW by 0.0058 (CVS exam) and
0.0043 (SF exam)**. That is an economically attractive dominance result — spend 2.0× on labels and
add structure rather than 16.1× on labels — and it is all it is: since SF-32k is both pricier and
worse here, the comparison does not isolate what representation would buy against a teacher that
actually helps. Nor does it close the gap to using both: SF-32k/HYBRID is still worse than
CVS-4k/HYBRID, which is S13's Stockfish-cost finding, unchanged by representation.

## Economics and accounting (per cell)

| | CVS cells | SF cells |
|---|---|---|
| teacher nodes per label | 4,000 (realized 3,939) | 32,000 (realized 31,636) |
| teacher nodes total | 74,570,561 (2.0×) | 599,592,682 (16.1×) |
| teacher ms/label (frozen S13 measurement) | 7.5 | 79.7 |
| student presentations | 194,560 (identical everywhere) | 194,560 |
| persistent parameters H4 | RAW 3,081 / HYBRID 3,249 | same |
| training-only parameters | 0 everywhere (S15's auxiliary arm is not in this matrix) | 0 |
| train wall time H4 (mean) | RAW 5.22 s / HYBRID 8.41 s | RAW ~5.9 s / HYBRID 9.45 s |
| model bytes H4 | 67,857 / 71,803 | 67,779 / 71,836 |

Inference throughput is comparable across cells (same RAW-768/HYBRID-810 numpy path; measured rates
in `s16-result.json`), and HYBRID's extra 42 input columns cost ~6 % of the deployed parameter count
at this scale.

## What the factorial says (and what it does not)

* **Substitutes?** Not identified. The dominance result (cheap teacher + HYBRID beats expensive
  teacher + RAW) is real, but substitution cannot be tested against a teacher that is both pricier
  and worse.
* **Complements?** Consistently signed, partly resolved: all four interaction estimates are
  negative — HYBRID attenuates the Stockfish penalty — with one interval excluding zero.
* **Independent?** Not established either: three intervals cross zero, so independence is one of
  several readings the data are consistent with, not a finding.
* **Authority-specific?** No: both exams agree on the teacher ordering, on the representation
  crossover, and on the interaction's sign.

**Non-claims.** No pooled score across the two exams; no game, search or strength claims; static
losses only. One population, one optimizer recipe, one student family, two capacities, 20 seeds,
one machine. GEO and the S15 auxiliary arm are absent by predeclared rule, not by outcome. The
sixteen-week question "which teacher is better in general" is not asked or answered — S16 measures
*this* CVS contract against *this* Stockfish contract at *these* capacities.

## The next axis

The static phase's own evidence says the binding constraint is **capacity**, not teacher authority
or representation: RAW and HYBRID both bottom out around 3K parameters (S14), HYBRID at 3K already
beats every larger RAW model, and the teacher ordering did not change any conclusion here. The next
experiment should therefore push **scale within the static contract** (wider students at matched
compute, and the fixed-compute regime's overfitting that S14 exposed) before any move to search —
that is the axis with unresolved headroom, and it is a precondition for asking whether a stronger
teacher ever pays once the student can use it.

## Evidence bundle

| artifact | path |
|---|---|
| preregistration + hash | `tools/s16/s16-preregistration.json` / `.hash` |
| frozen S13/S14/S15 references + cell manifest + selection rules | `tools/s16/s16_common.py` |
| driver (`preregister` \| `experiment` \| `run`) | `tools/s16/s16_run.py` |
| dual-exam pass + analysis + seal | `tools/s16/s16_analyse.py` |
| dual-exam scores (validated against recorded metrics) | `tools/s16/s16-dual-exam.json` / `.hash` |
| state / result / summary | `tools/s16/s16-state.json`, `s16-result.json`, `s16-result-summary.json` |
| wording erratum (interpretation claims) | `tools/s16/s16-wording-erratum.json` / `.hash` |
| sealed lattice | `labstore/X/X0015.json` (`SEALED`, 160/160) |
