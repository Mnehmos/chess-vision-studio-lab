# S10 — Is 4k the economic sweet spot between 8k and 2k? (sealed)

Experiment **X0004**, sealed with `record_hash sha256:bca0d06e9155d7c32…`, 240/240 cells
COMPLETED, 0 INVALID. Preregistration: `tools/s10/s10-preregistration.json`, hash `sha256:d3bda9acabe8e23901b2315ced13bb527850eb883704df8a755e47e8258a807e`.

**Answer.** 4k halves the teacher cost **and** halves the penalty: its point estimates
(+0.0002 … +0.0007, mean +0.00049) are about half of 2k's (+0.0008 … +0.0012, mean +0.00098), and
every 4k estimate sits **below** the preregistered +0.001 margin. But no width's *upper* bound is
below the margin, so **4K_NONINFERIOR is not demonstrated** — the decision is **INCONCLUSIVE**,
exactly as it was for 2k. What the evidence does support is a clean, monotone cost/quality curve.

## Design — three depths, one frozen position set, identical student compute

* Same **9,495 positions** as S9/S8, same order, verified in all three datasets (8k and 2k reused,
  4k newly frozen over the same ids).
* **Zero new teacher labels:** all three streams already covered these positions.
* Identical student compute: 760 updates × batch 256 = 194,560 presentations; same optimizer, LR,
  init, RAW-768, H1/H4/H16/H32, exam E0004/D0010; 20 seeds.
* 3 depths × 4 widths × 20 seeds = **240 cells**, frozen as X0004 before any outcome.

## The three-point tradeoff curve

`d` positive means the shallower (cheaper) depth is worse. Bounds are separate one-sided 95 %
lower/upper bounds (t = 1.729, df = 19).

| comparison | H1 | H4 | H16 | H32 | mean |
|---|---|---|---|---|---|
| **4k − 8k** (primary) | +0.000732 | +0.000344 | +0.000213 | +0.000671 | **+0.00049** |
| 2k − 4k | +0.0004 | +0.0006 | +0.0008 | +0.0001 | +0.00048 |
| 2k − 8k | +0.0012 | +0.0009 | +0.0011 | +0.0008 | +0.00098 |

Mean `test_loss`: **8k 0.018866 < 4k 0.019356 < 2k 0.019842** — monotone in teacher depth, with the
penalty roughly doubling as the label gets 4× cheaper.

## Realized economics (same 9,495 positions)

| depth | realized teacher nodes | vs 8k |
|---|---:|---:|
| 8k | 74,553,268 | 1.00× |
| 4k | 37,334,303 | **2.00×** cheaper |
| 2k | 18,690,779 | **3.99×** cheaper |

## Decision

Separate one-sided 95 % upper bounds for 4k − 8k: +0.001026 (H1), +0.001088 (H4), +0.001005 (H16),
+0.001952 (H32). Every point estimate is below the +0.001 margin, but every upper bound is above it
— so `4K_NONINFERIOR` fails and `4K_TAXED` is unsupported (the estimates are below the margin, not
above it). **INCONCLUSIVE**, and the secondary equivalence test (two-sided CI inside ±0.001) fails at
all four widths for the same reason: the intervals are still wide enough to admit a penalty slightly
larger than the margin.

**Reading.** On this evidence 4k is the better operating point than 2k — same design, half the
penalty, half the saving — but neither is *proven* non-inferior at δ = 0.001. The honest engineering
statement is: **halving the teacher budget from 8k to 4k costs ≈0.0005 test-loss (≈2.5 % relative)
on the same positions, and halving it again to 2k costs ≈0.001 (≈5 %)**. For a budget decision that
is a usable curve; for a non-inferiority claim at 0.001 it is not yet enough evidence.

## One quantified accounting discrepancy, disclosed

The realized totals here are summed **over the frozen 9,495 ids**. S8 and S9 recorded their 8k total
as 74,568,086 — the sum over the same *number* of rows but drawn in label-purchase order, which is a
stride permutation of the walk (the parallel labeler appends per worker chunk). The difference is
**14,818 nodes, 0.02 %**, it does not touch any dataset, run, metric or decision, and the corrected
value for the same quantity is 74,553,268. Future bundles sum over the frozen id list.

## Non-claims

* Static-instrument `test_loss` only; no playing strength, no search, no games.
* One position set (the S7-S/S8 9,495), one teacher family, RAW-768, H1/H4/H16/H32, fixed-update
  regime. No 1k or 512 arm yet, no GEO, no Stockfish, no optimizer tuning.
* INCONCLUSIVE is the preregistered verdict for 4k as well as for 2k; nothing here is a
  non-inferiority claim.

## What follows

The curve is now measured at three points with a consistent design. Two natural next moves, in
order: (1) tighten the interval rather than widen the study — more seeds at 4k vs 8k would resolve
whether the ~0.0005 penalty is below the margin, and seeds are far cheaper than teacher compute; and
(2) only then extend downward to 1k and 512 on a larger position set, where the tax is expected to
grow past the margin and the floor becomes visible.

## Evidence bundle

`tools/s10/s10-preregistration.json` (+ `.hash`), `tools/s10/s10-state.json`,
`tools/s10/s10-result.json`, `tools/s10/s10-result-summary.json`, `tools/s10/s10_run.py`, and the
sealed `X0004` object. This branch is stacked on PR #36 (S9), which carries the frozen 9,495-id
evidence S10 reuses.
