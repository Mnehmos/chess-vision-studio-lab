# S12 — The lower economic frontier: 4k / 2k / 1k / 512 (sealed)

Experiment **X0008**, sealed with `record_hash sha256:c1e3853f55d9e72e378d282168617c51868f153d1d4ff91dc96bf53e4d957cac`,
**320/320 cells COMPLETED**, 0 INVALID, 0 duplicates. Preregistration:
`tools/s12/s12-preregistration.json`, hash `sha256:7295062363f0d61ae4ad62de094c21a66828f2ba37a1240c5eb667d9f6e3a3bf`.

**Answer.** The quality penalty does **not** accelerate as teacher search keeps halving. At one
fixed teacher budget (74,570,982 nodes ≈ the S7-S 2× scale), the 512-node arm buys **7.8× the
coverage** of the 4k arm — 147,513 rows against 18,953 — and is **not worse**: at H32 it is
*strictly better* (−0.002601, one-sided `[−0.003485, −0.001717]`), at H4 and H16 the point estimates
favour 512 with upper bounds of `+0.000063` and `+0.000172` (unresolved, but one to two seed-widths
from resolved), and at H1 a small adverse `+0.000867 [−0.000363, +0.002096]` straddles zero. Under
the preregistered rule the decision is **INCONCLUSIVE** — neither global verdict fires — but the
*bend where the savings stop paying was not found*: the curve is flat-to-favourable all the way
down to 512 nodes per label.

## Design — one teacher budget, four label depths, an append-only population

* **Question.** How far can teacher search keep halving (4k → 2k → 1k → 512) before the quality
  penalty accelerates enough that the compute savings stop being worth it?
* **Population.** The S7-S universe (79,483 records) extended **append-only** by 11,200 new
  EPD-start games (357,329 raw positions, seed 20260920) → normalization **N0014**: 183,482
  records, 299 removed by the leakage join against the full 575-record N0010 exam source, then 36
  old records excluded as exam-bridged by the new games' transpositions → **183,183 candidates**,
  order hash `sha256:41375c2d…`. N0013 (extension games only, missing the old positions) was
  discarded before any run.
* **The one budget.** Every arm spends the same **74,570,982 teacher nodes** and buys as many rows
  as its label depth allows: 4k → 18,953 rows, 2k → 37,898, 1k → 75,627, 512 → 147,513. Realized
  nodes are 1.0000 of target in all four arms (worst deviation 0.0006 %).
* **Nesting.** 4k ⊂ 2k ⊂ 1k ⊂ 512 along **one frozen order** (closest-prefix under the node
  target at each depth), asserted at arm construction; each arm's `D####` is frozen over its prefix.
* **Zero new labels at 4k and 2k.** Those streams were carried by record identity; 1k and 512 were
  bought along the extended order: 77,929 labels / 76,840,566 nodes and 152,029 / 76,845,415. The
  cost-only calibration (64 positions, cost only — no value ever inspected) predicted 985.6 and
  505.2 mean nodes per row; the streams realized 986.0 and 505.5.
* **Equal student compute** (the S8 control): `MAX_UPDATES 760 × BATCH 256 = 194,560`
  presentations for every arm — 10.3 passes over the 4k arm's 18,953 rows, 1.3 over the 512 arm's
  147,513. Same optimizer, LR, init, RAW-768, H1/H4/H16/H32, seeds 0–19, exam E0004/D0010.
* **Frozen before any outcome:** the complete 4-arm × 4-width × 20-seed lattice as **X0008**, with
  its analysis and decision rule fixed in the preregistration.

## The frontier curve

`d` positive means the shallower (cheaper) depth is worse. Bounds are separate one-sided 95 %
lower/upper bounds, exact Student-t at each width's actual df (t = 1.7291, df = 19).

| primary `512 − 4k` | d | one-sided bounds | verdict |
|---|---:|---|---|
| H1 | **+0.000867** | [−0.000363, +0.002096] | unresolved |
| H4 | **−0.000848** | [−0.001760, **+0.000063**] | unresolved (upper 6.3e-5 above 0) |
| H16 | **−0.000842** | [−0.001856, **+0.000172**] | unresolved (upper 1.7e-4 above 0) |
| H32 | **−0.002601** | [**−0.003485, −0.001717**] | **512 strictly better** |

Adjacent halvings (descriptive, same convention; the columns sum to the primary):

| contrast | H1 | H4 | H16 | H32 |
|---|---:|---:|---:|---:|
| 2k − 4k | +0.001017 | −0.000122 | −0.000458 | −0.001713 |
| 1k − 2k | +0.001022 | +0.000358 | −0.000273 | −0.000225 |
| 512 − 1k | −0.001172 | −0.001084 | −0.000111 | −0.000663 |
| **512 − 4k** | **+0.000867** | **−0.000848** | **−0.000842** | **−0.002601** |

Mean `test_loss` (lower is better; bold = best arm at that width):

| arm | H1 | H4 | H16 | H32 |
|---|---:|---:|---:|---:|
| 4k | **0.016271** | 0.016261 | 0.017991 | 0.019040 |
| 2k | 0.017288 | 0.016139 | 0.017533 | 0.017327 |
| 1k | 0.018310 | 0.016496 | 0.017259 | 0.017102 |
| 512 | 0.017138 | **0.015413** | **0.017148** | **0.016439** |

## Does the per-halving tax accelerate? No

The preregistration asked whether the per-halving tax grows — that bend is where savings stop
paying. Across the three halvings below 4k:

* **H32 and H16**: every halving is at or below zero (−0.0017 / −0.0002 / −0.0007 and
  −0.0005 / −0.0003 / −0.0001). The tax does not merely fail to accelerate; there is no positive
  tax to accelerate.
* **H4**: −0.0001 / +0.0004 / −0.0011 — the middle step is the only positive one.
* **H1**: +0.0010 / +0.0010 / −0.0012 — the smallest model pays about +0.001 for each of the first
  two halvings (unresolved; it is also the noisiest arm, se ≈ 0.0007) and then *gains* 0.0012 going
  to 512.
* **No width** shows a positive, growing penalty. The single width whose tax sequence is strictly
  increasing (H16) increases from −0.00046 to −0.00011 — still negative at every step.

So the economic reading is the opposite of "the tax eats the savings": at the widest model the
cheap-label arm wins the same teacher budget outright, and the two near-unresolved widths lean the
same way.

## Economics — same teacher budget, 7.8× the coverage

| arm | label depth | rows | nodes / row | realized teacher nodes | shard of target | passes over its own rows |
|---|---:|---:|---:|---:|---:|---:|
| 4k | 4,000 | 18,953 | 3,934.5 | 74,570,561 | 0.99999 | 10.27 |
| 2k | 2,000 | 37,898 | 1,967.7 | 74,571,153 | 1.00000 | 5.13 |
| 1k | 1,000 | 75,627 | 986.0 | 74,571,388 | 1.00001 | 2.57 |
| **512** | **512** | **147,513** | **505.5** | 74,571,000 | 1.00000 | **1.32** |

Coverage ratio 512/4k = **7.78×**; per-row cost ratio 4k/512 = **7.78×**. The 512 arm spends 80.5 %
of the 183,183-candidate universe. Student compute is *equal by construction*, so the trade being
measured is exactly "precision per label vs number of positions", never "more total compute".

## What the instrument contract establishes

All 320 runs were re-verified against their stored payloads and share **one exam**:
`E0004` / `D0010` / eval TargetSpec `sha256:80e8e3de…` — `assert_runs_comparable` over all 320 run
ids returns that single hash. The four arms pin four *distinct* training TargetSpecs (one per label
depth) and a single, uniform `supervision_divergence` declaration, so the treatment really is the
declared label depth. No run is missing, duplicated or non-terminal.

## The preregistered decision, and what INCONCLUSIVE means here

The rule frozen before any run: **SHALLOWER_FRONTIER** iff the one-sided upper bound of
`test_loss(512) − test_loss(4k)` is below zero at every width; **FRONTIER_EXHAUSTED** iff the lower
bound is above zero at every width; **INCONCLUSIVE** otherwise. Verdict: **INCONCLUSIVE** — H4 and
H16 miss the first rule by 6.3e-5 and 1.7e-4.

INCONCLUSIVE here is not "no information". The preregistered framing anticipated the *opposite*
failure: it was written to detect an accelerating penalty, and both global verdicts require all
four widths to agree in sign. What the evidence shows is that three of four widths favour the
cheaper labels (one decisively) and the fourth is a small adverse estimate inside its interval.
512 is therefore **not** shown to be globally non-inferior — and equally, no accelerating penalty
is shown at any width.

**How many seeds would settle it** (projection only, holding the observed effect and sd fixed;
`se ≈ se₂₀·√(20/n)`, exact one-sided t at each df):

| width | seeds needed | what it would resolve |
|---|---:|---|
| H4 | ~23 | upper < 0 → 512 better |
| H16 | ~29 | upper < 0 → 512 better |
| H32 | 20 (already) | 512 strictly better |
| H1 | ~39 | lower > 0 → 512 worse (the adverse direction is what n can resolve) |

A targeted extension (S11's pattern: extra seeds only where variance limits the verdict) at ~30
seeds on two widths would convert the decision to SHALLOWER_FRONTIER *if* the H4/H16 estimates
hold. That is a projection, not a result.

## Errata — one parenthetical in the sealed provenance text

The sealed result's `provenance.prefix_reproduction` says "8 of them lay inside the old 4k prefix
and 32 inside the old 2k prefix". Those were *symmetric set differences* (excluded records **plus**
records displaced by the node-target landing point), not counts of excluded records lying inside
the prefixes; and S9/X0003 has no 4k arm. The corrected accounting, recomputed positionally from
the two stored orders, is: of the 36 excluded records, **7** lie inside the old 4k prefix (18,961)
and **16** inside the old 2k prefix (37,898); the new 4k prefix is a strict subset of the old one
with 18,953 rows (7 excluded + 1 displaced by the node target), and the 2k arm reaches the same
37,898 rows with 16 replacements from just beyond the old boundary. The consequence the sealed text
states is unchanged: the carried arms are not row-identical to S8's, and **no exact
cross-experiment value reproduction is claimed**. Full record: `tools/s12/s12-result-errata.json`.
The sealed result is not rewritten (`update_experiment` refuses a SEALED study; the repository's
precedent is to correct additively), and `tools/s12/s12_analyse.py` deliberately still emits the
sealed wording so a re-run reproduces the sealed result byte-for-byte.

## Non-claims

* No margin/equivalence test: S12's decision rule is sign-based by preregistration. "512 not shown
  globally non-inferior" is not "512 is equivalent to 4k at a +0.001 margin" — that test was never
  preregistered here.
* No 8k arm: 8k/4k/2k belongs to S9–S11 on a different population.
* Label depth and row count are deliberately confounded (one budget, more rows as labels cheapen).
  S12 measures the *economic* trade, not a mechanism.
* Widths are never pooled; one exam; 20 seeds; RAW-768; H ≤ 32.

## What follows

1. **Targeted seed extension at 512** (H4/H16 only, ~30 seeds) — the cheapest path to a
   non-INCONCLUSIVE verdict, exactly S11's targeted-precision pattern.
2. **The universe, not the budget, is now the binding constraint**: the 512 arm already spends
   80.5 % of the 183,183 candidates. The next halving (256) would want ~295,000 rows for the same
   budget, which this universe cannot supply — the natural successor question is whether the
   *population* must grow again, or whether 512 is simply where the frontier is.
3. Not attempted: 256, GEO/Stockfish routing, new scales, wider models.

## Evidence bundle

| artifact | path |
|---|---|
| preregistration + hash | `tools/s12/s12-preregistration.json` / `.hash` |
| driver (extend → calibrate → labels → arms → experiment → run) | `tools/s12/s12_run.py` |
| analysis + seal | `tools/s12/s12_analyse.py` |
| state (universe, calibration, labels, arms, queue) | `tools/s12/s12-state.json` |
| result (full) | `tools/s12/s12-result.json` |
| result (summary) | `tools/s12/s12-result-summary.json` |
| errata | `tools/s12/s12-result-errata.json` |
| sealed lattice | `labstore/X/X0008.json` (`SEALED`, membership 320/320) |
