# S11 — Targeted precision: 4k vs 8k, seeds where they are needed (sealed)

Experiment **X0006**, sealed with `record_hash sha256:d96280b9d851…`, **1040/1040 cells
COMPLETED, 0 INVALID**. Preregistration: `tools/s11/s11-preregistration.json`, hash
`sha256:a31c5f55f7b1509260a124d5177163e9dde9bf0b5f93ca183d4f5806dad53c8c`.

**Question.** S10 left `4K_NONINFERIOR` unresolved: all four point estimates were below the +0.001
margin but every upper bound crossed it, and H32 was the identified bottleneck. This study spends
seeds where variance actually limits instead of repeating a balanced matrix.

## Design — the axis that carries the seed budget

Each **(depth, width)** pair is its own arm, which is what lets a width carry its own seed budget:
H1/H4/H16 get **40 seeds**, H32 gets **400** (`tools/s11/s11-power-planning.json`). The plan came
from S10's own point estimates and standard errors; 280 seeds would have put H32's upper bound at
≈0.000998 against a 0.001 margin, so 400 was preregistered to avoid the knife edge. Identical
positions (the frozen 9,495), identical student compute (760 × 256), identical exam, **zero new
teacher labels**. 8 arms × 1040 targeted cells.

## Result — the bottleneck moved, and it moved exactly as planned

`d = test_loss(4k) − test_loss(8k)`, positive = 4k worse; separate one-sided 95 % bounds:

| width | seeds | d | one-sided 95 % | upper < +0.001 ? |
|---|---:|---|---|---|
| H1 | 40 | +0.000681 | [+0.000509, +0.000853] | **yes** |
| H4 | 40 | +0.000256 | [−0.000256, +0.000767] | **yes** |
| H16 | 40 | +0.000878 | [+0.000263, +0.001493] | no |
| H32 | **400** | +0.000722 | [+0.000497, +0.000947] | **yes** |

**Decision: INCONCLUSIVE** — three of four widths now clear the preregistered margin, and the rule
requires all four to. H32, the width the planning targeted, resolved precisely as predicted
(upper bound 0.000947). The failure is now H16 alone.

**Why H16 needs more than the plan said.** The planning used S10's H16 point estimate (+0.000213)
and standard error; the fresh 40-seed estimate is **+0.000878**, the largest of the four and the
closest to the margin. Planned from S11's own evidence, H16 would need roughly **975 seeds** — and
because its estimate sits only ~0.0001 below the margin, its true penalty may be *at or above* the
margin, in which case no number of seeds makes 4k non-inferior at that width.

## Determinism across experiments: verified

Seeds 0–19 of every 4k width reproduce **X0004's stored `test_loss` values exactly** — 80
comparisons, **0 mismatches**. Two separately frozen experiments, one deterministic pipeline.

## What this says about the method

* Targeting seeds by variance works: the bottleneck moved from H32 (resolved, 400 seeds) to H16.
* Planning from a single experiment's point estimates is optimistic: H16's S10 estimate was on the
  low side of its sampling distribution, so the required n was understated by ~25×.
* A precision study should therefore re-plan from the pooled evidence after each round, and should
  preregister the *possibility* that a width's true penalty lies above the margin (which would make
  non-inferiority unattainable there rather than merely unproven).

## Standing picture of the 4k tradeoff (S10 + S11, same positions)

| width | d (4k − 8k) | 95 % bounds | status |
|---|---|---|---|
| H1 | +0.0007 | [+0.0005, +0.0009] | inside the margin |
| H4 | +0.0003 | [−0.0003, +0.0008] | inside the margin |
| H32 | +0.0007 | [+0.0005, +0.0009] | inside the margin |
| H16 | +0.0009 | [+0.0003, +0.0015] | unresolved; may be at/above the margin |

Halving the teacher budget 8k→4k therefore costs ≈0.0003–0.0009 `test_loss` (≈1.5–4 % relative)
depending on width, and for three of four widths that penalty is now *demonstrated* to be inside the
0.001 margin.

## Non-claims

Static-instrument `test_loss` only; no playing strength. Same pool, same teacher family, RAW-768,
fixed-update regime. No new labels, no 1k/512 arm, no GEO, no Stockfish, no games, no optimizer
tuning. INCONCLUSIVE is the preregistered verdict; the per-width detail is the result.

## Evidence bundle

`tools/s11/s11-preregistration.json` (+ `.hash`), `s11-power-planning.json`, `s11-state.json`,
`s11-result.json`, `s11-result-summary.json`, `s11_run.py`, and the sealed `X0006` object. This
branch also carries the contract generalization the design required: an arm may declare its own
seed subset, and width coverage is validated per supervision depth (a design may carry the ladder
in one arm or distribute it across arms; a missing width still fails closed).

---

## Confirmatory analysis on fresh seeds (the primary result)

S11 chose its seed counts from S10's seeds 0–19 and then included those observations in its
intervals, which makes an all-seed interval a *data-dependent precision estimate*. The confirmatory
test therefore uses only seeds the planning never touched: **20–39 for H1/H4/H16, 20–399 for H32**,
with exact Student-t criticals at each width's own df (`tools/s11/s11_confirmatory.py`).

| width | seeds | d | one-sided 95 % | inside +0.001 ? |
|---|---:|---|---|---|
| H1 | 20 | +0.000630 | [+0.000414, +0.000847] | **yes** |
| H4 | 20 | +0.000167 | [−0.000628, +0.000962] | **yes** |
| H32 | 380 | +0.000725 | [+0.000496, +0.000954] | **yes** |
| H16 | 20 | **+0.001544** | [+0.000570, +0.002518] | **no** |

**Fresh data moved H16's estimated penalty above the margin, but H16 remains statistically
unresolved around it.** The point estimate (+0.001544) is above +0.001, yet the interval
[+0.000570, +0.002518] still includes values below the margin, so H16 is *neither* proven
non-inferior *nor* proven materially worse than δ — it is unresolved, with its fresh estimate lying
above the margin, and larger than the +0.000878 the all-seed precision estimate showed (the earlier
number was partly carried by the very seeds used to size the study).

**Conclusion.** 4k is non-inferior to 8k at H1, H4 and H32 under δ = 0.001. H16 remains unresolved,
with its fresh estimate lying above the margin. Therefore 4k is not established as globally
non-inferior across widths — and that is enough evidence to move on to the 1k/512 frontier rather
than spend another ~1,000 seeds on H16.

**Precision estimates (all seeds, secondary):** H1 +0.000681 [+0.000505, +0.000857], H4 +0.000256
[−0.000268, +0.000780], H16 +0.000878 [+0.000249, +0.001508], H32 +0.000722 [+0.000496, +0.000948].
These now use exact df-specific Student-t values (df=39 → 1.6849, df=399 → 1.6487) rather than the
normal fallback the first S11 bundle used.
