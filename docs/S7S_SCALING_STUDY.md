# S7-S — Supervision-density scaling at 2×/5×/10×/20× (sealed)

Experiment **X0001**, sealed with `record_hash sha256:9fa39059299f113c…`, 320/320 cells
COMPLETED, 0 INVALID. Preregistration:
`tools/s7/s7-scaling-preregistration.json`, hash
`sha256:c1e68acab2d0453d7d4a94b9836e0aed81fdc55ff60d563da99836766eb9a3f3`.

**Teaching statement.** *We increased the teacher's total thinking budget from 2× to 20×. At
every scale we spent exactly that budget in four different ways: a few very deep lessons
through many shallow lessons. Because the datasets and scales are nested, we can see both how
more supervision helps and whether the best allocation of that supervision changes as compute
grows.*

## Provenance

| link | identity |
|---|---|
| source population | the engine's committed EPD suite, 4,910 positions, LF-normalized sha256 `4d96e552…` (the Windows checkout hashes the same file `a4776725…`; the loader normalizes, so the pin is platform-independent) |
| generator | `cvslab.pool.selfplay` v3, `start_source="epd-file"`, per-game RNG, seed 20260920 |
| pool | 2,600 games → 82,909 raw rows → `sha256:fd742a5a00aa5265…` |
| snapshots | **S0011** (corrected: S0010's manifest-level `openingSourceSha256` held the 12-line default while its config block carried the EPD hash; the corrected snapshot re-records the same bytes with the frozen hash) |
| normalization | **N0012**, 79,483 records |
| leakage join | joint components against the **full 575-record N0010 evaluation source**: **0** records removed (0 instrument-adjacent, 0 population-adjacent) |
| universe | 79,483 clean candidates; universe hash `sha256:10e3ccf2f156a3cb5…`; order hash `sha256:bbdb9a3b4f26d7390…` (one order, seed 20260920) |
| instrument | **E0004 / D0010** (200 records, 400k deep targets), unchanged and component-disjoint |

## Arms — 16 nested prefixes, each filled to its scale target

Realized nodes are summed from the observations themselves; the ratio column is
realized / frozen scale target.

| scale | 400k | 160k | 64k | 16k |
|---|---:|---:|---:|---:|
| 2× (74,570,982) | 193 rows · 1.0016 | 478 · 0.9998 | 1,199 · 1.0000 | 4,772 · 1.0000 |
| 5× (186,427,455) | 481 · 1.0000 | 1,194 · 1.0002 | 2,982 · 1.0000 | 11,894 · 1.0000 |
| 10× (372,854,910) | 961 · 1.0001 | 2,388 · 1.0001 | 5,963 · 0.9999 | 23,786 · 1.0000 |
| 20× (745,709,820) | 1,927 · 0.9998 | 4,788 · 1.0000 | 11,953 · 1.0000 | **47,586 · 1.0000** |

Budget parity is within ±0.02 % for all sixteen arms. Nesting is asserted in both dimensions
(`2× ⊂ 5× ⊂ 10× ⊂ 20×` at fixed depth, `400k ⊂ 160k ⊂ 64k ⊂ 16k` at fixed scale) and holds.

**Labels were bought once.** Each depth was labeled along the single frozen order until the
20× crossing resolved; every smaller scale reuses a prefix of the same purchased stream.
Realized teacher cost: **3,080,112,644 nodes** (1,981 + 4,952 + 12,366 + 49,200 labels at
400k/160k/64k/16k) against the preregistered estimate of 2,982,839,280 — **1.033×**, the excess
being the probe/margin labels beyond the crossing. Buying sixteen independent datasets would
have cost 5.52 G; the nested construction cost 3.08 G.

## The frontier: at every scale, many shallow lessons beat a few deep ones

Paired seed differences, five seeds, 95 % CI (t = 2.776); negative favours **16k**.

| scale | verdict | H1 | H4 | H16 | H32 |
|---|---|---|---|---|---|
| 2× | WIDE_SHALLOW_BETTER_AT_EVERY_WIDTH | −0.0273 | −0.0139 | −0.0316 | −0.0374 |
| 5× | WIDE_SHALLOW_BETTER_AT_EVERY_WIDTH | −0.0278 | −0.0286 | −0.0389 | −0.0404 |
| 10× | WIDE_SHALLOW_BETTER_AT_EVERY_WIDTH | −0.0268 | −0.0315 | −0.0286 | −0.0280 |
| 20× | WIDE_SHALLOW_BETTER_AT_EVERY_WIDTH | −0.0209 | −0.0250 | −0.0161 | −0.0164 |

Every one of the sixteen width × scale cells has a CI entirely below zero. The magnitude is
large: the smallest effect is −0.0139 and the largest −0.0404 on a pooled `test_loss` level of
about 0.043, i.e. **32 %–94 % relative** — far beyond the pilot's noise-scale effects.

The adjacent contrasts at 20× show the curve is **monotone in depth** at every width:
160k−400k −0.0066…−0.0169, 64k−160k −0.0016…−0.0085, 16k−64k −0.0010…−0.0079. Mean
`test_loss` at 20× (H16) falls 0.03453 → 0.02795 → 0.02635 → **0.01841** as supervision gets
shallower.

## The scaling: more total supervision helps, but modestly

Paired 20× − 2× per depth and width (negative = more supervision improved `test_loss`):

| depth | H1 | H4 | H16 | H32 |
|---|---|---|---|---|
| 400k | −0.0095 | −0.0011 | −0.0207 | −0.0285 |
| 160k | −0.0233 | −0.0227 | −0.0333 | −0.0370 |
| 64k | −0.0232 | −0.0260 | −0.0139 | −0.0198 |
| 16k | −0.0031 | −0.0123 | −0.0053 | −0.0074 |

A tenfold budget increase helps in 32 of 32 cells (28 with CIs entirely below zero), but the
effect is **smaller than the depth-allocation effect**: moving from 400k to 16k at a fixed
budget buys ~2–3× more than multiplying the budget tenfold at a fixed depth.

## The interaction: the optimal allocation does not move

The best depth is **16k at every scale and every width** (2×, 5×, 10×, 20× × H1, H4, H16, H32
= 16 of 16 cells). Over a tenfold range of total supervision, the winning allocation does not
shift deeper — coverage continues to dominate label precision, and the preference for breadth
is already saturated at 2×.

## Student-side compute (reported, not hidden)

Fixed-epoch training means the broader arms receive far more optimizer work:
19,607,200 examples seen across the 320 runs. Per cell the result bundle records rows,
examples seen, CPU/wall seconds and examples per second
(`tools/s7/s7-scaling-result.json`, `student_compute`). The claim above is therefore about
**supervision scaling under one fixed training recipe**, not about teacher compute isolated
from student compute.

## Non-claims

* Not a claim about playing strength: `test_loss` on a frozen static instrument, no search, no games.
* Not a universal scaling law: four scales, one teacher, RAW-768, one pool.
* The 20×/16k arm holds 47,586 of the 79,483 candidates (59.9 %); the preregistered adequacy rule
  (≥ 1.15× the largest prefix) is met with 1.67× headroom, but that arm is not a small sample of
  the universe.
* GEO/HYBRID, other teachers, other optimizers and the exploratory 2k arm are out of scope by design.

## Evidence bundle

`tools/s7/s7-scaling-preregistration.json` (+ `.hash`), `tools/s7/s7-scaling-state.json`
(universe, order, calibration, per-depth label totals, arms, experiment, run statuses),
`tools/s7/s7-scaling-result.json` (all cell values, every contrast, student compute),
`tools/s7/s7-scaling-result-summary.json`, `tools/s7/s7_scaling_pool.py`,
`s7_scaling_run.py`, `s7_scaling_resnapshot.py`, `s7_scaling_analyse.py`, and the sealed
`X0001` object in the lab store.
