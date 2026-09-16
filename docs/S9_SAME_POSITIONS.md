# S9 — Same positions, cheaper labels (sealed)

Experiment **X0003**, sealed with `record_hash sha256:18d8d34f40938301e…`, 160/160 cells
COMPLETED, 0 INVALID. Preregistration: `tools/s9/s9-preregistration.json`, hash
`sha256:ee2313728516554ed7fac2977ece1faed36baa13e0e18ee6a8f123a4819edd46` (recorded in `tools/s9/s9-preregistration.hash`).

**The question.** If we train on the *exact same chess positions*, can 2k labels replace 8k
labels without materially hurting the student? S8 removed the student-compute confound; S9
removes breadth as a variable entirely. The only thing that changes is how hard the teacher
thought when producing each target.

## Design — unchanged except the teacher's depth

* **Positions:** the S8 8k arm's **9,495 positions**, frozen as one id list
  (`hash_obj(ids)` recorded). Both datasets contain exactly those ids **in the same order**
  (verified row-by-row), so the same training seed gives the same data order in both arms.
* **Zero new teacher labels:** the 2k stream already covers the 8k prefix.
* **Student compute identical:** fixed updates, 760 × batch 256 = 194,560 presentations per arm;
  same optimizer, LR, init, RAW-768, H1/H4/H16/H32, exam E0004/D0010.
* **20 seeds** (0–19) because this is an equivalence question where interval width matters:
  2 depths × 4 widths × 20 seeds = **160 cells**, frozen as X0003 before any outcome.

## Primary result — non-inferiority is NOT established at δ = +0.001

`d = test_loss(2k) − test_loss(8k)`, positive means 2k is worse; one-sided 95 % bounds
(t = 1.729, df = 19):

| width | d | one-sided 95 % | upper < +0.001 ? |
|---|---|---|---|
| H1 | +0.001155 | [+0.000768, +0.001541] | no |
| H4 | +0.000927 | [+0.000179, +0.001674] | no |
| H16 | +0.001056 | [+0.000348, +0.001763] | no |
| H32 | +0.000769 | [−0.000061, +0.001599] | no |

**Decision: INCONCLUSIVE.** The point estimate is positive (2k slightly worse) at every width and
every interval's upper bound exceeds the preregistered margin. `FLOOR_CROSSED` is *not* supported
either — the estimates sit well below the margin. The secondary equivalence test also fails at
every width (each two-sided CI crosses +0.001).

Mean `test_loss`: 8k 0.018866, 2k 0.019842 — the penalty is **0.00098 absolute, ~5 % relative**.
For scale: the 16k→8k improvements that S8 treated as meaningful were 0.003–0.007, three to seven
times larger than this penalty.

## The economics, from frozen observations

Both arms trained on the same 9,495 positions, so the counterfactual is exact:

| labels on the same 9,495 positions | realized teacher nodes |
|---|---:|
| 8k | 74,568,086 |
| 2k | 18,690,779 |
| ratio | **3.99×** |

So 2k supervision is **4.0× cheaper for the same positions**, and the measured cost is a ~5 %
relative `test_loss` penalty. Whether that trade is acceptable depends on the margin you require:
it is **not** non-inferior at δ = 0.001, but the penalty is an order of magnitude smaller than the
effects S8 measured.

## Secondary: the teachers disagree far more than the students do

On the same positions (200 mate-sentinel rows excluded, 9,570 compared):

| statistic | value |
|---|---|
| median \|Δcp\| | 34 |
| p90 / p99 \|Δcp\| | 135 / 386 |
| rate > 25 cp / > 50 cp / > 100 cp | 58.3 % / 36.7 % / 16.5 % |
| sign disagreement | 7.9 % |
| different best move | 52.5 % |

The teachers disagree substantially per position — over a third differ by more than 50 cp and half
give a different best move — while students trained on them differ by ~5 % in mean `test_loss`.
That is consistent with the hypothesis that the network does not need highly precise labels: the
teacher may only need to place a position in roughly the right part of evaluation space. It is a
secondary observation on this pool, not a measured mechanism.

## Non-claims

* Static-instrument `test_loss` only; no playing strength, no search, no games.
* INCONCLUSIVE is the preregistered verdict; the correct summary is "not non-inferior at
  δ = 0.001, and not shown to cross the floor either".
* One pool, one teacher family, RAW-768, H1/H4/H16/H32, fixed-update regime.
* No new pool, no new labels, no GEO/HYBRID, no 1k or 512 arm, no Stockfish, no optimizer tuning.

## What follows

Per the preregistration's own map, 2k is **not yet a default labeling budget** at a 0.001 margin —
but the measured penalty (~0.001 absolute, ~5 % relative) is small enough that the next step is
still the lower frontier: a larger population testing 2k → 1k → 512, run under this same
"same positions, same student compute, one teacher depth different" design, which is the cheapest
possible way to locate the floor.

## Evidence bundle

`tools/s9/s9-preregistration.json` (+ `.hash`), `tools/s9/s9-state.json`, `tools/s9/s9-result.json`
(all cell values, per-seed differences, disagreement stats, student compute),
`tools/s9/s9-result-summary.json`, `tools/s9/s9_run.py`, `tools/s9/s9_analyse.py`, and the sealed
`X0003` object. This PR also carries the compute-accounting repair: epoch-bounded runs count every
sample (partial tail batch included) as X0001 recorded, fixed-update runs count exactly
MAX_UPDATES × BATCH, and the run log states which regime produced the number.
