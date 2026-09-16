# S8 — The lower supervision frontier, with a student-compute control (sealed)

Experiment **X0002**, sealed with `record_hash sha256:65f310d07218865f…`, 160/160 cells
COMPLETED, 0 INVALID. Preregistration: `tools/s8/s8-preregistration.json`, hash
`sha256:a7eca1e81d757c1ac8f9982277c62415830e59cff588cee34515d588066ff949` (recorded in `tools/s8/s8-preregistration.hash`).

**The question.** S7-S showed that spending a fixed teacher budget on ~25× more 16k-supervised
positions beat 400k supervision at every scale — but wider datasets also received more optimizer
work under the fixed-epoch recipe. S8 asks whether that advantage is about the *labels* or about
the *student's extra training*.

## Design

* **One teacher scale:** 2× = 74,570,982 realized nodes.
* **Four depths, one frozen order:** 16k ⊂ 8k ⊂ 4k ⊂ 2k — 4,772 ⊂ 9,495 ⊂ 18,961 ⊂ 37,898 rows,
  every prefix at 1.0000 of the target.
* **Two student regimes:** the historical 40-epoch schedule, and a fixed-update schedule of
  exactly **760 optimizer updates at batch 256** = 194,560 sample presentations for every arm
  (deterministic sampler: shuffle, walk without replacement, drop the tail partial batch,
  reshuffle on exhaustion, stop at exactly 760).
* 4 depths × 2 regimes × H1/H4/H16/H32 × 5 seeds = **160 cells**, frozen as X0002 before any
  outcome was inspected.

**Reuse and cost.** The universe, order and instrument are S7-S's, re-verified: 79,483 records,
order `sha256:bbdb9a3b…`, **zero** collisions with the full 575-record evaluation source, and the
existing 2×/16k arm reproduced exactly (records are the order prefix, spec, realized nodes
74,573,451) so it was **not repurchased**. The new 8k/4k/2k streams cost **230,147,701 realized
teacher nodes** (9,770 / 19,500 / 38,989 labels) against the 223,712,946 estimate — 1.029×, the
excess being closest-prefix probe labels.

## Primary result — matched student compute: the direction holds, the strict all-width test does not

2k − 16k under equal optimizer updates (negative favours 2k):

| width | effect | 95 % CI | resolved |
|---|---|---|---|
| H1 | −0.001393 | [−0.003764, +0.000978] | no |
| H4 | −0.007734 | [−0.013848, −0.001620] | yes |
| H16 | −0.006561 | [−0.010620, −0.002502] | yes |
| H32 | −0.010285 | [−0.014555, −0.006014] | yes |

Preregistered decision: **INCONCLUSIVE** — the point estimate favours 2k at every width and three
of four CIs exclude zero, but H1 leaves the "every width" condition unmet. This is reported as
inconclusive, not as a win.

## Adjacent steps — the curve flattens *below* 8k, it does not turn

| step | H1 | H4 | H16 | H32 |
|---|---|---|---|---|
| 8k − 16k | −0.0033 | −0.0072 | −0.0061 | −0.0054 |
| 4k − 8k | +0.0014 | −0.0004 | −0.0016 | −0.0036 |
| 2k − 4k | +0.0005 | −0.0001 | +0.0011 | −0.0014 |

8k already beats 16k at every width, and 2k − 4k is flat everywhere. There is **no evidence of a
floor above 2k**: the useful supervision depth is at most 8k, and 2k shows no degradation.

## The confound, resolved

Mean `test_loss` by arm: 16k 0.02352 (epochs) / 0.02318 (updates), 8k 0.01984 / 0.01769,
4k 0.02021 / 0.01667, 2k 0.01776 / **0.01668**.

2k − 16k under the fixed-epoch regime is negative at all four widths (−0.0026, −0.0109, −0.0037,
−0.0058; two significant) — the same direction as the matched branch. The between-regime
difference is small (+0.0013, +0.0032, −0.0029, −0.0045), i.e. **the depth effect is essentially
the same whether or not the student gets extra optimization work**.

**So the breadth advantage is not a student-compute artifact.** S7-S's confound is closed: the
cheaper labels are doing the work, not the extra epochs that came with wider datasets.

## Scope and non-claims

* Static-instrument `test_loss` only; no search, no games, no playing strength.
* One teacher scale (2×), four depths, RAW-768, H1/H4/H16/H32 — no 1k arm, no GEO/HYBRID, no
  Stockfish, no priority selection, no optimizer tuning.
* The preregistered primary decision is INCONCLUSIVE, and the honest reading is *direction
  consistent at every width, resolved at three*. It is not a claim that 2k is optimal; 2k − 4k
  being flat means the data cannot distinguish 2k from 4k.
* Per the preregistration's own map, the next step if the shallow advantage holds is to build a
  larger population and test 2k → 1k → 512 before scaling 2k upward.

## Evidence bundle

`tools/s8/s8-preregistration.json` (+ `.hash`), `tools/s8/s8-state.json` (verification, calibration,
per-depth label totals, arms, experiment, run statuses), `tools/s8/s8-result.json` (all cell
values, every contrast, student compute), `tools/s8/s8-result-summary.json`,
`tools/s8/s8_run.py`, `tools/s8/s8_analyse.py`, and the sealed `X0002` object.
