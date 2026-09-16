# S13 — Teacher authority: CVS vs Stockfish under dual frozen static exams

Experiment **X0009**, sealed with `record_hash sha256:72b8fd2efeace6f01ff3a04c734ab5013aa74aa3dead1c2d4e0fd9a551eae8c2`,
**240/240 cells COMPLETED**, 0 INVALID, 0 duplicates. Preregistration:
`tools/s13/s13-preregistration.json`, hash `sha256:d2da7de839df74a192bb76be41b8eef75aceffd738e7c611f0ca8b111bab440a`.
Issue [Mnehmos/chess-vision-studio-lab#40](https://github.com/Mnehmos/chess-vision-studio-lab/issues/40),
the first narrow slice of #20.

**Answer.** Changing the teacher changes what the student learns — and here it changed it for the
worse. With identical positions, identical rows, identical student compute and identical seeds, the
**cheap legacy CVS teacher produced the better static student on BOTH exams at every width**,
including the exam built from the modern engine's own supervision. At H32 the paired gaps are
**+0.004663 [+0.003700, +0.005627]** on CVS-DEEP and **+0.002554 [+0.001479, +0.003628]** on
SF-DEEP (positive = the Stockfish-trained student is worse). The dose arm points the same way:
**SF-1M, which spends 495.9× CVS-4k's teacher nodes, is worse than SF-32k on both exams at every
width**. A small own-authority interaction exists (each student is *relatively* better on its own
teacher's exam at H4/H16/H32), but it is dwarfed by the absolute CVS advantage. The preregistered
verdict vocabulary named only two clean patterns and this is a third one, so `decision` stays
**INCONCLUSIVE** by the letter and the realized pattern is reported beside it as
**CVS_BETTER_ON_BOTH_EXAMS (4/4 widths, strict)**.

## The teachers, pinned by bytes

| | CVS-4k | SF-32k | SF-1M |
|---|---|---|---|
| authority | `legacy.cvs.search.shallow` | `oracle.stockfish.18` | `oracle.stockfish.18` |
| producer | `36e9b159…` (analyze build) | `c86215fa…` (binary SHA256) | `c86215fa…` |
| per-label budget | `{"nodeBudget": 4000}` | `{"nodes": 32000}` | `{"nodes": 1000000}` |
| measured cost | 7.5 ms/label | 41.9 ms/label | 1562 ms/label |
| realized/label | 3,939 nodes | 31,634 nodes | 975,562 nodes |
| realized depth | 4–6 | 18.4 | 28.6 |
| total teacher spend | 74,570,561 (2.0×) | 599,554,132 (16.1×) | 18,489,830,782 (495.9×) |

`×` = the arm's realized teacher spend ÷ the 37,285,491 reference unit. The conditions are **not a
grid**, so the study declares an explicit cell list and each arm carries its own honest scale —
never a nominal shared budget two of the arms do not spend.

Stockfish identity is recorded as bytes and handshake, not a nickname: `Stockfish 18`, binary
SHA256 `c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911`, options
Threads 1 / Hash 16 / MultiPV 1, and **both** embedded nets (`nn-c288c895ea92.nnue` 125 MiB,
`nn-37f18f62d772.nnue` 6 MiB). SF 18 chooses the small net per position
(`Eval::use_smallnet`: `|simple_eval| > 962`) deterministically — recorded in the identity, not
hidden. `{"nodes": N}` is SF's budget key and can never subset-match a CVS spec's `{"nodeBudget": N}`.

## Calibration: the convergence rule failed, so the design controls on cost and dose

The preregistered rule (`tools/s13/s13-calibration.json`, hash `sha256:478aea89…`) asked for the
cheapest rung whose labels agree with a 2M-node reference (≥ 0.99 sign, ≥ 0.99 best move,
≤ 15 mean |Δcp|) on a frozen 64-position set. **No rung qualified — not even 4M** (best-move
agreement 0.898, mean |Δcp| 14.6). Stockfish labels are not converged at any affordable budget.

| budget | sign vs 2M | best move vs 2M | mean \|Δcp\| | depth | wall/label |
|---|---:|---:|---:|---:|---:|
| 32k | 0.983 | 0.746 | 40.5 | 18.2 | 50 ms |
| 128k | 1.000 | 0.797 | 33.9 | 21.4 | 174 ms |
| 512k | 0.983 | 0.814 | 18.5 | 25.6 | 676 ms |
| 1M | 1.000 | 0.848 | 16.5 | 27.4 | 1299 ms |
| 4M | 1.000 | 0.898 | 14.6 | 35.1 | 5014 ms |

Equal labeling spend is also unreachable in a useful range: CVS-4k costs **7.5 ms/label**, the
cheapest SF rung **50 ms (6.7×)**. The amendment (`tools/s13/s13-calibration-amendment.json`, hash
`sha256:1c5a31c8…`) therefore fixes **SF-32k as the primary** Stockfish condition (cheapest
calibrated rung, more expensive than CVS — so a CVS win cannot be a spending artefact) and
**SF-1M as the dose condition** (measured at 495.9× CVS-4k's teacher spend), with the **SF-DEEP
exam at 4M nodes/label** (deterministic: 8/8 repeats identical in cp and best move). No student
existed when this was chosen.

## Design

* **One population, one row set.** The 18,953 rows of `D0036` (S12's 4k arm; `N0014`, the
  order-preserving filtered extension) for every condition. A missing teacher label fails closed —
  it never shrinks a dataset. The CVS condition reuses `D0036` unchanged and its TargetSpec hash is
  asserted equal to the sealed X0008 4k arm's training spec.
* **Identical student compute.** `MAX_UPDATES 760 × BATCH 256 = 194,560` presentations, T0004-derived
  recipe, RAW-768, H ∈ {1, 4, 16, 32}, seeds 0–19, one supervision-divergence declaration.
* **Three conditions, each with its own honest spend.** The conditions are not a grid, so the
  experiment declares an **explicit cell list** (a lab capability added for this study); each arm's
  scale value is the teacher spend it actually made ÷ the 37,285,491 reference unit, never a
  nominal shared scale.
* **Two frozen static exams, never averaged.** `CVS-DEEP` = `E0004`/`D0010`, 400k nodes, 200 held-out
  identities (the in-lab exam `X####` binds). `SF-DEEP` = `E0005`/`D0040`, 4M nodes/label, **the same
  200 identities**. Both losses are carried per student; neither exam is called ground truth, and
  absolute losses are not compared across exams (different target distributions).
* **The dual-exam pass is validated, not trusted.** The lab's `Run` carries one exam, so
  `tools/s13/s13_analyse.py dual` scores every stored model on both instruments with the lab's own
  `nnue.evaluate`, and refuses to proceed unless every recomputed CVS-DEEP loss equals the run's own
  recorded metric exactly.

## Teacher disagreement (descriptive, before any training)

Written by `tools/s13/s13_labels.py disagreement` into `tools/s13/s13-disagreement.json` (hash
`sha256:47ac1cfb…`). Mate-sentinel rows (`|cp| ≥ 100k`) are excluded from cp statistics and counted.
This is characterization, never a filter.

| comparison | positions | mean \|Δcp\| | median | p90 | sign disagreement | best-move disagreement |
|---|---:|---:|---:|---:|---:|---:|
| CVS-4k vs SF-32k | 18,034 usable | **106.6** | 77 | 230 | 10.4 % | **63.5 %** |
| CVS-4k vs SF-1M | 17,454 usable | **132.6** | 96 | 278 | 10.3 % | **63.5 %** |
| SF-32k vs SF-1M (same teacher) | 17,454 usable | 46.9 | — | — | 1.6 % | 29.9 % |
| CVS-400k vs SF-DEEP (the 200 exam identities) | 199 usable | 69.0 | — | — | 6.5 % | not recorded on the CVS side |

Mate-sentinel counts differ sharply too: on the training rows CVS-4k marks 346 positions as mate
while SF-32k marks 910 and SF-1M 1,496. Two engines that disagree on the best move in two of three
positions, on 100+ centipawns of value, and by 10 % on the sign are not two rulers for one quantity
— which is exactly the worldview question the student experiment then tests.

## Results

`d` positive = the shallower/cheaper teacher's student is BETTER (d = loss(SF-32k) − loss(CVS-4k)).
Bounds are separate one-sided 95 % exact-t bounds at each width's df (19). The two exam columns are
separate measurements: absolute losses are **not** comparable across exams (different target
distributions), only the paired differences within an exam are.

| width | exam | mean CVS-4k-trained | mean SF-32k-trained | d | bounds | verdict |
|---|---|---:|---:|---:|---|---|
| H1 | CVS-DEEP | 0.016270 | 0.018841 | **+0.002568** | [+0.002047, +0.003089] | SF worse |
| H1 | SF-DEEP | 0.026245 | 0.029196 | **+0.002956** | [+0.002420, +0.003492] | SF worse |
| H4 | CVS-DEEP | 0.016261 | 0.018836 | **+0.002579** | [+0.001918, +0.003241] | SF worse |
| H4 | SF-DEEP | 0.026943 | 0.028993 | **+0.002051** | [+0.001372, +0.002729] | SF worse |
| H16 | CVS-DEEP | 0.017991 | 0.022343 | **+0.004353** | [+0.003360, +0.005347] | SF worse |
| H16 | SF-DEEP | 0.029287 | 0.031557 | **+0.002267** | [+0.000933, +0.003602] | SF worse |
| H32 | CVS-DEEP | 0.019040 | 0.023703 | **+0.004663** | [+0.003700, +0.005627] | SF worse |
| H32 | SF-DEEP | 0.030305 | 0.032866 | **+0.002554** | [+0.001479, +0.003628] | SF worse |

Means per condition and exam (all 20 seeds):

| condition | CVS-DEEP H1 / H4 / H16 / H32 | SF-DEEP H1 / H4 / H16 / H32 |
|---|---|---|
| CVS-4k | 0.01627 / 0.01626 / 0.01799 / 0.01904 | 0.02624 / 0.02694 / 0.02929 / 0.03031 |
| SF-32k | 0.01884 / 0.01884 / 0.02234 / 0.02370 | 0.02919 / 0.02899 / 0.03156 / 0.03287 |
| SF-1M | 0.02004 / 0.01973 / 0.02362 / 0.02521 | 0.03021 / 0.03026 / 0.03282 / 0.03435 |

**Dose (does more teacher compute help?).** SF-1M − SF-32k is positive on both exams at every width
(strictly, at every width on both), so spending 31× more teacher compute than SF-32k and 496× more
than CVS made the student *worse*. The full dose contrasts are in `tools/s13/s13-result.json`.

**Interaction (the worldview component).** `exam_effect(T) = loss_SF-DEEP(T) − loss_CVS-DEEP(T)`;
interaction = `exam_effect(CVS-trained) − exam_effect(SF-trained)`. Positive = the CVS-trained
student is relatively worse on the SF exam than the SF-trained student is, i.e. part of each
student's fit is authority-specific:

| width | interaction | bounds |
|---|---:|---|
| H1 | −0.000388 | [−0.000549, −0.000227] |
| H4 | +0.000529 | [+0.000059, +0.000999] |
| H16 | +0.002086 | [+0.001524, +0.002648] |
| H32 | +0.002110 | [+0.001527, +0.002692] |

So there **is** a measurable own-authority component at H4/H16/H32 (each student is relatively
better on its own teacher's exam) — but it is an order of magnitude smaller than the absolute CVS
advantage, and it never turns the cross-exam comparison around. No width shows the clean
"CVS better on CVS-DEEP, SF better on SF-DEEP" pattern (`widths_own_authority: []`).

**Decision.** Preregistered vocabulary: `WORLDVIEW_DISAGREEMENT` if the own-authority pattern held
at every width; `GENERIC_SF_GAIN` if the SF student were strictly better on both exams at every
width; otherwise `INCONCLUSIVE`. The realized pattern is neither, so the sealed verdict is
**INCONCLUSIVE** — and the third pattern is reported beside it, never substituted for it:
`observed_pattern = CVS_BETTER_ON_BOTH_EXAMS` (4/4 widths, strict), `dose arm worse` (4/4 widths).

## Why might a deeper teacher train a worse student? A candidate, not a mechanism

The label distributions differ in a way that a bounded student feels (all measured, none of it used
to select anything):

| supervision | mate sentinels | mean \|cp\| | share \|cp\| > 500 | cp sd |
|---|---:|---:|---:|---:|
| CVS-4k | 1.9 % | 288.9 | 19.9 % | 364 |
| SF-32k | 4.8 % | 319.0 | 22.8 % | 392 |
| SF-1M | 7.9 % | 336.4 | 25.6 % | 440 |
| SF-DEEP exam | 0.5 % | 224.4 | 13.1 % | 313 |

The deeper the teacher, the more saturated and mate-heavy its opinions on this position set — and
the loss is a K=256 tanh-squashed cp fit, so saturated targets ask a small student for outputs it
cannot represent, on 5–8 % of rows. This is a **plausible covariate consistent with the result**,
not a demonstrated mechanism: S13 varies authority and budget together (by design, since equal
spend was unreachable) and does not isolate label shape from label authority. A successor could
hold the label distribution fixed (e.g. clamp or subsample mate rows, or rank-normalize targets)
and ask whether the CVS advantage survives.

## What this does and does not say

* It says: on this population, with this student family and compute, **teacher authority changed the
  student, and the cheaper authority won on both instruments** — a static-loss result, not a
  statement about engine strength, playing strength, or which engine is "right".
* It does **not** say Stockfish is a bad teacher in general. The SF conditions were not converged
  (calibration: no rung achieved 0.99 best-move agreement with 2M), the target distribution is
  sharper than the exams' own, and the student is tiny (RAW-768, H ≤ 32, 194,560 presentations).
* It does not average the two exams, and it does not call either one ground truth.
* Mate-sentinel rows (`|cp| ≥ 100k`) are counted and reported; the exam splits contain 1 such row of
  200 in each instrument.

## Non-claims

* No playing-strength claim, no games, no GEO/HYBRID inputs, no outcome or blended targets, no
  per-arm optimizer tuning, no post-hoc teacher-budget tuning (the budget was chosen from the
  outcome-blind calibration and amended, in writing, before any student existed).
* One population (18,953 positions of one frozen order), one student family, one exam pair, 20
  seeds, one machine. Nothing here generalizes to other pools, other students, or other engines
  without its own experiment.
* The lab's `Run` carries one exam; the second instrument is applied by a recorded pass that
  reproduces every run's in-lab metric exactly before any SF number is believed
  (`tools/s13/s13-dual-exam.json`, validated for all 240 cells).

## Evidence bundle

| artifact | path |
|---|---|
| preregistration + hash | `tools/s13/s13-preregistration.json` / `.hash` |
| calibration (failed rule) + amendment | `tools/s13/s13-calibration.json`, `s13-calibration-amendment.json` |
| disagreement characterization | `tools/s13/s13-disagreement.json` |
| driver | `tools/s13/s13_run.py` (`exams` \| `preregister` \| `experiment` \| `run`) |
| dual-exam pass + analysis + seal | `tools/s13/s13_analyse.py` (`dual` \| `verify` \| `report` \| `seal`) |
| state / result / summary | `tools/s13/s13-state.json`, `s13-result.json`, `s13-result-summary.json` |
| Stockfish teacher (lab module + tests) | `cvslab/funnel/stockfish.py`, `tests/test_funnel_stockfish.py` |
| sealed lattice | `labstore/X/X0009.json` (`SEALED`, membership 240/240) |
