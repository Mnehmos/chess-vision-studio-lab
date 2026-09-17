# S13 — Teacher authority: CVS vs Stockfish under dual frozen static exams

Issue [Mnehmos/chess-vision-studio-lab#40](https://github.com/Mnehmos/chess-vision-studio-lab/issues/40),
the first narrow slice of #20.

**Two experiments, one finding.** `X0009` (240/240) measured the comparison first; its Stockfish
supervision was **not cold per label**, so the dose contrast mixed node budget with hidden
search-state history. `X0009` is preserved, sealed and **superseded for causal claims about teacher
budget** (`tools/s13/s13-x0009-supersession.json`). **`X0010`** is the replacement, sealed with
`record_hash sha256:c228700f7458c90ddca14c87496196eb0c85aa88696d4f2516b946ebb3ddb788`,
**240/240 cells COMPLETED**, 0 INVALID, preregistration
`tools/s13/s13-preregistration-v2.json` hash `sha256:ac8be438295ae7d3e78fa715de8ecc211d08e308d138abc91ec388987ef0fc1f`.
The cold rerun reproduces the finding.

**Answer.** Changing the teacher changes what the student learns — and here it changed it for the
worse, under a contract where the only difference between arms is *who* produced the supervision and
*at what price*. With identical positions, identical rows, identical student compute and identical
seeds, the **cheap legacy CVS teacher produced the better static student on BOTH exams at every
width**, including the exam built from the modern engine's own supervision:

| width | CVS-DEEP: SF-32k − CVS-4k | SF-DEEP: SF-32k − CVS-4k |
|---|---:|---:|
| H1 | +0.002435 [+0.001877, +0.002993] | +0.002690 [+0.002119, +0.003260] |
| H4 | +0.002321 [+0.001666, +0.002976] | +0.002005 [+0.001285, +0.002724] |
| H16 | +0.003763 [+0.002837, +0.004688] | +0.001841 [+0.000475, +0.003206] |
| H32 | +0.004196 [+0.003097, +0.005294] | +0.002302 [+0.001158, +0.003447] |

Positive = the Stockfish-trained student is worse. **The dose arm confirms the direction: SF-1M,
which spends 495.7× CVS-4k's teacher nodes, is worse than SF-32k on both exams at every width.**
A smaller own-authority component exists — each student is *relatively* better on its own teacher's
exam at H16/H32 (+0.0019, bounds excluding 0) — but it never turns the cross-exam comparison around.
The preregistered verdict vocabulary named only two patterns, so the sealed `decision` stays
**INCONCLUSIVE** and the realized third pattern is reported beside it:
`observed_pattern = CVS_BETTER_ON_BOTH_EXAMS` (4/4 widths, strict).

## The confound that forced a replacement (recorded, not patched over)

The first provider sent `position fen …` then `go nodes …` and never cleared the engine's search
state between labels, while one process served a whole worker chunk. **Stockfish's `position`
command does not clear the transposition table**: it runs `Engine::set_position`
(`src/uci.cpp`, `UCIEngine::position`), while only `ucinewgame` and `setoption name Clear Hash`
reach `Engine::search_clear` (`src/uci.cpp:135/281/350/410`, `src/engine.cpp:99-100`). Ordinary UCI
`position` does not establish search independence. So a label was a function of
`(position, budget, preceding positions in that worker)`, and because the two arms used different
worker counts (8 for 32k, 12 for 1M) the dose contrast changed *both* node budget and hidden
search-state partition.

It is not theoretical: on 8 identical positions at the pinned engine and budget, **5 of 8 cp values
differ between warm and cold at 32k nodes, and 8 of 8 at 1M**. X0009's SF evidence is therefore
preserved as superseded evidence (`tools/s13/s13-x0009-supersession.json`, hash `sha256:0beaa8a7…`),
and X0010 replaces it under an explicit contract.

## The cold contract

* `ucinewgame` → `setoption name Clear Hash` → `isready` → **`readyok` is the sync point** (the
  engine notes a clear "may take a while", so the next `go` must not race it) → `position fen` →
  `go nodes N`. Cold is the provider default; `cold=False` is opt-in and every row it writes is
  marked `WARM`.
* Regression test proves the invariant rather than trusting it:
  `tests/test_funnel_stockfish.py::test_every_label_is_preceded_by_a_search_state_reset` drives a
  fake UCI engine that records a transcript, and fails if any labelled `go` is not preceded by the
  full reset triplet. Eight provider tests run in CI with no Stockfish binary present.
* **The worker partition is proven irrelevant**, not assumed:
  `tools/s13/s13-cold-worker-check.json` labels the same 64 positions with **4** and with **12**
  workers and requires identical `(cp, best move, nodes)` — 0 differences. The calibration
  re-checks this at 32k, 1M and 4M (`determinism_across_worker_counts: true`).
* Every cold label is registered under a distinct authority, `oracle.stockfish.18.cold`, so the
  warm streams remain in the store, matchable by their own authority and never silently mixed with
  the replacement's supervision.

## The teachers, pinned by bytes

| | **CVS-4k** | **SF-32k (cold)** | **SF-1M (cold)** |
| authority | `legacy.cvs.search.shallow` | `oracle.stockfish.18.cold` | `oracle.stockfish.18.cold` |
| producer | `36e9b159…` (analyze build) | `c86215fa…` (binary SHA256) | `c86215fa…` |
| per-label budget | `{"nodeBudget": 4000}` | `{"nodes": 32000}` | `{"nodes": 1000000}` |
| contract | cold single query (analyze `--serve`) | cold per label | cold per label |
| ms/label (as purchased) | 7.5 | 79.7 | 1439 |
| realized/label | 3,939 nodes | 31,636 nodes | 975,075 nodes |
| realized depth | 4–6 | 18.6 | 28.7 |
| total teacher spend | 74,570,561 (**2.0×**) | 599,592,682 (**16.1×**) | 18,480,595,717 (**495.7×**) |

`×` = the arm's realized teacher spend ÷ the 37,285,491 reference unit. ms/label is the mean wall
clock *under the purchase's own parallelism* (12 workers); the uncontended ladder is in the
calibration artifact (32k 56 ms, 128k 160 ms, 512k 577 ms, 1M 1115 ms, 4M 4646 ms).

Stockfish identity is recorded as bytes and handshake, not a nickname: `Stockfish 18`, binary
SHA256 `c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911`, options
Threads 1 / Hash 16 / MultiPV 1, and **both** embedded nets (`nn-c288c895ea92.nnue` 125 MiB,
`nn-37f18f62d772.nnue` 6 MiB) — SF 18 picks the small net per position
(`Eval::use_smallnet`: `|simple_eval| > 962`), which the identity records instead of hiding.
`{"nodes": N}` is SF's budget key and can never subset-match CVS's `{"nodeBudget": N}`.

## Calibration (cold), and the rule that refused to invent a budget

The ladder was re-measured cold on the same frozen 64-position set against a 2M reference
(`tools/s13/s13-calibration-cold.json`, hash `sha256:318ecf41…`):

| budget | sign vs 2M | best move vs 2M | mean \|Δcp\| | depth | wall/label |
|---|---:|---:|---:|---:|---:|
| 32k | 0.983 | 0.848 | 42.9 | 18.0 | 56 ms |
| 128k | 0.966 | 0.864 | 32.8 | 21.2 | 160 ms |
| 512k | 0.983 | 0.915 | 22.9 | 24.9 | 577 ms |
| 1M | 0.966 | 0.932 | 13.0 | 27.1 | 1115 ms |
| 4M | 0.983 | 0.898 | 11.3 | 34.7 | 4646 ms |

**No rung converges**: even 4M-vs-2M best-move agreement is 0.898. Equal labeling spend is also
unreachable in a useful range (CVS-4k 7.5 ms/label vs the cheapest SF rung 56 ms). The applied rule
therefore took its amendment branch — cheapest rung as the primary condition, 1M as the dose
condition, 4M as the SF-DEEP exam budget — and the artifact records *which branch ran and why*
rather than pretending a converged teacher was chosen.

## Design

* **One population, one row set.** The 18,953 rows of `D0036` for every condition; a missing teacher
  label fails closed. The CVS condition reuses `D0036` unchanged and its TargetSpec hash is asserted
  equal to the sealed X0008 4k arm's training spec.
* **Identical student compute.** `MAX_UPDATES 760 × BATCH 256 = 194,560` presentations, T0004-derived
  recipe, RAW-768, H ∈ {1, 4, 16, 32}, seeds 0–19, one supervision-divergence declaration.
* **An explicit, non-grid cell list.** The three conditions have honestly unequal teacher spend, so
  each arm declares its own scale value instead of inheriting a nominal shared budget (the Lab
  gained `declared_cells` for exactly this, with refusal tests for missing/undeclared/repeated cells).
* **Two frozen static exams, never averaged.** `CVS-DEEP` = `E0004`/`D0010` (400k nodes, 200 held-out
  identities; the in-lab exam `X0010` binds). `SF-DEEP` = `E0006`/`D0043` (cold Stockfish at 4M
  nodes/label, **the same 200 identities**). Absolute losses are not comparable across exams.
* **The dual-exam pass validates itself.** The Lab's `Run` carries one exam, so the SF instrument is
  applied to the stored weights by a recorded pass that **refuses to proceed unless it reproduces
  every run's recorded CVS loss exactly** — it does, for all 240 cells.

## Teacher disagreement (descriptive, before training)

`tools/s13/s13-disagreement.json` (hash `sha256:92afacdd…`); mate-sentinel rows (`|cp| ≥ 100k`)
excluded from cp statistics and counted; correlation on non-mate paired rows. The expected-score
transform is the Lab's own supervision target, `sigmoid(cp / K)`, K = 256 — the quantity a student
is trained to predict.

| comparison (cold) | pairs | mean \|Δcp\| | sign dis. | best-move dis. | Pearson cp | Spearman cp | Pearson expected |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVS-4k vs SF-32k | 18,027 | 105.8 | 10.5 % | **63.2 %** | 0.914 | 0.918 | 0.920 |
| CVS-4k vs SF-1M | 17,467 | 131.6 | 10.4 % | **63.3 %** | 0.871 | 0.907 | 0.907 |
| SF-32k vs SF-1M (same engine) | 17,472 | 45.9 | 1.4 % | 25.8 % | 0.943 | **0.995** | 0.995 |
| CVS-400k vs SF-DEEP (the 200 exam ids) | 199 | 69.7 | 6.5 % | not recorded on the CVS side | 0.967 | 0.953 | 0.958 |

Two authorities that rank positions similarly (r ≈ 0.91) still disagree on the best move in **two of
three** positions and on the sign in one of ten. The same engine at 16× more nodes keeps the ranking
almost perfectly (Spearman 0.995) while shifting the level (mean |Δcp| 45.9) — the deep teacher is a
different *rule* from CVS far more than it is a different *ordering*.

## Results (X0010, cold)

`d = loss(SF-32k) − loss(CVS-4k)`, paired by seed, separate one-sided 95 % exact-t bounds (df 19):

| width | exam | mean CVS-4k-trained | mean SF-32k-trained | d | bounds | verdict |
|---|---|---:|---:|---:|---|---|
| H1 | CVS-DEEP | 0.016270 | 0.018706 | **+0.002435** | [+0.001877, +0.002993] | SF worse |
| H1 | SF-DEEP | 0.026594 | 0.029284 | **+0.002690** | [+0.002119, +0.003260] | SF worse |
| H4 | CVS-DEEP | 0.016261 | 0.018582 | **+0.002321** | [+0.001666, +0.002976] | SF worse |
| H4 | SF-DEEP | 0.027280 | 0.029285 | **+0.002005** | [+0.001285, +0.002724] | SF worse |
| H16 | CVS-DEEP | 0.017991 | 0.021754 | **+0.003763** | [+0.002837, +0.004688] | SF worse |
| H16 | SF-DEEP | 0.029613 | 0.031454 | **+0.001841** | [+0.000475, +0.003206] | SF worse |
| H32 | CVS-DEEP | 0.019040 | 0.023236 | **+0.004196** | [+0.003097, +0.005294] | SF worse |
| H32 | SF-DEEP | 0.030577 | 0.032879 | **+0.002302** | [+0.001158, +0.003447] | SF worse |

Means per condition and exam (all 20 seeds; not comparable across exams):

| condition | CVS-DEEP H1 / H4 / H16 / H32 | SF-DEEP H1 / H4 / H16 / H32 |
|---|---|---|
| CVS-4k | 0.01627 / 0.01626 / 0.01799 / 0.01904 | 0.02659 / 0.02728 / 0.02961 / 0.03058 |
| SF-32k (cold) | 0.01871 / 0.01858 / 0.02175 / 0.02324 | 0.02928 / 0.02928 / 0.03145 / 0.03288 |
| SF-1M (cold) | 0.01993 / 0.01959 / 0.02405 / 0.02523 | 0.03047 / 0.03028 / 0.03359 / 0.03486 |

**Dose.** SF-1M − SF-32k is positive on both exams at every width: spending 30.8× more teacher
compute than SF-32k, and 495.7× more than CVS-4k, made the student *worse* at every width on both
instruments. Full contrasts in `tools/s13/s13-result-cold.json`.

**Interaction.** `exam_effect(T) = loss_SF-DEEP(T) − loss_CVS-DEEP(T)`;
interaction = `exam_effect(CVS-trained) − exam_effect(SF-trained)`:

| width | interaction | bounds | reading |
|---|---:|---|---|
| H1 | −0.000254 | [−0.000405, −0.000104] | reversed at the narrowest model |
| H4 | +0.000316 | [−0.000150, +0.000782] | unresolved |
| H16 | +0.001922 | [+0.001391, +0.002453] | own-authority component, strictly positive |
| H32 | +0.001893 | [+0.001311, +0.002476] | own-authority component, strictly positive |

So at H16/H32 a measurable part of each student's fit is authority-specific — but it is an order of
magnitude smaller than the absolute CVS advantage, and it never makes the SF-trained student the
better one on any exam.

## Why might a deeper teacher train a worse student? A candidate, not a mechanism

| supervision | mate sentinels | mean \|cp\| | share \|cp\| > 500 | cp sd |
|---|---:|---:|---:|---:|
| CVS-4k | 1.9 % | 288.9 | 19.9 % | 364 |
| SF-32k (cold) | 4.9 % | 316.7 | 22.7 % | 393 |
| SF-1M (cold) | 7.8 % | 334.5 | 25.3 % | 436 |
| CVS-DEEP exam | 0.5 % | 171.0 | 8.0 % | 259 |
| SF-DEEP exam | 0.5 % | 223.2 | 13.6 % | 309 |

The deeper the teacher, the more saturated and mate-heavy its opinions on this position set, and the
loss is a K=256 squashed cp fit: saturated targets ask a tiny student for outputs it cannot
represent, on 5–8 % of rows. This is **consistent with** the result — X0009's warm/cold difference
also shows how sensitive a tiny student's fit is to small supervision perturbations — but S13 varies
authority and budget together by design and does **not** isolate label shape from label authority.
A successor can hold the distribution fixed (clamp or subsample mate rows, rank-normalize targets)
and re-ask.

## Errata on the sealed X0010 metadata

Two *metadata* fields in X0010's sealed result carried warm identifiers from a hard-coded string in
the report code: `cells.exams` cited `SF-DEEP (E0005/D0040)` where the cold contract froze
`E0006/D0043`, and `evidence.preregistration` pointed at `s13-preregistration.json` where X0010 is
bound to **v2** (`sha256:ac8be438…`). No number, bound, verdict, mean or membership is affected —
every per-width figure came from the cold dual-exam artifact, which records `E0006/D0043` correctly,
and the experiment's own `preregistration_hash` was always the v2 value. The sealed record and its
byte-identical artifact are **not** rewritten (`update_experiment` refuses a sealed study); the
correction is recorded additively in `tools/s13/s13-result-cold-erratum.json` (hash
`sha256:7070f111…`), the report code now derives both the cited exam identities (from the store) and
the preregistration path (from the contract), and `tests/test_s13_report_metadata.py` fails if a warm
identifier ever reappears as a code literal.

## Non-claims

* No Stockfish-as-truth, no playing-strength claim, no games, no GEO/HYBRID inputs, no outcome or
  blended targets, no per-arm optimizer tuning, no post-hoc teacher-budget tuning (the ladder was
  measured outcome-blind and the branch is recorded in the artifact).
* One population (18,953 positions of one frozen order), one student family (RAW-768, H ≤ 32,
  194,560 presentations), one exam pair, 20 seeds, one machine. Nothing here generalizes to other
  pools, students or engines without its own experiment.
* X0009's *statistical* result was internally coherent and points the same way; it is superseded
  only for causal claims about teacher budget, and it remains sealed and byte-identical.
* The CVS teacher's own single-query analyze protocol is not re-litigated here; this review's
  finding concerns the Stockfish provider.

## Evidence bundle

| artifact | path |
|---|---|
| preregistration v2 + hash (X0010's contract) | `tools/s13/s13-preregistration-v2.json` / `.hash` |
| preregistration v1 + hash (X0009, superseded) | `tools/s13/s13-preregistration.json` / `.hash` |
| X0009 supersession record | `tools/s13/s13-x0009-supersession.json` / `.hash` |
| cold calibration + warm calibration + amendment | `s13-calibration-cold.json`, `s13-calibration.json`, `s13-calibration-amendment.json` |
| worker-partition proof | `tools/s13/s13-cold-worker-check.json` |
| disagreement with correlation | `tools/s13/s13-disagreement.json` |
| driver / analysis | `tools/s13/s13_run.py`, `tools/s13/s13_analyse.py` |
| state / results / dual-exam table | `s13-state.json`, `s13-result-cold.json`, `s13-result-summary-cold.json`, `s13-dual-exam-cold.json` |
| metadata erratum (sealed X0010 result) | `tools/s13/s13-result-cold-erratum.json` / `.hash` |
| Stockfish teacher + tests | `cvslab/funnel/stockfish.py`, `tests/test_funnel_stockfish.py`, `tests/fake_uci_engine.py` |
| sealed lattices | `labstore/X/X0010.json` (`SEALED`, 240/240 — the finding), `X0009` (sealed, superseded) |
