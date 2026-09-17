# S14 — Representation: RAW vs GEO vs HYBRID at matched learned-parameter budgets (sealed)

Experiment **X0012**, sealed with `record_hash` recorded in `labstore/X/X0012.json`, **260/260 cells
COMPLETED**, 0 INVALID, 33 cells carrying a recorded crash attempt replaced by its retry
(`replaced_cells` in the membership: the crash stays immutable evidence, the retry counts).
Preregistration: `tools/s14/s14-preregistration.json`, hash `sha256:39eb793bfaf8a0999df967b3ff1e59f408684f8bcc8013d1f048764196429df9`.
Issue [Mnehmos/chess-vision-studio-lab#41](https://github.com/Mnehmos/chess-vision-studio-lab/issues/41),
the controlled slice of #18.

**Answer, in three sentences.** Named deterministic chess structure **cannot replace** learned
capacity — GEO alone loses to RAW by ~+0.007 exam loss at every matched budget. But explicit
geometry **adds information the raw network does not recover efficiently**: HYBRID beats RAW
strictly from the 3K budget up (−0.0031 at 3K growing to −0.0044 at 25K), for the cost of 42 extra
input columns. And the white-box floor is startlingly high: **43 readable parameters** (one signed
weight per named registry column, no hidden layer) reach 0.02498 exam loss — better than the
25,000-parameter hidden GEO model (0.02634).

## The contract (one variable changes)

Supervision and exam are S13's selected CVS side, byte-identical: teacher = **CVS-4k over D0036**
(the 18,953 rows; TargetSpec hash asserted equal to X0008's sealed 4k arm), exam = **E0004**
(CVS-DEEP 400k, the 200 held-out identities). Student compute is fixed for every cell
(MAX_UPDATES 760 × BATCH 256 = 194,560 presentations), T0004-derived recipe, seeds 0–19, cold
starts, no per-family tuning. **The only changing variable is the input representation** (and the
matched parameter budget along the ladder).

## The three representations

| | columns | what they are |
|---|---:|---|
| **RAW** | 768 | piece-square planes, side-to-move relative (colour swap + mirror) |
| **GEO** | 42 | two named columns per registry family: the White-POV signed delta ÷ the family's **own** bucket-3 threshold (the registry's magnitude unit), and its 0–3 bucket ÷ 3; sign-flipped for black to move |
| **HYBRID** | 810 | RAW ++ GEO in one input vector |

The GEO vocabulary is **not invented for this study**: it is `cvslab.facts`' frozen geometry
registry (21 families, registry hash `sha256:a91c902a…`), and every column resolves to a named
`(family, kind)` entry in `tools/s14/s14-geometry-manifest.json` (hash `sha256:a35dd752…`). There is
no missingness path — the registry emits every family for every position — and no label, teacher
score, best move or outcome enters the representation: it is a pure function of the position.
Architectures: `crelu1` (inputs → H clipped-ReLU → 1) and `linear` (inputs → 1, the semantic floor).

## The matched ladder (generated from the code, frozen before outcomes)

| budget target | RAW (H) | GEO (H) | HYBRID (H) | exact params RAW / GEO / HYBRID |
|---|---|---|---|---|
| ~1K | 771 (1) | 1,013 (23) | 813 (1) | 771 / 1013 / 813 |
| ~3K | 3,081 (4) | 2,993 (68) | 3,249 (4) | 3081 / 2993 / 3249 |
| ~12K | 12,321 (16) | 12,013 (273) | 12,181 (15) | 12321 / 12013 / 12181 |
| ~25K | 24,641 (32) | 24,993 (568) | 25,173 (31) | 24641 / 24993 / 25173 |
| linear floor | — | **43** (linear) | — | 43 |

H is the exact-count nearest neighbour of each target under the formula `inputs·H + 2H + 1`
(`linear`: `inputs + 1`), computed by `tools/s14/s14_common.py::matched_h` — never hand-entered.
Budget mismatch is reported beside every comparison (largest: −359 params).

## The frontier

Exam loss on E0004 (CVS-DEEP, 200 identities), mean over 20 seeds. Contrasts are paired by seed
with separate one-sided 95 % exact-t bounds (df = 19); positive = the first family is **worse**.

| budget | RAW | GEO | HYBRID | GEO − RAW | HYBRID − RAW | HYBRID − GEO |
|---|---:|---:|---:|---|---|---|
| 1K | 0.016271 | 0.023049 | 0.017685 | +0.0068 [+0.0059, +0.0076] | **+0.0014** [+0.0001, +0.0027] | −0.0054 |
| 3K | 0.016261 | 0.023681 | **0.013181** | +0.0074 [+0.0065, +0.0083] | **−0.0031** [−0.0041, −0.0021] | −0.0105 |
| 12K | 0.017991 | 0.025354 | **0.013851** | +0.0074 [+0.0066, +0.0081] | **−0.0041** [−0.0051, −0.0031] | −0.0115 |
| 25K | 0.019040 | 0.026335 | **0.014684** | +0.0073 [+0.0065, +0.0081] | **−0.0044** [−0.0056, −0.0032] | −0.0117 |
| floor (43 p) | — | 0.024978 | — | | | |

Every interval excludes zero. The family-by-budget frontier — no global winner score:

1. **GEO never replaces RAW.** ~+0.007 at every budget: 21 named families carry real signal (the
   43-parameter floor proves it) but nowhere near what 768 piece-square columns carry.
2. **HYBRID strictly beats RAW from 3K up**, and the gap *widens* with budget. Adding the 42 named
   columns is the cheapest capacity in the study.
3. **The crossover is real and non-monotone**: at the 1K budget HYBRID is slightly *worse* than RAW
   (+0.0014, strictly) — with too few hidden units the extra inputs cost more than they give.
4. **RAW overfits the fixed-compute regime**: its exam loss *rises* from 0.0163 (3K) to 0.0190 (25K)
   at constant 194,560 presentations; HYBRID's optimum sits at 3K (0.0132). More parameters at
   fixed student compute are not free for any family.
5. **Phase structure**: HYBRID improves RAW in every phase at 12K, with the largest relative gain in
   the endgame (0.0441 → 0.0330); the linear floor is weakest exactly there (0.0606).

## The white-box floor (Stage A)

The direct linear model — 43 parameters, one signed weight per named column, fully readable —
reaches **0.02498**. For calibration: the hidden GEO model with **581× more parameters** (25K) is
*worse* (0.02634). The report distinguishes the three observability levels explicitly:
named-input observability (GEO-LINEAR: inputs *and* weights readable), named inputs with an opaque
hidden layer (GEO/HYBRID crelu1 — never called interpretable here), and full opacity (RAW).

## What semantic structure costs at inference

| measurement | rate |
|---|---:|
| RAW feature encoding | 187,755 positions/s |
| GEO feature encoding | 5,652 positions/s (**33× slower**) |
| RAW forward pass (25K) | 434,665 positions/s |
| HYBRID forward pass (25K) | 396,627 positions/s (−9 %) |

Measured in the python implementation as shipped (`extract_geometry` does real per-position work:
attack maps, pawn files, king zones). A compiled implementation would move both numbers, but the
*work ratio* is intrinsic: constructing named structure costs far more than setting 768 indices,
while the forward pass barely notices 42 extra columns.

## Secondary semantic ablation (preregistered sentinel: HYBRID 12K, seed 0)

Input lesions at inference on the frozen weights (base loss 0.011893; no retraining, no
combinatorial ablations):

| group (families) | columns | effect of removal |
|---|---:|---:|
| bishop_pair (1) | 2 | **+0.00157** |
| mobility (4) | 8 | +0.00107 |
| hanging_material (1) | 2 | +0.00101 |
| king_safety (8) | 16 | +0.00006 |
| pawn_structure (4) | 8 | −0.00020 |
| rook_files (3) | 6 | −0.00006 |

The fitted evaluator actually *uses* the bishop pair, mobility and hanging-material columns; the
eight king-safety families are nearly unused at this scale, and removing pawn-structure or
rook-file columns minutely *helps* (small fitted noise). All six results are preserved as measured,
including the negative ones.

## Costs and artifacts (per model, 12K budget)

Train wall time: RAW 7.2 s / GEO 8.4 s / HYBRID 8.6 s (mean per run). Serialized size:
RAW 266,753 B / GEO 260,395 B / HYBRID 264,329 B. Train-vs-exam distribution gaps
(train-split loss minus exam loss; no held-out validation split exists, so this is a distribution
gap, not a classic overfit gap): RAW −0.0140 / GEO −0.0043 / HYBRID −0.0093 — every family fits the
exam distribution better than the training rows.

## Non-claims

* Static-loss results only; no search, playing-strength or engine claims.
* Teacher authority does not change inside S14 (CVS-4k throughout); no auxiliary geometry
  prediction loss (S15); no active learning.
* One population (18,953 positions), one exam, one optimizer family, 20 seeds, python-measured
  inference. The 33 replaced cells are crashes from an encoding bug found and fixed mid-run
  (empty validation split), never silently dropped — each crash run remains in the store beside its
  retry.
* GEO's vocabulary is the frozen registry's; no feature was added, tuned or dropped for this study.

## Evidence bundle

| artifact | path |
|---|---|
| preregistration + hash | `tools/s14/s14-preregistration.json` / `.hash` |
| geometry manifest + hash | `tools/s14/s14-geometry-manifest.json` / `.hash` |
| driver (manifest \| preregister \| experiment \| run) | `tools/s14/s14_run.py` |
| shared contract + ladder generator | `tools/s14/s14_common.py` |
| analysis + seal | `tools/s14/s14_analyse.py` |
| state / result / summary | `tools/s14/s14-state.json`, `s14-result.json`, `s14-result-summary.json` |
| representations + linear arch | `cvslab/nnue.py`, `cvslab/families.py` |
| tests | `tests/test_nnue_representation.py`, `tests/test_experiment.py` (replacement semantics) |
| sealed lattice | `labstore/X/X0012.json` (`SEALED`, 260/260, 33 replaced cells recorded) |
