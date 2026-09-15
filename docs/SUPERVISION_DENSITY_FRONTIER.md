# S7 — The Supervision-Density Frontier

Status: **design and preregistration. Not implemented, not launched, no identity created.**
Companions: `docs/RESEARCH_PROTOCOL.md`, `docs/DATA_LIFECYCLE.md`,
`docs/TRAINING_DATA_RESEARCH_PLAN.md`, `tools/s6/s6-campaign-result-final.json`,
`tools/s6/s6-preregistration-amendment-{1,2}.json`, `tools/s7/s7-preregistration.json`.

Every number below is one of three things, and each is labelled:

* **[M]** measured — recorded in a lab artifact or a committed engine artifact;
* **[D]** documented — written in a repository document;
* **[E]** estimate — derived arithmetic from [M]/[D] numbers, or a plan value.

Nothing in this document is from recollection. Source paths are given for every quantity.

---

## 0. The question

The platform can now, cheaply and deterministically:

* generate broad, provenance-carrying position populations (S5);
* compute exact geometry/motif/strategy facts for essentially every position
  (`cvslab/facts.py`, mirroring the engine's Feature Registry v1);
* run any node-budget search on any subset and keep the layered result as canonical,
  append-only labels (S1–S4).

The open research question is **how much expensive search supervision is actually
needed** once broad coverage and deterministic semantics are cheap:

> At a fixed teacher/search-compute budget, is it better to train from a small number
> of very accurate/deep labels, or a much larger number of cheaper/shallow labels?

S6 answered a different question (selection, not depth) and answered it negatively.
S7 is the density frontier — the first experiment that varies *label quality per row*
against *row count* at constant teacher compute.

---

## 1. Two training-data architectures, compared

### 1.1 The intended layered CVS funnel

```
  positions (self-play / live / search distribution)
        │
        ├─ tier0  deterministic facts        ALL positions        ~6 ms each      [M]
        ├─ tier1  shallow CVS at 2k/16k      ALL positions        ~26 ms each     [M]
        │
        ├─ triage priority-v1 ranks ALL positions by information gain
        │
        ├─ tier3  deep CVS 400k              SELECTED subset      ~325 ms each    [M]
        └─ tier4  Stockfish d18 + 3 s cap    SMALL samples + all holdout
                                                              ~451 ms each        [M]
  (all four layers land in separate provenance classes and are kept, not collapsed)
```

What is *retained*: facts for every position, shallow telemetry for every position, a
deep label for the selected subset, an oracle label for a small subset, plus the
selection policy, its components and its hash. What is *discarded*: nothing. The pool's
duplicate multiplicity and game identity survive to normalization
(`cvslab/funnel/pool.py` module docstring; `S0004` → `N0005` records 1367 duplicates
**[M]**).

Intended consumers (README phases 1–3 **[D]**): RAW NNUE first, then GEO from the facts
layer, then matched RAW/GEO/HYBRID. The facts layer is the reason the funnel exists at
all: it is the only layer that gives *every* position a semantically meaningful target
without search.

### 1.2 What Gen10 actually did

```
  gen9 shard corpus (7,861,317 rows)                     [M] engine docs/GENERATIONS.md
        │
        └─ corpus-d20: self-play sampled every ply       599,363 rows / 43,536 unique  [M]
              └─ Stockfish 17.1, `go depth 20`, 1 thread  ~0.54 s/position             [M]
                    cp = White-POV SF score · cp_play = CVS play score · res = result
                    kept: fen, res, cp, cp_play, label_depth, label_nodes, sf_best,
                          features, features_bitset  (CVS core ids, registry v1)
        parallel gen10 corpora:
              corpus            113,207 rows  SF d16 (+ d12 shallow), stability-filtered  [M]
              corpus-static   1,500,000 rows  SF static eval                                [M]
              corpus-static-full 4,143,821 rows — but only 2,364,478 unique (43 % dup)     [M]
              corpus-lich     21,168,494 rows  Lichess eval DB d16                          [M]
```

Documented Gen10 facts **[D]** (`chess-vision-studio-rust-engine/training/gen10/README.md`,
`docs/GENERATIONS.md`, commit messages):

| fact | value |
|---|---|
| architecture | `768x256cReLU-1(cp)`, 197,120 serialized params, outputScaleCp 400 |
| target | `0.6·σ(cp/256) + 0.4·res` (CVS-NNUE) or `sigmoid/linear/`Huber on `cp` (raw CP variants) |
| recipe | Adam 1e-3, batch 16,384, 24–30 epochs, holdout `rows % 50 == 7` (row-level — the trainer itself refuses to run without `--allow-unsafe-row-split`, calling its own artifact not promotion-eligible) |
| outcome | "not yet shippable"; best measured r 0.886 vs incumbent 0.920; no gen10 gate record exists |
| provenance | the lab's intake catalog: 22 of 23 models lack `trainingCommit`/`datasetManifestHash` |

What Gen10 *retained*: `cp`, `res`, `cp_play`, per-row `label_depth`/`label_nodes`,
`sf_best`, CVS core feature ids. What it *discarded*: geometry beyond raw feature ids
(no bucket/Δ semantics), shallow−deep deltas as first-class labels, per-position
selection policy identity, and — in `corpus-static-full` — 43 % of the corpus was
duplicate rows, silently re-labelled (the loader does not dedup; the append-on-resume
path re-labelled FENs already on disk).

**The architectural difference is not the teacher.** It is that Gen10 bought one
expensive label for ~14 rows per unique position and kept only search-derived numbers,
while the funnel buys a *ladder* of labels per position, spends its expensive labels on
a subset, and keeps a deterministic semantic description of every position for free.
S6 tested the *selection* half of that claim and got a negative result. The *density*
half is untested.

### 1.3 Cost per unique position, both regimes

| regime | labels | unique positions | labels/unique | teacher time per unique position |
|---|---:|---:|---:|---:|
| Gen10 `corpus-d20` | 599,363 | 43,536 **[M]** | 13.8 | **7.44 s** (0.54 s × 13.8) **[E]** |
| Gen10 `corpus-static-full` | 4,143,821 | 2,364,478 **[M]** | 1.75 | 0.53 ms **[E]** (≈3,300 labels/s) |
| Funnel tier0 facts | 1 per position | 1 per position | 1.0 | 6.5 ms **[M]** |
| Funnel wide-shallow (S7 widest arm) | 1 per position | 1 per position | 1.0 | **19 ms** (16k nodes) **[E]** |
| Funnel narrow-deep (S7 narrowest arm) | 1 per position | 1 per position | 1.0 | **325 ms** (400k nodes) **[M]** |

The frontier is the first experiment that prices those rows against each other on a
common instrument.

---

## 2. What S6 established — and what it did not

**Established [M]** (`tools/s6/s6-campaign-result-final.json`):

* 40/40 cells COMPLETED, R0049–R0088, arms D0008/D0009 (94/94 rows, all-train), recipe
  T0004, instrument D0010/E0004 (200 records, 10 held-out openings), 5 paired seeds.
* Per-width paired effects (Priority − Uniform on `test_loss`):
  H1 +0.000092 [−0.003302, +0.003486]; H4 +0.003406 [−0.000253, +0.007065];
  H16 +0.006624 [+0.002125, +0.011124]; H32 +0.000796 [−0.001341, +0.002933].
* Decision: **INCONCLUSIVE** (no width favors PRIORITY; H16 significantly favors UNIFORM).
* Factory-level result: *how you select 94 positions out of this pool does not buy
  generalization at these capacities, and at H16 selective selection was worse.*

**Not established by S6** (each of these is a live hypothesis still):

1. **Geometry as input** — S6 was RAW-768 only. The facts layer was never an input.
2. **Shallow vs deep labels** — both arms received identical 400k supervision.
3. **Row count as such** — both arms had 94 rows.
4. **The layered funnel** — S6 tested one layer under one budget, not the ladder.
5. **Any claim about other pools** — preregistration scope says "this candidate pool
   only; independent-pool replication is #16".
6. Priority-v1's *training* value at other budgets, row counts, or with mixture designs.

S6's own numbers also say something encouraging about the machinery: the two arms'
realized deep-node totals differed by 0.79 % (36,990,516 vs 37,285,491 nodes **[M]**),
i.e. an equal-teacher-compute claim here is *measurable*, not assumed. S7 is built on
that capability.

---

## 3. Inventory — labels that exist right now, and what they cost

All of the following are on disk and reusable **without buying anything** **[M]**
(`labstore/canonical/N0005/labels/*`, `N0006/labels/*`, `labstore/objects/runs/R0008.json`).

| label | family / authority | budget | rows | measured nodes | engine seconds | ms/label |
|---|---|---|---:|---:|---:|---:|
| tier0 facts (engine shape) | `facts` / `legacy.cvs.tier0.facts` | — | 938 | 0 | 6.1361 | 6.54 |
| motif (engine shape) | `motif` / `legacy.cvs.tier0.tactics` | — | 938 | 0 | (tier0) | — |
| strategy (engine shape) | `strategy` / `legacy.cvs.tier0.structures` | — | 938 | 0 | (tier0) | — |
| priority | `priority` / `triage.priority-v1` | — | 938 | 0 | 0 | 0 |
| outcome | `outcome` / `outcome.game_result.v1` | — | 80 | 0 | 0 | 0 |
| shallow | `search_shallow_cp` / `legacy.cvs.search.shallow` | 2,000 | 938 | 1,871,572 | — | 2.4 |
| shallow | same | 16,000 | **938** | 14,940,202 | 20.0886 (both budgets) | 19.0 |
| deep | `search_deep_cp` / `legacy.cvs.search.deep` | 400,000 | 210 | 83,076,007 | 68.2984 | 325.2 |
| deep (instrument) | same | 400,000 | 200 | 79,600,000 | 68.65 | 343.3 |
| oracle | `oracle_cp` / `external.oracle.sf16` | depth 22 | 3 | — | — | — |

Measured efficiencies, used throughout this document's arithmetic **[M]**:
16k labels realized **99.55 %** of nominal (mean 15,927.7 nodes); 400k labels realized
**98.9 %** (mean 395,600.0 nodes); engine throughput 837k nodes/s at shallow budgets and
1.216M nodes/s at 400k.

Two facts make the pilot below possible:

* all **938** records of the S5 pool carry a 16k label, and **926** of them are
  train-eligible (the 12 hash-holdout records are excluded by `trainEligible=False`);
* the S6 arms already bought 400k labels for 210 records (198 train-eligible) and the
  instrument bought 200 more on a disjoint pool.

Target semantics are *usable and identical in currency* across the ladder: every
16k/deep label carries `scoreCpStm` at `pov="stm"`; they differ only in where the value
lives (`value.scoreCpStm` for shallow, `value.targets.scoreCpStm` for deep, because the
deep row also carries `shallowToDeep` and `to`-targets). A `TargetSpec` pins
family + authority + producer + budget + value_path + POV + type + K + LAMBDA, so a 16k
spec and a 400k spec are distinct, first-class identities
(`cvslab/targets.py`, `TARGET_SPEC_VERSION = 1`). The evaluation instrument already
pins its own spec by hash (`T0004`/`E0004`, spec `sha256:80e8e3de…`).

---

## 4. S7-P — the pilot (existing artifacts only, zero new labels)

### 4.1 Why it is worth running

The existing label inventory *already* contains a 2×2 over the two axes that matter:

* **rows** ∈ {94, 926} and **depth** ∈ {2k, 16k, 400k}

at these measured costs (nodes, **[M]** for the 94-row arms and 16k/2k rows):

| | 94 rows | 926 rows |
|---|---:|---:|
| **400k** | 37.29M nodes (D0009, exists) | impossible (only 198 train-eligible 400k labels) |
| **16k** | 1.50M nodes | **14.94M nodes** |
| **2k** | 0.19M nodes | **1.87M nodes** |

The four reachable cells are free of new labelling. They give:

* **P_depth** — rows held at 94, depth 16k vs 400k → *causal for depth at fixed rows*;
* **P_rows** — depth held at 16k, rows 94 vs 926 (nested; the 94 ⊂ the 926) →
  *causal for adding 832 rows at fixed depth*;
* **P_econ** — 926×16k vs 94×400k: 2.8× cheaper teacher bill, 9.9× more rows —
  the economic screening contrast. **Not matched-compute.**

None of the three is the frontier. All three are cheap, and P_econ is the first
empirical evidence about which end of the density axis looks promising.

### 4.2 Arms

| arm | rows | supervision | source | status |
|---|---:|---|---|---|
| P1 | 94 | 400k `search_deep_cp` | D0009 + T0004, cells **already exist** (S6 UNIFORM: R0054–R0058, R0064–R0068, R0074–R0078, R0084–R0088) | reuse |
| P2 | 94 (= P1's exact rows) | 16k `search_shallow_cp` | new D, existing labels | 20 new runs |
| P3 | 926 (= all train-eligible) | 16k | new D, existing labels | 20 new runs |
| P4 (exploratory) | 926 | 2k | new D, existing labels | 20 new runs |

Widths H∈{1,4,16,32}, seeds {0,1,2,3,4}, recipe parameters identical to T0004 except
`TARGET_SPEC`, instrument **E0004/D0010 unchanged** (disjointness re-verified below).

### 4.3 Preregistered reading

Per width, paired by seed, 95 % CI (Student t, df=4, t=2.776):

* **P_depth** = test_loss(P2) − test_loss(P1); **P_rows** = test_loss(P3) − test_loss(P2);
  **P_econ** = test_loss(P3) − test_loss(P1).
* Reporting rule: all three contrasts at all four widths are reported with CIs,
  including the case where widths disagree.
* Interpretation rule, decided now, before any number exists:
  * if **P_econ**'s CI upper bound < 0 at every width → "broad-shallow beats
    narrow-deep at 40 % of the teacher compute" — a strong economic signal that
    motivates S7-F and redirects the whole program toward the wide end;
  * if **P_econ**'s CI contains 0 → "no detectable difference at 40 % of the compute" —
    still an economic signal (the cheaper regime is not worse);
  * if **P_econ**'s CI lower bound > 0 at every width → depth is load-bearing even
    against 9.9× more rows.
* **Non-claims:** P is not matched-compute (P_econ); P3 is the *entire* train-eligible
  pool, not a sample (so P_rows is "add rows", not "sample more broadly"); P2/P3/P4 use
  a recipe whose spec differs from the instrument's — a divergence the service
  currently forbids (§5.8), so the pilot runs *after* prerequisite P1 is merged.
* Pilot results never enter the frontier's decision rule. They are a screen that tells
  us whether the frontier's answer is likely to be "wide". They cannot replace it.

### 4.4 Pilot cost

60 new training runs (P2/P3/P4), 0 new labels, 0 new engine work beyond training.
Reuses 20 completed S6 cells (P1) that used the identical instrument.

---

## 5. S7-F — the matched-compute supervision-density frontier

### 5.1 Design

One new frozen training pool. One deterministic uniform selection order. Four arms that
partition it into disjoint blocks, each buying *the same total number of teacher nodes*
at a different node budget. One instrument, four capacities, five paired seeds.
RAW-768 only.

```
  T = 37,285,491 teacher nodes  (measured; = S6 UNIFORM arm realized deep nodes [M])
  │
  ├─ A1  400k/row ×   94 rows   = 37.19M nodes   (deepest, narrowest)
  ├─ A2  160k/row ×  235 rows   = 37.22M
  ├─ A3   64k/row ×  588 rows   = 37.26M
  └─ A4   16k/row × 2341 rows   = 37.29M   (shallowest, widest)
                                   ────────
                                   149.1M nodes total ≈ 2.5–3 min of engine time [E]
```

Arm sizes above are **estimates**; the authoritative rule (§5.3) recomputes them from a
calibration measured on the new pool, materialized into
`tools/s7/s7-preregistration.json` **before** any arm label is bought or any training
starts.

### 5.2 Pool construction (new S/N identities)

| step | value | rationale |
|---|---|---|
| generator | `cvslab.pool.selfplay` v1, `DEFAULT_ENGINE_ARGS`, engine identity recorded truthfully via `engine_identity()` | the S6 provenance defect must not repeat; S0009 needed a byte-identical attestation to repair it |
| seed | `20260916` | fresh; independent of S5 (`20260915`) and the instrument (`777001`) |
| openings | built-in `OPENING_LINES` (12 lines) | disjoint from the instrument's 10 held-out opening lines; same opening-source hash as S5 `sha256:588f3688…`, so opening provenance is comparable |
| games | batches of 400 up to a hard cap of 2,400 games; stop as soon as unique records ≥ 12,000 | batch rule mirrors Amendment 2's fail-closed enlargement; it may look at *record counts only*, never at label values |
| play budget | 20,000 nodes/move, diversification at plies (8,10,12), 3 candidates, ±25 cp, 2,000 nodes | identical to S5 |
| truncation | `max_plies=70`, `sample_every=2`, `min_ply=6` | identical to S5 |
| recorded | positions.jsonl + games.jsonl + pool-manifest with real engine identity, `openingSourceSha256` | raw source keeps game multiplicity; **no dedup here** |
| expected yield | ≈13.4 unique records/game at 70 games [M from S5] falling with pool size; 12,000 unique needs ≈1,000–1,400 games [E] | |

Then `snapshot_source` → **S####** (rows carry explicit `game`), `normalize` → **N####**
(EPD identity, transposition components via `group`, duplicates counted and reported).

**Leakage control on the training pool** (before any arm is built):

1. Compute the joint transposition components of the new pool ∪ the instrument's 575
   clean records, using the same union rule S6 used (shared EPD record_id → union of
   `group`s; `tools/s6/s6_round3.py` lines 119–155).
2. If **zero** of D0010's 200 records touch a new-pool component → `E0004`/`D0010` stays
   the instrument, byte-identical.
3. Otherwise: **remove the colliding new-pool records from the candidate universe**
   (deterministic, preregistered, and it never inspects a label value). If the removed
   fraction exceeds 5 % of the pool, **STOP** and build a fresh instrument (new held-out
   openings, new labels, new `D`/`E`) *before* any training.
4. The candidate universe is then frozen and hashed; every later selection is over it.

Reuse is expected to succeed: the instrument's 575 clean records were proven disjoint
from **all 12 components** of the S5 pool, and it was built from a disjoint opening
partition. The check is still mandatory because the new pool is new.

### 5.3 Calibration and the arm-sizing rule

1. Draw **64** calibration records by deterministic hash order from the candidate
   universe (seed `20260916`).
2. Buy one label per budget {16k, 64k, 160k, 400k} for each calibration record through
   the pinned analyze identity. **Read only `nodes` and wall time. Never read
   `scoreCpStm`.**
3. `mean_b` = mean realized nodes per label at budget *b* over the 64 records.
4. `n_b = round(T / mean_b)`; the four sizes are materialized into the preregistration
   file, then **frozen**.
5. Calibration labels are ordinary labels of their budget; if a calibration record later
   falls in an arm, its label is reused (identical semantics; re-buying is refused by the
   duplicate guard anyway).
6. Preregistered sanity band: every `mean_b` must lie in **[0.90·b, b]**. Outside →
   **STOP** (engine drift or an unhonoured node budget).

### 5.4 Arms — exact definitions

Selection: one deterministic order over the candidate universe,
`order = sort(universe, key=sha256(f"{20260916}:{record_id}"))`. The arms are disjoint
prefix blocks of that order:

| arm | block | budget | family | authority | value_path | pov | type | K | λ |
|---|---|---|---|---|---|---|---|---|---|
| **A1** | `order[0:94]` | 400,000 | `search_deep_cp` | `legacy.cvs.search.deep` | `targets.scoreCpStm` | stm | cp | 256 | 1.0 |
| **A2** | `order[94:329]` | 160,000 | `search_shallow_cp` | `legacy.cvs.search.shallow` | `scoreCpStm` | stm | cp | 256 | 1.0 |
| **A3** | `order[329:917]` | 64,000 | `search_shallow_cp` | `legacy.cvs.search.shallow` | `scoreCpStm` | stm | cp | 256 | 1.0 |
| **A4** | `order[917:3258]` | 16,000 | `search_shallow_cp` | `legacy.cvs.search.shallow` | `scoreCpStm` | stm | cp | 256 | 1.0 |

(Block boundaries are shown with the estimated sizes; the materialized sizes are the
authoritative ones.) All four specs share producer = the pinned analyze sha256
`36e9b159…`, `target_type=cp`, `K=256`, `LAMBDA=1`.

Budget, not family name, is the identity: a 160k label is not interchangeable with a 16k
label, and the spec hash proves which observation a run consumed. The 400k family keeps
the S6/tier3 value shape because it is the shape the instrument's targets already use;
the 160k/64k/16k family keeps the legacy tier1 value shape so that the pilot's existing
16k labels and the frontier's new 16k labels are *the same kind of observation*.

Label buying: budgets {16k, 64k, 160k} at `pv_plies=8`; budget 400k at `pv_plies=12`
(the tier1/tier3 conventions). Per arm, buy exactly its rows at its budget; register one
label set per arm under the new normalization; record measured nodes and wall ms per
label. Then verify `|realized_nodes(arm) − T| / T ≤ 0.02` **for every arm** (§5.7).

### 5.5 Identities to be created

| identity | content |
|---|---|
| **S####** | new raw pool snapshot, 400-game batches, truthful engine identity, game ids preserved |
| **N####** | canonical normalization (EPD identity, components, duplicates reported) |
| facts label set | `cvslab.facts.label_facts` on every record — **not consumed by S7 training**; this is the first deposit of the "geometry everywhere" layer |
| 4 × search label sets | one per arm, as in §5.4 |
| **D_A1..D_A4** | `freeze_dataset(N, record_ids=<arm block>, required_labels=[family], target_spec=<spec_b>, fractions=(1,0,0), campaign={...})` — all-train layout, S6 Amendment 1 convention |
| **T_A1..T_A4** | four recipes: identical params (`EPOCHS 40, BATCH 256, LR 0.003, OPTIMIZER adam, INIT_STD 0.05, K 256, LAMBDA 1.0`), each pinning its own `TARGET_SPEC` |
| **E0004** | reused instrument (D0010, split test, 200 records, 400k spec) — reused iff §5.2 proves disjointness |
| **A####** | 16 ablations: per arm, one baseline (H=16) + three H∈{1,4,32} ablations, each recording the supervision-divergence declaration (§5.8) |
| **R####** | 16 ablations × 5 seeds = **80 runs**; every run pins `eval_dataset_id = D0010`, `eval_protocol_id = E0004`, its own dataset/recipe |
| **H####** | hypothesis: "at fixed teacher node budget, supervision density (more rows × shallower labels) beats a few deep labels on the common instrument" |
| **F####** | finding, sealed, with the decision and all per-cell numbers |

### 5.6 Statistical analysis (preregistered)

* **Primary endpoint:** `test_loss` on E0004/D0010 (200 records, 400k targets).
* **Unit:** paired seed difference; 5 seeds {0,1,2,3,4}; 95 % CI via Student t, df=4,
  t=2.776 — the same method and the same helper (`paired_effect`) as S6.
* **Primary contrast:** `d_H = mean_seed [ test_loss(A4,H,seed) − test_loss(A1,H,seed) ]`
  per width H ∈ {1,4,16,32}. Negative = wide-shallow better.
* **Decision rule (intersection–union over widths, as S6):**
  * `BREADTH_WINS` iff every width's CI upper bound < 0;
  * `DEPTH_WINS` iff every width's CI lower bound > 0;
  * otherwise `INCONCLUSIVE`.
  A single-width win is *reported but not decisive* — exactly the treatment S6's H16
  result received.
* **Secondary, no decision authority:** adjacent contrasts (A2−A1, A3−A2, A4−A3);
  intermediate arms vs A1; `cp_mae` and `sign_agreement`; monotonicity of the four arm
  means per width; width × density interaction (paired seed-level contrasts, as
  `interaction_effects` did for S6).
* **Required reporting:** all 16 cell means, all per-seed differences, every CI, the
  realized node totals per arm, and the full identity table. No metric substitution
  after the fact; no seed dropped; a failed cell is `INVALID`, never omitted.
* **Scope statement to be sealed with the finding:** "this pool, this teacher, these
  four budgets, RAW-768."

### 5.7 Fail-closed conditions

1. **Engine identity drift** — any label bought from an analyze whose sha256 ≠
   `36e9b159…`, or whose net hashes differ from the pinned set → STOP.
2. **Determinism gate** — re-label 8 stored positions (4 at 16k, 4 at 400k) whose labels
   already exist; the replay must reproduce the stored `scoreCpStm` exactly. Mismatch →
   STOP (this is the S1-parity discipline applied to a new session).
3. **Pool adequacy** — unique records < 4 × n_A4 (≈9,364), or the 2,400-game cap reached
   without reaching it → STOP.
4. **Instrument collisions** — > 5 % of pool records collide with the instrument's 200
   records → STOP (build a fresh instrument).
5. **Calibration band** — any `mean_b` outside [0.90·b, b] → STOP.
6. **Arm parity** — any arm's realized nodes outside ±2 % of T → those cells INVALID
   (report; never silently resize).
7. **Disjointness** — pairwise arm intersections must be exactly 0 (guaranteed by block
   construction; asserted anyway).
8. **Strict cardinality** — any dataset row with zero or more than one matching label
   under its spec → freeze/training raises (existing behaviour, deliberately retained).
9. **Instrument identity across runs** — all 80 runs must share one `protocol_hash` and
   one `eval_dataset_manifest_hash`; any deviation → those cells INVALID.
10. **No tuning** — after the first arm label is bought, no change to budgets, sizes,
    selection order, widths, seeds, epochs, LR, batch or initialization. Any change is a
    new experiment with a new preregistration.
11. **No peeking** — target values are not inspected before arm construction, and arm
    datasets are not trained before the instrument identity is frozen.
12. **Declared divergence only** — train/eval spec divergence is allowed *only* through
    the explicit, hashed declaration of §5.8; silent divergence remains a hard failure.

### 5.8 Required code change (prerequisite P1)

`cvslab/service.py` currently hard-fails when the training TargetSpec and the evaluation
TargetSpec differ (`target_spec_consistent` → `_PreflightFailed("training and evaluation
TargetSpecs differ")`, lines 519–528). That check was written for S6, where both were
400k; **the density frontier cannot run under it**, because arms are trained on their own
budget and measured on the common 400k instrument.

Required behaviour, minimal and fail-closed by default:

* divergence stays a **hard failure** unless the ablation declares it explicitly at
  creation time (e.g. `create_ablation(..., allow_supervision_divergence="<reason>")`),
  with the reason string stored in the A####/R#### record and included in the ablation
  identity hash, so it cannot be added after the fact;
* when declared, record both spec hashes in the run's integrity checks (`warn=True`,
  visible in every run record) and keep the strict per-row cardinality checks unchanged —
  those are what actually caught the S6 `eval_cp` failure;
* add, at the campaign level, the hard invariant that a declared-divergence campaign's
  runs all pin **one** protocol hash (condition 9 above).

This is the only code change S7 requires. Everything else reuses shipped machinery.

### 5.9 Cost

| item | teacher nodes | engine time | wall clock |
|---|---:|---:|---:|
| pool generation (≈1,200 games × 70 plies × 20k nodes) | ≈1.68 G **[E]** | ≈25–30 min **[E]** | 30–60 min **[E]** |
| tier0/facts on ≈12,000 records (lab-native) | 0 | — | ≈1–3 min **[E]** |
| calibration (64 × 640k nodes) | 41 M **[E]** | ≈40 s **[E]** | <1 min |
| arm labels (4 × T) | 149.1 M | ≈2.5–3 min **[E]** | <5 min |
| determinism gate (8 positions) | 1.7 M | ≈2 s | — |
| training (80 runs, ≤3,258 rows, 40 epochs) | 0 | 0 | ≈10–30 min **[E]** |
| evaluation (80 runs × 200 records) | 0 | 0 | <1 min |
| **total** | **≈1.87 G** | **≈30–35 min** | **≈1–2 h** |

The instrument cost is zero if reuse succeeds; if a fresh instrument is required, add
≈80 M nodes (≈69 s **[M]** for 200 labels) plus its own pool.

For scale: the *entire* S7 teacher bill is smaller than **three minutes** of the
Gen10 `corpus-d20` labelling campaign's 0.54 s/position rate × 4 threads — i.e. Gen10
spent ~90 single-thread hours on a corpus that took ~14 labels per unique position,
while S7 buys a 4-point density curve over 3,258 unique positions for ~30 minutes of
engine time, of which the labelling itself is under 3 minutes.

---

## 6. What S7 does not establish

* Not a statement about GEO or HYBRID inputs — RAW-768 only, deliberately.
* Not a statement about playing strength — `test_loss`/`cp_mae`/`sign_agreement` on a
  frozen static instrument; no search, no games, no gate.
* Not a statement about priority-v1 or any selection policy — selection is uniform.
* Not a statement about Stockfish/oracle labels — the teacher is the CVS analyze binary
  at four budgets.
* Not a statement about other pools or other teachers; replication stays issue #16.
* The wider arms receive proportionally more optimizer steps at fixed epochs. A
  breadth win therefore does not cleanly separate "more data" from "more updates"; a
  breadth *loss* is the stronger inference because it happened despite both. This
  asymmetry is declared, not hidden — and the fixed-epochs contract is the one S6
  already used (T0004).

---

## 7. Where this fits in the program

```
  S6  selection: priority-top vs uniform, equal depth     → INCONCLUSIVE (done)
  ├─ S7-P  screen: depth-at-fixed-rows, rows-at-fixed-depth, economics (existing labels)
  ├─ S7-F  density frontier: 400k×94 vs 160k×235 vs 64k×588 vs 16k×2341 at T  ← this doc
  ├─ S8    economically efficient labelling regime: pick the depth the frontier's
  │        curve says to buy; run the funnel at that budget on a NEW pool; verify the
  │        curve repeats with fresh games (this is where "efficient regime" becomes a
  │        standing recommendation rather than one pool's result)
  ├─ S9    GEO white-box: train on the deterministic facts layer that S7 already
  │        attaches to every record (no search teacher at all); the facts are free at
  │        the margin, so the question is what they are *worth* as input
  ├─ S10   matched RAW vs GEO vs HYBRID at comparable parameter and supervision budgets
  └─ S11   selection revisited, as *mixture*: many representative rows + a small
           prioritized fraction. This is also where the "wrong objective" question is
           answered by design, not by argument:
              arms = uniform | priority-top | priority-BOTTOM | fresh random
              all at equal rows and equal per-row budget on the same instrument.
              If priority-bottom ≈ uniform ≈ priority-top → the ranking carries no
              training-value signal at this scale (it is a diagnostic, not a sampler).
              If priority-bottom < uniform → right direction, small effect.
              If priority-bottom > uniform → priority-v1 selects against the training
              distribution (the "wrong objective" hypothesis, made falsifiable).
           priority-top and uniform already exist as S6 arms, so the test adds only two
           arms (≈188 new 400k labels ≈ 74 M nodes ≈ 60 s of engine time).
```

S7-F's outcome determines S8's budget, and S9/S10 inherit the facts layer that S7-F
deposits. Nothing in S9–S11 may start before S7-F's finding is sealed, because each of
them uses S7-F's answer as an input to its design.

---

## 8. The economics, directly

| | Gen10 `corpus-d20` **[M]**/**[D]** | S7-F frontier **[M]**/**[E]** |
|---|---|---|
| teacher | Stockfish 17.1, `go depth 20` | CVS analyze, fixed node budgets |
| labels bought | 599,363 | 3,258 |
| unique positions covered | 43,536 | 3,258 |
| labels per unique position | 13.8 | 1.0 |
| teacher time per unique position | 7.44 s | 19 ms (wide arm) / 325 ms (narrow arm) |
| teacher time for the whole campaign | ≈90 single-thread hours | ≈30–35 min wall, of which labelling <3 min |
| semantics retained | cp, res, cp_play, label_depth/nodes, sf_best, raw feature ids | cp at 4 distinct, hashed budgets + facts/motif/strategy for every record |
| provenance | 22/23 models lack training commit + dataset hash; one corpus was 43 % duplicate rows | every label carries family/authority/producer/budget/registry; every dataset and protocol is hash-pinned |
| selection policy identity | none recorded | none needed (uniform) — but the *pool*, *order seed* and *arm blocks* are all frozen |

**Answer to the thesis question, stated as the thing S7-F can actually falsify:** the
winning economics for a *static evaluator on this instrument* are
"broad representative rows at the cheapest depth that still carries signal" if and only
if `BREADTH_WINS`. If `DEPTH_WINS`, the economics stay "few deep labels", the funnel's
value collapses to its facts layer and its oracle sample, and the platform's effort
should move to S9 (what the free geometry is worth) rather than to wider shallow
corpora. If `INCONCLUSIVE`, the honest conclusion is that between these four regimes at
this scale the teacher budget is not the binding constraint — which would itself be a
major result, because it would say the field's default of "label everything deeply" is
paying for something invisible to this instrument.

---

## 9. Facts vs estimates

**Documented/measured (reusable as ground truth):** S4 run R0008 tier timings and node
totals; label counts and means per budget; S6 arm parity, per-width effects and
decision; D0010/E0004/T0004 identities and the S6 spec hash; Gen10 corpus row/unique
counts, SF 17.1 `go depth 20`, 0.54 s/position, architecture and target formulas,
missing provenance; the 43 % duplicate-row defect in `corpus-static-full`; engine
funnel README tier timings.

**Estimates (to be replaced by measurement during execution):** the new pool's yield per
game and therefore its game count; the 160k/64k realized-node means (calibration exists
precisely to replace them); arm sizes before calibration; all wall-clock figures;
the new pool's component count and its collision rate with the instrument.

**Unknown and left open:** whether CVS 16k-node targets carry enough signal to train a
useful evaluator at all — that is the experiment's question, not an assumption.
