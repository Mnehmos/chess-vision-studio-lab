# Training-Data Generation: Research Plan and Architecture Audit

Status: audit + design (no code moved). Companion to `docs/DATA_LIFECYCLE.md` and
`docs/RESEARCH_PROTOCOL.md`. This document decides **how the proven information-gain funnel
becomes a first-class CVS Lab research instrument** without losing it along the way.

> **The generator produces evidence. The lab decides what becomes a dataset.**
> A funnel output directory must never implicitly become "the training set."

## 1. The question this architecture must answer

Does selective information-gain labeling produce a stronger evaluator than uniform deep labeling
**at equal compute**? The first measured funnel run (legacy engine, 20k positions) was
INCONCLUSIVE: the priority arm concentrated shallow→deep score change (1.34×, partly circular),
and the non-circular Stockfish-disagreement signal pointed the right way (55.6% vs 50.4%) but at
~1.4σ. The missing step is a training experiment, not a better prioritizer. **Do not optimize
priority-v1 before that baseline exists.**

## 2. Audit — the legacy funnel as built

`chess-vision-studio-rust-engine/training/funnel/` (1,240-line `labeling_funnel.py`, 160-line
README, 226-line stdlib-only test suite, `funnel-config.v1.json`).

Pipeline: `positions → tier0 facts (all) → tier1 shallow CVS at multiple node budgets (all) →
tier2 priority-v1 triage → {deep (top-k), uniform (k), audit (low-priority sample), holdout
(hash-fixed)} → tier3 deep CVS → tier4 sparse Stockfish → manifest/report/coverage`.

What is genuinely proven here and must survive:

- **Deterministic sampling** by hash rank (independent of shard order), position identity
  = FEN without move counters.
- **priority-v1**: weighted sum of 9 auditable components (score instability, best-move change,
  trajectory instability, PV disagreement, tactical density, rarity vs accumulated coverage,
  outcome disagreement, selectivity edge, forcedness) with per-component `{value, weight,
  contribution}` recorded per position.
- **Holdout assignment is a pure function of (holdoutSeed, position id)** — stable across pool
  sizes and prioritizer versions; never train-eligible; the untouched oracle set.
- **Equal-compute arms**: deep vs uniform at identical row counts and node budgets; audit samples
  the low-priority remainder; `trainEligible` flags gate training use.
- **Leakage guard**: `assert_static_input_safe` — a Tier-0 record may carry only
  `deterministic_geometry` + `bounded_tactical_proof`; search/outcome/oracle stay in other files.
- **Provenance**: manifest pins config sha256, tooling git state, analyze binary sha256, net
  hashes, every search option, Stockfish sha256, taxonomy version+sha256, prioritizer version.
- **Resumability**: every stage skips ids already present in its output file.
- **Cost accounting**: per-tier engine-seconds and wall-ms; per-arm deep-node parity is reported
  (790.6M vs 795.4M in the measured run).
- **13 stdlib-only tests** covering priority decomposition, selection determinism/disjointness,
  holdout stability, rarity direction, leak guard — these become the parity oracle.

### 2.1 Engine dependency surface (the decisive audit finding)

The entire coupling to the Rust engine is:

| dependency | shape |
|---|---|
| `Serve` class | persistent `analyze --serve` subprocess, one JSON request → one JSON line |
| request `{"cmd":"facts", fenBefore, playedMoveUci, options}` | tier0 bundle |
| request `{"cmd":"go", fen, nodeBudget}` | tier1/tier3 search response |
| `TAXONOMY` file | `benchmarks/data/motif-taxonomy.json` (versioned, hashed into the manifest) |
| Stockfish | plain UCI subprocess (not engine-specific) |
| `board_geometry` | in-process python-chess (not engine-specific) |

Everything else — sampling, dedup, prioritization, selection, coverage, reporting, manifests —
is **engine-agnostic Python**. The provider seam the refactor needs already exists de facto at
`Serve` + `summarize_facts` + `compact_search`. This is the key evidence for the decision in §4.

### 2.2 Output formats (files a lab import must map)

`positions.jsonl` (id, fen, source file/line, gameOutcome) · `tier0.jsonl` (deterministic_geometry,
bounded_tactical_proof, taxonomy slugs/families, uncomputed, factsRegistryVersion, cost) ·
`tier1.jsonl` (per-budget score/mate/bestmove/PV/trajectory, derived scoreDelta/moveChange/PV
prefix, cost) · `triage.jsonl` (priority, rank, components, reasons, selection flags,
trainEligible) · `tier3.jsonl` (deep label, shallowToDeep delta, targets scoreCpStm/White,
expectedScoreStm) · `tier4.jsonl` (SF score/mate/bestmove/PV/depth/nodes) ·
`coverage.json` (taxonomy slug counts) · `report.json|md` (tier costs, arm stats, coverage,
audit-miss estimate, informativeYieldRatio) · `manifest.json` (all identities).

## 3. Audit — what CVS Lab already owns

| lab capability | where | status |
|---|---|---|
| `S####` immutable sources (hashes, importer version, counts) | `cvslab/data.py` | done |
| `N####` versioned standardization, EPD record identity, dedup | `data.py` | done |
| L2 append-only labels, provenance per label (family/authority/producer/registry_version/budget) | `data.py register_labels` | done |
| Label families include `priority`, `outcome`, `oracle_cp`, `search_shallow_cp`, `search_deep_cp`, `facts`, `motif`, `strategy` | `data.py LABEL_FAMILIES` | done — the funnel's provenance classes already have homes |
| Deterministic CVS facts (21-family geometry registry v1, ChessTempo motifs, strategy tags) | `cvslab/facts.py` | done (Python mirror) |
| Faceted catalog (motif/strategy/fact filters, facets, sort) | `data.py catalog` + `/api/catalog` | done |
| Stacking with explicit policies, available-vs-effective preview, immutable freeze | `data.py freeze_dataset`, `StackArm` | done |
| Leakage-safe splits by source game (`group` = source:game) | `data.py _split_for` | done |
| Funnel run intake as immutable evidence (R#### `FunnelRun`: tiers, arms, oracle arms, priority histogram/reasons, coverage, compute, file hashes) | `cvslab/intake.py import_funnel_run` | done |
| Compute accounting, findings, promotion gates, GUI (overview/data/map/matrix/scaling/findings) | `service.py`, `web/` | done |

Gaps relative to a lab-native funnel: no orchestration (tier stages, workers, resume), no
prioritizer implementation, no candidate-pool generation, no per-position funnel-row import,
no equal-compute arm builder that freezes D#### from one pool, and no triage-policy identity.

## 4. Copy vs refactor vs adapter — evaluated against the code

**A. Copy `labeling_funnel.py` into the lab.** Would work (it is portable), but duplicates the
taxonomy/facts semantics that `cvslab/facts.py` now mirrors, freezes the prioritizer inside a
script, and forks the engine integration in two places. Rejected: the code shows no need — the
orchestration is already separable, so copying buys nothing a refactor does not.

**B. Refactor into lab-owned orchestration + provider adapters.** The audit supports this: ~85%
of the funnel is engine-agnostic, and its engine contact is one 2-command JSON protocol. The lab
gains versioned TriagePolicy/T#### identities, coverage-aware priors from real lab coverage,
D####-only training, and can swap teachers (legacy CVS today, lab engine later, Stockfish
alongside) without touching science. Cost: cross-repo packaging discipline.

**C. Keep it in the legacy repo + lab adapter (`import_funnel_run`).** Already exists and should
be **kept working as the legacy evidence + parity reference** — but not as the future home: the
prioritizer and policy would stay unversioned-by-lab, teacher swaps become new forks, and the
"generator vs dataset" boundary stays implicit (a run dir looks like a dataset). Rejected as the
target, retained as the baseline.

**Decision: B, staged, with C preserved as evidence.** Move orchestration + triage + manifests
into the lab behind a provider contract; keep the legacy funnel untouched, import its runs as
`FunnelRun` evidence, and gate retirement on fixture parity (§7).

## 5. Target architecture

```
candidate sources            analysis producers                 lab
─────────────────            ──────────────────                 ───
legacy corpora (S####)  ┐    CVS facts (legacy analyze)     ┐
fresh legacy self-play  ├──► legacy CVS shallow (analyze go)├──► funnel run (lab-owned)
human/lichess PGN (S####)│   legacy CVS deep (analyze go)   │    ↓ evidence files + manifest
search-tree sampling    ┘    Stockfish (UCI)                 │    import per-position labels
                             lab facts (cvslab.facts)        ┘    into canonical records (N#### + L2)
                                                                  ↓ catalog / coverage / rarity priors
                                                                  ↓ stack + freeze → D#### (UNIFORM/PRIORITY/MIXED)
                                                                  ↓ train (tiny raw NNUE) → E#### suite → finding
```

Responsibilities, kept separate and each owned by exactly one layer:

| responsibility | owner | artifact |
|---|---|---|
| Position generation | provider(s) | new `S####` (self-play PGN/JSONL) or existing sources |
| Analysis / labeling | providers (teacher-specific) | append-only label sets under an `N####` |
| Triage / allocation | lab `TriagePolicy` (versioned) | triage rows + selection flags; policy identity |
| Canonical corpus management | lab (`data.py`) | `N####` + accumulated labels + coverage |
| Dataset definition | lab (`freeze_dataset`/stacking) | immutable `D####` only |
| Training | lab runs | `R####` pinned to one `D####` manifest |

## 6. Provider contract (proposed)

Python protocols in a new `cvslab/funnel/` package. Providers are thin; all science lives behind
them. Everything a provider returns is evidence with provenance, never a dataset.

```python
class PositionSource(Protocol):            # candidate pool generation
    id: str                                # e.g. "legacy-selfplay-v1"
    def positions(self, *, count: int, seed: int, pool: PoolSpec) -> Iterator[Position]:
        # Position: {fen, source_ref: (S####, row), game_id, opening_id, ply}
        # Deterministic; dedup by EPD is the lab's job (canonical record identity), not the source's.

class DeterministicLabelProducer(Protocol):
    provenance_class = "deterministic_geometry" | "bounded_tactical_proof"
    def label(self, position: Position, *, options: dict) -> LabelBatch
        # Tier-0 equivalent: facts/motif/strategy rows; must pass a static-input leak guard
        # equivalent to assert_static_input_safe before rows may feed static model inputs.

class SearchLabelProducer(Protocol):
    provenance_class = "search_derived"
    family: str                            # search_shallow_cp | search_deep_cp
    def search(self, position: Position, *, node_budget: int, pv_plies: int) -> SearchLabel
        # score/mate/bestmove/PV/trajectory + measured nodes/wall-ms; cold isolation = default

class OracleProducer(Protocol):
    provenance_class = "external_oracle"
    def evaluate(self, position: Position, *, depth: int, movetime_ms: int) -> OracleLabel

class TriagePolicy(Protocol):              # lab-owned, versioned
    version: str                           # "priority-v1" initially
    def components(self, tier0, tier1, outcome, coverage) -> dict[str, float]
    def select(self, ranked, *, cfg, seed) -> Selection   # deep/uniform/audit/holdout flags
```

Built-in providers: `AnalyzeServeProvider` (wraps the legacy `analyze --serve` protocol —
`facts` and `go` — with pinned binary hash), `StockfishProvider` (UCI), `LabFactsProvider`
(`cvslab/facts.py`), and later `LabEngineProvider` (clean-room engine, same `go` shape).
Triage policies are stored as immutable lab objects: **reuse `T####`** with
`trainer="cvslab.triage.priority"`, `trainer_version=1`, `params={weights, caps, versions}` —
no new prefix, same hashing/immutability rules.

## 7. Mapping funnel evidence → canonical lab objects

| funnel field/file | lab destination |
|---|---|
| `positions.jsonl` rows | canonical records: new `S####` snapshot of the pool + `N####` normalization (EPD identity dedup); `game_id` → record `group` (leakage-safe splits); `gameOutcome` → `outcome` label |
| `tier0.deterministic_geometry`, `bounded_tactical_proof`, `taxonomy`, `uncomputed` | `facts` / `motif` / `strategy` label sets (authority `legacy.cvs.analyze.<sha8>`, registry_version = `factsRegistryVersion`); `uncomputed` preserved per record |
| `tier1` per-budget search + derived | `search_shallow_cp` labels, one per node budget; `budget={"nodeBudget": n}`; derived deltas preserved in label `note` |
| `triage` priority/components/selection/trainEligible | `priority` labels (value = priority, `note` = top reasons, `budget` = component contributions) + selection flags recorded in the triage manifest |
| `tier3` deep + shallowToDeep + targets | `search_deep_cp` labels (scoreCpStm/White, expectedScoreStm) with `budget={"nodeBudget":400000}` |
| `tier4` Stockfish | `oracle_cp` labels (authority `oracle.stockfish.<ver>`, `budget={depth, movetimeMs, nodes}`) |
| `manifest.json`, `report.json`, `coverage.json`, `triage-misses.jsonl` | already imported losslessly as `FunnelRun` (R####) by `intake.import_funnel_run`; add `position_pool` link to the new `S####`/`N####` |

**Missing lab schema fields (minimal, additive):** (1) `FunnelRun.position_pool: Optional[str]` —
the `S####`/`N####` the per-position rows were imported into; (2) `FunnelRun.triage_policy_id:
Optional[str]` — the `T####` identity of the prioritizer used (today only the version string is
funnel-internal); (3) optional `LabelSetRef.policy_id` mirroring (2) on triage-produced label
sets. Nothing else is required: label rows already carry family/authority/producer/registry_
version/budget/note, and `FunnelRun` already preserves tier costs, arms, oracle arms, coverage,
priority histogram/reasons, audit-miss, compute, and file hashes.

## 8. Parity strategy (retire nothing until parity is proven)

1. **Pin the reference**: keep `runs/gen10-d20-20k`-style manifests and the measured-run tables
   in git history; add the funnel's `manifest.json` hash as a pinned constant in the parity test.
2. **Fixture replay**: small frozen fixture (~2k positions from old shards, committed as a test
   `S####`). Run the *legacy* funnel and the *lab* funnel over it with the same config, the same
   `analyze` binary sha, same seed.
3. **Compare, per position**: position-id set, priority value + per-component contributions
   (exact float equality expected — same formula, same inputs), deep/uniform/audit/holdout
   membership, `trainEligible`, arm sizes, deep-node parity, coverage counts, informative rates.
4. **Explain every difference**: differences are allowed only if listed in the test as
   intentional (e.g. lab-side label POV normalization documented once); anything else fails.
5. **Promote**: only after fixture parity passes, the lab funnel becomes the default producer;
   the legacy script stays frozen as the parity oracle and legacy-evidence source.

## 9. First scientific campaign (the missing causal test)

One pool, one funnel run, three frozen datasets, identical downstream everything:

1. **Pool**: ~100K–1M candidates from fresh legacy-CVS self-play with opening diversification
   (root-move diversity among near-equal moves), sampling throughout games, EPD dedup; retain
   `game_id`/`opening_id` for leakage-safe splits. Size follows measured Tier-0/1 throughput.
2. **Cheap pass on all**: tier0 facts (lab facts and/or legacy analyze) + tier1 shallow at
   `[2k, 16k]` nodes.
3. **Triage**: priority-v1 (unchanged). Emit deep/uniform/audit/holdout selections at identical
   row counts (e.g. 10% deep, 10% uniform, 2% audit, 1% hash-holdout).
4. **Deep label**: tier3 at identical node budget per row across arms — assert equal total deep
   nodes in the run manifest before training.
5. **Import**: per-position rows → canonical records + L2 labels (§7); freeze three `D####`
   datasets with the same normalization, splits and dedup:
   `UNIFORM` (uniform arm), `PRIORITY` (deep arm), `MIXED` (natural blend), later
   `SEMANTIC_BALANCED` (stack with balance arms — machinery already exists).
6. **Train**: the same tiny raw NNUE (H=16, one T#### recipe, seeds {0,1,2}) on each `D####`.
7. **Evaluate + finding**: frozen tiny suite `E####`; decision rule pre-declared (primary:
   `test_loss`, secondary: cp_mae, sign agreement). Result counts as evidence whether
   SUPPORTED/REJECTED/INCONCLUSIVE — negative results stay visible.
8. **Only then** touch prioritizer weights.

## 10. Staged implementation plan (each slice independently shippable)

Tracked as issues #8–#13. Each slice ships independently; S1's fixture parity gates everything
after it.

| stage | deliverable | acceptance |
|---|---|---|
| S1 | `cvslab/funnel/` package skeleton + `AnalyzeServeProvider` + leak-guard port + fixture replay harness | lab funnel reproduces legacy tier0/tier1 rows byte-for-byte-equivalent on the fixture (exact field compare) |
| S2 | Per-position import: funnel run rows → `S####`/`N####` + label sets (§7) + `FunnelRun.position_pool` | importing the measured legacy run yields records whose facts/motifs match tier0; catalog facets reproduce coverage.json counts |
| S3 | Triage as lab object: `T####` policy identity + `select()` port + parity test vs `triage.jsonl` | per-position priority/flags identical on fixture; policy hash pinned |
| S4 | Orchestrator: stages, workers, resume, cost accounting, manifest write (lab-shaped) | resume test: kill mid-tier1, rerun, no duplicates; manifest pins all identities |
| S5 | Candidate pool generation: self-play provider + opening diversification + sampling | deterministic pool from seed; EPD dedup reported; game/opening ids retained |
| S6 | Arms → `D####`: UNIFORM/PRIORITY/MIXED builder with equal-compute assertions | freeze refuses arms whose deep-node totals differ beyond tolerance; manifests expose arm vs effective rows |
| S7 | Campaign runner + GUI visibility (tier positions, compute, reasons, coverage, disagreement, audit misses, source mix) | first campaign runs end-to-end and renders in the Data/Lab Map screens |

## 11. Risks and guardrails

- **Parity drift**: taxonomy or tie-breaking drift silently changes the instrument → fixture
  replay in CI (S1) is non-negotiable before any retirement.
- **Prioritizer over-tuning**: do not change weights or components until §9's causal test
  reports; version every change as a new `T####`.
- **Leakage**: tier0 label sets must keep the static-input guard; search/outcome/oracle labels
  and `bucket`/`sort` filters must never feed static model inputs unless an ablation explicitly
  requests it (declare in the ablation, not in the data).
- **Implicit datasets**: lab funnel output is evidence; only `freeze_dataset` produces training
  inputs. The orchestrator must refuse to consume a run dir directly.
- **Compute honesty**: equal-compute claims are asserted from measured nodes/engine-seconds in
  the run record, never assumed from configuration.
- **Legacy preservation**: the old funnel and its runs are not deleted, rewritten, or "cleaned";
  they are the parity oracle and the historical evidence trail.
