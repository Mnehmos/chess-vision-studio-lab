# Chess Vision Studio Lab

A first-principles research lab for understanding where chess-engine capability comes from.

The lab studies four separable axes:

1. **Learned capacity** — parameter count and architecture.
2. **Representation** — raw piece-square inputs, CVS geometry, motifs, strategies, and semantic modules.
3. **Executed compute** — search nodes, depth, time, and search policy.
4. **Training signal** — deterministic facts, shallow/deep CVS search, game outcome, semantic labels, and sparse external-oracle labels.

The core rule is simple:

> Earn every parameter. Earn every feature. Earn every label. Earn every unit of search compute.

We start deliberately tiny, establish controls, then scale only ideas that survive controlled ablation.

## Quickstart (Phase 0)

```bash
pip install -e .[dev]                 # python >= 3.11, numpy, pydantic, fastapi, python-chess
python -m pytest                      # 49 tests: schemas, immutability, fail-closed INVALID, API, CLI

cvslab init --engine-repo /path/to/chess-vision-studio-rust-engine   # pin the engine (optional)
cvslab demo                           # one tiny experiment through every layer:
                                      # S -> N -> D -> T -> E -> H -> A -> runs -> sealed finding card

cvslab serve                          # local API + built GUI at http://127.0.0.1:8000
```

The GUI ships the six Phase 0 screens — lab overview, ablation builder (exact diff and parameter
count before launch), run queue/history, switch matrix, scaling explorer, and finding cards — plus
the data workbench (corpus catalog with faceted filtering, dataset stack builder with
available-vs-effective preview, and the lineage/migration view). It contains no scientific logic:
it edits canonical specs and renders immutable evidence served by the API. Rebuild it with
`cd web && npm install && npm run gen:types && npm run build`.

The data lifecycle (issue #2) is end-to-end on the CLI as well: `cvslab source-fixture` /
`source-import` / `source-pgn` → `cvslab normalize` → `cvslab labels-append` (append-only L2 label
sets with per-authority provenance) → `cvslab catalog` (faceted corpus filtering) →
`cvslab stack-preview` + `cvslab dataset --arm '{"name":"mid","filter":{"phase":"middlegame"},"policy":"fraction","fraction":0.5}'`
(explicit stacking, frozen into D####) → `cvslab migrate` + `cvslab migration-diff` +
`cvslab dataset-rebuild` (re-standardization into a new N#### with a full diff, and descendant
datasets — history is never rewritten).

Everything lives in `./labstore` (set `CVSLAB_HOME` to move it): one JSON file per canonical
object, artifacts per run, sealed with record hashes once terminal. `cvslab --help` lists every
command; `cvslab schema` exports the JSON Schema shared by the CLI, API and generated TypeScript
types (`schemas/cvslab.schema.json` → `web/src/generated/schemas.ts`).

## Research model

CVS Lab treats the production engine as a pinned subject under test. The engine repository is not silently modified during an experiment. Every experiment records the exact engine commit, model identities, dataset manifests, search budget, seeds, hardware, and evaluation protocol.

The lab uses permanent identities:

- `H####` — hypothesis
- `A####` — ablation
- `M####` — model artifact
- `S####` — immutable source snapshot
- `N####` — normalization / standardization recipe
- `D####` — frozen dataset manifest/view
- `T####` — training recipe
- `E####` — evaluation protocol
- `R####` — execution/run

Ablation display names use the compact form:

`A#### FAMILY-GEN-PARAMS @ BASELINE : OVERRIDES`

Examples:

- `A0042 NNUE-G01-P12K @ RAW : H=16`
- `A0107 NNUE-G03-P50K @ HYBRID : KING_SAFETY=0`
- `A0217 LLM-G01-P4.8M @ SEED : TYPED=1,LOOP=4`

The short label is for humans. Every run also stores the complete immutable configuration and hashes.

## Data model

Data is treated as a versioned lineage rather than a mutable training folder:

`S#### raw source -> N#### normalization -> canonical records -> accumulated labels -> D#### frozen dataset`

Raw sources remain immutable. Labels retain their producer and authority. Live filters and source stacks can be explored interactively, but a training run consumes only a frozen `D####` manifest.

When parsers, schemas, motif definitions, POV conventions, deduplication, or facts registries change, the lab creates a new `N####` and rebuilds from the original source. Historical datasets and runs remain unchanged, and the migration is diffed and documented.

See `docs/DATA_LIFECYCLE.md`.

## Initial research phases

### Phase 0 — Lab foundation

Build schemas, provenance, immutable run evidence, data lineage/standardization, compute accounting, frozen evaluation suites, and the human GUI.

### Phase 1 — Raw NNUE scaling

Establish the smallest useful raw evaluator by sweeping hidden width from absurdly small networks upward under one frozen training/evaluation contract.

### Phase 2 — Semantic geometry scaling

Repeat the scaling study using CVS geometry inputs. Determine how much capability explicit structure buys per learned parameter.

### Phase 3 — Matched-budget architecture comparisons

At matched learned-parameter budgets, compare raw-only, geometry-only, and hybrid evaluators.

### Phase 4 — Semantic ablations

Systematically switch motif, strategy, geometry, and semantic families on/off. Preserve interactions and negative results.

### Phase 5 — Supervision ablations

Hold architecture fixed while varying training signal: scalar eval, deterministic facts, motif labels, transition labels, game outcome, deep CVS search, and sparse external-oracle supervision.

### Phase 6 — Search-compute scaling

Freeze evaluators and vary executed search compute. Measure capability as a function of both stored parameters and executed compute.

### Phase 7 — Information-gain labeling

Use cheap board facts and shallow analysis at scale, then spend deep CVS/Stockfish compute only on uncertain, rare, unstable, or high-value positions.

### Phase 8 — Scale only winners

Architectures and ideas advance only after smaller controlled experiments justify the additional compute.

## Human GUI

The GUI exists primarily to let a human understand the research system.

It should visualize:

- live and historical model training performance;
- supervision composition and label provenance;
- corpus composition and filterable labeled data;
- dataset stacking/sampling and effective training contribution;
- raw -> standardized -> labeled -> frozen-dataset lineage;
- re-standardization and migration differences;
- ablation/control differences;
- capability vs parameters/search compute;
- promotion evidence, missing gates, and contradictory results;
- standardized findings, including negative and inconclusive results.

The GUI may create canonical experiment/data specifications, but hidden scientific logic belongs in the backend. Every displayed point must drill down to its immutable evidence and lineage.

See `docs/GUI.md`.

## Scientific result states

Use evidence-oriented states rather than `good`/`bad`:

- `PROPOSED`
- `RUNNING`
- `SUPPORTED`
- `REJECTED`
- `INCONCLUSIVE`
- `SUPERSEDED`
- `INVALID`

Promotion is a separate evidence decision and must expose its control, gates, uncertainty, integrity state, and dataset differences.

A negative result is still a completed research result. An invalid run is never silently promoted into evidence.

## Legacy intake

The engine and app repos predate the lab. Before Phase 0 assigns permanent identities,
`tools/intake/legacy_catalog.py` inventories what already exists, so nothing has to be
rediscovered:

- `catalog/models.json`: every net and calibration (content hash, locations, metadata,
  exact serialized parameter count, known role, whether training provenance is complete)
- `catalog/sources.json`: every training corpus (per-file hashes, rows, sampled fields
  mapped to provenance classes)
- `catalog/evaluations.json`: position suites, opening books, SPRT gate protocols
- `catalog/evidence.json`: every gate and anchor record, mapped to lab result states
- `catalog/engines.json` and `catalog/switches.json`: the engine identity registry and
  the generated search-switch registry
- `catalog/manifest.json`: the repo commits the catalog was generated from

No lab IDs are assigned here. The registry imports entries by content hash. See
`catalog/README.md`.

## Principle

CVS Lab is not trying to build the largest chess network possible.

It is trying to measure, from first principles, how much chess capability can be produced by learned parameters, explicit structure, targeted supervision, and reusable search computation — and where each additional unit actually pays for itself.
