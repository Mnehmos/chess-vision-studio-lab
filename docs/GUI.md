# Human GUI Contract

The CVS Lab GUI is primarily an observability and research-workbench surface for a human researcher.

Its first job is to make the research legible:

- how a model is training;
- what supervision it is receiving;
- what data it is seeing;
- what an ablation changed;
- how the intervention compares with its control;
- whether evidence is sufficient for promotion;
- what data, model, and protocol versions produced a finding.

The GUI may launch canonical experiments, but it must not become the source of scientific truth or hide logic that cannot be reproduced from the CLI/backend.

## Design goals

- Local-first and cheap to run.
- Human-readable before machine-clever.
- Visualization/inspection first; configuration second.
- Every displayed result links back to immutable evidence.
- Every graph can be filtered back to the records that produced it.
- Every launch shows the exact intervention relative to the chosen control.
- No experiment requires manual JSON/YAML editing.
- GUI and CLI operate on the same schemas and produce the same run records.
- Raw data, standardized data, labels, dataset manifests, runs, and findings all expose lineage.

## Primary screens

### 1. Lab overview

The opening screen should answer: "What is happening in the lab?"

Show:

- active generation;
- current frozen baselines;
- queued/running/completed experiments;
- current training jobs and progress;
- compute consumed by CPU/GPU/oracle tier;
- recent findings and promotion decisions;
- current parameter/search Pareto frontier;
- unresolved/inconclusive hypotheses;
- corpus health and migration warnings;
- current data/normalization/ontology versions.

### 2. Training monitor

This is the primary model-performance visualization.

For an active or completed training run, show where available:

- train/validation loss by step/epoch;
- held-out evaluation metrics;
- learning-rate schedule;
- throughput and elapsed compute;
- parameter count and architecture;
- checkpoint progression;
- gradient/activation diagnostics only when explicitly collected;
- evaluation-slice performance by motif, strategy, game phase, material bucket, and label provenance;
- comparison with the frozen control under the same instrument.

The GUI must distinguish training metrics from actual playing-strength evidence. A better loss curve is not displayed as a promotion claim.

### 3. Supervision inspector

Show exactly what is teaching the model.

For a training run or dataset manifest, visualize:

- scalar evaluation targets;
- deterministic geometry/fact labels;
- motif/strategy labels;
- position/move/transition supervision;
- shallow CVS labels;
- deep CVS labels;
- game outcomes;
- external-oracle labels;
- source mixture and sampling weights;
- label authority/provenance classes;
- missing/unavailable labels;
- class balance and rare categories.

A human should be able to answer: "What percentage of this run's effective training examples came from each source and supervision type?"

### 4. Dataset explorer / corpus workbench

This is a first-class screen, not a static row-count page.

The complete canonical catalog should be filterable by combinations of:

- source/source version;
- normalization version;
- engine generation/commit;
- model/evaluator identity;
- motif and motif family;
- strategy/semantic family;
- deterministic fact presence;
- label provenance/authority;
- shallow/deep/oracle tier;
- game phase;
- material configuration;
- side to move;
- game result/WDL;
- evaluation bucket;
- shallow/deep disagreement;
- best-move/PV stability;
- information-gain priority;
- duplicate/quarantine status;
- dataset and split membership.

Every filter update should show:

- retained row/position/transition count;
- unique count;
- source composition;
- motif/strategy coverage;
- label-tier composition;
- distribution shifts relative to the unfiltered corpus;
- estimated available deep/oracle labels;
- effective training contribution when stacking/weighting is active.

The user can create a live data view, stack sources or subpopulations, preview the resulting distribution, and then freeze the exact selection into an immutable `D####` manifest.

See `docs/DATA_LIFECYCLE.md`.

### 5. Data standardization and migration view

The GUI should make schema/ontology evolution understandable.

Show the lineage:

`S#### raw source -> N#### normalization -> canonical records -> labels -> D#### frozen dataset`

When a normalization recipe, parser, motif definition, POV convention, deduplication rule, or facts registry changes, show:

- old vs new standardization version;
- rows added/removed/changed;
- duplicate changes;
- changed labels;
- coverage deltas;
- affected frozen datasets/models;
- records requiring relabeling;
- migration/rebuild status.

Provide an explicit "rebuild under new standard" workflow. It creates a new canonical version and a new descendant `D####`; it never mutates historical evidence in place.

### 6. Ablation matrix

A compact grid for systematic intervention review.

Rows are ablations. Columns are registered switches/variables.

Examples for NNUE:

- RAW
- GEO
- KING_SAFETY
- PAWN_STRUCTURE
- TACTICS
- MOBILITY
- COORDINATION
- ROOK_FILE
- motif-family switches
- strategy-family switches
- hidden width / exact parameter count
- supervision switches
- search budget where relevant

Cells show enabled/disabled/value plus result state. Filters include generation, baseline, parameter budget, dataset, training recipe, evaluation suite, and promotion state.

The matrix should support control-vs-intervention diff highlighting so a human can immediately see what changed.

### 7. Scaling explorer

Plot and tabulate:

- capability vs learned parameters;
- capability vs search nodes/time;
- capability vs total compute;
- model throughput vs parameters;
- label quality/value vs labeling CPU cost;
- performance by semantic/motif slice as scale changes.

Overlay architecture families such as raw, geometry, hybrid, motif-supervised, strategy-supervised, and active-labeled models.

The user should be able to click a point and inspect the exact model, dataset, run, and evidence behind it.

### 8. Promotion view

Promotion is an evidence review, not a green button.

For a candidate model/ablation, show:

- frozen control identity;
- exact intervention;
- training/instrument metrics;
- held-out slice performance;
- fixed-node/fixed-time search comparisons as applicable;
- game-gate evidence and uncertainty;
- parameter/throughput/search-compute tradeoffs;
- integrity checks;
- data provenance and dataset-manifest differences;
- required gates and their pass/hold/fail/invalid state;
- contradictory evidence;
- compute spent to reach the decision.

Promotion state should be explicit, for example:

- `NOT_ELIGIBLE`
- `READY_FOR_GATE`
- `GATING`
- `PROMOTED`
- `REJECTED`
- `HOLD`
- `INVALID`

A human must be able to see why the decision was reached and which evidence is missing.

### 9. Run detail

Every `R####` page should show:

- hypothesis and ablation;
- exact control/intervention;
- engine commit and model hashes;
- dataset/training/evaluation identities;
- source/normalization/ontology versions;
- hardware and compute budget;
- stdout/stderr/log artifacts;
- training curves and metrics;
- supervision composition;
- evaluation slices and uncertainty;
- integrity checks;
- resulting model artifacts;
- comparison against the frozen control;
- evidence and promotion state.

### 10. Finding card

Every completed ablation should render a compact finding card:

- Hypothesis
- Control
- Intervention
- Evidence
- Result: SUPPORTED / REJECTED / INCONCLUSIVE / INVALID
- Promotion consequence if any
- Interpretation
- What this does *not* establish
- Next justified experiment

Negative findings remain visible and searchable.

## Interaction rules

- The GUI never mutates a completed run record.
- Re-running the same ablation creates a new `R####`.
- Changing the intervention creates a new `A####`.
- Changing the scientific regime enough to invalidate direct comparison requires a new generation.
- Parameter count is calculated, not manually typed into result records.
- Hidden/default settings must be displayed in the run identity even if omitted from the compact human label.
- Live dataset filters are exploratory until frozen into `D####`.
- Historical `D####` manifests never change when standardization changes.
- Re-standardization creates a new `N####` and new descendant datasets.
- Charts must distinguish measured values, derived values, and estimates.

## First GUI milestone

Keep implementation small, but make the first vertical slice representative:

1. overview;
2. training monitor;
3. supervision inspector;
4. dataset explorer with filters;
5. one versioned standardization lineage view;
6. ablation matrix;
7. run detail;
8. promotion evidence view;
9. finding card.

The first version does not need elaborate styling or every possible chart. It does need trustworthy lineage and drill-down.

## Suggested implementation boundary

Use one local web app talking to the lab CLI/service layer. The backend owns schemas, data indexing, standardization, validation, run execution, evidence persistence, and promotion gates. The frontend presents those canonical objects and allows safe creation of new views/specifications.

A lightweight TypeScript/React frontend with a small local API is reasonable, but framework choice is secondary to the scientific contract.
