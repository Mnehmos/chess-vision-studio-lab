# Human GUI Contract

The CVS Lab GUI is the human-facing control surface for the research system. It must remain thin: the GUI edits and launches versioned experiment specifications; it does not contain hidden scientific logic that cannot be reproduced from the CLI.

## Design goals

- Local-first and cheap to run.
- Human-readable before machine-clever.
- Every displayed result links back to immutable evidence.
- Every launch shows the exact intervention relative to the chosen control.
- No experiment requires manual JSON/YAML editing.
- GUI and CLI operate on the same schemas and produce the same run records.

## Primary screens

### 1. Lab overview

Show:

- active generation;
- current frozen baselines;
- queued/running/completed experiments;
- compute consumed by CPU/GPU/oracle tier;
- recent findings;
- current parameter/search Pareto frontier;
- unresolved/inconclusive hypotheses.

### 2. Ablation builder

Start from a frozen baseline and expose only explicit interventions.

Controls should include:

- family: NNUE / LLM / later research families;
- generation;
- baseline;
- exact parameter target or architectural dimensions;
- raw/geometry/hybrid branches;
- semantic-family switches;
- motif/strategy switches where defined;
- training-signal switches;
- search budget;
- dataset manifest;
- seeds/repetitions;
- evaluation suite.

Before launch, display:

- canonical ablation ID/signature;
- exact learned parameter count;
- changed fields vs baseline;
- estimated compute budget when available;
- controls that remain frozen;
- leakage/provenance warnings;
- whether the comparison is valid under the current protocol.

### 3. Switch matrix

A compact grid for systematic ablation review.

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
- hidden width / parameter count

Cells show enabled/disabled/value plus result state. The matrix must support filtering by generation, baseline, parameter budget, dataset, and evaluation suite.

### 4. Scaling explorer

Plot and tabulate:

- capability vs learned parameters;
- capability vs search nodes/time;
- capability vs total compute;
- model throughput vs parameters;
- label quality/value vs labeling CPU cost.

Overlay architecture families such as raw, geometry, hybrid, motif-supervised, and active-labeled models.

### 5. Dataset explorer

Show corpus composition rather than only row count.

Track:

- motif and semantic-family coverage;
- game phase;
- material buckets;
- outcomes;
- evaluation buckets;
- provenance/authority classes;
- shallow/deep/oracle label tiers;
- rare and underrepresented categories;
- cost per accepted high-priority example.

### 6. Run detail

Every `R####` page should show:

- hypothesis and ablation;
- exact control/intervention;
- engine commit and model hashes;
- dataset/training/evaluation identities;
- hardware and compute budget;
- stdout/stderr/log artifacts;
- metrics and uncertainty;
- integrity checks;
- resulting model artifacts;
- comparison against the frozen control;
- evidence state.

### 7. Finding card

Every completed ablation should render a compact finding card:

- Hypothesis
- Control
- Intervention
- Evidence
- Result: SUPPORTED / REJECTED / INCONCLUSIVE / INVALID
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
- Hidden/default settings must be displayed in the run identity even if they are omitted from the compact human label.

## First GUI milestone

Keep the first GUI deliberately small:

1. overview;
2. baseline selector;
3. ablation builder;
4. run queue/history;
5. switch-matrix comparison;
6. finding-card view.

Do not build a large analytics product before the underlying experiment schemas and first end-to-end runs exist.

## Suggested implementation boundary

Use one local web app talking to the lab CLI/service layer. The backend owns schemas, validation, run execution, and evidence persistence. The frontend only presents and edits those canonical objects.

A reasonable implementation is a lightweight TypeScript/React frontend with a small local API, but framework choice is secondary to the contract above.
