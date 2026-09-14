# Data Lifecycle and Corpus Contract

CVS Lab treats data as a versioned research asset, not as a collection of ad-hoc training files.

The goals are:

- preserve every raw source exactly as acquired;
- standardize heterogeneous sources into one canonical record model;
- accumulate labels without erasing their provenance;
- compose/stack datasets reproducibly through queries and manifests;
- allow the entire corpus to be re-standardized when schemas, ontologies, or validators improve;
- make every training run reconstructible from immutable inputs and versioned transformations.

## Data identities

The lab extends the core identity system with:

- `S####` — immutable source snapshot
- `N####` — normalization / standardization recipe
- `D####` — frozen dataset manifest/view

Existing research identities remain:

- `H####` hypothesis
- `A####` ablation
- `M####` model
- `T####` training recipe
- `E####` evaluation protocol
- `R####` run

A source is not a dataset. A normalized corpus is not a training split. A dataset is a frozen selection over canonical records.

## Four data layers

### L0 — Raw source

Store or reference source material exactly as acquired.

Examples:

- CVS self-play games
- live CVS games
- Lichess / Chess.com game exports where licensing permits
- engine-search traces
- Stockfish audit labels
- synthetic motif positions

Every source snapshot records:

- source identity and origin;
- acquisition date/version;
- content hash(es);
- licensing/usage metadata where relevant;
- parser/importer version;
- row/game counts;
- whether the bytes are local, remote, or externally referenced.

L0 is immutable. Fixing a parser or changing a definition never rewrites the raw source.

### L1 — Canonical standardized records

`N####` transforms one or more L0 sources into the current canonical schema.

The canonical representation should normalize at least:

- board/FEN representation;
- side to move;
- castling/en-passant state;
- move encoding;
- game/ply identity;
- position and transition identity;
- game result;
- source provenance;
- duplicate identity;
- game phase/material buckets when defined deterministically.

A normalization recipe records code commit, schema version, all transformation settings, filters, deduplication policy, and hashes.

Canonical records receive stable content-derived identities wherever practical so the same chess state/transition can be recognized across sources and re-standardization passes.

### L2 — Enriched labels

Labels attach to canonical records without replacing the underlying record.

Examples:

- deterministic CVS board facts;
- motif/strategy labels;
- move and transition labels;
- SEE/bounded tactical proof;
- shallow CVS search telemetry;
- deep CVS search targets;
- game outcomes;
- Stockfish/external-oracle labels;
- future human annotations.

Every label carries provenance:

- label family and semantic identity;
- authority class;
- producer/engine/model identity;
- search budget where applicable;
- ontology/facts-registry version;
- label schema version;
- timestamp/run identity;
- confidence/validity state where defined.

Labels from different authorities are not silently collapsed into one value. The same position may legitimately have several evaluation labels from different engines, depths, generations, or dates.

### L3 — Dataset views and frozen manifests

A dataset is a reproducible query/selection over L1 + L2.

Examples:

- all quiet middlegames with deterministic geometry labels;
- all positions containing an interference motif;
- balanced samples across motif families;
- CVS self-play only;
- Lichess + self-play stacked at a declared ratio;
- positions with shallow/deep disagreement above a threshold;
- semantic corpus with no expensive centipawn target;
- evaluation corpus requiring deep-search labels.

The GUI may create a live preview of a query, but training requires freezing it into `D####`.

A frozen `D####` records:

- source snapshots used;
- normalization recipe(s);
- canonical schema version;
- ontology/facts-registry versions;
- filter/query specification;
- stacking/sampling weights;
- deduplication policy;
- required label provenance/tier;
- exact row identities or deterministic selection seed;
- train/validation/test split identities;
- counts and coverage summaries;
- manifest hash.

A completed run references `D####`; it never references an unversioned folder such as `data/latest`.

## Dataset stacking

Stacking means composing several source populations under an explicit policy.

A stack must declare whether components are:

- concatenated naturally;
- sampled to fixed row counts;
- weighted by source;
- balanced by motif/strategy family;
- balanced by game phase/material state;
- stratified by label authority;
- curriculum ordered.

The GUI should show both raw available rows and effective sampled contribution. A 20M-row source that contributes 10% of a training mixture must not visually appear as if it dominates the actual training distribution.

No silent oversampling or source weighting is allowed.

## Re-standardization and migrations

Standardization will change. That is expected.

Examples:

- a motif definition is corrected;
- a POV convention changes;
- a duplicate detector improves;
- a canonical field is added;
- facts-registry semantics change;
- a parser bug is fixed.

When this happens:

1. create a new `N####` recipe/version;
2. re-run it from immutable L0 source snapshots;
3. produce a new canonical corpus version;
4. recompute labels whose meaning/input dependency changed;
5. retain labels that are provably unaffected;
6. generate a migration/diff report;
7. never mutate prior `D####` manifests or completed `R####` evidence.

The migration report should show:

- records added/removed/changed;
- deduplication changes;
- label additions/removals/value changes;
- motif/strategy coverage deltas;
- source contribution deltas;
- split-membership changes;
- affected historical dataset manifests and models.

The lab should support "rebuild this dataset under the new standard" as an explicit action. That creates a new `D####` descended from the old one, with the lineage and differences visible.

## Filterable catalog

The GUI dataset explorer should allow filtering across the complete canonical catalog by combinations of:

- source / source version;
- normalization version;
- CVS engine generation/commit;
- evaluator/model identity;
- motif and motif family;
- strategy / semantic family;
- deterministic fact presence;
- label provenance/authority class;
- shallow/deep/oracle label tier;
- game phase;
- material configuration;
- side to move;
- result/WDL;
- evaluation bucket;
- best-move/PV stability;
- shallow/deep disagreement;
- priority/information-gain score;
- duplicate status;
- validation state;
- dataset/split membership.

Filters must expose counts and distribution changes before a dataset is frozen.

## Corpus health

The GUI should visualize:

- total canonical positions/transitions;
- unique vs duplicate counts;
- source composition;
- semantic/motif coverage;
- rare/underrepresented classes;
- labels by authority/tier;
- unlabeled gaps;
- invalid/quarantined rows;
- train/validation/test contamination checks;
- standardization version coverage;
- rows requiring migration/relabeling;
- labeling compute spent by family and source.

## Immutability and lineage rules

- Raw source snapshots are immutable.
- Completed runs are immutable.
- Frozen dataset manifests are immutable.
- New standardization creates a new normalized version, never an in-place rewrite of historical evidence.
- New labels append evidence; they do not erase older labels.
- A changed label definition requires a new semantic/registry version.
- Every model must identify the exact frozen dataset manifest used to train it.
- Every GUI chart/filter must be traceable to underlying row identities and provenance.

## Core principle

> Keep the raw evidence forever, standardize through versioned transforms, accumulate labels with provenance, and train only from frozen reproducible views.
