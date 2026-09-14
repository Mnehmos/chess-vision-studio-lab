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

## Research model

CVS Lab treats the production engine as a pinned subject under test. The engine repository is not silently modified during an experiment. Every experiment records the exact engine commit, model identities, dataset manifests, search budget, seeds, hardware, and evaluation protocol.

The lab uses permanent identities:

- `H####` — hypothesis
- `A####` — ablation
- `M####` — model artifact
- `D####` — dataset
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

## Initial research phases

### Phase 0 — Lab foundation

Build schemas, provenance, immutable run evidence, compute accounting, frozen evaluation suites, and the human GUI.

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

The lab must be usable without editing config files. The GUI is part of the core research system, not a separate product.

A researcher should be able to:

- browse generations, hypotheses, ablations, models, datasets, and runs;
- create an ablation from a baseline using switches and numeric controls;
- see exact parameter counts before launching;
- configure search/data/label budgets;
- queue and monitor runs;
- compare controls and interventions;
- filter by motif, strategy, architecture, generation, parameter budget, and result state;
- inspect coverage and labeling-compute spend;
- view capability-vs-parameters-vs-search-compute charts;
- read standardized finding cards for supported, rejected, and inconclusive experiments.

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

A negative result is still a completed research result. An invalid run is never silently promoted into evidence.

## Principle

CVS Lab is not trying to build the largest chess network possible.

It is trying to measure, from first principles, how much chess capability can be produced by learned parameters, explicit structure, targeted supervision, and reusable search computation — and where each additional unit actually pays for itself.
