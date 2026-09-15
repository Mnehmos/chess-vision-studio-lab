"""CVS Lab funnel: lab-owned generation orchestration behind provider contracts.

This package is slice S1 of `docs/TRAINING_DATA_RESEARCH_PLAN.md` — contracts,
the legacy `analyze --serve` provider, the Tier-0 static-input leak guard, and
the fixture parity harness. Triage/orchestration deliberately do not live here
yet (slices S3/S4); nothing in this package may consume its own output as a
dataset — only `cvslab.data.freeze_dataset` produces training inputs.

Boundary rule: providers return evidence with provenance; the lab decides what
becomes a dataset.
"""
from .protocols import (
    DeterministicLabelProducer,
    LabelBatch,
    OracleLabel,
    OracleProducer,
    Position,
    PositionSource,
    SearchLabelProducer,
    SearchLabel,
)
from .leakguard import LEAK_GUARD_KEYS, assert_static_input_safe

__all__ = [
    "DeterministicLabelProducer", "LabelBatch", "OracleLabel", "OracleProducer",
    "Position", "PositionSource", "SearchLabel", "SearchLabelProducer",
    "LEAK_GUARD_KEYS", "assert_static_input_safe",
]
