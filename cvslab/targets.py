"""Frozen supervision target specification (S6).

The S6 treatment is *which records were selected for expensive labeling*; supervision must
therefore be an explicit, frozen spec rather than a hard-coded label family. Matching is
strict: exactly one label per record must satisfy the complete spec. Zero matches and
multiple matches are both integrity failures — the old "first eval_cp wins" behaviour is
deliberately eliminated.

Spec fields (all part of the identity):
family, authority, producer, budget match, value path, POV, target type, K, LAMBDA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hashing import hash_obj
from .store import LabError

TARGET_SPEC_VERSION = 1


@dataclass(frozen=True)
class TargetSpec:
    family: str
    authority: str
    producer: str
    budget: dict = field(default_factory=dict)   # e.g. {"nodeBudget": 400000}
    value_path: tuple[str, ...] = ("value",)     # e.g. ("targets", "scoreCpStm")
    pov: str = "stm"
    target_type: str = "cp"
    k: float = 256.0
    lam: float = 1.0

    def canonical(self) -> dict:
        return {
            "version": TARGET_SPEC_VERSION, "family": self.family, "authority": self.authority,
            "producer": self.producer, "budget": dict(sorted(self.budget.items())),
            "value_path": list(self.value_path), "pov": self.pov, "target_type": self.target_type,
            "k": self.k, "lam": self.lam,
        }

    def spec_hash(self) -> str:
        return hash_obj(self.canonical())


def _dig(value, path: tuple[str, ...]):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def matching_labels(record: dict, spec: TargetSpec) -> list[dict]:
    """Every label satisfying the COMPLETE spec (family, authority, producer, budget, value path)."""
    out = []
    for label in record.get("labels", []):
        if label.get("family") != spec.family:
            continue
        if label.get("authority") != spec.authority or label.get("producer") != spec.producer:
            continue
        budget = label.get("budget") or {}
        if any(budget.get(key) != value for key, value in spec.budget.items()):
            continue
        value = label.get("value")
        if spec.value_path != ("value",):
            value = _dig(value, spec.value_path)
        if value is None:
            continue
        out.append({**label, "_target_value": value})
    return out


def extract_target(record: dict, spec: TargetSpec) -> float:
    """The single supervision value for a record; cardinality must be exactly one."""
    matches = matching_labels(record, spec)
    if not matches:
        raise LabError(f"{record.get('record_id', '?')}: no label satisfies the frozen TargetSpec "
                       f"(family {spec.family}, authority {spec.authority}, producer {spec.producer[:16]}…, "
                       f"budget {spec.budget}, path {'.'.join(spec.value_path)})")
    if len(matches) > 1:
        raise LabError(f"{record.get('record_id', '?')}: {len(matches)} labels satisfy the frozen "
                       "TargetSpec; exactly one is required (ambiguous supervision is an integrity failure)")
    return float(matches[0]["_target_value"])


def validate_records(records: list[dict], spec: TargetSpec) -> int:
    """Validate the spec against a whole population before freezing; returns the match count."""
    for record in records:
        extract_target(record, spec)
    return len(records)
