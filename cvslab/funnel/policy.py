"""Content-addressed triage policy (S3, #15).

A policy is a canonical JSON document holding every value that materially affects
triage; `policy_hash = sha256(canonical_json(policy))`. Policies live immutably at
`policies/<hex>.json` and are referenced everywhere by (policy_version, policy_hash).
No new object prefix is introduced, and `T####` remains exclusively TrainingRecipe.

Build from a legacy funnel config's `tiers.tier2` + `tiers.tier4` + seed, which is
exactly what the frozen priority-v1 implementation consumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..hashing import canonical_json, read_jsonl
from ..store import LabError, Store

POLICY_DIR = "policies"


def prior_counts_hash(counts: dict) -> str:
    """Content identity for a coverage prior: hash of the canonical counts mapping."""
    import hashlib
    return "sha256:" + hashlib.sha256(canonical_json({k: int(v) for k, v in sorted(counts.items())})).hexdigest()


def build_policy(config: dict, *, policy_version: Optional[str] = None,
                 prior_counts: Optional[dict] = None) -> dict:
    """Extract a canonical policy document from a funnel config (tier2 + tier4 + seed).

    A non-null ``priorCountsPath`` must come with its content (``prior_counts``):
    the coverage prior is part of the scientific identity, so it is
    content-addressed — never just a path that could point at different bytes later.
    """
    if "tiers" not in config or "tier2" not in config.get("tiers", {}):
        raise LabError("funnel config has no tiers.tier2 section; cannot extract a triage policy")
    tier2 = config["tiers"]["tier2"]
    tier4 = config["tiers"].get("tier4", {})
    coverage = tier2.get("coverage", {})
    prior_path = coverage.get("priorCountsPath")
    if prior_path and prior_counts is None:
        raise LabError("config sets tiers.tier2.coverage.priorCountsPath; pass its counts so the "
                       "coverage prior can be content-addressed instead of merely referenced by path")
    prior_identity = None
    if prior_path:
        prior_identity = {"source_path": str(prior_path), "counts_hash": prior_counts_hash(prior_counts)}
    return {
        "policy_version": policy_version or config.get("prioritizerVersion") or "priority-v1",
        "seed": int(config["seed"]),
        "selection": {
            "deep_fraction": float(tier2["deepFraction"]),
            "audit_fraction": float(tier2["auditFraction"]),
            "holdout_fraction": float(tier2["holdoutFraction"]),
            "holdout_seed": int(tier2["holdoutSeed"]),
        },
        "weights": {name: float(value) for name, value in sorted(tier2["weights"].items())},
        "caps": {name: float(value) for name, value in sorted(tier2["caps"].items())},
        "coverage_prior": {
            "target_share": float(coverage.get("targetShare", 0.02)),
            "min_target": int(coverage.get("minTarget", 5)),
            "prior_counts": prior_identity,
        },
        "stockfish": {
            "enabled": bool(tier4.get("enabled", True)),
            "priority_fraction": float(tier4.get("priorityFraction", 0.0)),
            "uniform_fraction": float(tier4.get("uniformFraction", 0.0)),
            "audit_fraction": float(tier4.get("auditFraction", 0.0)),
        },
    }


def policy_hash(policy: dict) -> str:
    """sha256 over canonical JSON — the identity referenced by manifests and labels."""
    import hashlib
    return "sha256:" + hashlib.sha256(canonical_json(policy)).hexdigest()


def policy_path(store: Store, digest: str) -> Path:
    hex_digest = digest.split(":", 1)[1]
    return store.abs(f"{POLICY_DIR}/{hex_digest}.json")


def store_policy(store: Store, policy: dict) -> str:
    """Write the policy immutably (idempotent) and return its hash."""
    digest = policy_hash(policy)
    path = policy_path(store, digest)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(policy) + b"\n")
        store.make_readonly(path)
    return digest


def load_policy(store: Store, digest: str) -> dict:
    import json
    path = policy_path(store, digest)
    if not path.is_file():
        raise LabError(f"policy {digest} not found at {path}")
    import hashlib
    payload = path.read_bytes()
    actual = "sha256:" + hashlib.sha256(payload.rstrip(b"\n")).hexdigest()
    if actual != digest:
        raise LabError(f"policy {digest} fails hash verification (stored content hashes to {actual})")
    return json.loads(payload)


def policies_dir(store: Store) -> list[Path]:
    folder = store.abs(POLICY_DIR)
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def prepare_policy(config: dict, *, artifact_root=None, store=None):
    """Build the run's policy, resolving a configured coverage prior **explicitly**.

    A non-null ``priorCountsPath`` is read (relative to the artifact root when not
    absolute); its content hash becomes part of the policy identity. Returns
    (policy, stored_policy_hash_if_store_given).
    """
    import json
    from pathlib import Path as _Path

    tier2 = config.get("tiers", {}).get("tier2", {})
    prior_path = (tier2.get("coverage") or {}).get("priorCountsPath")
    prior_counts = None
    if prior_path:
        resolved = _Path(prior_path)
        if not resolved.is_absolute() and artifact_root:
            resolved = _Path(artifact_root) / prior_path
        if not resolved.is_file():
            raise LabError(f"coverage prior not found: {resolved} (configured as {prior_path})")
        prior_counts = json.loads(resolved.read_text(encoding="utf-8"))["counts"]
    policy = build_policy(config, prior_counts=prior_counts)
    digest = store_policy(store, policy) if store is not None else None
    return policy, digest
