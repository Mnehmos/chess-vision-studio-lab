"""File-backed canonical object store with fail-closed immutability.

Layout under the store root::

    objects/<kind>/<ID>.json     canonical objects (one file per identity)
    sources/ canonical/ datasets/ data layers L0, L1, L3
    artifacts/<R####>/           run artifacts (model, eval, log)
    lab.json                     lab configuration (active generation, engine repo)

Rules enforced here rather than trusted to callers:

* identities are never reused or overwritten;
* completed/invalid runs and findings are sealed with a record hash, written
  read-only, and verified on every read;
* hypotheses and ablations may only change their evidence state.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Mapping, Optional, Union

from pydantic import Field, TypeAdapter

from .hashing import hash_obj, sha256_file
from .schemas import (
    KINDS,
    MUTABLE_STATE_PREFIXES,
    TERMINAL_RUN_STATUSES,
    EvidenceState,
    LabModel,
    Run,
    StateChange,
    utc_now,
)


class LabError(Exception):
    status_code = 400


class NotFound(LabError):
    status_code = 404


class ImmutableError(LabError):
    status_code = 409


class TamperError(LabError):
    status_code = 409


ID_RE = re.compile(r"^([A-Z])(\d{4,})$")

# Fields of a run fixed at queue time; execution may never alter them.
RUN_IDENTITY_FIELDS = (
    "ablation_id",
    "seed",
    "effective_config",
    "config_hash",
    "identity_hash",
    "param_count",
    "dataset_id",
    "dataset_manifest_hash",
    "training_recipe_id",
    "recipe_hash",
    "eval_protocol_id",
    "protocol_hash",
    # supervision identities and the divergence declaration are fixed at queue time like
    # every other identity field: execution may never rewrite what taught or judged the model
    "train_target_spec_hash",
    "eval_target_spec_hash",
    "supervision_divergence",
    "experiment_id",
    "queued_at",
)

# Fields of an experiment fixed when it is created: runs are produced against them, so the
# design may not drift while the study is running (only status/result/state_history move).
EXPERIMENT_FROZEN_FIELDS = (
    "name", "family", "generation", "preregistration_hash", "reference_unit_nodes", "scales",
    "node_budgets", "source_id", "normalization_id", "candidate_universe_hash", "order_seed",
    "order_hash", "arms", "eval_dataset_id", "eval_dataset_manifest_hash", "eval_protocol_id",
    "eval_protocol_hash", "eval_target_spec_hash", "widths", "seeds", "expected_run_count",
    "analysis",
)


def parse_id(obj_id: str) -> tuple[str, int]:
    match = ID_RE.match(obj_id or "")
    if not match or match.group(1) not in KINDS:
        raise NotFound(f"not a lab identity: {obj_id!r}")
    return match.group(1), int(match.group(2))


def payload_hash(payload: Mapping[str, object]) -> str:
    """Hash of a RECORDED payload: the JSON as written, minus the hash field itself."""
    return hash_obj({key: value for key, value in payload.items() if key != "record_hash"})


def compute_record_hash(obj: LabModel) -> str:
    """Hash of the object as it will be recorded."""
    return payload_hash(obj.model_dump(mode="json"))


@lru_cache(maxsize=None)
def model_adapter(prefix: str) -> TypeAdapter:
    classes = KINDS[prefix][1]
    if len(classes) == 1:
        return TypeAdapter(classes[0])
    return TypeAdapter(Annotated[Union[classes], Field(discriminator="origin")])


def _is_sealed(obj: LabModel) -> bool:
    """Runs seal when they finish, experiments when their result is recorded, everything else
    with a record hash seals on creation."""
    if isinstance(obj, Run):
        return obj.status in TERMINAL_RUN_STATUSES
    if type(obj).__name__ == "Experiment":
        return getattr(obj, "status", "") == "SEALED"
    return "record_hash" in type(obj).model_fields


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._reserved: dict[str, int] = {}
        for folder, _ in KINDS.values():
            (self.root / "objects" / folder).mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, home: Optional[str | Path] = None) -> "Store":
        return cls(home or os.environ.get("CVSLAB_HOME") or "labstore")

    # -- paths ---------------------------------------------------------------

    def object_path(self, obj_id: str) -> Path:
        prefix, _ = parse_id(obj_id)
        return self.root / "objects" / KINDS[prefix][0] / f"{obj_id}.json"

    def abs(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if self.root not in path.parents and path != self.root:
            raise LabError(f"path escapes the lab store: {rel}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    # -- identities ----------------------------------------------------------

    def _max_number(self, prefix: str) -> int:
        folder = self.root / "objects" / KINDS[prefix][0]
        numbers = [parse_id(p.stem)[1] for p in folder.glob(f"{prefix}*.json") if ID_RE.match(p.stem)]
        return max(numbers + [self._reserved.get(prefix, 0)])

    def peek_id(self, prefix: str) -> str:
        with self._lock:
            return f"{prefix}{self._max_number(prefix) + 1:04d}"

    def next_id(self, prefix: str) -> str:
        with self._lock:
            number = self._max_number(prefix) + 1
            self._reserved[prefix] = number
            return f"{prefix}{number:04d}"

    # -- reads ---------------------------------------------------------------

    def exists(self, obj_id: str) -> bool:
        return self.object_path(obj_id).exists()

    def get(self, obj_id: str, *, verify: bool = True):
        path = self.object_path(obj_id)
        if not path.exists():
            raise NotFound(f"{obj_id} not found")
        raw = path.read_bytes()
        obj = model_adapter(obj_id[0]).validate_json(raw)
        if verify:
            self._verify_recorded(obj_id, json.loads(raw))
        return obj

    def get_as(self, obj_id: str, cls: type):
        """Fetch an identity and require a specific object class (e.g. a lab Run, not imported evidence)."""
        obj = self.get(obj_id)
        if not isinstance(obj, cls):
            raise LabError(f"{obj_id} is a {type(obj).__name__}, not a {cls.__name__}")
        return obj

    def verify(self, obj: LabModel) -> None:
        """Verify sealed evidence against the payload AS RECORDED.

        The seal covers what was written, not the object as the current code models it, so
        a later schema addition can never turn historical evidence into apparent tampering
        (and tampering with anything actually recorded is still detected).
        """
        path = self.object_path(obj.id)
        if path.exists():
            self._verify_recorded(obj.id, json.loads(path.read_bytes()))
            return
        # not yet persisted (an object under construction): verify the in-memory dump
        sealed = getattr(obj, "record_hash", None)
        if sealed is not None and compute_record_hash(obj) != sealed:
            raise TamperError(f"{obj.id}: record hash mismatch — object does not match its seal")

    def _verify_recorded(self, obj_id: str, payload: Mapping[str, object]) -> None:
        sealed = payload.get("record_hash")
        if sealed is not None and payload_hash(payload) != sealed:
            raise TamperError(f"{obj_id}: record hash mismatch — sealed evidence was modified on disk")

    def list(self, prefix: str, *, verify: bool = True, kind: Optional[type] = None) -> list:
        folder = self.root / "objects" / KINDS[prefix][0]
        ids = sorted((p.stem for p in folder.glob(f"{prefix}*.json") if ID_RE.match(p.stem)), key=lambda s: parse_id(s)[1])
        objects = [self.get(obj_id, verify=verify) for obj_id in ids]
        return [o for o in objects if isinstance(o, kind)] if kind else objects

    # -- writes --------------------------------------------------------------

    def _write(self, path: Path, obj: LabModel) -> None:
        payload = json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        if _is_sealed(obj) or obj.id[0] not in MUTABLE_STATE_PREFIXES | {"R", "X"}:
            self.make_readonly(path)

    def create(self, obj: LabModel):
        expected = KINDS[parse_id(obj.id)[0]][1]
        if not isinstance(obj, expected):
            raise LabError(f"{obj.id} must be one of {', '.join(c.__name__ for c in expected)}")
        with self._lock:
            path = self.object_path(obj.id)
            if path.exists():
                raise ImmutableError(f"{obj.id} already exists; identities are never reused or overwritten")
            if _is_sealed(obj):
                obj.record_hash = None
                obj.record_hash = compute_record_hash(obj)
            self._write(path, obj)
        return obj

    def update_run(self, run: Run) -> Run:
        with self._lock:
            current = self.get(run.id)
            if not isinstance(current, Run):
                raise ImmutableError(f"{run.id} is imported {type(current).__name__} evidence and cannot be updated")
            if current.status in TERMINAL_RUN_STATUSES:
                raise ImmutableError(
                    f"{run.id} is {current.status.value}; completed run records are immutable. "
                    "Re-running the ablation creates a new R####."
                )
            for field in RUN_IDENTITY_FIELDS:
                if getattr(current, field) != getattr(run, field):
                    raise ImmutableError(f"{run.id}: '{field}' is fixed when the run is queued")
            run.record_hash = None
            if run.status in TERMINAL_RUN_STATUSES:
                run.record_hash = compute_record_hash(run)
            self._write(self.object_path(run.id), run)
        return run

    def update_experiment(self, experiment):
        """Move an experiment from FROZEN to SEALED (result + state history), or refuse.

        The design frozen at creation is immutable: only status, result and state history may
        change, so a study cannot be redefined once its runs exist.
        """
        with self._lock:
            current = self.get(experiment.id)
            if type(current).__name__ != "Experiment":
                raise ImmutableError(f"{experiment.id} is {type(current).__name__}, not an experiment")
            if current.status == "SEALED":
                raise ImmutableError(f"{experiment.id} is SEALED; a finished study is immutable")
            for field in EXPERIMENT_FROZEN_FIELDS:
                if getattr(current, field) != getattr(experiment, field):
                    raise ImmutableError(f"{experiment.id}: '{field}' is fixed when the experiment is created")
            experiment.record_hash = None
            if experiment.status == "SEALED":
                experiment.record_hash = compute_record_hash(experiment)
            self._write(self.object_path(experiment.id), experiment)
        return experiment

    def update_state(self, obj_id: str, state: EvidenceState, reason: str):
        prefix, _ = parse_id(obj_id)
        if prefix not in MUTABLE_STATE_PREFIXES:
            raise ImmutableError(f"{obj_id}: only hypotheses and ablations carry a mutable evidence state")
        with self._lock:
            obj = self.get(obj_id)
            obj.state = state
            obj.state_history.append(StateChange(state=state, at=utc_now(), reason=reason))
            self._write(self.object_path(obj_id), obj)
        return obj

    def write_artifact(self, owner_id: str, name: str, payload: bytes) -> tuple[str, str]:
        rel = f"artifacts/{owner_id}/{name}"
        path = self.abs(rel)
        with self._lock:
            if path.exists():
                raise ImmutableError(f"artifact {rel} already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.make_readonly(path)
        return rel, sha256_file(path)

    @staticmethod
    def make_readonly(path: Path) -> None:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    # -- lab configuration ---------------------------------------------------

    def read_config(self) -> dict:
        path = self.root / "lab.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def write_config(self, config: dict) -> None:
        (self.root / "lab.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
