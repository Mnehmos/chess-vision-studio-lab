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
from typing import Annotated, Optional, Union

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
    "queued_at",
)


def parse_id(obj_id: str) -> tuple[str, int]:
    match = ID_RE.match(obj_id or "")
    if not match or match.group(1) not in KINDS:
        raise NotFound(f"not a lab identity: {obj_id!r}")
    return match.group(1), int(match.group(2))


def compute_record_hash(obj: LabModel) -> str:
    return hash_obj(obj.model_dump(mode="json", exclude={"record_hash"}))


@lru_cache(maxsize=None)
def model_adapter(prefix: str) -> TypeAdapter:
    classes = KINDS[prefix][1]
    if len(classes) == 1:
        return TypeAdapter(classes[0])
    return TypeAdapter(Annotated[Union[classes], Field(discriminator="origin")])


def _is_sealed(obj: LabModel) -> bool:
    """Runs seal when they finish; every other object with a record hash seals on creation."""
    if isinstance(obj, Run):
        return obj.status in TERMINAL_RUN_STATUSES
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
        obj = model_adapter(obj_id[0]).validate_json(path.read_bytes())
        if verify:
            self.verify(obj)
        return obj

    def get_as(self, obj_id: str, cls: type):
        """Fetch an identity and require a specific object class (e.g. a lab Run, not imported evidence)."""
        obj = self.get(obj_id)
        if not isinstance(obj, cls):
            raise LabError(f"{obj_id} is a {type(obj).__name__}, not a {cls.__name__}")
        return obj

    def verify(self, obj: LabModel) -> None:
        sealed = getattr(obj, "record_hash", None)
        if sealed is not None and compute_record_hash(obj) != sealed:
            raise TamperError(f"{obj.id}: record hash mismatch — sealed evidence was modified on disk")

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
        if _is_sealed(obj) or obj.id[0] not in MUTABLE_STATE_PREFIXES | {"R"}:
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
