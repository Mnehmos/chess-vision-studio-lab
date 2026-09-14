"""Canonical serialization and hashing. Every identity hash in the lab goes through here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, NaN rejected (fail closed)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> str:
    """Write rows as canonical JSON lines with LF endings; return the file hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        for row in rows:
            fh.write(canonical_json(row))
            fh.write(b"\n")
    return sha256_file(path)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "rb") as fh:
        return [json.loads(line) for line in fh if line.strip()]
