"""Corrected snapshot for the S7-S pool, and a re-checked universe.

S0010 was created while write_pool still filled the manifest-level openingSourceSha256 from its
`opening_lines` argument (the 12-line default, 588f3688…) instead of the config's declared
source. Its config block carried the correct EPD hash (4d96e552…), so the pool bytes were never
in doubt; the snapshot's recorded source hash was. This step re-snapshots the SAME pool bytes
with the fixed code (S####), re-normalizes (N####), and proves the universe and its order are
unchanged, because both are pure functions of the record ids.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import chess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.pool import snapshot_source
from cvslab.hashing import hash_obj
from cvslab.schemas import Dataset
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s7scaling")
POOL = OUT / "pool"
STATE = TOOLS / "s7-scaling-state.json"
ORDER_SEED = 20260920
UNIVERSE_TARGET = 60_000


def record_id(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def main() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8"))
    service = LabService(Store(str(LAB / "labstore")))
    manifest = json.loads((POOL / "pool-manifest.json").read_text(encoding="utf-8"))
    generated = {"games": read_jsonl(POOL / "games.jsonl"), "positions": read_jsonl(POOL / "positions.jsonl")}

    corrected = dict(manifest)
    corrected["openingSourceSha256"] = manifest["config"]["openingSourceSha256"]
    corrected.pop("manifestSha256", None)
    corrected["manifestSha256"] = hash_obj(corrected)
    (POOL / "pool-manifest-corrected.json").write_text(json.dumps(corrected, indent=1, sort_keys=True) + "\n",
                                                       encoding="utf-8")

    source = snapshot_source(service.store, POOL, name="s7s-pool-corrected", license="engine self-play corpus",
                             generated=generated, manifest=corrected)
    normalization = service.normalize([source.id], name="s7s-canonical-corrected")
    records = _load_canonical(service.store, service.store.get(normalization.id))
    print(f"corrected source {source.id}: openingSourceSha256 {source.generator['openingSourceSha256'][:24]}…; "
          f"normalization {normalization.id}: {len(records)} records", flush=True)

    previous = json.loads(pathlib.Path(TOOLS / "s7-scaling-state.json").read_text(encoding="utf-8"))
    assert source.generator["openingSourceSha256"] == "sha256:4d96e552fd257e66fe39d78420e1e48cfed6753795dd0a3e20cd222a3757e6b3"

    eval_records = _load_canonical(service.store, service.store.get("N0010"))
    d10: Dataset = service.store.get_as("D0010", Dataset)
    instrument_ids = sorted(row["record_id"] for row in read_jsonl(service.store.abs(d10.splits[2].path)))
    parent: dict = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    component_of = {record["record_id"]: record["group"] for record in records}
    for record in eval_records:
        if record["record_id"] in component_of:
            union(component_of[record["record_id"]], record["group"])
        component_of.setdefault(record["record_id"], record["group"])
    for record in records:
        find(record["group"])
    protected = {find(component_of[rid]) for rid in instrument_ids}
    population = {find(component_of[record["record_id"]]) for record in eval_records}
    removed = {r["record_id"] for r in records
               if find(component_of[r["record_id"]]) in protected or
               find(component_of[r["record_id"]]) in population}
    clean = sorted(record["record_id"] for record in records if record["record_id"] not in removed)
    order = sorted(clean, key=lambda rid: hashlib.sha256(f"{ORDER_SEED}:{rid}".encode()).hexdigest())
    if len(clean) < UNIVERSE_TARGET:
        print(f"STOP: only {len(clean)} clean candidates")
        return 3
    if previous["universe"]["universe_hash"] != hash_obj(clean) or previous["order"] != order:
        print("STOP: the corrected normalization yields a different universe/order; the pipeline "
              "must be restarted from the universe step")
        return 3
    state = dict(previous)
    state["universe"] = dict(previous["universe"], source=source.id, normalization=normalization.id,
                             corrected_from=previous["universe"]["source"],
                             note="S0010's manifest-level openingSourceSha256 held the 12-line default; "
                                  "the config block carried the frozen EPD hash; the pool bytes and the "
                                  "universe/order are unchanged (verified equal)")
    STATE.write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"source": source.id, "normalization": normalization.id, "clean": len(clean),
                      "universe_hash": hash_obj(clean)[:24] + "…", "order_hash": hash_obj(order)[:24] + "…",
                      "universe_unchanged": True}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
