"""S13 shared: paths, the pinned Stockfish teacher, the frozen population, and label purchase.

The teacher is pinned by BYTES: binary SHA256, the UCI name/author it reports, the NNUE net it
announces, and the exact options (Threads/Hash/MultiPV). Every label row records the budget it was
bought under (`{"nodes": N}`) and the realized nodes/depth, so the economics are measured, not
claimed. A Stockfish node is never assumed comparable to a CVS node.
"""
from __future__ import annotations

import concurrent.futures
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.protocols import Position
from cvslab.funnel.stockfish import StockfishProvider, StockfishTransport
from cvslab.hashing import hash_obj

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S13 = LAB / "tools" / "s13"
S12_STATE = LAB / "tools" / "s12" / "s12-state.json"
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s13")

SF_EXE = r"F:\tools\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
SF_SHA256 = "c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911"
SF_NETS = ("nn-c288c895ea92.nnue", "nn-37f18f62d772.nnue")  # big + small, both embedded; SF picks per position
SF_NAME = "Stockfish 18"
SF_OPTIONS = {"Threads": 1, "Hash": 16, "MultiPV": 1}

NORMALIZATION = "N0014"            # the frozen S12 union corpus (merged at a3bec11)
CVS_ARM_DATASET = "D0036"          # S12's 4k arm: 18,953 rows, CVS nodeBudget 4000
CVS_ARM_BUDGET = {"nodeBudget": 4000}
EXAM_DATASET = "D0010"             # the 200 held-out identities, shared by both exams
EXAM_PROTOCOL_CVS = "E0004"        # CVS-DEEP, 400k nodes (existing, unchanged)

AUTHORITY = "oracle.stockfish.18"            # the WARM streams X0009 used (preserved, superseded)
AUTHORITY_COLD = "oracle.stockfish.18.cold"  # cold per label: ucinewgame + Clear Hash + isready
FAMILY = "oracle_cp"
MATE_CP_LIMIT = 100_000            # |cp| at or above this is a mate sentinel, excluded from cp stats

ORDER_SEED = 20260920


def s12_state() -> dict:
    return json.loads(S12_STATE.read_text(encoding="utf-8"))


def order() -> list[str]:
    """The frozen S12 universe order (183,183 ids, order-preserving filtered extension)."""
    return s12_state()["order"]


def training_rows() -> list[str]:
    """The 18,953 rows the CVS arm trained on (D0036's prefix of the frozen order)."""
    state = s12_state()
    arm = next(a for a in state["arms"] if a["arm_id"] == "4k")
    return state["order"][: arm["prefix_size"]]


def record_fens(ids, *, normalization: str = NORMALIZATION) -> dict[str, str]:
    """Stream one canonical corpus once, collecting the FENs of the requested ids."""
    wanted = set(ids)
    found: dict[str, str] = {}
    path = LAB / "labstore" / "canonical" / normalization / "records.jsonl"
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["record_id"]
            if rid in wanted:
                found[rid] = row["fen"]
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} ids are not in {normalization}: {sorted(missing)[:3]}")
    return found


def exam_fens() -> dict[str, str]:
    """The 200 held-out exam identities (D0010's test split) as FENs, from the exam corpus N0006."""
    from cvslab.hashing import read_jsonl
    from cvslab.store import Store

    store = Store(str(LAB / "labstore"))
    dataset = store.get("D0010")
    split = next(entry for entry in dataset.splits if entry.name == "test")
    ids = [row["record_id"] for row in read_jsonl(store.abs(split.path))]
    return record_fens(ids, normalization="N0006")


def sf_provider(cold: bool = True, authority: str = AUTHORITY_COLD) -> StockfishProvider:
    transport = StockfishTransport([SF_EXE], options=SF_OPTIONS)
    transport._start()
    identity = transport.identity
    assert identity.binary_sha256 == SF_SHA256, f"the pinned binary changed: {identity.binary_sha256}"
    assert identity.nets == SF_NETS, f"the pinned nets changed: {identity.nets}"
    return StockfishProvider(transport, authority=authority, family=FAMILY)


def identity_canonical() -> dict:
    provider = sf_provider()
    try:
        canonical = provider.identity.canonical()
    finally:
        provider.transport.close()
    canonical["expected"] = {"binarySha256": SF_SHA256, "nets": list(SF_NETS), "options": SF_OPTIONS}
    return canonical


def _worker(start: int, chunk: list[tuple[str, str]], budget: int,
            cold: bool = True) -> list[dict]:
    provider = sf_provider(cold=cold)
    rows = []
    try:
        for rid, fen in chunk:
            label = provider.search(Position(fen=fen), node_budget=budget, cold=cold)
            rows.append({"record_id": rid,
                         "value": {"scoreCpStm": label.score_cp_stm, "mate": label.mate,
                                   "bestMove": label.best_move, "pv": list(label.pv),
                                   "depth": label.reached_depth, "nodes": label.nodes},
                         "budget_nodes": int(label.nodes), "node_budget": int(budget),
                         "wall_ms": label.wall_ms})
    finally:
        provider.transport.close()
    return rows


def buy(items: list[tuple[str, str]], budget: int, *, workers: int = 8, log: bool = True,
        cold: bool = True) -> list[dict]:
    """Label `items` (record_id, fen) at `budget` nodes, in worker processes, order preserved."""
    chunks = [items[index::workers] for index in range(workers)]
    rows: list[list[dict]] = [[] for _ in chunks]
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, index, chunk, budget, cold)
                   for index, chunk in enumerate(chunks)]
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            rows[index] = future.result()
            if log:
                done = sum(len(part) for part in rows)
                print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(items)} labels at {budget} nodes "
                      f"({time.time() - started:.0f}s)", flush=True)
    flat = [row for part in rows for row in part]
    by_id = {row["record_id"]: row for row in flat}
    assert len(by_id) == len(items), f"{len(by_id)} distinct labels for {len(items)} items"
    return [by_id[rid] for rid, _ in items]


def label_rows(rows: list[dict], *, note: str = "") -> list[dict]:
    """register_labels rows: value + the budget contract of THIS teacher (+ a note when given)."""
    out = []
    for row in rows:
        entry = {"record_id": row["record_id"], "value": row["value"],
                 "budget": {"nodes": row["node_budget"]}}
        if note:
            entry["note"] = note
        out.append(entry)
    return out


def write_json(path: pathlib.Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return hash_obj(payload)
