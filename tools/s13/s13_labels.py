"""S13 labels: buy the Stockfish teacher's streams and the SF-DEEP exam labels.

    python tools/s13/s13_labels.py training   # 18,953 rows at 32k and 1M nodes -> N0014
    python tools/s13/s13_labels.py exam       # 200 held-out ids at 4M nodes -> N0006
    python tools/s13/s13_labels.py disagreement  # CVS vs SF, descriptive only

Every stream is registered as its own authority (`oracle.stockfish.18`) with the executed binary's
SHA256 as producer and `{"nodes": N}` as the budget contract. Positions are the SAME identities the
CVS arm uses: a teacher-specific missing row is a failure, never a silently smaller dataset.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s13_common import (AUTHORITY, FAMILY, LAB, MATE_CP_LIMIT, OUT, S13, SF_SHA256, buy,
                        exam_fens, identity_canonical, label_rows, order, record_fens,
                        training_rows, write_json)

from cvslab.hashing import hash_obj, read_jsonl
from cvslab.service import LabService
from cvslab.store import Store

TRAINING_BUDGETS = {"SF-32k": 32_000, "SF-1M": 1_000_000}
EXAM_BUDGET = 4_000_000
EXAM_NORMALIZATION = "N0006"


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def state_path() -> pathlib.Path:
    return S13 / "s13-state.json"


def state() -> dict:
    path = state_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save(**updates) -> None:
    state_path().write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n",
                            encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def stream_path(label: str) -> pathlib.Path:
    return OUT / f"labels-{label}.jsonl"


def step_training() -> int:
    ids = training_rows()
    fens = record_fens(ids)
    items = [(rid, fens[rid]) for rid in ids]
    print(f"training rows: {len(items)} (D0036's prefix of the frozen order)", flush=True)
    service = svc()
    labels: dict = {}
    for label, budget in TRAINING_BUDGETS.items():
        path = stream_path(label)
        if path.is_file():
            rows = read_jsonl(path)
            print(f"[cached] {label}: {len(rows)} labels", flush=True)
        else:
            print(f"buying {label} ({budget} nodes) for {len(items)} rows", flush=True)
            workers = 8 if budget <= 32_000 else 12
            rows = buy(items, budget, workers=workers)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        by_id = {row["record_id"]: row for row in rows}
        missing = [rid for rid in ids if rid not in by_id]
        if missing:
            raise RuntimeError(f"{label}: {len(missing)} training rows lack a label; refusing")
        registered = service.register_labels(
            "N0014", family=FAMILY, authority=AUTHORITY, producer=SF_SHA256, pov="stm",
            rows=label_rows([by_id[rid] for rid in ids],
                            note=f"{label}: pinned Stockfish 18, {TRAINING_BUDGETS[label]} nodes per label"))
        total = sum(by_id[rid]["budget_nodes"] for rid in ids)
        wall = sum(by_id[rid]["wall_ms"] for rid in ids)
        labels[label] = {"labels": len(ids), "node_budget": TRAINING_BUDGETS[label],
                         "realized_nodes": total, "mean_realized_nodes": total / len(ids),
                         "sum_wall_ms": round(wall, 1),
                         "mean_wall_ms": round(wall / len(ids), 2),
                         "mean_depth": sum(by_id[rid]["value"]["depth"] for rid in ids) / len(ids),
                         "registration": {"path": registered.path, "file_hash": registered.file_hash,
                                          "rows": registered.rows}}
        print(f"{label}: {len(ids)} labels, {total:,} nodes, mean {total / len(ids):.0f}/label, "
              f"mean depth {labels[label]['mean_depth']:.1f}", flush=True)
    save(labels=labels)
    return 0


def step_exam() -> int:
    fens = exam_fens()
    ids = sorted(fens)
    items = [(rid, fens[rid]) for rid in ids]
    print(f"exam identities: {len(items)}", flush=True)
    path = stream_path("SF-DEEP")
    if path.is_file():
        rows = read_jsonl(path)
        print(f"[cached] SF-DEEP: {len(rows)} labels", flush=True)
    else:
        rows = buy(items, EXAM_BUDGET, workers=12)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    by_id = {row["record_id"]: row for row in rows}
    missing = [rid for rid in ids if rid not in by_id]
    if missing:
        raise RuntimeError(f"SF-DEEP: {len(missing)} exam ids lack a label; refusing")
    service = svc()
    registered = service.register_labels(
        EXAM_NORMALIZATION, family=FAMILY, authority=AUTHORITY, producer=SF_SHA256, pov="stm",
        rows=label_rows([by_id[rid] for rid in ids],
                        note=f"SF-DEEP exam: pinned Stockfish 18, {EXAM_BUDGET} nodes per label"))
    total = sum(by_id[rid]["budget_nodes"] for rid in ids)
    wall = sum(by_id[rid]["wall_ms"] for rid in ids)
    save(exam_labels={"labels": len(ids), "node_budget": EXAM_BUDGET, "realized_nodes": total,
               "mean_realized_nodes": total / len(ids), "mean_wall_ms": round(wall / len(ids), 2),
               "mean_depth": sum(by_id[rid]["value"]["depth"] for rid in ids) / len(ids),
               "registration": {"path": registered.path, "file_hash": registered.file_hash,
                                "rows": registered.rows},
               "normalization": EXAM_NORMALIZATION})
    print(f"SF-DEEP: {len(ids)} labels, mean depth "
          f"{sum(by_id[rid]['value']['depth'] for rid in ids) / len(ids):.1f}", flush=True)
    return 0


def _read_registered(normalization: str, family: str, authority: str, budget_key: str, budget: int,
                     ids: set[str]) -> dict[str, dict]:
    """Stream the corpus's label files; return the rows of one (family, authority, budget)."""
    found: dict[str, dict] = {}
    for path in sorted((LAB / "labstore" / "canonical" / normalization / "labels").glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("family") == family and row.get("authority") == authority
                        and (row.get("budget") or {}).get(budget_key) == budget
                        and row["record_id"] in ids):
                    found[row["record_id"]] = row
    return found


def _cp(value: dict):
    cp = value.get("scoreCpStm")
    if isinstance(cp, str):
        cp = int(cp)
    return None if cp is None or abs(cp) >= MATE_CP_LIMIT else cp


def _describe(cvs: dict[str, dict], sf: dict[str, dict], label: str) -> dict:
    pairs = [(cvs[rid], sf[rid]) for rid in cvs if rid in sf]
    usable = [(a, b) for a, b in pairs if _cp(a["value"]) is not None and _cp(b["value"]) is not None]
    diffs = [abs(_cp(a["value"]) - _cp(b["value"])) for a, b in usable]
    sign_disagree = sum(1 for a, b in usable
                        if (_cp(a["value"]) > 0) != (_cp(b["value"]) > 0)
                        and _cp(a["value"]) != 0 and _cp(b["value"]) != 0)
    moves = sum(1 for a, b in usable if a["value"].get("bestMove") != b["value"].get("bestMove"))
    n = max(len(usable), 1)
    return {"positions": len(pairs), "non_mate_pairs": len(usable),
            "mate_rows_cvs": sum(1 for a, _ in pairs if _cp(a["value"]) is None),
            "mate_rows_sf": sum(1 for _, b in pairs if _cp(b["value"]) is None),
            "mean_abs_cp": sum(diffs) / len(diffs) if diffs else None,
            "median_abs_cp": sorted(diffs)[len(diffs) // 2] if diffs else None,
            "p90_abs_cp": sorted(diffs)[int(0.9 * (len(diffs) - 1))] if diffs else None,
            "sign_disagreement_rate": sign_disagree / n,
            "best_move_disagreement_rate": moves / n,
            "note": f"{label}: descriptive evidence, never a selection mechanism"}


def step_disagreement() -> int:
    ids = training_rows()
    id_set = set(ids)
    cvs4k = _read_registered("N0014", "search_shallow_cp", "legacy.cvs.search.shallow",
                             "nodeBudget", 4000, id_set)
    blocks = {}
    for label, budget in TRAINING_BUDGETS.items():
        sf = _read_registered("N0014", FAMILY, AUTHORITY, "nodes", budget, id_set)
        blocks[f"CVS-4k vs {label}"] = _describe(cvs4k, sf, label)
        print(f"CVS-4k vs {label}: {blocks[f'CVS-4k vs {label}']}", flush=True)
    # SF dose ladder, same positions and same teacher
    sf32 = _read_registered("N0014", FAMILY, AUTHORITY, "nodes", 32_000, id_set)
    sf1m = _read_registered("N0014", FAMILY, AUTHORITY, "nodes", 1_000_000, id_set)
    blocks["SF-32k vs SF-1M"] = _describe(sf32, sf1m, "SF-32k vs SF-1M")

    exam_ids = set(exam_fens())
    cvs_deep = _read_registered(EXAM_NORMALIZATION, "search_deep_cp", "legacy.cvs.search.deep",
                                "nodeBudget", 400_000, exam_ids)
    sf_deep = _read_registered(EXAM_NORMALIZATION, FAMILY, AUTHORITY, "nodes", EXAM_BUDGET, exam_ids)
    blocks["CVS-400k vs SF-DEEP (exam identities)"] = _describe(cvs_deep, sf_deep, "exam ids")

    artifact = {"id": "s13-teacher-disagreement",
                "purpose": "characterize the authorities on the frozen positions BEFORE training; "
                           "no label value was used to select or filter a position",
                "authority_cvs": "legacy.cvs.search.shallow/deep (producer 36e9b159…)",
                "authority_sf": f"oracle.stockfish.18 (producer {SF_SHA256[:16]}…)",
                "matcher": f"|cp| >= {MATE_CP_LIMIT} treated as a mate sentinel and excluded from cp "
                           "statistics; mate rows counted and reported",
                "blocks": blocks,
                "teacher_identity": identity_canonical()}
    digest = write_json(S13 / "s13-disagreement.json", artifact)
    (S13 / "s13-disagreement.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"disagreement artifact {digest}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["training", "exam", "disagreement"])
    args = parser.parse_args()
    return {"training": step_training, "exam": step_exam,
            "disagreement": step_disagreement}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
