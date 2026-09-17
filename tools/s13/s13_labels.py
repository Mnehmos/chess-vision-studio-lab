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
import math
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


def stream_path(label: str, suffix: str = "") -> pathlib.Path:
    return OUT / f"labels-{label}{suffix}.jsonl"


def purchase_training(*, cold: bool, suffix: str, authority: str, state_key: str) -> int:
    """Buy (or reuse) every training stream and register it; identical logic for both contracts.

    `cold` is not cosmetic: the cold streams clear the engine's search state before every label,
    so a label cannot depend on which positions that worker happened to see before it. The warm
    streams (no reset) are the superseded X0009 evidence and stay byte-identical.
    """
    ids = training_rows()
    fens = record_fens(ids)
    items = [(rid, fens[rid]) for rid in ids]
    print(f"training rows: {len(items)} (D0036's prefix of the frozen order); cold={cold}", flush=True)
    service = svc()
    labels: dict = {}
    for label, budget in TRAINING_BUDGETS.items():
        path = stream_path(label, suffix)
        if path.is_file():
            rows = read_jsonl(path)
            print(f"[cached] {label}{suffix}: {len(rows)} labels", flush=True)
        else:
            print(f"buying {label}{suffix} ({budget} nodes, cold={cold}) for {len(items)} rows",
                  flush=True)
            workers = 12                     # the SAME worker count for both arms: with cold labels
            rows = buy(items, budget, workers=workers, cold=cold)   # the partition cannot matter
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        by_id = {row["record_id"]: row for row in rows}
        missing = [rid for rid in ids if rid not in by_id]
        if missing:
            raise RuntimeError(f"{label}: {len(missing)} training rows lack a label; refusing")
        contract = ("cold per label: ucinewgame + Clear Hash + isready before every search" if cold
                    else "WARM: search state reused across labels (superseded)")
        registered = service.register_labels(
            "N0014", family=FAMILY, authority=authority, producer=SF_SHA256, pov="stm",
            rows=label_rows([by_id[rid] for rid in ids],
                            note=f"{label}: pinned Stockfish 18, {budget} nodes per label; {contract}"))
        total = sum(by_id[rid]["budget_nodes"] for rid in ids)
        wall = sum(by_id[rid]["wall_ms"] for rid in ids)
        labels[label] = {"labels": len(ids), "node_budget": budget, "cold_per_label": cold,
                         "authority": authority, "workers": 12,
                         "realized_nodes": total, "mean_realized_nodes": total / len(ids),
                         "sum_wall_ms": round(wall, 1),
                         "mean_wall_ms": round(wall / len(ids), 2),
                         "mean_depth": sum(by_id[rid]["value"]["depth"] for rid in ids) / len(ids),
                         "registration": {"path": registered.path, "file_hash": registered.file_hash,
                                          "rows": registered.rows}}
        print(f"{label}{suffix}: {len(ids)} labels, {total:,} nodes, mean {total / len(ids):.0f}/label, "
              f"mean depth {labels[label]['mean_depth']:.1f}", flush=True)
    save(**{state_key: labels})
    return 0


def step_training() -> int:
    from s13_common import AUTHORITY
    return purchase_training(cold=False, suffix="", authority=AUTHORITY, state_key="labels")


def step_training_cold() -> int:
    from s13_common import AUTHORITY_COLD
    return purchase_training(cold=True, suffix="-cold", authority=AUTHORITY_COLD,
                             state_key="labels_cold")


def step_coldcheck() -> int:
    """The confound's own regression test, on the real engine: the worker partition must not matter.

    The same 64 positions are labelled at 32k nodes with 4 workers and with 12 workers. Under the
    cold contract the two streams must be identical in (cp, best move, nodes) — the warm contract
    is exactly the case where they are not.
    """
    from s13_common import AUTHORITY_COLD
    ids = training_rows()[:64]
    fens = record_fens(ids)
    items = [(rid, fens[rid]) for rid in ids]
    first = buy(items, 32_000, workers=4, log=False)
    second = buy(items, 32_000, workers=12, log=False)
    diffs = [{"record_id": a["record_id"], "cp": [a["value"]["scoreCpStm"], b["value"]["scoreCpStm"]],
              "bestMove": [a["value"]["bestMove"], b["value"]["bestMove"]]}
             for a, b in zip(first, second)
             if (a["value"]["scoreCpStm"], a["value"]["bestMove"]) !=
                (b["value"]["scoreCpStm"], b["value"]["bestMove"])]
    artifact = {"id": "s13-cold-worker-check", "authority": AUTHORITY_COLD,
                "positions": len(items), "budget_nodes": 32_000, "workers_compared": [4, 12],
                "identical": not diffs, "differences": diffs[:10], "difference_count": len(diffs),
                "reading": "cold labels are a function of (engine, options, position, budget) alone; "
                           "the worker partition is not a treatment variable"}
    digest = write_json(S13 / "s13-cold-worker-check.json", artifact)
    (S13 / "s13-cold-worker-check.hash").write_text(digest + "\n", encoding="utf-8")
    save(coldcheck={"identical_across_worker_counts": not diffs, "differences": len(diffs),
                    "artifact": "tools/s13/s13-cold-worker-check.json", "hash": digest})
    print(f"cold worker check: {len(diffs)} differences across 4 vs 12 workers ({digest})", flush=True)
    if diffs:
        print("FAIL: the cold contract did not make the worker partition irrelevant", flush=True)
        return 3
    return 0


def step_exam(*, cold: bool = False, suffix: str = "", authority: str = None,
              state_key: str = "exam_labels") -> int:
    from s13_common import AUTHORITY, AUTHORITY_COLD
    authority = authority or (AUTHORITY_COLD if cold else AUTHORITY)
    fens = exam_fens()
    ids = sorted(fens)
    items = [(rid, fens[rid]) for rid in ids]
    print(f"exam identities: {len(items)}", flush=True)
    path = stream_path("SF-DEEP", suffix)
    if path.is_file():
        rows = read_jsonl(path)
        print(f"[cached] SF-DEEP: {len(rows)} labels", flush=True)
    else:
        rows = buy(items, EXAM_BUDGET, workers=12, cold=cold)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    by_id = {row["record_id"]: row for row in rows}
    missing = [rid for rid in ids if rid not in by_id]
    if missing:
        raise RuntimeError(f"SF-DEEP: {len(missing)} exam ids lack a label; refusing")
    service = svc()
    contract = ("cold per label: ucinewgame + Clear Hash + isready before every search" if cold
                else "WARM: search state reused across labels (superseded)")
    registered = service.register_labels(
        EXAM_NORMALIZATION, family=FAMILY, authority=authority, producer=SF_SHA256, pov="stm",
        rows=label_rows([by_id[rid] for rid in ids],
                        note=f"SF-DEEP exam: pinned Stockfish 18, {EXAM_BUDGET} nodes per label; {contract}"))
    total = sum(by_id[rid]["budget_nodes"] for rid in ids)
    wall = sum(by_id[rid]["wall_ms"] for rid in ids)
    save(**{state_key: {"labels": len(ids), "node_budget": EXAM_BUDGET, "realized_nodes": total,
               "mean_realized_nodes": total / len(ids), "mean_wall_ms": round(wall / len(ids), 2),
               "mean_depth": sum(by_id[rid]["value"]["depth"] for rid in ids) / len(ids),
                    "registration": {"path": registered.path, "file_hash": registered.file_hash,
                                     "rows": registered.rows},
                    "authority": authority, "cold_per_label": cold,
                    "normalization": EXAM_NORMALIZATION}})
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
    """The stm cp of either label shape: flat (search_shallow/oracle_cp) or the legacy
    `targets` wrapper (the imported 400k deep labels). A string "None" means the row carries no
    finite cp; a mate sentinel is excluded from cp statistics but counted separately."""
    cp = value.get("scoreCpStm")
    if cp is None and isinstance(value.get("targets"), dict):
        cp = value["targets"].get("scoreCpStm")
    if isinstance(cp, str):
        try:
            cp = int(cp)
        except ValueError:
            return None
    return None if cp is None or abs(cp) >= MATE_CP_LIMIT else cp


def _move(value: dict):
    """The best move of either shape; None where the label never recorded one."""
    move = value.get("bestMove")
    if move is None and isinstance(value.get("targets"), dict):
        move = value["targets"].get("bestMove")
    return move if isinstance(move, str) and move else None


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank); no scipy, no external dependency."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 3:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def _correlations(pairs: list[tuple[dict, dict]]) -> dict:
    """CP and expected-score correlation between two authorities on the same positions.

    The expected-score transform is the Lab's OWN supervision contract (`sigmoid(cp / K)`,
    K = 256, `nnue.targets`), so this is the correlation of the quantities a student is trained to
    predict, not an invented conversion.
    """
    cps_a, cps_b = [], []
    for a, b in pairs:
        ca, cb = _cp(a["value"]), _cp(b["value"])
        if ca is None or cb is None:
            continue
        cps_a.append(float(ca))
        cps_b.append(float(cb))
    if len(cps_a) < 3:
        return {"pairs": len(cps_a), "pearson_cp": None, "spearman_cp": None,
                "pearson_expected_score": None}
    k = 256.0
    expected_a = [1.0 / (1.0 + math.exp(-cp / k)) for cp in cps_a]
    expected_b = [1.0 / (1.0 + math.exp(-cp / k)) for cp in cps_b]
    return {"pairs": len(cps_a), "pearson_cp": _pearson(cps_a, cps_b),
            "spearman_cp": _pearson(_ranks(cps_a), _ranks(cps_b)),
            "pearson_expected_score": _pearson(expected_a, expected_b),
            "expected_score_transform": "sigmoid(cp / K), K = 256 (the Lab's supervision target)"}


def _describe(cvs: dict[str, dict], sf: dict[str, dict], label: str) -> dict:
    pairs = [(cvs[rid], sf[rid]) for rid in cvs if rid in sf]
    usable = [(a, b) for a, b in pairs if _cp(a["value"]) is not None and _cp(b["value"]) is not None]
    diffs = [abs(_cp(a["value"]) - _cp(b["value"])) for a, b in usable]
    sign_disagree = sum(1 for a, b in usable
                        if (_cp(a["value"]) > 0) != (_cp(b["value"]) > 0)
                        and _cp(a["value"]) != 0 and _cp(b["value"]) != 0)
    comparable_moves = [(a, b) for a, b in usable if _move(a["value"]) and _move(b["value"])]
    moves = sum(1 for a, b in comparable_moves if _move(a["value"]) != _move(b["value"]))
    n = max(len(usable), 1)
    return {"positions": len(pairs), "non_mate_pairs": len(usable),
            "mate_rows_cvs": sum(1 for a, _ in pairs if _cp(a["value"]) is None),
            "mate_rows_sf": sum(1 for _, b in pairs if _cp(b["value"]) is None),
            "mean_abs_cp": sum(diffs) / len(diffs) if diffs else None,
            "median_abs_cp": sorted(diffs)[len(diffs) // 2] if diffs else None,
            "p90_abs_cp": sorted(diffs)[int(0.9 * (len(diffs) - 1))] if diffs else None,
            "sign_disagreement_rate": sign_disagree / n,
            "correlations": _correlations(usable),
            "best_move_pairs": len(comparable_moves),
            "best_move_disagreement_rate": (moves / len(comparable_moves)) if comparable_moves else None,
            "note": f"{label}: descriptive evidence, never a selection mechanism"}


def step_disagreement() -> int:
    """Characterize the authorities on the frozen positions; no label value selects a position.

    Both contracts are reported: the warm streams X0009 used (superseded) and the cold streams the
    replacement experiment uses. Correlation is computed on non-mate paired rows in cp and in the
    Lab's own expected-score transform (sigmoid(cp/K), K = 256).
    """
    from s13_common import AUTHORITY_COLD
    ids = training_rows()
    id_set = set(ids)
    cvs4k = _read_registered("N0014", "search_shallow_cp", "legacy.cvs.search.shallow",
                             "nodeBudget", 4000, id_set)
    blocks = {}
    for contract, authority in (("cold", AUTHORITY_COLD), ("warm (superseded)", AUTHORITY)):
        for label, budget in TRAINING_BUDGETS.items():
            sf = _read_registered("N0014", FAMILY, authority, "nodes", budget, id_set)
            if not sf:
                continue
            key = f"CVS-4k vs {label} [{contract}]"
            blocks[key] = _describe(cvs4k, sf, label)
            print(f"{key}: {blocks[key]}", flush=True)
        sf32 = _read_registered("N0014", FAMILY, authority, "nodes", 32_000, id_set)
        sf1m = _read_registered("N0014", FAMILY, authority, "nodes", 1_000_000, id_set)
        if sf32 and sf1m:
            blocks[f"SF-32k vs SF-1M [{contract}]"] = _describe(sf32, sf1m, "SF-32k vs SF-1M")

    exam_ids = set(exam_fens())
    cvs_deep = _read_registered(EXAM_NORMALIZATION, "search_deep_cp", "legacy.cvs.search.deep",
                                "nodeBudget", 400_000, exam_ids)
    for contract, authority in (("cold", AUTHORITY_COLD), ("warm (superseded)", AUTHORITY)):
        sf_deep = _read_registered(EXAM_NORMALIZATION, FAMILY, authority, "nodes", EXAM_BUDGET, exam_ids)
        if sf_deep:
            blocks[f"CVS-400k vs SF-DEEP (exam ids) [{contract}]"] = _describe(
                cvs_deep, sf_deep, "exam ids")

    artifact = {"id": "s13-teacher-disagreement",
                "purpose": "characterize the authorities on the frozen positions BEFORE training; "
                           "no label value was used to select or filter a position",
                "authority_cvs": "legacy.cvs.search.shallow/deep (producer 36e9b159…)",
                "authorities_sf": {"cold": AUTHORITY_COLD, "warm": AUTHORITY},
                "matcher": f"|cp| >= {MATE_CP_LIMIT} treated as a mate sentinel and excluded from cp "
                           "statistics; mate rows counted and reported",
                "correlation_note": "Pearson and Spearman on paired non-mate cp values, plus Pearson "
                                    "on the Lab's expected-score transform sigmoid(cp/K), K = 256 — "
                                    "the quantity a student is trained to predict",
                "blocks": blocks,
                "teacher_identity": identity_canonical()}
    digest = write_json(S13 / "s13-disagreement.json", artifact)
    (S13 / "s13-disagreement.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"disagreement artifact {digest}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["training", "training-cold", "coldcheck", "exam",
                                         "exam-cold", "disagreement"])
    args = parser.parse_args()
    return {"training": step_training, "training-cold": step_training_cold,
            "coldcheck": step_coldcheck, "exam": step_exam,
            "exam-cold": lambda: step_exam(cold=True, suffix="-cold",
                                           authority=__import__("s13_common").AUTHORITY_COLD,
                                           state_key="exam_labels_cold"),
            "disagreement": step_disagreement}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
