"""S7-S pre-freeze yield probe: measure what the EPD start-position source actually yields.

Bounded, deterministic, read-only: generate `--games` games from distinct frozen starts in
per-game mode (v3) and report raw rows, unique candidates, and the marginal unique yield per
batch. The study may only be frozen if this shows the pool can reach the required size.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.pool import DEFAULT_ENGINE_ARGS, AnalyzeTeacher, PoolConfig, SelfPlayGenerator
from cvslab.funnel.providers import AnalyzeTransport

TOOLS = pathlib.Path(__file__).resolve().parent
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
EPD = r"F:\Github\chess-vision-studio-rust-engine\benchmarks\suites\openings-inv1-20260910.epd"
EPD_SHA = "sha256:4d96e552fd257e66fe39d78420e1e48cfed6753795dd0a3e20cd222a3757e6b3"
SEED = 20260920
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s7s\probe")


def config(count: int) -> PoolConfig:
    return PoolConfig(seed=SEED, games=count, max_plies=70, sample_every=2, min_ply=6,
                      diversification_plies=(8, 10, 12), diversification_candidates=3,
                      diversification_window_cp=25, play_node_budget=20000,
                      diversification_node_budget=2000, rng_mode="per-game",
                      start_source="epd-file", epd_path=EPD, epd_sha256=EPD_SHA)


def rid(fen: str) -> str:
    import chess
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def worker(start: int, count: int) -> dict:
    transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
    try:
        generated = SelfPlayGenerator(config(count), AnalyzeTeacher(transport)).generate(
            start_index=start, count=count)
    finally:
        transport.close()
    path = OUT / f"probe-{start:06d}.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in generated["positions"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"start": start, "count": count, "rows": len(generated["positions"]), "file": path.name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("probe-*.jsonl"):
        stale.unlink()

    per_worker = args.games // args.workers
    starts = [index * per_worker for index in range(args.workers)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda start: worker(start, per_worker), starts))

    seen: set[str] = set()
    batches = []
    for result in sorted(results, key=lambda r: r["start"]):
        ids = set()
        with open(OUT / result["file"], "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(rid(json.loads(line)["fen"]))
        before = len(seen)
        seen |= ids
        batches.append({"games": result["count"], "rows": result["rows"], "unique_in_batch": len(ids),
                        "new_unique": len(seen) - before, "cumulative_unique": len(seen)})
    report = {
        "start_source": "epd-file", "epd_path": EPD, "epd_sha256": EPD_SHA,
        "generator_version": 3, "seed": SEED, "games": args.games,
        "rows": sum(result["rows"] for result in results),
        "unique": len(seen),
        "unique_per_game": round(len(seen) / args.games, 2),
        "batches": batches,
        "note": "bounded pre-freeze probe; distinct EPD starts, per-game RNG, play unchanged",
    }
    (TOOLS / "s7-scaling-epd-probe.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("games", "rows", "unique", "unique_per_game")}, indent=1))
    print("first batch new_unique:", batches[0]["new_unique"], "| last batch new_unique:", batches[-1]["new_unique"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
