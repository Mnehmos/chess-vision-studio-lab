"""S7-S scale study: deterministic parallel pool generation, one worker per game batch.

Game k in per-game mode is a pure function of (config, k, teacher), so disjoint index ranges
can be generated concurrently and concatenated in index order — byte-identical to a serial
run of the same config (verified by tests/test_funnel_pool.py and the driver's own check).

    python tools/s7/s7_pool_worker.py --start 0 --count 400 --out <dir>

Writes positions.jsonl, games.jsonl and batch.json for its range. No dedup, no snapshotting:
the driver merges batches and the freeze step records the S#### afterwards.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, AnalyzeTeacher, PoolConfig, SelfPlayGenerator)
from cvslab.funnel.providers import AnalyzeTransport
from cvslab.hashing import sha256_file

ENGINE = r"F:\Github\chess-vision-studio-rust-engine"


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = PoolConfig(seed=args.seed, games=args.count, max_plies=70, sample_every=2, min_ply=6,
                        diversification_plies=(8, 10, 12), diversification_candidates=3,
                        diversification_window_cp=25, play_node_budget=20000,
                        diversification_node_budget=2000, rng_mode="per-game")
    started = time.time()
    transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
    try:
        generated = SelfPlayGenerator(config, AnalyzeTeacher(transport)).generate(
            start_index=args.start, count=args.count)
    finally:
        transport.close()

    write_jsonl(out / "positions.jsonl", generated["positions"])
    write_jsonl(out / "games.jsonl", [{k: v for k, v in game.items() if k != "positions"}
                                      for game in generated["games"]])
    (out / "batch.json").write_text(json.dumps({
        "start": args.start, "count": args.count, "seed": args.seed, "rng_mode": config.rng_mode,
        "generatorVersion": config.generator_version,
        "positionsSha256": sha256_file(out / "positions.jsonl"),
        "gamesSha256": sha256_file(out / "games.jsonl"),
        "rows": len(generated["positions"]), "openingSourceSha256": config.canonical()["openingSourceSha256"],
        "seconds": round(time.time() - started, 2), "report": generated["report"],
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"start": args.start, "count": args.count, "rows": len(generated["positions"]),
                      "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
