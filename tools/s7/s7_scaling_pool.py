"""S7-S step 1: generate the candidate pool from the frozen EPD starts (generator v3).

Deterministic contiguous batches, parallel workers, resumable. Writes the merged pool
(positions.jsonl, games.jsonl, pool-manifest.json with the real engine identity) so the
freeze step can snapshot it as an S#### source.

    python tools/s7/s7_scaling_pool.py --games 2600 --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, AnalyzeTeacher, PoolConfig, SelfPlayGenerator,
                                engine_identity, write_pool)
from cvslab.funnel.providers import AnalyzeTransport

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s7scaling")
POOL = OUT / "pool"
BATCHES = OUT / "batches"
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
EPD = r"F:\Github\chess-vision-studio-rust-engine\benchmarks\suites\openings-inv1-20260910.epd"
EPD_SHA = "sha256:4d96e552fd257e66fe39d78420e1e48cfed6753795dd0a3e20cd222a3757e6b3"
SEED = 20260920
BATCH = 100


def config(count: int) -> PoolConfig:
    return PoolConfig(seed=SEED, games=count, max_plies=70, sample_every=2, min_ply=6,
                      diversification_plies=(8, 10, 12), diversification_candidates=3,
                      diversification_window_cp=25, play_node_budget=20000,
                      diversification_node_budget=2000, rng_mode="per-game",
                      start_source="epd-file", epd_path=EPD, epd_sha256=EPD_SHA)


def worker(start: int, count: int) -> pathlib.Path:
    directory = BATCHES / f"b{start:06d}"
    if (directory / "done.json").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
    try:
        generated = SelfPlayGenerator(config(count), AnalyzeTeacher(transport)).generate(
            start_index=start, count=count)
    finally:
        transport.close()
    with open(directory / "positions.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in generated["positions"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(directory / "games.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for game in generated["games"]:
            handle.write(json.dumps({k: v for k, v in game.items() if k != "positions"}, sort_keys=True) + "\n")
    (directory / "done.json").write_text(json.dumps({"start": start, "count": count,
                                                     "rows": len(generated["positions"])}) + "\n",
                                         encoding="utf-8")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    POOL.mkdir(parents=True, exist_ok=True)
    BATCHES.mkdir(parents=True, exist_ok=True)

    starts = list(range(0, args.games, BATCH))
    pending = [start for start in starts if not (BATCHES / f"b{start:06d}" / "done.json").is_file()]
    print(f"batches: {len(starts)} total, {len(pending)} to generate", flush=True)
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, _ in enumerate(pool.map(lambda start: worker(start, BATCH), pending), start=1):
                if index % 8 == 0 or index == len(pending):
                    print(f"[{time.strftime('%H:%M:%S')}] {index}/{len(pending)} batches done", flush=True)

    positions: list[dict] = []
    games: list[dict] = []
    for start in starts:
        directory = BATCHES / f"b{start:06d}"
        with open(directory / "positions.jsonl", "r", encoding="utf-8") as handle:
            positions.extend(json.loads(line) for line in handle if line.strip())
        with open(directory / "games.jsonl", "r", encoding="utf-8") as handle:
            games.extend(json.loads(line) for line in handle if line.strip())

    identity = engine_identity(ENGINE, args=DEFAULT_ENGINE_ARGS)
    manifest = write_pool(POOL, {"games": games, "positions": positions,
                                 "report": {"games": len(games), "rawPositions": len(positions)}},
                          {"config": config(len(starts) * BATCH).canonical(), "engine": identity},
                          opening_lines=None)
    print(json.dumps({"games": len(games), "rawRows": len(positions),
                      "positionsSha256": manifest["positionsFile"]["sha256"],
                      "generatorVersion": manifest["generatorVersion"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
