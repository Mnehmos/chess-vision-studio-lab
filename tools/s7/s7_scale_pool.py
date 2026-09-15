"""S7-S: generate the scaling study's candidate pool in parallel deterministic batches.

Design constraints this driver must honour (tools/s7/s7-scaling-preregistration.json):

  * ONE generator contract for the whole pool: cvslab.pool.selfplay, per-game RNG mode
    (generatorVersion 2), seed 20260920, the 12 built-in opening lines, max_plies 70,
    sample_every 2, min_ply 6, play 20k / diversification 2k;
  * batches are contiguous index ranges, merged in index order, byte-identical to a serial
    run of the same config (checked here by regenerating a two-batch prefix serially);
  * generate until there are at least `--target` unique EPD candidates, which must also be
    at least 1.15x the calibration-predicted size of the largest arm (checked at freeze);
  * never truncate the experiment: if the target is not reached, keep generating.

    python tools/s7/s7_scale_pool.py --workers 8 --target 60000 [--batch 400] [--cap 12000]
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

import chess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, AnalyzeTeacher, PoolConfig, SelfPlayGenerator)
from cvslab.funnel.providers import AnalyzeTransport
from cvslab.hashing import sha256_file

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
POOL = pathlib.Path(r"F:\Github\_parity_tmp\s7s\pool")
STATE = TOOLS / "s7-scaling-pool-state.json"
SEED = 20260920
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
WORKER = TOOLS / "s7_pool_worker.py"


def record_id(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else \
        {"seed": SEED, "batches": [], "rawRows": 0, "uniqueCandidates": 0}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")


UNIQUE = POOL / "unique.txt"


def batch_ids(batch: dict):
    path = POOL / f"batch-{batch['start']:07d}" / "positions.jsonl"
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield record_id(json.loads(line)["fen"])


def load_unique(state: dict) -> set[str]:
    """The union of every recorded batch, from unique.txt when it is current."""
    if UNIQUE.is_file() and state.get("uniqueFileBatches") == len(state["batches"]):
        return {line.strip() for line in UNIQUE.read_text(encoding="utf-8").splitlines() if line.strip()}
    seen: set[str] = set()
    for batch in sorted(state["batches"], key=lambda b: b["start"]):
        seen.update(batch_ids(batch))
    UNIQUE.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")
    return seen


def run_batch(start: int, count: int) -> dict:
    out = POOL / f"batch-{start:07d}"
    try:
        subprocess.run([sys.executable, str(WORKER), "--start", str(start), "--count", str(count),
                        "--out", str(out), "--seed", str(SEED)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-600:]
        print(f"worker for games [{start}, {start + count}) failed:\n{tail}", flush=True)
        raise
    meta = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    return {"start": start, "count": count, "rows": meta["rows"], "seconds": meta["seconds"],
            "positionsSha256": meta["positionsSha256"], "gamesSha256": meta["gamesSha256"]}


def verify_parallel_equivalence(prefix_batches: list[dict]) -> bool:
    """Regenerate the same game range SERIALLY and compare positions byte-for-byte."""
    count = sum(b["count"] for b in prefix_batches)
    config = PoolConfig(seed=SEED, games=count, max_plies=70, sample_every=2, min_ply=6,
                        diversification_plies=(8, 10, 12), diversification_candidates=3,
                        diversification_window_cp=25, play_node_budget=20000,
                        diversification_node_budget=2000, rng_mode="per-game")
    transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
    try:
        serial = SelfPlayGenerator(config, AnalyzeTeacher(transport)).generate(count=count)
    finally:
        transport.close()
    rows = []
    for batch in sorted(prefix_batches, key=lambda b: b["start"]):
        with open(POOL / f"batch-{batch['start']:07d}" / "positions.jsonl", "r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows == serial["positions"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=400)
    parser.add_argument("--cap", type=int, default=12000)
    parser.add_argument("--verify-first", action="store_true",
                        help="regenerate the first round serially and compare (one extra round of compute)")
    args = parser.parse_args()

    POOL.mkdir(parents=True, exist_ok=True)
    state = load_state()
    done_starts = {b["start"] for b in state["batches"]}
    unique = load_unique(state) if state["batches"] else set()
    print(f"resuming: {len(state['batches'])} batches, {len(unique)} unique candidates", flush=True)

    while len(unique) < args.target and state["rawRows"] < args.cap * 33:
        # batches tile [0, total_games) contiguously; resume from the tile edge, never from
        # a max-start guess (that would skip games when the batch size changes between runs)
        total_games = sum(batch["count"] for batch in state["batches"])
        if done_starts and total_games != max(done_starts) + max(
                b["count"] for b in state["batches"] if b["start"] == max(done_starts)):
            print("STOP: batches do not tile contiguously; refusing to guess the next range")
            return 3
        round_start = total_games
        starts = [round_start + index * args.batch for index in range(args.workers)]
        if starts[-1] + args.batch > args.cap:
            starts = [start for start in starts if start + args.batch <= args.cap]
            if not starts:
                print(f"cap {args.cap} games reached with {len(unique)} unique candidates", flush=True)
                break
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda start: run_batch(start, args.batch), starts))
        for result in results:
            state["batches"].append(result)
            done_starts.add(result["start"])
            unique.update(batch_ids(result))          # only the NEW batch is scanned
        state["rawRows"] += sum(r["rows"] for r in results)
        state["uniqueCandidates"] = len(unique)
        UNIQUE.write_text("\n".join(sorted(unique)) + "\n", encoding="utf-8")
        state["uniqueFileBatches"] = len(state["batches"])
        save_state(state)
        print(f"[{time.strftime('%H:%M:%S')}] round {len(results)} batches in {time.time() - t0:.0f}s: "
              f"raw {state['rawRows']} rows, unique {len(unique)} ({len(unique) / args.target:.0%} of target)",
              flush=True)

        if args.verify_first and round_start == 0:
            ok = verify_parallel_equivalence(results)
            state["parallelEquivalenceChecked"] = ok
            save_state(state)
            print(f"parallel/serial equivalence on the first round: {ok}", flush=True)
            if not ok:
                print("STOP: parallel batches are not identical to a serial run")
                return 3

    state["uniqueCandidates"] = len(unique)
    save_state(state)
    print(f"done: {len(state['batches'])} batches, raw {state['rawRows']} rows, "
          f"{len(unique)} unique candidates", flush=True)
    return 0 if len(unique) >= args.target else 3


if __name__ == "__main__":
    raise SystemExit(main())
