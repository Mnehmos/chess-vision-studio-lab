"""Reproducible inventory of a legacy JSONL corpus (engine repo) — row counts, unique
positions by FEN and by EPD, duplicate rows, and per-row field statistics.

Why this exists: the S7 design document cites unique-position counts for Gen10 corpora, and
those counts differed between the engine's committed documentation and a direct scan. A
number that decides how a contrast is described must be reproducible from a committed tool,
not asserted. This script is that tool; its JSON outputs are committed next to it.

    python tools/intake/gen10_corpus_inventory.py \
        --shards F:/Github/chess-vision-studio-rust-engine/training/gen10/corpus-d20/shards \
        --out tools/intake/gen10-corpus-d20-inventory.json --fields label_depth,features

Read-only: it never writes into the corpus directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def epd_of(fen: str) -> str:
    import chess
    return chess.Board(fen).epd()


def inventory(paths: list[Path], *, fields: list[str], epd: bool, max_epd_rows: int) -> dict:
    rows = 0
    fen_seen: set[str] = set()
    fen_noclocks_seen: set[str] = set()
    epd_seen: set[str] = set()
    dup_rows_by_fen = 0
    dup_rows_by_fen_noclocks = 0
    dup_rows_by_epd = 0
    fen_counts: Counter = Counter()
    field_stats: dict[str, Counter] = {name: Counter() for name in fields}
    label_nodes_max = 0
    per_file: list[dict] = []
    epd_done = 0
    epd_skipped = False

    for path in paths:
        file_rows = 0
        file_fen: set[str] = set()
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows += 1
                file_rows += 1
                fen = row.get("fen")
                if fen is None:
                    continue
                if fen in fen_seen:
                    dup_rows_by_fen += 1
                else:
                    fen_seen.add(fen)
                noclocks = " ".join(fen.split()[:4])
                if noclocks in fen_noclocks_seen:
                    dup_rows_by_fen_noclocks += 1
                else:
                    fen_noclocks_seen.add(noclocks)
                file_fen.add(fen)
                fen_counts[fen] += 1
                if epd and not epd_skipped:
                    if epd_done >= max_epd_rows:
                        epd_skipped = True
                    else:
                        key = epd_of(fen)
                        epd_done += 1
                        if key in epd_seen:
                            dup_rows_by_epd += 1
                        else:
                            epd_seen.add(key)
                for name in fields:
                    value = row.get(name)
                    if isinstance(value, list):
                        field_stats[name][f"rows_with_len_{len(value)}"] += 1 if not value else 0
                        field_stats[name]["rows_with_empty_list"] += 1 if not value else 0
                        field_stats[name]["rows_nonempty"] += 1 if value else 0
                    else:
                        field_stats[name][str(value)] += 1
                        if name == "label_nodes" and isinstance(value, int):
                            label_nodes_max = max(label_nodes_max, value)
        per_file.append({"file": path.name, "rows": file_rows, "uniqueFen": len(file_fen)})

    out = {
        "files": len(paths),
        "rows": rows,
        "uniqueFen": len(fen_seen),
        "duplicateRowsByFen": dup_rows_by_fen,
        "uniqueFenNoClocks": len(fen_noclocks_seen),
        "duplicateRowsByFenNoClocks": dup_rows_by_fen_noclocks,
        "maxFenMultiplicity": max(fen_counts.values()) if fen_counts else 0,
        "uniqueEpd": (len(epd_seen) if epd and not epd_skipped else None),
        "duplicateRowsByEpd": (dup_rows_by_epd if epd and not epd_skipped else None),
        "epdRowsExamined": epd_done,
        "epdSkippedDueToCap": epd_skipped,
        "fieldStats": {name: dict(sorted(counter.items(), key=lambda kv: -kv[1])[:12])
                       for name, counter in field_stats.items() if counter},
        "perFile": per_file,
    }
    if "label_nodes" in fields:
        out["labelNodesMax"] = label_nodes_max
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, help="directory of *.jsonl shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fields", default="", help="comma-separated extra fields to profile")
    parser.add_argument("--epd", action="store_true", help="also count unique EPDs (python-chess)")
    parser.add_argument("--max-epd-rows", type=int, default=5_000_000)
    args = parser.parse_args()

    shard_dir = Path(args.shards)
    paths = sorted(shard_dir.glob("*.jsonl"))
    if not paths:
        print(f"no *.jsonl under {shard_dir}")
        return 2
    result = inventory(paths, fields=[f for f in args.fields.split(",") if f],
                       epd=args.epd, max_epd_rows=args.max_epd_rows)
    result["source"] = str(shard_dir).replace("\\", "/")
    result["shardFileHashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in paths}
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rows", "uniqueFen", "duplicateRowsByFen",
                                             "uniqueEpd", "duplicateRowsByEpd")}, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
