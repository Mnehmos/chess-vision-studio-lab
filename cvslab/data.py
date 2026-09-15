"""Data lineage: S#### sources -> N#### normalization -> canonical records -> D#### frozen datasets."""
from __future__ import annotations

import random
import re
import shutil
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import chess

from .hashing import hash_obj, read_jsonl, sha256_bytes, sha256_file, write_jsonl
from .schemas import (
    Compute,
    ConfigValue,
    Dataset,
    DatasetSplitChange,
    IntegrityCheck,
    LabelSetRef,
    MigrationDiff,
    Normalization,
    SourceSnapshot,
    SplitManifest,
    StackArm,
    StackPolicy,
    make_check,
    utc_now,
)
from .store import LabError, Store, TamperError

FIXTURE_GENERATOR = "cvslab.fixture.random_play"
FIXTURE_GENERATOR_VERSION = 1
IMPORTER = "cvslab.import.jsonl"
IMPORTER_VERSION = 1
PGN_IMPORTER = "cvslab.import.pgn"
PGN_IMPORTER_VERSION = 1
MATERIAL_AUTHORITY = "deterministic.material.v1"
NORMALIZER = "cvslab.normalize"
NORMALIZER_VERSION = 1
CANONICAL_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1
DEDUP_POLICIES = ("exact-epd-keep-first", "none")
SELECTABLE_FIELDS = ("phase", "stm", "source_id")
SPLIT_NAMES = ("train", "val", "test")

# L2 label registry: a family must be registered here before any producer may emit it.
# tier groups families the way the GUI and supervision inspector reason about them.
LABEL_FAMILIES: dict[str, dict] = {
    "eval_cp": {"tier": "deterministic", "description": "scalar evaluation in centipawns, white POV"},
    "facts": {"tier": "deterministic", "description": "deterministic CVS board facts"},
    "motif": {"tier": "deterministic", "description": "tactical/structural motif presence"},
    "strategy": {"tier": "deterministic", "description": "strategy/semantic family labels"},
    "see_cp": {"tier": "deterministic", "description": "bounded tactical proof / SEE value"},
    "move_best": {"tier": "transition", "description": "best move / transition label"},
    "search_shallow_cp": {"tier": "search", "description": "shallow CVS search evaluation"},
    "search_deep_cp": {"tier": "search", "description": "deep CVS search evaluation"},
    "outcome": {"tier": "outcome", "description": "game outcome from the side to move"},
    "oracle_cp": {"tier": "oracle", "description": "external oracle (e.g. Stockfish) evaluation"},
    "human_cp": {"tier": "human", "description": "human annotation"},
    "priority": {"tier": "derived", "description": "information-gain labeling priority"},
}
LABEL_SCALAR_KEYS = ("value",)  # must be JSON scalars inside label rows
LABEL_ALLOWED_KEYS = frozenset({"record_id", "family", "value", "budget", "confidence", "note"})
SCALAR = (bool, int, float, str)

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}


class _Meter:
    def __init__(self):
        self.cpu, self.wall = time.process_time(), time.perf_counter()

    def compute(self, **fields) -> Compute:
        return Compute(cpu_seconds=round(time.process_time() - self.cpu, 4),
                       wall_seconds=round(time.perf_counter() - self.wall, 4), **fields)


def material_cp_white(board: chess.Board) -> int:
    return sum(value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
               for piece, value in PIECE_VALUES.items())


def game_phase(board: chess.Board, opening_min: int = 5600, middlegame_min: int = 2600) -> str:
    non_pawn = sum(value * len(board.pieces(piece, color))
                   for piece, value in PIECE_VALUES.items() if piece != chess.PAWN
                   for color in (chess.WHITE, chess.BLACK))
    return "opening" if non_pawn >= opening_min else "middlegame" if non_pawn >= middlegame_min else "endgame"


def material_signature(board: chess.Board) -> str:
    def side(color: chess.Color) -> str:
        return "".join(f"{chess.piece_symbol(p).upper()}{len(board.pieces(p, color))}"
                       for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN))
    return f"{side(chess.WHITE)}/{side(chess.BLACK).lower()}"


def _ply_from_fen(fen: str) -> int:
    fields = fen.split()
    fullmove = int(fields[5]) if len(fields) > 5 else 1
    return 2 * (fullmove - 1) + (1 if len(fields) > 1 and fields[1] == "b" else 0)


def infer_games(rows: Sequence[dict]) -> list[str]:
    """Game identity: explicit `game` field, else a new game whenever the ply counter does not advance."""
    games, game, previous = [], 0, None
    for row in rows:
        if "game" in row:
            games.append(str(row["game"]))
            continue
        ply = _ply_from_fen(row.get("fen", ""))
        if previous is not None and ply <= previous:
            game += 1
        previous = ply
        games.append(f"g{game:06d}")
    return games


# ---------------------------------------------------------------------------
# L0 sources
# ---------------------------------------------------------------------------


def create_fixture_source(store: Store, *, name: str = "fixture-random-play", n_games: int = 300, seed: int = 20260914,
                          max_plies: int = 80, sample_every: int = 2, min_ply: int = 6) -> SourceSnapshot:
    """Deterministic, license-free tiny corpus: seeded random legal games labelled with material balance."""
    if n_games < 1 or max_plies < 1 or sample_every < 1:
        raise LabError("fixture needs n_games, max_plies and sample_every >= 1")
    meter = _Meter()
    rng = random.Random(seed)
    rows = []
    for game in range(n_games):
        board = chess.Board()
        for ply in range(1, max_plies + 1):
            moves = sorted(board.legal_moves, key=lambda move: move.uci())
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
            if ply >= min_ply and ply % sample_every == 0:
                rows.append({"game": f"g{game:05d}", "ply": ply, "fen": board.fen(), "cp": material_cp_white(board),
                             "res": None, "label_authority": MATERIAL_AUTHORITY})
    source_id = store.next_id("S")
    rel = f"sources/{source_id}/positions.jsonl"
    content_hash = write_jsonl(store.abs(rel), rows)
    store.make_readonly(store.abs(rel))
    return store.create(SourceSnapshot(
        id=source_id, name=name, source_origin=f"generated by {FIXTURE_GENERATOR} v{FIXTURE_GENERATOR_VERSION}",
        locality="local",
        generator={"name": FIXTURE_GENERATOR, "version": FIXTURE_GENERATOR_VERSION, "seed": seed, "n_games": n_games,
                   "max_plies": max_plies, "sample_every": sample_every, "min_ply": min_ply},
        license="generated by the lab; contains no third-party content",
        importer=FIXTURE_GENERATOR, importer_version=FIXTURE_GENERATOR_VERSION, path=rel, content_hash=content_hash,
        row_count=len(rows), game_count=len({row["game"] for row in rows}), label_authorities=[MATERIAL_AUTHORITY],
        compute=meter.compute(labels_generated={MATERIAL_AUTHORITY: len(rows)}, accepted_examples=len(rows)),
        created_at=utc_now(),
    ))


def import_jsonl_source(store: Store, path: str | Path, *, name: str, license: str = "unspecified",
                        label_authority: str = "imported.unspecified", origin: Optional[str] = None) -> SourceSnapshot:
    """Snapshot a `{fen, cp, res}` JSONL corpus byte-for-byte into the store (L0 is never rewritten)."""
    src = Path(path)
    if not src.is_file():
        raise LabError(f"source file not found: {src}")
    meter = _Meter()
    source_id = store.next_id("S")
    rel = f"sources/{source_id}/{src.name}"
    dest = store.abs(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    store.make_readonly(dest)
    rows = read_jsonl(dest)
    return store.create(SourceSnapshot(
        id=source_id, name=name, source_origin=origin or str(src.resolve()), locality="local", generator=None, license=license,
        importer=IMPORTER, importer_version=IMPORTER_VERSION, path=rel, content_hash=sha256_file(dest),
        row_count=len(rows), game_count=len(set(infer_games(rows))), label_authorities=[label_authority],
        compute=meter.compute(accepted_examples=len(rows)), created_at=utc_now(),
    ))


def import_pgn_source(store: Store, path: str | Path, *, name: str, license: str = "unspecified",
                      sample_every: int = 2, min_ply: int = 6,
                      origin: Optional[str] = None) -> SourceSnapshot:
    """Second heterogeneous importer: PGN game files -> the same `{fen, cp, res}` row contract.

    Positions are sampled from the mainline; eval labels are deterministic material balance and
    `res` carries the game result (white POV) so outcome supervision can accumulate too.
    """
    import chess.pgn

    src = Path(path)
    if not src.is_file():
        raise LabError(f"source file not found: {src}")
    if sample_every < 1:
        raise LabError("sample_every must be >= 1")
    meter = _Meter()
    rows: list[dict] = []
    game_index = 0
    with open(src, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            game_index += 1
            result = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(game.headers.get("Result", "*"))
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if board.is_game_over():
                    break
                if ply >= min_ply and ply % sample_every == 0:
                    rows.append({"game": f"g{game_index:05d}", "ply": ply, "fen": board.fen(),
                                 "cp": material_cp_white(board), "res": result,
                                 "label_authority": MATERIAL_AUTHORITY})
    source_id = store.next_id("S")
    rel = f"sources/{source_id}/{src.name}"
    content_hash = write_jsonl(store.abs(rel), rows)
    store.make_readonly(store.abs(rel))
    return store.create(SourceSnapshot(
        id=source_id, name=name, source_origin=origin or str(src.resolve()), locality="local", generator=None,
        license=license, importer=PGN_IMPORTER, importer_version=PGN_IMPORTER_VERSION, path=rel,
        content_hash=content_hash, row_count=len(rows), game_count=game_index,
        label_authorities=[MATERIAL_AUTHORITY],
        compute=meter.compute(labels_generated={MATERIAL_AUTHORITY: len(rows)}, accepted_examples=len(rows)),
        created_at=utc_now(),
    ))


# ---------------------------------------------------------------------------
# L1 canonical records
# ---------------------------------------------------------------------------


def normalize(store: Store, source_ids: Sequence[str], *, name: str, dedup: str = "exact-epd-keep-first",
              settings_overrides: Optional[Mapping[str, ConfigValue]] = None) -> Normalization:
    if dedup not in DEDUP_POLICIES:
        raise LabError(f"unknown dedup policy {dedup!r}; choose one of {DEDUP_POLICIES}")
    if not source_ids:
        raise LabError("normalization needs at least one source")
    meter = _Meter()
    sources: list[SourceSnapshot] = [store.get_as(sid, SourceSnapshot) for sid in source_ids]
    for src in sources:
        if sha256_file(store.abs(src.path)) != src.content_hash:
            raise TamperError(f"{src.id}: source bytes no longer match the snapshot hash")
    settings: dict[str, ConfigValue] = {
        "label_family": "eval_cp",
        "label_pov": "white",
        "phase_opening_min_material": 5600,
        "phase_middlegame_min_material": 2600,
        "record_identity": "sha256(EPD) — position without move clocks",
    }
    for key, value in (settings_overrides or {}).items():
        if key not in ("phase_opening_min_material", "phase_middlegame_min_material"):
            raise LabError(f"unknown setting {key!r}; overridable settings: "
                           "phase_opening_min_material, phase_middlegame_min_material")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise LabError(f"setting {key!r} must be a non-negative integer material threshold")
        settings[key] = value
    settings["phase_rule"] = (f"non-pawn material >= {settings['phase_opening_min_material']} opening, "
                              f">= {settings['phase_middlegame_min_material']} middlegame, else endgame")
    seen: set[str] = set()  # canonical record identities (EPD), keep-first dedup
    records, duplicates, rejected = [], 0, 0
    # Games connected by any shared canonical position (transpositions) must not be split
    # apart later: leakage safety is a property of the whole connected component, so
    # components are resolved before dedup and every record of a component carries the
    # component's group. Union-find over game ids; deterministic by construction.
    parent: dict[str, str] = {}

    def find(game: str) -> str:
        parent.setdefault(game, game)
        while parent[game] != game:
            parent[game] = parent[parent[game]]
            game = parent[game]
        return game

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # stable merge direction: lexicographically smaller root wins
            high, low = (ra, rb) if ra > rb else (rb, ra)
            parent[high] = low

    game_of_row: dict[tuple[str, int], str] = {}
    games_of_record: dict[str, set[str]] = {}
    for src in sources:
        rows = read_jsonl(store.abs(src.path))
        inferred = infer_games(rows)
        for row_index, (row, inferred_game) in enumerate(zip(rows, inferred)):
            if row.get("game_unknown"):
                continue
            game = f"{src.id}:{inferred_game}"
            game_of_row[(src.id, row_index)] = game
            try:
                epd = chess.Board(row["fen"]).epd()
            except (KeyError, ValueError):
                continue
            games_of_record.setdefault(epd, set()).add(game)
    for games in games_of_record.values():
        if len(games) > 1:
            first = min(games)
            for other in sorted(games):
                union(first, other)
    for game in set(game_of_row.values()):  # materialize every game, linked or not
        find(game)

    groups: dict[tuple[str, int], tuple[str, list[str]]] = {}
    component_members: dict[str, list[str]] = {}
    for game in sorted(parent):
        component_members.setdefault(find(game), []).append(game)
    for (src_id, row_index), game in game_of_row.items():
        root = find(game)
        members = component_members[root]
        if len(members) == 1:
            groups[(src_id, row_index)] = (game, [game])
        else:
            # The component label must not depend on which source the surviving row came
            # from: it is derived solely from the sorted fully-qualified member games, so a
            # component spanning S0001 and S0002 gets ONE group (and one eventual split).
            label = f"xc{hash_obj(sorted(members))[7:15]}"
            groups[(src_id, row_index)] = (label, members)

    for src in sources:
        rows = read_jsonl(store.abs(src.path))
        inferred = infer_games(rows)
        for row_index, (row, inferred_game) in enumerate(zip(rows, inferred)):
            # Grouping is a claim about provenance: only explicit game identity or the
            # row-level game_unknown marker are trusted; a heuristic is never upgraded
            # into an identity.
            game_unknown = bool(row.get("game_unknown"))
            if game_unknown:
                group_label, member_games = f"{src.id}:UNKNOWN", []
            else:
                group_label, member_games = groups[(src.id, row_index)]
            try:
                board = chess.Board(row["fen"])
                if not board.is_valid():
                    raise ValueError("illegal position")
            except (KeyError, ValueError):
                rejected += 1
                continue
            epd = board.epd()
            record_id = "pos_" + sha256_bytes(epd.encode())[7:27]
            if dedup == "exact-epd-keep-first":
                if record_id in seen:
                    duplicates += 1
                    continue
                seen.add(record_id)
            labels = []
            if row.get("cp") is not None:
                labels.append({"family": "eval_cp", "value": row["cp"], "pov": "white",
                               "authority": row.get("label_authority") or src.label_authorities[0],
                               "producer": src.id, "label_schema_version": 1})
            record = {
                "record_id": record_id, "group": group_label, "source_id": src.id, "source_row": row_index,
                "ply": row.get("ply", _ply_from_fen(row["fen"])), "fen": board.fen(), "epd": epd,
                "stm": "w" if board.turn == chess.WHITE else "b",
                "phase": game_phase(board, int(settings["phase_opening_min_material"]),
                                    int(settings["phase_middlegame_min_material"])),
                "material": material_signature(board), "labels": labels, "result": row.get("res"),
            }
            if game_unknown:
                record["group_unknown"] = True
            else:
                record["games"] = member_games  # full provenance: every game that can reach it
            records.append(record)
    normalization_id = store.next_id("N")
    rel = f"canonical/{normalization_id}/records.jsonl"
    output_hash = write_jsonl(store.abs(rel), records)
    store.make_readonly(store.abs(rel))
    source_hashes = {src.id: src.content_hash for src in sources}
    return store.create(Normalization(
        id=normalization_id, name=name, source_ids=list(source_ids), source_hashes=source_hashes, normalizer=NORMALIZER,
        normalizer_version=NORMALIZER_VERSION, canonical_schema_version=CANONICAL_SCHEMA_VERSION, settings=settings,
        dedup_policy=dedup,
        recipe_hash=hash_obj({"source_hashes": source_hashes, "normalizer": NORMALIZER, "version": NORMALIZER_VERSION,
                              "canonical_schema_version": CANONICAL_SCHEMA_VERSION, "settings": settings, "dedup": dedup}),
        path=rel, output_hash=output_hash, record_count=len(records), duplicate_count=duplicates, rejected_count=rejected,
        compute=meter.compute(accepted_examples=len(records), discarded_examples=duplicates + rejected),
        created_at=utc_now(),
    ))


# ---------------------------------------------------------------------------
# L3 frozen datasets
# ---------------------------------------------------------------------------


def _split_for(group: str, seed: int, fractions: Sequence[float]) -> str:
    u = int(sha256_bytes(f"{seed}:{group}".encode())[7:15], 16) / 2**32
    if u < fractions[0]:
        return "train"
    return "val" if u < fractions[0] + fractions[1] else "test"


def _cp_bucket(record: dict) -> str:
    label = next((lab for lab in record["labels"] if lab["family"] == "eval_cp"), None)
    if label is None:
        return "unlabelled"
    cp = abs(float(label["value"]))
    return "0" if cp == 0 else "1-99" if cp < 100 else "100-299" if cp < 300 else "300-899" if cp < 900 else "900+"


def dataset_manifest_hash(dataset: Dataset) -> str:
    return hash_obj(dataset.model_dump(mode="json", exclude={"id", "created_at", "manifest_hash", "compute"}))


def freeze_dataset(store: Store, normalization_id: str, *, name: str, selection: Optional[Mapping[str, str]] = None,
                   split_seed: int = 0, fractions: Sequence[float] = (0.8, 0.1, 0.1),
                   required_labels: Sequence[str] = ("eval_cp",), parent_id: Optional[str] = None,
                   arms: Optional[Sequence[Mapping]] = None,
                   record_ids: Optional[Sequence[str]] = None,
                   campaign: Optional[Mapping[str, object]] = None,
                   target_spec=None,
                   unknown_grouping: str = "refuse") -> Dataset:
    """Freeze a live view into an immutable D####.

    Two equivalent forms: a flat `selection` over record fields, or an explicit `stack` of
    filtered subpopulations with per-arm sampling policies. Both pin every L2 label file that
    exists for the normalization at freeze time and join those labels into the split rows.
    """
    selection = {k: str(v) for k, v in (selection or {}).items()}
    bad = sorted(set(selection) - set(SELECTABLE_FIELDS))
    if bad:
        raise LabError(f"cannot select on {bad}; selectable fields: {SELECTABLE_FIELDS}")
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-9:
        raise LabError("split fractions must be three non-negative numbers summing to 1")
    if unknown_grouping not in ("refuse", "shared"):
        raise LabError("unknown_grouping must be 'refuse' or 'shared'")
    meter = _Meter()
    norm: Normalization = store.get(normalization_id)
    if sha256_file(store.abs(norm.path)) != norm.output_hash:
        raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    records = _load_canonical(store, norm)

    arm_list: list[StackArm] = []
    stack_overlap = 0
    if record_ids is not None:
        if arms is not None or selection:
            raise LabError("record_ids is an exact selection; do not combine it with arms or a flat selection")
        wanted = list(dict.fromkeys(record_ids))
        by_id = {record["record_id"]: record for record in records}
        missing_ids = [record_id for record_id in wanted if record_id not in by_id]
        if missing_ids:
            raise LabError(f"{len(missing_ids)} requested record ids are not in the corpus "
                           f"(e.g. {missing_ids[0]})")
        kept = [by_id[record_id] for record_id in wanted]
        excluded, missing = len(records) - len(kept), 0
        kept = [record for record in kept
                if set(required_labels) <= {lab["family"] for lab in record["labels"]}]
    elif arms is not None:
        arm_list = [_coerce_arm(arm) for arm in arms]
        if not arm_list:
            raise LabError("arms provided but empty; pass no arms for a flat selection")
        if selection:
            raise LabError("a stack freeze takes arms, not a flat selection")
        chosen: dict[str, int] = {}
        for arm in arm_list:
            _, selected = _apply_arm(records, arm)
            for index in selected:
                record_id = records[index]["record_id"]
                if record_id in chosen:
                    stack_overlap += 1
                else:
                    chosen[record_id] = index
        kept = [records[i] for i in chosen.values()]
        excluded = len(records) - len(chosen)
        missing = sum(1 for record in kept
                      if not set(required_labels) <= {lab["family"] for lab in record["labels"]})
        kept = [record for record in kept
                if set(required_labels) <= {lab["family"] for lab in record["labels"]}]
    else:
        kept, excluded, missing = [], 0, 0
        for record in records:
            if any(str(record.get(field)) != value for field, value in selection.items()):
                excluded += 1
                continue
            if not set(required_labels) <= {lab["family"] for lab in record["labels"]}:
                missing += 1
                continue
            kept.append(record)
    if not kept:
        raise LabError("selection retains no records; nothing to freeze")
    if target_spec is not None:
        # detect malformed supervision BEFORE materialising runs: every kept record must
        # have exactly one label satisfying the complete spec (family alone is not enough)
        from .targets import validate_records
        validate_records(kept, target_spec)
    unknown = [record for record in kept if record.get("group_unknown")]
    if unknown and unknown_grouping == "refuse":
        raise LabError(
            f"{len(unknown)} position(s) have unknown game grouping; a leakage-safe split cannot be "
            "trusted while same-game positions may be grouped apart. Repair the grouping (re-join "
            "source_ref to the original sources) or freeze with unknown_grouping='shared' to place "
            "every unknown-grouping position in one common group (safe, but unusable for balanced splits).")
    if unknown:
        for record in unknown:
            record["group"] = "UNKNOWN-GROUP"  # one shared group: cannot straddle splits

    dataset_id = store.next_id("D")
    by_split: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES}
    for record in kept:
        by_split[_split_for(record["group"], split_seed, fractions)].append(record)
    manifests = []
    for split_name in SPLIT_NAMES:
        rows = by_split[split_name]
        rel = f"datasets/{dataset_id}/{split_name}.jsonl"
        file_hash = write_jsonl(store.abs(rel), rows)
        store.make_readonly(store.abs(rel))
        manifests.append(SplitManifest(
            name=split_name, count=len(rows), group_count=len({r["group"] for r in rows}), path=rel, file_hash=file_hash,
            record_ids_hash=hash_obj(sorted(r["record_id"] for r in rows)),
        ))
    if arm_list:
        sampling_policy = "stack union (cross-arm duplicates dropped): " + "; ".join(
            f"{arm.name}={arm.policy.value}" for arm in arm_list)
    else:
        sampling_policy = "natural concatenation of all selected records; no weighting or oversampling"
    dataset = Dataset(
        id=dataset_id, name=name, normalization_id=norm.id, normalization_recipe_hash=norm.recipe_hash,
        source_ids=norm.source_ids, canonical_schema_version=norm.canonical_schema_version,
        label_requirements=list(required_labels), selection=dict(selection),
        sampling_policy=sampling_policy,
        dedup_policy=norm.dedup_policy,
        split_policy={"method": "group-hash", "group_key": "source game", "seed": split_seed,
                      "unknown_grouping": unknown_grouping, "unknown_group_positions": len(unknown),
                      "train": float(fractions[0]), "val": float(fractions[1]), "test": float(fractions[2])},
        splits=manifests,
        stack=arm_list,
        label_sets=[_label_set_ref(store, path) for path in label_set_paths(store, norm.id)],
        counts={"records": len(kept), "groups": len({r["group"] for r in kept}), "excluded_by_selection": excluded,
                "missing_required_labels": missing, "stack_overlap": stack_overlap},
        coverage={
            "split": {s: len(by_split[s]) for s in SPLIT_NAMES},
            "phase": dict(Counter(r["phase"] for r in kept)),
            "stm": dict(Counter(r["stm"] for r in kept)),
            "abs_cp_bucket": dict(Counter(_cp_bucket(r) for r in kept)),
            "source": dict(Counter(r["source_id"] for r in kept)),
            **({"stack": dict(_stack_effectives(arm_list, records))} if arm_list else {}),
        },
        label_provenance=dict(Counter(lab["authority"] for r in kept for lab in r["labels"])),
        parent_id=parent_id, campaign=dict(campaign or {}), manifest_hash="",
        compute=meter.compute(accepted_examples=len(kept), discarded_examples=excluded + missing),
        created_at=utc_now(),
    )
    dataset.manifest_hash = dataset_manifest_hash(dataset)
    return store.create(dataset)


def _stack_effectives(arm_list: list[StackArm], records: list[dict]) -> list[tuple[str, int]]:
    """Per-arm effective counts recomputed from the same deterministic rules as the preview."""
    out = []
    for arm in arm_list:
        _, selected = _apply_arm(records, arm)
        out.append((arm.name, len(selected)))
    return out


def load_split(store: Store, dataset: Dataset, split_name: str) -> list[dict]:
    manifest = next((s for s in dataset.splits if s.name == split_name), None)
    if manifest is None:
        raise LabError(f"{dataset.id} has no split {split_name!r}")
    return read_jsonl(store.abs(manifest.path))


def verify_dataset(store: Store, dataset: Dataset) -> list[IntegrityCheck]:
    checks = [make_check("dataset_manifest_hash", dataset_manifest_hash(dataset) == dataset.manifest_hash,
                         f"{dataset.id} manifest recomputes to its recorded hash")]
    problems: list[str] = []
    ids: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for manifest in dataset.splits:
        path = store.abs(manifest.path)
        if not path.exists():
            problems.append(f"{manifest.name}: file missing")
            continue
        if sha256_file(path) != manifest.file_hash:
            problems.append(f"{manifest.name}: file hash mismatch")
            continue
        rows = read_jsonl(path)
        row_ids = [r["record_id"] for r in rows]
        if len(rows) != manifest.count or hash_obj(sorted(row_ids)) != manifest.record_ids_hash:
            problems.append(f"{manifest.name}: record identities differ from manifest")
        ids[manifest.name] = set(row_ids)
        groups[manifest.name] = {r["group"] for r in rows}
    checks.append(make_check("dataset_files", not problems,
                             "; ".join(problems) or "every split file matches its manifest hash and record identities"))
    overlaps = [f"{a}/{b}: {len(ids[a] & ids[b])} positions, {len(groups[a] & groups[b])} games"
                for a, b in combinations(ids, 2) if ids[a] & ids[b] or groups[a] & groups[b]]
    checks.append(make_check("split_disjoint", not overlaps,
                             "; ".join(overlaps) or "no position or game appears in more than one split"))
    return checks


# ---------------------------------------------------------------------------
# L2 — append-only accumulated labels
# ---------------------------------------------------------------------------


def label_set_paths(store: Store, normalization_id: str) -> list[Path]:
    folder = store.abs(f"canonical/{normalization_id}/labels")
    return sorted(folder.glob("*.jsonl")) if folder.is_dir() else []


def _label_set_ref(store: Store, path: Path) -> LabelSetRef:
    rows = read_jsonl(path)
    families: Counter = Counter()
    authorities: Counter = Counter()
    producers: Counter = Counter()
    registry = 1
    policy_hashes: set[str] = set()
    for row in rows:
        families[row["family"]] += 1
        authorities[row["authority"]] += 1
        producers[row["producer"]] += 1
        registry = int(row.get("registry_version", 1))
        if row.get("policy_hash"):
            policy_hashes.add(str(row["policy_hash"]))
    if len(policy_hashes) > 1:
        raise LabError(f"{path.name} carries {len(policy_hashes)} distinct policy hashes "
                       f"({', '.join(sorted(policy_hashes))}); one label set must belong to exactly "
                       "one policy identity")
    return LabelSetRef(path=store.rel(path), file_hash=sha256_file(path), rows=len(rows),
                       families=dict(families), authorities=dict(authorities), producers=dict(producers),
                       registry_version=registry, label_schema_version=LABEL_SCHEMA_VERSION,
                       policy_hash=next(iter(policy_hashes), None))


def _load_canonical(store: Store, normalization: Normalization) -> list[dict]:
    """Canonical records with every registered L2 label file joined in registration order."""
    records = read_jsonl(store.abs(normalization.path))
    for record in records:
        record["labels"] = list(record.get("labels") or [])
    for path in label_set_paths(store, normalization.id):
        by_id: dict[str, list[dict]] = {}
        for row in read_jsonl(path):
            by_id.setdefault(row["record_id"], []).append(row)
        for record in records:
            record["labels"].extend(by_id.get(record["record_id"], ()))
    return records


def _scalarize(value):
    """Label payloads may be scalars, lists of scalars, or flat dicts of scalars."""
    if isinstance(value, SCALAR):
        return value
    if isinstance(value, list):
        return [str(v) if not isinstance(v, SCALAR) else v for v in value]
    if isinstance(value, dict):
        return {str(k): _scalarize(v) for k, v in value.items()}
    return str(value)


def register_labels(store: Store, normalization_id: str, *, family: str, producer: str, authority: str,
                    rows: Sequence[Mapping], pov: str = "white", registry_version: int = 1) -> LabelSetRef:
    """Append one label set as a new immutable file; existing records and labels are never modified."""
    norm: Normalization = store.get_as(normalization_id, Normalization)
    if family not in LABEL_FAMILIES:
        raise LabError(f"unknown label family {family!r}; register it in LABEL_FAMILIES first: "
                       f"{', '.join(sorted(LABEL_FAMILIES))}")
    if sha256_file(store.abs(norm.path)) != norm.output_hash:
        raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    if not rows:
        raise LabError("a label set needs at least one row")
    known = {record["record_id"] for record in read_jsonl(store.abs(norm.path))}
    seen: set[tuple[str, str]] = set()  # (record_id, value-hash): budgets may add rows, accidents may not
    stamped = utc_now()
    clean_rows: list[dict] = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id not in known:
            raise LabError(f"label row references unknown record {record_id!r} in {norm.id}")
        if "value" not in row:
            raise LabError(f"label for {record_id} carries no value")
        value = _scalarize(row["value"])
        clean: dict = {"record_id": record_id, "family": family,
                       "value": value,
                       "pov": pov, "authority": authority, "producer": producer,
                       "registry_version": int(registry_version), "label_schema_version": LABEL_SCHEMA_VERSION,
                       "produced_at": stamped}
        for key in ("budget", "confidence", "note", "components", "policy_hash"):
            if key in row:
                clean[key] = _scalarize(row[key])
        # Dedup identity is the whole observation (record + value + budget + components):
        # two searches at different node budgets may legitimately return identical values.
        dedup_key = (record_id, hash_obj({key: clean.get(key) for key in ("value", "budget", "components")}))
        if dedup_key in seen:
            raise LabError(f"duplicate observation for {record_id} in one set (identical value and budget); "
                           "separate authorities go in separate sets")
        seen.add(dedup_key)
        clean_rows.append(clean)
    sequence = len(label_set_paths(store, normalization_id)) + 1
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", producer)[:24] or "producer"
    rel = f"canonical/{normalization_id}/labels/labels-{sequence:04d}-{slug}.jsonl"
    write_jsonl(store.abs(rel), clean_rows)
    store.make_readonly(store.abs(rel))
    return _label_set_ref(store, store.abs(rel))


def copy_label_sets(store: Store, from_normalization_id: str, to_normalization_id: str) -> dict[str, int]:
    """Carry label sets into a new canonical corpus, keeping only labels whose record identity survived.

    Records whose EPD identity survived re-standardization keep their labels unchanged ("provably
    unaffected"); dropped rows are counted and reported, and changed-semantics labels must be
    re-registered by their producer under a new registry version.
    """
    source: Normalization = store.get_as(from_normalization_id, Normalization)
    target: Normalization = store.get_as(to_normalization_id, Normalization)
    if source.id == target.id:
        raise LabError("copy_label_sets needs two different normalizations")
    if sha256_file(store.abs(source.path)) != source.output_hash or sha256_file(store.abs(target.path)) != target.output_hash:
        raise TamperError("canonical corpus no longer matches its normalization output hash")
    target_ids = {record["record_id"] for record in read_jsonl(store.abs(target.path))}
    kept_total = dropped_total = sets_copied = 0
    for path in label_set_paths(store, source.id):
        rows = read_jsonl(path)
        kept_rows = [row for row in rows if row["record_id"] in target_ids]
        kept_total += len(kept_rows)
        dropped_total += len(rows) - len(kept_rows)
        if not kept_rows:
            continue
        sets_copied += 1
        rel = f"canonical/{target.id}/labels/copied-{sets_copied:04d}-{path.name}"
        write_jsonl(store.abs(rel), kept_rows)
        store.make_readonly(store.abs(rel))
    return {"sets_copied": sets_copied, "labels_kept": kept_total, "labels_dropped": dropped_total}


# ---------------------------------------------------------------------------
# Catalog filtering and dataset stacking
# ---------------------------------------------------------------------------

ARM_FILTER_FIELDS = ("source_id", "phase", "stm", "material", "result", "group", "ply_min", "ply_max",
                     "eval_bucket", "label_family", "authority", "tier", "producer", "motif", "strategy")
CATALOG_SORT_KEYS = ("record_id", "ply", "phase", "material", "label_count")
CATALOG_ONLY_FIELDS = ("dataset", "split")
CATALOG_FILTER_FIELDS = ARM_FILTER_FIELDS + CATALOG_ONLY_FIELDS
BALANCE_BUCKETS = ("phase", "stm", "material", "source_id", "eval_bucket")


def _label_matches(record: dict, key: str, wanted: str) -> bool:
    for label in record["labels"]:
        if key == "label_family" and label["family"] == wanted:
            return True
        if key == "authority" and label.get("authority") == wanted:
            return True
        if key == "producer" and label.get("producer") == wanted:
            return True
        if key == "tier" and LABEL_FAMILIES.get(label["family"], {}).get("tier") == wanted:
            return True
    return False


def _record_label_values(record: dict, family: str) -> list:
    out = []
    for label in record["labels"]:
        if label["family"] == family:
            value = label["value"]
            out.extend(value if isinstance(value, list) else [value])
    return out


def _strategy_tag_matches(record: dict, wanted: str) -> bool:
    """`strategy=<tag>` matches tag presence; `tag:value` (or >=/<=/>/</!=) compares its value."""
    for operator in ("!=", ">=", "<=", ">", "<", ":"):
        if operator in wanted:
            tag, raw = wanted.split(operator, 1)
            break
    else:
        tag, operator, raw = wanted, "", None
    for value in _record_label_values(record, "strategy"):
        if not isinstance(value, dict):
            continue
        for key, entry in value.items():
            if str(key).lower() != tag.lower():
                continue
            if not operator:
                return True
            if operator == ":":
                if str(entry).lower() == raw.lower():
                    return True
                continue
            try:
                left, right = float(entry), float(raw)
            except (TypeError, ValueError):
                left, right = str(entry), raw
            if operator == ">" and left > right:
                return True
            if operator == "<" and left < right:
                return True
            if operator == ">=" and left >= right:
                return True
            if operator == "<=" and left <= right:
                return True
            if operator == "!=" and left != right:
                return True
            if operator == "==" or (operator not in (">", "<", ">=", "<=", "!=") and left == right):
                return True
    return False


def _filter_matches(record: dict, flt: Mapping[str, ConfigValue]) -> bool:
    for key, value in flt.items():
        if key in ("label_family", "authority", "tier", "producer"):
            if not _label_matches(record, key, str(value)):
                return False
        elif key == "eval_bucket":
            if _cp_bucket(record) != str(value):
                return False
        elif key == "motif":
            if str(value).lower() not in {str(v).lower() for v in _record_label_values(record, "motif")}:
                return False
        elif key == "strategy":
            if not _strategy_tag_matches(record, str(value)):
                return False
        elif key == "ply_min":
            if record["ply"] < int(value):
                return False
        elif key == "ply_max":
            if record["ply"] > int(value):
                return False
        elif str(record.get(key)) != str(value):
            return False
    return True


def _coerce_arm(raw: Mapping) -> StackArm:
    try:
        arm = StackArm(**{k: v for k, v in raw.items()})
    except Exception as exc:
        raise LabError(f"invalid stack arm {raw.get('name', '?')!r}: {exc}") from None
    unknown = sorted(set(arm.filter) - set(ARM_FILTER_FIELDS))
    if unknown:
        raise LabError(f"arm {arm.name!r} filters on unknown field(s) {unknown}; allowed: {ARM_FILTER_FIELDS}")
    if arm.policy == StackPolicy.FIXED_ROWS and (arm.rows is None or arm.rows < 1):
        raise LabError(f"arm {arm.name!r}: fixed_rows needs rows >= 1")
    if arm.policy == StackPolicy.FRACTION and (arm.fraction is None or not 0 < arm.fraction <= 1):
        raise LabError(f"arm {arm.name!r}: fraction needs 0 < fraction <= 1")
    if arm.policy == StackPolicy.BALANCE and (arm.balance_bucket not in BALANCE_BUCKETS
                                              or not arm.balance_cap or arm.balance_cap < 1):
        raise LabError(f"arm {arm.name!r}: balance needs balance_bucket in {BALANCE_BUCKETS} and balance_cap >= 1")
    return arm


def _bucket_of(record: dict, bucket: str) -> str:
    if bucket == "eval_bucket":
        return _cp_bucket(record)
    if bucket in ("phase", "stm", "material", "source_id"):
        return str(record[bucket])
    raise LabError(f"cannot bucket by {bucket!r}; buckets: {BALANCE_BUCKETS}")


def _apply_arm(records: list[dict], arm: StackArm) -> tuple[list[int], list[int]]:
    """Indices matching the arm's filter ('available') and indices selected by its policy ('effective')."""
    matching = [i for i, record in enumerate(records) if _filter_matches(record, arm.filter)]
    if arm.policy == StackPolicy.ALL:
        return matching, matching
    rng = random.Random(arm.seed)

    def sample(count: int) -> list[int]:
        if count > len(matching):
            raise LabError(f"arm {arm.name!r}: requested {count} rows but only {len(matching)} match its filter")
        if count == len(matching):
            return matching
        return sorted(rng.sample(matching, count))

    if arm.policy == StackPolicy.FIXED_ROWS:
        return matching, sample(int(arm.rows))
    if arm.policy == StackPolicy.FRACTION:
        return matching, sample(max(1, round(len(matching) * float(arm.fraction))))
    # BALANCE: seeded stratified cap per bucket value
    buckets: dict[str, list[int]] = {}
    for index in matching:
        buckets.setdefault(_bucket_of(records[index], str(arm.balance_bucket)), []).append(index)
    chosen: list[int] = []
    for value in sorted(buckets):
        indices = buckets[value]
        if len(indices) > int(arm.balance_cap):
            indices = sorted(rng.sample(indices, int(arm.balance_cap)))
        chosen.extend(indices)
    return matching, sorted(chosen)


def stack_preview(store: Store, normalization_id: str, arms: Sequence[Mapping]) -> "StackPreview":
    """Raw available vs effective sampled contribution per arm, plus combined distributions."""
    from .schemas import ArmPreview, StackPreview

    norm: Normalization = store.get_as(normalization_id, Normalization)
    if sha256_file(store.abs(norm.path)) != norm.output_hash:
        raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    if not arms:
        raise LabError("a stack needs at least one arm")
    records = _load_canonical(store, norm)
    previews: list[ArmPreview] = []
    union: dict[str, int] = {}
    for raw in arms:
        arm = _coerce_arm(raw)
        matching, selected = _apply_arm(records, arm)
        previews.append(ArmPreview(name=arm.name, policy=arm.policy.value, filter=dict(arm.filter),
                                   available=len(matching), effective=len(selected)))
        for index in selected:
            union.setdefault(records[index]["record_id"], index)
    rows = [records[i] for i in union.values()]
    distributions = {
        "phase": dict(Counter(r["phase"] for r in rows)),
        "stm": dict(Counter(r["stm"] for r in rows)),
        "source": dict(Counter(r["source_id"] for r in rows)),
        "abs_cp_bucket": dict(Counter(_cp_bucket(r) for r in rows)),
        "tier": dict(Counter(tier for r in rows
                             for tier in {LABEL_FAMILIES.get(lab["family"], {}).get("tier", "unknown")
                                          for lab in r["labels"]})),
    }
    return StackPreview(normalization_id=norm.id, arms=previews, effective_total=len(rows),
                        unique_records=len(rows), distributions=distributions)


def _fact_bucket_any(record: dict, key: str) -> int:
    for label in record["labels"]:
        if label["family"] == "facts" and isinstance(label["value"], dict):
            entry = label["value"].get(key)
            if isinstance(entry, dict):
                return int(entry.get("bucket", 0))
    return 0


def catalog(store: Store, normalization_id: str, *, filters: Optional[Mapping[str, ConfigValue]] = None,
            offset: int = 0, limit: int = 50, sort_by: Optional[str] = None) -> "CatalogResponse":
    """Filterable corpus catalog: facet counts plus one page of canonical rows, labels joined.

    Sorting accepts ``record_id|ply|phase|material|label_count`` or ``fact:<KEY>``
    (descending facts bucket, e.g. ``fact:HANGING_MATERIAL``); ``fact_key`` +
    ``fact_min`` filter on a geometry fact bucket, ``motif``/``strategy`` filter
    on the CVS motif/strategy label values.
    """
    from .schemas import CatalogRecord, CatalogResponse

    norm: Normalization = store.get_as(normalization_id, Normalization)
    if sha256_file(store.abs(norm.path)) != norm.output_hash:
        raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    flt = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
    unknown = sorted(set(flt) - set(CATALOG_FILTER_FIELDS)
                     - {"fact_key", "fact_min", "fact_favors"})
    if unknown:
        raise LabError(f"unknown catalog filter(s) {unknown}; allowed: {sorted(CATALOG_FILTER_FIELDS) + ['fact_key', 'fact_min', 'fact_favors']}")
    records = _load_canonical(store, norm)

    fact_key = str(flt.pop("fact_key", "") or "")
    fact_min = int(flt.pop("fact_min", 0) or 0)
    fact_favors = str(flt.pop("fact_favors", "") or "")
    if fact_key and fact_key not in {key for key, _ in __import__("cvslab.facts", fromlist=["GEOMETRY_FAMILIES"]).GEOMETRY_FAMILIES}:
        raise LabError(f"unknown geometry fact {fact_key!r}")

    def fact_bucket(record: dict) -> int:
        for label in record["labels"]:
            if label["family"] == "facts" and isinstance(label["value"], dict):
                entry = label["value"].get(fact_key)
                if isinstance(entry, dict):
                    return int(entry.get("bucket", 0))
        return 0

    if fact_key:
        records = [r for r in records if fact_bucket(r) >= fact_min]
    if fact_favors:
        records = [r for r in records
                   if any(label["family"] == "facts" and isinstance(label["value"], dict)
                          and isinstance((entry := label["value"].get(fact_key or "")), dict)
                          and entry.get("favors") == fact_favors
                          for label in r["labels"])]

    memberships: dict[str, dict[str, list[str]]] = {}
    for dataset in store.list("D", verify=False, kind=Dataset):
        if dataset.normalization_id != norm.id:
            continue
        for manifest in dataset.splits:
            path = store.abs(manifest.path)
            if not path.exists():
                continue
            for row in read_jsonl(path):
                memberships.setdefault(row["record_id"], {}).setdefault(dataset.id, []).append(manifest.name)
    if "dataset" in flt:
        wanted = str(flt.pop("dataset"))
        records = [r for r in records if wanted in memberships.get(r["record_id"], {})]
    if "split" in flt:
        wanted = str(flt.pop("split"))
        records = [r for r in records
                   if any(wanted in splits for splits in memberships.get(r["record_id"], {}).values())]
    matching = [r for r in records if _filter_matches(r, flt)]

    if sort_by:
        if sort_by.startswith("fact:"):
            matching.sort(key=lambda r: _fact_bucket_any(r, sort_by.split(":", 1)[1]), reverse=True)
        elif sort_by == "label_count":
            matching.sort(key=lambda r: -len(r["labels"]))
        elif sort_by in CATALOG_SORT_KEYS and sort_by != "record_id":
            matching.sort(key=lambda r: str(r.get(sort_by)))
        elif sort_by != "record_id":
            raise LabError(f"sort_by must be one of {CATALOG_SORT_KEYS} or 'fact:<KEY>'")

    def facet(source_records: list[dict], key: str) -> dict[str, int]:
        counts: Counter = Counter()
        for record in source_records:
            if key == "eval_bucket":
                counts[_cp_bucket(record)] += 1
            elif key == "label_family":
                counts.update(label["family"] for label in record["labels"])
            elif key == "authority":
                counts.update(label.get("authority", "?") for label in record["labels"])
            elif key == "producer":
                counts.update(label.get("producer", "?") for label in record["labels"])
            elif key == "tier":
                counts.update(LABEL_FAMILIES.get(label["family"], {}).get("tier", "unknown")
                              for label in record["labels"])
            elif key == "dataset":
                counts.update(memberships.get(record["record_id"], {}).keys())
            else:
                counts[str(record.get(key))] += 1
        return dict(counts)

    facets = {key: facet(matching, key) for key in
              ("source_id", "phase", "stm", "material", "result", "dataset")}
    for key in ("eval_bucket", "label_family", "authority", "tier", "producer"):
        facets[key] = facet(matching, key)
    motif_counts: Counter = Counter()
    strategy_counts: Counter = Counter()
    for record in matching:
        # facet semantics: a record counts once per facet value it carries
        motif_counts.update({str(v).lower() for v in _record_label_values(record, "motif")})
        for value in _record_label_values(record, "strategy"):
            if isinstance(value, dict):
                strategy_counts.update(value.keys())
    facets["motif"] = dict(motif_counts)
    facets["strategy"] = dict(strategy_counts)
    label_count = sum(facets["label_family"].values())

    window = matching[max(0, offset): max(0, offset) + max(1, min(int(limit), 500))]
    page = [CatalogRecord(
        record_id=r["record_id"], group=r["group"], source_id=r["source_id"], source_row=r["source_row"],
        ply=r["ply"], fen=r["fen"], epd=r["epd"], stm=r["stm"], phase=r["phase"], material=r["material"],
        result=r.get("result"), labels=r["labels"],
        label_tiers=dict(Counter(LABEL_FAMILIES.get(lab["family"], {}).get("tier", "unknown")
                                 for lab in r["labels"])),
        memberships=memberships.get(r["record_id"], {}),
    ) for r in window]
    return CatalogResponse(
        normalization_id=norm.id, record_count=norm.record_count, unique_count=norm.record_count,
        duplicate_count=norm.duplicate_count, label_count=label_count, filters_applied=dict(flt),
        facets=facets, page=page, page_offset=max(0, offset), page_size=max(1, min(int(limit), 500)),
        total_matching=len(matching),
    )


# ---------------------------------------------------------------------------
# Re-standardization: a new N#### from the same immutable sources, with a diff
# ---------------------------------------------------------------------------


def _label_map(normalization: Normalization, store: Store) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for record in _load_canonical(store, normalization):
        for label in record["labels"]:
            key = (record["record_id"], label["family"], label.get("authority"), label.get("producer"))
            out[key] = label
    return out


def _coverage(records: list[dict]) -> dict[str, dict[str, int]]:
    return {
        "phase": dict(Counter(r["phase"] for r in records)),
        "stm": dict(Counter(r["stm"] for r in records)),
        "abs_cp_bucket": dict(Counter(_cp_bucket(r) for r in records)),
    }


def migration_diff(store: Store, old_normalization_id: str, new_normalization_id: str) -> "MigrationDiff":
    from .schemas import MigrationDiff as Diff

    old_n = store.get_as(old_normalization_id, Normalization)
    new_n = store.get_as(new_normalization_id, Normalization)
    if old_n.id == new_n.id:
        raise LabError("migration diff needs two different normalizations")
    for norm in (old_n, new_n):
        if sha256_file(store.abs(norm.path)) != norm.output_hash:
            raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    old_rows = read_jsonl(store.abs(old_n.path))
    new_rows = read_jsonl(store.abs(new_n.path))
    old_map = {r["record_id"]: r for r in old_rows}
    new_map = {r["record_id"]: r for r in new_rows}
    shared = old_map.keys() & new_map.keys()

    def payload(record: dict) -> dict:
        return {k: v for k, v in record.items() if k != "labels"}

    changed = sum(1 for rid in shared if payload(old_map[rid]) != payload(new_map[rid]))

    old_labels, new_labels = _label_map(old_n, store), _label_map(new_n, store)
    per_family: dict[str, Counter] = {}
    for _rid, family, _auth, _prod in new_labels.keys() - old_labels.keys():
        per_family.setdefault(family, Counter())["added"] += 1
    for _rid, family, _auth, _prod in old_labels.keys() - new_labels.keys():
        per_family.setdefault(family, Counter())["removed"] += 1
    for key in old_labels.keys() & new_labels.keys():
        if old_labels[key]["value"] != new_labels[key]["value"]:
            per_family.setdefault(key[1], Counter())["value_changed"] += 1
    label_changes = {family: dict(counts) for family, counts in sorted(per_family.items())}

    # split membership: recompute each affected dataset's group-hash split for surviving records
    split_changes: list[DatasetSplitChange] = []
    affected_datasets: list[str] = []
    for dataset in store.list("D", verify=False, kind=Dataset):
        if dataset.normalization_id != old_n.id:
            continue
        affected_datasets.append(dataset.id)
        fractions = (float(dataset.split_policy["train"]), float(dataset.split_policy["val"]),
                     float(dataset.split_policy["test"]))
        seed = int(dataset.split_policy["seed"])
        changes = sum(1 for rid in shared
                      if _split_for(old_map[rid]["group"], seed, fractions)
                      != _split_for(new_map[rid]["group"], seed, fractions))
        split_changes.append(DatasetSplitChange(dataset_id=dataset.id, surviving_records=len(shared),
                                                split_changes=changes))

    settings_delta = sorted(k for k in set(old_n.settings) | set(new_n.settings)
                            if old_n.settings.get(k) != new_n.settings.get(k))
    notes = [f"recipe hash {old_n.recipe_hash[:19]}... -> {new_n.recipe_hash[:19]}..."]
    if settings_delta:
        notes.append("settings changed: " + ", ".join(
            f"{k} ({old_n.settings.get(k)!r} -> {new_n.settings.get(k)!r})" for k in settings_delta))
    if old_n.dedup_policy != new_n.dedup_policy:
        notes.append(f"dedup policy {old_n.dedup_policy} -> {new_n.dedup_policy}")

    return Diff(
        from_normalization_id=old_n.id, to_normalization_id=new_n.id,
        records_from=len(old_rows), records_to=len(new_rows),
        added=len(new_map.keys() - old_map.keys()), removed=len(old_map.keys() - new_map.keys()),
        changed=changed,
        duplicate_delta=new_n.duplicate_count - old_n.duplicate_count,
        rejected_delta=new_n.rejected_count - old_n.rejected_count,
        label_changes=label_changes,
        coverage_from=_coverage(list(old_map.values())), coverage_to=_coverage(list(new_map.values())),
        source_contribution_from=dict(Counter(r["source_id"] for r in old_rows)),
        source_contribution_to=dict(Counter(r["source_id"] for r in new_rows)),
        split_changes=split_changes, affected_dataset_ids=affected_datasets, affected_run_ids=[],
        notes=notes,
    )


def rebuild_dataset(store: Store, dataset_id: str, new_normalization_id: str) -> Dataset:
    """'Rebuild this dataset under the new standard': a descendant D####; the original never changes."""
    old = store.get_as(dataset_id, Dataset)
    new_norm = store.get_as(new_normalization_id, Normalization)
    if new_norm.id == old.normalization_id:
        raise LabError(f"{dataset_id} is already frozen on {new_norm.id}")
    for existing in store.list("D", verify=False, kind=Dataset):
        if existing.parent_id == old.id and existing.normalization_id == new_norm.id:
            raise LabError(f"{old.id} was already rebuilt under {new_norm.id} as {existing.id}; "
                           "a repeated rebuild would duplicate an identical descendant")
    fractions = (float(old.split_policy["train"]), float(old.split_policy["val"]), float(old.split_policy["test"]))
    arms = [arm.model_dump() for arm in old.stack] or None
    return freeze_dataset(store, new_norm.id, name=old.name, selection=dict(old.selection) or None,
                          split_seed=int(old.split_policy["seed"]), fractions=fractions,
                          required_labels=list(old.label_requirements), parent_id=old.id, arms=arms)


def label_set_refs(store: Store, normalization_id: str) -> list[LabelSetRef]:
    """Every L2 label set registered under a normalization, in registration order."""
    return [_label_set_ref(store, path) for path in label_set_paths(store, normalization_id)]
