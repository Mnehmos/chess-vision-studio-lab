"""Deterministic CVS analysis facts for canonical records — per position.

Three fact classes attach to every record as regular L2 labels:

**Geometry** is a faithful Python mirror of the engine's Feature Registry v1
(``chess-vision-studio-rust-engine/src/eval/cvs_features.rs``): the same 21
family keys in the same order, the same White-POV signed deltas, the same
magnitude thresholds and 0-3 bucketing. The deltas themselves re-express
``src/eval/rung2.rs`` with python-chess so the lab can compute them without
the engine binary; the registry contract (keys + thresholds + bucket order)
is what models and migrations pin, and ``facts_registry_hash()`` covers it.

**Motifs** follow the ChessTempo tactical-motifs taxonomy
(reference: chesstempo.com/tactical-motifs, accessed 2026-09-14). The catalog
lists every taxonomy motif; ``facts.py`` statically detects the well-defined
subset that needs no search (fork, pin, skewer, ...). Search-time motifs
(zwischenzug, sacrifice, quiet move, ...) stay in the catalog for producers
and are never fabricated statically.

**Strategy** tags summarize the position along the lab's research axes
(material balance, center control, development, king safety, structure).

All three are deterministic: same FEN + registry version -> same labels.
"""
from __future__ import annotations

import chess

from .hashing import hash_obj

CVS_FACTS_REGISTRY_VERSION = 1
GEOMETRY_AUTHORITY = "deterministic.cvs.geometry.v1"
MOTIF_AUTHORITY = "deterministic.cvs.motifs.v1"
STRATEGY_AUTHORITY = "deterministic.cvs.strategy.v1"
FACTS_PRODUCER = "cvslab.facts"

SEE_VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900}
PHASE_VALUE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
MAX_PHASE = 24

# Family keys with |White-POV delta| thresholds for buckets 1/2/3. ORDER IS THE
# CONTRACT — mirrors cvs_features.rs FAMILIES; any change bumps the registry.
GEOMETRY_FAMILIES: list[tuple[str, tuple[float, float, float]]] = [
    ("KING_DANGER", (2.0, 8.0, 20.0)),
    ("KING_ZONE_PRESSURE", (1.0, 3.0, 6.0)),
    ("KING_OPEN_FILE", (1.0, 2.0, 3.0)),
    ("KING_SHIELD", (1.0, 2.0, 3.0)),
    ("KING_CENTRAL_EXPOSURE", (1.0, 2.0, 4.0)),
    ("ENEMY_QUEEN_NEAR_KING", (1.0, 2.0, 3.0)),
    ("OPEN_CENTER_KING", (1.0, 2.0, 3.0)),
    ("KING_ESCAPE_DEFICIT", (1.0, 2.0, 3.0)),
    ("HANGING_MATERIAL", (1.0, 3.0, 6.0)),
    ("MOBILITY_KNIGHT", (2.0, 5.0, 9.0)),
    ("MOBILITY_BISHOP", (2.0, 6.0, 11.0)),
    ("MOBILITY_ROOK", (2.0, 6.0, 12.0)),
    ("MOBILITY_QUEEN", (3.0, 9.0, 16.0)),
    ("PASSED_PAWN", (1.0, 2.0, 3.0)),
    ("CONNECTED_PASSED_PAWN", (1.0, 2.0, 3.0)),
    ("ROOK_OPEN_FILE", (1.0, 2.0, 3.0)),
    ("ROOK_SEMI_OPEN_FILE", (1.0, 2.0, 3.0)),
    ("ROOK_SEVENTH", (1.0, 2.0, 3.0)),
    ("DOUBLED_PAWN", (1.0, 2.0, 3.0)),
    ("ISOLATED_PAWN", (1.0, 2.0, 3.0)),
    ("BISHOP_PAIR", (0.5, 1.0, 1.5)),
]


def facts_registry_hash() -> str:
    """Stable hash of the geometry registry (keys, order, thresholds, buckets)."""
    return hash_obj({"families": GEOMETRY_FAMILIES, "buckets": 3, "version": CVS_FACTS_REGISTRY_VERSION})


def scan(bitboard: int):
    """Iterate square indices of a python-chess bitboard (LSB first)."""
    while bitboard:
        sq = (bitboard & -bitboard).bit_length() - 1
        bitboard &= bitboard - 1
        yield sq


def _bucket(magnitude: float, thresholds: tuple[float, float, float]) -> int:
    if magnitude >= thresholds[2]:
        return 3
    if magnitude >= thresholds[1]:
        return 2
    if magnitude >= thresholds[0]:
        return 1
    return 0


def _chebyshev(a: int, b: int) -> int:
    return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3)))


def _pawns_by_file(board: chess.Board, color: chess.Color) -> list[int]:
    files = [0] * 8
    for sq in scan(board.pawns & board.occupied_co[color]):
        files[sq & 7] += 1
    return files


def _shield_pawns(pawns: int, king_sq: int, color: chess.Color) -> int:
    kf, kr = king_sq & 7, king_sq >> 3
    count = 0
    for df in (-1, 0, 1):
        ff = kf + df
        if not 0 <= ff <= 7:
            continue
        for step in (1, 2):
            rr = kr + step if color == chess.WHITE else kr - step
            if 0 <= rr <= 7 and pawns & chess.BB_SQUARES[rr * 8 + ff]:
                count += 1
    return count


def _king_zone_attacked(board: chess.Board, king_sq: int, by: chess.Color) -> int:
    zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    count = 0
    while zone:
        sq = (zone & -zone).bit_length() - 1
        zone &= zone - 1
        if board.attackers_mask(by, sq):
            count += 1
    return count


def _is_passed(enemy_pawns: int, sq: int, color: chess.Color) -> bool:
    f, r = sq & 7, sq >> 3
    for ff in range(max(0, f - 1), min(7, f + 1) + 1):
        rr = r + 1 if color == chess.WHITE else r - 1
        while 0 <= rr <= 7:
            if enemy_pawns & chess.BB_SQUARES[rr * 8 + ff]:
                return False
            rr += 1 if color == chess.WHITE else -1
    return True


def extract_geometry(board: chess.Board) -> dict[str, dict]:
    """White-POV signed delta and 0-3 magnitude bucket per registry family."""
    w, b = chess.WHITE, chess.BLACK
    w_occ, b_occ = board.occupied_co[w], board.occupied_co[b]
    w_pawns, b_pawns = board.pawns & w_occ, board.pawns & b_occ

    phase_units = sum(board.pieces_mask(chess.PIECE_SYMBOLS.index(symbol.lower()), color).bit_count()
                      * PHASE_VALUE[symbol.upper()]
                      for symbol in PHASE_VALUE for color in (w, b))
    phase_units = min(phase_units, MAX_PHASE)
    mg_w = phase_units / MAX_PHASE
    eg_w = 1.0 - mg_w

    values: dict[str, float] = {}

    # Mobility: pseudo-legal attack squares onto empty-or-enemy squares.
    for key, piece_type in (("MOBILITY_KNIGHT", chess.KNIGHT), ("MOBILITY_BISHOP", chess.BISHOP),
                            ("MOBILITY_ROOK", chess.ROOK), ("MOBILITY_QUEEN", chess.QUEEN)):
        delta = 0.0
        for color, sign in ((w, 1.0), (b, -1.0)):
            not_own = ~board.occupied_co[color]
            for sq in scan(board.pieces_mask(piece_type, color)):
                delta += sign * (board.attacks_mask(sq) & not_own).bit_count()
        values[key] = delta

    # Bishop pair, tapered.
    pair = (1 if board.pieces_mask(chess.BISHOP, w).bit_count() >= 2 else 0) \
        - (1 if board.pieces_mask(chess.BISHOP, b).bit_count() >= 2 else 0)
    values["BISHOP_PAIR"] = pair * (mg_w + eg_w)

    # Pawn structure.
    w_files, b_files = _pawns_by_file(board, w), _pawns_by_file(board, b)
    doubled = isolated = 0
    for file in range(8):
        if w_files[file] > 1:
            doubled -= w_files[file] - 1
        if b_files[file] > 1:
            doubled += b_files[file] - 1
        adj_w = (w_files[file - 1] if file > 0 else 0) + (w_files[file + 1] if file < 7 else 0)
        adj_b = (b_files[file - 1] if file > 0 else 0) + (b_files[file + 1] if file < 7 else 0)
        if w_files[file] > 0 and adj_w == 0:
            isolated -= w_files[file]
        if b_files[file] > 0 and adj_b == 0:
            isolated += b_files[file]
    values["DOUBLED_PAWN"] = float(doubled)
    values["ISOLATED_PAWN"] = float(isolated)

    passed, connected = 0.0, 0.0
    for color, sign in ((w, 1.0), (b, -1.0)):
        own_files = w_files if color == w else b_files
        enemy_pawns = b_pawns if color == w else w_pawns
        for sq in scan(board.pawns & board.occupied_co[color]):
            if _is_passed(enemy_pawns, sq, color):
                passed += sign * ((sq >> 3) if color == w else (6 - (sq >> 3)))
                adjacent = sum(own_files[f] for f in ((sq & 7) - 1, (sq & 7) + 1) if 0 <= f <= 7)
                if adjacent > 0:
                    connected += sign
    values["PASSED_PAWN"] = passed * (mg_w + eg_w)
    values["CONNECTED_PASSED_PAWN"] = connected

    # Rook activity on open/semi-open files and the seventh rank.
    rook_open = rook_semi = rook_seventh = 0.0
    for color, sign in ((w, 1.0), (b, -1.0)):
        own_files = w_files if color == w else b_files
        enemy_files = b_files if color == w else w_files
        for sq in scan(board.pieces_mask(chess.ROOK, color)):
            file = sq & 7
            if own_files[file] == 0 and enemy_files[file] == 0:
                rook_open += sign
            elif own_files[file] == 0:
                rook_semi += sign
            if (color == w and sq >> 3 == 6) or (color == b and sq >> 3 == 1):
                rook_seventh += sign
    values["ROOK_OPEN_FILE"] = rook_open
    values["ROOK_SEMI_OPEN_FILE"] = rook_semi
    values["ROOK_SEVENTH"] = rook_seventh

    # King safety: shield, zone pressure, file exposure.
    w_king, b_king = board.king(w), board.king(b)
    values["KING_SHIELD"] = float(_shield_pawns(w_pawns, w_king, w) - _shield_pawns(b_pawns, b_king, b))
    values["KING_ZONE_PRESSURE"] = float(_king_zone_attacked(board, b_king, w) - _king_zone_attacked(board, w_king, b))

    def file_exposure(own_files: list[int], king_sq: int) -> int:
        kf = king_sq & 7
        return sum(1 for df in (-1, 0, 1) if 0 <= kf + df <= 7 and own_files[kf + df] == 0)

    values["KING_OPEN_FILE"] = float(file_exposure(b_files, b_king) - file_exposure(w_files, w_king))

    # Enemy-queen proximity and the conditioned exposure trio.
    near = 0
    for sq in scan(board.pieces_mask(chess.QUEEN, w)):
        near += max(0, 4 - _chebyshev(sq, b_king))
    for sq in scan(board.pieces_mask(chess.QUEEN, b)):
        near -= max(0, 4 - _chebyshev(sq, w_king))
    values["ENEMY_QUEEN_NEAR_KING"] = float(near)

    def exposure_terms(king_sq: int, color: chess.Color, own_pawns: int, enemy_queen: bool):
        if not enemy_queen:
            return 0.0, 0.0, 0.0
        home = 0 if color == w else 7
        displacement = abs((king_sq >> 3) - home)
        central = 2 <= (king_sq & 7) <= 5
        exposure = (displacement + (1 if central else 0)) * mg_w
        open_center = mg_w if central and _shield_pawns(own_pawns, king_sq, color) == 0 else 0.0
        flights = 0
        escapes = chess.BB_KING_ATTACKS[king_sq] & ~board.occupied_co[color]
        while escapes:
            sq = (escapes & -escapes).bit_length() - 1
            escapes &= escapes - 1
            if not board.attackers_mask(not color, sq):
                flights += 1
        return exposure, open_center, float(max(0, 3 - flights))

    we, wo, wd = exposure_terms(w_king, w, w_pawns, bool(board.pieces_mask(chess.QUEEN, b)))
    be, bo, bd = exposure_terms(b_king, b, b_pawns, bool(board.pieces_mask(chess.QUEEN, w)))
    values["KING_CENTRAL_EXPOSURE"] = be - we
    values["OPEN_CENTER_KING"] = bo - wo
    values["KING_ESCAPE_DEFICIT"] = bd - wd

    # Nonlinear king-danger index: attack units over the king zone, quadratic,
    # gated on >= 2 attackers (knight/bishop 2, rook 3, queen 5, heavy-file x-ray 2).
    def zone_danger(king_sq: int, color: chess.Color, own_pawns: int) -> float:
        zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
        enemy = not color
        units = attackers = 0
        for piece_type, weight in ((chess.KNIGHT, 2), (chess.BISHOP, 2), (chess.ROOK, 3), (chess.QUEEN, 5)):
            for sq in scan(board.pieces_mask(piece_type, enemy)):
                if board.attacks_mask(sq) & zone:
                    units += weight
                    attackers += 1
        if attackers < 2:
            return 0.0
        units += max(0, 3 - _shield_pawns(own_pawns, king_sq, color))
        return min(units * units / 16.0, 40.0)

    values["KING_DANGER"] = (zone_danger(b_king, b, b_pawns) - zone_danger(w_king, w, w_pawns)) * mg_w

    # Hanging material: attacked-and-undefended non-king pieces, in pawns.
    hanging = {"white": 0, "black": 0}
    for color in (w, b):
        side = "white" if color == w else "black"
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in scan(board.pieces_mask(piece_type, color)):
                if board.attackers_mask(not color, sq) and not board.attackers_mask(color, sq):
                    hanging[side] += SEE_VALUE[chess.piece_symbol(piece_type).upper()]
    values["HANGING_MATERIAL"] = (hanging["black"] - hanging["white"]) / 100.0

    facts: dict[str, dict] = {}
    for key, thresholds in GEOMETRY_FAMILIES:
        value = values[key]
        bucket = _bucket(abs(value), thresholds)
        facts[key] = {
            "value": round(value, 4),
            "bucket": bucket,
            "favors": "white" if value > 0 else "black" if value < 0 else "balanced",
        }
    return facts


# ChessTempo tactical-motifs taxonomy (chesstempo.com/tactical-motifs, accessed
# 2026-09-14). `static=True` motifs are detected deterministically below; the
# rest are listed so search-time producers can attach them under the same
# taxonomy — the static detector never fabricates them.
MOTIF_TAXONOMY: dict[str, bool] = {
    "fork": True, "pin": True, "skewer": True, "discovered_attack": True,
    "double_attack": True, "hanging_piece": True, "removal_of_defender": True,
    "trapped_piece": True, "back_rank_weakness": True, "mate_in_1": True,
    "promotion_available": True,
    "deflection": False, "decoy": False, "interference": False, "overloading": False,
    "xray_attack": False, "zwischenzug": False, "sacrifice": False, "desperado": False,
    "quiet_move": False, "zugzwang": False, "perpetual_check": False, "smothered_mate": False,
    "king_hunt": False, "clearance": False, "blocking": False, "underpromotion": False,
}

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _attack_targets(board: chess.Board, sq: int) -> list[tuple[int, chess.Piece]]:
    targets = []
    attacks = board.attacks_mask(sq)
    while attacks:
        target = (attacks & -attacks).bit_length() - 1
        attacks &= attacks - 1
        piece = board.piece_at(target)
        if piece and piece.color != board.piece_at(sq).color:
            targets.append((target, piece))
    return targets


def _worth_attacking(piece: chess.Piece) -> bool:
    return piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _slider_rays(board: chess.Board, sq: int) -> list[tuple[int, list[int]]]:
    """Slider rays: (first blocker square, squares beyond) per ray direction."""
    rays = []
    piece = board.piece_at(sq)
    directions: list[tuple[int, int]] = []
    if piece.piece_type in (chess.BISHOP, chess.QUEEN):
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if piece.piece_type in (chess.ROOK, chess.QUEEN):
        directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    f, r = sq & 7, sq >> 3
    for df, dr in directions:
        first, beyond, steps = None, [], 0
        ff, rr = f + df, r + dr
        while 0 <= ff <= 7 and 0 <= rr <= 7:
            target = rr * 8 + ff
            if board.piece_at(target):
                if first is None:
                    first = target
                else:
                    beyond.append(target)
            elif first is not None:
                beyond.append(target)
            ff += df
            rr += dr
        if first is not None:
            rays.append((first, beyond))
    return rays


def extract_motifs(board: chess.Board) -> list[dict]:
    """Deterministic static motif detections (ChessTempo taxonomy subset)."""
    motifs: list[dict] = []
    mover = board.turn

    # Mate in 1 / promotion availability (exact, via legal move enumeration).
    for move in board.legal_moves:
        if board.gives_check(move) and not board.is_castling(move):
            board.push(move)
            if board.is_checkmate():
                motifs.append({"motif": "mate_in_1", "detail": {"by": "white" if mover == chess.WHITE else "black"}})
            board.pop()
            break
    if any(move.promotion for move in board.legal_moves):
        motifs.append({"motif": "promotion_available", "detail": {"by": "white" if mover == chess.WHITE else "black"}})

    for color in (chess.WHITE, chess.BLACK):
        side = "white" if color == chess.WHITE else "black"
        enemy = not color

        # Hanging enemy pieces: attacked by us and undefended by their own side
        # (the engine's hanging_material rule).
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN):
            for sq in scan(board.pieces_mask(piece_type, enemy)):
                if board.attackers_mask(color, sq) and not board.attackers_mask(enemy, sq):
                    motifs.append({"motif": "hanging_piece", "detail": {"side": side, "square": chess.square_name(sq)}})

        # Forks and double attacks: one piece attacking 2+ worth-attacking targets.
        for piece_type in (chess.KNIGHT, chess.PAWN, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in scan(board.pieces_mask(piece_type, color)):
                targets = [t for t, piece in _attack_targets(board, sq) if _worth_attacking(piece)]
                if len(targets) >= 2:
                    motif = "fork" if piece_type in (chess.KNIGHT, chess.PAWN) else "double_attack"
                    motifs.append({"motif": motif, "detail": {
                        "side": side, "piece": chess.piece_symbol(piece_type),
                        "targets": [chess.square_name(t) for t in targets[:4]]}})

        # Pins and skewers: own slider, first blocker, meaningful piece behind.
        for piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in scan(board.pieces_mask(piece_type, color)):
                for first, beyond in _slider_rays(board, sq):
                    blocker = board.piece_at(first)
                    if blocker is None or blocker.color != enemy:
                        continue
                    for target in beyond:
                        behind = board.piece_at(target)
                        if behind is None or behind.color != enemy:
                            continue
                        if behind.piece_type == chess.KING:
                            motifs.append({"motif": "pin", "detail": {
                                "side": side, "piece": chess.piece_symbol(piece_type),
                                "pinned": chess.square_name(first)}})
                        elif blocker.piece_type == chess.KING:
                            break
                        elif PIECE_ORDER.index(blocker.piece_type) > PIECE_ORDER.index(behind.piece_type):
                            motifs.append({"motif": "skewer", "detail": {
                                "side": side, "piece": chess.piece_symbol(piece_type),
                                "front": chess.square_name(first), "behind": chess.square_name(target)}})
                        break

        # Discovered attack: own slider behind own blocker aiming at an enemy target.
        for piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in scan(board.pieces_mask(piece_type, color)):
                for first, beyond in _slider_rays(board, sq):
                    blocker = board.piece_at(first)
                    if blocker is None or blocker.color != color:
                        continue
                    for target in beyond:
                        behind = board.piece_at(target)
                        if behind is not None and behind.color == enemy and _worth_attacking(behind):
                            motifs.append({"motif": "discovered_attack", "detail": {
                                "side": side, "slider": chess.square_name(sq),
                                "blocker": chess.square_name(first), "target": chess.square_name(target)}})
                        break

        # Removal of defender: an enemy defender we can profitably capture guards a target.
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for defender_sq in scan(board.pieces_mask(piece_type, enemy)):
                defended = board.attacks_mask(defender_sq)
                if not defended:
                    continue
                attackers = board.attackers_mask(color, defender_sq)
                if not attackers:
                    continue
                cheapest = min((board.piece_at(sq).piece_type for sq in
                                iter_bits(attackers) if board.piece_at(sq)),
                               default=None, key=lambda pt: PIECE_ORDER.index(pt))
                if cheapest is None or PIECE_ORDER.index(cheapest) > PIECE_ORDER.index(piece_type):
                    continue
                for guarded in iter_bits(defended):
                    victim = board.piece_at(guarded)
                    if victim and victim.color == enemy and victim.piece_type != chess.KING:
                        if board.attackers_mask(enemy, guarded) and not board.attackers_mask(color, guarded):
                            continue
                        motifs.append({"motif": "removal_of_defender", "detail": {
                            "side": side, "defender": chess.square_name(defender_sq),
                            "target": chess.square_name(guarded)}})
                        break

        # Trapped piece: enemy minor with no safe move.
        for piece_type in (chess.KNIGHT, chess.BISHOP):
            for sq in scan(board.pieces_mask(piece_type, enemy)):
                if board.attackers_mask(color, sq) == 0:
                    continue
                safe = 0
                escapes = board.attacks_mask(sq) & ~board.occupied_co[enemy]
                while escapes:
                    escape = (escapes & -escapes).bit_length() - 1
                    escapes &= escapes - 1
                    if not board.attackers_mask(color, escape):
                        safe += 1
                if safe == 0:
                    motifs.append({"motif": "trapped_piece", "detail": {
                        "side": side, "square": chess.square_name(sq)}})

        # Back-rank weakness: a settled king on its home rank, trapped by its own
        # army (<=1 safe flight square), while the enemy keeps a heavy piece that
        # could deliver the back-rank blow. Excludes un-castled d/e-file kings so
        # opening positions are not flagged.
        king_sq = board.king(color)
        if king_sq is not None:
            home = 0 if color == chess.WHITE else 7
            castled_file = (king_sq & 7) in (0, 1, 2, 5, 6, 7)
            heavy = board.pieces_mask(chess.ROOK, enemy) | board.pieces_mask(chess.QUEEN, enemy)
            if king_sq >> 3 == home and castled_file and heavy:
                flights = 0
                escapes = chess.BB_KING_ATTACKS[king_sq] & ~board.occupied_co[color]
                while escapes:
                    escape = (escapes & -escapes).bit_length() - 1
                    escapes &= escapes - 1
                    if not board.attackers_mask(enemy, escape):
                        flights += 1
                if flights <= 1 and _shield_pawns(board.pawns & board.occupied_co[color], king_sq, color) >= 2:
                    motifs.append({"motif": "back_rank_weakness", "detail": {"side": side}})

    seen: set[str] = set()
    unique = []
    for motif in motifs:
        key = motif["motif"] + ":" + str(sorted(motif.get("detail", {}).items()))
        if key not in seen:
            seen.add(key)
            unique.append(motif)
    return unique


def iter_bits(bitboard: int):
    while bitboard:
        sq = (bitboard & -bitboard).bit_length() - 1
        bitboard &= bitboard - 1
        yield sq


def extract_strategies(board: chess.Board) -> dict[str, object]:
    """Coarse strategic tags along the lab's research axes."""
    w, b = chess.WHITE, chess.BLACK
    center = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
    center_delta = 0
    for sq in iter_bits(center):
        for color, sign in ((w, 1), (b, -1)):
            if board.occupied_co[color] & chess.BB_SQUARES[sq]:
                center_delta += sign
            if board.attackers_mask(color, sq):
                center_delta += sign
    development = 0
    for color, sign in ((w, 1), (b, -1)):
        home = chess.BB_RANKS[0] if color == w else chess.BB_RANKS[7]
        minor = board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)
        queen_off_home = bool(board.pieces_mask(chess.QUEEN, color)) and not (
            board.pieces_mask(chess.QUEEN, color) & home)
        development += sign * (minor & ~home).bit_count() + sign * (1 if queen_off_home else 0)
    material = 0
    for piece_type, value in SEE_VALUE.items():
        pt = chess.PIECE_SYMBOLS.index(piece_type.lower())
        material += value * (board.pieces_mask(pt, w).bit_count() - board.pieces_mask(pt, b).bit_count())
    phase_units = min(sum(board.pieces_mask(chess.PIECE_SYMBOLS.index(symbol.lower()), color).bit_count()
                          * PHASE_VALUE[symbol.upper()]
                          for symbol in PHASE_VALUE for color in (w, b)), MAX_PHASE)
    return {
        "material_balance_cp": material,
        "center_control": center_delta,
        "development": development,
        "phase_weight": round(phase_units / MAX_PHASE, 3),
    }


def analyze(fen: str) -> dict:
    """Full per-record CVS analysis: geometry facts, motifs, strategy tags."""
    board = chess.Board(fen)
    return {
        "registry_version": CVS_FACTS_REGISTRY_VERSION,
        "registry_hash": facts_registry_hash(),
        "geometry": extract_geometry(board),
        "motifs": extract_motifs(board),
        "strategy": extract_strategies(board),
    }


def label_facts(store, normalization_id: str, *, registry_version: int = CVS_FACTS_REGISTRY_VERSION,
                note: str = "", batch: int = 2000) -> dict:
    """Compute CVS analysis facts for every record in a normalization and register them.

    Three append-only label sets are produced (geometry facts, motifs, strategy
    tags) with deterministic authorities. Records already labelled by a previous
    registry version are untouched; re-labelling under a bumped version creates a
    new set and both stay inspectable.
    """
    from .data import register_labels, read_jsonl
    from .hashing import sha256_file
    from .schemas import Normalization
    from .store import Store, TamperError

    norm: Normalization = store.get_as(normalization_id, Normalization)
    if sha256_file(store.abs(norm.path)) != norm.output_hash:
        raise TamperError(f"{norm.id}: canonical records no longer match the normalization output hash")
    detail = note or f"registry {facts_registry_hash()[:24]}"
    rows_facts, rows_motifs, rows_strategy = [], [], []
    for record in read_jsonl(store.abs(norm.path)):
        analysis = analyze(record["fen"])
        rows_facts.append({
            "record_id": record["record_id"], "value": analysis["geometry"],
            "note": detail, "registry_version": registry_version,
        })
        motifs = [m["motif"] for m in analysis["motifs"]]
        if motifs:
            rows_motifs.append({"record_id": record["record_id"], "value": motifs,
                                "note": detail, "registry_version": registry_version})
        if analysis["strategy"]:
            rows_strategy.append({"record_id": record["record_id"], "value": analysis["strategy"],
                                  "note": detail, "registry_version": registry_version})
    geometry = register_labels(store, normalization_id, family="facts", producer=FACTS_PRODUCER,
                               authority=GEOMETRY_AUTHORITY, rows=rows_facts, pov="white",
                               registry_version=registry_version)
    motifs = register_labels(store, normalization_id, family="motif", producer=FACTS_PRODUCER,
                             authority=MOTIF_AUTHORITY, rows=rows_motifs, pov="white",
                             registry_version=registry_version) if rows_motifs else None
    strategy = register_labels(store, normalization_id, family="strategy", producer=FACTS_PRODUCER,
                               authority=STRATEGY_AUTHORITY, rows=rows_strategy, pov="white",
                               registry_version=registry_version)
    return {
        "normalization_id": normalization_id, "records": norm.record_count,
        "facts_label_set": geometry, "motif_label_set": motifs, "strategy_label_set": strategy,
        "records_with_motifs": len(rows_motifs), "registry_hash": facts_registry_hash(),
    }
