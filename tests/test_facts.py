import chess
import pytest

from cvslab import data
from cvslab.facts import (
    GEOMETRY_FAMILIES,
    analyze,
    extract_geometry,
    facts_registry_hash,
    label_facts,
)
from cvslab.store import LabError


@pytest.fixture()
def labelled_corpus(lab):
    """A normalization with all three CVS fact label sets registered."""
    lab.create_fixture_source(n_games=10, seed=4)
    normalization = lab.normalize([lab.store.list("S")[0].id], name="v1")
    lab.norm_id = normalization.id
    return label_facts(lab.store, normalization.id)


def test_geometry_registry_matches_the_engine_contract():
    assert [key for key, _ in GEOMETRY_FAMILIES][:9] == [
        "KING_DANGER", "KING_ZONE_PRESSURE", "KING_OPEN_FILE", "KING_SHIELD",
        "KING_CENTRAL_EXPOSURE", "ENEMY_QUEEN_NEAR_KING", "OPEN_CENTER_KING",
        "KING_ESCAPE_DEFICIT", "HANGING_MATERIAL"]
    assert len(GEOMETRY_FAMILIES) == 21
    assert facts_registry_hash().startswith("sha256:")
    facts = extract_geometry(chess.Board(chess.STARTING_FEN))
    assert set(facts) == {key for key, _ in GEOMETRY_FAMILIES}
    assert all(entry["bucket"] in (0, 1, 2, 3) for entry in facts.values())
    assert facts["BISHOP_PAIR"]["bucket"] == 0
    assert facts["HANGING_MATERIAL"]["bucket"] == 0


def test_start_position_is_quiet():
    analysis = analyze(chess.STARTING_FEN)
    assert analysis["motifs"] == []
    assert analysis["strategy"]["material_balance_cp"] == 0
    assert analysis["strategy"]["center_control"] == 0


def test_analyze_is_deterministic():
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    assert analyze(fen) == analyze(fen)


def test_motif_detections():
    # an undefended black queen attacked by a white rook on the same file
    hanging = analyze("k7/8/8/8/8/8/R7/q3K3 w - - 0 1")
    motifs = {motif["motif"] for motif in hanging["motifs"]}
    assert "hanging_piece" in motifs

    # white queen d1 attacks h5 rook and b3 pawn along two rays
    double = analyze("k7/8/8/7r/8/8/3P4/K6Q b - - 0 1")
    assert any(motif["motif"] in ("double_attack", "fork", "skewer", "hanging_piece")
               for motif in double["motifs"])

    # every emitted motif belongs to the ChessTempo-reference static subset
    for motif in analyze("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")["motifs"]:
        assert motif["motif"] in {
            "fork", "pin", "skewer", "discovered_attack", "double_attack", "hanging_piece",
            "removal_of_defender", "trapped_piece", "back_rank_weakness", "mate_in_1",
            "promotion_available"}


def test_label_facts_registers_all_three_families(lab, labelled_corpus):
    assert labelled_corpus["records"] > 0
    assert set(labelled_corpus["facts_label_set"].families) == {"facts"}
    assert set(labelled_corpus["facts_label_set"].authorities) == {"deterministic.cvs.geometry.v1"}
    assert labelled_corpus["motif_label_set"] is not None
    assert set(labelled_corpus["motif_label_set"].families) == {"motif"}
    assert set(labelled_corpus["strategy_label_set"].families) == {"strategy"}
    assert labelled_corpus["registry_hash"].startswith("sha256:")


def test_catalog_facts_filters_and_sort(lab, labelled_corpus):
    catalog = lab.catalog(lab.norm_id, sort_by="fact:HANGING_MATERIAL", limit=50)
    buckets = []
    for record in catalog.page:
        facts_label = next(label for label in record.labels if label["family"] == "facts")
        buckets.append(facts_label["value"]["HANGING_MATERIAL"]["bucket"])
    assert buckets == sorted(buckets, reverse=True)

    motifs = catalog.facets["motif"]
    assert motifs  # random-play corpora contain tactical patterns
    top_motif = max(motifs, key=motifs.get)
    by_motif = lab.catalog(lab.norm_id, filters={"motif": top_motif}, limit=500)
    assert by_motif.total_matching == motifs[top_motif]

    assert lab.catalog(lab.norm_id, filters={"fact_key": "HANGING_MATERIAL", "fact_min": 1},
                       limit=500).total_matching > 0
    with pytest.raises(LabError, match="unknown geometry fact"):
        lab.catalog(lab.norm_id, filters={"fact_key": "VIBES"})


def test_frozen_dataset_carries_facts_labels_into_splits(lab, labelled_corpus):
    dataset = lab.freeze_dataset(lab.norm_id, name="with-facts")
    split_rows = data.read_jsonl(lab.store.abs(dataset.splits[0].path))
    assert split_rows
    families = {label["family"] for label in split_rows[0]["labels"]}
    assert {"eval_cp", "facts", "strategy"} <= families
    assert {"facts", "motif", "strategy"} <= {key for ref in dataset.label_sets for key in ref.families}


def test_map_projections_are_read_only_canonical_views(lab):
    lab.result = lab.run_phase0_proof(n_games=8, epochs=1, seeds=(0,))
    research = lab.map_view("research")
    kinds = {node.kind for node in research.nodes}
    assert {"baseline", "ablation", "run", "model", "hypothesis", "finding"} <= kinds
    edges = {(edge.src, edge.dst) for edge in research.edges}
    assert (lab.result["hypothesis"], lab.result["ablation"]) in edges
    promotion = lab.map_view("promotion")
    assert {node.kind for node in promotion.nodes} <= {"baseline", "ablation", "finding"}
    data_view = lab.map_view("data")
    kinds = {node.kind for node in data_view.nodes}
    assert {"source", "normalization", "dataset"} <= kinds
    assert lab.map_view("research", generation="G99").nodes == []
    with pytest.raises(LabError, match="unknown projection"):
        lab.map_view("galaxy")
