"""P1 evidence demonstration: how the model now distinguishes the four things.

Run: python tools/p1_evidence_demo.py   (writes nothing outside a temp store)
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cvslab.service import LabService, protocol_target_spec, recipe_target_spec
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

TRAIN = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer="a" * 64,
                   budget={"nodeBudget": 16000}, value_path=("scoreCpStm",))
EVAL = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep", producer="a" * 64,
                  budget={"nodeBudget": 400000}, value_path=("targets", "scoreCpStm"))
REASON = "arm trained at 16k, measured on the frozen 400k instrument"


def main() -> int:
    lab = LabService(Store(pathlib.Path(tempfile.mkdtemp()) / "labstore"))
    source = lab.create_fixture_source(n_games=6, seed=99)
    normalization = lab.normalize([source.id], name="n")
    from cvslab.data import _load_canonical
    evaluated = sorted(r["record_id"] for r in _load_canonical(lab.store, lab.store.get(normalization.id)))
    lab.register_labels(normalization.id, family="search_shallow_cp", producer="a" * 64,
                        authority="legacy.cvs.search.shallow", pov="stm",
                        rows=[{"record_id": r, "value": {"scoreCpStm": 30 + i, "bestMove": "e2e4", "nodes": 16000},
                               "budget": {"nodeBudget": 16000, "nodes": 16000}} for i, r in enumerate(evaluated)])
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": r, "value": {"targets": {"scoreCpStm": -100 + 10 * i}},
                               "budget": {"nodeBudget": 400000, "nodes": 400000}} for i, r in enumerate(evaluated)])
    train = lab.freeze_dataset(normalization.id, name="d-train-16k", required_labels=["search_shallow_cp"],
                               target_spec=TRAIN)
    exam = lab.freeze_dataset(normalization.id, name="d-exam-400k", required_labels=["search_deep_cp"],
                              target_spec=EVAL, fractions=(1.0, 0.0, 0.0))
    recipe = lab.create_training_recipe(name="t-16k", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 2},
                                        target_spec=TRAIN)
    protocol = lab.create_eval_protocol(name="e-400k", dataset_id=exam.id, split="train", k=256.0, lam=1.0,
                                        target_spec=EVAL)
    print("1. WHAT TEACHES  (recipe", recipe.id, ") train spec :", recipe_target_spec(recipe).spec_hash())
    print("2. WHAT JUDGES  (protocol", protocol.id, ") eval spec  :", protocol_target_spec(protocol).spec_hash())

    print("\n3. DIVERGENCE WITHOUT A DECLARATION IS REFUSED")
    undeclared = lab.register_baseline(name="UNDECLARED", dataset_id=train.id, training_recipe_id=recipe.id,
                                       eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4})
    [run_a] = lab.queue_runs(undeclared.id, seeds=(0,))
    finished = lab.execute_run(run_a.id)
    print("   ", run_a.id, "->", finished.status.value, "|",
          {c.name: c.status for c in finished.integrity if c.name in ("supervision_divergence", "preflight")})
    print("    run records both specs anyway:", run_a.train_target_spec_hash[:19] + "…", "/",
          run_a.eval_target_spec_hash[:19] + "…")

    print("\n4. THE SAME DIVERGENCE, DECLARED BEFORE THE RUN EXISTS, EXECUTES")
    declared = lab.register_baseline(name="DECLARED", dataset_id=train.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                     supervision_divergence=REASON)
    [run_b] = lab.queue_runs(declared.id, seeds=(0,))
    finished_b = lab.execute_run(run_b.id)
    detail = {c.name: c for c in finished_b.integrity}["supervision_divergence"]
    print("   ", run_b.id, "->", finished_b.status.value, "|", detail.status, "|", detail.detail)
    print("    ablation identity changes with the declaration:",
          declared.identity_hash != undeclared.identity_hash)

    print("\n5. COMPARABILITY: one instrument per comparison")
    exam16 = lab.freeze_dataset(normalization.id, name="d-exam-16k", required_labels=["search_shallow_cp"],
                                target_spec=TRAIN, fractions=(1.0, 0.0, 0.0))
    protocol16 = lab.create_eval_protocol(name="e-16k", dataset_id=exam16.id, split="train", k=256.0, lam=1.0,
                                          target_spec=TRAIN)
    arm16 = lab.register_baseline(name="EXAM16", dataset_id=train.id, training_recipe_id=recipe.id,
                                  eval_protocol_id=protocol16.id, model_config={"INPUT": "RAW", "H": 4},
                                  supervision_divergence="cheaper exam")
    [run_c] = lab.queue_runs(arm16.id, seeds=(0,))
    lab.execute_run(run_c.id)
    print("    two completed runs:", run_b.id, "(exam 400k)", run_c.id, "(exam 16k)")
    try:
        lab.assert_runs_comparable([run_b.id, run_c.id])
        print("    UNEXPECTED: they compared")
    except LabError as exc:
        print("    refused:", exc)
    [run_d] = lab.queue_runs(declared.id, seeds=(1,))
    lab.execute_run(run_d.id)
    print("    same instrument, different seeds ->", lab.assert_runs_comparable([run_b.id, run_d.id])[:19] + "…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
