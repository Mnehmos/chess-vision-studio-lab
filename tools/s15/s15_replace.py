"""S15 replacement: X0013 preserved as superseded, X0014 frozen under the split-stream contract.

    python tools/s15/s15_replace.py supersede   # the record + preregistration v2
    python tools/s15/s15_replace.py experiment  # freeze the replacement lattice
    python tools/s15/s15_replace.py run         # execute it
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s15_common import (ABLATION_MODES, ABLATION_WIDTH, BATCH, DATASET, EXAM_PROTOCOL, LAB,
                        MAX_UPDATES, SEEDS, S15, WIDTHS, aux_manifest, freeze_hash, save, state,
                        width_ladder)
import s15_run

sys.path.insert(0, str(LAB))

from cvslab.schemas import Experiment
from cvslab.service import LabService
from cvslab.store import LabError, Store

SUPERSESSION = S15 / "s15-x0013-supersession.json"
PREREG_V2 = S15 / "s15-preregistration-v2.json"
PREREG_V2_HASH = S15 / "s15-preregistration-v2.hash"


def step_supersede() -> int:
    if SUPERSESSION.is_file():
        print("[cached] supersede")
        return 0
    service = LabService(Store(str(LAB / "labstore")))
    x0013: Experiment = service.store.get_as("X0013", Experiment)
    supersession = {
        "id": "s15-x0013-supersession",
        "experiment": "X0013",
        "sealed_record_hash": x0013.record_hash,
        "preserved_not_rewritten": "X0013 stays sealed and byte-identical; this record and the "
                                  "replacement experiment are additive",
        "defect": {
            "what": "one generator served the persistent weights, the auxiliary-head weights AND the "
                    "minibatch sampler: np.random.default_rng(run.seed) was handed to "
                    "RawNnue.initialize, the auxiliary draws advanced it, and the same object was "
                    "then given to nnue.train for shuffling",
            "consequence": "paired RAW-SCORE and AUX-GEO runs at one seed had identical persistent "
                           "starting weights but DIFFERENT minibatch orders, so the comparison was "
                           "'score-only loss + order A' versus 'auxiliary loss + order B' rather "
                           "than 'same examples in the same order + different loss'",
            "why_it_matters": "the measured effects are 0.001-0.002; a different stochastic "
                              "training path is a plausible contributor at that scale",
            "invalidates": ["the causal attribution of the S15 primary contrasts to the auxiliary "
                            "loss alone"],
            "does_not_invalidate": ["the measurements themselves: both arms trained and were "
                                    "examined exactly as recorded",
                                    "the parametric accounting, throughput and gap numbers"]
        },
        "repair": {
            "streams": "np.random.SeedSequence(seed).spawn(3) -> (persistent init, auxiliary init, "
                       "minibatch sampling); the sampler stream is never touched by any "
                       "initialisation draw",
            "sampler": "the fixed-update sampler is now nnue.batch_stream, shared by train and by "
                       "the regression test, so the tested sampler IS the training sampler",
            "tests": ["tests/test_nnue_aux.py::test_paired_aux_and_control_arms_share_init_and_batch_order "
                      "(identical persistent weights and identical first 20 batches across paired arms)",
                      "tests/test_nnue_aux.py::test_the_sampler_stream_is_independent_of_other_draws"]
        },
        "replacement": {"experiment": "frozen by tools/s15/s15_replace.py under "
                                      "tools/s15/s15-preregistration-v2.json",
                        "design": "unchanged: 2 arms x 4 widths x 20 seeds, same teacher/exam "
                                  "contract, same auxiliary manifest and AUX_WEIGHT"},
        "doc": "docs/S15_AUXILIARY_GEOMETRY.md",
    }
    digest = freeze_hash(supersession)
    SUPERSESSION.write_text(json.dumps(supersession, indent=1, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    (S15 / "s15-x0013-supersession.hash").write_text(digest + "\n", encoding="utf-8")
    save(supersession={"file": "tools/s15/s15-x0013-supersession.json", "hash": digest})
    print(f"supersession record {digest}", flush=True)
    return 0


def step_preregister_v2() -> int:
    if PREREG_V2.is_file():
        print("[cached] preregister v2")
        return 0
    v1 = json.loads((S15 / "s15-preregistration.json").read_text(encoding="utf-8"))
    calibration = json.loads((S15 / "s15-aux-weight-calibration.json").read_text(encoding="utf-8"))
    v2 = {**v1,
          "version": 2,
          "supersedes": "X0013 (preserved, sealed; its arms did not share a minibatch stream — see "
                        "tools/s15/s15-x0013-supersession.json)",
          "random_streams": {
              "contract": "np.random.SeedSequence(run.seed).spawn(3) -> persistent init, auxiliary "
                          "init, minibatch sampling; no stream is shared and no draw advances another",
              "consequence": "paired arms of a seed start from identical persistent weights and see "
                             "identical example orders; the only difference between them is the loss",
              "test": "tests/test_nnue_aux.py::test_paired_aux_and_control_arms_share_init_and_batch_order",
              "inference_note": "this is a training-side contract only; deployed models are unchanged"},
          "unchanged_from_v1": ["teacher/exam contract", "auxiliary manifest",
                                "AUX_WEIGHT calibration rule and value", "width ladder",
                                "primary analysis", "family-ablation trigger rule"]}
    digest = freeze_hash(v2)
    PREREG_V2.write_text(json.dumps(v2, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    PREREG_V2_HASH.write_text(digest + "\n", encoding="utf-8")
    save(preregistration_v2={"file": "tools/s15/s15-preregistration-v2.json", "hash": digest})
    print(f"preregistration v2 {digest}", flush=True)
    return 0


def step_experiment() -> int:
    if "experiment_v2" in state():
        print("[cached] experiment v2")
        return 0
    service = s15_run.svc()
    weight = float(state()["calibration"]["aux_weight"])
    conditions = ([(f"RAW-SCORE", h, "none", 0.0) for h in WIDTHS]
                  + [(f"AUX-GEO", h, "geo", weight) for h in WIDTHS])
    original = s15_run.PREREG_HASH
    try:
        s15_run.PREREG_HASH = PREREG_V2_HASH          # the replacement binds the v2 hash
        xid, record = s15_run._freeze(
            service, name="S15 auxiliary geometry supervision (split RNG streams)",
            conditions=conditions,
            notes="2 arms x 4 widths x 20 seeds; identical stream discipline for both arms")
    finally:
        s15_run.PREREG_HASH = original
    save(xid_v2=xid, experiment_v2=record)
    print(f"experiment {record['id']}: {record['arms']} arms, {record['cells']} cells", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = s15_run.svc()
    queued = st.get("queued_v2")
    if not queued:
        experiment = service.store.get_as(st["xid_v2"], Experiment)
        queued = {ablation_id: [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
                  for arm in experiment.arms for ablation_id in arm.ablations}
        save(queued_v2=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    return s15_run._execute(service, st["xid_v2"], st["experiment_v2"]["cells"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["supersede", "preregister", "experiment", "run"])
    args = parser.parse_args()
    return {"supersede": step_supersede, "preregister": step_preregister_v2,
            "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
