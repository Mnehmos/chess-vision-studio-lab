"""Lab service: the single implementation of the lab's scientific logic.

The CLI and HTTP API are thin shells over this module and the GUI only talks to
the API. Derived facts — parameter counts, diffs, hashes, integrity checks and
result states — are always recomputed here from canonical objects, never
accepted from a caller.
"""
from __future__ import annotations

import json
import math
import logging
import os
import platform
import re
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import data, facts, families, intake, nnue
from .hashing import canonical_json, hash_obj, sha256_file
from .schemas import (
    KINDS,
    TERMINAL_RUN_STATUSES,
    Ablation,
    AblationPreview,
    ArmSummary,
    Compute,
    ComputeTotals,
    Dataset,
    Effect,
    EpochLog,
    EvaluationProtocol,
    EvidenceState,
    Finding,
    FunnelRun,
    Hypothesis,
    IntegrityCheck,
    IntakeRecord,
    LegacyEvidence,
    LegacyEvaluation,
    LegacyModel,
    LegacyModelPoint,
    LegacySource,
    LegacySummary,
    MatrixCell,
    MatrixResponse,
    MatrixRow,
    MetricValue,
    ModelArtifact,
    Normalization,
    Overview,
    PromotionGate,
    PromotionState,
    Run,
    RunEnvironment,
    RunStatus,
    ScalingPoint,
    ScalingResponse,
    SearchBacklog,
    SourceSnapshot,
    StateChange,
    TrainingRecipe,
    make_check,
    utc_now,
)
from .store import LabError, NotFound, Store, TamperError

log = logging.getLogger("cvslab")

LAB_REPO = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY = "NNUE"
TRAINER_ID = "cvslab.nnue.raw_numpy"
TRAINER_VERSION = 1
RECIPE_AXES = ("training", "signal", "compute")
PROMOTION_PHASE0 = ("NOT_ELIGIBLE — Phase 0 static-evaluation evidence only; no fixed-node, fixed-time or "
                    "game-gate instrument exists yet, so no promotion decision can follow from this finding.")
# Gates a finding would have to pass before any promotion decision; each is named
# here so the GUI can show exactly what is missing rather than a bare refusal.
PROMOTION_GATES = (
    ("fixed_node_benchmark", "no fixed-node benchmark instrument is registered in Phase 0"),
    ("fixed_time_benchmark", "no fixed-time benchmark instrument is registered in Phase 0"),
    ("game_gate_sprt", "no cutechess/SPRT game gate is registered in Phase 0"),
)
DEFAULT_NON_CLAIMS = [
    "Static-evaluation fit on a frozen held-out split is not playing strength: this protocol runs no search, "
    "fixed-node comparison or game gate.",
    "The result holds only for this frozen dataset distribution and label authority; it does not transfer to other "
    "corpora or labels without a new experiment.",
]


def recipe_target_spec(recipe: TrainingRecipe):
    """The frozen TRAINING TargetSpec a recipe pins (canonical JSON in its params)."""
    from .targets import TargetSpec
    raw = recipe.params.get("TARGET_SPEC")
    if not raw:
        return None
    payload = json.loads(str(raw))
    return TargetSpec(family=payload["family"], authority=payload["authority"],
                      producer=payload["producer"], budget=payload.get("budget") or {},
                      value_path=tuple(payload.get("value_path") or ["value"]),
                      pov=payload.get("pov", "stm"), target_type=payload.get("target_type", "cp"),
                      k=float(payload.get("k", 256.0)), lam=float(payload.get("lam", 1.0)))


def protocol_target_spec(protocol: EvaluationProtocol):
    """The frozen EVALUATION TargetSpec a protocol pins (stored as canonical JSON in its params)."""
    from .targets import TargetSpec
    raw = protocol.params.get("TARGET_SPEC")
    if not raw:
        return None
    payload = json.loads(str(raw))
    return TargetSpec(family=payload["family"], authority=payload["authority"],
                      producer=payload["producer"], budget=payload.get("budget") or {},
                      value_path=tuple(payload.get("value_path") or ["value"]),
                      pov=payload.get("pov", "stm"), target_type=payload.get("target_type", "cp"),
                      k=float(payload.get("k", 256.0)), lam=float(payload.get("lam", 1.0)))


def recipe_hash(recipe: TrainingRecipe) -> str:
    return hash_obj(recipe.model_dump(mode="json", exclude={"id", "created_at", "recipe_hash"}))


def protocol_hash(protocol: EvaluationProtocol) -> str:
    return hash_obj(protocol.model_dump(mode="json", exclude={"id", "created_at", "protocol_hash"}))


def _git(repo: Optional[str | Path], *args: str) -> Optional[str]:
    if not repo:
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def capture_environment(engine_repo: Optional[str]) -> RunEnvironment:
    lab_commit = _git(LAB_REPO, "rev-parse", "HEAD")
    status = _git(LAB_REPO, "status", "--porcelain") if lab_commit else None
    return RunEnvironment(
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        machine=platform.machine(), processor=platform.processor() or "unknown", cpu_count=os.cpu_count() or 0,
        lab_commit=lab_commit, lab_dirty=None if status is None else bool(status),
        engine_repo=engine_repo, engine_commit=_git(engine_repo, "rev-parse", "HEAD"),
    )


def _failed(checks: Sequence[IntegrityCheck]) -> list[IntegrityCheck]:
    return [c for c in checks if c.status == "fail"]


class _PreflightFailed(Exception):
    pass


class LabService:
    def __init__(self, store: Store):
        self.store = store
        self._queue_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ config

    def config(self) -> dict:
        return {"active_generation": "G01", "engine_repo": None, **self.store.read_config()}

    def init_lab(self, *, generation: Optional[str] = None, engine_repo: Optional[str] = None) -> dict:
        config = self.config()
        if generation:
            if not re.fullmatch(r"G\d{2,}", generation):
                raise LabError("generation must look like G01")
            config["active_generation"] = generation
        if engine_repo:
            path = Path(engine_repo).resolve()
            if _git(path, "rev-parse", "HEAD") is None:
                raise LabError(f"{path} is not a git repository with a commit; the engine must be pinnable")
            config["engine_repo"] = str(path)
        self.store.write_config(config)
        return config

    def _generation(self, generation: Optional[str]) -> str:
        return generation or self.config()["active_generation"]

    # ------------------------------------------------------------------ data lineage

    def create_fixture_source(self, **kwargs):
        return data.create_fixture_source(self.store, **kwargs)

    def import_jsonl_source(self, path: str, **kwargs):
        return data.import_jsonl_source(self.store, path, **kwargs)

    def import_pgn_source(self, path: str, **kwargs):
        return data.import_pgn_source(self.store, path, **kwargs)

    def normalize(self, source_ids: Sequence[str], **kwargs):
        return data.normalize(self.store, source_ids, **kwargs)

    def freeze_dataset(self, normalization_id: str, **kwargs):
        return data.freeze_dataset(self.store, normalization_id, **kwargs)

    def register_labels(self, normalization_id: str, **kwargs):
        """Append one L2 label set; records and earlier labels are never modified."""
        return data.register_labels(self.store, normalization_id, **kwargs)

    def copy_label_sets(self, from_normalization_id: str, to_normalization_id: str) -> dict[str, int]:
        return data.copy_label_sets(self.store, from_normalization_id, to_normalization_id)

    def catalog(self, normalization_id: str, *, filters: Optional[Mapping[str, object]] = None,
                offset: int = 0, limit: int = 50, sort_by: Optional[str] = None):
        return data.catalog(self.store, normalization_id, filters=filters, offset=offset, limit=limit,
                            sort_by=sort_by)

    def label_facts(self, normalization_id: str, **kwargs) -> dict:
        """Attach deterministic CVS analysis facts (geometry, motifs, strategy) per record."""
        return facts.label_facts(self.store, normalization_id, **kwargs)

    def stack_preview(self, normalization_id: str, arms: Sequence[Mapping[str, object]]):
        return data.stack_preview(self.store, normalization_id, arms)

    def migrate(self, from_normalization_id: str, *, name: str, settings_overrides: Optional[Mapping[str, object]] = None,
                dedup: str = "exact-epd-keep-first") -> Normalization:
        """Re-standardize from the same immutable sources under new settings: a new N####, never a rewrite."""
        previous: Normalization = self.store.get_as(from_normalization_id, Normalization)
        return data.normalize(self.store, previous.source_ids, name=name, dedup=dedup,
                              settings_overrides=settings_overrides)

    def migration_diff(self, old_normalization_id: str, new_normalization_id: str):
        diff = data.migration_diff(self.store, old_normalization_id, new_normalization_id)
        diff.affected_run_ids = [r.id for r in self.store.list("R", verify=False, kind=Run)
                                 if r.dataset_id in diff.affected_dataset_ids]
        return diff

    def rebuild_dataset(self, dataset_id: str, new_normalization_id: str) -> Dataset:
        return data.rebuild_dataset(self.store, dataset_id, new_normalization_id)

    def import_legacy_catalog(self, catalog_dir: str, **kwargs) -> IntakeRecord:
        """Bring the pre-lab engine/app inventory in as LEGACY evidence; never inherited, never promotable."""
        return intake.import_legacy_catalog(self.store, catalog_dir, **kwargs)

    def import_funnel_run(self, run_dir: str, **kwargs) -> FunnelRun:
        return intake.import_funnel_run(self.store, run_dir, **kwargs)

    # ------------------------------------------------------------------ recipes and protocols

    def create_training_recipe(self, *, name: str, params: Optional[Mapping[str, object]] = None,
                               family: str = DEFAULT_FAMILY, generation: Optional[str] = None,
                               description: str = "", target_spec=None) -> TrainingRecipe:
        registry = families.switch_map(family)
        params = dict(params or {})
        families.check_registered(family, params)
        not_recipe = sorted(k for k in params if registry[k].axis not in RECIPE_AXES)
        if not_recipe:
            raise LabError(f"{', '.join(not_recipe)} belong to the model configuration, not the training recipe")
        coerced = {sw.key: families.coerce(sw, params.get(sw.key, sw.default))
                   for sw in families.switches(family) if sw.axis in RECIPE_AXES}
        if target_spec is not None:
            # the recipe's K/LAMBDA must be the spec's: hashing one transformation while
            # executing another is exactly the ambiguity the spec exists to remove
            if abs(float(coerced["K"]) - target_spec.k) > 1e-9 or abs(float(coerced["LAMBDA"]) - target_spec.lam) > 1e-9:
                raise LabError(f"recipe K/LAMBDA ({coerced['K']}/{coerced['LAMBDA']}) do not match the "
                               f"TargetSpec ({target_spec.k}/{target_spec.lam})")
            coerced["TARGET_SPEC"] = json.dumps(target_spec.canonical(), sort_keys=True)
        recipe = TrainingRecipe(id=self.store.next_id("T"), name=name, family=family,
                                generation=self._generation(generation), trainer=TRAINER_ID,
                                trainer_version=TRAINER_VERSION, params=coerced, description=description,
                                recipe_hash="", created_at=utc_now())
        recipe.recipe_hash = recipe_hash(recipe)
        return self.store.create(recipe)

    def create_eval_protocol(self, *, name: str, dataset_id: str, family: str = DEFAULT_FAMILY,
                             generation: Optional[str] = None, split: str = "test", bootstrap_samples: int = 1000,
                             bootstrap_seed: int = 0, k: float = 256.0, lam: float = 1.0, description: str = "",
                             non_claims: Optional[list[str]] = None,
                             target_spec=None) -> EvaluationProtocol:
        dataset: Dataset = self.store.get_as(dataset_id, Dataset)
        if split not in {s.name for s in dataset.splits}:
            raise LabError(f"{dataset_id} has no split {split!r}")
        if target_spec is not None and (
                abs(float(k) - target_spec.k) > 1e-9 or abs(float(lam) - target_spec.lam) > 1e-9):
            raise LabError(f"protocol K/LAMBDA ({k}/{lam}) do not match the TargetSpec "
                           f"({target_spec.k}/{target_spec.lam})")
        if bootstrap_samples < 100:
            raise LabError("bootstrap_samples must be at least 100")
        protocol = EvaluationProtocol(
            id=self.store.next_id("E"), name=name, family=family, generation=self._generation(generation),
            dataset_id=dataset.id, dataset_manifest_hash=dataset.manifest_hash, split=split,
            metrics=nnue.METRIC_DEFS,
            params={"K": float(k), "LAMBDA": float(lam),
                    **({"TARGET_SPEC": json.dumps(target_spec.canonical(), sort_keys=True)}
                       if target_spec is not None else {})},
            bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed, protocol_version=1,
            non_claims=non_claims or DEFAULT_NON_CLAIMS, description=description, protocol_hash="",
            created_at=utc_now(),
        )
        protocol.protocol_hash = protocol_hash(protocol)
        return self.store.create(protocol)

    # ------------------------------------------------------------------ hypotheses

    def create_hypothesis(self, *, title: str, statement: str, metric: str = "test_loss",
                          predicted_direction: str = "decrease", min_effect: float = 0.0,
                          family: str = DEFAULT_FAMILY, generation: Optional[str] = None,
                          tags: Sequence[str] = ()) -> Hypothesis:
        families.switches(family)
        known = {m.name for m in nnue.METRIC_DEFS}
        if metric not in known:
            raise LabError(f"unknown metric {metric!r}; registered metrics: {', '.join(sorted(known))}")
        now = utc_now()
        return self.store.create(Hypothesis(
            id=self.store.next_id("H"), title=title, statement=statement, family=family,
            generation=self._generation(generation), metric=metric, predicted_direction=predicted_direction,
            min_effect=min_effect, tags=list(tags), state=EvidenceState.PROPOSED,
            state_history=[StateChange(state=EvidenceState.PROPOSED, at=now, reason="created")], created_at=now,
        ))

    # ------------------------------------------------------------------ ablations

    def _frozen_inputs(self, dataset_id: str, recipe_id: str, protocol_id: str, family: str, generation: str):
        dataset: Dataset = self.store.get(dataset_id)
        recipe: TrainingRecipe = self.store.get_as(recipe_id, TrainingRecipe)
        protocol: EvaluationProtocol = self.store.get_as(protocol_id, EvaluationProtocol)
        for obj in (recipe, protocol):
            if (obj.family, obj.generation) != (family, generation):
                raise LabError(f"{obj.id} belongs to {obj.family}-{obj.generation}, not {family}-{generation}")
        # The protocol owns its evaluation dataset and may point at a COMMON instrument
        # shared by several training datasets (S6 evaluates every arm on one D_EVAL).
        eval_dataset: Dataset = self.store.get_as(protocol.dataset_id, Dataset)
        if eval_dataset.manifest_hash != protocol.dataset_manifest_hash:
            raise LabError(f"{protocol.id} pins {protocol.dataset_manifest_hash} but "
                           f"{eval_dataset.id} records {eval_dataset.manifest_hash}")
        return dataset, recipe, protocol

    @staticmethod
    def _identity_hash(family, generation, config, dataset, recipe, protocol,
                       supervision_divergence: str = "", experiment_id: Optional[str] = None) -> str:
        """Identity of one intervention.

        A declared supervision divergence is part of that identity: the contract "this arm
        trains against spec A and is measured against spec B" must exist before the run
        does, and cannot be added once results are visible. An UNDECLARED ablation hashes
        exactly as it did before the declaration existed, so every historical A#### keeps
        the identity it was created with.
        """
        payload = {"family": family, "generation": generation, "effective_config": config,
                   "dataset_manifest_hash": dataset.manifest_hash, "recipe_hash": recipe.recipe_hash,
                   "protocol_hash": protocol.protocol_hash}
        if supervision_divergence:
            payload["supervision_divergence"] = supervision_divergence
        if experiment_id:
            payload["experiment_id"] = experiment_id
        return hash_obj(payload)

    def _new_ablation(self, *, family, generation, baseline_name, is_baseline, control_id, hypothesis_id, config,
                      diff, dataset, recipe, protocol, notes, supervision_divergence: str = "",
                      experiment_id: Optional[str] = None) -> Ablation:
        ablation_id = self.store.next_id("A")
        shapes = families.parameter_shapes(family, config)
        n_params = families.count_parameters(shapes)
        now = utc_now()
        return self.store.create(Ablation(
            id=ablation_id, family=family, generation=generation, baseline_name=baseline_name,
            is_baseline=is_baseline, control_id=control_id, hypothesis_id=hypothesis_id,
            overrides={d.key: d.value for d in diff}, effective_config=config, config_hash=hash_obj(config),
            identity_hash=self._identity_hash(family, generation, config, dataset, recipe, protocol,
                                              supervision_divergence, experiment_id), diff=diff,
            param_count=n_params, parameter_shapes=shapes,
            display_label=families.display_label(ablation_id, family, generation, n_params, baseline_name,
                                                 [(d.key, d.value) for d in diff]),
            dataset_id=dataset.id, training_recipe_id=recipe.id, eval_protocol_id=protocol.id, notes=notes,
            supervision_divergence=supervision_divergence, experiment_id=experiment_id,
            state=EvidenceState.PROPOSED, state_history=[StateChange(state=EvidenceState.PROPOSED, at=now,
                                                                     reason="created")],
            created_at=now,
        ))

    def register_baseline(self, *, name: str, dataset_id: str, training_recipe_id: str, eval_protocol_id: str,
                          model_config: Optional[Mapping[str, object]] = None, family: str = DEFAULT_FAMILY,
                          generation: Optional[str] = None, notes: str = "",
                          supervision_divergence: str = "",
                          experiment_id: Optional[str] = None) -> Ablation:
        name = name.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise LabError("baseline names are upper-case identifiers such as RAW or HYBRID")
        generation = self._generation(generation)
        for existing in self.store.list("A"):
            if existing.is_baseline and (existing.family, existing.generation, existing.baseline_name) == (family, generation, name):
                raise LabError(f"baseline {name} is already frozen as {existing.id} in {family}-{generation}; "
                               "a different control needs a new name or a new generation")
        dataset, recipe, protocol = self._frozen_inputs(dataset_id, training_recipe_id, eval_protocol_id, family, generation)
        model_config = dict(model_config or {})
        families.check_registered(family, model_config)
        in_recipe = sorted(k for k in model_config if families.switch_map(family)[k].axis in RECIPE_AXES)
        if in_recipe:
            raise LabError(f"{', '.join(in_recipe)} are fixed by {recipe.id}; register a different training recipe instead")
        config = families.merge_config(family, {**recipe.params, **model_config}, {})
        return self._new_ablation(family=family, generation=generation, baseline_name=name, is_baseline=True,
                                  control_id=None, hypothesis_id=None, config=config, diff=[], dataset=dataset,
                                  recipe=recipe, protocol=protocol, notes=notes,
                                  supervision_divergence=supervision_divergence,
                                  experiment_id=experiment_id)

    def preview_ablation(self, baseline_id: str, overrides: Mapping[str, object], *,
                         supervision_divergence: str = "") -> AblationPreview:
        base = self.store.get(baseline_id)
        if not isinstance(base, Ablation) or not base.is_baseline:
            raise LabError(f"{baseline_id} is not a frozen baseline; ablations are defined against a registered control")
        registry = families.switch_map(base.family)
        families.check_registered(base.family, overrides)
        coerced = {key: families.coerce(registry[key], value) for key, value in overrides.items()}
        warnings = [f"{key}={families.format_value(value)} equals the baseline and is not a deviation."
                    for key, value in coerced.items() if base.effective_config.get(key) == value]
        config = families.merge_config(base.family, base.effective_config, coerced)
        diff = families.diff_configs(base.family, base.effective_config, config)
        if not diff:
            warnings.append("No deviation from the baseline: queue another run of the baseline instead.")
        if len(diff) > 1:
            warnings.append(f"Multi-variable intervention ({len(diff)} fields). Prefer one-variable ablations; "
                            "declare interactions explicitly by also testing each change alone.")
        axes = sorted({d.axis for d in diff})
        if len(axes) > 1:
            warnings.append(f"Changes span independent axes ({', '.join(axes)}); their effects cannot be attributed separately.")
        dataset, recipe, protocol = self._frozen_inputs(base.dataset_id, base.training_recipe_id, base.eval_protocol_id,
                                                        base.family, base.generation)
        identity = self._identity_hash(base.family, base.generation, config, dataset, recipe, protocol,
                                       supervision_divergence)
        duplicate = next((a.id for a in self.store.list("A") if a.identity_hash == identity), None)
        if duplicate and diff:
            warnings.append(f"This exact configuration already exists as {duplicate}; queue a new run of it instead.")
        shapes = families.parameter_shapes(base.family, config)
        n_params = families.count_parameters(shapes)
        return AblationPreview(
            baseline_id=base.id, baseline_label=base.display_label, overrides={d.key: d.value for d in diff},
            effective_config=config, diff=diff, param_count=n_params, control_param_count=base.param_count,
            parameter_shapes=shapes,
            display_label=families.display_label(self.store.peek_id("A"), base.family, base.generation, n_params,
                                                 base.baseline_name, [(d.key, d.value) for d in diff]),
            supervision_divergence=supervision_divergence,
            identity_hash=identity, warnings=warnings, duplicate_of=duplicate, can_create=bool(diff) and duplicate is None,
        )

    def create_ablation(self, *, baseline_id: str, overrides: Mapping[str, object], hypothesis_id: Optional[str] = None,
                        notes: str = "", supervision_divergence: str = "",
                        experiment_id: Optional[str] = None) -> Ablation:
        """Create one intervention.

        `supervision_divergence` declares, before any run exists, that this intervention
        intentionally trains against the recipe's TargetSpec and is measured against the
        protocol's. The declaration is stored on the A#### and folded into its identity
        hash, so it can neither be added after results are visible nor silently dropped.
        """
        preview = self.preview_ablation(baseline_id, overrides, supervision_divergence=supervision_divergence)
        if not preview.can_create:
            raise LabError(" ".join(preview.warnings))
        base: Ablation = self.store.get(baseline_id)
        if hypothesis_id:
            hypothesis: Hypothesis = self.store.get(hypothesis_id)
            if (hypothesis.family, hypothesis.generation) != (base.family, base.generation):
                raise LabError(f"{hypothesis_id} belongs to {hypothesis.family}-{hypothesis.generation}")
        dataset, recipe, protocol = self._frozen_inputs(base.dataset_id, base.training_recipe_id, base.eval_protocol_id,
                                                        base.family, base.generation)
        return self._new_ablation(family=base.family, generation=base.generation, baseline_name=base.baseline_name,
                                  is_baseline=False, control_id=base.id, hypothesis_id=hypothesis_id,
                                  config=preview.effective_config, diff=preview.diff, dataset=dataset, recipe=recipe,
                                  protocol=protocol, notes=notes, supervision_divergence=supervision_divergence,
                                  experiment_id=experiment_id or base.experiment_id)

    # ------------------------------------------------------------------ runs

    def queue_runs(self, ablation_id: str, seeds: Sequence[int] = (0,)) -> list[Run]:
        ablation: Ablation = self.store.get(ablation_id)
        dataset, recipe, protocol = self._frozen_inputs(ablation.dataset_id, ablation.training_recipe_id,
                                                        ablation.eval_protocol_id, ablation.family, ablation.generation)
        train_spec = recipe_target_spec(recipe)
        eval_spec = protocol_target_spec(protocol)
        experiment = None
        arm = None
        if ablation.experiment_id:
            experiment = self.store.get(ablation.experiment_id)
            if experiment.status == "SEALED":
                raise LabError(f"{experiment.id} is SEALED; it accepts no further runs")
            arm = self.arm_of(experiment, ablation.id)
            if arm is None:
                raise LabError(f"{ablation.id} is not an arm of {experiment.id}; a run cannot join an "
                               "experiment it was not frozen into")
            problems = self.verify_experiment_arm(arm, dataset, recipe)
            if problems:
                raise LabError(f"{ablation.id}: " + "; ".join(problems))
            allowed = {int(value) for value in (arm.seeds or experiment.seeds)}
            outside = [int(seed) for seed in seeds if int(seed) not in allowed]
            if outside:
                raise LabError(f"seeds {outside} are outside arm {arm.arm_id}'s frozen seed list "
                               f"[{min(allowed)}..{max(allowed)}] ({len(allowed)} seeds); "
                               "an experiment's cells are its design")
        runs = []
        for seed in seeds:
            runs.append(self.store.create(Run(
                id=self.store.next_id("R"), ablation_id=ablation.id, hypothesis_id=ablation.hypothesis_id,
                display_label=ablation.display_label, seed=int(seed), status=RunStatus.QUEUED,
                effective_config=ablation.effective_config, config_hash=ablation.config_hash,
                identity_hash=ablation.identity_hash, param_count=ablation.param_count, dataset_id=dataset.id,
                dataset_manifest_hash=dataset.manifest_hash, training_recipe_id=recipe.id,
                recipe_hash=recipe.recipe_hash, eval_protocol_id=protocol.id, protocol_hash=protocol.protocol_hash,
                eval_dataset_id=protocol.dataset_id, eval_dataset_manifest_hash=protocol.dataset_manifest_hash,
                train_target_spec_hash=(train_spec.spec_hash() if train_spec is not None else None),
                eval_target_spec_hash=(eval_spec.spec_hash() if eval_spec is not None else None),
                supervision_divergence=ablation.supervision_divergence,
                experiment_id=ablation.experiment_id,
                queued_at=utc_now(),
            )))
        self._wake.set()
        return runs

    def execute_run(self, run_id: str) -> Run:
        run: Run = self.store.get_as(run_id, Run)
        if run.status != RunStatus.QUEUED:
            raise LabError(f"{run_id} is {run.status.value}; only QUEUED runs can be executed")
        ablation: Ablation = self.store.get(run.ablation_id)
        run.status, run.started_at = RunStatus.RUNNING, utc_now()
        run.environment = capture_environment(self.config().get("engine_repo"))
        self.store.update_run(run)
        if ablation.state == EvidenceState.PROPOSED:
            self.store.update_state(ablation.id, EvidenceState.RUNNING, f"{run.id} started")

        lines: list[str] = []
        checks: list[IntegrityCheck] = []
        compute = Compute()
        cpu0, wall0 = time.process_time(), time.perf_counter()

        def emit(message: str) -> None:
            lines.append(f"{utc_now()} {message}")

        try:
            dataset: Dataset = self.store.get(run.dataset_id)
            recipe: TrainingRecipe = self.store.get(run.training_recipe_id)
            protocol: EvaluationProtocol = self.store.get(run.eval_protocol_id)
            eval_dataset: Dataset = self.store.get_as(protocol.dataset_id, Dataset)
            checks.extend(data.verify_dataset(self.store, dataset))
            if eval_dataset.id != dataset.id:
                checks.extend(data.verify_dataset(self.store, eval_dataset))
                checks.append(make_check("eval_dataset_pinned",
                                         eval_dataset.manifest_hash == protocol.dataset_manifest_hash
                                         == (run.eval_dataset_manifest_hash or protocol.dataset_manifest_hash),
                                         f"evaluation corpus {eval_dataset.id} matches the protocol binding"))
            checks += [
                make_check("dataset_pinned", dataset.manifest_hash == run.dataset_manifest_hash,
                           f"run pinned {run.dataset_manifest_hash}; {dataset.id} records {dataset.manifest_hash}"),
                make_check("config_identity", hash_obj(run.effective_config) == run.config_hash == ablation.config_hash,
                           "effective configuration hashes to the ablation's config hash"),
                make_check("recipe_frozen", recipe_hash(recipe) == recipe.recipe_hash == run.recipe_hash,
                           f"{recipe.id} recomputes to the recipe hash pinned at queue time"),
                make_check("protocol_frozen", protocol_hash(protocol) == protocol.protocol_hash == run.protocol_hash
                           and protocol.dataset_manifest_hash == eval_dataset.manifest_hash,
                           f"{protocol.id} recomputes to its pinned hash and is pinned to {eval_dataset.id}"),
                make_check("param_count_formula",
                           families.param_count(ablation.family, run.effective_config) == run.param_count,
                           f"shape-derived parameter count {run.param_count}"),
            ]
            contract_checks, contract_problems = self.experiment_contract(run)
            checks.extend(contract_checks)
            if contract_problems:
                raise _PreflightFailed("experiment contract violated: " + "; ".join(contract_problems))
            if _failed(checks):
                raise _PreflightFailed("pre-flight integrity checks failed; training was not started")

            cfg = run.effective_config
            # training supervision is pinned by T####; evaluation supervision by E####;
            # the two hashes must agree or the run cannot claim they measure the same thing
            train_spec = recipe_target_spec(recipe)
            eval_spec = protocol_target_spec(protocol)
            checks.append(make_check("eval_dataset_identity",
                                     run.eval_dataset_id == protocol.dataset_id,
                                     f"run pins evaluation corpus {run.eval_dataset_id}; protocol binds "
                                     f"{protocol.dataset_id}"))
            # Supervision identity: the recipe pins what TEACHES the model, the protocol
            # pins what JUDGES it. Both are recorded on the run, and a divergence is
            # permitted only when this ablation declared it at creation time (the
            # declaration is part of the ablation identity hash). The default stays
            # fail-closed: an undeclared divergence is refused exactly as before.
            train_hash = train_spec.spec_hash() if train_spec is not None else None
            eval_hash = eval_spec.spec_hash() if eval_spec is not None else None
            declared = (ablation.supervision_divergence or "").strip()
            if train_spec is not None or eval_spec is not None:
                # The run fields are EVIDENCE; the frozen T#### and E#### objects are AUTHORITY.
                # Execution proves they agree before training starts, so a sealed run can never
                # display a supervision identity the recipe or protocol does not pin.
                def _short(value):
                    return value[:19] + "…" if value else "none"
                recorded_ok = (run.train_target_spec_hash == train_hash
                               and run.eval_target_spec_hash == eval_hash)
                checks.append(make_check(
                    "supervision_authority", recorded_ok,
                    f"run records train {_short(run.train_target_spec_hash)} / eval "
                    f"{_short(run.eval_target_spec_hash)}; {recipe.id} pins {_short(train_hash)} / "
                    f"{protocol.id} pins {_short(eval_hash)}"))
                if not recorded_ok:
                    raise _PreflightFailed(
                        "queued supervision identity does not match the frozen recipe/protocol: "
                        "the T#### and E#### objects are the authority")
                if train_hash is None or eval_hash is None:
                    checks.append(make_check(
                        "supervision_declared", False,
                        f"training TargetSpec {'present' if train_spec else 'MISSING'}; "
                        f"evaluation TargetSpec {'present' if eval_spec else 'MISSING'}"))
                    raise _PreflightFailed("both the training recipe and the evaluation protocol must pin a "
                                           "TargetSpec, or neither may")
                checks.append(make_check(
                    "supervision_divergence", bool(declared) or train_hash == eval_hash,
                    f"train spec {train_hash[:19]}… eval spec {eval_hash[:19]}… — "
                    + ("identical" if train_hash == eval_hash
                       else (f"declared divergence: {declared}" if declared
                             else "divergence NOT declared by the ablation; create it with "
                                  "supervision_divergence='<reason>' to train and measure "
                                  "against different supervision"))))
                if train_hash != eval_hash and not declared:
                    raise _PreflightFailed("training and evaluation TargetSpecs differ")
            train_split, dropped_train = nnue.encode_records(data.load_split(self.store, dataset, "train"),
                                                             target_spec=train_spec)
            val_split, dropped_val = nnue.encode_records(data.load_split(self.store, dataset, "val"),
                                                         target_spec=train_spec)
            eval_split, dropped_eval = nnue.encode_records(
                data.load_split(self.store, eval_dataset, protocol.split), target_spec=eval_spec)
            compute.accepted_examples = len(train_split) + len(val_split) + len(eval_split)
            compute.discarded_examples = dropped_train + dropped_val + dropped_eval
            if not len(train_split) or not len(eval_split):
                raise _PreflightFailed("the training or evaluation split is empty")
            emit(f"{run.display_label} seed={run.seed} train={len(train_split)} val={len(val_split)} "
                 f"{protocol.split}={len(eval_split)} discarded={compute.discarded_examples}")

            rng = np.random.default_rng(run.seed)
            model = nnue.RawNnue.initialize(nnue.INPUT_DIM, int(cfg["H"]), float(cfg["INIT_STD"]),
                                            float(cfg["OUT_SCALE_CP"]), rng)

            def on_epoch(entry: dict) -> None:
                run.training_curve.append(EpochLog(**entry))
                emit(f"epoch {entry['epoch']} train={entry['train_loss']} val={entry['val_loss']} "
                     f"ex/s={entry['examples_per_second']}")

            nnue.train(model, train_split, val_split, cfg, rng, on_epoch=on_epoch)
            # Two regimes, one definition each (the S8 patch briefly redefined the historical
            # one by counting only full batches; see the S9 repair note):
            #   epoch-bounded: every sample in every epoch counts, partial tail batch included
            #   fixed-update:  exactly MAX_UPDATES batches of BATCH samples
            max_updates = int(cfg.get("MAX_UPDATES", 0) or 0)
            batch = int(cfg["BATCH"])
            if max_updates > 0:
                updates = max_updates
                compute.train_examples_seen = updates * batch
                regime = (f"fixed updates: {updates} optimizer updates x batch {batch} = "
                          f"{compute.train_examples_seen} sample presentations")
            else:
                epochs = int(cfg["EPOCHS"])
                updates = epochs * math.ceil(len(train_split) / batch)
                compute.train_examples_seen = epochs * len(train_split)
                regime = (f"fixed epochs: {epochs} passes over {len(train_split)} rows = "
                          f"{compute.train_examples_seen} sample presentations "
                          f"({updates} full batches incl. partial tails)")
            emit(f"student compute: {regime}; "
                 f"{compute.train_examples_seen / max(len(train_split), 1):.2f} effective passes")

            payload = nnue.serialize(model, cfg, {
                "run_id": run.id, "ablation_id": ablation.id, "display_label": run.display_label, "seed": run.seed,
                "dataset_manifest_hash": run.dataset_manifest_hash, "recipe_hash": run.recipe_hash,
            })
            model_rel, model_hash = self.store.write_artifact(run.id, "model.json", json.dumps(payload).encode("utf-8"))
            run.artifact_hashes["model.json"] = model_hash
            stored = self.store.abs(model_rel)
            checks.append(make_check("artifact_hash", sha256_file(stored) == model_hash, f"{model_rel} {model_hash}"))
            reloaded_payload = json.loads(stored.read_bytes())
            serialized_params = nnue.serialized_param_count(reloaded_payload)
            checks.append(make_check("param_count_serialized", serialized_params == run.param_count,
                                     f"serialized artifact holds {serialized_params} learned parameters; "
                                     f"computed {run.param_count}"))
            reloaded = nnue.load_serialized(reloaded_payload)
            checks.append(make_check("finite_weights", reloaded.all_finite(), "all serialized weights are finite"))

            per_position = nnue.evaluate(reloaded, eval_split, protocol.params)
            for metric in protocol.metrics:
                mean, low, high, n = nnue.bootstrap_mean(per_position[metric.name], protocol.bootstrap_samples,
                                                         protocol.bootstrap_seed)
                if mean is None:
                    # the primary metric must be measurable; a conditional metric with no
                    # applicable positions is "not applicable", recorded as a warning
                    checks.append(make_check(f"metric_{metric.name}", False,
                                             "no finite per-position values",
                                             warn=metric.name != "test_loss"))
                    continue
                run.metrics.append(MetricValue(name=metric.name, value=mean, ci_low=low, ci_high=high, n=n,
                                               kind="measured", split=protocol.split))
            eval_payload = {
                "run_id": run.id, "eval_protocol_id": protocol.id, "split": protocol.split,
                "record_ids": eval_split.record_ids,
                "metrics": {name: [float(v) if np.isfinite(v) else None for v in values]
                            for name, values in per_position.items()},
            }
            _, eval_hash = self.store.write_artifact(run.id, "eval.json", canonical_json(eval_payload))
            run.artifact_hashes["eval.json"] = eval_hash

            artifact = self.store.create(ModelArtifact(
                id=self.store.next_id("M"), run_id=run.id, ablation_id=ablation.id, family=ablation.family,
                arch=payload["arch"], format="cvs-raw-nnue-json-v1", path=model_rel, artifact_hash=model_hash,
                param_count=serialized_params, parameter_shapes=nnue.serialized_shapes(reloaded_payload),
                dataset_id=dataset.id, training_recipe_id=recipe.id, promotable=False, engine_loadable=True,
                created_at=utc_now(),
            ))
            run.model_id = artifact.id
        except _PreflightFailed as exc:
            checks.append(make_check("preflight", False, str(exc)))
            emit(str(exc))
        except Exception as exc:  # any crash is recorded as invalid evidence, never silently dropped
            run.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            checks.append(make_check("execution", False, f"run crashed: {run.error}"))
            emit(traceback.format_exc())

        env = run.environment
        checks.append(make_check("lab_code_clean", bool(env and env.lab_dirty is False),
                                 "lab code at a clean commit" if env and env.lab_dirty is False
                                 else "lab code has uncommitted changes or is not a git checkout", warn=True))
        checks.append(make_check("engine_pinned", bool(env and env.engine_commit),
                                 f"engine {env.engine_commit}" if env and env.engine_commit
                                 else "no engine repository configured; this static-eval run executes no engine search",
                                 warn=True))
        compute.cpu_seconds = round(time.process_time() - cpu0, 4)
        compute.wall_seconds = round(time.perf_counter() - wall0, 4)
        run.compute = compute
        run.integrity = checks
        run.status = RunStatus.INVALID if _failed(checks) else RunStatus.COMPLETED
        emit(f"finished {run.status.value}")
        run.log_path, run.artifact_hashes["run.log"] = self.store.write_artifact(run.id, "run.log", "\n".join(lines).encode("utf-8"))
        run.finished_at = utc_now()
        self.store.update_run(run)
        return run

    def work_queue(self) -> list[Run]:
        executed = []
        with self._queue_lock:
            while True:
                queued = [r for r in self.store.list("R", verify=False, kind=Run) if r.status == RunStatus.QUEUED]
                if not queued:
                    return executed
                executed.append(self.execute_run(queued[0].id))

    def start_worker(self, poll_seconds: float = 2.0) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.work_queue()
                except Exception:
                    log.exception("run worker failed")
                self._wake.wait(poll_seconds)
                self._wake.clear()

        self._worker = threading.Thread(target=loop, name="cvslab-worker", daemon=True)
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker:
            self._worker.join(timeout=30)

    # ------------------------------------------------------------------ findings

    # ------------------------------------------------------------ experiments (X####)

    @staticmethod
    def _arm_index(experiment) -> dict:
        return {arm.arm_id: arm for arm in experiment.arms}

    def arm_of(self, experiment, ablation_id: str):
        for arm in experiment.arms:
            if ablation_id in arm.ablations:
                return arm
        return None

    def verify_experiment_arm(self, arm, dataset, recipe) -> list[str]:
        """The arm contract: the dataset, recipe and training supervision it was frozen with."""
        problems: list[str] = []
        spec = recipe_target_spec(recipe)
        spec_hash = spec.spec_hash() if spec is not None else None
        if dataset.id != arm.dataset_id or dataset.manifest_hash != arm.dataset_manifest_hash:
            problems.append(f"dataset {dataset.id}/{dataset.manifest_hash[:19]}… is not the arm's "
                            f"{arm.dataset_id}/{arm.dataset_manifest_hash[:19]}…")
        if recipe.id != arm.training_recipe_id or recipe_hash(recipe) != arm.recipe_hash:
            problems.append(f"recipe {recipe.id} does not reproduce the arm's frozen recipe hash")
        if (spec_hash or "") != arm.train_target_spec_hash:
            problems.append(f"recipe training spec {spec_hash} is not the arm's {arm.train_target_spec_hash}")
        split = next((candidate for candidate in dataset.splits if candidate.name == "train"), None)
        if split is None:
            problems.append(f"{dataset.id} has no train split; an experiment arm is all-train")
        else:
            if split.count != arm.prefix_size:
                problems.append(f"arm declares prefix_size {arm.prefix_size} but {dataset.id}'s train "
                                f"split holds {split.count} rows")
            if split.record_ids_hash != arm.record_ids_hash:
                problems.append(f"arm declares record_ids_hash {arm.record_ids_hash[:19]}… but {dataset.id}'s "
                                f"train split records {split.record_ids_hash[:19]}…")
        return problems

    def verify_experiment_run(self, run: Run, experiment, arm) -> list[str]:
        """Everything a run must match to belong to this experiment, checked from the run record."""
        problems: list[str] = []
        if run.experiment_id != experiment.id:
            problems.append(f"run belongs to {run.experiment_id}, not {experiment.id}")
        if run.ablation_id not in arm.ablations:
            problems.append(f"ablation {run.ablation_id} is not listed in arm {arm.arm_id}")
        for field, expected in (("dataset_id", arm.dataset_id),
                                ("dataset_manifest_hash", arm.dataset_manifest_hash),
                                ("training_recipe_id", arm.training_recipe_id),
                                ("recipe_hash", arm.recipe_hash),
                                ("train_target_spec_hash", arm.train_target_spec_hash)):
            actual = getattr(run, field)
            if (actual or "") != expected:
                problems.append(f"{field} {actual} is not the arm's {expected}")
        if run.eval_protocol_id != experiment.eval_protocol_id:
            problems.append(f"evaluation protocol {run.eval_protocol_id} is not the study's "
                            f"{experiment.eval_protocol_id}")
        if (run.eval_dataset_id or "") != experiment.eval_dataset_id:
            problems.append(f"evaluation dataset {run.eval_dataset_id} is not the study's "
                            f"{experiment.eval_dataset_id}")
        if (run.eval_target_spec_hash or "") != experiment.eval_target_spec_hash:
            problems.append(f"evaluation supervision {run.eval_target_spec_hash} is not the study's "
                            f"{experiment.eval_target_spec_hash}")
        return problems

    def create_experiment(self, *, name: str, preregistration_hash: str, reference_unit_nodes: int,
                          scales: Mapping[str, int], node_budgets: Sequence[int], arms: Sequence[Mapping],
                          eval_protocol_id: str, widths: Sequence[int], seeds: Sequence[int],
                          source_id: str, normalization_id: str, candidate_universe_hash: str,
                          order_seed: int, order_hash: str, analysis: Mapping[str, object],
                          family: str = DEFAULT_FAMILY, generation: Optional[str] = None,
                          notes: str = "", experiment_id: Optional[str] = None) -> "Experiment":
        """Freeze one study. Every arm is verified against its frozen D####/T#### BEFORE the X
        exists, and the exam is taken from the protocol rather than trusted from the caller."""
        from .schemas import Experiment, ExperimentArm

        generation = self._generation(generation)
        protocol: EvaluationProtocol = self.store.get_as(eval_protocol_id, EvaluationProtocol)
        eval_spec = protocol_target_spec(protocol)
        if eval_spec is None:
            raise LabError(f"{protocol.id} pins no TargetSpec; an experiment needs a frozen exam")

        declared_scales = {str(key) for key in scales}
        declared_budgets = {int(budget) for budget in node_budgets}
        numeric_scale = {str(key): int(value) for key, value in scales.items()}
        declared_widths = {int(width) for width in widths}
        # Arms may partition a condition: several width-specific arms can share one
        # (scale, supervision depth) cell, which is what a targeted precision design needs.
        # What must hold is that the SET of cells is exactly the declared lattice.
        cells = {(str(arm["scale"]), int(arm["node_budget"])) for arm in arms}
        lattice = {(scale, budget) for scale in declared_scales for budget in declared_budgets}
        for arm in arms:
            if str(arm["scale"]) not in numeric_scale:
                raise LabError(f"arm {arm['arm_id']} names scale {arm['scale']!r}, which the design does "
                               f"not declare (declared: {sorted(numeric_scale)})")
            # the scale NAME must carry its frozen numeric target: a cell called "20x" with a
            # 2x budget would otherwise pass every check and quietly misspend the study
            if int(arm["scale_nodes"]) != numeric_scale[str(arm["scale"])]:
                raise LabError(f"arm {arm['arm_id']} declares scale_nodes {arm['scale_nodes']} for scale "
                               f"{arm['scale']!r}, but the design freezes {numeric_scale[str(arm['scale'])]}")
        missing_cells = sorted(lattice - cells)
        undeclared_cells = sorted(cells - lattice)
        if missing_cells or undeclared_cells:
            raise LabError(f"the design must be exactly the declared lattice: missing {missing_cells}, "
                           f"not declared {undeclared_cells}")

        frozen_arms: list[ExperimentArm] = []
        per_cell: dict = {}
        seen: set[str] = set()
        for arm in arms:
            arm_id = str(arm["arm_id"])
            if arm_id in seen:
                raise LabError(f"duplicate arm_id {arm_id!r}")
            seen.add(arm_id)
            dataset: Dataset = self.store.get_as(str(arm["dataset_id"]), Dataset)
            recipe: TrainingRecipe = self.store.get_as(str(arm["training_recipe_id"]), TrainingRecipe)
            # the arm's identity is DECLARED by the caller and then verified against the frozen
            # objects; it is never silently normalized from them, or a wrong declaration would
            # be healed instead of refused
            missing = [key for key in ("dataset_manifest_hash", "recipe_hash", "train_target_spec_hash",
                                       "record_ids_hash") if not arm.get(key)]
            if missing:
                raise LabError(f"arm {arm_id} must declare {', '.join(missing)}; the arm identity is "
                               "verified against the frozen objects, not inferred from them")
            entry = ExperimentArm(
                arm_id=arm_id, scale=str(arm["scale"]), scale_nodes=int(arm["scale_nodes"]),
                node_budget=int(arm["node_budget"]), prefix_size=int(arm["prefix_size"]),
                realized_nodes=int(arm["realized_nodes"]), dataset_id=str(arm["dataset_id"]),
                dataset_manifest_hash=str(arm["dataset_manifest_hash"]),
                training_recipe_id=str(arm["training_recipe_id"]), recipe_hash=str(arm["recipe_hash"]),
                train_target_spec_hash=str(arm["train_target_spec_hash"]),
                record_ids_hash=str(arm["record_ids_hash"]),
                ablations=[str(a) for a in arm.get("ablations") or []],
                seeds=[int(s) for s in arm.get("seeds") or []])
            problems = self.verify_experiment_arm(entry, dataset, recipe)
            if problems:
                raise LabError(f"arm {arm_id} does not match its frozen identities: " + "; ".join(problems))
            arm_widths: dict = {}
            for ablation_id in entry.ablations:
                ablation: Ablation = self.store.get_as(ablation_id, Ablation)
                if (ablation.family, ablation.generation) != (family, generation):
                    raise LabError(f"{ablation_id} belongs to {ablation.family}-{ablation.generation}")
                if (ablation.dataset_id, ablation.training_recipe_id, ablation.eval_protocol_id) != \
                        (dataset.id, recipe.id, protocol.id):
                    raise LabError(f"{ablation_id} is not defined on this arm's dataset/recipe/protocol")
                if "H" not in ablation.effective_config:
                    raise LabError(f"{ablation_id} carries no H; an arm must cover the declared widths")
                arm_widths.setdefault(int(ablation.effective_config["H"]), []).append(ablation_id)
            covered = set(arm_widths)
            if not covered or not covered <= declared_widths:
                raise LabError(f"arm {arm_id} covers widths {sorted(covered)}, which are not a subset of "
                               f"the declared widths {sorted(declared_widths)}")
            duplicated = {width: ids for width, ids in arm_widths.items() if len(ids) > 1}
            if duplicated:
                raise LabError(f"arm {arm_id} lists more than one ablation for width(s) "
                               f"{sorted(duplicated)}: {duplicated}")
            per_cell.setdefault((entry.scale, int(entry.node_budget)), []).append((arm_id, covered))
            frozen_arms.append(entry)

        # The X<->A binding must be REAL: take the id before creation so ablations can carry it,
        # then require every listed ablation to carry it back and every carrying ablation to be
        # listed. A run's membership therefore rests on a verified back-reference, not a promise.
        xid = experiment_id or self.store.next_id("X")
        listed = {ablation_id for arm in frozen_arms for ablation_id in arm.ablations}
        for arm in frozen_arms:
            for ablation_id in arm.ablations:
                ablation: Ablation = self.store.get_as(ablation_id, Ablation)
                if ablation.experiment_id != xid:
                    raise LabError(f"{ablation_id} carries experiment_id={ablation.experiment_id!r}, "
                                   f"not {xid!r}: it is not bound to this experiment")
        for ablation in self.store.list("A", verify=True, kind=Ablation):
            if ablation.experiment_id == xid and ablation.id not in listed:
                raise LabError(f"{ablation.id} claims {xid} but no arm lists it; refusing an "
                               "experiment with unlisted members")

        # Each (scale, supervision depth) cell must present the full width ladder across its arms:
        # one arm may carry all four widths (S7-S/S9/S10), or several arms may carry one width each
        # (S11's targeted precision design). The union, not a per-arm requirement: a missing width
        # still fails closed, and a width may not appear twice within one arm.
        for (scale, budget), entries in sorted(per_cell.items()):
            covered_here = {width for _arm_id, covered in entries for width in covered}
            if covered_here != declared_widths:
                raise LabError(f"cell ({scale}, {budget}) presents widths {sorted(covered_here)} but the "
                               f"design declares {sorted(declared_widths)}")

        declared_seeds = {int(value) for value in seeds}
        for entry in frozen_arms:
            outside = [int(value) for value in entry.seeds if int(value) not in declared_seeds]
            if outside:
                raise LabError(f"arm {entry.arm_id} declares seeds {outside} outside the experiment's "
                               f"seed list {sorted(declared_seeds)}")
        # the expected matrix size follows the furniture: an arm with its own seed subset
        # contributes only that subset, so a targeted precision design seals a truthful count
        run_count = sum(len(arm.ablations) * len(arm.seeds or declared_seeds) for arm in frozen_arms)
        experiment = Experiment(
            id=xid, name=name, family=family, generation=generation,
            preregistration_hash=preregistration_hash, reference_unit_nodes=int(reference_unit_nodes),
            scales={str(k): int(v) for k, v in scales.items()}, node_budgets=[int(b) for b in node_budgets],
            source_id=source_id, normalization_id=normalization_id,
            candidate_universe_hash=candidate_universe_hash, order_seed=int(order_seed),
            order_hash=order_hash, arms=frozen_arms, eval_dataset_id=protocol.dataset_id,
            eval_dataset_manifest_hash=protocol.dataset_manifest_hash, eval_protocol_id=protocol.id,
            eval_protocol_hash=protocol.protocol_hash, eval_target_spec_hash=eval_spec.spec_hash(),
            widths=[int(w) for w in widths], seeds=[int(x) for x in seeds],
            expected_run_count=run_count, analysis=dict(analysis), notes=notes,
            status="FROZEN", created_at=utc_now())
        return self.store.create(experiment)

    def experiment_contract(self, run: Run) -> tuple:
        """(checks, problems) for a run's membership in its experiment, or ([], []) if none."""
        if not run.experiment_id:
            return [], []
        checks: list[IntegrityCheck] = []
        experiment = self.store.get(run.experiment_id)
        arm = self.arm_of(experiment, run.ablation_id)
        if arm is None:
            return [], [f"ablation {run.ablation_id} is not an arm of {experiment.id}"]
        problems = self.verify_experiment_run(run, experiment, arm)
        checks.append(make_check(
            "experiment_contract", not problems,
            f"{experiment.id} arm {arm.arm_id} ({arm.scale}, {arm.node_budget} nodes, {arm.prefix_size} rows)"
            if not problems else "; ".join(problems)))
        return checks, problems

    def expected_membership(self, experiment) -> dict:
        """The Cartesian product the study declared: every (ablations x seeds) cell.

        Derived from the FROZEN design, not from whatever runs happen to exist, so a partial
        study cannot masquerade as a complete one.
        """
        keys: dict = {}
        declared = {int(value) for value in experiment.seeds}
        for arm in experiment.arms:
            if not arm.ablations:
                raise LabError(f"arm {arm.arm_id} lists no ablations; expected membership cannot "
                               "be derived from this design")
            arm_seeds = [int(value) for value in (arm.seeds or experiment.seeds)]
            outside = [seed for seed in arm_seeds if seed not in declared]
            if outside:
                raise LabError(f"arm {arm.arm_id} declares seeds {outside} outside the experiment's "
                               f"seed list {sorted(declared)}")
            for ablation_id in arm.ablations:
                for seed in arm_seeds:
                    key = (ablation_id, int(seed))
                    if key in keys:
                        raise LabError(f"cell {key} is claimed by two arms "
                                       f"({keys[key]['arm_id']} and {arm.arm_id})")
                    keys[key] = {"arm_id": arm.arm_id, "seed": int(seed), "scale": arm.scale,
                                 "node_budget": arm.node_budget}
        return keys

    def membership_report(self, experiment) -> dict:
        """Exactly one terminal run per expected cell, no duplicates, no extras."""
        expected = self.expected_membership(experiment)
        runs = [r for r in self.store.list("R", verify=True, kind=Run)
                if r.experiment_id == experiment.id]
        by_key: dict = {}
        for run in runs:
            by_key.setdefault((run.ablation_id, run.seed), []).append(run)
        missing = sorted(key for key in expected if key not in by_key)
        duplicates = {f"{key[0]}:{key[1]}": [r.id for r in value]
                      for key, value in by_key.items() if len(value) > 1}
        extra = sorted(key for key in by_key if key not in expected)
        nonterminal = sorted(run.id for run in runs if run.status not in TERMINAL_RUN_STATUSES)
        completed = sorted(key for key, value in by_key.items()
                           if len(value) == 1 and value[0].status == RunStatus.COMPLETED)
        invalid = sorted(key for key, value in by_key.items()
                         if len(value) == 1 and value[0].status == RunStatus.INVALID)
        return {"experiment": experiment.id, "expected_cells": len(expected), "runs_found": len(runs),
                "completed_cells": len(completed), "invalid_cells": len(invalid),
                "missing": [[key[0], key[1]] for key in missing], "duplicates": duplicates,
                "extra": [[key[0], key[1]] for key in extra], "nonterminal": nonterminal}

    def assert_experiment_complete(self, experiment) -> dict:
        """Fail closed unless the frozen matrix exists exactly once and is fully terminal."""
        report = self.membership_report(experiment)
        problems = []
        if report["missing"]:
            problems.append(f"{len(report['missing'])} expected cells have no run "
                            f"(first: {report['missing'][0]})")
        if report["duplicates"]:
            problems.append(f"{len(report['duplicates'])} cells have more than one run "
                            f"({list(report['duplicates'])[0]})")
        if report["extra"]:
            problems.append(f"{len(report['extra'])} runs are not in the design "
                            f"(first: {report['extra'][0]})")
        if report["nonterminal"]:
            problems.append(f"{len(report['nonterminal'])} runs are not terminal "
                            f"(first: {report['nonterminal'][0]})")
        if problems:
            raise LabError(f"{experiment.id} membership is not the frozen design: " + "; ".join(problems))
        return report

    def assert_experiment_runs(self, run_ids: Sequence[str]) -> str:
        """Analysis-time gate: one experiment, complete membership, every contract intact."""
        runs = [self.store.get_as(run_id, Run) for run_id in run_ids]
        if not runs:
            raise LabError("no runs to check")
        experiments = {run.experiment_id for run in runs}
        if len(experiments) != 1 or None in experiments:
            raise LabError(f"runs span {sorted(str(e) for e in experiments)}; a comparison must stay "
                           "inside one experiment")
        experiment = self.store.get(next(iter(experiments)))
        expected = self.expected_membership(experiment)
        for run in runs:
            arm = self.arm_of(experiment, run.ablation_id)
            if arm is None:
                raise LabError(f"{run.id}: ablation {run.ablation_id} is not an arm of {experiment.id}")
            if (run.ablation_id, run.seed) not in expected:
                raise LabError(f"{run.id}: cell ({run.ablation_id}, {run.seed}) is not in the design")
            problems = self.verify_experiment_run(run, experiment, arm)
            if problems:
                raise LabError(f"{run.id}: " + "; ".join(problems))
        # a comparison inside a study is only meaningful when the whole frozen matrix exists
        self.assert_experiment_complete(experiment)
        self.assert_runs_comparable([run.id for run in runs])
        return experiment.id

    def seal_experiment(self, experiment_id: str, result: Mapping[str, object]) -> "Experiment":
        from .schemas import Experiment

        experiment: Experiment = self.store.get_as(experiment_id, Experiment)
        membership = self.assert_experiment_complete(experiment)   # no sealing a partial study
        experiment.status = "SEALED"
        experiment.result = {**dict(result), "membership": membership,
                             "expected_run_count": experiment.expected_run_count}
        experiment.state_history = list(experiment.state_history) + [
            StateChange(state=EvidenceState.PROPOSED, at=utc_now(), reason="result sealed")]
        return self.store.update_experiment(experiment)

    def evaluation_spec_hash(self, run: Run) -> str:
        """The evaluation TargetSpec that judged this run, read from the protocol it pins.

        The protocol is the authority: a run records `eval_target_spec_hash` as evidence for
        the reader, but what actually judged the model is the frozen protocol's spec.
        """
        protocol: EvaluationProtocol = self.store.get_as(run.eval_protocol_id, EvaluationProtocol)
        spec = protocol_target_spec(protocol)
        return spec.spec_hash() if spec is not None else ""

    def instrument_identity(self, run: Run) -> tuple:
        """The EXAM a run sat: (eval protocol id, protocol hash, eval dataset id, eval dataset
        manifest hash, evaluation TargetSpec hash).

        Built from the evaluation side ONLY. A run's training dataset is the treatment in a
        supervision experiment — two S7 arms necessarily train on different D#### — so the
        training manifest must never enter the instrument identity, or legitimate arms would
        look incomparable. Historical runs that predate the split-dataset convention evaluated
        on their own dataset; for them the protocol's own binding is the evaluation side.
        """
        protocol: EvaluationProtocol = self.store.get_as(run.eval_protocol_id, EvaluationProtocol)
        return (run.eval_protocol_id, run.protocol_hash,
                run.eval_dataset_id or protocol.dataset_id,
                run.eval_dataset_manifest_hash or protocol.dataset_manifest_hash,
                self.evaluation_spec_hash(run))

    @staticmethod
    def _exam_differences(exams: list[tuple]) -> list[str]:
        """Human-readable reasons two exam identities differ, in the reviewer's terms."""
        reasons: list[str] = []
        protocols = {(exam[0], exam[1]) for exam in exams}
        datasets = {(exam[2], exam[3]) for exam in exams}
        specs = {exam[4] for exam in exams}
        if len(protocols) > 1:
            reasons.append(f"{len(protocols)} distinct evaluation protocols")
        if len(datasets) > 1:
            reasons.append(f"{len(datasets)} distinct evaluation datasets")
        if len(specs) > 1:
            reasons.append("evaluation TargetSpecs "
                           + ", ".join(sorted(h[:19] + "…" for h in specs)))
        return reasons

    def assert_runs_comparable(self, run_ids: Sequence[str]) -> str:
        """Every run in one comparison must share one evaluation instrument.

        Two runs measured against different exams are not the same measurement, however
        similar their metric names look: a 16k-node evaluation and a 400k-node evaluation
        may both be a number called test_loss. Returns the shared evaluation TargetSpec
        hash, or raises naming which part of the exam differs.

        Different training datasets are EXPECTED here — that is the treatment. This is the
        invariant S7 turns into an experiment identity (X####): one shared evaluation
        protocol, evaluation dataset and evaluation TargetSpec per experiment, while each
        treatment arm pins its own training TargetSpec.
        """
        runs = [self.store.get_as(run_id, Run) for run_id in run_ids]
        if len(runs) < 2:
            raise LabError("a comparison needs at least two runs")
        exams = [self.instrument_identity(run) for run in runs]
        reasons = self._exam_differences(exams)
        if reasons:
            raise LabError("runs were not measured on one shared evaluation instrument ("
                           + "; ".join(reasons) + ") and are not comparable")
        return exams[0][4]

    def draft_finding(self, *, hypothesis_id: str, control_ablation_id: str, intervention_ablation_id: str,
                      interpretation: Optional[str] = None, next_experiment: Optional[str] = None,
                      non_claims: Optional[list[str]] = None) -> Finding:
        hypothesis: Hypothesis = self.store.get(hypothesis_id)
        control: Ablation = self.store.get(control_ablation_id)
        intervention: Ablation = self.store.get(intervention_ablation_id)
        if not isinstance(hypothesis, Hypothesis) or not isinstance(control, Ablation) or not isinstance(intervention, Ablation):
            raise LabError("expected a hypothesis and two ablations")
        if intervention.control_id != control.id:
            raise LabError(f"{intervention.id} is defined against {intervention.control_id}, not {control.id}")
        if intervention.hypothesis_id and intervention.hypothesis_id != hypothesis.id:
            raise LabError(f"{intervention.id} tests {intervention.hypothesis_id}, not {hypothesis.id}")
        protocol: EvaluationProtocol = self.store.get_as(intervention.eval_protocol_id, EvaluationProtocol)
        metric = next((m for m in protocol.metrics if m.name == hypothesis.metric), None)
        if metric is None:
            raise LabError(f"{protocol.id} does not measure {hypothesis.metric}")

        checks: list[IntegrityCheck] = []
        all_runs: list[Run] = self.store.list("R", verify=False, kind=Run)
        completed: dict[str, list[Run]] = {}
        invalid: dict[str, list[str]] = {}
        for role, arm in (("control", control), ("intervention", intervention)):
            finished = [r for r in all_runs if r.ablation_id == arm.id and r.status in TERMINAL_RUN_STATUSES]
            if not finished:
                raise LabError(f"{arm.id} has no finished runs; execute runs of both arms before drafting a finding")
            tampered = []
            for r in finished:
                try:
                    self.store.verify(r)
                except TamperError:
                    tampered.append(r.id)
            completed[arm.id] = [r for r in finished if r.status == RunStatus.COMPLETED and r.id not in tampered]
            invalid[arm.id] = [r.id for r in finished if r.status == RunStatus.INVALID]
            checks.append(make_check(f"{role}_evidence", bool(completed[arm.id]) and not tampered,
                                     f"{len(completed[arm.id])} completed run(s); invalid runs excluded: "
                                     f"{', '.join(invalid[arm.id]) or 'none'}; tampered: {', '.join(tampered) or 'none'}"))
        used = completed[control.id] + completed[intervention.id]
        # one definition of "the exam", shared with assert_runs_comparable: evaluation side
        # only, so different training datasets never make legitimate arms look incomparable
        exams = [self.instrument_identity(run) for run in used]
        exam_differences = self._exam_differences(exams)
        checks.append(make_check(
            "same_instrument", not any("protocol" in r or "dataset" in r for r in exam_differences),
            "all runs share one evaluation protocol and one evaluation dataset"
            if not exam_differences else "; ".join(exam_differences)))
        eval_specs = {exam[4] for exam in exams}
        checks.append(make_check(
            "same_evaluation_target_spec", len(eval_specs) <= 1,
            "two runs measured against different evaluation TargetSpecs are not the same measurement: "
            + (f"all {len(used)} compared runs share {next(iter(eval_specs))[:19]}…" if len(eval_specs) <= 1
               else f"found {len(eval_specs)} distinct evaluation specs: "
                    + ", ".join(sorted(h[:19] + "…" for h in eval_specs)))))

        values: dict[str, np.ndarray] = {}
        reference_ids, artifacts_ok, aligned = None, True, True
        for r in used:
            path = self.store.abs(f"artifacts/{r.id}/eval.json")
            if not path.exists() or sha256_file(path) != r.artifact_hashes.get("eval.json"):
                artifacts_ok = False
                continue
            payload = json.loads(path.read_bytes())
            if reference_ids is None:
                reference_ids = payload["record_ids"]
            elif payload["record_ids"] != reference_ids:
                aligned = False
            values[r.id] = np.array([np.nan if v is None else v for v in payload["metrics"][metric.name]], dtype=np.float64)
        checks.append(make_check("eval_artifacts", artifacts_ok, "per-position evaluation artifacts match their run hashes"))
        checks.append(make_check("paired_positions", aligned, "every run was evaluated on the same ordered positions"))

        def arm_summary(arm: Ablation, arm_values: Optional[np.ndarray]) -> ArmSummary:
            runs = completed[arm.id]
            mean = low = high = None
            if arm_values is not None:
                mean, low, high, _ = nnue.bootstrap_mean(arm_values, protocol.bootstrap_samples, protocol.bootstrap_seed)
            return ArmSummary(ablation_id=arm.id, display_label=arm.display_label, param_count=arm.param_count,
                              run_ids=[r.id for r in runs], invalid_run_ids=invalid[arm.id],
                              seeds=[r.seed for r in runs], metric_mean=mean, ci_low=low, ci_high=high,
                              cpu_seconds=round(sum(r.compute.cpu_seconds for r in runs), 4))

        effect = None
        if _failed(checks):
            result = EvidenceState.INVALID
            control_summary, intervention_summary = arm_summary(control, None), arm_summary(intervention, None)
        else:
            control_values = np.mean([values[r.id] for r in completed[control.id]], axis=0)
            intervention_values = np.mean([values[r.id] for r in completed[intervention.id]], axis=0)
            mask = np.isfinite(control_values) & np.isfinite(intervention_values)
            control_summary = arm_summary(control, control_values[mask])
            intervention_summary = arm_summary(intervention, intervention_values[mask])
            difference, low, high, n = nnue.bootstrap_mean(intervention_values[mask] - control_values[mask],
                                                           protocol.bootstrap_samples, protocol.bootstrap_seed)
            effect = Effect(metric=metric.name, difference=difference, ci_low=low, ci_high=high, n=n,
                            method=f"paired percentile bootstrap over {n} held-out positions "
                                   f"({protocol.bootstrap_samples} resamples, seed {protocol.bootstrap_seed}); "
                                   "per-position values averaged across each arm's seeds")
            result = self._decide(hypothesis, effect)

        seeds = min(len(control_summary.seeds), len(intervention_summary.seeds))
        dataset: Dataset = self.store.get(intervention.dataset_id)
        recipe: TrainingRecipe = self.store.get(intervention.training_recipe_id)
        changes = ", ".join(f"{d.key} {families.format_value(d.control)} -> {families.format_value(d.value)}"
                            for d in intervention.diff)
        previous = [f for f in self.store.list("F", verify=False)
                    if (f.hypothesis_id, f.control.ablation_id, f.intervention.ablation_id)
                    == (hypothesis.id, control.id, intervention.id)]
        default_non_claims = list(protocol.non_claims)
        if seeds < 2:
            default_non_claims.append("Uncertainty covers held-out position sampling only; seed-to-seed variance is "
                                      "not estimated with fewer than two seeds per arm.")
        if any(d.axis == "capacity" for d in intervention.diff):
            default_non_claims.append("Changing capacity changes learned parameters only; it does not isolate "
                                      "representation, executed search compute or training signal.")

        finding = Finding(
            id=self.store.next_id("F"), hypothesis_id=hypothesis.id, hypothesis_title=hypothesis.title,
            hypothesis_statement=hypothesis.statement, eval_protocol_id=protocol.id, dataset_id=dataset.id,
            metric=metric.name, predicted_direction=hypothesis.predicted_direction, min_effect=hypothesis.min_effect,
            control=control_summary, intervention=intervention_summary,
            frozen_controls=[
                f"Control {control.display_label}",
                f"Dataset {dataset.id} '{dataset.name}' manifest {dataset.manifest_hash}",
                f"Training recipe {recipe.id} '{recipe.name}' ({recipe.trainer} v{recipe.trainer_version}) {recipe.recipe_hash}",
                f"Evaluation protocol {protocol.id} '{protocol.name}' on split '{protocol.split}' {protocol.protocol_hash}",
                "All other registered switches identical to the control",
            ],
            intervention_diff=intervention.diff,
            evidence=[
                f"Control runs {', '.join(control_summary.run_ids) or 'none'} (seeds {control_summary.seeds}); "
                f"intervention runs {', '.join(intervention_summary.run_ids) or 'none'} (seeds {intervention_summary.seeds})",
                f"Metric '{metric.name}': {metric.description}",
                "Measured on serialized model artifacts reloaded from disk, not on in-memory weights",
                f"Exact learned parameters: control {control.param_count}, intervention {intervention.param_count}",
                f"CPU seconds: control {control_summary.cpu_seconds}, intervention {intervention_summary.cpu_seconds}; "
                "search nodes: 0 (no search executed)",
            ],
            effect=effect, result=result, decision_rule=self._decision_rule(hypothesis), integrity=checks,
            promotion_state=PromotionState.NOT_ELIGIBLE,
            promotion_gates=[PromotionGate(name=name, status="missing", detail=detail)
                             for name, detail in PROMOTION_GATES],
            promotion_consequence=PROMOTION_PHASE0,
            interpretation=interpretation or self._interpretation(result, hypothesis, metric, effect, changes, control, dataset),
            non_claims=non_claims or default_non_claims,
            next_experiment=next_experiment or self._next_experiment(result, seeds, intervention),
            supersedes=previous[-1].id if previous else None, created_at=utc_now(),
        )
        self.store.create(finding)
        self.store.update_state(intervention.id, result, f"{finding.id}")
        if result != EvidenceState.INVALID:
            self.store.update_state(hypothesis.id, result, f"{finding.id}")
        return finding

    @staticmethod
    def _decision_rule(h: Hypothesis) -> str:
        if h.predicted_direction == "decrease":
            return (f"Effect = intervention - control on {h.metric}. SUPPORTED if the 95% CI upper bound < "
                    f"-{h.min_effect:g}; REJECTED if the CI lower bound > -{h.min_effect:g}; otherwise INCONCLUSIVE. "
                    "INVALID if any integrity check fails.")
        return (f"Effect = intervention - control on {h.metric}. SUPPORTED if the 95% CI lower bound > {h.min_effect:g}; "
                f"REJECTED if the CI upper bound < {h.min_effect:g}; otherwise INCONCLUSIVE. "
                "INVALID if any integrity check fails.")

    @staticmethod
    def _decide(h: Hypothesis, effect: Effect) -> EvidenceState:
        if h.predicted_direction == "decrease":
            if effect.ci_high < -h.min_effect:
                return EvidenceState.SUPPORTED
            if effect.ci_low > -h.min_effect:
                return EvidenceState.REJECTED
        else:
            if effect.ci_low > h.min_effect:
                return EvidenceState.SUPPORTED
            if effect.ci_high < h.min_effect:
                return EvidenceState.REJECTED
        return EvidenceState.INCONCLUSIVE

    @staticmethod
    def _interpretation(result, h, metric, effect, changes, control, dataset) -> str:
        if result == EvidenceState.INVALID or effect is None:
            return ("The evidence failed integrity checks, so no scientific claim is made. The invalid runs stay on "
                    "record; repair the failing checks and repeat the comparison.")
        better = (effect.difference < 0) == metric.lower_is_better
        verdict = {
            EvidenceState.SUPPORTED: f"The interval lies beyond the pre-declared minimum effect ({h.min_effect:g}) in the predicted direction.",
            EvidenceState.REJECTED: "The interval excludes an effect of the pre-declared size in the predicted direction.",
            EvidenceState.INCONCLUSIVE: "The interval is compatible with both a meaningful effect and no meaningful effect.",
        }[result]
        return (f"Changing {changes} moved {metric.name} by {effect.difference:+.4g} (95% CI {effect.ci_low:+.4g} to "
                f"{effect.ci_high:+.4g}; {'better' if better else 'worse'} for this metric) relative to "
                f"{control.display_label}, on {effect.n} held-out positions from {dataset.id}. {verdict}")

    @staticmethod
    def _next_experiment(result, seeds: int, intervention: Ablation) -> str:
        if result == EvidenceState.INVALID:
            return "Repair the failing integrity checks, then repeat both arms under the same frozen dataset, recipe and protocol."
        if seeds < 2:
            return "Replicate both arms with at least three seeds to estimate seed variance before any scaling decision."
        if result == EvidenceState.SUPPORTED and any(d.key == "H" for d in intervention.diff):
            return ("Run the full width ladder H = 1, 2, 4, 8, 16, 32, 64, 128, 256 under the same frozen D/T/E to "
                    "locate where held-out returns per parameter flatten.")
        if result == EvidenceState.REJECTED:
            return "Keep this negative result visible; test a different rung or axis instead of re-running this intervention."
        return "Increase held-out evidence (a larger frozen evaluation split or more seeds) before drawing a conclusion."

    def list_findings(self, *, query: Optional[str] = None, result: Optional[str] = None) -> list[Finding]:
        findings = self.store.list("F", verify=False)
        superseded = {f.supersedes for f in findings if f.supersedes}
        out = []
        for f in findings:
            state = EvidenceState.SUPERSEDED.value if f.id in superseded else f.result.value
            if result and result.upper() not in {state, f.result.value}:
                continue
            if query and query.lower() not in json.dumps(f.model_dump(mode="json")).lower():
                continue
            out.append(f)
        return out

    # ------------------------------------------------------------------ views

    def matrix(self, *, family: str = DEFAULT_FAMILY, generation: Optional[str] = None, baseline: Optional[str] = None,
               dataset_id: Optional[str] = None, min_params: Optional[int] = None, max_params: Optional[int] = None,
               state: Optional[str] = None, metric: str = "test_loss") -> MatrixResponse:
        switches = families.switches(family)
        ablations = {a.id: a for a in self.store.list("A") if a.family == family}
        runs = self.store.list("R", verify=False, kind=Run)
        findings = self.store.list("F", verify=False)
        rows = []
        for a in ablations.values():
            if generation and a.generation != generation or baseline and a.baseline_name != baseline.upper():
                continue
            if dataset_id and a.dataset_id != dataset_id or state and a.state.value != state.upper():
                continue
            if min_params is not None and a.param_count < min_params or max_params is not None and a.param_count > max_params:
                continue
            control_cfg = a.effective_config if a.is_baseline else ablations[a.control_id].effective_config
            mine = [r for r in runs if r.ablation_id == a.id]
            scores = [m.value for r in mine if r.status == RunStatus.COMPLETED for m in r.metrics if m.name == metric]
            rows.append(MatrixRow(
                ablation_id=a.id, display_label=a.display_label, is_baseline=a.is_baseline, control_id=a.control_id,
                generation=a.generation, baseline_name=a.baseline_name, dataset_id=a.dataset_id,
                param_count=a.param_count, state=a.state,
                cells=[MatrixCell(key=sw.key, value=a.effective_config.get(sw.key),
                                  changed=control_cfg.get(sw.key) != a.effective_config.get(sw.key)) for sw in switches],
                run_counts=dict(Counter(r.status.value for r in mine)),
                metric_mean=float(np.mean(scores)) if scores else None,
                finding_ids=[f.id for f in findings if a.id in (f.control.ablation_id, f.intervention.ablation_id)],
            ))
        return MatrixResponse(family=family, metric=metric, switches=switches, rows=rows)

    def scaling(self, *, metric: str = "test_loss", family: str = DEFAULT_FAMILY) -> ScalingResponse:
        definition = next((m for m in nnue.METRIC_DEFS if m.name == metric), None)
        if definition is None:
            raise LabError(f"unknown metric {metric!r}")
        ablations = {a.id: a for a in self.store.list("A") if a.family == family}
        points = []
        for r in self.store.list("R", verify=False, kind=Run):
            a = ablations.get(r.ablation_id)
            if a is None or r.status not in TERMINAL_RUN_STATUSES:
                continue
            value = next((m for m in r.metrics if m.name == metric), None)
            points.append(ScalingPoint(
                run_id=r.id, ablation_id=a.id, display_label=a.display_label, family=a.family,
                generation=a.generation, representation=str(a.effective_config.get("INPUT")), is_baseline=a.is_baseline,
                status=r.status, seed=r.seed, param_count=r.param_count, search_nodes=r.compute.search_nodes,
                cpu_seconds=r.compute.cpu_seconds, value=value.value if value else None,
                ci_low=value.ci_low if value else None, ci_high=value.ci_high if value else None,
                kind=value.kind if value else None,
            ))
        legacy_models = [LegacyModelPoint(
            model_id=m.id, name=m.name, role=m.role, model_kind=m.model_kind,
            serialized_param_count=m.serialized_param_count, provenance_complete=m.provenance_complete,
        ) for m in self.store.list("M", verify=False, kind=LegacyModel)]
        return ScalingResponse(metric=metric, lower_is_better=definition.lower_is_better,
                               available_metrics=nnue.METRIC_DEFS, points=points, legacy_models=legacy_models)

    def _latest_intake(self) -> Optional[IntakeRecord]:
        records = self.store.list("I", verify=False, kind=IntakeRecord)
        return records[-1] if records else None

    def _legacy_summary(self) -> LegacySummary:
        """Summarize imported legacy evidence from the objects themselves; never from the catalog files."""
        record = self._latest_intake()
        models = self.store.list("M", verify=False, kind=LegacyModel)
        sources = self.store.list("S", verify=False, kind=LegacySource)
        evaluations = self.store.list("E", verify=False, kind=LegacyEvaluation)
        evidence = self.store.list("R", verify=False, kind=LegacyEvidence)
        states: Counter = Counter(e.evidence_state.value for e in evidence if e.evidence_state is not None)
        unstates: Counter = Counter(e.recorded_state for e in evidence if e.evidence_state is None)
        switch_states: Counter = Counter(s.legacy_state.value for s in record.switches) if record else Counter()
        roles: Counter = Counter(m.role for m in models)
        return LegacySummary(
            intake_id=record.id if record else None,
            catalog_generated_at=record.catalog_generated_at if record else None,
            production_lineage="LEGACY — imported evidence is inspectable but never inherited into the clean lineage"
                               if (record or models or sources) else "nothing imported",
            champion=record.champion if record else None,
            roots=record.roots if record else [],
            engines=record.engines if record else [],
            flagship_model_ids=record.flagship_model_ids if record else [],
            models=len(models), models_provenance_incomplete=sum(not m.provenance_complete for m in models),
            model_roles=dict(roles), sources=len(sources), source_rows=sum(s.rows for s in sources),
            source_bytes=sum(s.bytes for s in sources), evaluations=len(evaluations), evidence=len(evidence),
            evidence_states={**{k: v for k, v in unstates.items()}, **{k: states[k] + unstates.get(k, 0) for k in states}},
            switch_legacy_states=dict(switch_states), findings=record.findings if record else [],
        )

    MAP_PROJECTIONS = ("data", "research", "promotion")

    def map_view(self, projection: str = "data", *, generation: Optional[str] = None,
                 state: Optional[str] = None, family: Optional[str] = None):
        """Lab Map: nodes and provenance edges derived only from canonical lab objects.

        Projections: `data` (sources -> normalizations + label producers -> datasets ->
        models), `research` (hypotheses -> ablations -> runs -> models -> findings),
        `promotion` (frozen baselines vs intervention ablations with finding decisions).
        """
        from .schemas import MapEdge, MapNode, MapResponse

        projection = (projection or "data").lower()
        if projection not in self.MAP_PROJECTIONS:
            raise LabError(f"unknown projection {projection!r}; choose one of {self.MAP_PROJECTIONS}")

        nodes: dict[str, MapNode] = {}
        edges: dict[tuple[str, str, str], MapEdge] = {}

        def add(node_id: str, kind: str, label: str, *, node_state: Optional[str] = None,
                link: str = "", gen: Optional[str] = None, fam: Optional[str] = None, **detail) -> None:
            if generation and gen != generation:
                return
            if family and fam and fam.lower() != family.lower():
                return
            if node_id in nodes:
                return
            nodes[node_id] = MapNode(id=node_id, kind=kind, label=label, state=node_state,
                                     link=link, detail=detail)

        def edge(src: str, dst: str, kind: str) -> None:
            if src in nodes and dst in nodes and (src, dst, kind) not in edges:
                edges[(src, dst, kind)] = MapEdge(src=src, dst=dst, kind=kind)

        ablations = self.store.list("A", verify=False, kind=Ablation)
        findings = self.list_findings()
        runs = self.store.list("R", verify=False, kind=Run)
        models = {m.id: m for m in self.store.list("M", verify=False, kind=ModelArtifact)}
        ablation_by_id = {a.id: a for a in ablations}

        if projection == "data":
            for source in self.store.list("S", verify=False, kind=SourceSnapshot):
                add(source.id, "source", f"{source.id} {source.name}", link=f"/object/{source.id}",
                    rows=f"{source.row_count:,}", importer=source.importer)
            for norm in self.store.list("N", verify=False):
                add(norm.id, "normalization", f"{norm.id} {norm.name}", link=f"/object/{norm.id}",
                    records=f"{norm.record_count:,}", duplicates=norm.duplicate_count)
                for source_id in norm.source_ids:
                    edge(source_id, norm.id, "standardized")
                for ref in data.label_set_refs(self.store, norm.id):
                    for authority, rows in ref.authorities.items():
                        producer_id = f"labels:{norm.id}:{authority}"
                        add(producer_id, "labels", f"label set {authority}", link="/data",
                            rows=f"{rows} rows", families=", ".join(ref.families))
                        edge(producer_id, norm.id, "annotated")
            for dataset in self.store.list("D", verify=False, kind=Dataset):
                add(dataset.id, "dataset", f"{dataset.id} {dataset.name}",
                    link=f"/object/{dataset.id}", records=f"{dataset.counts['records']:,}",
                    stack="+".join(arm.name for arm in dataset.stack) or "flat")
                edge(dataset.normalization_id, dataset.id, "frozen")
            runs_by_dataset: dict[str, set[str]] = {}
            for run in runs:
                if run.model_id:
                    runs_by_dataset.setdefault(run.dataset_id, set()).add(run.model_id)
            for dataset_id, model_ids in runs_by_dataset.items():
                for model_id in model_ids:
                    model = models[model_id]
                    ablation = ablation_by_id.get(model.ablation_id)
                    add(model_id, "model", f"{model_id} {model.arch}", link=f"/object/{model_id}",
                        gen=ablation.generation if ablation else None,
                        fam=ablation.family if ablation else None,
                        parameters=f"{model.param_count:,}", run=model.run_id)
                    edge(dataset_id, model_id, "trained_from")
        else:
            hypotheses = {h.id: h for h in self.store.list("H")}
            show: set[str] = set()
            if projection == "promotion":
                for finding in findings:
                    show.update((finding.control.ablation_id, finding.intervention.ablation_id, finding.id))
            for ablation in ablations:
                if projection == "promotion" and ablation.id not in show and not ablation.is_baseline:
                    continue
                kind = "baseline" if ablation.is_baseline else "ablation"
                add(ablation.id, kind, ablation.display_label, node_state=ablation.state.value,
                    link=f"/object/{ablation.id}", gen=ablation.generation, fam=ablation.family,
                    parameters=f"{ablation.param_count:,}",
                    runs=str(sum(1 for r in runs if r.ablation_id == ablation.id)))
                if projection == "research" and ablation.hypothesis_id in hypotheses:
                    hypothesis = hypotheses[ablation.hypothesis_id]
                    add(hypothesis.id, "hypothesis", f"{hypothesis.id} {hypothesis.title}",
                        node_state=hypothesis.state.value, link=f"/object/{hypothesis.id}",
                        gen=hypothesis.generation, fam=hypothesis.family)
                    edge(hypothesis.id, ablation.id, "tested_by")
                if projection == "research":
                    for run in (r for r in runs if r.ablation_id == ablation.id):
                        add(run.id, "run", f"{run.id} seed {run.seed}", node_state=run.status.value,
                            link=f"/runs/{run.id}", gen=ablation.generation, fam=ablation.family)
                        edge(ablation.id, run.id, "executed")
                        model = models.get(run.model_id) if run.model_id else None
                        if model:
                            add(run.model_id, "model", f"{run.model_id} {model.arch}",
                                link=f"/object/{run.model_id}", gen=ablation.generation,
                                fam=ablation.family, parameters=f"{model.param_count:,}")
                            edge(run.id, run.model_id, "produced")
            for finding in findings:
                for role, ablation_id in (("control", finding.control.ablation_id),
                                          ("intervention", finding.intervention.ablation_id)):
                    ablation = ablation_by_id.get(ablation_id)
                    add(finding.id, "finding", f"{finding.id} {finding.hypothesis_title}",
                        node_state=finding.result.value, link=f"/findings/{finding.id}",
                        effect=f"{finding.effect.difference:+.4g}" if finding.effect else "n/a",
                        fam=ablation.family if ablation else None)
                    edge(finding.id, ablation_id,
                         "compares_against" if role == "control" else "decides")

        return MapResponse(
            projection=projection, projections_available=list(self.MAP_PROJECTIONS),
            nodes=list(nodes.values()), edges=list(edges.values()),
            filters_applied={k: v for k, v in (("generation", generation), ("state", state),
                                               ("family", family)) if v},
        )

    def search_backlog(self) -> SearchBacklog:
        """Search switches imported from the engine, ordered for clean-lineage re-testing."""
        record = self._latest_intake()
        if record is None:
            return SearchBacklog(intake_id=None, switches=[], search_profiles=[], engine_args=[],
                                 legacy_state_counts={})
        counts: Counter = Counter(s.legacy_state.value for s in record.switches)
        return SearchBacklog(intake_id=record.id, switches=record.switches, search_profiles=record.search_profiles,
                             engine_args=record.engine_args, legacy_state_counts=dict(counts))

    def overview(self) -> Overview:
        config = self.config()
        alerts: list[str] = []
        runs = self.store.list("R", verify=False, kind=Run)
        for r in runs:
            try:
                self.store.verify(r)
            except TamperError as exc:
                alerts.append(str(exc))
            if r.status == RunStatus.INVALID:
                failing = ", ".join(c.name for c in r.integrity if c.status == "fail")
                alerts.append(f"{r.id} INVALID ({r.display_label}): {failing}")
        unpinned = sum(1 for r in runs if r.status in TERMINAL_RUN_STATUSES
                       and any(c.name == "engine_pinned" and c.status == "warn" for c in r.integrity))
        if unpinned:
            alerts.append(f"{unpinned} finished run(s) recorded no engine commit (run `cvslab init --engine-repo PATH`)")
        funnel_runs = self.store.list("R", verify=False, kind=FunnelRun)
        lineage = ([self.store.list("S", verify=False, kind=SourceSnapshot)]
                   + [self.store.list("N", verify=False)]
                   + [self.store.list("D", verify=False, kind=Dataset)])
        spent = ([r.compute for r in runs] + [f.compute for f in funnel_runs]
                 + [obj.compute for objs in lineage for obj in objs])
        labels: Counter = Counter()
        for c in spent:
            labels.update(c.labels_generated)
        findings = self.list_findings()
        return Overview(
            active_generation=config["active_generation"], engine_repo=config.get("engine_repo"),
            engine_commit=_git(config.get("engine_repo"), "rev-parse", "HEAD"), lab_commit=_git(LAB_REPO, "rev-parse", "HEAD"),
            baselines=[a for a in self.store.list("A") if a.is_baseline],
            run_counts={s.value: sum(1 for r in runs if r.status == s) for s in RunStatus},
            active_runs=[r for r in runs if r.status in (RunStatus.QUEUED, RunStatus.RUNNING)],
            recent_runs=[r for r in runs if r.status in TERMINAL_RUN_STATUSES][-10:][::-1],
            recent_findings=findings[-5:][::-1],
            open_hypotheses=[h for h in self.store.list("H")
                             if h.state in (EvidenceState.PROPOSED, EvidenceState.RUNNING, EvidenceState.INCONCLUSIVE)],
            datasets=[obj for obj in lineage[2]],
            compute=ComputeTotals(
                cpu_seconds=round(sum(c.cpu_seconds for c in spent), 3), gpu_seconds=round(sum(c.gpu_seconds for c in spent), 3),
                wall_seconds=round(sum(c.wall_seconds for c in spent), 3), search_nodes=sum(c.search_nodes for c in spent),
                oracle_labels=sum(c.oracle_labels for c in spent),
                train_examples_seen=sum(c.train_examples_seen for c in spent), labels_generated=dict(labels),
                legacy_engine_seconds=round(sum(f.total_engine_seconds for f in funnel_runs), 3),
            ),
            object_counts={prefix: len(list((self.store.root / "objects" / folder).glob("*.json")))
                           for prefix, (folder, _) in KINDS.items()},
            legacy=self._legacy_summary(), funnel_runs=funnel_runs,
            integrity_alerts=alerts,
        )

    def run_log(self, run_id: str) -> str:
        run: Run = self.store.get_as(run_id, Run)
        if not run.log_path:
            raise NotFound(f"{run_id} has no log yet")
        return self.store.abs(run.log_path).read_text(encoding="utf-8")

    # ------------------------------------------------------------------ first end-to-end proof

    def run_phase0_proof(self, *, n_games: int = 300, epochs: int = 40, seeds: Sequence[int] = (0, 1, 2),
                         control_width: int = 1, intervention_width: int = 16) -> dict[str, object]:
        """Drive one scientifically trivial experiment through every layer of the lab."""
        source = self.create_fixture_source(n_games=n_games)
        normalization = self.normalize([source.id], name="canonical-v1")
        dataset = self.freeze_dataset(normalization.id, name="phase0-tiny", split_seed=0)
        recipe = self.create_training_recipe(name="raw-nnue-tiny", params={"EPOCHS": epochs},
                                             description="Frozen Phase 0 raw-NNUE trainer settings")
        protocol = self.create_eval_protocol(name="tiny-heldout-v1", dataset_id=dataset.id,
                                             description="Frozen tiny held-out static-eval suite")
        hypothesis = self.create_hypothesis(
            title="Width buys held-out fit at tiny scale",
            statement=f"Under one frozen raw-NNUE contract, widening the hidden layer from H={control_width} to "
                      f"H={intervention_width} lowers held-out loss.",
            metric="test_loss", predicted_direction="decrease", min_effect=0.0, tags=["phase0", "capacity", "proof"],
        )
        baseline = self.register_baseline(name="RAW", dataset_id=dataset.id, training_recipe_id=recipe.id,
                                          eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": control_width},
                                          notes="Frozen raw-NNUE control for the Phase 0 proof")
        ablation = self.create_ablation(baseline_id=baseline.id, overrides={"H": intervention_width},
                                        hypothesis_id=hypothesis.id, notes="Tiny-width intervention")
        runs = self.queue_runs(baseline.id, seeds) + self.queue_runs(ablation.id, seeds)
        executed = [self.execute_run(r.id) for r in runs]
        finding = self.draft_finding(hypothesis_id=hypothesis.id, control_ablation_id=baseline.id,
                                     intervention_ablation_id=ablation.id)
        return {"source": source.id, "normalization": normalization.id, "dataset": dataset.id, "recipe": recipe.id,
                "protocol": protocol.id, "hypothesis": hypothesis.id, "baseline": baseline.id, "ablation": ablation.id,
                "runs": {r.id: r.status.value for r in executed}, "finding": finding.id, "result": finding.result.value}
