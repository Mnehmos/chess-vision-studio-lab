"""Canonical lab schemas.

These pydantic models are the single definition of every lab object. The CLI,
the HTTP API and the GUI (via generated TypeScript types in
``web/src/generated/schemas.ts``) all use them. Regenerate the exported JSON
Schema with ``python -m cvslab schema export`` after any change.

Objects created by the lab itself carry ``origin="lab"``. Evidence brought in
from the pre-lab engine/app system carries ``origin="legacy_catalog"`` or
``origin="funnel"`` and ``lineage="LEGACY"``: it stays permanently inspectable
but is never inherited into the clean lineage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import models_json_schema

SCHEMA_VERSION = 1

ConfigValue = Union[bool, int, float, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _id(prefix: str):
    return Field(pattern=rf"^{prefix}\d{{4,}}$")


class LabModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_serialization_defaults_required=True,
    )


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


class EvidenceState(str, Enum):
    PROPOSED = "PROPOSED"
    RUNNING = "RUNNING"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUPERSEDED = "SUPERSEDED"
    INVALID = "INVALID"


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INVALID = "INVALID"


class PromotionState(str, Enum):
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    READY_FOR_GATE = "READY_FOR_GATE"
    GATING = "GATING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    HOLD = "HOLD"
    INVALID = "INVALID"


TERMINAL_RUN_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.INVALID})


class StateChange(LabModel):
    state: EvidenceState
    at: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------


class Switch(LabModel):
    """A registered experimental variable. Only registered switches may appear in a configuration."""

    key: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    label: str
    kind: Literal["int", "float", "bool", "enum"]
    default: ConfigValue
    choices: list[str] = []
    min: Optional[float] = None
    max: Optional[float] = None
    axis: Literal["capacity", "representation", "compute", "signal", "training"]
    description: str = ""


class ConfigDiff(LabModel):
    key: str
    control: Optional[ConfigValue]
    value: ConfigValue
    axis: str


class Compute(LabModel):
    """Executed compute, recorded as a research variable. All values are measured."""

    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    search_nodes: int = 0
    train_examples_seen: int = 0
    accepted_examples: int = 0
    discarded_examples: int = 0
    labels_generated: dict[str, int] = {}
    oracle_labels: int = 0


class IntegrityCheck(LabModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str


def make_check(name: str, ok: bool, detail: str, *, warn: bool = False) -> IntegrityCheck:
    return IntegrityCheck(name=name, status="pass" if ok else ("warn" if warn else "fail"), detail=detail)


class MetricDef(LabModel):
    name: str
    lower_is_better: bool
    description: str


class MetricValue(LabModel):
    name: str
    value: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    n: int
    kind: Literal["measured", "derived", "estimate"]
    split: str


class EpochLog(LabModel):
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    examples_per_second: float
    elapsed_seconds: float


class RunEnvironment(LabModel):
    python: str
    numpy: str
    platform: str
    machine: str
    processor: str
    cpu_count: int
    lab_commit: Optional[str]
    lab_dirty: Optional[bool]
    engine_repo: Optional[str]
    engine_commit: Optional[str]


class SplitManifest(LabModel):
    name: str
    count: int
    group_count: int
    path: str
    file_hash: str
    record_ids_hash: str


class LabelSetRef(LabModel):
    """One append-only L2 label file pinned into a dataset manifest."""

    path: str
    file_hash: str
    rows: int
    families: dict[str, int]
    authorities: dict[str, int]
    producers: dict[str, int]
    registry_version: int
    label_schema_version: int
    policy_hash: Optional[str] = None


class StackPolicy(str, Enum):
    ALL = "all"
    FIXED_ROWS = "fixed_rows"
    FRACTION = "fraction"
    BALANCE = "balance"


class StackArm(LabModel):
    """One component of a dataset stack: a filtered subpopulation plus an explicit sampling policy."""

    name: str
    filter: dict[str, ConfigValue] = {}
    policy: StackPolicy = StackPolicy.ALL
    rows: Optional[int] = None
    fraction: Optional[float] = None
    balance_bucket: Optional[str] = None
    balance_cap: Optional[int] = None
    seed: int = 0


class ArtifactLocation(LabModel):
    root: str
    path: str


FileVerification = Literal["sha256-match", "sha256-mismatch", "size-match", "size-mismatch", "missing", "unverifiable"]


# ---------------------------------------------------------------------------
# Lab-originated objects
# ---------------------------------------------------------------------------


class Hypothesis(LabModel):
    id: str = _id("H")
    schema_version: int = SCHEMA_VERSION
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    family: str
    generation: str = Field(pattern=r"^G\d{2,}$")
    metric: str
    predicted_direction: Literal["decrease", "increase"]
    min_effect: float = Field(ge=0)
    tags: list[str] = []
    state: EvidenceState = EvidenceState.PROPOSED
    state_history: list[StateChange] = []
    created_at: str


class SourceSnapshot(LabModel):
    """S#### — immutable L0 source snapshot."""

    id: str = _id("S")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["lab"] = "lab"
    name: str
    locality: Literal["local", "remote", "external_reference"]
    source_origin: str
    generator: Optional[dict[str, ConfigValue]] = None
    license: str
    importer: str
    importer_version: int
    path: str
    content_hash: str
    row_count: int
    game_count: int
    label_authorities: list[str]
    compute: Compute
    created_at: str


class Normalization(LabModel):
    """N#### — versioned standardization of sources into canonical records."""

    id: str = _id("N")
    schema_version: int = SCHEMA_VERSION
    name: str
    source_ids: list[str]
    source_hashes: dict[str, str]
    normalizer: str
    normalizer_version: int
    canonical_schema_version: int
    settings: dict[str, ConfigValue]
    dedup_policy: str
    recipe_hash: str
    path: str
    output_hash: str
    record_count: int
    duplicate_count: int
    rejected_count: int
    compute: Compute
    created_at: str


class Dataset(LabModel):
    """D#### — frozen dataset manifest: a reproducible selection over canonical records."""

    id: str = _id("D")
    schema_version: int = SCHEMA_VERSION
    name: str
    normalization_id: str
    normalization_recipe_hash: str
    source_ids: list[str]
    canonical_schema_version: int
    label_requirements: list[str]
    selection: dict[str, ConfigValue]
    sampling_policy: str
    dedup_policy: str
    split_policy: dict[str, ConfigValue]
    splits: list[SplitManifest]
    stack: list[StackArm] = []
    label_sets: list[LabelSetRef] = []
    counts: dict[str, int]
    coverage: dict[str, dict[str, int]]
    label_provenance: dict[str, int]
    parent_id: Optional[str] = None
    frozen: bool = True
    campaign: dict[str, object] = {}
    manifest_hash: str
    compute: Compute
    created_at: str


class TrainingRecipe(LabModel):
    """T#### — frozen trainer identity and training/signal/compute settings."""

    id: str = _id("T")
    schema_version: int = SCHEMA_VERSION
    name: str
    family: str
    generation: str
    trainer: str
    trainer_version: int
    params: dict[str, ConfigValue]
    description: str = ""
    recipe_hash: str
    created_at: str


class EvaluationProtocol(LabModel):
    """E#### — frozen evaluation instrument pinned to one dataset manifest."""

    id: str = _id("E")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["lab"] = "lab"
    name: str
    family: str
    generation: str
    dataset_id: str
    dataset_manifest_hash: str
    split: str
    metrics: list[MetricDef]
    params: dict[str, ConfigValue]
    bootstrap_samples: int
    bootstrap_seed: int
    protocol_version: int
    non_claims: list[str]
    description: str = ""
    protocol_hash: str
    created_at: str


class Ablation(LabModel):
    """A#### — a scientific intervention relative to a frozen baseline (or the baseline itself)."""

    id: str = _id("A")
    schema_version: int = SCHEMA_VERSION
    family: str
    generation: str
    baseline_name: str
    is_baseline: bool
    control_id: Optional[str]
    hypothesis_id: Optional[str]
    overrides: dict[str, ConfigValue]
    effective_config: dict[str, ConfigValue]
    config_hash: str
    identity_hash: str
    diff: list[ConfigDiff]
    param_count: int
    parameter_shapes: dict[str, list[int]]
    display_label: str
    dataset_id: str
    training_recipe_id: str
    eval_protocol_id: str
    notes: str = ""
    # declared at creation and part of `identity_hash`: this ablation intentionally trains
    # against one TargetSpec and is measured against another. Empty = divergence refused.
    supervision_divergence: str = ""
    state: EvidenceState = EvidenceState.PROPOSED
    state_history: list[StateChange] = []
    created_at: str


class ModelArtifact(LabModel):
    """M#### — a serialized model produced by exactly one lab run."""

    id: str = _id("M")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["lab"] = "lab"
    lineage: Literal["LAB"] = "LAB"
    run_id: str
    ablation_id: str
    family: str
    arch: str
    format: str
    path: str
    artifact_hash: str
    param_count: int
    parameter_shapes: dict[str, list[int]]
    dataset_id: str
    training_recipe_id: str
    warm_start_model_id: Optional[str] = None
    promotable: bool = False
    engine_loadable: bool
    created_at: str


class Run(LabModel):
    """R#### — one execution of an ablation. Immutable once COMPLETED or INVALID."""

    id: str = _id("R")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["lab"] = "lab"
    ablation_id: str
    hypothesis_id: Optional[str]
    display_label: str
    seed: int
    status: RunStatus
    effective_config: dict[str, ConfigValue]
    config_hash: str
    identity_hash: str
    param_count: int
    dataset_id: str
    dataset_manifest_hash: str
    training_recipe_id: str
    recipe_hash: str
    eval_protocol_id: str
    protocol_hash: str
    # evaluation may use a DIFFERENT (common) dataset than training; explicit so no
    # future reader assumes dataset_id is also the evaluation corpus
    eval_dataset_id: Optional[str] = None
    eval_dataset_manifest_hash: Optional[str] = None
    # what taught the model and what judged it, independently and on the run itself:
    # a reader of R#### must not have to reconstruct this from a recipe and a protocol
    train_target_spec_hash: Optional[str] = None
    eval_target_spec_hash: Optional[str] = None
    supervision_divergence: str = ""
    queued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    environment: Optional[RunEnvironment] = None
    compute: Compute = Compute()
    training_curve: list[EpochLog] = []
    metrics: list[MetricValue] = []
    integrity: list[IntegrityCheck] = []
    model_id: Optional[str] = None
    artifact_hashes: dict[str, str] = {}
    log_path: Optional[str] = None
    error: Optional[str] = None
    record_hash: Optional[str] = None


class ArmSummary(LabModel):
    ablation_id: str
    display_label: str
    param_count: int
    run_ids: list[str]
    invalid_run_ids: list[str]
    seeds: list[int]
    metric_mean: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    cpu_seconds: float


class Effect(LabModel):
    metric: str
    difference: float
    ci_low: float
    ci_high: float
    n: int
    method: str


class PromotionGate(LabModel):
    name: str
    status: Literal["pass", "hold", "fail", "invalid", "missing"]
    detail: str


class Finding(LabModel):
    """F#### — the standard finding card. Immutable; a revision supersedes, never edits."""

    id: str = _id("F")
    schema_version: int = SCHEMA_VERSION
    hypothesis_id: str
    hypothesis_title: str
    hypothesis_statement: str
    eval_protocol_id: str
    dataset_id: str
    metric: str
    predicted_direction: Literal["decrease", "increase"]
    min_effect: float
    control: ArmSummary
    intervention: ArmSummary
    frozen_controls: list[str]
    intervention_diff: list[ConfigDiff]
    evidence: list[str]
    effect: Optional[Effect]
    result: EvidenceState
    decision_rule: str
    integrity: list[IntegrityCheck]
    promotion_state: PromotionState
    promotion_gates: list[PromotionGate]
    promotion_consequence: str
    interpretation: str
    non_claims: list[str]
    next_experiment: str
    supersedes: Optional[str] = None
    created_at: str
    record_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Legacy evidence (imported, never inherited)
# ---------------------------------------------------------------------------


class LegacyModel(LabModel):
    """M#### imported from the legacy catalog: inspectable, never promotable, never a warm start by default."""

    id: str = _id("M")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["legacy_catalog"]
    lineage: Literal["LEGACY"] = "LEGACY"
    intake_id: str
    name: str
    sha256: str
    bytes: int
    model_kind: str
    serialized_param_count: Optional[int]
    metadata: dict[str, Optional[ConfigValue]]
    role: str
    role_evidence: Optional[str]
    provenance_complete: bool
    promotable: Literal[False] = False
    locations: list[ArtifactLocation]
    verification: FileVerification
    created_at: str


class LegacyFile(LabModel):
    root: str
    path: str
    sha256: str
    bytes: int
    rows: int


class LegacySource(LabModel):
    id: str = _id("S")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["legacy_catalog"]
    lineage: Literal["LEGACY"] = "LEGACY"
    intake_id: str
    name: str
    content_sha256: str
    files: int
    rows: int
    bytes: int
    sample_fields: list[str]
    field_authority: dict[str, str]
    note: Optional[str]
    bytes_availability: str
    file_manifest: list[LegacyFile]
    verification: FileVerification
    verification_detail: str
    created_at: str


class LegacyEvaluation(LabModel):
    id: str = _id("E")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["legacy_catalog"]
    lineage: Literal["LEGACY"] = "LEGACY"
    intake_id: str
    name: str
    kind: str
    sha256: str
    bytes: int
    location: ArtifactLocation
    verification: FileVerification
    created_at: str


class LegacyEvidence(LabModel):
    """R#### imported gate or measurement record from the legacy engine."""

    id: str = _id("R")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["legacy_catalog"]
    lineage: Literal["LEGACY"] = "LEGACY"
    intake_id: str
    type: str
    location: ArtifactLocation
    sha256: str
    date: Optional[str] = None
    baseline_id: Optional[str] = None
    candidate_id: Optional[str] = None
    games: Optional[int] = None
    wdl: Optional[list[int]] = None
    llr: Optional[float] = None
    boundary: Optional[str] = None
    decision: Optional[str] = None
    recorded_state: str
    evidence_state: Optional[EvidenceState]
    is_measurement: bool
    note: Optional[str] = None
    switch_ids: list[str] = []
    verification: FileVerification
    created_at: str
    record_hash: Optional[str] = None


class FunnelTier(LabModel):
    name: str
    positions: int
    engine_seconds: float
    ms_per_position: float
    nodes: int
    wall_seconds: Optional[float]
    provenance_classes: list[str]


class FunnelArm(LabModel):
    name: str
    n: int
    deep_nodes: int
    deep_engine_seconds: float
    informative: int
    informative_rate: float
    move_change_rate: float
    abs_shallow_deep_delta_cp: dict[str, float]


class FunnelOracleArm(LabModel):
    name: str
    n: int
    disagreement_rate: float
    move_agreement_rate: float
    abs_cp: dict[str, float]


class FunnelFile(LabModel):
    name: str
    sha256: str
    bytes: int
    rows: Optional[int]
    copied_to: Optional[str]


class FunnelRun(LabModel):
    """R#### imported information-gain labeling funnel run (engine `training/funnel`)."""

    id: str = _id("R")
    schema_version: int = SCHEMA_VERSION
    origin: Literal["funnel"]
    lineage: Literal["LEGACY"] = "LEGACY"
    lineage_note: str
    position_pool: Optional[str] = None
    triage_policy_version: Optional[str] = None
    triage_policy_hash: Optional[str] = None
    name: str
    run_dir: str
    funnel_schema_version: int
    funnel_config_version: int
    prioritizer_version: str
    config_sha256: str
    source_created_at: str
    tooling_commit: Optional[str]
    tooling_dirty: Optional[bool]
    engine_binary_sha256: Optional[str]
    engine_artifacts: dict[str, str]
    engine_artifact_model_ids: dict[str, str]
    engine_options: dict[str, ConfigValue]
    stockfish_sha256: Optional[str]
    taxonomy_version: Optional[int]
    taxonomy_sha256: Optional[str]
    facts_registry_version: Optional[int]
    provenance_classes: dict[str, str]
    sample: dict[str, ConfigValue]
    tiers: list[FunnelTier]
    total_engine_seconds: float
    provenance_records: dict[str, int]
    coverage: dict[str, dict[str, int]]
    underrepresented: list[str]
    selection: dict[str, int]
    priority_histogram: list[int]
    priority_reasons: dict[str, int]
    arms: list[FunnelArm]
    oracle_arms: list[FunnelOracleArm]
    audit_miss: dict[str, float]
    experiment_question: str
    informative_yield_ratio: Optional[float]
    experiment_caveat: str
    research_state: Optional[EvidenceState]
    research_state_source: Optional[str]
    compute: Compute
    compute_note: str
    files: list[FunnelFile]
    integrity: list[IntegrityCheck]
    created_at: str
    record_hash: Optional[str] = None


class LegacyRoot(LabModel):
    name: str
    path: Optional[str]
    catalog_commit: Optional[str]
    current_commit: Optional[str]
    dirty: Optional[bool]
    remote: Optional[str]


class LegacyEngine(LabModel):
    id: str
    display_name: Optional[str]
    generation: Optional[int]
    status: Optional[str]
    architecture: Optional[str]
    search_profile: Optional[str]
    main_net: Optional[str]
    main_net_model_id: Optional[str]


class SearchProfile(LabModel):
    name: str
    status: Optional[str]
    args: list[str]
    effective_options: dict[str, ConfigValue]
    notes: Optional[str]


class SwitchGateRecord(LabModel):
    record: str
    evidence_id: Optional[str]
    date: Optional[str]
    games: Optional[int]
    llr: Optional[float]
    boundary: Optional[str]
    decision: Optional[str]
    legacy_state: str


class SearchSwitch(LabModel):
    """An engine search switch as a clean-lineage backlog item: legacy evidence sets priority, not truth."""

    id: str
    kind: Literal["toggle", "value"]
    field: str
    on_flag: Optional[str]
    off_flag: Optional[str]
    value_flag: Optional[str]
    default: Optional[ConfigValue]
    doc: Optional[str]
    legacy_state: EvidenceState
    current: Optional[SwitchGateRecord]
    history: list[SwitchGateRecord]
    clean_state: EvidenceState = EvidenceState.PROPOSED
    backlog_tier: int
    backlog_reason: str


class LegacyChampion(LabModel):
    engine_id: str
    role: str
    search_profile: Optional[str]
    live_bot_flags: list[str]
    main_net_model_id: Optional[str]
    helper_net_model_id: Optional[str]
    source_path: str
    sha256: str
    open_followups: list[str]


class IntakeRecord(LabModel):
    """I#### — one import of the legacy catalog: what was verified, assigned and reused."""

    id: str = _id("I")
    schema_version: int = SCHEMA_VERSION
    tool: str
    catalog_dir: str
    catalog_generated_at: str
    catalog_file_hashes: dict[str, str]
    roots: list[LegacyRoot]
    assigned: dict[str, list[str]]
    reused: dict[str, int]
    engines: list[LegacyEngine]
    search_profiles: list[SearchProfile]
    switches: list[SearchSwitch]
    engine_args: list[dict[str, str]]
    unmatched_gate_records: int
    champion: Optional[LegacyChampion]
    flagship_model_ids: list[str]
    findings: list[str]
    integrity: list[IntegrityCheck]
    compute: Compute
    created_at: str
    record_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Views returned by the service (rendered by CLI and GUI)
# ---------------------------------------------------------------------------


class AblationPreview(LabModel):
    baseline_id: str
    baseline_label: str
    overrides: dict[str, ConfigValue]
    effective_config: dict[str, ConfigValue]
    diff: list[ConfigDiff]
    param_count: int
    control_param_count: int
    parameter_shapes: dict[str, list[int]]
    display_label: str
    identity_hash: str
    supervision_divergence: str = ""
    warnings: list[str]
    duplicate_of: Optional[str]
    can_create: bool


class MatrixCell(LabModel):
    key: str
    value: Optional[ConfigValue]
    changed: bool


class MatrixRow(LabModel):
    ablation_id: str
    display_label: str
    is_baseline: bool
    control_id: Optional[str]
    generation: str
    baseline_name: str
    dataset_id: str
    param_count: int
    state: EvidenceState
    cells: list[MatrixCell]
    run_counts: dict[str, int]
    metric_mean: Optional[float]
    finding_ids: list[str]


class MatrixResponse(LabModel):
    family: str
    metric: str
    switches: list[Switch]
    rows: list[MatrixRow]


class ScalingPoint(LabModel):
    run_id: str
    ablation_id: str
    display_label: str
    family: str
    generation: str
    representation: str
    is_baseline: bool
    status: RunStatus
    seed: int
    param_count: int
    search_nodes: int
    cpu_seconds: float
    value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    kind: Optional[Literal["measured", "derived", "estimate"]]


class LegacyModelPoint(LabModel):
    model_id: str
    name: str
    role: str
    model_kind: str
    serialized_param_count: Optional[int]
    provenance_complete: bool


class ScalingResponse(LabModel):
    metric: str
    lower_is_better: bool
    available_metrics: list[MetricDef]
    points: list[ScalingPoint]
    legacy_models: list[LegacyModelPoint]


class SearchBacklog(LabModel):
    intake_id: Optional[str]
    switches: list[SearchSwitch]
    search_profiles: list[SearchProfile]
    engine_args: list[dict[str, str]]
    legacy_state_counts: dict[str, int]


class CatalogRecord(LabModel):
    """One canonical record rendered in the corpus catalog, labels joined and memberships resolved."""

    record_id: str
    group: str
    source_id: str
    source_row: int
    ply: int
    fen: str
    epd: str
    stm: str
    phase: str
    material: str
    result: Optional[Union[bool, int, float, str]]
    labels: list[dict[str, object]]
    label_tiers: dict[str, int]
    memberships: dict[str, list[str]]


class CatalogResponse(LabModel):
    normalization_id: str
    record_count: int
    unique_count: int
    duplicate_count: int
    label_count: int
    filters_applied: dict[str, ConfigValue]
    facets: dict[str, dict[str, int]]
    page: list[CatalogRecord]
    page_offset: int
    page_size: int
    total_matching: int


class MapNode(LabModel):
    """One spatial object on the Lab Map; `detail` is display evidence, `link` the 2D page."""

    id: str
    kind: str
    label: str
    state: Optional[str] = None
    link: str = ""
    detail: dict[str, object] = {}


class MapEdge(LabModel):
    src: str
    dst: str
    kind: str


class MapResponse(LabModel):
    projection: str
    projections_available: list[str]
    nodes: list[MapNode]
    edges: list[MapEdge]
    filters_applied: dict[str, object]


class ArmPreview(LabModel):
    name: str
    policy: str
    filter: dict[str, ConfigValue]
    available: int
    effective: int


class StackPreview(LabModel):
    normalization_id: str
    arms: list[ArmPreview]
    effective_total: int
    unique_records: int
    distributions: dict[str, dict[str, int]]


class DatasetSplitChange(LabModel):
    dataset_id: str
    surviving_records: int
    split_changes: int


class MigrationDiff(LabModel):
    from_normalization_id: str
    to_normalization_id: str
    records_from: int
    records_to: int
    added: int
    removed: int
    changed: int
    duplicate_delta: int
    rejected_delta: int
    label_changes: dict[str, dict[str, int]]
    coverage_from: dict[str, dict[str, int]]
    coverage_to: dict[str, dict[str, int]]
    source_contribution_from: dict[str, int]
    source_contribution_to: dict[str, int]
    split_changes: list[DatasetSplitChange]
    affected_dataset_ids: list[str]
    affected_run_ids: list[str]
    notes: list[str]


class ComputeTotals(LabModel):
    cpu_seconds: float
    gpu_seconds: float
    wall_seconds: float
    search_nodes: int
    oracle_labels: int
    train_examples_seen: int
    labels_generated: dict[str, int]
    legacy_engine_seconds: float


class LegacySummary(LabModel):
    intake_id: Optional[str]
    catalog_generated_at: Optional[str]
    production_lineage: str
    champion: Optional[LegacyChampion]
    roots: list[LegacyRoot]
    engines: list[LegacyEngine]
    flagship_model_ids: list[str]
    models: int
    models_provenance_incomplete: int
    model_roles: dict[str, int]
    sources: int
    source_rows: int
    source_bytes: int
    evaluations: int
    evidence: int
    evidence_states: dict[str, int]
    switch_legacy_states: dict[str, int]
    findings: list[str]


class Overview(LabModel):
    active_generation: str
    engine_repo: Optional[str]
    engine_commit: Optional[str]
    lab_commit: Optional[str]
    baselines: list[Ablation]
    run_counts: dict[str, int]
    active_runs: list[Run]
    recent_runs: list[Run]
    recent_findings: list[Finding]
    open_hypotheses: list[Hypothesis]
    datasets: list[Dataset]
    compute: ComputeTotals
    object_counts: dict[str, int]
    legacy: LegacySummary
    funnel_runs: list[FunnelRun]
    integrity_alerts: list[str]


# ---------------------------------------------------------------------------
# Requests (shared by CLI arguments, API bodies and GUI forms)
# ---------------------------------------------------------------------------


class HypothesisCreate(LabModel):
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    metric: str = "test_loss"
    predicted_direction: Literal["decrease", "increase"] = "decrease"
    min_effect: float = Field(default=0.0, ge=0)
    family: str = "NNUE"
    generation: Optional[str] = None
    tags: list[str] = []


class AblationRequest(LabModel):
    baseline_id: str
    overrides: dict[str, ConfigValue] = {}
    hypothesis_id: Optional[str] = None
    notes: str = ""
    supervision_divergence: str = ""


class RunQueueRequest(LabModel):
    ablation_id: str
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)


class FindingRequest(LabModel):
    hypothesis_id: str
    control_ablation_id: str
    intervention_ablation_id: str
    interpretation: Optional[str] = None
    next_experiment: Optional[str] = None
    non_claims: Optional[list[str]] = None


class LabelsAppendRequest(LabModel):
    """Append a new label set to a canonical corpus; existing records and labels are never modified."""

    normalization_id: str
    family: str
    producer: str
    authority: str
    registry_version: int = 1
    pov: str = "white"
    rows: list[dict[str, object]] = Field(min_length=1)


class StackFreezeRequest(LabModel):
    normalization_id: str
    name: str
    arms: list[dict[str, object]] = Field(default_factory=list)
    fractions: list[float] = [0.8, 0.1, 0.1]
    split_seed: int = 0
    required_labels: list[str] = ["eval_cp"]


class MigrateRequest(LabModel):
    from_normalization_id: str
    name: str
    settings_overrides: dict[str, ConfigValue] = {}
    dedup: str = "exact-epd-keep-first"


class RebuildDatasetRequest(LabModel):
    new_normalization_id: str


# ---------------------------------------------------------------------------
# Registry and export
# ---------------------------------------------------------------------------

# prefix -> (storage folder, models discriminated by `origin` when more than one)
KINDS: dict[str, tuple[str, tuple[type[LabModel], ...]]] = {
    "H": ("hypotheses", (Hypothesis,)),
    "A": ("ablations", (Ablation,)),
    "M": ("models", (ModelArtifact, LegacyModel)),
    "S": ("sources", (SourceSnapshot, LegacySource)),
    "N": ("normalizations", (Normalization,)),
    "D": ("datasets", (Dataset,)),
    "T": ("training_recipes", (TrainingRecipe,)),
    "E": ("evaluation_protocols", (EvaluationProtocol, LegacyEvaluation)),
    "R": ("runs", (Run, LegacyEvidence, FunnelRun)),
    "F": ("findings", (Finding,)),
    "I": ("intakes", (IntakeRecord,)),
}

# Only the evidence state of hypotheses and ablations may change after creation.
MUTABLE_STATE_PREFIXES = frozenset({"H", "A"})

OBJECT_MODELS = [cls for _, classes in KINDS.values() for cls in classes]
VIEW_MODELS = [Switch, AblationPreview, MatrixResponse, ScalingResponse, SearchBacklog, Overview,
               CatalogResponse, StackPreview, MigrationDiff, MapResponse]
REQUEST_MODELS = [HypothesisCreate, AblationRequest, RunQueueRequest, FindingRequest,
                  LabelsAppendRequest, StackFreezeRequest, MigrateRequest, RebuildDatasetRequest]


def export_json_schema() -> dict:
    models = OBJECT_MODELS + VIEW_MODELS + REQUEST_MODELS
    _, top = models_json_schema(
        [(m, "validation" if m in REQUEST_MODELS else "serialization") for m in models],
        ref_template="#/$defs/{model}",
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CVSLab",
        "description": f"CVS Lab canonical schemas, schema_version {SCHEMA_VERSION}. Generated by `python -m cvslab schema export`; do not edit.",
        "type": "object",
        "properties": {m.__name__: {"$ref": f"#/$defs/{m.__name__}"} for m in models},
        "required": [m.__name__ for m in models],
        "additionalProperties": False,
        "$defs": top["$defs"],
    }
