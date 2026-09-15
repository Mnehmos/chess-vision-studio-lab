// Generated from schemas/cvslab.schema.json by `npm run gen:types` — do not edit.

export type Id = string;
export type SchemaVersion = number;
export type Title = string;
export type Statement = string;
export type Family = string;
export type Generation = string;
export type Metric = string;
export type PredictedDirection = "decrease" | "increase";
export type MinEffect = number;
export type Tags = string[];
export type EvidenceState =
  "PROPOSED" | "RUNNING" | "SUPPORTED" | "REJECTED" | "INCONCLUSIVE" | "SUPERSEDED" | "INVALID";
export type EvidenceState1 =
  "PROPOSED" | "RUNNING" | "SUPPORTED" | "REJECTED" | "INCONCLUSIVE" | "SUPERSEDED" | "INVALID";
export type At = string;
export type Reason = string;
export type StateHistory = StateChange[];
export type CreatedAt = string;
export type Id1 = string;
export type SchemaVersion1 = number;
export type Family1 = string;
export type Generation1 = string;
export type BaselineName = string;
export type IsBaseline = boolean;
export type ControlId = string | null;
export type HypothesisId = string | null;
export type ConfigHash = string;
export type IdentityHash = string;
export type Key = string;
export type Control = boolean | number | string | null;
export type Value = boolean | number | string;
export type Axis = string;
export type Diff = ConfigDiff[];
export type ParamCount = number;
export type DisplayLabel = string;
export type DatasetId = string;
export type TrainingRecipeId = string;
export type EvalProtocolId = string;
export type Notes = string;
export type EvidenceState2 =
  "PROPOSED" | "RUNNING" | "SUPPORTED" | "REJECTED" | "INCONCLUSIVE" | "SUPERSEDED" | "INVALID";
export type StateHistory1 = StateChange[];
export type CreatedAt1 = string;
export type Id2 = string;
export type SchemaVersion2 = number;
export type Origin = "lab";
export type Lineage = "LAB";
export type RunId = string;
export type AblationId = string;
export type Family2 = string;
export type Arch = string;
export type Format = string;
export type Path = string;
export type ArtifactHash = string;
export type ParamCount1 = number;
export type DatasetId1 = string;
export type TrainingRecipeId1 = string;
export type WarmStartModelId = string | null;
export type Promotable = boolean;
export type EngineLoadable = boolean;
export type CreatedAt2 = string;
export type Id3 = string;
export type SchemaVersion3 = number;
export type Origin1 = "legacy_catalog";
export type Lineage1 = "LEGACY";
export type IntakeId = string;
export type Name = string;
export type Sha256 = string;
export type Bytes = number;
export type ModelKind = string;
export type SerializedParamCount = number | null;
export type Role = string;
export type RoleEvidence = string | null;
export type ProvenanceComplete = boolean;
export type Promotable1 = false;
export type Root = string;
export type Path1 = string;
export type Locations = ArtifactLocation[];
export type Verification =
  "sha256-match" | "sha256-mismatch" | "size-match" | "size-mismatch" | "missing" | "unverifiable";
export type CreatedAt3 = string;
export type Id4 = string;
export type SchemaVersion4 = number;
export type Origin2 = "lab";
export type Name1 = string;
export type Locality = "local" | "remote" | "external_reference";
export type SourceOrigin = string;
export type Generator = {
  [k: string]: boolean | number | string;
} | null;
export type License = string;
export type Importer = string;
export type ImporterVersion = number;
export type Path2 = string;
export type ContentHash = string;
export type RowCount = number;
export type GameCount = number;
export type LabelAuthorities = string[];
export type CpuSeconds = number;
export type WallSeconds = number;
export type GpuSeconds = number;
export type SearchNodes = number;
export type TrainExamplesSeen = number;
export type AcceptedExamples = number;
export type DiscardedExamples = number;
export type OracleLabels = number;
export type CreatedAt4 = string;
export type Id5 = string;
export type SchemaVersion5 = number;
export type Origin3 = "legacy_catalog";
export type Lineage2 = "LEGACY";
export type IntakeId1 = string;
export type Name2 = string;
export type ContentSha256 = string;
export type Files = number;
export type Rows = number;
export type Bytes1 = number;
export type SampleFields = string[];
export type Note = string | null;
export type BytesAvailability = string;
export type Root1 = string;
export type Path3 = string;
export type Sha2561 = string;
export type Bytes2 = number;
export type Rows1 = number;
export type FileManifest = LegacyFile[];
export type Verification1 =
  "sha256-match" | "sha256-mismatch" | "size-match" | "size-mismatch" | "missing" | "unverifiable";
export type VerificationDetail = string;
export type CreatedAt5 = string;
export type Id6 = string;
export type SchemaVersion6 = number;
export type Name3 = string;
export type SourceIds = string[];
export type Normalizer = string;
export type NormalizerVersion = number;
export type CanonicalSchemaVersion = number;
export type DedupPolicy = string;
export type RecipeHash = string;
export type Path4 = string;
export type OutputHash = string;
export type RecordCount = number;
export type DuplicateCount = number;
export type RejectedCount = number;
export type CreatedAt6 = string;
export type Id7 = string;
export type SchemaVersion7 = number;
export type Name4 = string;
export type NormalizationId = string;
export type NormalizationRecipeHash = string;
export type SourceIds1 = string[];
export type CanonicalSchemaVersion1 = number;
export type LabelRequirements = string[];
export type SamplingPolicy = string;
export type DedupPolicy1 = string;
export type Name5 = string;
export type Count = number;
export type GroupCount = number;
export type Path5 = string;
export type FileHash = string;
export type RecordIdsHash = string;
export type Splits = SplitManifest[];
export type Name6 = string;
export type StackPolicy = "all" | "fixed_rows" | "fraction" | "balance";
export type Rows2 = number | null;
export type Fraction = number | null;
export type BalanceBucket = string | null;
export type BalanceCap = number | null;
export type Seed = number;
export type Stack = StackArm[];
export type Path6 = string;
export type FileHash1 = string;
export type Rows3 = number;
export type RegistryVersion = number;
export type LabelSchemaVersion = number;
export type LabelSets = LabelSetRef[];
export type ParentId = string | null;
export type Frozen = boolean;
export type ManifestHash = string;
export type CreatedAt7 = string;
export type Id8 = string;
export type SchemaVersion8 = number;
export type Name7 = string;
export type Family3 = string;
export type Generation2 = string;
export type Trainer = string;
export type TrainerVersion = number;
export type Description = string;
export type RecipeHash1 = string;
export type CreatedAt8 = string;
export type Id9 = string;
export type SchemaVersion9 = number;
export type Origin4 = "lab";
export type Name8 = string;
export type Family4 = string;
export type Generation3 = string;
export type DatasetId2 = string;
export type DatasetManifestHash = string;
export type Split = string;
export type Name9 = string;
export type LowerIsBetter = boolean;
export type Description1 = string;
export type Metrics = MetricDef[];
export type BootstrapSamples = number;
export type BootstrapSeed = number;
export type ProtocolVersion = number;
export type NonClaims = string[];
export type Description2 = string;
export type ProtocolHash = string;
export type CreatedAt9 = string;
export type Id10 = string;
export type SchemaVersion10 = number;
export type Origin5 = "legacy_catalog";
export type Lineage3 = "LEGACY";
export type IntakeId2 = string;
export type Name10 = string;
export type Kind = string;
export type Sha2562 = string;
export type Bytes3 = number;
export type Verification2 =
  "sha256-match" | "sha256-mismatch" | "size-match" | "size-mismatch" | "missing" | "unverifiable";
export type CreatedAt10 = string;
export type Id11 = string;
export type SchemaVersion11 = number;
export type Origin6 = "lab";
export type AblationId1 = string;
export type HypothesisId1 = string | null;
export type DisplayLabel1 = string;
export type Seed1 = number;
export type RunStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "INVALID";
export type ConfigHash1 = string;
export type IdentityHash1 = string;
export type ParamCount2 = number;
export type DatasetId3 = string;
export type DatasetManifestHash1 = string;
export type TrainingRecipeId2 = string;
export type RecipeHash2 = string;
export type EvalProtocolId1 = string;
export type ProtocolHash1 = string;
export type QueuedAt = string;
export type StartedAt = string | null;
export type FinishedAt = string | null;
export type Python = string;
export type Numpy = string;
export type Platform = string;
export type Machine = string;
export type Processor = string;
export type CpuCount = number;
export type LabCommit = string | null;
export type LabDirty = boolean | null;
export type EngineRepo = string | null;
export type EngineCommit = string | null;
export type Epoch = number;
export type TrainLoss = number | null;
export type ValLoss = number | null;
export type ExamplesPerSecond = number;
export type ElapsedSeconds = number;
export type TrainingCurve = EpochLog[];
export type Name11 = string;
export type Value1 = number;
export type CiLow = number | null;
export type CiHigh = number | null;
export type N = number;
export type Kind1 = "measured" | "derived" | "estimate";
export type Split1 = string;
export type Metrics1 = MetricValue[];
export type Name12 = string;
export type Status = "pass" | "warn" | "fail";
export type Detail = string;
export type Integrity = IntegrityCheck[];
export type ModelId = string | null;
export type LogPath = string | null;
export type Error = string | null;
export type RecordHash = string | null;
export type Id12 = string;
export type SchemaVersion12 = number;
export type Origin7 = "legacy_catalog";
export type Lineage4 = "LEGACY";
export type IntakeId3 = string;
export type Type = string;
export type Sha2563 = string;
export type Date = string | null;
export type BaselineId = string | null;
export type CandidateId = string | null;
export type Games = number | null;
export type Wdl = number[] | null;
export type Llr = number | null;
export type Boundary = string | null;
export type Decision = string | null;
export type RecordedState = string;
export type IsMeasurement = boolean;
export type Note1 = string | null;
export type SwitchIds = string[];
export type Verification3 =
  "sha256-match" | "sha256-mismatch" | "size-match" | "size-mismatch" | "missing" | "unverifiable";
export type CreatedAt11 = string;
export type RecordHash1 = string | null;
export type Id13 = string;
export type SchemaVersion13 = number;
export type Origin8 = "funnel";
export type Lineage5 = "LEGACY";
export type LineageNote = string;
export type Name13 = string;
export type RunDir = string;
export type FunnelSchemaVersion = number;
export type FunnelConfigVersion = number;
export type PrioritizerVersion = string;
export type ConfigSha256 = string;
export type SourceCreatedAt = string;
export type ToolingCommit = string | null;
export type ToolingDirty = boolean | null;
export type EngineBinarySha256 = string | null;
export type StockfishSha256 = string | null;
export type TaxonomyVersion = number | null;
export type TaxonomySha256 = string | null;
export type FactsRegistryVersion = number | null;
export type Name14 = string;
export type Positions = number;
export type EngineSeconds = number;
export type MsPerPosition = number;
export type Nodes = number;
export type WallSeconds1 = number | null;
export type ProvenanceClasses1 = string[];
export type Tiers = FunnelTier[];
export type TotalEngineSeconds = number;
export type Underrepresented = string[];
export type PriorityHistogram = number[];
export type Name15 = string;
export type N1 = number;
export type DeepNodes = number;
export type DeepEngineSeconds = number;
export type Informative = number;
export type InformativeRate = number;
export type MoveChangeRate = number;
export type Arms = FunnelArm[];
export type Name16 = string;
export type N2 = number;
export type DisagreementRate = number;
export type MoveAgreementRate = number;
export type OracleArms = FunnelOracleArm[];
export type ExperimentQuestion = string;
export type InformativeYieldRatio = number | null;
export type ExperimentCaveat = string;
export type ResearchStateSource = string | null;
export type ComputeNote = string;
export type Name17 = string;
export type Sha2564 = string;
export type Bytes4 = number;
export type Rows4 = number | null;
export type CopiedTo = string | null;
export type Files1 = FunnelFile[];
export type Integrity1 = IntegrityCheck[];
export type CreatedAt12 = string;
export type RecordHash2 = string | null;
export type Id14 = string;
export type SchemaVersion14 = number;
export type HypothesisId2 = string;
export type HypothesisTitle = string;
export type HypothesisStatement = string;
export type EvalProtocolId2 = string;
export type DatasetId4 = string;
export type Metric1 = string;
export type PredictedDirection1 = "decrease" | "increase";
export type MinEffect1 = number;
export type AblationId2 = string;
export type DisplayLabel2 = string;
export type ParamCount3 = number;
export type RunIds = string[];
export type InvalidRunIds = string[];
export type Seeds = number[];
export type MetricMean = number | null;
export type CiLow1 = number | null;
export type CiHigh1 = number | null;
export type CpuSeconds1 = number;
export type FrozenControls = string[];
export type InterventionDiff = ConfigDiff[];
export type Evidence = string[];
export type Metric2 = string;
export type Difference = number;
export type CiLow2 = number;
export type CiHigh2 = number;
export type N3 = number;
export type Method = string;
export type DecisionRule = string;
export type Integrity2 = IntegrityCheck[];
export type PromotionState =
  "NOT_ELIGIBLE" | "READY_FOR_GATE" | "GATING" | "PROMOTED" | "REJECTED" | "HOLD" | "INVALID";
export type Name18 = string;
export type Status1 = "pass" | "hold" | "fail" | "invalid" | "missing";
export type Detail1 = string;
export type PromotionGates = PromotionGate[];
export type PromotionConsequence = string;
export type Interpretation = string;
export type NonClaims1 = string[];
export type NextExperiment = string;
export type Supersedes = string | null;
export type CreatedAt13 = string;
export type RecordHash3 = string | null;
export type Id15 = string;
export type SchemaVersion15 = number;
export type Tool = string;
export type CatalogDir = string;
export type CatalogGeneratedAt = string;
export type Name19 = string;
export type Path7 = string | null;
export type CatalogCommit = string | null;
export type CurrentCommit = string | null;
export type Dirty = boolean | null;
export type Remote = string | null;
export type Roots = LegacyRoot[];
export type Id16 = string;
export type DisplayName = string | null;
export type Generation4 = number | null;
export type Status2 = string | null;
export type Architecture = string | null;
export type SearchProfile = string | null;
export type MainNet = string | null;
export type MainNetModelId = string | null;
export type Engines = LegacyEngine[];
export type Name20 = string;
export type Status3 = string | null;
export type Args = string[];
export type Notes1 = string | null;
export type SearchProfiles = SearchProfile1[];
export type Id17 = string;
export type Kind2 = "toggle" | "value";
export type Field = string;
export type OnFlag = string | null;
export type OffFlag = string | null;
export type ValueFlag = string | null;
export type Default = boolean | number | string | null;
export type Doc = string | null;
export type Record = string;
export type EvidenceId = string | null;
export type Date1 = string | null;
export type Games1 = number | null;
export type Llr1 = number | null;
export type Boundary1 = string | null;
export type Decision1 = string | null;
export type LegacyState = string;
export type History = SwitchGateRecord[];
export type EvidenceState3 =
  "PROPOSED" | "RUNNING" | "SUPPORTED" | "REJECTED" | "INCONCLUSIVE" | "SUPERSEDED" | "INVALID";
export type BacklogTier = number;
export type BacklogReason = string;
export type Switches = SearchSwitch[];
export type EngineArgs = {
  [k: string]: string;
}[];
export type UnmatchedGateRecords = number;
export type EngineId = string;
export type Role1 = string;
export type SearchProfile2 = string | null;
export type LiveBotFlags = string[];
export type MainNetModelId1 = string | null;
export type HelperNetModelId = string | null;
export type SourcePath = string;
export type Sha2565 = string;
export type OpenFollowups = string[];
export type FlagshipModelIds = string[];
export type Findings = string[];
export type Integrity3 = IntegrityCheck[];
export type CreatedAt14 = string;
export type RecordHash4 = string | null;
export type Key1 = string;
export type Label = string;
export type Kind3 = "int" | "float" | "bool" | "enum";
export type Default1 = boolean | number | string;
export type Choices = string[];
export type Min = number | null;
export type Max = number | null;
export type Axis1 = "capacity" | "representation" | "compute" | "signal" | "training";
export type Description3 = string;
export type BaselineId1 = string;
export type BaselineLabel = string;
export type Diff1 = ConfigDiff[];
export type ParamCount4 = number;
export type ControlParamCount = number;
export type DisplayLabel3 = string;
export type IdentityHash2 = string;
export type Warnings = string[];
export type DuplicateOf = string | null;
export type CanCreate = boolean;
export type Family5 = string;
export type Metric3 = string;
export type Switches1 = Switch[];
export type AblationId3 = string;
export type DisplayLabel4 = string;
export type IsBaseline1 = boolean;
export type ControlId1 = string | null;
export type Generation5 = string;
export type BaselineName1 = string;
export type DatasetId5 = string;
export type ParamCount5 = number;
export type Key2 = string;
export type Value2 = boolean | number | string | null;
export type Changed = boolean;
export type Cells = MatrixCell[];
export type MetricMean1 = number | null;
export type FindingIds = string[];
export type Rows5 = MatrixRow[];
export type Metric4 = string;
export type LowerIsBetter1 = boolean;
export type AvailableMetrics = MetricDef[];
export type RunId1 = string;
export type AblationId4 = string;
export type DisplayLabel5 = string;
export type Family6 = string;
export type Generation6 = string;
export type Representation = string;
export type IsBaseline2 = boolean;
export type Seed2 = number;
export type ParamCount6 = number;
export type SearchNodes1 = number;
export type CpuSeconds2 = number;
export type Value3 = number | null;
export type CiLow3 = number | null;
export type CiHigh3 = number | null;
export type Kind4 = ("measured" | "derived" | "estimate") | null;
export type Points = ScalingPoint[];
export type ModelId1 = string;
export type Name21 = string;
export type Role2 = string;
export type ModelKind1 = string;
export type SerializedParamCount1 = number | null;
export type ProvenanceComplete1 = boolean;
export type LegacyModels = LegacyModelPoint[];
export type IntakeId4 = string | null;
export type Switches2 = SearchSwitch[];
export type SearchProfiles1 = SearchProfile1[];
export type EngineArgs1 = {
  [k: string]: string;
}[];
export type ActiveGeneration = string;
export type EngineRepo1 = string | null;
export type EngineCommit1 = string | null;
export type LabCommit1 = string | null;
export type Baselines = Ablation[];
export type ActiveRuns = Run[];
export type RecentRuns = Run[];
export type RecentFindings = Finding[];
export type OpenHypotheses = Hypothesis[];
export type Datasets = Dataset[];
export type CpuSeconds3 = number;
export type GpuSeconds1 = number;
export type WallSeconds2 = number;
export type SearchNodes2 = number;
export type OracleLabels1 = number;
export type TrainExamplesSeen1 = number;
export type LegacyEngineSeconds = number;
export type IntakeId5 = string | null;
export type CatalogGeneratedAt1 = string | null;
export type ProductionLineage = string;
export type Roots1 = LegacyRoot[];
export type Engines1 = LegacyEngine[];
export type FlagshipModelIds1 = string[];
export type Models = number;
export type ModelsProvenanceIncomplete = number;
export type Sources = number;
export type SourceRows = number;
export type SourceBytes = number;
export type Evaluations = number;
export type Evidence1 = number;
export type Findings1 = string[];
export type FunnelRuns = FunnelRun[];
export type IntegrityAlerts = string[];
export type NormalizationId1 = string;
export type RecordCount1 = number;
export type UniqueCount = number;
export type DuplicateCount1 = number;
export type LabelCount = number;
export type RecordId = string;
export type Group = string;
export type SourceId = string;
export type SourceRow = number;
export type Ply = number;
export type Fen = string;
export type Epd = string;
export type Stm = string;
export type Phase = string;
export type Material = string;
export type Result = boolean | number | string | null;
export type Labels = {
  [k: string]: unknown;
}[];
export type Page = CatalogRecord[];
export type PageOffset = number;
export type PageSize = number;
export type TotalMatching = number;
export type NormalizationId2 = string;
export type Name22 = string;
export type Policy = string;
export type Available = number;
export type Effective = number;
export type Arms1 = ArmPreview[];
export type EffectiveTotal = number;
export type UniqueRecords = number;
export type FromNormalizationId = string;
export type ToNormalizationId = string;
export type RecordsFrom = number;
export type RecordsTo = number;
export type Added = number;
export type Removed = number;
export type Changed1 = number;
export type DuplicateDelta = number;
export type RejectedDelta = number;
export type DatasetId6 = string;
export type SurvivingRecords = number;
export type SplitChanges1 = number;
export type SplitChanges = DatasetSplitChange[];
export type AffectedDatasetIds = string[];
export type AffectedRunIds = string[];
export type Notes2 = string[];
export type Projection = string;
export type ProjectionsAvailable = string[];
export type Id18 = string;
export type Kind5 = string;
export type Label1 = string;
export type State = string | null;
export type Link = string;
export type Nodes1 = MapNode[];
export type Src = string;
export type Dst = string;
export type Kind6 = string;
export type Edges = MapEdge[];
export type Title1 = string;
export type Statement1 = string;
export type Metric5 = string;
export type PredictedDirection2 = "decrease" | "increase";
export type MinEffect2 = number;
export type Family7 = string;
export type Generation7 = string | null;
export type Tags1 = string[];
export type BaselineId2 = string;
export type HypothesisId3 = string | null;
export type Notes3 = string;
export type AblationId5 = string;
/**
 * @minItems 1
 */
export type Seeds1 = [number, ...number[]];
export type HypothesisId4 = string;
export type ControlAblationId = string;
export type InterventionAblationId = string;
export type Interpretation1 = string | null;
export type NextExperiment1 = string | null;
export type NonClaims2 = string[] | null;
export type NormalizationId3 = string;
export type Family8 = string;
export type Producer = string;
export type Authority = string;
export type RegistryVersion1 = number;
export type Pov = string;
/**
 * @minItems 1
 */
export type Rows6 = [
  {
    [k: string]: unknown;
  },
  ...{
    [k: string]: unknown;
  }[]
];
export type NormalizationId4 = string;
export type Name23 = string;
export type Arms2 = {
  [k: string]: unknown;
}[];
export type Fractions = number[];
export type SplitSeed = number;
export type RequiredLabels = string[];
export type FromNormalizationId1 = string;
export type Name24 = string;
export type Dedup = string;
export type NewNormalizationId = string;

/**
 * CVS Lab canonical schemas, schema_version 1. Generated by `python -m cvslab schema export`; do not edit.
 */
export interface CVSLab {
  Hypothesis: Hypothesis;
  Ablation: Ablation;
  ModelArtifact: ModelArtifact;
  LegacyModel: LegacyModel;
  SourceSnapshot: SourceSnapshot;
  LegacySource: LegacySource;
  Normalization: Normalization;
  Dataset: Dataset;
  TrainingRecipe: TrainingRecipe;
  EvaluationProtocol: EvaluationProtocol;
  LegacyEvaluation: LegacyEvaluation;
  Run: Run;
  LegacyEvidence: LegacyEvidence;
  FunnelRun: FunnelRun;
  Finding: Finding;
  IntakeRecord: IntakeRecord;
  Switch: Switch;
  AblationPreview: AblationPreview;
  MatrixResponse: MatrixResponse;
  ScalingResponse: ScalingResponse;
  SearchBacklog: SearchBacklog;
  Overview: Overview;
  CatalogResponse: CatalogResponse;
  StackPreview: StackPreview;
  MigrationDiff: MigrationDiff;
  MapResponse: MapResponse;
  HypothesisCreate: HypothesisCreate;
  AblationRequest: AblationRequest;
  RunQueueRequest: RunQueueRequest;
  FindingRequest: FindingRequest;
  LabelsAppendRequest: LabelsAppendRequest;
  StackFreezeRequest: StackFreezeRequest;
  MigrateRequest: MigrateRequest;
  RebuildDatasetRequest: RebuildDatasetRequest;
}
export interface Hypothesis {
  id: Id;
  schema_version: SchemaVersion;
  title: Title;
  statement: Statement;
  family: Family;
  generation: Generation;
  metric: Metric;
  predicted_direction: PredictedDirection;
  min_effect: MinEffect;
  tags: Tags;
  state: EvidenceState;
  state_history: StateHistory;
  created_at: CreatedAt;
}
export interface StateChange {
  state: EvidenceState1;
  at: At;
  reason: Reason;
}
/**
 * A#### — a scientific intervention relative to a frozen baseline (or the baseline itself).
 */
export interface Ablation {
  id: Id1;
  schema_version: SchemaVersion1;
  family: Family1;
  generation: Generation1;
  baseline_name: BaselineName;
  is_baseline: IsBaseline;
  control_id: ControlId;
  hypothesis_id: HypothesisId;
  overrides: Overrides;
  effective_config: EffectiveConfig;
  config_hash: ConfigHash;
  identity_hash: IdentityHash;
  diff: Diff;
  param_count: ParamCount;
  parameter_shapes: ParameterShapes;
  display_label: DisplayLabel;
  dataset_id: DatasetId;
  training_recipe_id: TrainingRecipeId;
  eval_protocol_id: EvalProtocolId;
  notes: Notes;
  state: EvidenceState2;
  state_history: StateHistory1;
  created_at: CreatedAt1;
}
export interface Overrides {
  [k: string]: boolean | number | string;
}
export interface EffectiveConfig {
  [k: string]: boolean | number | string;
}
export interface ConfigDiff {
  key: Key;
  control: Control;
  value: Value;
  axis: Axis;
}
export interface ParameterShapes {
  [k: string]: number[];
}
/**
 * M#### — a serialized model produced by exactly one lab run.
 */
export interface ModelArtifact {
  id: Id2;
  schema_version: SchemaVersion2;
  origin: Origin;
  lineage: Lineage;
  run_id: RunId;
  ablation_id: AblationId;
  family: Family2;
  arch: Arch;
  format: Format;
  path: Path;
  artifact_hash: ArtifactHash;
  param_count: ParamCount1;
  parameter_shapes: ParameterShapes1;
  dataset_id: DatasetId1;
  training_recipe_id: TrainingRecipeId1;
  warm_start_model_id: WarmStartModelId;
  promotable: Promotable;
  engine_loadable: EngineLoadable;
  created_at: CreatedAt2;
}
export interface ParameterShapes1 {
  [k: string]: number[];
}
/**
 * M#### imported from the legacy catalog: inspectable, never promotable, never a warm start by default.
 */
export interface LegacyModel {
  id: Id3;
  schema_version: SchemaVersion3;
  origin: Origin1;
  lineage: Lineage1;
  intake_id: IntakeId;
  name: Name;
  sha256: Sha256;
  bytes: Bytes;
  model_kind: ModelKind;
  serialized_param_count: SerializedParamCount;
  metadata: Metadata;
  role: Role;
  role_evidence: RoleEvidence;
  provenance_complete: ProvenanceComplete;
  promotable: Promotable1;
  locations: Locations;
  verification: Verification;
  created_at: CreatedAt3;
}
export interface Metadata {
  [k: string]: boolean | number | string | null;
}
export interface ArtifactLocation {
  root: Root;
  path: Path1;
}
/**
 * S#### — immutable L0 source snapshot.
 */
export interface SourceSnapshot {
  id: Id4;
  schema_version: SchemaVersion4;
  origin: Origin2;
  name: Name1;
  locality: Locality;
  source_origin: SourceOrigin;
  generator: Generator;
  license: License;
  importer: Importer;
  importer_version: ImporterVersion;
  path: Path2;
  content_hash: ContentHash;
  row_count: RowCount;
  game_count: GameCount;
  label_authorities: LabelAuthorities;
  compute: Compute;
  created_at: CreatedAt4;
}
/**
 * Executed compute, recorded as a research variable. All values are measured.
 */
export interface Compute {
  cpu_seconds: CpuSeconds;
  wall_seconds: WallSeconds;
  gpu_seconds: GpuSeconds;
  search_nodes: SearchNodes;
  train_examples_seen: TrainExamplesSeen;
  accepted_examples: AcceptedExamples;
  discarded_examples: DiscardedExamples;
  labels_generated: LabelsGenerated;
  oracle_labels: OracleLabels;
}
export interface LabelsGenerated {
  [k: string]: number;
}
export interface LegacySource {
  id: Id5;
  schema_version: SchemaVersion5;
  origin: Origin3;
  lineage: Lineage2;
  intake_id: IntakeId1;
  name: Name2;
  content_sha256: ContentSha256;
  files: Files;
  rows: Rows;
  bytes: Bytes1;
  sample_fields: SampleFields;
  field_authority: FieldAuthority;
  note: Note;
  bytes_availability: BytesAvailability;
  file_manifest: FileManifest;
  verification: Verification1;
  verification_detail: VerificationDetail;
  created_at: CreatedAt5;
}
export interface FieldAuthority {
  [k: string]: string;
}
export interface LegacyFile {
  root: Root1;
  path: Path3;
  sha256: Sha2561;
  bytes: Bytes2;
  rows: Rows1;
}
/**
 * N#### — versioned standardization of sources into canonical records.
 */
export interface Normalization {
  id: Id6;
  schema_version: SchemaVersion6;
  name: Name3;
  source_ids: SourceIds;
  source_hashes: SourceHashes;
  normalizer: Normalizer;
  normalizer_version: NormalizerVersion;
  canonical_schema_version: CanonicalSchemaVersion;
  settings: Settings;
  dedup_policy: DedupPolicy;
  recipe_hash: RecipeHash;
  path: Path4;
  output_hash: OutputHash;
  record_count: RecordCount;
  duplicate_count: DuplicateCount;
  rejected_count: RejectedCount;
  compute: Compute;
  created_at: CreatedAt6;
}
export interface SourceHashes {
  [k: string]: string;
}
export interface Settings {
  [k: string]: boolean | number | string;
}
/**
 * D#### — frozen dataset manifest: a reproducible selection over canonical records.
 */
export interface Dataset {
  id: Id7;
  schema_version: SchemaVersion7;
  name: Name4;
  normalization_id: NormalizationId;
  normalization_recipe_hash: NormalizationRecipeHash;
  source_ids: SourceIds1;
  canonical_schema_version: CanonicalSchemaVersion1;
  label_requirements: LabelRequirements;
  selection: Selection;
  sampling_policy: SamplingPolicy;
  dedup_policy: DedupPolicy1;
  split_policy: SplitPolicy;
  splits: Splits;
  stack: Stack;
  label_sets: LabelSets;
  counts: Counts;
  coverage: Coverage;
  label_provenance: LabelProvenance;
  parent_id: ParentId;
  frozen: Frozen;
  manifest_hash: ManifestHash;
  compute: Compute;
  created_at: CreatedAt7;
}
export interface Selection {
  [k: string]: boolean | number | string;
}
export interface SplitPolicy {
  [k: string]: boolean | number | string;
}
export interface SplitManifest {
  name: Name5;
  count: Count;
  group_count: GroupCount;
  path: Path5;
  file_hash: FileHash;
  record_ids_hash: RecordIdsHash;
}
/**
 * One component of a dataset stack: a filtered subpopulation plus an explicit sampling policy.
 */
export interface StackArm {
  name: Name6;
  filter: Filter;
  policy: StackPolicy;
  rows: Rows2;
  fraction: Fraction;
  balance_bucket: BalanceBucket;
  balance_cap: BalanceCap;
  seed: Seed;
}
export interface Filter {
  [k: string]: boolean | number | string;
}
/**
 * One append-only L2 label file pinned into a dataset manifest.
 */
export interface LabelSetRef {
  path: Path6;
  file_hash: FileHash1;
  rows: Rows3;
  families: Families;
  authorities: Authorities;
  producers: Producers;
  registry_version: RegistryVersion;
  label_schema_version: LabelSchemaVersion;
}
export interface Families {
  [k: string]: number;
}
export interface Authorities {
  [k: string]: number;
}
export interface Producers {
  [k: string]: number;
}
export interface Counts {
  [k: string]: number;
}
export interface Coverage {
  [k: string]: {
    [k: string]: number;
  };
}
export interface LabelProvenance {
  [k: string]: number;
}
/**
 * T#### — frozen trainer identity and training/signal/compute settings.
 */
export interface TrainingRecipe {
  id: Id8;
  schema_version: SchemaVersion8;
  name: Name7;
  family: Family3;
  generation: Generation2;
  trainer: Trainer;
  trainer_version: TrainerVersion;
  params: Params;
  description: Description;
  recipe_hash: RecipeHash1;
  created_at: CreatedAt8;
}
export interface Params {
  [k: string]: boolean | number | string;
}
/**
 * E#### — frozen evaluation instrument pinned to one dataset manifest.
 */
export interface EvaluationProtocol {
  id: Id9;
  schema_version: SchemaVersion9;
  origin: Origin4;
  name: Name8;
  family: Family4;
  generation: Generation3;
  dataset_id: DatasetId2;
  dataset_manifest_hash: DatasetManifestHash;
  split: Split;
  metrics: Metrics;
  params: Params1;
  bootstrap_samples: BootstrapSamples;
  bootstrap_seed: BootstrapSeed;
  protocol_version: ProtocolVersion;
  non_claims: NonClaims;
  description: Description2;
  protocol_hash: ProtocolHash;
  created_at: CreatedAt9;
}
export interface MetricDef {
  name: Name9;
  lower_is_better: LowerIsBetter;
  description: Description1;
}
export interface Params1 {
  [k: string]: boolean | number | string;
}
export interface LegacyEvaluation {
  id: Id10;
  schema_version: SchemaVersion10;
  origin: Origin5;
  lineage: Lineage3;
  intake_id: IntakeId2;
  name: Name10;
  kind: Kind;
  sha256: Sha2562;
  bytes: Bytes3;
  location: ArtifactLocation;
  verification: Verification2;
  created_at: CreatedAt10;
}
/**
 * R#### — one execution of an ablation. Immutable once COMPLETED or INVALID.
 */
export interface Run {
  id: Id11;
  schema_version: SchemaVersion11;
  origin: Origin6;
  ablation_id: AblationId1;
  hypothesis_id: HypothesisId1;
  display_label: DisplayLabel1;
  seed: Seed1;
  status: RunStatus;
  effective_config: EffectiveConfig1;
  config_hash: ConfigHash1;
  identity_hash: IdentityHash1;
  param_count: ParamCount2;
  dataset_id: DatasetId3;
  dataset_manifest_hash: DatasetManifestHash1;
  training_recipe_id: TrainingRecipeId2;
  recipe_hash: RecipeHash2;
  eval_protocol_id: EvalProtocolId1;
  protocol_hash: ProtocolHash1;
  queued_at: QueuedAt;
  started_at: StartedAt;
  finished_at: FinishedAt;
  environment: RunEnvironment | null;
  compute: Compute1;
  training_curve: TrainingCurve;
  metrics: Metrics1;
  integrity: Integrity;
  model_id: ModelId;
  artifact_hashes: ArtifactHashes;
  log_path: LogPath;
  error: Error;
  record_hash: RecordHash;
}
export interface EffectiveConfig1 {
  [k: string]: boolean | number | string;
}
export interface RunEnvironment {
  python: Python;
  numpy: Numpy;
  platform: Platform;
  machine: Machine;
  processor: Processor;
  cpu_count: CpuCount;
  lab_commit: LabCommit;
  lab_dirty: LabDirty;
  engine_repo: EngineRepo;
  engine_commit: EngineCommit;
}
/**
 * Executed compute, recorded as a research variable. All values are measured.
 */
export interface Compute1 {
  cpu_seconds: CpuSeconds;
  wall_seconds: WallSeconds;
  gpu_seconds: GpuSeconds;
  search_nodes: SearchNodes;
  train_examples_seen: TrainExamplesSeen;
  accepted_examples: AcceptedExamples;
  discarded_examples: DiscardedExamples;
  labels_generated: LabelsGenerated;
  oracle_labels: OracleLabels;
}
export interface EpochLog {
  epoch: Epoch;
  train_loss: TrainLoss;
  val_loss: ValLoss;
  examples_per_second: ExamplesPerSecond;
  elapsed_seconds: ElapsedSeconds;
}
export interface MetricValue {
  name: Name11;
  value: Value1;
  ci_low: CiLow;
  ci_high: CiHigh;
  n: N;
  kind: Kind1;
  split: Split1;
}
export interface IntegrityCheck {
  name: Name12;
  status: Status;
  detail: Detail;
}
export interface ArtifactHashes {
  [k: string]: string;
}
/**
 * R#### imported gate or measurement record from the legacy engine.
 */
export interface LegacyEvidence {
  id: Id12;
  schema_version: SchemaVersion12;
  origin: Origin7;
  lineage: Lineage4;
  intake_id: IntakeId3;
  type: Type;
  location: ArtifactLocation;
  sha256: Sha2563;
  date: Date;
  baseline_id: BaselineId;
  candidate_id: CandidateId;
  games: Games;
  wdl: Wdl;
  llr: Llr;
  boundary: Boundary;
  decision: Decision;
  recorded_state: RecordedState;
  evidence_state: EvidenceState1 | null;
  is_measurement: IsMeasurement;
  note: Note1;
  switch_ids: SwitchIds;
  verification: Verification3;
  created_at: CreatedAt11;
  record_hash: RecordHash1;
}
/**
 * R#### imported information-gain labeling funnel run (engine `training/funnel`).
 */
export interface FunnelRun {
  id: Id13;
  schema_version: SchemaVersion13;
  origin: Origin8;
  lineage: Lineage5;
  lineage_note: LineageNote;
  name: Name13;
  run_dir: RunDir;
  funnel_schema_version: FunnelSchemaVersion;
  funnel_config_version: FunnelConfigVersion;
  prioritizer_version: PrioritizerVersion;
  config_sha256: ConfigSha256;
  source_created_at: SourceCreatedAt;
  tooling_commit: ToolingCommit;
  tooling_dirty: ToolingDirty;
  engine_binary_sha256: EngineBinarySha256;
  engine_artifacts: EngineArtifacts;
  engine_artifact_model_ids: EngineArtifactModelIds;
  engine_options: EngineOptions;
  stockfish_sha256: StockfishSha256;
  taxonomy_version: TaxonomyVersion;
  taxonomy_sha256: TaxonomySha256;
  facts_registry_version: FactsRegistryVersion;
  provenance_classes: ProvenanceClasses;
  sample: Sample;
  tiers: Tiers;
  total_engine_seconds: TotalEngineSeconds;
  provenance_records: ProvenanceRecords;
  coverage: Coverage1;
  underrepresented: Underrepresented;
  selection: Selection1;
  priority_histogram: PriorityHistogram;
  priority_reasons: PriorityReasons;
  arms: Arms;
  oracle_arms: OracleArms;
  audit_miss: AuditMiss;
  experiment_question: ExperimentQuestion;
  informative_yield_ratio: InformativeYieldRatio;
  experiment_caveat: ExperimentCaveat;
  research_state: EvidenceState1 | null;
  research_state_source: ResearchStateSource;
  compute: Compute;
  compute_note: ComputeNote;
  files: Files1;
  integrity: Integrity1;
  created_at: CreatedAt12;
  record_hash: RecordHash2;
}
export interface EngineArtifacts {
  [k: string]: string;
}
export interface EngineArtifactModelIds {
  [k: string]: string;
}
export interface EngineOptions {
  [k: string]: boolean | number | string;
}
export interface ProvenanceClasses {
  [k: string]: string;
}
export interface Sample {
  [k: string]: boolean | number | string;
}
export interface FunnelTier {
  name: Name14;
  positions: Positions;
  engine_seconds: EngineSeconds;
  ms_per_position: MsPerPosition;
  nodes: Nodes;
  wall_seconds: WallSeconds1;
  provenance_classes: ProvenanceClasses1;
}
export interface ProvenanceRecords {
  [k: string]: number;
}
export interface Coverage1 {
  [k: string]: {
    [k: string]: number;
  };
}
export interface Selection1 {
  [k: string]: number;
}
export interface PriorityReasons {
  [k: string]: number;
}
export interface FunnelArm {
  name: Name15;
  n: N1;
  deep_nodes: DeepNodes;
  deep_engine_seconds: DeepEngineSeconds;
  informative: Informative;
  informative_rate: InformativeRate;
  move_change_rate: MoveChangeRate;
  abs_shallow_deep_delta_cp: AbsShallowDeepDeltaCp;
}
export interface AbsShallowDeepDeltaCp {
  [k: string]: number;
}
export interface FunnelOracleArm {
  name: Name16;
  n: N2;
  disagreement_rate: DisagreementRate;
  move_agreement_rate: MoveAgreementRate;
  abs_cp: AbsCp;
}
export interface AbsCp {
  [k: string]: number;
}
export interface AuditMiss {
  [k: string]: number;
}
export interface FunnelFile {
  name: Name17;
  sha256: Sha2564;
  bytes: Bytes4;
  rows: Rows4;
  copied_to: CopiedTo;
}
/**
 * F#### — the standard finding card. Immutable; a revision supersedes, never edits.
 */
export interface Finding {
  id: Id14;
  schema_version: SchemaVersion14;
  hypothesis_id: HypothesisId2;
  hypothesis_title: HypothesisTitle;
  hypothesis_statement: HypothesisStatement;
  eval_protocol_id: EvalProtocolId2;
  dataset_id: DatasetId4;
  metric: Metric1;
  predicted_direction: PredictedDirection1;
  min_effect: MinEffect1;
  control: ArmSummary;
  intervention: ArmSummary;
  frozen_controls: FrozenControls;
  intervention_diff: InterventionDiff;
  evidence: Evidence;
  effect: Effect | null;
  result: EvidenceState1;
  decision_rule: DecisionRule;
  integrity: Integrity2;
  promotion_state: PromotionState;
  promotion_gates: PromotionGates;
  promotion_consequence: PromotionConsequence;
  interpretation: Interpretation;
  non_claims: NonClaims1;
  next_experiment: NextExperiment;
  supersedes: Supersedes;
  created_at: CreatedAt13;
  record_hash: RecordHash3;
}
export interface ArmSummary {
  ablation_id: AblationId2;
  display_label: DisplayLabel2;
  param_count: ParamCount3;
  run_ids: RunIds;
  invalid_run_ids: InvalidRunIds;
  seeds: Seeds;
  metric_mean: MetricMean;
  ci_low: CiLow1;
  ci_high: CiHigh1;
  cpu_seconds: CpuSeconds1;
}
export interface Effect {
  metric: Metric2;
  difference: Difference;
  ci_low: CiLow2;
  ci_high: CiHigh2;
  n: N3;
  method: Method;
}
export interface PromotionGate {
  name: Name18;
  status: Status1;
  detail: Detail1;
}
/**
 * I#### — one import of the legacy catalog: what was verified, assigned and reused.
 */
export interface IntakeRecord {
  id: Id15;
  schema_version: SchemaVersion15;
  tool: Tool;
  catalog_dir: CatalogDir;
  catalog_generated_at: CatalogGeneratedAt;
  catalog_file_hashes: CatalogFileHashes;
  roots: Roots;
  assigned: Assigned;
  reused: Reused;
  engines: Engines;
  search_profiles: SearchProfiles;
  switches: Switches;
  engine_args: EngineArgs;
  unmatched_gate_records: UnmatchedGateRecords;
  champion: LegacyChampion | null;
  flagship_model_ids: FlagshipModelIds;
  findings: Findings;
  integrity: Integrity3;
  compute: Compute;
  created_at: CreatedAt14;
  record_hash: RecordHash4;
}
export interface CatalogFileHashes {
  [k: string]: string;
}
export interface LegacyRoot {
  name: Name19;
  path: Path7;
  catalog_commit: CatalogCommit;
  current_commit: CurrentCommit;
  dirty: Dirty;
  remote: Remote;
}
export interface Assigned {
  [k: string]: string[];
}
export interface Reused {
  [k: string]: number;
}
export interface LegacyEngine {
  id: Id16;
  display_name: DisplayName;
  generation: Generation4;
  status: Status2;
  architecture: Architecture;
  search_profile: SearchProfile;
  main_net: MainNet;
  main_net_model_id: MainNetModelId;
}
export interface SearchProfile1 {
  name: Name20;
  status: Status3;
  args: Args;
  effective_options: EffectiveOptions;
  notes: Notes1;
}
export interface EffectiveOptions {
  [k: string]: boolean | number | string;
}
/**
 * An engine search switch as a clean-lineage backlog item: legacy evidence sets priority, not truth.
 */
export interface SearchSwitch {
  id: Id17;
  kind: Kind2;
  field: Field;
  on_flag: OnFlag;
  off_flag: OffFlag;
  value_flag: ValueFlag;
  default: Default;
  doc: Doc;
  legacy_state: EvidenceState1;
  current: SwitchGateRecord | null;
  history: History;
  clean_state: EvidenceState3;
  backlog_tier: BacklogTier;
  backlog_reason: BacklogReason;
}
export interface SwitchGateRecord {
  record: Record;
  evidence_id: EvidenceId;
  date: Date1;
  games: Games1;
  llr: Llr1;
  boundary: Boundary1;
  decision: Decision1;
  legacy_state: LegacyState;
}
export interface LegacyChampion {
  engine_id: EngineId;
  role: Role1;
  search_profile: SearchProfile2;
  live_bot_flags: LiveBotFlags;
  main_net_model_id: MainNetModelId1;
  helper_net_model_id: HelperNetModelId;
  source_path: SourcePath;
  sha256: Sha2565;
  open_followups: OpenFollowups;
}
/**
 * A registered experimental variable. Only registered switches may appear in a configuration.
 */
export interface Switch {
  key: Key1;
  label: Label;
  kind: Kind3;
  default: Default1;
  choices: Choices;
  min: Min;
  max: Max;
  axis: Axis1;
  description: Description3;
}
export interface AblationPreview {
  baseline_id: BaselineId1;
  baseline_label: BaselineLabel;
  overrides: Overrides1;
  effective_config: EffectiveConfig2;
  diff: Diff1;
  param_count: ParamCount4;
  control_param_count: ControlParamCount;
  parameter_shapes: ParameterShapes2;
  display_label: DisplayLabel3;
  identity_hash: IdentityHash2;
  warnings: Warnings;
  duplicate_of: DuplicateOf;
  can_create: CanCreate;
}
export interface Overrides1 {
  [k: string]: boolean | number | string;
}
export interface EffectiveConfig2 {
  [k: string]: boolean | number | string;
}
export interface ParameterShapes2 {
  [k: string]: number[];
}
export interface MatrixResponse {
  family: Family5;
  metric: Metric3;
  switches: Switches1;
  rows: Rows5;
}
export interface MatrixRow {
  ablation_id: AblationId3;
  display_label: DisplayLabel4;
  is_baseline: IsBaseline1;
  control_id: ControlId1;
  generation: Generation5;
  baseline_name: BaselineName1;
  dataset_id: DatasetId5;
  param_count: ParamCount5;
  state: EvidenceState1;
  cells: Cells;
  run_counts: RunCounts;
  metric_mean: MetricMean1;
  finding_ids: FindingIds;
}
export interface MatrixCell {
  key: Key2;
  value: Value2;
  changed: Changed;
}
export interface RunCounts {
  [k: string]: number;
}
export interface ScalingResponse {
  metric: Metric4;
  lower_is_better: LowerIsBetter1;
  available_metrics: AvailableMetrics;
  points: Points;
  legacy_models: LegacyModels;
}
export interface ScalingPoint {
  run_id: RunId1;
  ablation_id: AblationId4;
  display_label: DisplayLabel5;
  family: Family6;
  generation: Generation6;
  representation: Representation;
  is_baseline: IsBaseline2;
  status: RunStatus;
  seed: Seed2;
  param_count: ParamCount6;
  search_nodes: SearchNodes1;
  cpu_seconds: CpuSeconds2;
  value: Value3;
  ci_low: CiLow3;
  ci_high: CiHigh3;
  kind: Kind4;
}
export interface LegacyModelPoint {
  model_id: ModelId1;
  name: Name21;
  role: Role2;
  model_kind: ModelKind1;
  serialized_param_count: SerializedParamCount1;
  provenance_complete: ProvenanceComplete1;
}
export interface SearchBacklog {
  intake_id: IntakeId4;
  switches: Switches2;
  search_profiles: SearchProfiles1;
  engine_args: EngineArgs1;
  legacy_state_counts: LegacyStateCounts;
}
export interface LegacyStateCounts {
  [k: string]: number;
}
export interface Overview {
  active_generation: ActiveGeneration;
  engine_repo: EngineRepo1;
  engine_commit: EngineCommit1;
  lab_commit: LabCommit1;
  baselines: Baselines;
  run_counts: RunCounts1;
  active_runs: ActiveRuns;
  recent_runs: RecentRuns;
  recent_findings: RecentFindings;
  open_hypotheses: OpenHypotheses;
  datasets: Datasets;
  compute: ComputeTotals;
  object_counts: ObjectCounts;
  legacy: LegacySummary;
  funnel_runs: FunnelRuns;
  integrity_alerts: IntegrityAlerts;
}
export interface RunCounts1 {
  [k: string]: number;
}
export interface ComputeTotals {
  cpu_seconds: CpuSeconds3;
  gpu_seconds: GpuSeconds1;
  wall_seconds: WallSeconds2;
  search_nodes: SearchNodes2;
  oracle_labels: OracleLabels1;
  train_examples_seen: TrainExamplesSeen1;
  labels_generated: LabelsGenerated1;
  legacy_engine_seconds: LegacyEngineSeconds;
}
export interface LabelsGenerated1 {
  [k: string]: number;
}
export interface ObjectCounts {
  [k: string]: number;
}
export interface LegacySummary {
  intake_id: IntakeId5;
  catalog_generated_at: CatalogGeneratedAt1;
  production_lineage: ProductionLineage;
  champion: LegacyChampion | null;
  roots: Roots1;
  engines: Engines1;
  flagship_model_ids: FlagshipModelIds1;
  models: Models;
  models_provenance_incomplete: ModelsProvenanceIncomplete;
  model_roles: ModelRoles;
  sources: Sources;
  source_rows: SourceRows;
  source_bytes: SourceBytes;
  evaluations: Evaluations;
  evidence: Evidence1;
  evidence_states: EvidenceStates;
  switch_legacy_states: SwitchLegacyStates;
  findings: Findings1;
}
export interface ModelRoles {
  [k: string]: number;
}
export interface EvidenceStates {
  [k: string]: number;
}
export interface SwitchLegacyStates {
  [k: string]: number;
}
export interface CatalogResponse {
  normalization_id: NormalizationId1;
  record_count: RecordCount1;
  unique_count: UniqueCount;
  duplicate_count: DuplicateCount1;
  label_count: LabelCount;
  filters_applied: FiltersApplied;
  facets: Facets;
  page: Page;
  page_offset: PageOffset;
  page_size: PageSize;
  total_matching: TotalMatching;
}
export interface FiltersApplied {
  [k: string]: boolean | number | string;
}
export interface Facets {
  [k: string]: {
    [k: string]: number;
  };
}
/**
 * One canonical record rendered in the corpus catalog, labels joined and memberships resolved.
 */
export interface CatalogRecord {
  record_id: RecordId;
  group: Group;
  source_id: SourceId;
  source_row: SourceRow;
  ply: Ply;
  fen: Fen;
  epd: Epd;
  stm: Stm;
  phase: Phase;
  material: Material;
  result: Result;
  labels: Labels;
  label_tiers: LabelTiers;
  memberships: Memberships;
}
export interface LabelTiers {
  [k: string]: number;
}
export interface Memberships {
  [k: string]: string[];
}
export interface StackPreview {
  normalization_id: NormalizationId2;
  arms: Arms1;
  effective_total: EffectiveTotal;
  unique_records: UniqueRecords;
  distributions: Distributions;
}
export interface ArmPreview {
  name: Name22;
  policy: Policy;
  filter: Filter1;
  available: Available;
  effective: Effective;
}
export interface Filter1 {
  [k: string]: boolean | number | string;
}
export interface Distributions {
  [k: string]: {
    [k: string]: number;
  };
}
export interface MigrationDiff {
  from_normalization_id: FromNormalizationId;
  to_normalization_id: ToNormalizationId;
  records_from: RecordsFrom;
  records_to: RecordsTo;
  added: Added;
  removed: Removed;
  changed: Changed1;
  duplicate_delta: DuplicateDelta;
  rejected_delta: RejectedDelta;
  label_changes: LabelChanges;
  coverage_from: CoverageFrom;
  coverage_to: CoverageTo;
  source_contribution_from: SourceContributionFrom;
  source_contribution_to: SourceContributionTo;
  split_changes: SplitChanges;
  affected_dataset_ids: AffectedDatasetIds;
  affected_run_ids: AffectedRunIds;
  notes: Notes2;
}
export interface LabelChanges {
  [k: string]: {
    [k: string]: number;
  };
}
export interface CoverageFrom {
  [k: string]: {
    [k: string]: number;
  };
}
export interface CoverageTo {
  [k: string]: {
    [k: string]: number;
  };
}
export interface SourceContributionFrom {
  [k: string]: number;
}
export interface SourceContributionTo {
  [k: string]: number;
}
export interface DatasetSplitChange {
  dataset_id: DatasetId6;
  surviving_records: SurvivingRecords;
  split_changes: SplitChanges1;
}
export interface MapResponse {
  projection: Projection;
  projections_available: ProjectionsAvailable;
  nodes: Nodes1;
  edges: Edges;
  filters_applied: FiltersApplied1;
}
/**
 * One spatial object on the Lab Map; `detail` is display evidence, `link` the 2D page.
 */
export interface MapNode {
  id: Id18;
  kind: Kind5;
  label: Label1;
  state: State;
  link: Link;
  detail: Detail2;
}
export interface Detail2 {
  [k: string]: unknown;
}
export interface MapEdge {
  src: Src;
  dst: Dst;
  kind: Kind6;
}
export interface FiltersApplied1 {
  [k: string]: unknown;
}
export interface HypothesisCreate {
  title: Title1;
  statement: Statement1;
  metric?: Metric5;
  predicted_direction?: PredictedDirection2;
  min_effect?: MinEffect2;
  family?: Family7;
  generation?: Generation7;
  tags?: Tags1;
}
export interface AblationRequest {
  baseline_id: BaselineId2;
  overrides?: Overrides2;
  hypothesis_id?: HypothesisId3;
  notes?: Notes3;
}
export interface Overrides2 {
  [k: string]: boolean | number | string;
}
export interface RunQueueRequest {
  ablation_id: AblationId5;
  seeds?: Seeds1;
}
export interface FindingRequest {
  hypothesis_id: HypothesisId4;
  control_ablation_id: ControlAblationId;
  intervention_ablation_id: InterventionAblationId;
  interpretation?: Interpretation1;
  next_experiment?: NextExperiment1;
  non_claims?: NonClaims2;
}
/**
 * Append a new label set to a canonical corpus; existing records and labels are never modified.
 */
export interface LabelsAppendRequest {
  normalization_id: NormalizationId3;
  family: Family8;
  producer: Producer;
  authority: Authority;
  registry_version?: RegistryVersion1;
  pov?: Pov;
  rows: Rows6;
}
export interface StackFreezeRequest {
  normalization_id: NormalizationId4;
  name: Name23;
  arms?: Arms2;
  fractions?: Fractions;
  split_seed?: SplitSeed;
  required_labels?: RequiredLabels;
}
export interface MigrateRequest {
  from_normalization_id: FromNormalizationId1;
  name: Name24;
  settings_overrides?: SettingsOverrides;
  dedup?: Dedup;
}
export interface SettingsOverrides {
  [k: string]: boolean | number | string;
}
export interface RebuildDatasetRequest {
  new_normalization_id: NewNormalizationId;
}
