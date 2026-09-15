import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Ablation,
  AblationPreview,
  CatalogResponse,
  Dataset,
  Finding,
  FindingRequest,
  Hypothesis,
  HypothesisCreate,
  LabelSetRef,
  MapResponse,
  MatrixResponse,
  MigrationDiff,
  Normalization,
  Overview,
  Run,
  ScalingResponse,
  SearchBacklog,
  StackPreview,
  Switch,
} from "./generated/schemas";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.error === "string") detail = body.error;
    } catch {
      /* keep status line */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | undefined>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== "");
  return entries.length ? `?${new URLSearchParams(entries.map(([k, v]) => [k, String(v)]))}` : "";
}

export const api = {
  overview: () => request<Overview>("/api/overview"),

  switches: (family = "NNUE") => request<Switch[]>(`/api/switches${query({ family })}`),
  matrix: (params: Record<string, string | number | undefined>) =>
    request<MatrixResponse>(`/api/matrix${query(params)}`),
  scaling: (metric = "test_loss", family = "NNUE") =>
    request<ScalingResponse>(`/api/scaling${query({ metric, family })}`),
  backlog: () => request<SearchBacklog>("/api/backlog"),

  runs: (status?: string) => request<Run[]>(`/api/runs${query({ status })}`),
  run: (id: string) => request<Run>(`/api/runs/${id}`),
  runLog: async (id: string) => {
    const response = await fetch(`/api/runs/${id}/log`);
    if (!response.ok) throw new ApiError(response.status, response.statusText);
    return response.text();
  },
  queueRuns: (ablationId: string, seeds: number[]) =>
    request<Run[]>("/api/runs", { method: "POST", body: JSON.stringify({ ablation_id: ablationId, seeds }) }),
  executeRun: (id: string) => request<Run>(`/api/runs/${id}/execute`, { method: "POST" }),

  hypotheses: () => request<Hypothesis[]>("/api/hypotheses"),
  createHypothesis: (body: HypothesisCreate) =>
    request<Hypothesis>("/api/hypotheses", { method: "POST", body: JSON.stringify(body) }),

  baselines: () => request<Ablation[]>("/api/baselines"),
  ablations: () => request<Ablation[]>("/api/ablations"),
  previewAblation: (baselineId: string, overrides: Record<string, unknown>) =>
    request<AblationPreview>("/api/ablations/preview", {
      method: "POST",
      body: JSON.stringify({ baseline_id: baselineId, overrides }),
    }),
  createAblation: (body: { baseline_id: string; overrides: Record<string, unknown>; hypothesis_id?: string; notes?: string }) =>
    request<Ablation>("/api/ablations", { method: "POST", body: JSON.stringify(body) }),

  findings: (search?: string, result?: string) =>
    request<Finding[]>(`/api/findings${query({ query: search, result })}`),
  finding: (id: string) => request<Finding>(`/api/findings/${id}`),
  draftFinding: (body: FindingRequest) =>
    request<Finding>("/api/findings", { method: "POST", body: JSON.stringify(body) }),

  datasets: () => request<Dataset[]>("/api/datasets"),
  object: <T>(id: string) => request<T>(`/api/object/${id}`),

  normalizations: () => request<Normalization[]>("/api/objects/N"),
  catalog: (params: Record<string, string | number | undefined>) =>
    request<CatalogResponse>(`/api/catalog${query(params)}`),
  labelSets: (normalizationId: string) => request<LabelSetRef[]>(`/api/normalizations/${normalizationId}/labels`),
  appendLabels: (body: {
    normalization_id: string;
    family: string;
    producer: string;
    authority: string;
    registry_version?: number;
    pov?: string;
    rows: { record_id: string; value: unknown; budget?: unknown; note?: string }[];
  }) => request<LabelSetRef>("/api/labels", { method: "POST", body: JSON.stringify(body) }),
  stackPreview: (normalizationId: string, arms: object[]) =>
    request<StackPreview>("/api/stacks/preview", {
      method: "POST",
      body: JSON.stringify({ normalization_id: normalizationId, arms }),
    }),
  freezeStack: (body: {
    normalization_id: string;
    name: string;
    arms: object[];
    split_seed?: number;
    required_labels?: string[];
  }) => request<Dataset>("/api/datasets/freeze", { method: "POST", body: JSON.stringify(body) }),
  migrate: (body: { from_normalization_id: string; name: string; settings_overrides: Record<string, number> }) =>
    request<Normalization>("/api/normalizations", { method: "POST", body: JSON.stringify(body) }),
  migrationDiff: (fromId: string, toId: string) =>
    request<MigrationDiff>(`/api/migration-diff${query({ from_id: fromId, to_id: toId })}`),
  rebuildDataset: (datasetId: string, newNormalizationId: string) =>
    request<Dataset>(`/api/datasets/${datasetId}/rebuild`, {
      method: "POST",
      body: JSON.stringify({ new_normalization_id: newNormalizationId }),
    }),

  mapView: (projection: string, generation?: string, state?: string, family?: string) =>
    request<MapResponse>(`/api/map${query({ projection, generation, state, family })}`),
  labelFacts: (normalizationId: string) =>
    request<object>("/api/labels/facts", {
      method: "POST",
      body: JSON.stringify({ normalization_id: normalizationId }),
    }),
};

/** Fetch with periodic refresh; returns data, error and a manual refresh trigger. */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(() => {
    fetcherRef.current()
      .then((value) => {
        setData(value);
        setError(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, intervalMs);
    return () => clearInterval(timer);
  }, [refresh, intervalMs]);

  return { data, error, loading, refresh };
}
