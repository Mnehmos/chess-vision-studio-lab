import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Ablation,
  AblationPreview,
  Dataset,
  Finding,
  FindingRequest,
  Hypothesis,
  HypothesisCreate,
  MatrixResponse,
  Overview,
  Run,
  ScalingResponse,
  SearchBacklog,
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
