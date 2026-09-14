import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { MatrixResponse, MetricDef } from "../generated/schemas";
import { Badge, Empty, ErrorBox, Loading, fmt, formatParams, formatValue } from "../ui";

export default function Matrix() {
  const [family] = useState("NNUE");
  const [generation, setGeneration] = useState("");
  const [baseline, setBaseline] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [state, setState] = useState("");
  const [metric, setMetric] = useState("test_loss");
  const [minParams, setMinParams] = useState("");
  const [maxParams, setMaxParams] = useState("");
  const [data, setData] = useState<MatrixResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricDef[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.scaling(metric).then((response) => setMetrics(response.available_metrics)).catch(() => setMetrics([]));
  }, [metric]);

  useEffect(() => {
    setLoading(true);
    api
      .matrix({
        family,
        generation: generation || undefined,
        baseline: baseline || undefined,
        dataset_id: datasetId || undefined,
        state: state || undefined,
        metric,
        min_params: minParams || undefined,
        max_params: maxParams || undefined,
      })
      .then((response) => {
        setData(response);
        setError(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)))
      .finally(() => setLoading(false));
  }, [family, generation, baseline, datasetId, state, metric, minParams, maxParams]);

  const distinct = useMemo(() => {
    const rows = data?.rows ?? [];
    return {
      baselines: [...new Set(rows.map((row) => row.baseline_name))].sort(),
      datasets: [...new Set(rows.map((row) => row.dataset_id))].sort(),
      generations: [...new Set(rows.map((row) => row.generation))].sort(),
    };
  }, [data]);

  return (
    <>
      <h1>Switch matrix</h1>
      <p className="subtitle">
        Rows are ablations; columns are registered switches. Highlighted cells deviate from that row's control.
        Filters narrow by generation, baseline, dataset, parameter budget and result state.
      </p>

      <div className="filters">
        <label>
          Generation
          <select value={generation} onChange={(event) => setGeneration(event.target.value)}>
            <option value="">all</option>
            {distinct.generations.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Baseline
          <select value={baseline} onChange={(event) => setBaseline(event.target.value)}>
            <option value="">all</option>
            {distinct.baselines.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Dataset
          <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
            <option value="">all</option>
            {distinct.datasets.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          State
          <select value={state} onChange={(event) => setState(event.target.value)}>
            {["", "PROPOSED", "RUNNING", "SUPPORTED", "REJECTED", "INCONCLUSIVE", "SUPERSEDED", "INVALID"].map((value) => (
              <option key={value} value={value}>
                {value || "all"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Metric
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>
            {metrics.map((definition) => (
              <option key={definition.name} value={definition.name}>
                {definition.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Min params
          <input value={minParams} onChange={(event) => setMinParams(event.target.value)} placeholder="1" style={{ width: 90 }} />
        </label>
        <label>
          Max params
          <input value={maxParams} onChange={(event) => setMaxParams(event.target.value)} placeholder="∞" style={{ width: 90 }} />
        </label>
      </div>

      <ErrorBox error={error} />
      {loading && !data ? (
        <Loading />
      ) : !data || data.rows.length === 0 ? (
        <Empty>No ablations match these filters.</Empty>
      ) : (
        <div className="panel matrix" style={{ padding: 0, overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Ablation</th>
                <th>State</th>
                <th className="num">Params</th>
                <th className="num">Runs</th>
                <th className="num">{data.metric}</th>
                {data.switches.map((sw) => (
                  <th key={sw.key} title={`${sw.label} (${sw.axis})`}>
                    {sw.key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.ablation_id}>
                  <td title={row.display_label}>
                    <span className="mono">{row.ablation_id}</span>{" "}
                    {row.is_baseline ? <Badge>{row.baseline_name}</Badge> : null}
                    {row.finding_ids.map((findingId) => (
                      <Link key={findingId} to={`/findings/${findingId}`} className="mono" style={{ marginLeft: 6 }}>
                        {findingId}
                      </Link>
                    ))}
                  </td>
                  <td>
                    <Badge>{row.state}</Badge>
                  </td>
                  <td className="num mono" title={String(row.param_count)}>
                    {formatParams(row.param_count)}
                  </td>
                  <td className="num mono">
                    {Object.entries(row.run_counts)
                      .map(([status, count]) => `${count}${status[0]}`)
                      .join(" ") || "—"}
                  </td>
                  <td className="num mono">{row.metric_mean === null ? "—" : fmt(row.metric_mean)}</td>
                  {row.cells.map((cell) => (
                    <td key={cell.key} className={cell.changed ? "changed-cell" : "same-cell"}>
                      {formatValue(cell.value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.rows.length > 0 && (
        <p className="subtitle" style={{ marginTop: 8 }}>
          Run counts: C completed, I invalid, Q queued, R running.
        </p>
      )}
    </>
  );
}
