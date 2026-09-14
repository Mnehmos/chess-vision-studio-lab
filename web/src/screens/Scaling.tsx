import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { ScalingResponse } from "../generated/schemas";
import { Empty, ErrorBox, Loading, fmt, fmtInt, formatParams } from "../ui";

type XAxis = "param_count" | "cpu_seconds" | "search_nodes";

const X_AXIS_LABEL: Record<XAxis, string> = {
  param_count: "learned parameters",
  cpu_seconds: "executed CPU seconds",
  search_nodes: "executed search nodes",
};

export default function Scaling() {
  const [metric, setMetric] = useState("test_loss");
  const [xAxis, setXAxis] = useState<XAxis>("param_count");
  const [data, setData] = useState<ScalingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .scaling(metric)
      .then((response) => {
        setData(response);
        setError(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, [metric]);

  const plotted = (data?.points ?? []).filter((point) => point.value !== null);
  const legacy = (data?.legacy_models ?? []).filter((model) => model.serialized_param_count !== null);
  const width = 920;
  const height = 460;
  const pad = { left: 84, right: 20, top: 18, bottom: 58 };

  const values = plotted.map((point) => point.value as number);
  const ciLow = plotted.map((point) => point.ci_low ?? (point.value as number));
  const ciHigh = plotted.map((point) => point.ci_high ?? (point.value as number));
  const xs = plotted.map((point) => point[xAxis]);
  const legacyXs = xAxis === "param_count" ? legacy.map((model) => model.serialized_param_count as number) : [];
  const allX = [...xs, ...legacyXs];
  const hasX = allX.length > 0 && allX.some((value) => value > 0);

  const logX = xAxis === "param_count";
  const xMinRaw = allX.length ? Math.min(...allX) : 0;
  const xMaxRaw = allX.length ? Math.max(...allX) : 1;
  const xMin = logX ? Math.log10(Math.max(xMinRaw, 1)) : 0;
  const xMax = logX ? Math.log10(Math.max(xMaxRaw, 2)) : xMaxRaw * 1.05 || 1;
  const xSpan = xMax - xMin || 1;
  const px = (value: number) => pad.left + ((logX ? Math.log10(Math.max(value, 1)) : value) - xMin) / xSpan * (width - pad.left - pad.right);

  const yMin = values.length ? Math.min(...ciLow) : 0;
  const yMax = values.length ? Math.max(...ciHigh) : 1;
  const ySpan = yMax - yMin || Math.abs(yMax) || 1;
  const py = (value: number) => pad.top + (1 - (value - yMin) / ySpan) * (height - pad.top - pad.bottom);

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => xMin + fraction * xSpan);

  return (
    <>
      <h1>Scaling explorer</h1>
      <p className="subtitle">
        Capability (held-out metric) against learned parameters and executed compute. Every point is one immutable run;
        whiskers are 95% bootstrap intervals. Legacy nets appear as reference ticks on the parameter axis — they carry
        no clean-lineage measurement.
      </p>
      <div className="filters">
        <label>
          Metric
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>
            {(data?.available_metrics ?? []).map((definition) => (
              <option key={definition.name} value={definition.name}>
                {definition.name} {definition.lower_is_better ? "↓" : "↑"}
              </option>
            ))}
          </select>
        </label>
        <label>
          X axis
          <select value={xAxis} onChange={(event) => setXAxis(event.target.value as XAxis)}>
            <option value="param_count">learned parameters (log)</option>
            <option value="cpu_seconds">CPU seconds</option>
            <option value="search_nodes">search nodes</option>
          </select>
        </label>
      </div>
      <ErrorBox error={error} />
      {!data ? (
        <Loading />
      ) : plotted.length === 0 ? (
        <Empty>No completed runs measure {metric} yet.</Empty>
      ) : (
        <>
          <div className="scaling-wrap">
            <svg viewBox={`0 0 ${width} ${height}`} width="100%">
              <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} stroke="var(--border)" />
              <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="var(--border)" />
              {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
                const value = yMax - fraction * ySpan;
                return (
                  <g key={fraction}>
                    <line x1={pad.left} x2={width - pad.right} y1={py(value)} y2={py(value)} stroke="var(--border)" strokeDasharray="2 4" />
                    <text x={pad.left - 6} y={py(value) + 3} textAnchor="end">{value.toPrecision(3)}</text>
                  </g>
                );
              })}
              {xTicks.map((tick, index) => (
                <text key={index} x={px(logX ? 10 ** tick : tick)} y={height - pad.bottom + 16} textAnchor="middle">
                  {logX ? `1e${Math.round(tick)}` : fmtInt(Math.round(tick))}
                </text>
              ))}
              <text x={(width + pad.left) / 2} y={height - 10} textAnchor="middle">{X_AXIS_LABEL[xAxis]}</text>
              <text x={14} y={(height - pad.bottom + pad.top) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(height - pad.bottom + pad.top) / 2})`}>
                {metric} {data.lower_is_better ? "(lower is better)" : "(higher is better)"}
              </text>

              {plotted.map((point) => {
                const cx = px(point[xAxis]);
                const lowY = py(point.ci_low ?? point.value as number);
                const highY = py(point.ci_high ?? point.value as number);
                const cy = py(point.value as number);
                const color = point.status === "INVALID" ? "var(--invalid)" : point.is_baseline ? "var(--warn)" : "var(--accent)";
                return (
                  <g key={point.run_id}>
                    <title>{`${point.run_id} ${point.display_label}\n${metric}: ${fmt(point.value)} [${fmt(point.ci_low)}, ${fmt(point.ci_high)}]\nparams ${fmtInt(point.param_count)} · seed ${point.seed} · ${point.status}`}</title>
                    {point.ci_low !== null && point.ci_high !== null && (
                      <line x1={cx} x2={cx} y1={highY} y2={lowY} stroke={color} strokeWidth={1} opacity={0.7} />
                    )}
                    {point.is_baseline ? (
                      <rect x={cx - 4} y={cy - 4} width={8} height={8} fill={color} />
                    ) : (
                      <circle cx={cx} cy={cy} r={4} fill={color} />
                    )}
                  </g>
                );
              })}

              {legacyXs.map((value) => {
                const model = legacy.find((candidate) => candidate.serialized_param_count === value)!;
                return (
                  <g key={model.model_id}>
                    <title>{`${model.model_id} ${model.name}\nrole ${model.role}\n${fmtInt(value)} parameters (LEGACY — no clean measurement)`}</title>
                    <line x1={px(value)} x2={px(value)} y1={height - pad.bottom} y2={height - pad.bottom + 10} stroke="var(--muted)" />
                    <circle cx={px(value)} cy={height - pad.bottom + 10} r={2.5} fill="var(--muted)" />
                  </g>
                );
              })}
            </svg>
            <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>
              <span style={{ color: "var(--warn)" }}>■ baseline</span> · <span style={{ color: "var(--accent)" }}>● intervention</span>
              {legacyXs.length > 0 && <> · <span>● legacy nets (x-axis ticks only)</span></>}
              {!hasX && " · some runs recorded zero compute"}
            </div>
          </div>

          <div className="panel" style={{ padding: 0, marginTop: 16 }}>
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Ablation</th>
                  <th className="num">Params</th>
                  <th className="num">{metric}</th>
                  <th className="num">95% CI</th>
                  <th className="num">CPU s</th>
                  <th className="num">Seed</th>
                </tr>
              </thead>
              <tbody>
                {plotted
                  .slice()
                  .sort((a, b) => a.param_count - b.param_count)
                  .map((point) => (
                    <tr key={point.run_id}>
                      <td className="mono">
                        <Link to={`/runs/${point.run_id}`}>{point.run_id}</Link>
                      </td>
                      <td>{point.display_label}</td>
                      <td className="num mono" title={String(point.param_count)}>{formatParams(point.param_count)}</td>
                      <td className="num mono">{fmt(point.value)}</td>
                      <td className="num mono">[{fmt(point.ci_low)}, {fmt(point.ci_high)}]</td>
                      <td className="num mono">{fmt(point.cpu_seconds, 1)}</td>
                      <td className="num mono">{point.seed}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {legacy.length > 0 && (
            <div className="panel" style={{ padding: 0, marginTop: 16 }}>
              <table>
                <thead>
                  <tr>
                    <th>Legacy net</th>
                    <th>Role</th>
                    <th className="num">Serialized params</th>
                    <th>Provenance</th>
                  </tr>
                </thead>
                <tbody>
                  {legacy
                    .slice()
                    .sort((a, b) => (b.serialized_param_count ?? 0) - (a.serialized_param_count ?? 0))
                    .map((model) => (
                      <tr key={model.model_id}>
                        <td>
                          <span className="mono">{model.model_id}</span> {model.name}
                        </td>
                        <td className="mono">{model.role}</td>
                        <td className="num mono">{fmtInt(model.serialized_param_count)}</td>
                        <td>{model.provenance_complete ? "complete" : "incomplete — non-promotable"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </>
  );
}
