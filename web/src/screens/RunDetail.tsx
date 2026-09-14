import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import type { Run as RunType } from "../generated/schemas";
import { Badge, ErrorBox, Loading, Section, fmt, fmtInt, formatParams, formatValue, shortHash, timeAgo } from "../ui";

function Curve({ run }: { run: RunType }) {
  const points = run.training_curve;
  if (points.length === 0) return <p className="empty">No epochs recorded.</p>;
  const width = 860;
  const height = 220;
  const pad = { left: 70, right: 16, top: 12, bottom: 26 };
  const values = points.flatMap((point) => [point.train_loss, point.val_loss]).filter((value): value is number => value !== null);
  if (values.length === 0) return <p className="empty">Epoch logs hold no finite losses.</p>;
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const spanY = maxY - minY || 1e-9;
  const x = (index: number) => pad.left + (points.length === 1 ? 0 : (index / (points.length - 1)) * (width - pad.left - pad.right));
  const y = (value: number | null) => (value === null ? null : pad.top + (1 - (value - minY) / spanY) * (height - pad.top - pad.bottom));
  const line = (accessor: (point: RunType["training_curve"][number]) => number | null) =>
    points
      .map((point, index) => {
        const py = y(accessor(point));
        return py === null ? null : `${x(index).toFixed(1)},${py.toFixed(1)}`;
      })
      .filter((chunk): chunk is string => chunk !== null)
      .join(" ");

  return (
    <div className="scaling-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%">
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} stroke="var(--border)" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="var(--border)" />
        {[0, 0.5, 1].map((fraction) => {
          const value = maxY - fraction * spanY;
          return (
            <g key={fraction}>
              <line x1={pad.left} x2={width - pad.right} y1={y(value)!} y2={y(value)!} stroke="var(--border)" strokeDasharray="2 4" />
              <text x={pad.left - 6} y={(y(value) ?? 0) + 3} textAnchor="end">{value.toPrecision(3)}</text>
            </g>
          );
        })}
        {points.map((point, index) => (
          <text key={point.epoch} x={x(index)} y={height - 8} textAnchor="middle">{point.epoch}</text>
        ))}
        <polyline points={line((point) => point.val_loss)} fill="none" stroke="var(--warn)" strokeWidth={1.5} />
        <polyline points={line((point) => point.train_loss)} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      </svg>
      <div style={{ color: "var(--muted)", fontSize: 12 }}>
        <span style={{ color: "var(--accent)" }}>━ train loss</span> · <span style={{ color: "var(--warn)" }}>━ val loss</span> · epoch on x axis
      </div>
    </div>
  );
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunType | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    if (!runId) return;
    api
      .run(runId)
      .then((value) => {
        setRun(value);
        setError(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, [runId]);

  useEffect(() => {
    load();
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, [load]);

  useEffect(() => {
    if (!run || log !== null) return;
    api.runLog(run.id).then(setLog).catch(() => setLog("log unavailable until the run finishes"));
  }, [run, log]);

  async function execute() {
    if (!run) return;
    setBusy(true);
    try {
      setRun(await api.executeRun(run.id));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  if (!run) return error ? <ErrorBox error={error} /> : <Loading />;
  const failure = run.integrity.find((check) => check.status === "fail");

  return (
    <>
      <h1>
        {run.id} <Badge>{run.status}</Badge>
      </h1>
      <p className="subtitle">
        {run.display_label} · seed {run.seed} · {formatParams(run.param_count)} parameters
        {run.finished_at ? ` · finished ${timeAgo(run.finished_at)}` : ""}
      </p>
      {run.error && <div className="alert">{run.error}</div>}
      {failure && (
        <div className="alert">
          Integrity failure <span className="mono">{failure.name}</span>: {failure.detail}. INVALID runs stay on record
          and never silently become weak evidence.
        </div>
      )}
      {run.status === "QUEUED" && (
        <p>
          <button className="primary" disabled={busy} onClick={execute}>
            {busy ? "Executing…" : "Execute now"}
          </button>{" "}
          (the API worker also picks up queued runs automatically)
        </p>
      )}
      <ErrorBox error={error} />

      <Section title="Metrics (frozen evaluation split)">
        <div className="panel" style={{ padding: 0 }}>
          {run.metrics.length === 0 ? (
            <p className="empty">No metrics — the run did not reach evaluation.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="num">Value</th>
                  <th className="num">95% CI</th>
                  <th className="num">n</th>
                  <th>Kind</th>
                  <th>Split</th>
                </tr>
              </thead>
              <tbody>
                {run.metrics.map((metric) => (
                  <tr key={metric.name}>
                    <td className="mono">{metric.name}</td>
                    <td className="num mono">{fmt(metric.value)}</td>
                    <td className="num mono">
                      [{fmt(metric.ci_low)}, {fmt(metric.ci_high)}]
                    </td>
                    <td className="num mono">{fmtInt(metric.n)}</td>
                    <td>{metric.kind}</td>
                    <td className="mono">{metric.split}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Section>

      <Section title="Training curve">
        <Curve run={run} />
      </Section>

      <div className="grid cols-2">
        <Section title="Effective configuration (frozen at queue time)">
          <div className="panel" style={{ padding: 0 }}>
            <table>
              <tbody>
                {Object.entries(run.effective_config).map(([key, value]) => (
                  <tr key={key}>
                    <td className="mono">{key}</td>
                    <td className="num mono">{formatValue(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Pinned evidence">
          <div className="panel">
            <dl className="kv">
              <dt>Ablation</dt>
              <dd className="mono">{run.ablation_id}</dd>
              <dt>Dataset</dt>
              <dd className="mono">{run.dataset_id} · {shortHash(run.dataset_manifest_hash)}</dd>
              <dt>Training recipe</dt>
              <dd className="mono">{run.training_recipe_id} · {shortHash(run.recipe_hash)}</dd>
              <dt>Eval protocol</dt>
              <dd className="mono">{run.eval_protocol_id} · {shortHash(run.protocol_hash)}</dd>
              <dt>Config hash</dt>
              <dd className="mono">{shortHash(run.config_hash)}</dd>
              <dt>Model artifact</dt>
              <dd className="mono">{run.model_id ?? "—"}</dd>
              <dt>Record seal</dt>
              <dd className="mono">{shortHash(run.record_hash)}</dd>
            </dl>
          </div>
        </Section>
      </div>

      <Section title={`Integrity checks (${run.integrity.filter((c) => c.status === "pass").length}/${run.integrity.length} pass)`}>
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <tbody>
              {run.integrity.map((check) => (
                <tr key={check.name}>
                  <td>
                    <Badge>{check.status}</Badge>
                  </td>
                  <td className="mono">{check.name}</td>
                  <td>{check.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <div className="grid cols-2">
        <Section title="Executed compute">
          <div className="panel">
            <dl className="kv">
              <dt>CPU seconds</dt>
              <dd className="mono">{fmt(run.compute.cpu_seconds, 2)}</dd>
              <dt>Wall seconds</dt>
              <dd className="mono">{fmt(run.compute.wall_seconds, 2)}</dd>
              <dt>Train examples seen</dt>
              <dd className="mono">{fmtInt(run.compute.train_examples_seen)}</dd>
              <dt>Accepted / discarded</dt>
              <dd className="mono">{fmtInt(run.compute.accepted_examples)} / {fmtInt(run.compute.discarded_examples)}</dd>
              <dt>Search nodes</dt>
              <dd className="mono">{fmtInt(run.compute.search_nodes)} (static eval: no search executed)</dd>
            </dl>
          </div>
        </Section>
        <Section title="Environment">
          <div className="panel">
            <dl className="kv">
              <dt>Python / numpy</dt>
              <dd className="mono">{run.environment?.python ?? "—"} / {run.environment?.numpy ?? "—"}</dd>
              <dt>CPU</dt>
              <dd>{run.environment?.processor ?? "—"} × {run.environment?.cpu_count ?? "—"}</dd>
              <dt>Lab commit</dt>
              <dd className="mono">{run.environment?.lab_commit ? `${run.environment.lab_commit.slice(0, 10)}${run.environment.lab_dirty ? " (dirty)" : ""}` : "—"}</dd>
              <dt>Engine commit</dt>
              <dd className="mono">{run.environment?.engine_commit?.slice(0, 10) ?? "not pinned"}</dd>
            </dl>
          </div>
        </Section>
      </div>

      <Section
        title="Run log (immutable)"
        right={
          <a href="#/" onClick={(event) => { event.preventDefault(); api.runLog(run.id).then(setLog); }}>
            reload
          </a>
        }
      >
        <pre className="log">{log ?? "loading…"}</pre>
      </Section>
    </>
  );
}
