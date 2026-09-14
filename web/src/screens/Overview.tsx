import { Link } from "react-router-dom";
import { api, usePolling } from "../api";
import { Badge, Empty, ErrorBox, Loading, Section, fmt, fmtInt, formatParams, shortHash, timeAgo } from "../ui";

export default function OverviewScreen() {
  const { data: o, error, loading } = usePolling(api.overview, 5000);
  if (loading && !o) return <Loading />;
  if (!o) return <ErrorBox error={error} />;

  return (
    <>
      <h1>Lab overview</h1>
      <p className="subtitle">
        Active generation {o.active_generation}
        {o.engine_repo ? ` · engine pinned at ${shortHash(o.engine_commit)}` : " · no engine repository pinned"}
      </p>

      {o.integrity_alerts.length > 0 && (
        <div>
          {o.integrity_alerts.slice(0, 8).map((alert, index) => (
            <div className="alert" key={index}>
              {alert}
            </div>
          ))}
        </div>
      )}

      <div className="grid cols-4">
        <div className="stat">
          <div className="label">Completed runs</div>
          <div className="value">{o.run_counts.COMPLETED ?? 0}</div>
          <div className="note">{o.run_counts.INVALID ?? 0} invalid · {o.run_counts.QUEUED ?? 0} queued · {o.run_counts.RUNNING ?? 0} running</div>
        </div>
        <div className="stat">
          <div className="label">CPU seconds spent</div>
          <div className="value">{fmtInt(Math.round(o.compute.cpu_seconds))}</div>
          <div className="note">{fmtInt(Math.round(o.compute.wall_seconds))} wall · {fmtInt(o.compute.legacy_engine_seconds)} legacy engine</div>
        </div>
        <div className="stat">
          <div className="label">Train examples seen</div>
          <div className="value">{fmtInt(o.compute.train_examples_seen)}</div>
          <div className="note">{fmtInt(o.compute.oracle_labels)} oracle labels</div>
        </div>
        <div className="stat">
          <div className="label">Search nodes</div>
          <div className="value">{fmtInt(o.compute.search_nodes)}</div>
          <div className="note">{o.object_counts.A ?? 0} ablations · {o.object_counts.H ?? 0} hypotheses</div>
        </div>
      </div>

      <Section title="Frozen baselines" count={o.baselines.length}>
        {o.baselines.length === 0 ? (
          <Empty>No baselines registered yet. Run `cvslab demo` to populate the lab with the Phase 0 proof.</Empty>
        ) : (
          <div className="panel" style={{ padding: 0 }}>
            <table>
              <thead>
                <tr>
                  <th>Ablation</th>
                  <th>Parameters</th>
                  <th>State</th>
                  <th>Dataset</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {o.baselines.map((baseline) => (
                  <tr key={baseline.id}>
                    <td>{baseline.display_label}</td>
                    <td className="num mono">{formatParams(baseline.param_count)}</td>
                    <td>
                      <Badge>{baseline.state}</Badge>
                    </td>
                    <td className="mono">{baseline.dataset_id}</td>
                    <td>{timeAgo(baseline.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {o.active_runs.length > 0 && (
        <Section title="Active runs" count={o.active_runs.length}>
          <div className="panel" style={{ padding: 0 }}>
            <table>
              <tbody>
                {o.active_runs.map((run) => (
                  <tr key={run.id}>
                    <td className="mono">
                      <Link to={`/runs/${run.id}`}>{run.id}</Link>
                    </td>
                    <td>{run.display_label}</td>
                    <td className="num mono">seed {run.seed}</td>
                    <td>
                      <Badge>{run.status}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <div className="grid cols-2">
        <Section title="Recent runs" count={o.recent_runs.length}>
          <div className="panel" style={{ padding: 0 }}>
            {o.recent_runs.length === 0 ? (
              <Empty>No finished runs yet.</Empty>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Status</th>
                    <th className="num">test loss</th>
                    <th className="num">CPU s</th>
                    <th>Finished</th>
                  </tr>
                </thead>
                <tbody>
                  {o.recent_runs.map((run) => {
                    const loss = run.metrics.find((metric) => metric.name === "test_loss");
                    return (
                      <tr key={run.id}>
                        <td className="mono">
                          <Link to={`/runs/${run.id}`}>{run.id}</Link>
                        </td>
                        <td>
                          <Badge>{run.status}</Badge>
                        </td>
                        <td className="num mono">{loss ? fmt(loss.value) : "—"}</td>
                        <td className="num mono">{fmt(run.compute.cpu_seconds, 1)}</td>
                        <td>{timeAgo(run.finished_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </Section>

        <Section title="Recent findings" count={o.recent_findings.length}>
          <div className="panel" style={{ padding: 0 }}>
            {o.recent_findings.length === 0 ? (
              <Empty>No findings yet.</Empty>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Result</th>
                    <th>Hypothesis</th>
                    <th className="num">Effect</th>
                  </tr>
                </thead>
                <tbody>
                  {o.recent_findings.map((finding) => (
                    <tr key={finding.id}>
                      <td>
                        <Badge>{finding.result}</Badge>
                      </td>
                      <td>
                        <Link to={`/findings/${finding.id}`}>
                          {finding.hypothesis_id} {finding.hypothesis_title}
                        </Link>
                      </td>
                      <td className="num mono">
                        {finding.effect ? fmt(finding.effect.difference) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Section>
      </div>

      <div className="grid cols-2">
        <Section title="Open hypotheses" count={o.open_hypotheses.length}>
          <div className="panel">
            {o.open_hypotheses.length === 0 ? (
              <Empty>No open hypotheses.</Empty>
            ) : (
              <ul className="plain">
                {o.open_hypotheses.map((hypothesis) => (
                  <li key={hypothesis.id}>
                    <span className="mono">{hypothesis.id}</span> {hypothesis.title}{" "}
                    <Badge>{hypothesis.state}</Badge>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Section>

        <Section title="Frozen datasets" count={o.datasets.length}>
          <div className="panel" style={{ padding: 0 }}>
            {o.datasets.length === 0 ? (
              <Empty>No datasets frozen yet.</Empty>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th className="num">Records</th>
                    <th className="num">train / val / test</th>
                    <th>Manifest</th>
                  </tr>
                </thead>
                <tbody>
                  {o.datasets.map((dataset) => (
                    <tr key={dataset.id}>
                      <td>
                        <span className="mono">{dataset.id}</span> {dataset.name}
                      </td>
                      <td className="num mono">{fmtInt(dataset.counts.records)}</td>
                      <td className="num mono">{dataset.splits.map((split) => split.count).join(" / ")}</td>
                      <td className="mono" title={dataset.manifest_hash}>
                        {shortHash(dataset.manifest_hash)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Section>
      </div>

      {(o.legacy.models > 0 || o.legacy.sources > 0) && (
        <Section title="Legacy intake (LEGACY lineage — inspectable, never inherited)">
          <div className="panel">
            <div className="kv">
              <dt>Production lineage</dt>
              <dd>{o.legacy.production_lineage}</dd>
              <dt>Models</dt>
              <dd>
                {o.legacy.models} ({o.legacy.models_provenance_incomplete} with incomplete provenance)
              </dd>
              <dt>Roles</dt>
              <dd className="mono">{Object.entries(o.legacy.model_roles).map(([role, n]) => `${role} ${n}`).join(", ") || "—"}</dd>
              <dt>Sources</dt>
              <dd>
                {o.legacy.sources} corpora · {(o.legacy.source_bytes / 1e9).toFixed(1)} GB ·{" "}
                {fmtInt(o.legacy.source_rows)} rows
              </dd>
              <dt>Evidence states</dt>
              <dd className="mono">
                {Object.entries(o.legacy.evidence_states).map(([state, n]) => `${state} ${n}`).join(", ") || "—"}
              </dd>
              {o.legacy.champion && (
                <>
                  <dt>Legacy champion</dt>
                  <dd>
                    {o.legacy.champion.engine_id} (net {o.legacy.champion.main_net_model_id ?? "?"})
                  </dd>
                </>
              )}
            </div>
            {o.legacy.findings.length > 0 && (
              <ul className="plain" style={{ marginTop: 10 }}>
                {o.legacy.findings.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
            )}
          </div>
        </Section>
      )}

      {o.funnel_runs.length > 0 && (
        <Section title="Labeling funnel runs (legacy compute)" count={o.funnel_runs.length}>
          <div className="panel" style={{ padding: 0 }}>
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th className="num">Engine s</th>
                  <th className="num">Oracle labels</th>
                  <th>Question</th>
                </tr>
              </thead>
              <tbody>
                {o.funnel_runs.map((funnel) => (
                  <tr key={funnel.id}>
                    <td>
                      <span className="mono">{funnel.id}</span> {funnel.name}
                    </td>
                    <td className="num mono">{fmtInt(Math.round(funnel.total_engine_seconds))}</td>
                    <td className="num mono">{fmtInt(funnel.compute.oracle_labels)}</td>
                    <td>{funnel.experiment_question}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </>
  );
}
