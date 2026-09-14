import { useState } from "react";
import { Link } from "react-router-dom";
import { api, usePolling } from "../api";
import { Badge, Empty, ErrorBox, Loading, fmt, fmtInt, formatParams, timeAgo } from "../ui";

const STATUSES = ["", "QUEUED", "RUNNING", "COMPLETED", "INVALID"];

export default function Runs() {
  const [status, setStatus] = useState("");
  const { data: runs, error, loading, refresh } = usePolling(() => api.runs(status || undefined), 3000);

  return (
    <>
      <h1>Run queue &amp; history</h1>
      <p className="subtitle">
        One R#### per execution of an ablation. Completed and INVALID records are sealed — a rerun creates a new R####,
        it never edits history.
      </p>
      <div className="filters">
        <label>
          Status
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            {STATUSES.map((value) => (
              <option key={value} value={value}>
                {value || "ALL"}
              </option>
            ))}
          </select>
        </label>
        <button onClick={refresh}>Refresh</button>
      </div>
      <ErrorBox error={error} />
      {loading && !runs ? (
        <Loading />
      ) : !runs || runs.length === 0 ? (
        <Empty>No runs. Queue one from the ablation builder, or run `cvslab demo`.</Empty>
      ) : (
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Ablation</th>
                <th className="num">Seed</th>
                <th className="num">Params</th>
                <th>Status</th>
                <th className="num">test loss</th>
                <th className="num">CPU s</th>
                <th className="num">Integrity</th>
                <th>Finished</th>
              </tr>
            </thead>
            <tbody>
              {runs
                .slice()
                .sort((a, b) => a.id.localeCompare(b.id))
                .map((run) => {
                  const loss = run.metrics.find((metric) => metric.name === "test_loss");
                  const failed = run.integrity.filter((check) => check.status === "fail").length;
                  return (
                    <tr key={run.id}>
                      <td className="mono">
                        <Link to={`/runs/${run.id}`}>{run.id}</Link>
                      </td>
                      <td>{run.display_label}</td>
                      <td className="num mono">{run.seed}</td>
                      <td className="num mono" title={String(run.param_count)}>{formatParams(run.param_count)}</td>
                      <td>
                        <Badge>{run.status}</Badge>
                      </td>
                      <td className="num mono">{loss ? fmt(loss.value) : "—"}</td>
                      <td className="num mono">{fmt(run.compute.cpu_seconds, 1)}</td>
                      <td className="num mono" style={{ color: failed ? "var(--bad)" : undefined }}>
                        {failed ? `${failed} fail` : run.integrity.length === 0 ? "—" : "ok"}
                      </td>
                      <td>{timeAgo(run.finished_at ?? run.queued_at)}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}
      {runs && runs.length > 0 && (
        <p className="subtitle" style={{ marginTop: 8 }}>
          {fmtInt(runs.length)} run records shown.
        </p>
      )}
    </>
  );
}
