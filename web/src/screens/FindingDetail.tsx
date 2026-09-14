import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { Finding } from "../generated/schemas";
import { Badge, ErrorBox, Loading, Section, fmt, formatParams, formatValue } from "../ui";

export default function FindingDetail() {
  const { findingId } = useParams<{ findingId: string }>();
  const [finding, setFinding] = useState<Finding | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!findingId) return;
    api
      .finding(findingId)
      .then((value) => {
        setFinding(value);
        setError(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, [findingId]);

  if (!finding) return error ? <ErrorBox error={error} /> : <Loading />;

  return (
    <>
      <h1>
        {finding.id} <Badge>{finding.result}</Badge> <Badge>{finding.promotion_state}</Badge>
      </h1>
      <p className="subtitle">
        Finding card · {finding.hypothesis_id} · created {finding.created_at}
        {finding.supersedes ? ` · supersedes ${finding.supersedes}` : ""}
      </p>

      <Section title="Hypothesis">
        <div className="panel">
          <p style={{ marginTop: 0 }}>
            <strong>{finding.hypothesis_title}</strong>
          </p>
          <p>{finding.hypothesis_statement}</p>
          <dl className="kv">
            <dt>Metric</dt>
            <dd className="mono">{finding.metric}</dd>
            <dt>Predicted direction</dt>
            <dd>{finding.predicted_direction}</dd>
            <dt>Minimum effect</dt>
            <dd className="mono">{finding.min_effect}</dd>
            <dt>Dataset / protocol</dt>
            <dd className="mono">
              {finding.dataset_id} / {finding.eval_protocol_id}
            </dd>
          </dl>
        </div>
      </Section>

      <div className="grid cols-2">
        <Section title="Control arm">
          <div className="panel">
            <dl className="kv">
              <dt>Ablation</dt>
              <dd>{finding.control.display_label}</dd>
              <dt>Parameters</dt>
              <dd className="mono">{finding.control.param_count.toLocaleString("en-US")}</dd>
              <dt>Runs</dt>
              <dd className="mono">{finding.control.run_ids.map((id) => (
                <Link key={id} to={`/runs/${id}`} style={{ marginRight: 8 }}>{id}</Link>
              ))}</dd>
              {finding.control.invalid_run_ids.length > 0 && (
                <>
                  <dt>Invalid runs</dt>
                  <dd className="mono">{finding.control.invalid_run_ids.join(", ")}</dd>
                </>
              )}
              <dt>Seeds</dt>
              <dd className="mono">{finding.control.seeds.join(", ")}</dd>
              <dt>Metric mean</dt>
              <dd className="mono">
                {finding.control.metric_mean === null ? "—" : `${fmt(finding.control.metric_mean)} [${fmt(finding.control.ci_low)}, ${fmt(finding.control.ci_high)}]`}
              </dd>
              <dt>CPU seconds</dt>
              <dd className="mono">{fmt(finding.control.cpu_seconds, 1)}</dd>
            </dl>
          </div>
        </Section>
        <Section title="Intervention arm">
          <div className="panel">
            <dl className="kv">
              <dt>Ablation</dt>
              <dd>{finding.intervention.display_label}</dd>
              <dt>Parameters</dt>
              <dd className="mono">{finding.intervention.param_count.toLocaleString("en-US")} ({formatParams(finding.intervention.param_count)})</dd>
              <dt>Runs</dt>
              <dd className="mono">{finding.intervention.run_ids.map((id) => (
                <Link key={id} to={`/runs/${id}`} style={{ marginRight: 8 }}>{id}</Link>
              ))}</dd>
              {finding.intervention.invalid_run_ids.length > 0 && (
                <>
                  <dt>Invalid runs</dt>
                  <dd className="mono">{finding.intervention.invalid_run_ids.join(", ")}</dd>
                </>
              )}
              <dt>Seeds</dt>
              <dd className="mono">{finding.intervention.seeds.join(", ")}</dd>
              <dt>Metric mean</dt>
              <dd className="mono">
                {finding.intervention.metric_mean === null
                  ? "—"
                  : `${fmt(finding.intervention.metric_mean)} [${fmt(finding.intervention.ci_low)}, ${fmt(finding.intervention.ci_high)}]`}
              </dd>
              <dt>CPU seconds</dt>
              <dd className="mono">{fmt(finding.intervention.cpu_seconds, 1)}</dd>
            </dl>
          </div>
        </Section>
      </div>

      <Section title="Intervention — exactly what changed">
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Field</th>
                <th>Control</th>
                <th>Intervention</th>
                <th>Axis</th>
              </tr>
            </thead>
            <tbody>
              {finding.intervention_diff.map((diff) => (
                <tr key={diff.key}>
                  <td className="mono">{diff.key}</td>
                  <td className="mono">{formatValue(diff.control)}</td>
                  <td className="mono" style={{ color: "var(--accent)" }}>{formatValue(diff.value)}</td>
                  <td>
                    <span className={`axis ${diff.axis}`}>{diff.axis}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Frozen controls">
        <div className="panel">
          <ul className="plain">
            {finding.frozen_controls.map((control, index) => (
              <li key={index}>{control}</li>
            ))}
          </ul>
        </div>
      </Section>

      <Section title="Evidence">
        <div className="panel">
          <ul className="plain">
            {finding.evidence.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      </Section>

      <Section title="Result">
        <div className="panel">
          {finding.effect ? (
            <div className="kv" style={{ gridTemplateColumns: "max-content auto" }}>
              <dt>Effect (intervention − control)</dt>
              <dd className="mono">
                {fmt(finding.effect.difference)} · 95% CI [{fmt(finding.effect.ci_low)}, {fmt(finding.effect.ci_high)}] · n{" "}
                {finding.effect.n.toLocaleString("en-US")}
              </dd>
              <dt>Method</dt>
              <dd>{finding.effect.method}</dd>
            </div>
          ) : (
            <p className="empty">No effect estimate — integrity checks failed.</p>
          )}
          <p style={{ marginBottom: 0 }}>
            <strong>Decision rule (pre-declared):</strong> {finding.decision_rule}
          </p>
        </div>
      </Section>

      <Section title={`Integrity (${finding.integrity.filter((c) => c.status === "pass").length}/${finding.integrity.length} pass)`}>
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <tbody>
              {finding.integrity.map((check) => (
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

      <Section title="Promotion">
        <div className="panel">
          <dl className="kv">
            <dt>State</dt>
            <dd>
              <Badge>{finding.promotion_state}</Badge>
            </dd>
          </dl>
          <table style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>Gate</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {finding.promotion_gates.map((gate) => (
                <tr key={gate.name}>
                  <td className="mono">{gate.name}</td>
                  <td>
                    <Badge>{gate.status}</Badge>
                  </td>
                  <td>{gate.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginBottom: 0 }}>{finding.promotion_consequence}</p>
        </div>
      </Section>

      <Section title="Interpretation">
        <div className="panel">{finding.interpretation}</div>
      </Section>

      <Section title="Non-claims">
        <div className="panel">
          <ul className="plain">
            {finding.non_claims.map((claim, index) => (
              <li key={index}>{claim}</li>
            ))}
          </ul>
        </div>
      </Section>

      <Section title="Next experiment">
        <div className="panel">{finding.next_experiment}</div>
      </Section>
    </>
  );
}
