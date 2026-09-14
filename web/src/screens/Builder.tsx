import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { Ablation, AblationPreview, Hypothesis, Switch } from "../generated/schemas";
import { Badge, ErrorBox, Loading, Section, formatParams, formatValue } from "../ui";

export default function Builder() {
  const [baselines, setBaselines] = useState<Ablation[]>([]);
  const [switches, setSwitches] = useState<Switch[]>([]);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [baselineId, setBaselineId] = useState<string>("");
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [preview, setPreview] = useState<AblationPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [hypothesisId, setHypothesisId] = useState("");
  const [notes, setNotes] = useState("");
  const [seeds, setSeeds] = useState("0,1,2");
  const [created, setCreated] = useState<Ablation | null>(null);
  const [queuedRunIds, setQueuedRunIds] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const debounce = useRef(0);

  useEffect(() => {
    Promise.all([api.baselines(), api.switches(), api.hypotheses()]).then(([base, registry, hyps]) => {
      setBaselines(base);
      setSwitches(registry);
      setHypotheses(hyps);
      if (base.length > 0 && !baselineId) setBaselineId(base[0].id);
    }).catch((exc: unknown) => setActionError(exc instanceof Error ? exc.message : String(exc)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const baseline = baselines.find((candidate) => candidate.id === baselineId) ?? null;

  const overrides = useMemo(() => {
    const out: Record<string, string> = {};
    if (!baseline) return out;
    for (const [key, value] of Object.entries(draft)) {
      if (String(baseline.effective_config[key]) !== value) out[key] = value;
    }
    return out;
  }, [draft, baseline]);

  const runPreview = useCallback(() => {
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      if (!baselineId) return;
      api
        .previewAblation(baselineId, overrides)
        .then((result) => {
          setPreview(result);
          setPreviewError(null);
        })
        .catch((exc: unknown) => {
          setPreview(null);
          setPreviewError(exc instanceof Error ? exc.message : String(exc));
        });
    }, 300);
  }, [baselineId, overrides]);

  useEffect(() => {
    runPreview();
  }, [runPreview]);

  function setSwitch(key: string, value: string) {
    setDraft((previous) => ({ ...previous, [key]: value }));
    setCreated(null);
    setQueuedRunIds([]);
  }

  function resetSwitch(key: string) {
    setDraft((previous) => {
      const next = { ...previous };
      delete next[key];
      return next;
    });
  }

  async function createAblation() {
    if (!preview || !preview.can_create) return;
    setBusy(true);
    setActionError(null);
    try {
      const ablation = await api.createAblation({
        baseline_id: baselineId,
        overrides,
        hypothesis_id: hypothesisId || undefined,
        notes,
      });
      setCreated(ablation);
      setQueuedRunIds([]);
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function queueRuns() {
    if (!created) return;
    setBusy(true);
    setActionError(null);
    try {
      const parsedSeeds = seeds.split(",").map((part) => Number.parseInt(part.trim(), 10)).filter(Number.isInteger);
      if (parsedSeeds.length === 0) throw new Error("at least one integer seed is required");
      const runs = await api.queueRuns(created.id, parsedSeeds);
      setQueuedRunIds(runs.map((run) => run.id));
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  if (baselines.length === 0 && !actionError) return <Loading />;
  if (baselines.length === 0) return <ErrorBox error={actionError} />;

  return (
    <>
      <h1>Ablation builder</h1>
      <p className="subtitle">
        Choose a frozen baseline, flip registered switches, see the exact diff and parameter count, then queue runs.
        The backend recomputes everything — this form carries no scientific logic of its own.
      </p>
      <ErrorBox error={actionError} />

      <div className="filters">
        <label>
          Baseline
          <select value={baselineId} onChange={(event) => { setBaselineId(event.target.value); setDraft({}); setCreated(null); }}>
            {baselines.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.display_label}
              </option>
            ))}
          </select>
        </label>
        {baseline && (
          <label>
            Control parameters
            <input readOnly value={`${formatParams(baseline.param_count)} (${baseline.param_count})`} />
          </label>
        )}
      </div>

      {baseline && (
        <Section title="Registered switches" count={`${Object.keys(overrides).length} changed`}>
          <div className="switch-grid">
            {switches.map((sw) => {
              const effective = baseline.effective_config[sw.key];
              const edited = overrides[sw.key] !== undefined;
              return (
                <div key={sw.key} className={`switch-control${edited ? " changed" : ""}`}>
                  <div className="head">
                    <span className="key">{sw.key}</span>
                    <span className="axis">{sw.axis}</span>
                    {edited && (
                      <button style={{ marginLeft: "auto", padding: "0 8px" }} onClick={() => resetSwitch(sw.key)} title="reset to baseline">
                        ×
                      </button>
                    )}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    {sw.kind === "bool" ? (
                      <input
                        type="checkbox"
                        checked={edited ? overrides[sw.key] === "1" : Boolean(effective)}
                        onChange={(event) => setSwitch(sw.key, event.target.checked ? "1" : "0")}
                      />
                    ) : sw.kind === "enum" ? (
                      <select value={edited ? overrides[sw.key] : String(effective)} onChange={(event) => setSwitch(sw.key, event.target.value)}>
                        {sw.choices.map((choice) => (
                          <option key={choice} value={choice}>
                            {choice}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        step={sw.kind === "float" ? "any" : "1"}
                        value={edited ? overrides[sw.key] : String(effective)}
                        placeholder={String(sw.default)}
                        onChange={(event) => setSwitch(sw.key, event.target.value)}
                      />
                    )}
                  </div>
                  <div className="desc" title={sw.description ?? ""}>
                    {sw.label} · baseline {formatValue(effective)}
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      <Section title="Preview — computed by the backend before anything is written">
        {previewError && <ErrorBox error={previewError} />}
        {preview && (
          <>
            {preview.warnings.length > 0 && (
              <div>
                {preview.warnings.map((warning, index) => (
                  <div className="notice" key={index}>
                    {warning}
                  </div>
                ))}
              </div>
            )}
            <div className="panel finding-card">
              <div className="head">
                <span className="mono" style={{ fontSize: 14 }}>{preview.display_label}</span>
                {preview.duplicate_of && <Badge>duplicate of {preview.duplicate_of}</Badge>}
              </div>
              <div className="grid cols-4">
                <div className="stat">
                  <div className="label">Exact parameters</div>
                  <div className="value mono">{preview.param_count.toLocaleString("en-US")}</div>
                  <div className="note">control {preview.control_param_count.toLocaleString("en-US")} · Δ {preview.param_count - preview.control_param_count > 0 ? "+" : ""}{(preview.param_count - preview.control_param_count).toLocaleString("en-US")}</div>
                </div>
                <div className="stat">
                  <div className="label">Changed fields</div>
                  <div className="value">{preview.diff.length}</div>
                  <div className="note">{preview.diff.map((diff) => diff.key).join(", ") || "none"}</div>
                </div>
                <div className="stat">
                  <div className="label">Parameter shapes</div>
                  <div className="note mono" style={{ marginTop: 6 }}>
                    {Object.entries(preview.parameter_shapes).map(([name, shape]) => `${name} ${shape.join("×")}`).join("  ")}
                  </div>
                </div>
                <div className="stat">
                  <div className="label">Identity</div>
                  <div className="note mono" style={{ marginTop: 6 }}>{preview.identity_hash.slice(0, 27)}…</div>
                </div>
              </div>
              {preview.diff.length > 0 && (
                <div style={{ padding: 0 }}>
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
                      {preview.diff.map((diff) => (
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
              )}
            </div>
          </>
        )}
      </Section>

      <Section title="Create and queue">
        <div className="panel">
          <div className="filters">
            <label>
              Tests hypothesis
              <select value={hypothesisId} onChange={(event) => setHypothesisId(event.target.value)}>
                <option value="">— none —</option>
                {hypotheses.map((hypothesis) => (
                  <option key={hypothesis.id} value={hypothesis.id}>
                    {hypothesis.id} {hypothesis.title}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ flexGrow: 1 }}>
              Notes
              <input style={{ width: "100%", minWidth: 260 }} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="why this intervention" />
            </label>
            <label>
              Seeds
              <input value={seeds} onChange={(event) => setSeeds(event.target.value)} />
            </label>
            <button className="primary" disabled={!preview?.can_create || busy} onClick={createAblation}>
              Create ablation
            </button>
            {created && (
              <button disabled={busy} onClick={queueRuns}>
                Queue runs of {created.id}
              </button>
            )}
          </div>

          {created && (
            <div className="notice">
              Created <span className="mono">{created.display_label}</span>. Queue runs to execute it — the API
              server's worker picks up queued runs automatically, or run <span className="mono">cvslab worker</span>.
            </div>
          )}
          {queuedRunIds.length > 0 && (
            <div className="notice">
              Queued {queuedRunIds.map((id) => (
                <Link key={id} to={`/runs/${id}`} className="mono" style={{ marginRight: 10 }}>
                  {id}
                </Link>
              ))}
            </div>
          )}
        </div>
      </Section>
    </>
  );
}
