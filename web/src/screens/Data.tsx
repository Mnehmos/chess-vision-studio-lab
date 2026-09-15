import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type {
  ArmPreview,
  CatalogResponse,
  Dataset,
  LabelSetRef,
  MigrationDiff,
  Normalization,
  StackPreview,
} from "../generated/schemas";
import { Empty, ErrorBox, Loading, Section, fmtInt, formatValue } from "../ui";

type Tab = "catalog" | "stack" | "lineage";

const FILTER_FIELDS = ["phase", "stm", "source_id", "material", "eval_bucket", "result",
  "label_family", "authority", "tier", "producer", "motif", "strategy", "dataset", "split",
  "ply_min", "ply_max"] as const;

// Mirror of cvslab.facts GEOMETRY_FAMILIES keys (registry v1 — stable contract).
const FACT_KEYS = [
  "KING_DANGER", "KING_ZONE_PRESSURE", "KING_OPEN_FILE", "KING_SHIELD", "KING_CENTRAL_EXPOSURE",
  "ENEMY_QUEEN_NEAR_KING", "OPEN_CENTER_KING", "KING_ESCAPE_DEFICIT", "HANGING_MATERIAL",
  "MOBILITY_KNIGHT", "MOBILITY_BISHOP", "MOBILITY_ROOK", "MOBILITY_QUEEN", "PASSED_PAWN",
  "CONNECTED_PASSED_PAWN", "ROOK_OPEN_FILE", "ROOK_SEMI_OPEN_FILE", "ROOK_SEVENTH",
  "DOUBLED_PAWN", "ISOLATED_PAWN", "BISHOP_PAIR",
];

function Facet({ label, counts, onPick }: { label: string; counts: Record<string, number>; onPick?: (value: string) => void }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  if (entries.length === 0) return null;
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
        {entries.map(([value, count]) => (
          <button
            key={value}
            style={{ padding: "1px 8px", fontSize: 11, fontWeight: 400 }}
            title={onPick ? `filter ${label}=${value}` : undefined}
            onClick={onPick ? () => onPick(value) : undefined}
          >
            {value} <span style={{ color: "var(--muted)" }}>{fmtInt(count)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function CatalogTab({ normalizations }: { normalizations: Normalization[] }) {
  const [normalizationId, setNormalizationId] = useState("");
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [factKey, setFactKey] = useState("");
  const [factMin, setFactMin] = useState("1");
  const [factFavors, setFactFavors] = useState("");
  const [offset, setOffset] = useState(0);
  const [sortBy, setSortBy] = useState("");
  const limit = 25;
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!normalizationId && normalizations.length > 0) setNormalizationId(normalizations[normalizations.length - 1].id);
  }, [normalizations, normalizationId]);

  useEffect(() => {
    if (!normalizationId) return;
    const params: Record<string, string | number | undefined> = { normalization_id: normalizationId, offset, limit };
    if (sortBy) params.sort_by = sortBy;
    for (const [key, value] of Object.entries(filters)) if (value) params[key] = value;
    if (factKey) {
      params.fact_key = factKey;
      params.fact_min = factMin || "1";
      if (factFavors) params.fact_favors = factFavors;
    }
    api.catalog(params).then(setData).catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, [normalizationId, filters, offset, sortBy, factKey, factMin, factFavors]);

  const setFilter = (key: string, value: string) => {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setOffset(0);
  };

  return (
    <>
      <div className="filters">
        <label>
          Sort by
          <select value={sortBy} onChange={(event) => { setSortBy(event.target.value); setOffset(0); }}>
            <option value="">record id</option>
            <option value="fact:HANGING_MATERIAL">fact: hanging material ↓</option>
            <option value="fact:KING_DANGER">fact: king danger ↓</option>
            <option value="fact:PASSED_PAWN">fact: passed pawns ↓</option>
            <option value="fact:MOBILITY_KNIGHT">fact: knight mobility ↓</option>
            <option value="label_count">label count ↓</option>
            <option value="ply">ply</option>
          </select>
        </label>
        <label>
          CVS fact ≥ bucket
          <select value={factKey} onChange={(event) => { setFactKey(event.target.value); setOffset(0); }}>
            <option value="">any</option>
            {FACT_KEYS.map((key) => (
              <option key={key} value={key}>{key.toLowerCase().replace(/_/g, " ")}</option>
            ))}
          </select>
        </label>
        {factKey && (
          <>
            <label>
              Min bucket
              <select value={factMin} onChange={(event) => { setFactMin(event.target.value); setOffset(0); }}>
                {["1", "2", "3"].map((bucket) => <option key={bucket}>{bucket}</option>)}
              </select>
            </label>
            <label>
              Favors
              <select value={factFavors} onChange={(event) => { setFactFavors(event.target.value); setOffset(0); }}>
                <option value="">either side</option>
                <option value="white">white</option>
                <option value="black">black</option>
              </select>
            </label>
          </>
        )}
        <label>
          Normalization
          <select value={normalizationId} onChange={(event) => { setNormalizationId(event.target.value); setOffset(0); }}>
            {normalizations.map((norm) => (
              <option key={norm.id} value={norm.id}>
                {norm.id} {norm.name} ({fmtInt(norm.record_count)} records)
              </option>
            ))}
          </select>
        </label>
        {FILTER_FIELDS.map((field) => (
          <label key={field}>
            {field}
            <input
              value={filters[field] ?? ""}
              placeholder="all"
              style={{ width: 110 }}
              onChange={(event) => setFilter(field, event.target.value)}
            />
          </label>
        ))}
      </div>
      <ErrorBox error={error} />
      {!data ? <Loading /> : (
        <>
          <div className="grid cols-4" style={{ marginBottom: 16 }}>
            <div className="stat">
              <div className="label">Matching records</div>
              <div className="value">{fmtInt(data.total_matching)}</div>
              <div className="note">of {fmtInt(data.record_count)} canonical · {fmtInt(data.duplicate_count)} duplicates dropped</div>
            </div>
            <Facet label="phase" counts={data.facets.phase ?? {}} onPick={(value) => setFilter("phase", value)} />
            <Facet label="tier" counts={data.facets.tier ?? {}} onPick={(value) => setFilter("tier", value)} />
            <Facet label="authority" counts={data.facets.authority ?? {}} onPick={(value) => setFilter("authority", value)} />
            <Facet label="label_family" counts={data.facets.label_family ?? {}} onPick={(value) => setFilter("label_family", value)} />
            <Facet label="source" counts={data.facets.source_id ?? {}} onPick={(value) => setFilter("source_id", value)} />
            <Facet label="eval_bucket" counts={data.facets.eval_bucket ?? {}} onPick={(value) => setFilter("eval_bucket", value)} />
            <Facet label="dataset" counts={data.facets.dataset ?? {}} onPick={(value) => setFilter("dataset", value)} />
            <Facet label="motif (CVS tactics)" counts={data.facets.motif ?? {}} onPick={(value) => setFilter("motif", value)} />
            <Facet label="strategy tags" counts={data.facets.strategy ?? {}} onPick={(value) => setFilter("strategy", value)} />
          </div>
          {data.page.length === 0 ? <Empty>No records match these filters.</Empty> : (
            <div className="panel" style={{ padding: 0, overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Record</th>
                    <th>Phase</th>
                    <th>stm</th>
                    <th>Material</th>
                    <th className="num">Ply</th>
                    <th>Labels</th>
                    <th>In datasets</th>
                    <th>FEN</th>
                  </tr>
                </thead>
                <tbody>
                  {data.page.map((record) => (
                    <tr key={`${record.record_id}:${record.source_row}`}>
                      <td className="mono" title={record.epd}>{record.record_id.slice(0, 14)}…</td>
                      <td>{record.phase}</td>
                      <td className="mono">{record.stm}</td>
                      <td className="mono" title={record.material}>{record.material}</td>
                      <td className="num mono">{record.ply}</td>
                      <td title={record.labels.map((label) => `${label.family}=${formatValue(String(label.value))} (${label.authority})`).join("\n")}>
                        {record.labels.map((label) => (
                          <span key={`${label.family}:${label.authority}`} className="badge" style={{ marginRight: 4 }}>
                            {String(label.family)}
                          </span>
                        ))}
                      </td>
                      <td className="mono">{Object.entries(record.memberships).map(([datasetId, splits]) => `${datasetId}:${splits.join("/")}`).join(" ") || "—"}</td>
                      <td className="mono" style={{ color: "var(--muted)", maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis" }}>{record.fen}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="filters" style={{ marginTop: 10 }}>
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>Previous</button>
            <span className="subtitle">records {fmtInt(offset + 1)}–{fmtInt(Math.min(offset + limit, data.total_matching))} of {fmtInt(data.total_matching)}</span>
            <button disabled={offset + limit >= data.total_matching} onClick={() => setOffset(offset + limit)}>Next</button>
          </div>
        </>
      )}
    </>
  );
}

interface DraftArm {
  name: string;
  field: string;
  value: string;
  policy: string;
  rows: string;
  fraction: string;
  balanceBucket: string;
  balanceCap: string;
  seed: string;
}

function emptyArm(index: number): DraftArm {
  return { name: `arm${index + 1}`, field: "phase", value: "opening", policy: "all", rows: "1000",
           fraction: "0.5", balanceBucket: "stm", balanceCap: "100", seed: "0" };
}

function draftToArm(arm: DraftArm): object {
  const spec: Record<string, unknown> = {
    name: arm.name || "arm",
    filter: arm.value ? { [arm.field]: arm.value } : {},
    policy: arm.policy,
    seed: Number(arm.seed) || 0,
  };
  if (arm.policy === "fixed_rows") spec.rows = Number(arm.rows);
  if (arm.policy === "fraction") spec.fraction = Number(arm.fraction);
  if (arm.policy === "balance") { spec.balance_bucket = arm.balanceBucket; spec.balance_cap = Number(arm.balanceCap); }
  return spec;
}

function StackTab({ normalizations, datasets, onFrozen }: {
  normalizations: Normalization[];
  datasets: Dataset[];
  onFrozen: () => void;
}) {
  const [normalizationId, setNormalizationId] = useState("");
  const [arms, setArms] = useState<DraftArm[]>([emptyArm(0)]);
  const [name, setName] = useState("stacked-view");
  const [seed, setSeed] = useState("0");
  const [preview, setPreview] = useState<StackPreview | null>(null);
  const [frozen, setFrozen] = useState<Dataset | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!normalizationId && normalizations.length > 0) setNormalizationId(normalizations[normalizations.length - 1].id);
  }, [normalizations, normalizationId]);

  const runPreview = useCallback(() => {
    if (!normalizationId) return;
    api.stackPreview(normalizationId, arms.map(draftToArm))
      .then((value) => { setPreview(value); setError(null); })
      .catch((exc: unknown) => { setPreview(null); setError(exc instanceof Error ? exc.message : String(exc)); });
  }, [normalizationId, arms]);

  useEffect(() => { runPreview(); }, [runPreview]);

  async function freeze() {
    setBusy(true);
    setError(null);
    try {
      const dataset = await api.freezeStack({ normalization_id: normalizationId, name, arms: arms.map(draftToArm), split_seed: Number(seed) || 0 });
      setFrozen(dataset);
      onFrozen();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  function updateArm(index: number, patch: Partial<DraftArm>) {
    setArms((previous) => previous.map((arm, i) => (i === index ? { ...arm, ...patch } : arm)));
  }

  return (
    <>
      <p className="subtitle">
        Compose filtered subpopulations under explicit sampling policies. The preview shows raw
        available rows versus the effective sampled contribution of every arm — nothing is silent.
      </p>
      <div className="filters">
        <label>
          Normalization
          <select value={normalizationId} onChange={(event) => { setNormalizationId(event.target.value); setPreview(null); setFrozen(null); }}>
            {normalizations.map((norm) => (
              <option key={norm.id} value={norm.id}>{norm.id} {norm.name}</option>
            ))}
          </select>
        </label>
      </div>
      <ErrorBox error={error} />
      {arms.map((arm, index) => (
        <div className="panel" key={index}>
          <div className="filters" style={{ marginBottom: 0 }}>
            <label>
              Name
              <input value={arm.name} style={{ width: 110 }} onChange={(event) => updateArm(index, { name: event.target.value })} />
            </label>
            <label>
              Filter field
              <select value={arm.field} onChange={(event) => updateArm(index, { field: event.target.value })}>
                {["phase", "stm", "source_id", "material", "eval_bucket", "result"].map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>
              equals
              <input value={arm.value} style={{ width: 130 }} onChange={(event) => updateArm(index, { value: event.target.value })} />
            </label>
            <label>
              Policy
              <select value={arm.policy} onChange={(event) => updateArm(index, { policy: event.target.value })}>
                {["all", "fraction", "fixed_rows", "balance"].map((policy) => <option key={policy}>{policy}</option>)}
              </select>
            </label>
            {arm.policy === "fraction" && (
              <label>
                Fraction
                <input type="number" step="0.05" min="0.05" max="1" value={arm.fraction} style={{ width: 80 }}
                       onChange={(event) => updateArm(index, { fraction: event.target.value })} />
              </label>
            )}
            {arm.policy === "fixed_rows" && (
              <label>
                Rows
                <input type="number" min="1" value={arm.rows} style={{ width: 90 }}
                       onChange={(event) => updateArm(index, { rows: event.target.value })} />
              </label>
            )}
            {arm.policy === "balance" && (
              <>
                <label>
                  Bucket
                  <select value={arm.balanceBucket} onChange={(event) => updateArm(index, { balanceBucket: event.target.value })}>
                    {["stm", "phase", "material", "source_id", "eval_bucket"].map((bucket) => <option key={bucket}>{bucket}</option>)}
                  </select>
                </label>
                <label>
                  Cap / bucket
                  <input type="number" min="1" value={arm.balanceCap} style={{ width: 80 }}
                         onChange={(event) => updateArm(index, { balanceCap: event.target.value })} />
                </label>
              </>
            )}
            <label>
              Seed
              <input type="number" value={arm.seed} style={{ width: 80 }} onChange={(event) => updateArm(index, { seed: event.target.value })} />
            </label>
            {arms.length > 1 && <button onClick={() => setArms((previous) => previous.filter((_, i) => i !== index))}>Remove</button>}
          </div>
        </div>
      ))}
      <div className="filters">
        <button onClick={() => setArms((previous) => [...previous, emptyArm(previous.length)])}>Add arm</button>
      </div>

      {preview && (
        <Section title="Preview (live, nothing frozen)">
          <div className="panel" style={{ padding: 0 }}>
            <table>
              <thead>
                <tr><th>Arm</th><th>Policy</th><th>Filter</th><th className="num">Available</th><th className="num">Effective</th></tr>
              </thead>
              <tbody>
                {preview.arms.map((arm: ArmPreview) => (
                  <tr key={arm.name}>
                    <td className="mono">{arm.name}</td>
                    <td>{arm.policy}</td>
                    <td className="mono">{Object.entries(arm.filter).map(([key, value]) => `${key}=${formatValue(value)}`).join(", ") || "everything"}</td>
                    <td className="num mono">{fmtInt(arm.available)}</td>
                    <td className="num mono" style={{ color: "var(--accent)" }}>{fmtInt(arm.effective)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid cols-4" style={{ marginTop: 12 }}>
            <div className="stat">
              <div className="label">Effective total</div>
              <div className="value">{fmtInt(preview.effective_total)}</div>
              <div className="note">{fmtInt(preview.unique_records)} unique records</div>
            </div>
            <Facet label="phase mix" counts={preview.distributions.phase ?? {}} />
            <Facet label="source mix" counts={preview.distributions.source ?? {}} />
            <Facet label="label tier mix" counts={preview.distributions.tier ?? {}} />
          </div>
        </Section>
      )}

      <Section title="Freeze this view into an immutable D####">
        <div className="panel">
          <div className="filters" style={{ marginBottom: 0 }}>
            <label>
              Dataset name
              <input value={name} onChange={(event) => setName(event.target.value)} />
            </label>
            <label>
              Split seed
              <input type="number" value={seed} style={{ width: 80 }} onChange={(event) => setSeed(event.target.value)} />
            </label>
            <button className="primary" disabled={!preview || busy || !preview.effective_total} onClick={freeze}>
              Freeze dataset
            </button>
          </div>
          {frozen && (
            <div className="notice" style={{ marginTop: 10 }}>
              Frozen <span className="mono">{frozen.id}</span> {frozen.name} · {fmtInt(frozen.counts.records)} records ·
              manifest {frozen.manifest_hash.slice(0, 24)}… ·{" "}
              {datasets.filter((candidate) => candidate.parent_id === frozen.id).length} rebuilds
            </div>
          )}
        </div>
      </Section>
    </>
  );
}

function LineageTab({ normalizations, datasets, labelSets }: {
  normalizations: Normalization[];
  datasets: Dataset[];
  labelSets: Record<string, LabelSetRef[]>;
}) {
  const [migrateFrom, setMigrateFrom] = useState("");
  const [migrateName, setName] = useState("canon-v2");
  const [threshold, setThreshold] = useState("2600");
  const [diff, setDiff] = useState<MigrationDiff | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rebuildTarget, setRebuildTarget] = useState("");
  const [rebuildNorm, setRebuildNorm] = useState("");
  const [rebuilt, setRebuilt] = useState<Dataset | null>(null);

  const runDiff = useCallback((fromId: string, toId: string) => {
    if (!fromId || !toId || fromId === toId) { setDiff(null); return; }
    api.migrationDiff(fromId, toId).then(setDiff).catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, []);

  async function migrate() {
    setBusy(true);
    setError(null);
    try {
      const created = await api.migrate({
        from_normalization_id: migrateFrom,
        name: migrateName,
        settings_overrides: { phase_middlegame_min_material: Number(threshold) || 2600 },
      });
      runDiff(migrateFrom, created.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function rebuild() {
    setBusy(true);
    setError(null);
    try {
      setRebuilt(await api.rebuildDataset(rebuildTarget, rebuildNorm));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Section title="Lineage">
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr><th>Normalization</th><th className="num">Records</th><th className="num">Duplicates</th><th>Settings</th><th>Label sets (L2)</th><th>Recipe</th></tr>
            </thead>
            <tbody>
              {normalizations.map((norm) => (
                <tr key={norm.id}>
                  <td><span className="mono">{norm.id}</span> {norm.name}</td>
                  <td className="num mono">{fmtInt(norm.record_count)}</td>
                  <td className="num mono">{fmtInt(norm.duplicate_count)}</td>
                  <td className="mono" style={{ color: "var(--muted)" }}>
                    {Object.entries(norm.settings).filter(([key]) => key.startsWith("phase")).map(([key, value]) => `${key}=${formatValue(value)}`).join(" · ")}
                  </td>
                  <td className="mono" style={{ color: "var(--muted)" }}>
                    {(labelSets[norm.id] ?? []).map((ref) =>
                      `${ref.rows} rows [${Object.entries(ref.families).map(([family, count]) => `${family}×${count}`).join(", ")}]`
                    ).join(" · ") || "—"}
                  </td>
                  <td className="mono" title={norm.recipe_hash}>{norm.recipe_hash.slice(0, 19)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr><th>Dataset</th><th>On</th><th>Parent</th><th className="num">Records</th><th>Stack</th><th>Manifest</th></tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td><span className="mono">{dataset.id}</span> {dataset.name}</td>
                  <td className="mono">{dataset.normalization_id}</td>
                  <td className="mono">{dataset.parent_id ?? "—"}</td>
                  <td className="num mono">{fmtInt(dataset.counts.records)}</td>
                  <td>{dataset.stack.length > 0 ? dataset.stack.map((arm) => arm.name).join(" + ") : "flat selection"}</td>
                  <td className="mono" title={dataset.manifest_hash}>{dataset.manifest_hash.slice(0, 19)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <ErrorBox error={error} />
      <Section title="Re-standardize (creates a new N#### from the same immutable sources)">
        <div className="panel">
          <div className="filters" style={{ marginBottom: 0 }}>
            <label>
              From
              <select value={migrateFrom} onChange={(event) => setMigrateFrom(event.target.value)}>
                <option value="">— pick —</option>
                {normalizations.map((norm) => <option key={norm.id} value={norm.id}>{norm.id} {norm.name}</option>)}
              </select>
            </label>
            <label>
              New name
              <input value={migrateName} onChange={(event) => setName(event.target.value)} />
            </label>
            <label>
              phase_middlegame_min_material
              <input type="number" value={threshold} style={{ width: 90 }} onChange={(event) => setThreshold(event.target.value)} />
            </label>
            <button className="primary" disabled={!migrateFrom || busy} onClick={migrate}>Migrate</button>
          </div>
        </div>
      </Section>

      {diff && (
        <Section title={`Migration diff ${diff.from_normalization_id} → ${diff.to_normalization_id}`}>
          <div className="grid cols-4">
            <div className="stat"><div className="label">Records</div><div className="value">{fmtInt(diff.records_from)} → {fmtInt(diff.records_to)}</div><div className="note">+{fmtInt(diff.added)} / −{fmtInt(diff.removed)} / ~{fmtInt(diff.changed)} changed</div></div>
            <div className="stat"><div className="label">Duplicates</div><div className="value">{fmtInt(diff.duplicate_delta)}</div><div className="note">rejected {fmtInt(diff.rejected_delta)}</div></div>
            <div className="stat"><div className="label">Coverage shift</div>
              <div className="note mono" style={{ marginTop: 6 }}>
                {Object.keys(diff.coverage_to.phase).map((phase) => (
                  <div key={phase}>{phase}: {fmtInt(diff.coverage_from.phase[phase] ?? 0)} → {fmtInt(diff.coverage_to.phase[phase] ?? 0)}</div>
                ))}
              </div>
            </div>
            <div className="stat"><div className="label">Affected</div>
              <div className="note mono" style={{ marginTop: 6 }}>
                datasets {diff.affected_dataset_ids.join(", ") || "—"}<br />
                runs {diff.affected_run_ids.join(", ") || "—"}
              </div>
            </div>
          </div>
          {Object.keys(diff.label_changes).length > 0 && (
            <div className="panel">
              <strong>Label changes</strong>
              <ul className="plain">
                {Object.entries(diff.label_changes).map(([family, changes]) => (
                  <li key={family} className="mono">{family}: {Object.entries(changes).map(([kind, count]) => `${kind} ${count}`).join(", ")}</li>
                ))}
              </ul>
            </div>
          )}
          {diff.split_changes.length > 0 && (
            <div className="panel" style={{ padding: 0 }}>
              <table>
                <thead><tr><th>Dataset split membership</th><th className="num">Surviving records</th><th className="num">Split changes</th></tr></thead>
                <tbody>
                  {diff.split_changes.map((change) => (
                    <tr key={change.dataset_id}>
                      <td className="mono"><Link to={`/object/D${change.dataset_id.slice(1)}`}>{change.dataset_id}</Link></td>
                      <td className="num mono">{fmtInt(change.surviving_records)}</td>
                      <td className="num mono">{fmtInt(change.split_changes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {diff.notes.map((note, index) => <div className="notice" key={index}>{note}</div>)}
        </Section>
      )}

      <Section title="Rebuild a frozen dataset under a new standard (descendant D####)">
        <div className="panel">
          <div className="filters" style={{ marginBottom: 0 }}>
            <label>
              Dataset
              <select value={rebuildTarget} onChange={(event) => setRebuildTarget(event.target.value)}>
                <option value="">— pick —</option>
                {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.id} {dataset.name} (on {dataset.normalization_id})</option>)}
              </select>
            </label>
            <label>
              Under
              <select value={rebuildNorm} onChange={(event) => setRebuildNorm(event.target.value)}>
                <option value="">— pick —</option>
                {normalizations.map((norm) => <option key={norm.id} value={norm.id}>{norm.id} {norm.name}</option>)}
              </select>
            </label>
            <button disabled={!rebuildTarget || !rebuildNorm || busy} onClick={rebuild}>Rebuild</button>
          </div>
          {rebuilt && (
            <div className="notice" style={{ marginTop: 10 }}>
              Created descendant <span className="mono">{rebuilt.id}</span> from {rebuilt.parent_id} ·{" "}
              {fmtInt(rebuilt.counts.records)} records on {rebuilt.normalization_id} · the original manifest is untouched
            </div>
          )}
        </div>
      </Section>
    </>
  );
}

export default function Data() {
  const [tab, setTab] = useState<Tab>("catalog");
  const [normalizations, setNormalizations] = useState<Normalization[] | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [labelSets, setLabelSets] = useState<Record<string, LabelSetRef[]>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.normalizations().then((norms) => {
      setNormalizations(norms);
      const missing = norms.filter((norm) => !(norm.id in labelSets));
      if (missing.length === 0) return null;
      return Promise.all(missing.map((norm) => api.labelSets(norm.id)))
        .then((sets) => setLabelSets((previous) => {
          const next = { ...previous };
          missing.forEach((norm, index) => { next[norm.id] = sets[index]; });
          return next;
        }));
    }).catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
    api.datasets().then(setDatasets).catch(() => setDatasets([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  if (!normalizations) {
    if (error) return <ErrorBox error={error} />;
    return <Loading />;
  }
  if (normalizations.length === 0) {
    return <Empty>No normalizations yet — run `cvslab demo` or `cvslab normalize` first.</Empty>;
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: "catalog", label: "Corpus catalog" },
    { key: "stack", label: "Stack & freeze" },
    { key: "lineage", label: "Lineage & migration" },
  ];

  return (
    <>
      <h1>Data workbench</h1>
      <p className="subtitle">
        S#### raw sources → N#### normalization → canonical records → accumulated labels → D#### frozen datasets.
        Live views are exploratory; only frozen manifests can train.
      </p>
      <div className="filters">
        {tabs.map((entry) => (
          <button key={entry.key} className={tab === entry.key ? "primary" : ""} onClick={() => setTab(entry.key)}>
            {entry.label}
          </button>
        ))}
      </div>
      {tab === "catalog" && <CatalogTab normalizations={normalizations} />}
      {tab === "stack" && <StackTab normalizations={normalizations} datasets={datasets} onFrozen={refresh} />}
      {tab === "lineage" && <LineageTab normalizations={normalizations} datasets={datasets} labelSets={labelSets} />}
    </>
  );
}
