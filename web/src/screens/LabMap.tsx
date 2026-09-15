import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { MapNode, MapResponse } from "../generated/schemas";
import { Empty, ErrorBox, Loading, formatValue } from "../ui";

const PROJECTIONS = [
  { key: "data", label: "Data lineage" },
  { key: "research", label: "Research lineage" },
  { key: "promotion", label: "Promotion tree" },
];

const STATE_COLOR: Record<string, string> = {
  SUPPORTED: "var(--good)", COMPLETED: "var(--good)", PROMOTED: "var(--good)",
  RUNNING: "var(--accent)", GATING: "var(--accent)",
  QUEUED: "var(--muted)", PROPOSED: "var(--muted)", SUPERSEDED: "var(--muted)",
  INCONCLUSIVE: "var(--warn)", HOLD: "var(--warn)",
  REJECTED: "var(--bad)",
  INVALID: "var(--invalid)", NOT_ELIGIBLE: "var(--invalid)",
};

const KIND_ORDER: Record<string, string[]> = {
  data: ["source", "labels", "normalization", "dataset", "model"],
  research: ["hypothesis", "baseline", "ablation", "run", "model", "finding"],
  promotion: ["baseline", "ablation", "finding"],
};

const NODE_W = 236;
const NODE_H = 52;
const COL_GAP = 64;
const ROW_GAP = 18;
const PAD = 24;

interface Positioned extends MapNode {
  x: number;
  y: number;
  column: number;
}

function colorFor(node: MapNode): string {
  return (node.state && STATE_COLOR[node.state]) || "var(--accent)";
}

export default function LabMap() {
  const [projection, setProjection] = useState("data");
  const [generation, setGeneration] = useState("");
  const [state, setState] = useState("");
  const [family, setFamily] = useState("");
  const [data, setData] = useState<MapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<MapNode | null>(null);

  useEffect(() => {
    api
      .mapView(projection, generation || undefined, state || undefined, family || undefined)
      .then((response) => {
        setData(response);
        setError(null);
        setSelected(null);
      })
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, [projection, generation, state, family]);

  const positioned = useMemo<Positioned[]>(() => {
    if (!data) return [];
    const order = KIND_ORDER[projection] ?? [];
    const columns = new Map<string, MapNode[]>();
    for (const node of data.nodes) {
      const column = order.indexOf(node.kind);
      const key = String(column >= 0 ? column : 99);
      if (!columns.has(key)) columns.set(key, []);
      columns.get(key)!.push(node);
    }
    const sortedKeys = [...columns.keys()].sort((a, b) => Number(a) - Number(b));
    const out: Positioned[] = [];
    sortedKeys.forEach((key, columnIndex) => {
      const column = columns.get(key)!;
      column
        .slice()
        .sort((a, b) => a.id.localeCompare(b.id))
        .forEach((node, rowIndex) => {
          out.push({ ...node, column: columnIndex, x: PAD + columnIndex * (NODE_W + COL_GAP), y: PAD + rowIndex * (NODE_H + ROW_GAP) });
        });
    });
    return out;
  }, [data, projection]);

  const byId = useMemo(() => new Map(positioned.map((node) => [node.id, node])), [positioned]);
  const width = useMemo(() => Math.max(600, ...positioned.map((node) => node.x + NODE_W + PAD)), [positioned]);
  const height = useMemo(() => Math.max(400, ...positioned.map((node) => node.y + NODE_H + PAD)), [positioned]);

  return (
    <>
      <h1>Lab Map</h1>
      <p className="subtitle">
        Spatial view of the research topology, generated entirely from canonical lab objects — every
        node and edge resolves to real evidence. Provenance flows left to right; distance carries no
        metric meaning. Click a node for its evidence inspector.
      </p>
      <div className="filters">
        {PROJECTIONS.map((entry) => (
          <button key={entry.key} className={projection === entry.key ? "primary" : ""}
                  onClick={() => setProjection(entry.key)}>
            {entry.label}
          </button>
        ))}
        <label style={{ marginLeft: 16 }}>
          Generation
          <input value={generation} placeholder="all" style={{ width: 70 }}
                 onChange={(event) => setGeneration(event.target.value)} />
        </label>
        <label>
          State
          <input value={state} placeholder="all" style={{ width: 130 }}
                 onChange={(event) => setState(event.target.value)} />
        </label>
        <label>
          Family
          <input value={family} placeholder="all" style={{ width: 90 }}
                 onChange={(event) => setFamily(event.target.value)} />
        </label>
      </div>
      <ErrorBox error={error} />
      {!data ? (
        <Loading />
      ) : data.nodes.length === 0 ? (
        <Empty>No canonical objects match these filters.</Empty>
      ) : (
        <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
          <div className="scaling-wrap" style={{ flexGrow: 1, overflowX: "auto" }}>
            <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
              {positioned.map((node) =>
                data.edges
                  .filter((edge) => edge.src === node.id && byId.has(edge.dst))
                  .map((edge) => {
                    const target = byId.get(edge.dst)!;
                    const x1 = node.x + NODE_W;
                    const y1 = node.y + NODE_H / 2;
                    const x2 = target.x;
                    const y2 = target.y + NODE_H / 2;
                    const bend = Math.max(40, Math.abs(x2 - x1) / 2);
                    return (
                      <path key={`${edge.src}->${edge.dst}:${edge.kind}`}
                            d={`M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`}
                            fill="none" stroke="var(--border)" strokeWidth={1.5} opacity={0.9}>
                        <title>{edge.kind}</title>
                      </path>
                    );
                  }),
              )}
              {positioned.map((node) => (
                <g key={node.id} onClick={() => setSelected(node)} style={{ cursor: "pointer" }}>
                  <title>{node.label}</title>
                  <rect x={node.x + 3} y={node.y + 3} width={NODE_W} height={NODE_H} rx={8}
                        fill={selected?.id === node.id ? "rgba(88,166,255,0.25)" : "var(--panel)"}
                        stroke={colorFor(node)} strokeWidth={selected?.id === node.id ? 2 : 1.2} />
                  <rect x={node.x + 3} y={node.y + 3} width={5} height={NODE_H} rx={2} fill={colorFor(node)} />
                  <text x={node.x + 16} y={node.y + 22} fill="var(--text)" fontSize={12}
                        style={{ fontFamily: "var(--mono)" }}>
                    {node.label.length > 30 ? `${node.label.slice(0, 29)}…` : node.label}
                  </text>
                  <text x={node.x + 16} y={node.y + 40} fill="var(--muted)" fontSize={10}>
                    {node.state ? `${node.kind} · ${node.state}` : node.kind}
                  </text>
                </g>
              ))}
            </svg>
          </div>

          {selected && (
            <div className="panel" style={{ width: 340, flexShrink: 0 }}>
              <div className="head">
                <strong>{selected.kind}</strong> <span className="mono">{selected.id}</span>
              </div>
              <p style={{ margin: "6px 0" }}>{selected.label}</p>
              {selected.state && (
                <p style={{ margin: "6px 0" }}>
                  State: <span className="badge" style={{ color: colorFor(selected), borderColor: colorFor(selected) }}>
                    {selected.state}
                  </span>
                </p>
              )}
              <dl className="kv">
                {Object.entries(selected.detail).map(([key, value]) => (
                  <>
                    <dt key={`t${key}`} className="mono">{key}</dt>
                    <dd key={`d${key}`}>{formatValue(value as string)}</dd>
                  </>
                ))}
              </dl>
              {selected.link && (
                <p style={{ marginTop: 12 }}>
                  <Link to={selected.link} className="mono">open evidence page →</Link>
                </p>
              )}
            </div>
          )}
        </div>
      )}
      {data && data.nodes.length > 0 && (
        <p className="subtitle" style={{ marginTop: 8 }}>
          {data.nodes.length} objects · {data.edges.length} provenance edges · projection{" "}
          {data.projection}. Historical relationships are read-only here — precise comparisons live in
          the 2D workbench.
        </p>
      )}
    </>
  );
}
