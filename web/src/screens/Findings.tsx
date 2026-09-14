import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { Finding } from "../generated/schemas";
import { Badge, Empty, ErrorBox, Loading, Section, fmt, formatParams, timeAgo } from "../ui";

const RESULTS = ["", "SUPPORTED", "REJECTED", "INCONCLUSIVE", "INVALID", "SUPERSEDED"];

export default function Findings() {
  const [search, setSearch] = useState("");
  const [result, setResult] = useState("");
  const [findings, setFindings] = useState<Finding[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      api
        .findings(search || undefined, result || undefined)
        .then((response) => {
          setFindings(response);
          setError(null);
        })
        .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc)));
    }, 250);
    return () => clearTimeout(timer);
  }, [search, result]);

  return (
    <>
      <h1>Findings</h1>
      <p className="subtitle">
        Standardized finding cards. Negative and inconclusive results are first-class evidence — they stay visible and
        searchable, and a revision supersedes rather than edits.
      </p>
      <div className="filters">
        <label style={{ flexGrow: 1 }}>
          Search
          <input style={{ width: "100%", minWidth: 240 }} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="hypothesis, metric, interpretation…" />
        </label>
        <label>
          Result
          <select value={result} onChange={(event) => setResult(event.target.value)}>
            {RESULTS.map((value) => (
              <option key={value} value={value}>
                {value || "ALL"}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ErrorBox error={error} />
      {!findings ? (
        <Loading />
      ) : findings.length === 0 ? (
        <Empty>No findings match. Draft one with `cvslab finding` once both arms have completed runs.</Empty>
      ) : (
        <Section title="Cards" count={findings.length}>
          <div className="grid cols-2">
            {findings
              .slice()
              .reverse()
              .map((finding) => (
                <div className="panel finding-card" key={finding.id}>
                  <div className="head">
                    <Link to={`/findings/${finding.id}`} className="mono">
                      {finding.id}
                    </Link>
                    <Badge>{finding.result}</Badge>
                    <Badge>{finding.promotion_state}</Badge>
                    <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 12 }}>{timeAgo(finding.created_at)}</span>
                  </div>
                  <div>
                    <strong>{finding.hypothesis_title}</strong>
                  </div>
                  {finding.effect ? (
                    <div className="effect mono">
                      {finding.metric} {finding.effect.difference > 0 ? "+" : ""}
                      {fmt(finding.effect.difference)} [{fmt(finding.effect.ci_low)}, {fmt(finding.effect.ci_high)}] · n{" "}
                      {finding.effect.n.toLocaleString("en-US")}
                    </div>
                  ) : (
                    <div className="effect">no effect estimate (integrity failed)</div>
                  )}
                  <div className="arms">
                    control {finding.control.ablation_id} ({formatParams(finding.control.param_count)} params, seeds{" "}
                    {finding.control.seeds.join("/")}) → intervention {finding.intervention.ablation_id} (
                    {formatParams(finding.intervention.param_count)} params, seeds {finding.intervention.seeds.join("/")})
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: 12 }}>{finding.interpretation}</div>
                </div>
              ))}
          </div>
        </Section>
      )}
    </>
  );
}
