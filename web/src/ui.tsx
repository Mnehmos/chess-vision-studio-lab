import type { ReactNode } from "react";

/** Mirror of cvslab.families.format_params. */
export function formatParams(n: number): string {
  if (n < 1000) return String(n);
  for (const [divisor, suffix] of [
    [1_000_000_000, "B"],
    [1_000_000, "M"],
    [1_000, "K"],
  ] as const) {
    if (n >= divisor) {
      const value = n / divisor;
      const text = value >= 10 ? value.toFixed(0) : value.toFixed(1);
      return text.replace(/\.0$/, "") + suffix;
    }
  }
  return String(n);
}

/** Mirror of cvslab.families.format_value. */
export function formatValue(value: boolean | number | string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "1" : "0";
  return String(value);
}

export function fmt(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) return value.toExponential(2);
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US");
}

export function shortHash(hash: string | null | undefined): string {
  if (!hash) return "—";
  return hash.length > 19 ? `${hash.slice(0, 19)}…` : hash;
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(seconds)) return iso;
  if (seconds < 90) return `${Math.max(0, Math.round(seconds))}s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 129600) return `${Math.round(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export function Badge({ children, tone }: { children: ReactNode; tone?: string }) {
  const cls = tone ?? String(children ?? "");
  return <span className={`badge ${cls}`}>{String(children ?? "")}</span>;
}

export function Section({
  title,
  count,
  children,
  right,
}: {
  title: string;
  count?: number | string;
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <section>
      <div className="section-title">
        <h2>{title}</h2>
        {count !== undefined && <span className="count">{count}</span>}
        {right && <span style={{ marginLeft: "auto" }}>{right}</span>}
      </div>
      {children}
    </section>
  );
}

export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="panel">
      {title && <h2 style={{ marginTop: 0 }}>{title}</h2>}
      {children}
    </div>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="error-box">{error}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Loading() {
  return <span className="spin">loading…</span>;
}
