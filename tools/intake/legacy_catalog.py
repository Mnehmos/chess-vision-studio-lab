#!/usr/bin/env python3
"""Legacy intake catalog: inventory every pre-lab artifact so the Phase 0 registry can
import it instead of rediscovering it.

What already exists before the lab:
  engine repo   nets (nets/, target-cvs/), engine registry (benchmarks/engines.json),
                search switches (benchmarks/search-switches.json), eval suites
                (benchmarks/suites/), gate + anchor evidence (benchmarks/results/),
                training corpora (training/gen*/)
  app repo      arena datasets (arena/out/)
  tools dir     frozen baseline binaries/nets (cvs-baselines/), gen8-era label files

This tool assigns NO permanent lab identities (M/S/D/E/R/...). Every entry carries a
`proposedKind` and a content hash so the lab registry can assign IDs once, deterministically,
and detect duplicates. Output is regenerated wholesale; the manifest pins the repo commits it
was generated from.

  python tools/intake/legacy_catalog.py --engine ../chess-vision-studio-rust-engine \\
      --app ../chess-vision-studio --tools F:/tools --out catalog
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

TOOL_VERSION = "legacy-catalog-v1"

LAB_STATE = {"promote": "SUPPORTED", "reject": "REJECTED", "hold_for_more_data": "INCONCLUSIVE"}

# Field -> provenance class, for the corpus row formats the pipelines have written.
FIELD_AUTHORITY = {
    "fen": "record",
    "features": "deterministic_geometry", "features_bitset": "deterministic_geometry",
    "res": "game_outcome", "result": "game_outcome",
    "cp_play": "search_derived",
    "cpStatic": "external_oracle",
    "sf_best": "external_oracle", "label_depth": "external_oracle", "label_nodes": "external_oracle",
    "evalBefore": "external_oracle", "topMoves": "external_oracle",
}

# Known roles of named artifacts, each with the document that establishes it.
MODEL_ROLES = {
    "matrix-raw.json": ("flagship-main-net", "engine:benchmarks/N0-identity.json"),
    "matrix-residual.json": ("flagship-helper-net", "engine:benchmarks/N0-identity.json"),
    "eval-cal.json": ("flagship-eval-calibration", "engine:benchmarks/results/nnuecal-gate-20260911 (promote)"),
    "eval-cal-20260911.json": ("flagship-eval-calibration", "engine:benchmarks/results/nnuecal-gate-20260911 (promote)"),
    "matrix-flat.json": ("gen9-candidate", "engine:benchmarks/engines.json g9.flat-core104.h256"),
    "matrix-ranker.json": ("gen9-experimental-ranker", "engine:benchmarks/engines.json g9.hybrid-b.raw-plus-ranker"),
    "raw-nnue-h256-sf-d12-v3.json": ("gen7-frozen-baseline", "engine:benchmarks/engines.json g7.raw-h256.sf-d12.v3"),
    "gen11-lich.json": ("gen11-candidate", "engine:benchmarks/results/gen11-gate-20260913 (hold)"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "remote": git("remote", "get-url", "origin")}


def loc(root_name: str, root: Path, p: Path) -> dict:
    return {"root": root_name, "path": p.relative_to(root).as_posix()}


def count_params(obj) -> int:
    """Exact serialized parameter count: every numeric scalar inside list-valued fields."""
    if isinstance(obj, list):
        return sum(count_params(x) for x in obj)
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, (int, float)):
        return 1
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values() if isinstance(v, (list, dict)))
    return 0


# ---------------------------------------------------------------------------------------------


def scan_models(roots: dict[str, Path]) -> list[dict]:
    patterns = [("engine", "nets/*.json"), ("engine", "target-cvs/*.json"), ("tools", "cvs-baselines/*.json")]
    by_hash: dict[str, dict] = {}
    for root_name, pat in patterns:
        root = roots.get(root_name)
        if not root:
            continue
        for f in sorted(root.glob(pat)):
            if f.name.startswith("."):
                continue  # tool state (e.g. cargo's .rustc_info.json), not an artifact
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or not ("points" in data or "modelKind" in data
                                                  or any(isinstance(v, list) and len(v) > 64 for v in data.values())):
                continue  # not a model or calibration
            digest = sha256(f)
            if digest in by_hash:
                by_hash[digest]["locations"].append(loc(root_name, root, f))
                continue
            meta = {}
            kind = "unknown"
            params = None
            if isinstance(data, dict):
                meta = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
                if "points" in data:
                    kind = "eval-calibration"
                    params = len(data["points"]) * 2
                else:
                    kind = str(data.get("modelKind") or "nnue-unlabelled")
                    params = count_params(data)
            role, evidence = MODEL_ROLES.get(f.name, (None, None))
            if role is None and f.name.startswith("gen10-"):
                role, evidence = "gen10-experimental", "engine:training/gen10/README.md (not shippable)"
            by_hash[digest] = {
                "proposedKind": "M", "sha256": digest, "bytes": f.stat().st_size,
                "name": f.name, "modelKind": kind, "serializedParameterCount": params,
                "metadata": meta, "role": role or "unregistered", "roleEvidence": evidence,
                "provenanceComplete": bool(meta.get("trainingCommit") and meta.get("datasetManifestHash")),
                "locations": [loc(root_name, root, f)],
            }
    return sorted(by_hash.values(), key=lambda m: (m["role"], m["name"]))


def sample_fields(f: Path) -> list[str]:
    try:
        with open(f, encoding="utf-8") as fd:
            first = fd.readline()
        row = json.loads(first)
        return sorted(row.keys()) if isinstance(row, dict) else []
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []


def describe_source(name: str, root_name: str, root: Path, files: list[Path], note: str | None) -> dict:
    rows = 0
    entries = []
    for f in files:
        n = 0
        with open(f, "rb") as fd:
            for _ in fd:
                n += 1
        rows += n
        entries.append({**loc(root_name, root, f), "sha256": sha256(f), "bytes": f.stat().st_size, "rows": n})
    fields = sample_fields(files[0]) if files else []
    authority = {k: FIELD_AUTHORITY.get(k, "unknown-needs-audit") for k in fields}
    if "cp" in fields:
        authority["cp"] = "ambiguous: external_oracle or search_derived depending on the producing pipeline (see note)"
    if "cpShallow" in fields:
        authority["cpShallow"] = "external_oracle (shallow)"
    content = hashlib.sha256("".join(e["sha256"] for e in entries).encode()).hexdigest()
    return {"proposedKind": "S", "name": name, "contentSha256": content, "files": len(entries), "rows": rows,
            "bytes": sum(e["bytes"] for e in entries), "sampleFields": fields, "fieldAuthority": authority,
            "note": note, "bytesAvailability": "local", "fileManifest": entries}


def scan_sources(roots: dict[str, Path]) -> list[dict]:
    notes = {
        "gen9-cvs": "gen9 NNUE training shards (train_matrix.py); cp = gen9 corpus label; features = CVS core ids registry v1. Shard metas carry checksums.",
        "gen10/corpus": "build_corpus.py: cp = Stockfish d16, cpShallow = Stockfish d12, stability-filtered quiet positions sampled from gen9 shards.",
        "gen10/corpus-d20": "run_generation.py (Railway): CVS NNUE self-play; cp = Stockfish d20, cp_play = CVS play-depth score, res = game result.",
        "gen10/corpus-d20-local": "run_generation.py local run: same row contract as corpus-d20.",
        "gen10/corpus-static": "build_corpus.py --depth 0: cp = Stockfish static eval (static distillation), 1.5M quiet positions.",
        "gen10/corpus-static-full": "build_corpus.py --depth 0 over the whole gen9 shard source (4.1M quiet deduped positions).",
        "gen10/corpus-lich": "Lichess evaluation-database extract (app arena/evaldb_extract.py) used for gen10/gen11 training.",
    }
    out = []
    eng = roots.get("engine")
    if eng:
        gen9 = sorted((eng / "training/gen9/gen9-cvs").glob("shard-*.jsonl"))
        if gen9:
            out.append(describe_source("gen9-cvs", "engine", eng, gen9, notes["gen9-cvs"]))
        for d in sorted((eng / "training/gen10").glob("corpus*")):
            files = sorted(p for p in d.rglob("*.jsonl"))
            key = f"gen10/{d.name}"
            if files:
                out.append(describe_source(key, "engine", eng, files, notes.get(key)))
    tools = roots.get("tools")
    if tools:
        for f in sorted(tools.glob("*.jsonl")):
            if f.stat().st_size >= 1_000_000:
                out.append(describe_source(f"tools/{f.stem}", "tools", tools, [f], "gen8-era label/corpus file (see engine docs/GEN8_TRAINING_PLAN.md)"))
    app = roots.get("app")
    if app:
        for f in sorted((app / "arena/out").glob("*.jsonl")):
            if f.stat().st_size >= 1_000_000:
                out.append(describe_source(f"app/{f.stem}", "app", app, [f], "arena pipeline output (see app docs/RSI_OODA_PIPELINE.md)"))
    return out


def scan_evaluations(roots: dict[str, Path]) -> list[dict]:
    eng = roots.get("engine")
    if not eng:
        return []
    out = []
    suites = eng / "benchmarks/suites"
    for f in sorted(suites.iterdir()):
        if f.is_file() and f.name != "SHA256SUMS":
            kind = "opening-book" if f.suffix == ".epd" else "position-suite" if f.suffix == ".txt" else "suite-sidecar"
            out.append({"proposedKind": "E", "name": f.name, "kind": kind, "sha256": sha256(f),
                        "bytes": f.stat().st_size, "location": loc("engine", eng, f)})
    gates = {}
    for f in glob.glob(str(eng / "benchmarks/results/**/sprt*.json"), recursive=True):
        if "fixtures" in f:
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        key = (d.get("elo0"), d.get("elo1"), d.get("alpha"), d.get("beta"))
        gates.setdefault(key, 0)
        gates[key] += 1
    for (elo0, elo1, alpha, beta), n in sorted(gates.items(), key=lambda kv: -kv[1]):
        out.append({"proposedKind": "E", "name": f"sprt-elo0={elo0}-elo1={elo1}-a={alpha}-b={beta}", "kind": "game-gate-protocol",
                    "records": n, "runner": "engine:benchmarks/scripts/sprt_runner.py",
                    "note": "BayesElo SPRT; see engine benchmarks/README.md for time control, book and adjudication per gate"})
    return out


def scan_evidence(roots: dict[str, Path]) -> list[dict]:
    eng = roots.get("engine")
    if not eng:
        return []
    out = []
    for f in sorted(glob.glob(str(eng / "benchmarks/results/**/sprt*.json"), recursive=True)):
        p = Path(f)
        if "fixtures" in p.parts:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        date = re.search(r"(\d{8})", p.as_posix())
        out.append({"proposedKind": "R", "type": "sprt-gate", "location": loc("engine", eng, p), "sha256": sha256(p),
                    "date": date.group(1) if date else None, "baselineId": d.get("baselineId"),
                    "candidateId": d.get("candidateId"), "games": d.get("games"),
                    "wdl": [d.get("wins"), d.get("draws"), d.get("losses")], "llr": d.get("llr"),
                    "boundary": d.get("boundary"), "decision": d.get("decision"),
                    "labState": LAB_STATE.get(d.get("decision"), "INCONCLUSIVE")})
    for f in sorted((eng / "benchmarks").glob("ANCHOR_*.json")) + sorted((eng / "benchmarks").glob("SF_EFFICIENCY_*.json")):
        out.append({"proposedKind": "R", "type": "anchor" if "ANCHOR" in f.name else "efficiency",
                    "location": loc("engine", eng, f), "sha256": sha256(f), "labState": "SUPPORTED-AS-MEASUREMENT",
                    "note": "measurement record, not an intervention gate"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--app")
    ap.add_argument("--tools")
    ap.add_argument("--out", default="catalog")
    a = ap.parse_args(argv)
    roots = {k: Path(v).resolve() for k, v in (("engine", a.engine), ("app", a.app), ("tools", a.tools)) if v}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    eng = roots["engine"]
    engines = json.loads((eng / "benchmarks/engines.json").read_text(encoding="utf-8"))
    switches_path = eng / "benchmarks/search-switches.json"
    switches = json.loads(switches_path.read_text(encoding="utf-8")) if switches_path.exists() else None

    parts = {
        "models.json": scan_models(roots),
        "sources.json": scan_sources(roots),
        "evaluations.json": scan_evaluations(roots),
        "evidence.json": scan_evidence(roots),
        "engines.json": {"proposedKind": "engine-identity (M + search profile)", "source": "engine:benchmarks/engines.json",
                         "sha256": sha256(eng / "benchmarks/engines.json"), "registry": engines},
        "switches.json": {"source": "engine:benchmarks/search-switches.json",
                          "sha256": sha256(switches_path) if switches else None, "registry": switches},
    }
    for name, payload in parts.items():
        (out / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    counts = {k: (len(v) if isinstance(v, list) else None) for k, v in parts.items()}
    manifest = {
        "tool": TOOL_VERSION, "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsedSec": round(time.time() - t0, 1),
        "roots": {k: {"path": str(v), "git": git_state(v) if (v / ".git").exists() else None} for k, v in roots.items()},
        "counts": counts,
        "identityPolicy": "no permanent lab IDs assigned; import by content hash",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "README.md").write_text(render_readme(parts, manifest), encoding="utf-8", newline="\n")
    print(json.dumps(counts))
    return 0


def render_readme(parts: dict, manifest: dict) -> str:
    from collections import Counter

    models, sources = parts["models.json"], parts["sources.json"]
    evidence, evals = parts["evidence.json"], parts["evaluations.json"]
    roots = manifest["roots"]

    def pin(name):
        g = (roots.get(name) or {}).get("git") or {}
        return f"`{(g.get('commit') or '?')[:10]}`" + (" (dirty)" if g.get("dirty") else "") if g else "not a git repo"

    L = ["# Legacy intake catalog", "",
         f"Generated by `tools/intake/legacy_catalog.py` ({manifest['tool']}) at {manifest['generatedAt']}. "
         "Do not edit by hand; regenerate.", "",
         "Everything the engine and app repos produced before the lab existed, inventoried by content hash so the "
         "Phase 0 registry (issues #1, #2) can import it. **No lab identities are assigned here.** Each entry's "
         "`proposedKind` says what it would become.", "",
         "| root | pinned at |", "|---|---|"]
    L += [f"| {k} | {pin(k)} |" for k in roots]
    L += ["", "| file | proposed kind | entries | what |", "|---|---|---:|---|",
          f"| `models.json` | M | {len(models)} | nets and calibrations: sha256, locations, metadata, exact serialized parameter count, role, provenance completeness |",
          f"| `sources.json` | S | {len(sources)} | corpora: per-file sha256 and rows ({sum(s['rows'] for s in sources):,} rows, {sum(s['bytes'] for s in sources) / 1e9:.1f} GB), sampled fields mapped to provenance classes |",
          f"| `evaluations.json` | E | {len(evals)} | position suites, opening books, suite sidecars, SPRT gate protocols |",
          f"| `evidence.json` | R | {len(evidence)} | gate records mapped to lab states, plus anchor/efficiency measurements |",
          "| `engines.json` | M + search profile | 1 | engine identity registry (engine `benchmarks/engines.json`) |",
          "| `switches.json` | ablation columns | 1 | generated search-switch registry (engine `benchmarks/search-switches.json`) |",
          "| `manifest.json` | — | 1 | roots, repo commits, counts |", "",
          "## Findings the lab must resolve at import", "",
          f"- **Provenance:** {sum(not m['provenanceComplete'] for m in models)} of {len(models)} models lack "
          "`trainingCommit` + `datasetManifestHash`, so their training data cannot be reconstructed from metadata alone. "
          "Import them as development artifacts, not promotable models.",
          f"- **Roles:** {', '.join(f'{k} {v}' for k, v in sorted(Counter(m['role'] for m in models).items()))}.",
          f"- **Evidence states:** {', '.join(f'{k} {v}' for k, v in sorted(Counter(e['labState'] for e in evidence).items()))}. "
          "Superseded 2026-09-09 records are listed as recorded; engine `benchmarks/INV1_GATE_INTEGRITY_2026-09-09.md` "
          "says which were superseded.",
          f"- **Ambiguous labels:** {sum(1 for s in sources if any('ambiguous' in str(v) or 'unknown' in str(v) for v in s['fieldAuthority'].values()))} "
          "sources have a `cp` or other field whose authority depends on the producing pipeline. The first N#### "
          "normalization must assign each an explicit provenance class.",
          "- **Duplicates:** identical bytes appear once, with every location listed. Row-level duplicates across "
          "corpora (for example gen9-train vs gen9-cvs) are not detected here; that belongs to canonical record identity (#2).",
          ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
