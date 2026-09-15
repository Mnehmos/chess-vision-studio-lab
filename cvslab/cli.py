"""Command-line interface: a thin shell over :mod:`cvslab.service`.

Every command prints canonical JSON (except ``log``, which prints the run log
and ``schema``, which prints the exported JSON Schema). The GUI and this CLI
are peers over the same service and the same canonical objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .schemas import export_json_schema
from .service import LabService
from .store import KINDS, LabError, Store


def _service(args) -> LabService:
    return LabService(Store.open(getattr(args, "home", None)))


def _key_values(pairs: Optional[Sequence[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        out[key] = value
    return out


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part != ""]
    except ValueError:
        raise SystemExit(f"seeds must be comma-separated integers, got {text!r}") from None
    if not seeds:
        raise SystemExit("at least one seed is required")
    return seeds


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize {type(value)!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvslab", description="CVS Research Lab")
    parser.add_argument("--version", action="version", version=f"cvslab {__version__}")
    parser.add_argument("--home", default=None, help="lab store directory (default: $CVSLAB_HOME or ./labstore)")
    # Subcommands re-accept --home after the verb; SUPPRESS keeps an absent flag
    # from clobbering a value parsed before the verb.
    home = argparse.ArgumentParser(add_help=False)
    home.add_argument("--home", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_: str, parents=() ):
        sp = sub.add_parser(name, help=help_, parents=[home, *parents])
        return sp

    p = command("init", "set the active generation and pin the engine repository")
    p.add_argument("--generation", help="active generation, e.g. G01")
    p.add_argument("--engine-repo", help="path to the engine git checkout to pin")

    p = command("overview", "lab overview: baselines, runs, findings, compute, integrity")
    p = command("show", "print one canonical object by identity")
    p.add_argument("id")
    p = command("ls", "list canonical objects of one kind")
    p.add_argument("prefix", choices=sorted(KINDS), help="object prefix, e.g. A for ablations")

    p = command("schema", "export the canonical JSON Schema for every lab object")
    p.add_argument("--out", help="write to this file instead of stdout")

    # -- data lineage ---------------------------------------------------------
    p = command("source-fixture", "generate a deterministic labelled fixture source (S####)")
    p.add_argument("--name", default="fixture-random-play")
    p.add_argument("--n-games", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--max-plies", type=int, default=80)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--min-ply", type=int, default=6)

    p = command("source-import", "snapshot a {fen, cp, res} JSONL corpus as an S#### source")
    p.add_argument("path")
    p.add_argument("--name", required=True)
    p.add_argument("--license", default="unspecified")
    p.add_argument("--label-authority", default="imported.unspecified")
    p.add_argument("--origin", default=None)

    p = command("normalize", "standardize sources into canonical records (N####)")
    p.add_argument("sources", nargs="+")
    p.add_argument("--name", required=True)
    p.add_argument("--dedup", default="exact-epd-keep-first", choices=["exact-epd-keep-first", "none"])

    p = command("dataset", "freeze a dataset manifest with game-disjoint splits (D####)")
    p.add_argument("normalization")
    p.add_argument("--name", required=True)
    p.add_argument("--select", action="append", default=[], metavar="FIELD=VALUE",
                   help="selection filter; repeatable (phase, stm, source_id)")
    p.add_argument("--arm", action="append", default=[], metavar="JSON",
                   help='stack arm as JSON, e.g. \'{"name":"mid","filter":{"phase":"middlegame"},"policy":"all"}\'; repeatable')
    p.add_argument("--required-labels", nargs="*", default=["eval_cp"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))

    p = command("stack-preview", "preview a stack: available vs effective rows per arm; writes nothing")
    p.add_argument("normalization")
    p.add_argument("--arm", action="append", default=[], metavar="JSON", required=True)

    p = command("catalog", "filter the canonical corpus with facet counts")
    p.add_argument("normalization")
    p.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE",
                   help="catalog filter; repeatable (phase, stm, source_id, material, result, eval_bucket, "
                        "label_family, authority, tier, producer, dataset, split, ply_min, ply_max)")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=20)

    p = command("labels-append", "append an L2 label set (new immutable file) to a canonical corpus")
    p.add_argument("normalization")
    p.add_argument("--family", required=True, help="registered label family, e.g. eval_cp, search_deep_cp")
    p.add_argument("--producer", required=True)
    p.add_argument("--authority", required=True)
    p.add_argument("--rows", required=True, help="JSONL file with {record_id, value, ...} rows")
    p.add_argument("--pov", default="white")
    p.add_argument("--registry-version", type=int, default=1)

    p = command("copy-labels", "carry label sets into a new normalization, keeping surviving record identities")
    p.add_argument("from_normalization")
    p.add_argument("to_normalization")

    p = command("migrate", "re-standardize from the same immutable sources under new settings (new N####)")
    p.add_argument("from_normalization")
    p.add_argument("--name", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="settings override, repeatable")
    p.add_argument("--dedup", default="exact-epd-keep-first", choices=["exact-epd-keep-first", "none"])

    p = command("migration-diff", "diff two normalizations: records, labels, coverage, splits, affected evidence")
    p.add_argument("from_normalization")
    p.add_argument("to_normalization")

    p = command("dataset-rebuild", "rebuild a frozen dataset under a new standard as a descendant D####")
    p.add_argument("dataset")
    p.add_argument("new_normalization")

    p = command("source-pgn", "snapshot a PGN game file as an S#### source (second importer)")
    p.add_argument("path")
    p.add_argument("--name", required=True)
    p.add_argument("--license", default="unspecified")
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--min-ply", type=int, default=6)
    p.add_argument("--origin", default=None)

    # -- frozen instruments ---------------------------------------------------
    p = command("recipe", "freeze a training recipe (T####)")
    p.add_argument("--name", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="training/signal/compute switch, repeatable")
    p.add_argument("--description", default="")

    p = command("protocol", "freeze an evaluation protocol (E####) on a dataset split")
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--k", type=float, default=256.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--description", default="")

    p = command("hypothesis", "register a hypothesis (H####)")
    p.add_argument("--title", required=True)
    p.add_argument("--statement", required=True)
    p.add_argument("--metric", default="test_loss")
    p.add_argument("--direction", choices=["decrease", "increase"], default="decrease")
    p.add_argument("--min-effect", type=float, default=0.0)
    p.add_argument("--tags", nargs="*", default=[])

    # -- ablations ------------------------------------------------------------
    p = command("baseline", "register a frozen baseline ablation")
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="model-configuration switch (representation/capacity), repeatable")
    p.add_argument("--notes", default="")

    p = command("preview", "preview an ablation: diff, exact parameter count, warnings — writes nothing")
    p.add_argument("--baseline", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    p = command("ablation", "create an ablation against a baseline")
    p.add_argument("--baseline", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--notes", default="")

    # -- runs -----------------------------------------------------------------
    p = command("queue", "queue run(s) of an ablation for given seeds")
    p.add_argument("--ablation", required=True)
    p.add_argument("--seeds", type=_seeds, default=_seeds("0"))
    p.add_argument("--exec", dest="execute", action="store_true",
                   help="execute the queued runs immediately in this process")

    p = command("exec", "execute queued run(s) by identity")
    p.add_argument("runs", nargs="+")

    p = command("worker", "execute queued runs as they arrive until interrupted")
    p = command("log", "print a run's immutable log")
    p.add_argument("run")

    # -- findings and views ---------------------------------------------------
    p = command("finding", "draft a finding card from two ablations' completed runs")
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--intervention", required=True)
    p.add_argument("--interpretation", default=None)
    p.add_argument("--next", default=None, dest="next_experiment")

    p = command("findings", "list finding cards; negative and inconclusive results stay visible")
    p.add_argument("--query", default=None)
    p.add_argument("--result", default=None)

    p = command("matrix", "switch matrix: ablations x registered switches")
    p.add_argument("--family", default="NNUE")
    p.add_argument("--generation", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--state", default=None)
    p.add_argument("--metric", default="test_loss")
    p.add_argument("--min-params", type=int, default=None)
    p.add_argument("--max-params", type=int, default=None)

    p = command("scaling", "capability vs learned parameters and compute")
    p.add_argument("--metric", default="test_loss")
    p.add_argument("--family", default="NNUE")

    p = command("backlog", "engine search switches as clean-lineage re-test backlog")

    p = command("facts-label", "compute CVS analysis facts (geometry, motifs, strategy) per record")
    p.add_argument("normalization")

    p = command("map", "Lab Map: canonical objects and provenance edges for a projection")
    p.add_argument("--projection", default="data", choices=["data", "research", "promotion"])
    p.add_argument("--generation", default=None)
    p.add_argument("--state", default=None)
    p.add_argument("--family", default=None)

    p = command("pool", "candidate-pool generation (S5): fresh legacy-CVS self-play, raw source")
    pool_sub = p.add_subparsers(dest="pool_command", required=True)
    g = pool_sub.add_parser("generate", parents=[home])
    g.add_argument("--engine-root", required=True, help="legacy engine checkout (frozen teacher)")
    g.add_argument("--out", required=True, help="output directory for the raw pool")
    g.add_argument("--games", type=int, default=10)
    g.add_argument("--seed", type=int, default=20260915)
    g.add_argument("--max-plies", type=int, default=80)
    g.add_argument("--sample-every", type=int, default=2)
    g.add_argument("--min-ply", type=int, default=6)
    g.add_argument("--diversify-plies", type=int, nargs="*", default=[6, 8, 10])
    g.add_argument("--window-cp", type=int, default=25)
    g.add_argument("--play-nodes", type=int, default=20000)
    g.add_argument("--snapshot", action="store_true", help="also snapshot the raw pool as an S#### source")

    # -- intake ---------------------------------------------------------------
    p = command("intake-legacy", "import the legacy catalog as LEGACY evidence (I####)")
    p.add_argument("--catalog", default="catalog")
    p.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                   help="override a catalog root location, repeatable")
    p.add_argument("--hash-sources", action="store_true", help="hash every source byte instead of checking sizes")

    p = command("intake-funnel", "import a labeling funnel run directory as LEGACY evidence")
    p.add_argument("run_dir")
    p.add_argument("--name", default=None)
    p.add_argument("--research-state", default=None)
    p.add_argument("--state-source", default=None)

    # -- first end-to-end proof -------------------------------------------------
    p = command("demo", "run the Phase 0 end-to-end proof: one tiny experiment through every layer")
    p.add_argument("--n-games", type=int, default=300)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=_seeds, default=_seeds("0,1,2"))
    p.add_argument("--control-width", type=int, default=1)
    p.add_argument("--intervention-width", type=int, default=16)

    # -- serving ----------------------------------------------------------------
    p = command("serve", "serve the local HTTP API (and the built GUI from web/dist)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-worker", action="store_true", help="do not execute queued runs in the background")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except LabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _dispatch(args) -> int:
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(getattr(args, "home", None), worker=not args.no_worker),
                    host=args.host, port=args.port)
        return 0

    service = _service(args)
    cmd = args.command

    if cmd == "init":
        _emit(service.init_lab(generation=args.generation, engine_repo=args.engine_repo))
    elif cmd == "overview":
        _emit(service.overview())
    elif cmd == "show":
        _emit(service.store.get(args.id))
    elif cmd == "ls":
        _emit(service.store.list(args.prefix, verify=False))
    elif cmd == "schema":
        document = json.dumps(export_json_schema(), indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(document + "\n")
            print(f"wrote {args.out}")
        else:
            print(document)

    elif cmd == "source-fixture":
        _emit(service.create_fixture_source(name=args.name, n_games=args.n_games, seed=args.seed,
                                            max_plies=args.max_plies, sample_every=args.sample_every,
                                            min_ply=args.min_ply))
    elif cmd == "source-import":
        _emit(service.import_jsonl_source(args.path, name=args.name, license=args.license,
                                          label_authority=args.label_authority, origin=args.origin))
    elif cmd == "normalize":
        _emit(service.normalize(args.sources, name=args.name, dedup=args.dedup))
    elif cmd == "dataset":
        arms = [json.loads(arm) for arm in args.arm]
        _emit(service.freeze_dataset(args.normalization, name=args.name,
                                     selection=_key_values(args.select) or None, arms=arms or None,
                                     required_labels=args.required_labels, split_seed=args.split_seed,
                                     fractions=tuple(args.fractions)))
    elif cmd == "stack-preview":
        _emit(service.stack_preview(args.normalization, [json.loads(arm) for arm in args.arm]))
    elif cmd == "catalog":
        _emit(service.catalog(args.normalization, filters=_key_values(args.filter),
                              offset=args.offset, limit=args.limit))
    elif cmd == "labels-append":
        from .hashing import read_jsonl
        _emit(service.register_labels(args.normalization, family=args.family, producer=args.producer,
                                      authority=args.authority, rows=read_jsonl(args.rows), pov=args.pov,
                                      registry_version=args.registry_version))
    elif cmd == "copy-labels":
        _emit(service.copy_label_sets(args.from_normalization, args.to_normalization))
    elif cmd == "migrate":
        _emit(service.migrate(args.from_normalization, name=args.name,
                              settings_overrides=_key_values(args.set) or None, dedup=args.dedup))
    elif cmd == "migration-diff":
        _emit(service.migration_diff(args.from_normalization, args.to_normalization))
    elif cmd == "dataset-rebuild":
        _emit(service.rebuild_dataset(args.dataset, args.new_normalization))
    elif cmd == "source-pgn":
        _emit(service.import_pgn_source(args.path, name=args.name, license=args.license,
                                        sample_every=args.sample_every, min_ply=args.min_ply,
                                        origin=args.origin))

    elif cmd == "recipe":
        _emit(service.create_training_recipe(name=args.name, params=_key_values(args.set),
                                              description=args.description))
    elif cmd == "protocol":
        _emit(service.create_eval_protocol(name=args.name, dataset_id=args.dataset, split=args.split,
                                           bootstrap_samples=args.bootstrap_samples,
                                           bootstrap_seed=args.bootstrap_seed, k=args.k, lam=args.lam,
                                           description=args.description))
    elif cmd == "hypothesis":
        _emit(service.create_hypothesis(title=args.title, statement=args.statement, metric=args.metric,
                                        predicted_direction=args.direction, min_effect=args.min_effect,
                                        tags=args.tags))

    elif cmd == "baseline":
        _emit(service.register_baseline(name=args.name, dataset_id=args.dataset,
                                        training_recipe_id=args.recipe, eval_protocol_id=args.protocol,
                                        model_config=_key_values(args.set), notes=args.notes))
    elif cmd == "preview":
        _emit(service.preview_ablation(baseline_id=args.baseline, overrides=_key_values(args.set)))
    elif cmd == "ablation":
        _emit(service.create_ablation(baseline_id=args.baseline, overrides=_key_values(args.set),
                                      hypothesis_id=args.hypothesis, notes=args.notes))

    elif cmd == "queue":
        runs = service.queue_runs(args.ablation, args.seeds)
        if args.execute:
            runs = [service.execute_run(r.id) for r in runs]
        _emit(runs)
    elif cmd == "exec":
        _emit([service.execute_run(run_id) for run_id in args.runs])
    elif cmd == "worker":
        print("executing queued runs; Ctrl-C to stop", file=sys.stderr)
        while True:
            service.work_queue()
    elif cmd == "log":
        sys.stdout.write(service.run_log(args.run))

    elif cmd == "finding":
        _emit(service.draft_finding(hypothesis_id=args.hypothesis, control_ablation_id=args.control,
                                    intervention_ablation_id=args.intervention,
                                    interpretation=args.interpretation, next_experiment=args.next_experiment))
    elif cmd == "findings":
        _emit(service.list_findings(query=args.query, result=args.result))
    elif cmd == "matrix":
        _emit(service.matrix(family=args.family, generation=args.generation, baseline=args.baseline,
                             dataset_id=args.dataset, state=args.state, metric=args.metric,
                             min_params=args.min_params, max_params=args.max_params))
    elif cmd == "scaling":
        _emit(service.scaling(metric=args.metric, family=args.family))
    elif cmd == "backlog":
        _emit(service.search_backlog())

    elif cmd == "facts-label":
        _emit(service.label_facts(args.normalization))
    elif cmd == "map":
        _emit(service.map_view(args.projection, generation=args.generation,
                               state=args.state, family=args.family))
    elif cmd == "pool":
        from .funnel.pool import PoolConfig, generate_and_write
        result = generate_and_write(
            args.engine_root, args.out,
            config=PoolConfig(seed=args.seed, games=args.games, max_plies=args.max_plies,
                              sample_every=args.sample_every, min_ply=args.min_ply,
                              diversification_plies=tuple(args.diversify_plies),
                              diversification_window_cp=args.window_cp,
                              play_node_budget=args.play_nodes),
            store=service.store if args.snapshot else None)
        _emit({"report": result["report"], "source": result["source"],
               "configSha256": result["manifest"]["configSha256"],
               "manifestSha256": result["manifest"]["manifestSha256"]})
    elif cmd == "intake-legacy":
        roots = {}
        for pair in args.root:
            name, _, path = pair.partition("=")
            roots[name] = path
        _emit(service.import_legacy_catalog(args.catalog, roots=roots, hash_sources=args.hash_sources))
    elif cmd == "intake-funnel":
        _emit(service.import_funnel_run(args.run_dir, name=args.name, research_state=args.research_state,
                                        state_source=args.state_source))

    elif cmd == "demo":
        _emit(service.run_phase0_proof(n_games=args.n_games, epochs=args.epochs, seeds=args.seeds,
                                       control_width=args.control_width,
                                       intervention_width=args.intervention_width))
    else:  # pragma: no cover - argparse rejects unknown commands
        raise LabError(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
