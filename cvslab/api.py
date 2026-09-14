"""Local HTTP API: a thin shell over :mod:`cvslab.service`.

The GUI talks only to this API; it contains no scientific logic of its own.
Every mutating route names the canonical object it creates, and every view is
recomputed from the store. Run with ``cvslab serve`` or
``uvicorn cvslab.api:app``; ``CVSLAB_HOME`` (or ``create_app(home=...)``)
selects the store.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import data, families
from .schemas import (
    AblationRequest,
    Dataset,
    Finding,
    FindingRequest,
    HypothesisCreate,
    LabelsAppendRequest,
    MigrateRequest,
    Normalization,
    RebuildDatasetRequest,
    Run,
    RunQueueRequest,
    RunStatus,
    SourceSnapshot,
    StackFreezeRequest,
)
from .service import LAB_REPO, DEFAULT_FAMILY, LabService
from .store import KINDS, LabError, Store


class BaselineCreate(BaseModel):
    name: str
    dataset_id: str
    training_recipe_id: str
    eval_protocol_id: str
    overrides: dict[str, object] = {}
    family: str = DEFAULT_FAMILY
    notes: str = ""


class LabConfigUpdate(BaseModel):
    generation: Optional[str] = None
    engine_repo: Optional[str] = None


def create_app(home: Optional[str | Path] = None, *, worker: bool = True) -> FastAPI:
    service = LabService(Store.open(home))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker:
            service.start_worker()
        yield
        if worker:
            service.stop_worker()

    app = FastAPI(title="CVS Research Lab", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.state.service = service

    @app.exception_handler(LabError)
    async def lab_error_handler(_request: Request, exc: LabError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    # -- lab configuration ---------------------------------------------------

    @app.get("/api/config")
    def get_config():
        return service.config()

    @app.post("/api/config")
    def update_config(body: LabConfigUpdate):
        return service.init_lab(generation=body.generation, engine_repo=body.engine_repo)

    # -- views ---------------------------------------------------------------

    @app.get("/api/overview")
    def overview():
        return service.overview()

    @app.get("/api/switches")
    def registry(family: str = DEFAULT_FAMILY):
        return families.switches(family)

    @app.get("/api/matrix")
    def matrix(family: str = DEFAULT_FAMILY, generation: Optional[str] = None, baseline: Optional[str] = None,
               dataset_id: Optional[str] = None, min_params: Optional[int] = None, max_params: Optional[int] = None,
               state: Optional[str] = None, metric: str = "test_loss"):
        return service.matrix(family=family, generation=generation, baseline=baseline, dataset_id=dataset_id,
                              min_params=min_params, max_params=max_params, state=state, metric=metric)

    @app.get("/api/scaling")
    def scaling(metric: str = "test_loss", family: str = DEFAULT_FAMILY):
        return service.scaling(metric=metric, family=family)

    @app.get("/api/backlog")
    def backlog():
        return service.search_backlog()

    # -- canonical objects ---------------------------------------------------

    @app.get("/api/objects/{prefix}")
    def list_objects(prefix: str):
        if prefix not in KINDS:
            raise LabError(f"unknown object kind {prefix!r}; kinds: {', '.join(KINDS)}")
        return service.store.list(prefix, verify=False)

    @app.get("/api/object/{obj_id}")
    def get_object(obj_id: str):
        return service.store.get(obj_id)

    # -- hypotheses ----------------------------------------------------------

    @app.get("/api/hypotheses")
    def hypotheses():
        return service.store.list("H")

    @app.post("/api/hypotheses", status_code=201)
    def create_hypothesis(body: HypothesisCreate):
        return service.create_hypothesis(**body.model_dump())

    # -- ablations -----------------------------------------------------------

    @app.get("/api/baselines")
    def baselines():
        return [a for a in service.store.list("A") if a.is_baseline]

    @app.post("/api/baselines", status_code=201)
    def register_baseline(body: BaselineCreate):
        return service.register_baseline(name=body.name, dataset_id=body.dataset_id,
                                         training_recipe_id=body.training_recipe_id,
                                         eval_protocol_id=body.eval_protocol_id,
                                         model_config=body.overrides, family=body.family, notes=body.notes)

    @app.get("/api/ablations")
    def ablations():
        return service.store.list("A")

    @app.post("/api/ablations/preview")
    def preview_ablation(body: AblationRequest):
        return service.preview_ablation(baseline_id=body.baseline_id, overrides=body.overrides)

    @app.post("/api/ablations", status_code=201)
    def create_ablation(body: AblationRequest):
        return service.create_ablation(baseline_id=body.baseline_id, overrides=body.overrides,
                                       hypothesis_id=body.hypothesis_id, notes=body.notes)

    # -- runs ----------------------------------------------------------------

    @app.get("/api/runs")
    def runs(status: Optional[str] = Query(None)):
        wanted = status.upper() if status else None
        if wanted and wanted not in RunStatus.__members__:
            raise LabError(f"unknown run status {status!r}; one of {', '.join(RunStatus.__members__)}")
        return [r for r in service.store.list("R", verify=False, kind=Run)
                if wanted is None or r.status.value == wanted]

    @app.post("/api/runs", status_code=201)
    def queue_runs(body: RunQueueRequest):
        return service.queue_runs(body.ablation_id, body.seeds)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return service.store.get_as(run_id, Run)

    @app.post("/api/runs/{run_id}/execute")
    def execute_run(run_id: str):
        return service.execute_run(run_id)

    @app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(run_id: str):
        return service.run_log(run_id)

    # -- findings ------------------------------------------------------------

    @app.get("/api/findings")
    def findings(query: Optional[str] = None, result: Optional[str] = None):
        return service.list_findings(query=query, result=result)

    @app.get("/api/findings/{finding_id}")
    def get_finding(finding_id: str):
        return service.store.get_as(finding_id, Finding)

    @app.post("/api/findings", status_code=201)
    def draft_finding(body: FindingRequest):
        return service.draft_finding(**body.model_dump())

    # -- data lineage, catalog, stacking and migration -------------------------

    @app.get("/api/datasets")
    def datasets():
        return service.store.list("D", verify=False, kind=Dataset)

    @app.post("/api/sources/pgn", status_code=201)
    def import_pgn(body: dict):
        return service.import_pgn_source(body["path"], name=body["name"],
                                         license=body.get("license", "unspecified"),
                                         sample_every=int(body.get("sample_every", 2)),
                                         min_ply=int(body.get("min_ply", 6)),
                                         origin=body.get("origin"))

    @app.post("/api/normalizations", status_code=201)
    def migrate(body: MigrateRequest):
        return service.migrate(body.from_normalization_id, name=body.name,
                               settings_overrides=body.settings_overrides, dedup=body.dedup)

    @app.get("/api/normalizations/{normalization_id}/labels")
    def label_sets(normalization_id: str):
        service.store.get_as(normalization_id, Normalization)
        return data.label_set_refs(service.store, normalization_id)

    @app.post("/api/labels", status_code=201)
    def append_labels(body: LabelsAppendRequest):
        return service.register_labels(body.normalization_id, family=body.family, producer=body.producer,
                                       authority=body.authority, rows=body.rows, pov=body.pov,
                                       registry_version=body.registry_version)

    @app.post("/api/normalizations/{from_id}/copy-labels/{to_id}")
    def copy_labels(from_id: str, to_id: str):
        return service.copy_label_sets(from_id, to_id)

    @app.get("/api/catalog")
    def catalog(request: Request, normalization_id: str, offset: int = 0, limit: int = 50):
        reserved = ("normalization_id", "offset", "limit")
        filters = {key: value for key, value in request.query_params.items() if key not in reserved}
        return service.catalog(normalization_id, filters=filters, offset=offset, limit=limit)

    @app.post("/api/stacks/preview")
    def stack_preview(body: dict):
        return service.stack_preview(body["normalization_id"], body.get("arms", []))

    @app.post("/api/datasets/freeze", status_code=201)
    def freeze_stack(body: StackFreezeRequest):
        return service.freeze_dataset(body.normalization_id, name=body.name, arms=body.arms,
                                      split_seed=body.split_seed, fractions=tuple(body.fractions),
                                      required_labels=body.required_labels)

    @app.get("/api/migration-diff")
    def migration_diff(from_id: str, to_id: str):
        return service.migration_diff(from_id, to_id)

    @app.post("/api/datasets/{dataset_id}/rebuild", status_code=201)
    def rebuild(dataset_id: str, body: RebuildDatasetRequest):
        return service.rebuild_dataset(dataset_id, body.new_normalization_id)

    dist = LAB_REPO / "web" / "dist"
    if dist.is_dir():
        from fastapi.responses import FileResponse

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            # Client-side routes (e.g. /runs/R0001) fall back to index.html;
            # unknown API paths stay 404 instead of returning the app shell.
            if full_path.startswith("api/") or full_path == "api":
                raise LabError(f"no such API route: /{full_path}")
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
