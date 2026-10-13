"""RupeeTrace API.   uvicorn engine.api:app --port 8000

POST /score  | GET /cases | GET /cases/{id} | POST /cases/{id}/feedback | GET /lineage/{txn_id}
GET /metrics | GET /health | GET /fusion
Accounts are pseudonymous in every response unless ?analyst=true is passed (analyst view).
State is guarded by the engine's RLock; handlers are short and non-blocking (mock scoring < 1 ms).
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

import config
from engine.engine import RiskEngine, pseudonym
from engine.metrics import compute_metrics


class TxnIn(BaseModel):
    txn_id: str | None = None
    ts: str = Field(..., examples=["2026-01-25T02:30:00+00:00"])
    src_account: str
    dst_account: str
    amount: float = Field(..., gt=0)
    channel: str = "UPI"
    origin_type: str = "p2p"
    device_id: str | None = None
    ip: str | None = None
    city: str | None = None
    remark: str | None = None


class ComponentOut(BaseModel):
    score: float
    contribution: float


class DecisionOut(BaseModel):
    txn_id: str
    action: Literal["ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"]
    risk_score: float = Field(..., ge=0, le=1)
    base_risk_score: float
    cluster_risk: float
    reasons: list[str]
    components: dict[str, ComponentOut]
    recommended_intervention: str
    sla_minutes: int
    segment: str
    escalated: bool
    case_id: str | None = None
    latency_ms: float
    backend: str


class FeedbackIn(BaseModel):
    label: Literal["confirmed", "false_positive"]
    note: str = ""
    analyst: str = "analyst"


class Health(BaseModel):
    status: str
    backend: str
    use_mock: bool
    data_source: str
    fusion_mode: str
    cases: int
    scored: int


def _mask_case(c: dict, analyst: bool) -> dict:
    if analyst:
        return c
    c = dict(c)
    c["accounts"] = [pseudonym(a) for a in c.get("accounts", [])]
    c["txns"] = [{**t, "src": pseudonym(t["src"]), "dst": pseudonym(t["dst"])} for t in c.get("txns", [])]
    return c


def create_app(engine: RiskEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or RiskEngine()
        app.state.metrics_cache = (-1, None)
        yield
        app.state.engine.cases.flush()

    app = FastAPI(title="RupeeTrace Risk Engine", version="1.0", lifespan=lifespan)

    def eng(request: Request) -> RiskEngine:
        return request.app.state.engine

    @app.get("/health", response_model=Health)
    async def health(request: Request):
        e = eng(request)
        return Health(status="ok", backend=e.backend, use_mock=bool(getattr(config, "USE_MOCK", True)), data_source=e.data_source,
                      fusion_mode=e.fusion.mode, cases=len(e.cases.cases), scored=len(e.decisions))

    @app.post("/score", response_model=DecisionOut)
    async def score(txn: TxnIn, request: Request):
        e = eng(request)
        t = txn.model_dump()
        t["txn_id"] = t["txn_id"] or f"API-{uuid.uuid4().hex[:10]}"
        acc = e.acc.get(t["src_account"], {})
        t["device_id"] = t["device_id"] or acc.get("primary_device_id", "UNKNOWN")
        t["ip"] = t["ip"] or f"{acc.get('primary_ip_prefix', '0.0.0')}.1"
        t["city"] = t["city"] or acc.get("city", "")
        t["remark"] = t["remark"] or ""
        t0 = time.perf_counter()
        d = e.score_txn(t)
        d["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return d

    @app.get("/cases")
    async def list_cases(request: Request, status: Literal["open", "confirmed", "false_positive"] | None = None):
        return eng(request).cases.list(status)

    @app.get("/cases/{case_id}")
    async def get_case(case_id: str, request: Request, analyst: bool = Query(False)):
        e = eng(request)
        c = e.cases.get(case_id)
        if not c:
            raise HTTPException(404, "case not found")
        c = _mask_case(c, analyst)
        c["graph"] = e.case_graph(case_id)
        if not analyst:
            for n in c["graph"]["nodes"]:
                n["id"], n["device"] = n["pseudonym"], None
            for ed in c["graph"]["edges"]:
                ed["src"], ed["dst"] = pseudonym(ed["src"]), pseudonym(ed["dst"])
        return c

    @app.post("/cases/{case_id}/feedback")
    async def feedback(case_id: str, body: FeedbackIn, request: Request):
        e = eng(request)
        if case_id not in e.cases.cases:
            raise HTTPException(404, "case not found")
        return e.feedback(case_id, body.label, body.note, body.analyst)

    @app.get("/lineage/{txn_id}")
    async def lineage(txn_id: str, request: Request, analyst: bool = Query(False)):
        e = eng(request)
        lin = e.lineage(txn_id)
        if not lin.get("nodes"):
            raise HTTPException(404, "no lineage for this transaction")
        if not analyst:
            lin = {**lin, "account_path": [pseudonym(a) for a in lin["account_path"]],
                   "nodes": [{**n, "account": pseudonym(n["account"])} for n in lin["nodes"]]}
        return lin

    @app.get("/metrics")
    async def metrics(request: Request):
        e = eng(request)
        n, cached = request.app.state.metrics_cache
        if n != len(e.decisions) or cached is None:
            cached = compute_metrics(e)
            request.app.state.metrics_cache = (len(e.decisions), cached)
        return cached

    @app.get("/fusion")
    async def fusion(request: Request):
        return eng(request).fusion_info()

    return app


app = create_app()
