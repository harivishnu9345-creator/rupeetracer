"""RiskEngine: the single pipeline used by the API and the dashboard.

txn -> StreamScorer (mock or real) -> fusion -> watchlist boost -> orchestrator -> case manager
Ground-truth columns (label, typology, scenario_id, root_txn_id_gt, role_gt) are NEVER passed to the scorer.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine import orchestrator
from engine.cases import WATCHLIST_BOOST, CaseManager
from engine.fusion import COMPONENTS, Fusion
from engine.mock_provider import build_mock, generate_dataset

log = logging.getLogger("rupeetrace")
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "state"
GT_COLS = ["label", "typology", "scenario_id", "root_txn_id_gt"]
TXN_FIELDS = ["txn_id", "ts", "src_account", "dst_account", "amount", "channel", "origin_type", "device_id", "ip", "city", "remark"]
SPLIT = pd.Timestamp(config.SPLIT_TS)
SPLIT = SPLIT.tz_localize("UTC") if SPLIT.tzinfo is None else SPLIT.tz_convert("UTC")
REASON_TEXT = {
    "behavior": "Unusual payment behaviour for this account",
    "anomaly": "Pattern differs from this account's normal activity",
    "graph": "Connected accounts share devices / payment links",
    "provenance": "Money lineage shows the funds came from a larger, split-up source",
}


def pseudonym(account_id: str) -> str:
    h = hashlib.sha1(("rt-salt:" + str(account_id)).encode()).hexdigest()[:6].upper()
    return ("EXT-" if str(account_id).startswith("EXT_") else "ACC-") + h


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Contract CSVs from acc_1 if present; otherwise deterministic synthetic data."""
    a, t = ROOT / "data" / "out" / "accounts.csv", ROOT / "data" / "out" / "transactions.csv"
    if a.exists() and t.exists():
        acc, tx, src = pd.read_csv(a), pd.read_csv(t), "data/out (acc_1)"
    else:
        acc, tx = generate_dataset(config.SEED)
        src = "synthetic mock data (data/out not found)"
    tx = tx.copy()
    tx["_ts"] = pd.to_datetime(tx["ts"], format="ISO8601", utc=True)
    tx = tx.sort_values("_ts", kind="stable").reset_index(drop=True)
    for c in GT_COLS:
        if c not in tx:
            tx[c] = "" if c != "label" else 0
    tx["label"] = tx["label"].fillna(0).astype(int)
    return acc, tx, src


def build_backend(accounts: pd.DataFrame, use_mock: bool | None = None):
    """Single switch config.USE_MOCK: mock vs acc_3's real StreamScorer / LotLedger."""
    use_mock = getattr(config, "USE_MOCK", True) if use_mock is None else use_mock
    if not use_mock:
        try:
            from ledger.lots import LotLedger
            from models.stream_scorer import StreamScorer
            scorer = StreamScorer(str(getattr(config, "MODELS_DIR", ROOT / "models")))
            ledger = LotLedger()  # ASSUMPTION: no-arg constructor; contract only specifies get_lineage()
            return scorer, ledger, "real"
        except Exception as e:  # degrade, never crash the demo
            log.warning("Real backend unavailable (%s); falling back to mock", e)
            s, l = build_mock(accounts)
            return s, l, f"mock (fallback: {type(e).__name__})"
    s, l = build_mock(accounts)
    return s, l, "mock"


class RiskEngine:
    def __init__(self, use_mock: bool | None = None, state_dir: str | Path | None = None, persist: bool = True, warm: bool = True):
        self._lock = threading.RLock()
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.accounts, self.txns, self.data_source = load_data()
        self.acc = self.accounts.set_index("account_id").to_dict("index")
        self.scorer, self.ledger, self.backend = build_backend(self.accounts, use_mock)
        self.fusion = Fusion(self.state_dir / "fusion_weights.json")
        self.cases = CaseManager(self.state_dir / "cases.json" if persist else None, persist=persist)
        self.decisions: list[dict] = []
        self.train_frame = pd.DataFrame(columns=COMPONENTS + ["label"])
        self._lineage_cache: dict[str, dict] = {}
        self.train_df = self.txns[self.txns["_ts"] < SPLIT]
        self.test_df = self.txns[self.txns["_ts"] >= SPLIT]
        self.cursor = 0  # next test txn index for streaming
        if warm:
            self.warm_up()

    # -------------------------------------------------------------- training
    def warm_up(self) -> dict:
        """Stream the training split through the scorer (state warm-up, chronological) and fit the stacker."""
        with self._lock:
            rows = []
            for r in self.train_df.to_dict("records"):
                o = self.scorer.process({k: r[k] for k in TXN_FIELDS})
                rows.append([o[f"{c}_score"] for c in COMPONENTS] + [int(r["label"])])
            self.train_frame = pd.DataFrame(rows, columns=COMPONENTS + ["label"])
            meta = self.fusion.fit(self.train_frame[COMPONENTS].values, self.train_frame["label"].values)
            self.fusion.save()
            self.cursor = 0
            return meta

    # --------------------------------------------------------------- scoring
    def segment_of(self, txn: dict) -> tuple[str | None, int | None]:
        a = self.acc.get(txn["src_account"]) or self.acc.get(txn["dst_account"]) or {}
        return a.get("account_type"), a.get("kyc_tier")

    def score_txn(self, txn: dict, label_gt: int | None = None, keep: bool = True) -> dict:
        t0 = time.perf_counter()
        with self._lock:
            txn = {k: txn.get(k) for k in TXN_FIELDS if k in txn}
            out = self.scorer.process(txn)
            raw = {c: float(out[f"{c}_score"]) for c in COMPONENTS}
            fused = self.fusion.score(raw, self.cases.scale())
            risk = fused["risk_score"]
            hit = self.cases.watchlist_hit(txn["src_account"], txn["dst_account"])
            if hit:
                risk = min(1.0, risk + WATCHLIST_BOOST)
            atype, kyc = self.segment_of(txn)
            contrib = fused["contributions"]
            top = max(contrib, key=contrib.get)
            reasons = list(out.get("reasons", []))
            if risk >= 0.30:
                reasons.append(f"Largest contributor: {REASON_TEXT[top].lower()} ({contrib[top] / max(risk, 1e-9):.0%} of the score)")
            dec = orchestrator.decide(risk, float(out.get("cluster_risk", 0.0)), atype, kyc, reasons=reasons[:8],
                                      learned_offsets=self.cases.offsets(), watchlist_hit=hit)
            dec.update(account_type=atype, kyc_tier=kyc)
            dec["components"] = {c: {"score": round(raw[c], 4), "contribution": contrib[c]} for c in COMPONENTS}
            case_id = None
            if dec["opens_case"]:
                lin = self._lineage_cache.get(txn["txn_id"]) or self.ledger.get_lineage(txn["txn_id"])
                self._lineage_cache[txn["txn_id"]] = lin
                case_id = self.cases.record(txn, dec, lin, out.get("features"), top, contrib, raw)
            dec.update(txn_id=txn["txn_id"], ts=txn["ts"], src_account=txn["src_account"], dst_account=txn["dst_account"],
                       amount=float(txn["amount"]), channel=txn.get("channel"), case_id=case_id, watchlist_hit=hit,
                       fusion_mode=fused["mode"], backend=self.backend, scorer_latency_ms=round(float(out.get("latency_ms", 0.0)), 3),
                       features=out.get("features", {}), cluster_risk_raw=float(out.get("cluster_risk", 0.0)))
            if label_gt is not None:
                dec["label_gt"] = int(label_gt)
            dec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if keep:
                self.decisions.append(dec)
            return dec

    def next_batch(self, n: int = 1) -> list[dict]:
        """Stream the next n test-split transactions through the engine (used by the live monitor)."""
        out = []
        with self._lock:
            for r in self.test_df.iloc[self.cursor:self.cursor + n].to_dict("records"):
                d = self.score_txn({k: r[k] for k in TXN_FIELDS}, label_gt=r["label"])
                out.append(d)
            self.cursor += n
        return out

    def reset_stream(self) -> None:
        """Fresh scorer state, retrain, clear cases/decisions: used to restart a demo run."""
        with self._lock:
            self.scorer, self.ledger, self.backend = build_backend(self.accounts)
            self.decisions.clear(), self._lineage_cache.clear()
            self.cases.reset()
            self.warm_up()

    # ----------------------------------------------------------- lineage / graph
    def lineage(self, txn_id: str) -> dict:
        with self._lock:
            if txn_id not in self._lineage_cache:
                self._lineage_cache[txn_id] = self.ledger.get_lineage(txn_id)
            return self._lineage_cache[txn_id]

    def account_details(self, account_id: str) -> dict:
        a = self.acc.get(account_id)
        if not a:
            return {"account_id": account_id, "type": "external pseudo-account"}
        return {"account_id": account_id, "account_type": a["account_type"], "kyc_tier": a["kyc_tier"], "city": a["city"],
                "device": a["primary_device_id"], "ip_prefix": a["primary_ip_prefix"], "created_at": a["created_at"]}

    def case_graph(self, case_id: str, max_nodes: int = 60) -> dict:
        """Bounded network of the case's accounts: payment edges + shared-device links (highlighted)."""
        c = self.cases.get(case_id)
        if not c:
            return {"nodes": [], "edges": [], "truncated": False}
        deg: dict[str, int] = {}
        edges: dict[tuple, dict] = {}
        for t in c["txns"]:
            k = (t["src"], t["dst"])
            e = edges.setdefault(k, {"src": t["src"], "dst": t["dst"], "amount": 0.0, "count": 0, "kind": "payment"})
            e["amount"] += t["amount"]
            e["count"] += 1
            deg[t["src"]] = deg.get(t["src"], 0) + 1
            deg[t["dst"]] = deg.get(t["dst"], 0) + 1
        keep = set(sorted(deg, key=deg.get, reverse=True)[:max_nodes])
        by_dev: dict[str, list] = {}
        for a in keep:
            dv = (self.acc.get(a) or {}).get("primary_device_id")
            if dv:
                by_dev.setdefault(dv, []).append(a)
        e_out = [e for e in edges.values() if e["src"] in keep and e["dst"] in keep]
        for dv, members in by_dev.items():
            if len(members) > 1:
                for i in range(len(members) - 1):  # chain keeps edge count linear in members
                    e_out.append({"src": members[i], "dst": members[i + 1], "amount": 0.0, "count": 0, "kind": "shared_device", "device": dv})
        nodes = [{"id": a, "pseudonym": pseudonym(a), "degree": deg[a], "device": (self.acc.get(a) or {}).get("primary_device_id"),
                  "on_watchlist": a in self.cases.watchlist()} for a in keep]
        return {"nodes": nodes, "edges": e_out, "truncated": len(deg) > len(keep), "total_accounts": len(deg)}

    # -------------------------------------------------------------- feedback
    def feedback(self, case_id: str, label: str, note: str = "", analyst: str = "analyst") -> dict:
        with self._lock:
            res = self.cases.feedback(case_id, label, note, analyst)
            peak = self.cases.get(case_id)["peak"]
            if res.get("changed") and peak:
                b, a = res["before"], res["after"]
                scale_after, off_after = self.cases.scale(), self.cases.offsets()
                scale_before, off_before = dict(scale_after), dict(off_after)
                scale_before[b["driver"]] = b["component_scale"]
                off_before[b["segment"]] = b["segment_offset"]
                at, kyc = peak.get("account_type"), peak.get("kyc_tier")
                def rescore(scale, offs, wl):
                    r = self.fusion.score(peak["components"], scale)["risk_score"]
                    d = orchestrator.decide(min(1.0, r + (WATCHLIST_BOOST if wl else 0)), peak["cluster_risk"], at, kyc, learned_offsets=offs)
                    return {"risk_score": d["risk_score"], "action": d["action"]}
                res["rescore_peak_txn"] = {"txn_id": peak["txn_id"], "before": rescore(scale_before, off_before, False),
                                           "after": rescore(scale_after, off_after, False)}
            return res

    def fusion_info(self) -> dict:
        return {**self.fusion.describe(), "component_scale": self.cases.scale(), "segment_offsets": self.cases.offsets(),
                "watchlist_size": len(self.cases.watchlist())}
