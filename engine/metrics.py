"""Metrics for API /metrics and the dashboard Metrics page.

Reads acc_1's outputs from eval/out when present (predictions_{A_rules|B_behavior_anomaly|C_plus_graph|D_plus_provenance}.csv,
ablation table CSV, chart files) and degrades gracefully when missing. When predictions are absent,
an INTERNAL ablation is computed from the engine's own component scores (clearly labelled as such):
  A rules (amount >= ALERT_THRESHOLD) | B behaviour+anomaly | C +graph | D +provenance (all four, fused)
B, C, D each use their own stacker trained on the training split only; actions come from config.ACTION_BANDS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine import orchestrator
from engine.engine import SPLIT
from engine.fusion import COMPONENTS, Fusion

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "out"
VARIANTS = {"A_rules": "A. Rules (threshold)", "B_behavior_anomaly": "B. Behaviour + anomaly", "C_plus_graph": "C. + Graph",
            "D_plus_provenance": "D. + Provenance (full)"}
VARIANT_COMPONENTS = {"B_behavior_anomaly": ["behavior", "anomaly"], "C_plus_graph": ["behavior", "anomaly", "graph"], "D_plus_provenance": COMPONENTS}


def bucket_label(amount: float) -> str:
    for lo, hi in config.AMOUNT_BUCKETS:
        if lo <= amount < hi or (hi == float("inf") and amount >= lo):
            return f"< Rs {hi:,.0f}" if lo == 0 else (f"Rs {lo:,.0f} - {hi:,.0f}" if hi != float("inf") else f">= Rs {lo:,.0f}")
    return "?"


def _summ(df: pd.DataFrame, action_col: str = "action") -> dict:
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    flag = lambda x: (x[action_col] != "ALLOW")
    hard = lambda x: x[action_col].isin(["HOLD", "BLOCK", "ESCALATE"])
    tp = int(flag(pos).sum())
    fp = int(flag(neg).sum())
    by_bucket = []
    for lo, hi in config.AMOUNT_BUCKETS:
        p = pos[(pos["amount"] >= lo) & (pos["amount"] < hi)]
        by_bucket.append({"bucket": bucket_label(lo), "n": int(len(p)), "recall": float(flag(p).mean()) if len(p) else None})
    return {"recall": float(flag(pos).mean()) if len(pos) else None, "recall_hold_plus": float(hard(pos).mean()) if len(pos) else None,
            "precision": tp / (tp + fp) if tp + fp else None, "friction": float(flag(neg).mean()) if len(neg) else None,
            "friction_hard": float(hard(neg).mean()) if len(neg) else None, "n": int(len(df)), "n_pos": int(len(pos)),
            "recall_by_bucket": by_bucket}


def _latency(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {"median_ms": float(s.median()), "p95_ms": float(s.quantile(0.95)), "max_ms": float(s.max())} if len(s) else {}


def read_eval_out(eval_dir: Path = EVAL_DIR) -> dict:
    out = {"predictions": {}, "ablation_csv": None, "charts": []}
    if not eval_dir.exists():
        return out
    for v in VARIANTS:
        p = eval_dir / f"predictions_{v}.csv"
        if p.exists():
            try:
                out["predictions"][v] = pd.read_csv(p)
            except Exception:
                pass
    for p in sorted(eval_dir.glob("*ablation*.csv")):
        try:
            out["ablation_csv"] = pd.read_csv(p).to_dict("records")
            break
        except Exception:
            pass
    out["charts"] = [str(p) for p in sorted(eval_dir.glob("*")) if p.suffix.lower() in (".png", ".svg", ".html", ".jpg")]
    return out


def internal_variants(engine) -> dict[str, pd.DataFrame]:
    """Per-variant action frames for the test-split decisions the engine has produced so far."""
    d = pd.DataFrame(engine.decisions)
    if d.empty or "label_gt" not in d:
        return {}
    comp = pd.DataFrame([{c: r[c]["score"] for c in COMPONENTS} for r in d["components"]])
    base = pd.DataFrame({"txn_id": d["txn_id"], "label": d["label_gt"], "amount": d["amount"]})
    out = {}
    a = base.copy()
    a["action"] = np.where(d["amount"] >= config.ALERT_THRESHOLD_INR, "HOLD", "ALLOW")
    a["latency_ms"] = 0.01
    out["A_rules"] = a
    tf = engine.train_frame
    for v, comps in VARIANT_COMPONENTS.items():
        f = Fusion(engine.state_dir / f"fusion_{v}.json", components=comps)
        f.fit(tf[comps].values, tf["label"].values)
        risk = [f.score(r)["risk_score"] for r in comp[comps].to_dict("records")]
        x = base.copy()
        x["risk_score"], x["action"] = risk, [orchestrator.band_for(r) for r in risk]
        x["latency_ms"] = d["scorer_latency_ms"] if v != "D_plus_provenance" else d["latency_ms"]
        out[v] = x
    return out


def compute_metrics(engine=None, eval_dir: Path = EVAL_DIR) -> dict:
    ev = read_eval_out(eval_dir)
    result = {"source": None, "variants": {}, "ablation_csv": ev["ablation_csv"], "charts": ev["charts"], "latency": {}, "notes": []}
    frames, src = {}, None
    if ev["predictions"] and engine is not None:
        lab = engine.txns[["txn_id", "label", "amount", "_ts"]]
        for v, p in ev["predictions"].items():
            j = p.merge(lab, on="txn_id", how="inner")
            frames[v] = j[j["_ts"] >= SPLIT]
        src = "eval/out (acc_1 predictions)"
    elif engine is not None:
        frames = internal_variants(engine)
        src = "internal ablation from engine component scores (acc_1 predictions not found)" if frames else None
    if src is None:
        result["notes"].append("No predictions in eval/out and no engine decisions yet: run the live monitor or POST /score first.")
    result["source"] = src
    for v, df in frames.items():
        s = _summ(df)
        s["label"] = VARIANTS.get(v, v)
        if "latency_ms" in df:
            s["latency"] = _latency(df["latency_ms"])
        result["variants"][v] = s
    if engine is not None and engine.decisions:
        result["latency"] = {"engine_end_to_end": _latency(pd.Series([d["latency_ms"] for d in engine.decisions])), "backend": engine.backend}
        d = pd.DataFrame(engine.decisions)
        if "label_gt" in d:
            neg = d[d.label_gt == 0]
            result["live_friction"] = {"legit_txns": int(len(neg)), "stepped_up": int((neg.action == "STEP_UP").sum()),
                                       "held_or_worse": int(neg.action.isin(["HOLD", "BLOCK", "ESCALATE"]).sum())}
    return json.loads(json.dumps(result, default=lambda o: None if o is pd.NA else float(o) if isinstance(o, np.floating) else str(o)))


def pick_story_txn(engine) -> dict | None:
    """The demo story: a ~Rs 4,800 relayed fragment of a ~Rs 5,00,000 root that rules AND ML-only (and graph-only)
    miss or under-react to, but the full engine holds/blocks. Falls back to the best available fragment."""
    frames = internal_variants(engine)
    if not frames:
        return None
    act = {v: dict(zip(df["txn_id"], df["action"])) for v, df in frames.items()}
    cands = []
    for d in engine.decisions:
        f = d.get("features", {})
        if d.get("label_gt") == 1 and 4000 < d["amount"] < config.AMOUNT_BUCKETS[0][1] and f.get("root_aggregate_amount", 0) >= 400000 and f.get("hops") == 2:
            t = d["txn_id"]
            miss = (act["A_rules"][t] == "ALLOW") + (act["B_behavior_anomaly"][t] == "ALLOW") + (act["C_plus_graph"][t] in ("ALLOW", "STEP_UP"))
            caught = d["action"] in ("HOLD", "BLOCK", "ESCALATE")
            cands.append((caught, miss, f.get("fragment_count", 0) >= 30, d["amount"], d))
    return max(cands, key=lambda c: c[:4])[4] if cands else None


def story_outcomes(engine, txn_id: str) -> list[dict]:
    """What each ablation variant decided for one transaction."""
    out = []
    frames = internal_variants(engine)
    for v, df in frames.items():
        r = df[df["txn_id"] == txn_id]
        if len(r):
            out.append({"variant": VARIANTS[v], "action": r.iloc[0]["action"], "risk_score": float(r.iloc[0].get("risk_score", np.nan)) if "risk_score" in r else None})
    return out
