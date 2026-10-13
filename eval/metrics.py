"""Evaluation metrics for RupeeTrace predictions (test split only).

Definitions
  flag (action level)  : action tokens intersect {HOLD, BLOCK, ESCALATE}
  friction rate        : share of label==0 txns whose action has STEP_UP, HOLD or BLOCK
  score level          : risk_score >= SCORE_THRESHOLD (HOLD band lower edge, 0.55); PR-AUC uses raw scores
  detection latency    : per scenario, minutes from first label==1 test txn to the first flagged txn
                         of the same scenario; median / p95 over *detected* scenarios, plus scenario recall
  latency_ms           : p50 / p95 of the engine-reported per-txn latency

Run:  python -m eval.metrics [--pred-dir DIR]   -> eval/out/metrics_summary.{csv,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config as C
from data import schemas as S
from eval import _common as H

SCORE_THRESHOLD = 0.55


def load_truth(tx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Test-split ground truth. Evaluation-only columns are used here and nowhere else."""
    if tx is None:
        tx = S.load_transactions()
    t = tx.loc[tx["ts"] >= C.SPLIT_TS, ["txn_id", "ts", "amount", "label", "scenario_id", "typology"]].copy()
    t["ts_dt"] = pd.to_datetime(t["ts"], utc=True)
    return t.reset_index(drop=True)


def align(pred: pd.DataFrame, truth: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Join predictions onto truth. Missing predictions count as ALLOW / score 0 (warned)."""
    extra = set(pred["txn_id"]) - set(truth["txn_id"])
    if extra:
        raise ValueError(f"{name}: {len(extra)} predictions are outside the test split")
    df = truth.merge(pred, on="txn_id", how="left")
    miss = df["risk_score"].isna()
    if miss.any():
        print(f"[warn] {name}: {int(miss.sum())} test txns have no prediction; treated as ALLOW")
    df["risk_score"] = pd.to_numeric(df["risk_score"]).fillna(0.0)
    df["action"] = df["action"].fillna("ALLOW")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"])
    amap = {a: (C.is_flag_action(a), C.is_friction_action(a)) for a in df["action"].unique()}
    df["flag"] = df["action"].map(lambda a: amap[a][0])
    df["friction"] = df["action"].map(lambda a: amap[a][1])
    return df


def _prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return p, r, f1, fpr


def confusion(y: np.ndarray, flag: np.ndarray) -> Dict[str, int]:
    y, flag = y.astype(bool), flag.astype(bool)
    return dict(tp=int((y & flag).sum()), fp=int((~y & flag).sum()),
                fn=int((y & ~flag).sum()), tn=int((~y & ~flag).sum()))


def detection_latency(df: pd.DataFrame) -> Dict[str, float]:
    pos = df[(df["label"] == 1) & (df["scenario_id"] != "")]
    first = pos.groupby("scenario_id")["ts_dt"].min()
    alert = df[df["flag"] & (df["scenario_id"] != "")].groupby("scenario_id")["ts_dt"].min()
    j = pd.concat([first.rename("first"), alert.rename("alert")], axis=1).loc[first.index]
    det = j["alert"].notna()
    lat = ((j.loc[det, "alert"] - j.loc[det, "first"]).dt.total_seconds() / 60.0).clip(lower=0)
    return dict(
        scenarios_total=int(len(j)), scenarios_detected=int(det.sum()),
        scenario_recall=float(det.mean()) if len(j) else 0.0,
        det_latency_median_min=float(lat.median()) if len(lat) else float("nan"),
        det_latency_p95_min=float(lat.quantile(.95)) if len(lat) else float("nan"))


def compute_metrics(pred: pd.DataFrame, truth: pd.DataFrame, name: str = "") -> Dict[str, float]:
    df = align(pred, truth, name)
    y = df["label"].to_numpy()
    cm = confusion(y, df["flag"].to_numpy())
    p, r, f1, fpr = _prf(**cm)
    cs = confusion(y, (df["risk_score"] >= SCORE_THRESHOLD).to_numpy())
    sp, sr, sf1, sfpr = _prf(**cs)
    neg = df["label"] == 0
    out = dict(
        model=name, n_txn=int(len(df)), n_pos=int(y.sum()), prevalence=float(y.mean()),
        tp=cm["tp"], fp=cm["fp"], fn=cm["fn"], tn=cm["tn"],
        precision=p, recall=r, f1=f1, fpr=fpr,
        friction_rate=float(df.loc[neg, "friction"].mean()) if neg.any() else 0.0,
        alert_rate=float(df["flag"].mean()),
        pr_auc=float(average_precision_score(y, df["risk_score"])) if y.sum() else float("nan"),
        score_threshold=SCORE_THRESHOLD, score_tp=cs["tp"], score_fp=cs["fp"], score_fn=cs["fn"], score_tn=cs["tn"],
        score_precision=sp, score_recall=sr, score_f1=sf1, score_fpr=sfpr,
        latency_ms_p50=float(df["latency_ms"].quantile(.5)), latency_ms_p95=float(df["latency_ms"].quantile(.95)),
        latency_ms_p99=float(df["latency_ms"].quantile(.99)), latency_ms_mean=float(df["latency_ms"].mean()),
    )
    out.update(detection_latency(df))
    return out


def summarize(preds: Dict[str, pd.DataFrame], truth: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    truth = load_truth() if truth is None else truth
    return pd.DataFrame([compute_metrics(p, truth, m) for m, p in preds.items()])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=C.EVAL_OUT)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    preds = H.load_all_predictions(a.pred_dir)
    if not preds:
        raise SystemExit("no prediction files found")
    truth = load_truth()
    for m, p in preds.items():
        S.validate_predictions(p, set(truth["txn_id"]))
    df = summarize(preds, truth)
    out = a.out_dir or a.pred_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics_summary.csv", index=False)
    (out / "metrics_summary.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=float))
    show = ["model", "precision", "recall", "f1", "fpr", "friction_rate", "pr_auc", "scenario_recall",
            "det_latency_median_min", "det_latency_p95_min", "latency_ms_p50", "latency_ms_p95"]
    print(H.df_to_md(df[show]))
    return df


if __name__ == "__main__":
    main()
