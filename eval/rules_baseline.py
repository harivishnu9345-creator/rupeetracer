"""Model A: classic threshold rules engine (streaming, per-transaction latency).

Run:  python -m eval.rules_baseline   -> eval/out/predictions_A_rules.csv

Rules (each hit appends a reason string):
  R1  single txn >= ALERT_THRESHOLD_INR
  R2  more than 10 txns in the trailing hour for the sending account
  R3  sending account's outflow for the UTC calendar day > Rs 1,00,000
  R4  device never seen before for the account (and != primary device) AND amount >= LARGE_AMOUNT_INR
  R5  round amount just under threshold (45,000 <= amount < 50,000, multiple of 500)

risk_score = 1 - prod(1 - w_rule), so one hit lands in HOLD, two hits in BLOCK.
Only transaction fields and accounts.csv are read. label / typology / scenario_id /
root_txn_id_gt / role_gt are never touched (tests enforce this).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import config as C
from data import schemas as S

LARGE_AMOUNT_INR = 20_000
MAX_TXNS_PER_HOUR = 10
DAILY_OUTFLOW_INR = 100_000
ROUND_BAND = (0.9 * C.ALERT_THRESHOLD_INR, C.ALERT_THRESHOLD_INR)
WEIGHTS = {"R1": .60, "R2": .58, "R3": .58, "R4": .65, "R5": .57}
NEEDED = ["txn_id", "ts", "src_account", "amount", "device_id"]


def _epoch(ts: pd.Series) -> np.ndarray:
    d = pd.to_datetime(ts, utc=True)
    return ((d - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).astype("int64").to_numpy()


def score_transactions(tx: pd.DataFrame, accounts: pd.DataFrame, split_ts: str = C.SPLIT_TS,
                       measure_latency: bool = True) -> pd.DataFrame:
    """Stream every txn (state is warmed on the train period); emit test-split predictions."""
    tx = tx[NEEDED].reset_index(drop=True)
    primary = dict(zip(accounts["account_id"], accounts["primary_device_id"]))
    t = _epoch(tx["ts"])
    txn_id = tx["txn_id"].to_numpy()
    src = tx["src_account"].to_numpy()
    amt = tx["amount"].to_numpy(dtype=float)
    dev = tx["device_id"].to_numpy()
    is_test = (tx["ts"] >= split_ts).to_numpy()
    real = ~pd.Series(src).str.startswith("EXT_").to_numpy()

    hour_q = defaultdict(deque)           # account -> deque of ts in trailing hour
    day_out = {}                          # account -> [day, cumulative outflow]
    seen = defaultdict(set)               # account -> devices seen so far
    T, lo_b, hi_b = C.ALERT_THRESHOLD_INR, *ROUND_BAND
    out_id, out_score, out_action, out_lat, out_reasons = [], [], [], [], []
    clock = time.perf_counter_ns

    for i in range(len(tx)):
        t0 = clock()
        a, ts_i, s = amt[i], t[i], src[i]
        reasons, keep = [], 1.0
        if a >= T:
            reasons.append(f"R1_amount_ge_threshold: Rs {a:,.0f} >= Rs {T:,.0f}")
            keep *= 1 - WEIGHTS["R1"]
        if real[i]:
            q = hour_q[s]
            while q and q[0] <= ts_i - 3600:
                q.popleft()
            q.append(ts_i)
            if len(q) > MAX_TXNS_PER_HOUR:
                reasons.append(f"R2_velocity: {len(q)} txns in trailing hour (> {MAX_TXNS_PER_HOUR})")
                keep *= 1 - WEIGHTS["R2"]
            day = ts_i // 86400
            st = day_out.get(s)
            if st is None or st[0] != day:
                st = day_out[s] = [day, 0.0]
            st[1] += a
            if st[1] > DAILY_OUTFLOW_INR:
                reasons.append(f"R3_daily_outflow: Rs {st[1]:,.0f} today > Rs {DAILY_OUTFLOW_INR:,.0f}")
                keep *= 1 - WEIGHTS["R3"]
            d = dev[i]
            new = d not in seen[s] and d != primary.get(s)
            seen[s].add(d)
            if new and a >= LARGE_AMOUNT_INR:
                reasons.append(f"R4_new_device_large_amount: unseen device {d}, Rs {a:,.0f}")
                keep *= 1 - WEIGHTS["R4"]
        if lo_b <= a < hi_b and a % 500 == 0:
            reasons.append(f"R5_round_just_under_threshold: Rs {a:,.0f}")
            keep *= 1 - WEIGHTS["R5"]
        score = 1.0 - keep
        action = C.action_for_score(score)
        dt = (clock() - t0) / 1e6
        if is_test[i]:
            out_id.append(txn_id[i])
            out_score.append(round(score, 4))
            out_action.append(action)
            out_lat.append(round(dt, 4) if measure_latency else 0.0)
            out_reasons.append(json.dumps(reasons))

    return pd.DataFrame({"txn_id": out_id, "risk_score": out_score, "action": out_action,
                         "latency_ms": out_lat, "reasons": out_reasons})[S.PREDICTION_COLUMNS]


def main():
    acc = S.load_accounts()
    tx = S.load_transactions()
    pred = score_transactions(tx, acc)
    S.validate_predictions(pred, set(tx.loc[tx["ts"] >= C.SPLIT_TS, "txn_id"]))
    C.EVAL_OUT.mkdir(parents=True, exist_ok=True)
    path = C.EVAL_OUT / "predictions_A_rules.csv"
    pred.to_csv(path, index=False)
    flagged = pred["action"].map(C.is_flag_action).mean()
    print(f"wrote {path} rows={len(pred)} flagged={flagged:.3%} "
          f"latency p50={pred.latency_ms.median():.4f}ms p95={pred.latency_ms.quantile(.95):.4f}ms")


if __name__ == "__main__":
    main()
