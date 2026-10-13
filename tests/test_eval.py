import json

import numpy as np
import pandas as pd
import pytest

import config as C
from data import schemas as S
from eval import ablation, buckets, latency, metrics as M, rules_baseline as R


@pytest.fixture(scope="module")
def truth(small):
    return M.load_truth(small[1])


def make_pred(truth, flag, score=None, action=None, lat=1.0):
    n = len(truth)
    score = np.where(flag, .9, .05) if score is None else score
    action = np.where(flag, "BLOCK", "ALLOW") if action is None else action
    return pd.DataFrame({"txn_id": truth.txn_id, "risk_score": score, "action": action,
                         "latency_ms": np.full(n, lat), "reasons": "[]"})


def test_oracle_metrics(truth):
    p = make_pred(truth, truth.label.to_numpy() == 1)
    m = M.compute_metrics(p, truth, "oracle")
    assert m["precision"] == m["recall"] == m["f1"] == 1.0
    assert m["fpr"] == 0 and m["friction_rate"] == 0 and m["pr_auc"] == pytest.approx(1.0)
    assert m["scenario_recall"] == 1.0 and m["det_latency_median_min"] == 0.0
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(truth)


def test_all_allow(truth):
    m = M.compute_metrics(make_pred(truth, np.zeros(len(truth), bool)), truth, "none")
    assert m["recall"] == 0 and m["precision"] == 0 and m["friction_rate"] == 0
    assert m["scenarios_detected"] == 0 and np.isnan(m["det_latency_median_min"])


def test_all_flag_and_friction(truth):
    m = M.compute_metrics(make_pred(truth, np.ones(len(truth), bool)), truth, "all")
    assert m["recall"] == 1.0 and m["fpr"] == 1.0 and m["friction_rate"] == 1.0
    assert m["precision"] == pytest.approx(truth.label.mean())


def test_step_up_is_friction_not_flag(truth):
    p = make_pred(truth, np.zeros(len(truth), bool), action="STEP_UP")
    m = M.compute_metrics(p, truth, "stepup")
    assert m["tp"] + m["fp"] == 0 and m["friction_rate"] == 1.0


def test_escalate_and_combined_action(truth):
    p = make_pred(truth, np.zeros(len(truth), bool), action="ALLOW+ESCALATE")
    assert M.compute_metrics(p, truth, "esc")["recall"] == 1.0


def test_random_pr_auc_near_prevalence(truth):
    rng = np.random.default_rng(0)
    p = make_pred(truth, rng.random(len(truth)) > .9, score=rng.random(len(truth)))
    m = M.compute_metrics(p, truth, "rand")
    assert abs(m["pr_auc"] - truth.label.mean()) < 0.02


def test_detection_latency_positive(truth):
    # flag only the LAST txn of each scenario: latency must be > 0 for multi-txn scenarios
    t = truth.copy()
    last = t[t.label == 1].groupby("scenario_id")["ts_dt"].transform("max") == t.loc[t.label == 1, "ts_dt"]
    flag = np.zeros(len(t), bool)
    flag[t.index[t.label == 1][last.to_numpy()]] = True
    m = M.compute_metrics(make_pred(t, flag), t, "late")
    assert m["scenario_recall"] == 1.0 and m["det_latency_median_min"] > 0
    assert m["det_latency_p95_min"] >= m["det_latency_median_min"]


def test_missing_and_extra_predictions(truth):
    p = make_pred(truth, truth.label.to_numpy() == 1).iloc[:-5]
    assert M.compute_metrics(p, truth, "m")["n_txn"] == len(truth)
    extra = pd.concat([p, pd.DataFrame({"txn_id": ["T9999999"], "risk_score": [.5], "action": ["HOLD"],
                                        "latency_ms": [1.0], "reasons": ["[]"]})])
    with pytest.raises(ValueError):
        M.compute_metrics(extra, truth, "x")


def test_rules_baseline(small):
    acc, tx, _ = small
    pred = R.score_transactions(tx, acc)
    S.validate_predictions(pred, set(tx.loc[tx.ts >= C.SPLIT_TS, "txn_id"]))
    assert len(pred) == (tx.ts >= C.SPLIT_TS).sum()
    j = tx.set_index("txn_id").loc[pred.txn_id]
    hit1 = (j.amount.to_numpy() >= C.ALERT_THRESHOLD_INR)
    assert (pred.risk_score.to_numpy()[hit1] >= .55).all()               # R1 reaches HOLD
    assert all("R1" in json.loads(r)[0] for r in pred.reasons[hit1])
    assert (pred.action[~hit1 & (pred.reasons == "[]")] == "ALLOW").all()
    # leakage: identical output when all ground-truth columns are removed
    stripped = tx.drop(columns=S.LEAKAGE_COLUMNS[:4])
    pd.testing.assert_frame_equal(pred.drop(columns="latency_ms"),
                                  R.score_transactions(stripped, acc.drop(columns="role_gt")).drop(columns="latency_ms"))


def test_rules_miss_subthreshold_fragments(small):
    acc, tx, _ = small
    truth = M.load_truth(tx)
    m = M.compute_metrics(R.score_transactions(tx, acc), truth, "A_rules")
    b = buckets.bucket_table({"A_rules": R.score_transactions(tx, acc)}, truth).set_index("bucket")
    assert b["recall"].iloc[0] < 0.6 and m["recall"] < 0.9              # thresholds do not catch everything


def test_rule_velocity_and_round_amount():
    acc = pd.DataFrame({"account_id": ["A", "B"], "primary_device_id": ["d", "e"]})
    ts = pd.date_range("2026-01-25", periods=13, freq="1min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    tx = pd.DataFrame({"txn_id": [f"T{i:07d}" for i in range(1, 15)],
                       "ts": list(ts) + ["2026-01-26T00:00:00Z"],
                       "src_account": ["A"] * 13 + ["B"],
                       "amount": [100.0] * 13 + [49_500.0], "device_id": ["d"] * 13 + ["e"]})
    p = R.score_transactions(tx, acc).set_index("txn_id")
    assert "R2" not in p.reasons.iloc[9] and "R2" in p.reasons.iloc[10]   # 11th txn in the hour fires
    assert "R5" in p.reasons.iloc[13] and p.action.iloc[13] == "HOLD"


def test_buckets_partition(small, truth):
    p = {"o": make_pred(truth, truth.label.to_numpy() == 1)}
    b = buckets.bucket_table(p, truth)
    assert len(b) == len(C.AMOUNT_BUCKETS) and b.n_txn.sum() == len(truth)
    assert b.n_pos.sum() == truth.label.sum()
    assert (b.recall.dropna() == 1.0).all() and (b.precision.dropna() == 1.0).all()


def test_ablation_buckets_latency_scripts(tmp_path, truth):
    rng = np.random.default_rng(1)
    y = truth.label.to_numpy() == 1
    for i, m in enumerate(C.PREDICTION_MODELS):
        flag = y & (rng.random(len(y)) < .3 + .2 * i)
        make_pred(truth, flag).to_csv(tmp_path / f"predictions_{m}.csv", index=False)
    M_main = M.load_truth  # reuse the small truth instead of the full CSVs
    import eval.metrics as mm
    orig = mm.load_truth
    mm.load_truth = lambda tx=None: truth
    try:
        ablation.main(["--pred-dir", str(tmp_path)])
        buckets.main(["--pred-dir", str(tmp_path)])
        latency.main(["--pred-dir", str(tmp_path)])
    finally:
        mm.load_truth = orig
    for f in ["ablation.csv", "ablation.md", "ablation.html", "buckets.csv", "buckets_recall.html",
              "latency.csv", "latency.html"]:
        assert (tmp_path / f).exists(), f
    t = pd.read_csv(tmp_path / "ablation.csv")
    assert list(t.model) == C.PREDICTION_MODELS and t.recall.is_monotonic_increasing
