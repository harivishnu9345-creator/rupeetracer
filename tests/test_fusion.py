import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.cases import CaseManager  # noqa: E402
from engine.fusion import (COMPONENTS, DEFAULT_WEIGHTS, CORROBORATION_GAIN, OTHERS_LOW, SCALE_MAX, SCALE_MIN,  # noqa: E402
                           Fusion)
from engine.orchestrator import BAND_FLOORS, band_for, decide  # noqa: E402

HOLD = BAND_FLOORS["HOLD"]


def synth(n=4000, seed=0, prov_strength=1.0):
    """Synthetic component scores; every component is informative, provenance strongly so."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    X = np.column_stack([
        np.clip(rng.normal(0.15 + 0.45 * y, 0.15), 0, 1),
        np.clip(rng.normal(0.15 + 0.35 * y, 0.15), 0, 1),
        np.clip(rng.normal(0.10 + 0.40 * y, 0.15), 0, 1),
        np.clip(rng.normal(0.05 + 0.80 * prov_strength * y, 0.10), 0, 1),
    ])
    return X, y


@pytest.fixture(scope="module")
def trained():
    f = Fusion()
    X, y = synth()
    meta = f.fit(X, y)
    assert meta["trained"] and f.mode == "logistic"
    return f


def untrained():
    return Fusion()


def both(trained_fusion):
    return [untrained(), trained_fusion]


# ----------------------------------------------------------------- basics
def test_exports():
    assert COMPONENTS == ["behavior", "anomaly", "graph", "provenance"]
    assert 0 < SCALE_MIN < 1 < SCALE_MAX
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1) < 1e-9


def test_untrained_is_weighted_average():
    f = untrained()
    assert f.mode == "weighted_average"
    r = f.score({c: 0.5 for c in COMPONENTS})
    assert r["mode"] == "weighted_average" and r["risk_score"] == pytest.approx(0.5, abs=1e-6)
    assert f.score({c: 0.0 for c in COMPONENTS})["risk_score"] == 0.0
    assert f.score({c: 1.0 for c in COMPONENTS})["risk_score"] == pytest.approx(1.0)


def test_training_learns_signal(trained):
    low = trained.score({"behavior": 0.1, "anomaly": 0.1, "graph": 0.1, "provenance": 0.05})["risk_score"]
    high = trained.score({"behavior": 0.6, "anomaly": 0.5, "graph": 0.5, "provenance": 0.85})["risk_score"]
    assert low < 0.30 < high
    assert all(w >= 0 for w in trained.weights.values())


def test_too_little_data_stays_fallback():
    f = Fusion()
    meta = f.fit(np.random.rand(10, 4), np.zeros(10))
    assert meta["trained"] is False and f.mode == "weighted_average"
    f2 = Fusion()
    f2.fit(np.random.rand(500, 4), np.zeros(500))  # no positives
    assert not f2.trained


def test_fit_tolerates_nan_in_training_matrix():
    X, y = synth(2000, seed=3)
    X[::7, 1] = np.nan
    X[::11, 3] = np.inf
    f = Fusion()
    assert f.fit(X, y)["trained"]


def test_fit_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Fusion().fit(np.zeros((100, 3)), np.zeros(100))


# ------------------------------------------------------ range & bad input
BAD = [float("nan"), float("inf"), float("-inf"), None, "abc", -5, 9, [1], {}]


def test_output_always_in_unit_interval(trained):
    for f in both(trained):
        for combo in itertools.product([0.0, 0.5, 1.0, *BAD[:5]], repeat=4):
            r = f.score(dict(zip(COMPONENTS, combo)))
            assert 0.0 <= r["risk_score"] <= 1.0 and math.isfinite(r["risk_score"])
            assert all(math.isfinite(v) and v >= 0 for v in r["contributions"].values())


def test_missing_keys_and_garbage_input(trained):
    for f in both(trained):
        for inp in ({}, None, "x", [], {"behavior": 0.9}, {"zzz": 1}, {c: float("nan") for c in COMPONENTS}):
            r = f.score(inp)
            assert 0.0 <= r["risk_score"] <= 1.0
        r = f.score({"behavior": 0.9})
        assert r["mode"] == "weighted_average" and set(r["missing"]) == {"anomaly", "graph", "provenance"}


def test_missing_input_uses_fallback_even_when_trained(trained):
    r = trained.score({"behavior": 0.8, "anomaly": 0.7, "graph": 0.6, "provenance": float("nan")})
    assert r["mode"] == "weighted_average" and r["missing"] == ["provenance"]
    assert r["risk_score"] > 0.5  # renormalised over what is present, not diluted by the gap


def test_bad_scale_values_are_clamped(trained):
    comps = {c: 0.5 for c in COMPONENTS}
    for sc in ({"behavior": float("nan")}, {"behavior": 1e9}, {"behavior": -4}, "junk", None, {"x": 2}):
        assert 0 <= trained.score(comps, sc)["risk_score"] <= 1
    assert trained.score(comps, {"behavior": 1e9}) == trained.score(comps, {"behavior": SCALE_MAX})


# ----------------------------------------------- hard rule: provenance alone
def test_provenance_alone_cannot_reach_hold(trained):
    lows = [0.0, 0.05, 0.15, 0.25, OTHERS_LOW - 0.001]
    worst = 0.0
    for f in both(trained):
        for scale in (None, {"provenance": SCALE_MAX}, {c: SCALE_MAX for c in COMPONENTS}):
            for others in itertools.product(lows, repeat=3):
                for p in (0.0, 0.5, 0.9, 1.0):
                    r = f.score(dict(zip(COMPONENTS, [*others, p])), scale)
                    worst = max(worst, r["risk_score"])
                    assert r["risk_score"] < HOLD, (f.mode, others, p, scale, r)
                    assert band_for(r["risk_score"]) in ("ALLOW", "STEP_UP")
                    assert decide(r["risk_score"])["action"] in ("ALLOW", "STEP_UP")
    assert worst < HOLD


def test_provenance_alone_random_sweep(trained):
    rng = np.random.default_rng(7)
    for f in both(trained):
        for _ in range(3000):
            others = rng.uniform(0, OTHERS_LOW - 1e-6, 3)
            r = f.score(dict(zip(COMPONENTS, [*others, rng.uniform(0.7, 1.0)])), {"provenance": SCALE_MAX})
            assert r["risk_score"] < HOLD


def test_provenance_alone_with_missing_others(trained):
    for f in both(trained):
        r = f.score({"provenance": 1.0})
        assert r["risk_score"] < HOLD
        r = f.score({"behavior": 0.1, "anomaly": float("nan"), "graph": None, "provenance": 1.0})
        assert r["risk_score"] < HOLD


def test_cap_relaxes_smoothly_with_corroboration(trained):
    """No cliff at OTHERS_LOW: the ceiling rises continuously as the strongest other component grows."""
    for f in both(trained):
        prev = None
        for m in np.linspace(0.0, 0.8, 161):
            r = f.score({"behavior": m, "anomaly": 0.0, "graph": 0.0, "provenance": 1.0}, {"provenance": SCALE_MAX})["risk_score"]
            if prev is not None:
                assert r >= prev - 1e-9 and r - prev < 0.05   # monotone, no jumps
            prev = r
        edge = f.score({"behavior": OTHERS_LOW, "anomaly": 0, "graph": 0, "provenance": 1.0}, {"provenance": SCALE_MAX})
        assert edge["risk_score"] < HOLD
    assert CORROBORATION_GAIN > 0


def test_provenance_still_helps_when_others_are_elevated(trained):
    """The rule is a cap, not a mute button: with real corroboration provenance raises the score."""
    c = {"behavior": 0.35, "anomaly": 0.35, "graph": 0.45}
    for f in both(trained):
        assert f.score({**c, "provenance": 0.9})["risk_score"] > f.score({**c, "provenance": 0.0})["risk_score"]


def test_other_components_can_still_reach_hold(trained):
    assert untrained().score({"behavior": 1, "anomaly": 1, "graph": 1, "provenance": 0})["risk_score"] < 1  # sanity
    r = trained.score({"behavior": 0.7, "anomaly": 0.6, "graph": 0.6, "provenance": 0.9})
    assert r["risk_score"] >= HOLD


def test_provenance_not_in_subset_fusion_has_no_gate():
    f = Fusion(components=["behavior", "anomaly"])
    r = f.score({"behavior": 1.0, "anomaly": 1.0, "provenance": 0.0})
    assert r["risk_score"] == pytest.approx(1.0) and not r["gated"]
    assert set(r["contributions"]) == {"behavior", "anomaly"}


# --------------------------------------------------------- contributions
def test_contributions_sum_to_risk(trained):
    rng = np.random.default_rng(11)
    for f in both(trained):
        for scale in (None, {"graph": 1.3, "provenance": 0.7}, {"provenance": SCALE_MAX}):
            for _ in range(500):
                v = rng.random(4)
                if rng.random() < 0.3:
                    v[rng.integers(0, 4)] = float("nan")
                if rng.random() < 0.3:
                    v[:3] *= 0.25  # provoke the provenance cap
                r = f.score(dict(zip(COMPONENTS, v)), scale)
                assert set(r["contributions"]) == set(COMPONENTS)
                assert sum(r["contributions"].values()) == pytest.approx(r["risk_score"], abs=1e-4)
                assert all(x >= 0 for x in r["contributions"].values())


def test_zero_risk_zero_contributions():
    r = untrained().score({c: 0.0 for c in COMPONENTS})
    assert r["risk_score"] == 0.0 and all(v == 0.0 for v in r["contributions"].values())


def test_contribution_tracks_the_dominant_component(trained):
    r = trained.score({"behavior": 0.9, "anomaly": 0.0, "graph": 0.0, "provenance": 0.0})
    assert max(r["contributions"], key=r["contributions"].get) == "behavior"


def test_capped_provenance_gets_only_its_lift(trained):
    r = trained.score({"behavior": 0.0, "anomaly": 0.0, "graph": 0.0, "provenance": 1.0}, {"provenance": SCALE_MAX})
    assert r["gated"] and r["risk_score"] < HOLD
    c = r["contributions"]
    assert sum(c.values()) == pytest.approx(r["risk_score"], abs=1e-4)
    assert c["provenance"] <= r["risk_score"] and c["provenance"] == pytest.approx(r["risk_score"], abs=0.05)


# ------------------------------------------------------------ persistence
def test_weights_json_round_trip(tmp_path, trained):
    p = tmp_path / "state" / "fusion_weights.json"
    f = Fusion(p)
    f.fit(*synth(3000, seed=5))
    f.save()
    assert p.exists() and json.loads(p.read_text())["trained"] is True
    g = Fusion(p)  # loads on construction
    assert g.trained and g.weights == f.weights and g.intercept == f.intercept and g.mode == "logistic"
    rng = np.random.default_rng(2)
    for _ in range(200):
        c = dict(zip(COMPONENTS, rng.random(4)))
        assert g.score(c) == f.score(c)
    assert not list(p.parent.glob("*.tmp"))


def test_round_trip_untrained_and_bad_files(tmp_path):
    p = tmp_path / "w.json"
    Fusion(p).save()
    assert not Fusion(p).trained
    p.write_text("{not json")
    assert not Fusion(p).trained  # corrupt file ignored, no crash
    p.write_text(json.dumps({"components": ["behavior"], "trained": True, "weights": {"behavior": 1}}))
    assert not Fusion(p).trained  # component mismatch ignored
    Fusion(p, components=["behavior"]).load()


def test_describe_is_json_serialisable(trained):
    d = trained.describe()
    json.dumps(d)
    assert d["mode"] == "logistic" and set(d["weights"]) == set(COMPONENTS)


def test_subset_fusion_like_metrics_ablation():
    X, y = synth(2000, seed=9)
    comps = ["behavior", "anomaly"]
    f = Fusion(components=comps)
    f.fit(X[:, :2], y)
    assert f.trained
    r = f.score({"behavior": 0.5, "anomaly": 0.5, "graph": 0.9, "provenance": 0.9})
    assert 0 <= r["risk_score"] <= 1


# --------------------------------------- feedback loop (via cases.py)
def _case_manager():
    cm = CaseManager(persist=False)
    comps = {"behavior": 0.30, "anomaly": 0.10, "graph": 0.15, "provenance": 0.05}
    raw = {"behavior": 0.8, "anomaly": 0.4, "graph": 0.4, "provenance": 0.2}
    txn = {"txn_id": "T1", "ts": "2026-01-25T02:30:00+00:00", "src_account": "A1", "dst_account": "A2", "amount": 5000}
    dec = {"opens_case": True, "action": "HOLD", "segment": "personal|kyc1", "reasons": ["x"], "risk_score": 0.6,
           "base_risk_score": 0.6, "cluster_risk": 0.1, "account_type": "personal", "kyc_tier": 1}
    cid = cm.record(txn, dec, None, None, "behavior", comps, raw)
    return cm, cid, raw


def test_feedback_changes_weights_in_expected_direction():
    f = untrained()
    cm, cid, raw = _case_manager()
    r0 = f.score(raw, cm.scale())["risk_score"]

    res = cm.feedback(cid, "confirmed")
    assert res["changed"] and cm.scale()["behavior"] > 1.0
    r_up = f.score(raw, cm.scale())["risk_score"]
    assert r_up > r0
    assert cm.offsets()["personal|kyc1"] > 0

    res = cm.feedback(cid, "false_positive")  # flips: undo, then apply opposite
    assert cm.scale()["behavior"] < 1.0
    r_down = f.score(raw, cm.scale())["risk_score"]
    assert r_down < r0 < r_up
    assert cm.offsets()["personal|kyc1"] < 0


def test_feedback_direction_with_trained_stacker(trained):
    cm, cid, raw = _case_manager()
    base = trained.score(raw, cm.scale())["risk_score"]
    cm.feedback(cid, "confirmed")
    assert trained.score(raw, cm.scale())["risk_score"] > base
    cm.feedback(cid, "false_positive")
    assert trained.score(raw, cm.scale())["risk_score"] < base


def test_feedback_scale_stays_within_bounds_and_hard_rule_holds():
    from engine.cases import SCALE_STEP
    f = untrained()
    cm = CaseManager(persist=False)
    for i in range(30):  # many distinct confirmed cases, all driven by provenance
        cm.adjustments["component_scale"]["provenance"] = min(SCALE_MAX, cm.adjustments["component_scale"]["provenance"] * (1 + SCALE_STEP))
    assert cm.scale()["provenance"] == pytest.approx(SCALE_MAX)
    r = f.score({"behavior": 0.0, "anomaly": 0.0, "graph": 0.0, "provenance": 1.0}, cm.scale())
    assert r["risk_score"] < HOLD


def test_feedback_moves_decision_across_band_edge():
    """False-positive feedback lowers the segment offset enough to demote a borderline HOLD."""
    cm, cid, _ = _case_manager()
    for _ in range(3):
        cm.feedback(cid, "false_positive")  # idempotent: same label again changes nothing
    assert cm.offsets()["personal|kyc1"] == pytest.approx(-0.02)
    d = decide(0.56, 0.0, "personal", 1, learned_offsets=cm.offsets())
    assert d["action"] == "STEP_UP"
