import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import orchestrator as o  # noqa: E402
from engine.orchestrator import (ACTION_ORDER, INTERVENTION, OFFSET_LIMIT, SLA_MINUTES, band_for, decide,  # noqa: E402
                                 segment_key)

REQUIRED_KEYS = {"action", "risk_score", "base_risk_score", "cluster_risk", "reasons", "recommended_intervention",
                 "sla_minutes", "segment", "escalated", "opens_case"}


@pytest.mark.parametrize("score,expected", [
    (0.0, "ALLOW"), (0.2999999, "ALLOW"), (0.30, "STEP_UP"), (0.5499999, "STEP_UP"),
    (0.55, "HOLD"), (0.7999999, "HOLD"), (0.80, "BLOCK"), (1.0, "BLOCK"),
])
def test_band_boundaries(score, expected):
    assert band_for(score) == expected
    assert decide(score)["action"] == expected


def test_band_just_below_edges():
    for edge, below in ((0.30, "ALLOW"), (0.55, "STEP_UP"), (0.80, "HOLD")):
        assert band_for(math.nextafter(edge, 0)) == below


def test_band_for_bad_input():
    for bad in (float("nan"), None, "x", -3, float("-inf")):
        assert band_for(bad) == "ALLOW"
    assert band_for(7) == "BLOCK" and band_for(float("inf")) == "ALLOW"  # inf is non-finite -> treated as missing


def test_escalate_on_cluster_risk():
    d = decide(0.10, 0.70)
    assert d["action"] == "ESCALATE" and d["escalated"] and d["opens_case"]
    assert decide(0.10, 0.6999)["action"] == "ALLOW"
    assert decide(0.90, 0.95)["action"] == "ESCALATE"  # overrides BLOCK
    assert not decide(0.9, 0.2)["escalated"]


def test_opens_case_only_hold_and_above():
    assert {a: decide(s)["opens_case"] for a, s in
            [("ALLOW", 0.1), ("STEP_UP", 0.4), ("HOLD", 0.6), ("BLOCK", 0.9)]} == \
           {"ALLOW": False, "STEP_UP": False, "HOLD": True, "BLOCK": True}


def test_action_order_and_maps_complete():
    assert ACTION_ORDER == ["ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"]
    assert set(INTERVENTION) == set(SLA_MINUTES) == set(ACTION_ORDER)
    assert all(isinstance(t, str) and t for t in INTERVENTION.values())
    assert all(isinstance(m, int) and m >= 0 for m in SLA_MINUTES.values())


def test_decision_dict_contract():
    d = decide(0.6, 0.1, "personal", 2, reasons=["r1"])
    assert REQUIRED_KEYS <= set(d)
    assert d["recommended_intervention"] == INTERVENTION[d["action"]]
    assert d["sla_minutes"] == SLA_MINUTES[d["action"]]
    assert d["segment"] == segment_key("personal", 2) == "personal|kyc2"
    assert 0 <= d["risk_score"] <= 1 and d["reasons"][0] == "r1"


def test_offsets_never_exceed_limit():
    seg = segment_key("personal", 1)
    for off in (5.0, -5.0, OFFSET_LIMIT * 3, float("nan"), float("inf")):
        d = decide(0.5, 0.0, "personal", 1, learned_offsets={seg: off})
        assert abs(d["risk_score"] - 0.5) <= OFFSET_LIMIT + 1e-9
        assert abs(d["offset_applied"]) <= OFFSET_LIMIT
    rng = random.Random(1)
    for _ in range(500):
        s, off = rng.random(), rng.uniform(-1, 1)
        d = decide(s, 0.0, "merchant", 3, learned_offsets={segment_key("merchant", 3): off})
        assert abs(d["risk_score"] - s) <= OFFSET_LIMIT + 1e-6 and 0 <= d["risk_score"] <= 1


def test_offset_moves_band_in_right_direction():
    seg = segment_key("personal", 1)
    assert decide(0.54, 0, "personal", 1)["action"] == "STEP_UP"
    assert decide(0.54, 0, "personal", 1, learned_offsets={seg: 0.02})["action"] == "HOLD"   # confirmed -> stricter
    assert decide(0.56, 0, "personal", 1, learned_offsets={seg: -0.02})["action"] == "STEP_UP"  # false positive -> looser
    d = decide(0.54, 0, "personal", 1, learned_offsets={seg: 0.02})
    assert d["base_risk_score"] == 0.54 and d["base_action"] == "STEP_UP"


def test_offsets_are_per_segment():
    offs = {segment_key("personal", 1): 0.05}
    assert decide(0.52, 0, "personal", 1, learned_offsets=offs)["action"] == "HOLD"
    assert decide(0.52, 0, "personal", 2, learned_offsets=offs)["action"] == "STEP_UP"
    assert decide(0.52, 0, "merchant", 1, learned_offsets=offs)["action"] == "STEP_UP"


def test_segment_key_handles_missing_and_numpy_like():
    assert segment_key(None, None) == "unknown|kycunknown"
    assert segment_key(float("nan"), float("nan")) == "unknown|kyc" + "unknown"
    assert segment_key("personal", 2.0) == "personal|kyc2"


def test_nan_and_missing_inputs():
    d = decide(float("nan"), float("nan"), None, None)
    assert d["action"] == "ALLOW" and d["risk_score"] == 0.0 and REQUIRED_KEYS <= set(d)
    assert decide(None, None)["action"] == "ALLOW"


def test_reasons_not_mutated_and_watchlist_reason_once():
    r = ["a"]
    d = decide(0.6, 0.0, reasons=r, watchlist_hit=True)
    assert r == ["a"] and sum("watchlist" in x.lower() for x in d["reasons"]) == 1
    d2 = decide(0.6, 0.0, reasons=d["reasons"], watchlist_hit=True)
    assert sum("watchlist" in x.lower() for x in d2["reasons"]) == 1
    assert decide(0.6)["reasons"]  # never empty for a non-ALLOW action


def test_action_monotonic_in_risk():
    last = 0
    for i in range(0, 1001):
        idx = ACTION_ORDER.index(band_for(i / 1000))
        assert idx >= last
        last = idx
