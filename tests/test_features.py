"""Integration and logical tests for the feature extractors."""
import pytest
import datetime
from features.provenance import extract_features, FEATURE_NAMES as PROV_NAMES
from features.behavior import BehaviorExtractor, FEATURE_NAMES as BEH_NAMES
from features.graph import GraphExtractor, FEATURE_NAMES as GRAPH_NAMES

class MockConfig:
    ALERT_THRESHOLD_INR = 50000

class MockLot:
    def __init__(self, amount, root_txn_id, hop_count, origin_type, last_move_ts, account_path):
        self.amount = amount
        self.root_txn_id = root_txn_id
        self.hop_count = hop_count
        self.origin_type = origin_type
        self.last_move_ts = last_move_ts
        self.account_path = account_path

def test_feature_names_populated():
    assert len(PROV_NAMES) == 12
    assert len(BEH_NAMES) == 18
    assert len(GRAPH_NAMES) == 10

def test_provenance_smurfing_vs_shop():
    """
    Provenance must differentiate Smurfing (1 large root -> many sub-threshold fragments) 
    from a Legitimate Shop (many distinct roots -> 1 fragment each).
    """
    config = MockConfig()
    ts_now = datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()

    # --- Scenario A: Smurfing (Cash Root Rs 5,00,000 -> 100 fragments of Rs 4,900) ---
    root_stats_smurf = {
        "root_1": {
            "aggregate_amount": 500000.0,
            "fragments": [4900.0] * 100, 
            "destinations": {f"mule_{i}" for i in range(30)}
        }
    }
    lots_smurf = [
        MockLot(4900.0, "root_1", 2, "CASH", ts_now - 60, ["source", "relay", "mule_1"])
    ]
    txn_smurf = {"amount": 4900.0, "_ts_sec": ts_now}
    smurf_feats = extract_features(txn_smurf, lots_smurf, root_stats_smurf, {}, config)

    assert smurf_feats["prov_max_root_amount"] == 500000.0
    assert smurf_feats["prov_max_fragments"] == 100
    assert smurf_feats["prov_sub_threshold_fragments_ratio"] == 1.0
    assert smurf_feats["prov_origin_is_cash"] == 1.0

    # --- Scenario B: Shop Fan-In (100 customers pay Rs 5,000 each) ---
    root_stats_shop = {
        f"root_{i}": {
            "aggregate_amount": 5000.0,
            "fragments": [5000.0],
            "destinations": {"shop_1"}
        } for i in range(100)
    }
    lots_shop = [
        MockLot(5000.0, "root_42", 1, "SALARY", ts_now - 120, ["customer_42", "shop_1"])
    ]
    txn_shop = {"amount": 5000.0, "_ts_sec": ts_now}
    shop_feats = extract_features(txn_shop, lots_shop, root_stats_shop, {}, config)

    assert shop_feats["prov_max_root_amount"] == 5000.0
    assert shop_feats["prov_max_fragments"] == 1
    assert shop_feats["prov_origin_is_cash"] == 0.0

    # Primary assertion: Smurfing fragment counts and root amounts severely out-scale shop norms
    assert smurf_feats["prov_max_root_amount"] > shop_feats["prov_max_root_amount"]
    assert smurf_feats["prov_max_fragments"] > shop_feats["prov_max_fragments"]

def test_behavior_finite():
    extr = BehaviorExtractor()
    txn = {"src_account": "A", "dst_account": "B", "amount": 1000.0, "ts": "2026-01-01T10:00:00Z"}
    feats = extr.extract_and_update(txn)
    
    for k, v in feats.items():
        assert isinstance(v, float), f"Feature {k} is not a float"
        assert v == v, f"Feature {k} is NaN" 

def test_graph_refresh():
    extr = GraphExtractor()
    for i in range(2005):
        txn = {"src_account": f"u_{i}", "dst_account": f"u_{i+1}", "amount": 100.0}
        extr.extract_and_update(txn)

    assert len(extr.communities) > 0
    assert len(extr.comm_stats) > 0