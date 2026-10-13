import pytest
import time
import config
from engine.mock_provider import generate_dataset, build_mock

def test_determinism():
    a1, t1 = generate_dataset(42)
    a2, t2 = generate_dataset(42)
    assert a1.equals(a2)
    assert t1.equals(t2)

def test_contract_conformance():
    a, t = generate_dataset(config.SEED)
    assert "primary_device_id" in a.columns
    assert "account_id" in a.columns
    assert "txn_id" in t.columns
    assert "root_txn_id_gt" in t.columns
    assert "label" in t.columns
    assert "typology" in t.columns

def test_showcase_scenario_and_speed():
    a, t = generate_dataset(config.SEED)
    showcase = t[(t["label"] == 1) & (t["typology"] == "fragmentation")]
    assert not showcase.empty
    
    root = showcase[showcase["origin_type"] == "CASH"].iloc[0]
    assert root["amount"] == 500000.0

    story_txn = showcase[showcase["amount"] == 4800.0]
    assert not story_txn.empty
    st = story_txn.iloc[0]

    scorer, ledger = build_mock(a)
    st_res = None
    
    start = time.perf_counter()
    for _, txn in t.iterrows():
        res = scorer.process(txn.to_dict())
        if txn["txn_id"] == st["txn_id"]:
            st_res = res
    elapsed = time.perf_counter() - start

    assert (elapsed / len(t)) < 0.005  # Average well under 5ms

    assert st_res is not None
    assert st_res["provenance_score"] > 0.8
    assert st_res["behavior_score"] < 0.3
    assert st_res["anomaly_score"] < 0.3
    
    lin = ledger.get_lineage(st["txn_id"])
    assert lin["root_aggregate_amount"] == 500000.0
    assert len(lin["account_path"]) >= 2
    assert "hash_chain_ok" in lin