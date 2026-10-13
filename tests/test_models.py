# tests/test_models.py
"""Tests for model training, execution, determinism, and leakage."""
import os
import shutil
import pandas as pd
from pathlib import Path
from models.train import train_models
from models.stream_scorer import StreamScorer
import config

def setup_module():
    # Setup dummy data
    Path("data/out").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"txn_id": f"t{i}", "ts": f"2026-01-01T10:0{i%60}:00Z", "src_account": f"A{i%5}", 
         "dst_account": f"B{i%5}", "amount": 1000.0, "label": i%2, 
         "typology": "x", "scenario_id": "y", "root_txn_id_gt": "z", "role_gt": "m"}
        for i in range(2500)
    ])
    df.to_csv("data/out/transactions.csv", index=False)
    
    # Train dummy models
    train_models()

def teardown_module():
    shutil.rmtree("data/out", ignore_errors=True)
    shutil.rmtree("models/saved", ignore_errors=True)

def test_scorer_contract_and_bounds():
    scorer = StreamScorer("models")
    txn = {"txn_id": "tx1", "ts": "2026-01-10T00:00:00Z", "src_account": "A", "dst_account": "B", "amount": 100.0}
    out = scorer.process(txn)
    
    assert "txn_id" in out
    for k in ["behavior_score", "anomaly_score", "graph_score", "provenance_score", "cluster_risk"]:
        assert 0.0 <= out[k] <= 1.0
    assert isinstance(out["reasons"], list)
    assert isinstance(out["features"], dict)
    assert isinstance(out["latency_ms"], float)

def test_no_eval_leakage():
    scorer = StreamScorer("models")
    txn_clean = {"txn_id": "tx1", "ts": "2026-01-10T00:00:00Z", "src_account": "A", "dst_account": "B", "amount": 100.0}
    txn_dirty = {**txn_clean, "label": 1, "typology": "smurfing", "scenario_id": "s1"}
    
    out_clean = scorer.process(txn_clean)
    
    scorer2 = StreamScorer("models") # Fresh state
    out_dirty = scorer2.process(txn_dirty)
    
    # Scores must be perfectly identical
    assert out_clean["behavior_score"] == out_dirty["behavior_score"]
    assert out_clean["graph_score"] == out_dirty["graph_score"]

def test_fix_determinism():
    scorer_on = StreamScorer("models", enable_gc_fix=True)
    scorer_off = StreamScorer("models", enable_gc_fix=False)
    
    txn = {"txn_id": "tx1", "ts": "2026-01-10T00:00:00Z", "src_account": "A", "dst_account": "B", "amount": 100.0}
    
    # Push through boundary to trigger graph update
    for _ in range(2005):
        out_on = scorer_on.process(txn)
        out_off = scorer_off.process(txn)
        
    assert out_on["graph_score"] == out_off["graph_score"]
    assert out_on["cluster_risk"] == out_off["cluster_risk"]