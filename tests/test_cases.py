import pytest
import tempfile
from pathlib import Path
from engine.cases import CaseManager
from engine.fusion import COMPONENTS

def test_case_lifecycle_and_persistence():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cases.json"
        cm = CaseManager(path=path, persist=True)

        txn1 = {"txn_id": "t1", "ts": "2026-01-01T00:00:00Z", "src_account": "A1", "dst_account": "A2", "amount": 500}
        dec1 = {"action": "HOLD", "opens_case": True, "risk_score": 0.8, "cluster_risk": 0.5, "reasons": ["Risk"], "segment": "retail|kyc2"}
        lin1 = {"root_txn_id": "r1", "root_aggregate_amount": 100000, "fragment_count": 5, "origin_type": "CASH", "hops": 2}
        raw_comps = {c: 0.8 for c in COMPONENTS}
        comps = {c: 0.2 for c in COMPONENTS}

        cid1 = cm.record(txn1, dec1, lin1, {}, "provenance", comps, raw_comps)
        assert cid1 is not None

        txn2 = {"txn_id": "t2", "ts": "2026-01-01T00:01:00Z", "src_account": "A2", "dst_account": "A3", "amount": 200}
        dec2 = {"action": "BLOCK", "opens_case": True, "risk_score": 0.9, "cluster_risk": 0.5, "reasons": [], "segment": "retail|kyc2"}
        cid2 = cm.record(txn2, dec2, lin1, {}, "provenance", comps, raw_comps)
        
        assert cid1 == cid2

        cm.flush()

        cm_reloaded = CaseManager(path=path, persist=True)
        case = cm_reloaded.get(cid1)
        assert case["txn_count"] == 2
        assert case["status"] == "open"
        assert case["root_amount"] == 100000

        res_fp = cm_reloaded.feedback(cid1, "false_positive", "Looks okay")
        assert res_fp["status"] == "false_positive"
        assert cm_reloaded.scale()["provenance"] < 1.0

        res_conf = cm_reloaded.feedback(cid1, "confirmed", "Confirmed fraud")
        assert res_conf["status"] == "confirmed"
        assert "A1" in cm_reloaded.watchlist()