import pytest
from fastapi.testclient import TestClient
from engine.api import app
import config

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["use_mock"] is True

def test_score_and_cases(client):
    txn = {
        "txn_id": "TEST_TXN_001",
        "ts": "2026-01-25T10:00:00Z",
        "src_account": "ACC_T1",
        "dst_account": "ACC_T2",
        "amount": getattr(config, "ALERT_THRESHOLD_INR", 50000) + 1000,
        "channel": "UPI",
        "origin_type": "TRANSFER",
        "device_id": "D1",
        "ip": "1.1.1.1",
        "city": "Mumbai",
        "remark": "test"
    }
    
    res = client.post("/score", json=txn)
    assert res.status_code == 200
    dec = res.json()
    assert dec["latency_ms"] < 100
    assert dec["action"] in ("HOLD", "BLOCK", "ESCALATE")
    case_id = dec["case_id"]
    assert case_id is not None

    res = client.get("/cases")
    assert res.status_code == 200
    cases = res.json()
    assert any(c["case_id"] == case_id for c in cases)

    res = client.get(f"/cases/{case_id}")
    assert res.status_code == 200
    assert res.json()["case_id"] == case_id

    res = client.post(f"/cases/{case_id}/feedback", json={"label": "confirmed"})
    assert res.status_code == 200
    assert res.json()["status"] == "confirmed"

    res = client.get("/lineage/TEST_TXN_001")
    assert res.status_code == 200
    assert res.json()["txn_id"] == "TEST_TXN_001"

def test_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "variants" in res.json()