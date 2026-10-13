import time
from ledger.lots import LotLedger

class MockConfig:
    ALERT_THRESHOLD_INR = 50000
    HOP_CAP = 3
    WINDOW_HOURS = 24
    SPLIT_TS = "2026-01-01"
    SEED = 42
    USE_MOCK = False
    MODELS_DIR = ""
    ACTION_BANDS = []
    AMOUNT_BUCKETS = []

def test_proportional_split_and_conservation():
    ledger = LotLedger(MockConfig())

    txn1 = {'txn_id': 't1', 'ts': '2026-10-13T10:00:00Z', 'src_account': 'EXT_SALARY', 'dst_account': 'A', 'amount': 1000, 'origin_type': 'SALARY'}
    ledger.process(txn1)

    txn2 = {'txn_id': 't2', 'ts': '2026-10-13T10:01:00Z', 'src_account': 'A', 'dst_account': 'B', 'amount': 400}
    ledger.process(txn2)

    assert sum(l.amount for l in ledger.accounts['A']) == 600
    assert sum(l.amount for l in ledger.accounts['B']) == 400

    txn3 = {'txn_id': 't3', 'ts': '2026-10-13T10:02:00Z', 'src_account': 'C', 'dst_account': 'A', 'amount': 200}
    ledger.process(txn3)
    
    assert sum(l.amount for l in ledger.accounts['C']) == 0
    assert sum(l.amount for l in ledger.accounts['A']) == 800

    total_in_system = sum(sum(l.amount for l in lots) for lots in ledger.accounts.values())
    assert total_in_system == 1200

def test_hop_cap():
    ledger = LotLedger(MockConfig())
    for i in range(1, 6):
        src = 'EXT_CASH' if i == 1 else f'A{i-1}'
        ledger.process({'txn_id': f't{i}', 'ts': f'2026-10-13T10:00:0{i}Z', 'src_account': src, 'dst_account': f'A{i}', 'amount': 100, 'origin_type': 'CASH'})

    lots = ledger.accounts['A5']
    assert lots[0].hop_count <= MockConfig.HOP_CAP
    assert len(lots[0].account_path) == 0  # Should be cleared by HOP_CAP enforcement

def test_performance():
    ledger = LotLedger(MockConfig())
    start = time.time()
    for i in range(500):
        ledger.process({'txn_id': f'tx_{i}', 'ts': '2026-10-13T10:00:00Z', 'src_account': f'EXT_{i}', 'dst_account': f'U_{i}', 'amount': 10, 'origin_type': 'BUSINESS'})
    duration = time.time() - start
    avg_ms = (duration / 500) * 1000
    assert avg_ms < 5.0