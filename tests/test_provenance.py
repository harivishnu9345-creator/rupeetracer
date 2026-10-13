from ledger.lots import LotLedger
from tests.test_ledger import MockConfig

def test_no_leakage_and_fragmentation():
    ledger = LotLedger(MockConfig())

    # Scenario A: Legitimate Shop
    for i in range(10):
        ledger.process({'txn_id': f's{i}', 'ts': f'2026-10-13T10:0{i}:00Z', 'src_account': f'CUST_{i}', 'dst_account': 'SHOP', 'amount': 100})
    feats_shop = ledger.process({'txn_id': 's11', 'ts': '2026-10-13T10:15:00Z', 'src_account': 'CUST_11', 'dst_account': 'SHOP', 'amount': 100})

    # Scenario B: Synthetic Fragmentation
    ledger.process({'txn_id': 'f1', 'ts': '2026-10-13T11:00:00Z', 'src_account': 'EXT_CASH', 'dst_account': 'MULE_1', 'amount': 500000, 'origin_type': 'CASH'})
    for i in range(50):
        ledger.process({'txn_id': f'fs_{i}', 'ts': f'2026-10-13T11:05:{i:02d}Z', 'src_account': 'MULE_1', 'dst_account': f'RELAY_{i}', 'amount': 9000})

    # Reconvergence back to Target
    feats_recon = ledger.process({'txn_id': 'fr_0', 'ts': '2026-10-13T11:10:00Z', 'src_account': 'RELAY_0', 'dst_account': 'TARGET', 'amount': 9000})

    # Assert structural separation between legitimate multi-source vs single root fragmentation
    assert feats_recon['root_aggregate_amount'] >= 500000
    assert feats_shop['root_aggregate_amount'] < 500000
    
    assert feats_recon['fragment_count_per_root'] > 0
    assert feats_shop['fragment_count_per_root'] == 0

    # Leakage Test: Evaluation columns are strictly ignored
    txn_leak = {'txn_id': 'l1', 'ts': '2026-10-13T12:00:00Z', 'src_account': 'A', 'dst_account': 'B', 'amount': 10, 'label': 1, 'typology': 'fraud'}
    feats = ledger.process(txn_leak)
    assert 'label' not in feats
    assert 'typology' not in feats
    assert feats is not None