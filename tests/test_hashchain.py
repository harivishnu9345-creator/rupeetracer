from ledger.lots import LotLedger
from ledger.hashchain import verify_chain
from tests.test_ledger import MockConfig

def test_verify_chain_tampering():
    ledger = LotLedger(MockConfig())
    ledger.process({'txn_id': 't1', 'ts': '2026-10-13T10:00:00Z', 'src_account': 'EXT_CASH', 'dst_account': 'A', 'amount': 1000, 'origin_type': 'CASH'})
    ledger.process({'txn_id': 't2', 'ts': '2026-10-13T10:01:00Z', 'src_account': 'A', 'dst_account': 'B', 'amount': 400})

    lot = ledger.accounts['B'][0]
    assert verify_chain(lot.lot_id, ledger) is True

    # Tamper with the amount
    lot.amount = 999.0
    assert verify_chain(lot.lot_id, ledger) is False