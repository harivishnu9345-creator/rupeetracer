import hashlib

def generate_hash(parent_hashes: list, txn_id: str, amount: float, ts: float) -> str:
    """
    Generates a SHA-256 hash over sorted parent hashes, txn_id, amount, and timestamp.
    """
    m = hashlib.sha256()
    for ph in sorted(parent_hashes):
        m.update(ph.encode('utf-8'))
    m.update(str(txn_id).encode('utf-8'))
    m.update(str(float(amount)).encode('utf-8'))
    m.update(str(float(ts)).encode('utf-8'))
    return m.hexdigest()

def verify_chain(lot_id: str, ledger) -> bool:
    """
    Verifies the integrity of a lot's hash by matching it against its parents.
    Requires ledger access to resolve parent_lot_ids.
    """
    lot = ledger.lot_store.get(lot_id)
    if not lot:
        return False
    
    if not lot.parent_lot_ids:
        expected = generate_hash([], lot.created_by_txn_id, lot.amount, lot.last_move_ts)
        return lot.hash == expected

    parent_hashes = []
    for pid in lot.parent_lot_ids:
        p = ledger.lot_store.get(pid)
        if not p:
            # Cannot definitively verify if parent is purged, return False for strict integrity
            return False
        parent_hashes.append(p.hash)

    expected = generate_hash(parent_hashes, lot.created_by_txn_id, lot.amount, lot.last_move_ts)
    return lot.hash == expected