import datetime
from collections import defaultdict
from ledger.hashchain import generate_hash, verify_chain
from features.provenance import extract_features, FEATURE_NAMES

class Lot:
    __slots__ = ['lot_id', 'root_txn_id', 'created_by_txn_id', 'origin_type', 'amount', 
                 'holder', 'parent_lot_ids', 'hop_count', 'account_path', 'created_ts', 
                 'last_move_ts', 'hash']

    def __init__(self, lot_id, root_txn_id, created_by_txn_id, origin_type, amount, holder,
                 parent_lot_ids, hop_count, account_path, created_ts, last_move_ts, hash_val):
        self.lot_id = lot_id
        self.root_txn_id = root_txn_id
        self.created_by_txn_id = created_by_txn_id
        self.origin_type = origin_type
        self.amount = amount
        self.holder = holder
        self.parent_lot_ids = parent_lot_ids
        self.hop_count = hop_count
        self.account_path = account_path
        self.created_ts = created_ts
        self.last_move_ts = last_move_ts
        self.hash = hash_val

class LotLedger:
    def __init__(self, config, accounts_info=None):
        self.config = config
        self.accounts_info = accounts_info or {}
        
        self.accounts = defaultdict(list)
        self.lot_store = {}
        self.txn_lots = defaultdict(list)
        
        self.root_stats = defaultdict(lambda: {
            'aggregate_amount': 0.0,
            'fragments': [],
            'destinations': set()
        })
        self.lot_counter = 0

    def _next_lot_id(self):
        self.lot_counter += 1
        return f"L{self.lot_counter}"

    def _parse_ts(self, ts_str):
        try:
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            return datetime.datetime.fromisoformat(ts_str).timestamp()
        except:
            return 0.0

    def process(self, txn: dict) -> dict:
        src = txn['src_account']
        dst = txn['dst_account']
        amt = float(txn['amount'])
        ts_sec = self._parse_ts(txn['ts'])
        txn['_ts_sec'] = ts_sec

        src_lots = self.accounts[src]
        
        # Handle external sources
        if src.startswith('EXT_'):
            root_lot = Lot(
                lot_id=self._next_lot_id(),
                root_txn_id=txn['txn_id'],
                created_by_txn_id=txn['txn_id'],
                origin_type=txn.get('origin_type', 'UNKNOWN'),
                amount=amt,
                holder=src,
                parent_lot_ids=[],
                hop_count=0,
                account_path=[src],
                created_ts=ts_sec,
                last_move_ts=ts_sec,
                hash_val=generate_hash([], txn['txn_id'], amt, ts_sec)
            )
            src_lots = [root_lot]
            self.root_stats[txn['txn_id']]['aggregate_amount'] += amt

        total_balance = sum(l.amount for l in src_lots)

        # Handle shortfall by injecting UNKNOWN origin lot
        if not src.startswith('EXT_') and amt > total_balance:
            shortfall = amt - total_balance
            shortfall_lot = Lot(
                lot_id=self._next_lot_id(),
                root_txn_id=txn['txn_id'],
                created_by_txn_id=txn['txn_id'],
                origin_type='UNKNOWN',
                amount=shortfall,
                holder=src,
                parent_lot_ids=[],
                hop_count=0,
                account_path=[src],
                created_ts=ts_sec,
                last_move_ts=ts_sec,
                hash_val=generate_hash([], txn['txn_id'], shortfall, ts_sec)
            )
            src_lots.append(shortfall_lot)
            self.lot_store[shortfall_lot.lot_id] = shortfall_lot
            total_balance = amt

        # Compute proportional splits
        transferred_lots = []
        commit_actions = []
        remaining_amt = amt

        for l in src_lots:
            if remaining_amt <= 0: break
            if l.amount <= 0: continue

            send_amt = l.amount * (amt / total_balance)
            send_amt = min(send_amt, l.amount, remaining_amt)
            if send_amt < 1e-6: continue

            new_hop = l.hop_count + 1
            new_path = list(l.account_path)
            if not new_path or new_path[-1] != dst:
                new_path.append(dst)

            if new_hop > self.config.HOP_CAP:
                new_path = [] # Enforce hop cap by truncating path

            child_hash = generate_hash([l.hash], txn['txn_id'], send_amt, ts_sec)
            
            child = Lot(
                lot_id=self._next_lot_id(),
                root_txn_id=l.root_txn_id,
                created_by_txn_id=txn['txn_id'],
                origin_type=l.origin_type,
                amount=send_amt,
                holder=dst,
                parent_lot_ids=[l.lot_id],
                hop_count=new_hop,
                account_path=new_path,
                created_ts=l.created_ts,
                last_move_ts=ts_sec,
                hash_val=child_hash
            )
            
            transferred_lots.append(child)
            commit_actions.append((l, send_amt))
            remaining_amt -= send_amt

        # Extract features BEFORE committing state to maintain strict causality
        features = extract_features(txn, transferred_lots, self.root_stats, self.accounts_info, self.config)

        # Commit state mutations
        for l, send_amt in commit_actions:
            l.amount -= send_amt
            self.root_stats[l.root_txn_id]['fragments'].append(send_amt)
            self.root_stats[l.root_txn_id]['destinations'].add(dst)

        if not src.startswith('EXT_'):
            self.accounts[src] = [l for l in src_lots if l.amount > 1e-6]

        if not dst.startswith('EXT_CASH_OUT'):
            for child in transferred_lots:
                self.lot_store[child.lot_id] = child
                self.txn_lots[txn['txn_id']].append(child)
                self.accounts[dst].append(child)
            self._cap_lots(dst)

        return features

    def _cap_lots(self, account):
        lots = self.accounts[account]
        if len(lots) <= 50:
            return

        by_root = defaultdict(list)
        for l in lots:
            by_root[l.root_txn_id].append(l)

        while len(lots) > 50:
            root_to_merge = max(by_root.keys(), key=lambda r: len(by_root[r]))
            if len(by_root[root_to_merge]) < 2:
                break 

            root_lots = sorted(by_root[root_to_merge], key=lambda x: x.amount)
            l1, l2 = root_lots[0], root_lots[1]

            merged_amt = l1.amount + l2.amount
            merged_parents = list(set(l1.parent_lot_ids + l2.parent_lot_ids))
            merged_hash = generate_hash([l1.hash, l2.hash], "MERGE", merged_amt, max(l1.last_move_ts, l2.last_move_ts))

            merged = Lot(
                lot_id=self._next_lot_id(),
                root_txn_id=l1.root_txn_id,
                created_by_txn_id="MERGE",
                origin_type=l1.origin_type,
                amount=merged_amt,
                holder=account,
                parent_lot_ids=merged_parents,
                hop_count=max(l1.hop_count, l2.hop_count),
                account_path=l1.account_path,
                created_ts=min(l1.created_ts, l2.created_ts),
                last_move_ts=max(l1.last_move_ts, l2.last_move_ts),
                hash_val=merged_hash
            )

            lots.remove(l1)
            lots.remove(l2)
            lots.append(merged)
            by_root[root_to_merge].remove(l1)
            by_root[root_to_merge].remove(l2)
            by_root[root_to_merge].append(merged)
            self.lot_store[merged.lot_id] = merged

        self.accounts[account] = lots

    def get_lineage(self, txn_id: str) -> dict:
        child_lots = self.txn_lots.get(txn_id, [])
        if not child_lots:
            return {}

        nodes = []
        dwell_minutes = []
        chain_ok = True

        for l in child_lots:
            nodes.append({
                'lot_id': l.lot_id,
                'account': l.holder,
                'amount': l.amount,
                'ts': l.created_ts,
                'parent_lot_ids': l.parent_lot_ids
            })
            if not verify_chain(l.lot_id, self):
                chain_ok = False
            for pid in l.parent_lot_ids:
                p = self.lot_store.get(pid)
                if p:
                    dwell = max(0.0, l.created_ts - p.last_move_ts) / 60.0
                    dwell_minutes.append(dwell)

        l0 = child_lots[0]
        root_stat = self.root_stats.get(l0.root_txn_id, {})

        return {
            'txn_id': txn_id,
            'root_txn_id': l0.root_txn_id,
            'origin_type': l0.origin_type,
            'root_aggregate_amount': root_stat.get('aggregate_amount', 0.0),
            'fragment_count': len(root_stat.get('fragments', [])),
            'hops': l0.hop_count,
            'account_path': l0.account_path,
            'dwell_minutes': dwell_minutes,
            'hash_chain_ok': chain_ok,
            'nodes': nodes
        }