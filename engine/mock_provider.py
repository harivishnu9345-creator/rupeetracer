import time
import random
from datetime import timedelta
import pandas as pd
import config

class MockLotLedger:
    def __init__(self):
        self.acc_lots = {}
        self.lineages = {}
        self.seq = 0
        self.fanout = {}

    def process(self, txn: dict) -> dict:
        tid = txn["txn_id"]
        src = txn["src_account"]
        dst = txn["dst_account"]
        amt = float(txn["amount"])
        ts = pd.to_datetime(txn["ts"])
        origin = txn.get("origin_type", "TRANSFER")

        if origin != "TRANSFER":
            lot = {
                "id": f"L{self.seq}", "root": tid, "amt": amt,
                "orig": origin, "root_amt": amt, "path": [src],
                "ts": ts, "dwell": []
            }
            self.seq += 1
            self.acc_lots.setdefault(dst, []).append(lot)
            self.fanout[tid] = 1
            feats = {"root_aggregate_amount": amt, "hops": 0, "fragment_count": 1}
            self.lineages[tid] = self._build_lin(tid, origin, amt, 1, [lot], dst, ts)
            return feats

        src_lots = self.acc_lots.get(src, [])
        tot_bal = sum(l["amt"] for l in src_lots)

        if tot_bal < amt:
            deficit = amt - tot_bal
            lot = {
                "id": f"L{self.seq}", "root": f"EXT_{tid}", "amt": deficit,
                "orig": "UNKNOWN", "root_amt": deficit, "path": ["EXT_UNKNOWN", src],
                "ts": ts, "dwell": [0]
            }
            self.seq += 1
            src_lots.append(lot)
            tot_bal = amt

        moved_lots = []
        new_src = []
        for l in src_lots:
            frac = l["amt"] / tot_bal
            move = amt * frac
            if move > 0.01:
                dwell_m = (ts - l["ts"]).total_seconds() / 60.0
                nl = {
                    "id": f"L{self.seq}", "root": l["root"], "amt": move,
                    "orig": l["orig"], "root_amt": l["root_amt"],
                    "path": l["path"] + [src], "ts": ts, "dwell": l["dwell"] + [dwell_m]
                }
                self.seq += 1
                moved_lots.append(nl)
                self.acc_lots.setdefault(dst, []).append(nl)
            rem = l["amt"] - move
            if rem > 0.01:
                l["amt"] = rem
                new_src.append(l)

        self.acc_lots[src] = new_src

        if not moved_lots:
            self.lineages[tid] = self._build_lin(tid, "UNKNOWN", amt, 0, [], dst, ts)
            return {"root_aggregate_amount": 0, "hops": 0, "fragment_count": 0}

        dom = max(moved_lots, key=lambda x: x["amt"])
        self.fanout[dom["root"]] = self.fanout.get(dom["root"], 0) + 1

        feats = {
            "root_aggregate_amount": dom["root_amt"],
            "hops": len(dom["path"]) - 1,
            "fragment_count": self.fanout[dom["root"]]
        }
        self.lineages[tid] = self._build_lin(tid, dom["orig"], dom["root_amt"], self.fanout[dom["root"]], moved_lots, dst, ts)
        return feats

    def _build_lin(self, tid, orig, root_amt, frags, lots, dst, ts):
        if not lots: return {}
        dom = max(lots, key=lambda x: x["amt"])
        return {
            "txn_id": tid,
            "root_txn_id": dom["root"],
            "origin_type": dom["orig"],
            "root_aggregate_amount": dom["root_amt"],
            "fragment_count": frags,
            "hops": len(dom["path"]) - 1,
            "account_path": dom["path"] + [dst],
            "dwell_minutes": dom["dwell"],
            "hash_chain_ok": True,
            "nodes": [{"lot_id": l["id"], "account": dst, "amount": l["amt"], "ts": ts.isoformat(), "parent_lot_ids": []} for l in lots]
        }

    def get_lineage(self, txn_id: str) -> dict:
        return self.lineages.get(txn_id, {})

class MockScorer:
    def __init__(self, ledger: MockLotLedger):
        self.ledger = ledger

    def process(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        feats = self.ledger.process(txn)
        amt = float(txn["amount"])

        b_score, a_score, g_score, p_score, c_risk = 0.1, 0.1, 0.1, 0.1, 0.1
        threshold = getattr(config, "ALERT_THRESHOLD_INR", 50000)

        if amt >= threshold:
            b_score, a_score = 0.8, 0.8

        root_amt = feats.get("root_aggregate_amount", 0)
        hops = feats.get("hops", 0)

        # Showcase Scenario: Fragmented large root
        if root_amt >= 400000 and hops > 0 and amt < threshold:
            p_score = 0.95
            c_risk = 0.85
            b_score, a_score, g_score = 0.15, 0.15, 0.20

        # Hard Negative: Shop fan-in
        if root_amt < 100000 and hops == 1 and 500 <= amt <= 2000:
            b_score, a_score, g_score, p_score = 0.05, 0.05, 0.05, 0.05

        reasons = []
        if p_score > 0.8:
            reasons.append(f"Money is 1 of ~{feats.get('fragment_count', 1)} fragments of one Rs {root_amt:,.0f} {feats.get('origin_type', 'UNKNOWN')} deposit across {hops+1} layers")

        return {
            "behavior_score": b_score,
            "anomaly_score": a_score,
            "graph_score": g_score,
            "provenance_score": p_score,
            "cluster_risk": c_risk,
            "reasons": reasons,
            "features": feats,
            "latency_ms": (time.perf_counter() - t0) * 1000
        }

def build_mock(accounts: pd.DataFrame) -> tuple[MockScorer, MockLotLedger]:
    ledger = MockLotLedger()
    scorer = MockScorer(ledger)
    return scorer, ledger

def generate_dataset(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    split_time = pd.to_datetime(getattr(config, "SPLIT_TS", "2026-01-25T00:00:00Z")).to_pydatetime()
    start_time = split_time - timedelta(days=5)

    accounts = []
    # 200 normal users
    for i in range(1, 201):
        accounts.append({
            "account_id": f"ACC_{i:03d}", "account_type": "retail", "kyc_tier": 2,
            "city": rng.choice(["Mumbai", "Delhi", "Bengaluru"]),
            "primary_device_id": f"DEV_{i:03d}", "primary_ip_prefix": f"192.168.{i%255}",
            "created_at": (start_time - timedelta(days=100)).isoformat()
        })
    # 1 Shop
    accounts.append({
        "account_id": "SHOP_1", "account_type": "business", "kyc_tier": 3,
        "city": "Chennai", "primary_device_id": "DEV_SHOP", "primary_ip_prefix": "10.0.0",
        "created_at": (start_time - timedelta(days=300)).isoformat()
    })
    # Relays & Collectors
    for i in range(1, 35):
        accounts.append({
            "account_id": f"RELAY_{i:02d}", "account_type": "retail", "kyc_tier": 1,
            "city": "Kolkata", "primary_device_id": f"DEV_R_{i:02d}", "primary_ip_prefix": "172.16.0",
            "created_at": (start_time - timedelta(days=10)).isoformat()
        })
    for i in range(1, 3):
        accounts.append({
            "account_id": f"COL_{i}", "account_type": "retail", "kyc_tier": 2,
            "city": "Kolkata", "primary_device_id": f"DEV_C_{i}", "primary_ip_prefix": "172.16.1",
            "created_at": (start_time - timedelta(days=20)).isoformat()
        })

    txns = []
    curr_time = start_time

    def advance(minutes=1):
        nonlocal curr_time
        curr_time += timedelta(minutes=minutes)
        return curr_time.isoformat() + "Z"

    # 1. Normal traffic
    for _ in range(2000):
        src = rng.choice(accounts[:200])["account_id"]
        dst = rng.choice(accounts[:200])["account_id"]
        if src == dst: continue
        txns.append({
            "txn_id": f"TXN_N_{len(txns)}", "ts": advance(rng.randint(1, 5)),
            "src_account": src, "dst_account": dst, "amount": round(rng.uniform(100, 5000), 2),
            "channel": "UPI", "origin_type": "TRANSFER", "device_id": f"DEV_{src}", "ip": "1.1.1.1",
            "city": "Mumbai", "remark": rng.choice(["dinner", "rent", ""]),
            "label": 0, "typology": "normal", "scenario_id": "N1", "root_txn_id_gt": ""
        })

    # 2. Hard negatives
    for _ in range(500):
        src = rng.choice(accounts[:200])["account_id"]
        txns.append({
            "txn_id": f"TXN_S_{len(txns)}", "ts": advance(2),
            "src_account": src, "dst_account": "SHOP_1", "amount": round(rng.uniform(500, 2000), 2),
            "channel": "UPI", "origin_type": "TRANSFER", "device_id": f"DEV_{src}", "ip": "1.1.1.1",
            "city": "Chennai", "remark": "groceries",
            "label": 0, "typology": "fan_in", "scenario_id": "S1", "root_txn_id_gt": ""
        })
    txns.append({
        "txn_id": f"TXN_S_BIG", "ts": advance(5),
        "src_account": "ACC_001", "dst_account": "SHOP_1", "amount": 200000.0,
        "channel": "IMPS", "origin_type": "TRANSFER", "device_id": "DEV_001", "ip": "1.1.1.1",
        "city": "Chennai", "remark": "tuition",
        "label": 0, "typology": "large_payment", "scenario_id": "S2", "root_txn_id_gt": ""
    })

    # 3. Showcase fragmentation (Rs 5,00,000 root)
    root_txn_id = "TXN_ROOT"
    txns.append({
        "txn_id": root_txn_id, "ts": advance(10),
        "src_account": "EXT_CASH", "dst_account": "RELAY_01", "amount": 500000.0,
        "channel": "CASH_DEP", "origin_type": "CASH", "device_id": "UNKNOWN", "ip": "0.0.0.0",
        "city": "Kolkata", "remark": "",
        "label": 1, "typology": "fragmentation", "scenario_id": "SC1", "root_txn_id_gt": root_txn_id
    })
    # L1: R01 splits to R02..R11
    for i in range(2, 12):
        txns.append({
            "txn_id": f"TXN_L1_{i}", "ts": advance(10),
            "src_account": "RELAY_01", "dst_account": f"RELAY_{i:02d}", "amount": 50000.0,
            "channel": "UPI", "origin_type": "TRANSFER", "device_id": "DEV_R_01", "ip": "172.16.0.1",
            "city": "Kolkata", "remark": "",
            "label": 1, "typology": "fragmentation", "scenario_id": "SC1", "root_txn_id_gt": root_txn_id
        })
    # L2: R02..R11 splits to R12..R30
    for src_i in range(2, 12):
        for dst_i in range(12, 31):
            if dst_i % 10 == src_i % 10:
                amt = 4800.0 if src_i == 2 and dst_i == 12 else round(rng.uniform(3000, 9900), 2)
                txns.append({
                    "txn_id": f"TXN_L2_{src_i}_{dst_i}", "ts": advance(5),
                    "src_account": f"RELAY_{src_i:02d}", "dst_account": f"RELAY_{dst_i:02d}", "amount": amt,
                    "channel": "UPI", "origin_type": "TRANSFER", "device_id": f"DEV_R_{src_i:02d}", "ip": "172.16.0.2",
                    "city": "Kolkata", "remark": "",
                    "label": 1, "typology": "fragmentation", "scenario_id": "SC1", "root_txn_id_gt": root_txn_id
                })
    # L3: R12..R30 sweep to C1, C2
    for src_i in range(12, 31):
        col = "COL_1" if src_i % 2 == 0 else "COL_2"
        txns.append({
            "txn_id": f"TXN_L3_{src_i}", "ts": advance(5),
            "src_account": f"RELAY_{src_i:02d}", "dst_account": col, "amount": round(rng.uniform(2000, 9900), 2),
            "channel": "UPI", "origin_type": "TRANSFER", "device_id": f"DEV_R_{src_i:02d}", "ip": "172.16.0.3",
            "city": "Kolkata", "remark": "",
            "label": 1, "typology": "fragmentation", "scenario_id": "SC1", "root_txn_id_gt": root_txn_id
        })

    # 4. Account Takeover / Circular
    txns.append({
        "txn_id": "TXN_ATO_1", "ts": advance(10),
        "src_account": "ACC_005", "dst_account": "ACC_006", "amount": 15000.0,
        "channel": "UPI", "origin_type": "TRANSFER", "device_id": "DEV_HACKER", "ip": "8.8.8.8",
        "city": "Foreign", "remark": "",
        "label": 1, "typology": "ato", "scenario_id": "SC2", "root_txn_id_gt": ""
    })

    return pd.DataFrame(accounts), pd.DataFrame(txns).sort_values("ts")