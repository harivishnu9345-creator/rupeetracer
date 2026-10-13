"""Behavior and anomaly feature extractor (Strictly Causal & Stateful)."""
import datetime
import numpy as np

FEATURE_NAMES = [
    "beh_src_count_1h", "beh_src_amount_1h",
    "beh_src_count_24h", "beh_src_amount_24h",
    "beh_src_count_72h", "beh_src_amount_72h",
    "beh_src_amount_zscore",
    "beh_first_time_counterparty",
    "beh_new_device", "beh_new_ip", "beh_new_city",
    "beh_fan_in_72h", "beh_fan_out_72h",
    "beh_round_amount", "beh_just_below_threshold",
    "beh_passthrough_ratio", "beh_remark_risk_flag",
    "beh_time_of_day_dev"
]

class BehaviorExtractor:
    def __init__(self, alert_threshold=50000):
        self.alert_threshold = alert_threshold
        self.history = {}
        self.counterparties = {}
        self.devices = {}
        self.ips = {}
        self.cities = {}
        self.running_stats = {}

    def _parse_ts(self, ts_str) -> float:
        if not isinstance(ts_str, str):
            ts_str = str(ts_str) if ts_str is not None else ""
        try:
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            return datetime.datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            return 0.0
            
    def _hour_of_day(self, ts_str) -> int:
        if not isinstance(ts_str, str):
            ts_str = str(ts_str) if ts_str is not None else ""
        try:
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            return datetime.datetime.fromisoformat(ts_str).hour
        except Exception:
            return 12

    def extract_and_update(self, txn: dict) -> dict:
        src = str(txn.get('src_account', ''))
        dst = str(txn.get('dst_account', ''))
        amt = float(txn.get('amount', 0.0))
        ts_sec = self._parse_ts(txn.get('ts', ''))
        hour = self._hour_of_day(txn.get('ts', ''))
        
        device = str(txn.get('device_id', ''))
        ip = str(txn.get('ip', ''))
        city = str(txn.get('city', ''))
        remark = str(txn.get('remark', '')).lower()

        feats = {k: 0.0 for k in FEATURE_NAMES}

        # Initialize missing state for strict causality
        for acc in (src, dst):
            if acc not in self.history:
                self.history[acc] = []
                self.counterparties[acc] = set()
                self.devices[acc] = set()
                self.ips[acc] = set()
                self.cities[acc] = set()
                self.running_stats[acc] = {'sum': 0.0, 'sq_sum': 0.0, 'count': 0}

        # --- 1. STRICTLY CAUSAL EXTRACTION ---
        cutoff_1h = ts_sec - 3600
        cutoff_24h = ts_sec - 86400
        cutoff_72h = ts_sec - 72 * 3600

        self.history[src] = [h for h in self.history[src] if h[0] >= cutoff_72h]
        
        c_1h, a_1h, c_24h, a_24h, c_72h, a_72h = 0, 0.0, 0, 0.0, 0, 0.0
        fan_in_72h, fan_in_amount_72h, fan_out_amount_72h = 0, 0.0, 0.0

        for h_ts, h_amt, is_out in self.history[src]:
            if is_out:
                c_72h += 1
                a_72h += h_amt
                fan_out_amount_72h += h_amt
                if h_ts >= cutoff_24h:
                    c_24h += 1
                    a_24h += h_amt
                if h_ts >= cutoff_1h:
                    c_1h += 1
                    a_1h += h_amt
            else:
                fan_in_72h += 1
                fan_in_amount_72h += h_amt

        feats["beh_src_count_1h"] = float(c_1h)
        feats["beh_src_amount_1h"] = float(a_1h)
        feats["beh_src_count_24h"] = float(c_24h)
        feats["beh_src_amount_24h"] = float(a_24h)
        feats["beh_src_count_72h"] = float(c_72h)
        feats["beh_src_amount_72h"] = float(a_72h)
        feats["beh_fan_in_72h"] = float(fan_in_72h)
        feats["beh_fan_out_72h"] = float(c_72h)

        if fan_in_amount_72h > 0:
            feats["beh_passthrough_ratio"] = float(fan_out_amount_72h / fan_in_amount_72h)

        stats = self.running_stats[src]
        if stats['count'] > 1:
            mean = stats['sum'] / stats['count']
            var = (stats['sq_sum'] / stats['count']) - (mean ** 2)
            std = np.sqrt(max(0.0, var))
            if std > 1e-6:
                feats["beh_src_amount_zscore"] = float((amt - mean) / std)

        if dst and dst not in self.counterparties[src]:
            feats["beh_first_time_counterparty"] = 1.0
        if device and device not in self.devices[src] and len(self.devices[src]) > 0:
            feats["beh_new_device"] = 1.0
        if ip and ip not in self.ips[src] and len(self.ips[src]) > 0:
            feats["beh_new_ip"] = 1.0
        if city and city not in self.cities[src] and len(self.cities[src]) > 0:
            feats["beh_new_city"] = 1.0
        if amt % 100 == 0:
            feats["beh_round_amount"] = 1.0
        if (self.alert_threshold * 0.8) <= amt < self.alert_threshold:
            feats["beh_just_below_threshold"] = 1.0

        if 1 <= hour <= 5:
            feats["beh_time_of_day_dev"] = 1.0

        risk_words = ['urgent', 'loan', 'investment', 'prize', 'winner', 'crypto', 'bonus']
        if any(w in remark for w in risk_words):
            feats["beh_remark_risk_flag"] = 1.0

        # --- 2. CAUSAL STATE UPDATE ---
        self.history[src].append((ts_sec, amt, True))
        self.history[dst].append((ts_sec, amt, False))
        self.counterparties[src].add(dst)
        self.counterparties[dst].add(src)
        
        if device: self.devices[src].add(device)
        if ip: self.ips[src].add(ip)
        if city: self.cities[src].add(city)

        self.running_stats[src]['sum'] += amt
        self.running_stats[src]['sq_sum'] += (amt ** 2)
        self.running_stats[src]['count'] += 1

        for k, v in feats.items():
            if not np.isfinite(v):
                feats[k] = 0.0

        return feats