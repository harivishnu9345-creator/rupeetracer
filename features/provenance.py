"""Provenance feature extractor (money lots lineage)."""
import numpy as np

FEATURE_NAMES = [
    "prov_max_root_amount",
    "prov_max_hops",
    "prov_max_fragments",
    "prov_sub_threshold_fragments_ratio",
    "prov_fragment_root_share",
    "prov_dwell_time_min",
    "prov_dwell_time_mean",
    "prov_distinct_path_accounts",
    "prov_origin_is_cash",
    "prov_origin_is_unknown",
    "prov_just_below_threshold",
    "prov_hash_chain_ok"
]

def extract_features(txn: dict, transferred_lots: list, root_stats: dict, accounts_info: dict, config) -> dict:
    """
    Derives features from the lineage of money being transferred.
    Matches the exact signature called by ledger/lots.py.
    """
    features = {k: 0.0 for k in FEATURE_NAMES}
    if not transferred_lots:
        return features

    # Default to 1.0; hash chain validation requires full ledger access.
    features["prov_hash_chain_ok"] = 1.0

    max_root_amt = 0.0
    max_hops = 0
    max_frags = 0
    sub_thresh_frags = 0
    total_frags = 0
    total_lot_amt = 0.0
    dwell_times = []
    path_accounts = set()
    origin_cash = 0.0
    origin_unknown = 0.0

    # Handle missing timestamps gracefully
    ts_sec = float(txn.get('_ts_sec', 0.0))
    threshold = float(getattr(config, 'ALERT_THRESHOLD_INR', 50000.0))

    for lot in transferred_lots:
        r_stat = root_stats.get(lot.root_txn_id, {})
        r_amt = float(r_stat.get('aggregate_amount', 0.0))
        frags = r_stat.get('fragments', [])
        
        if r_amt > max_root_amt:
            max_root_amt = r_amt
        if lot.hop_count > max_hops:
            max_hops = lot.hop_count
            
        frag_count = len(frags)
        if frag_count > max_frags:
            max_frags = frag_count
            
        total_frags += frag_count
        sub_thresh_frags += sum(1 for f in frags if f < threshold)
        total_lot_amt += float(lot.amount)

        # Dwell time since the lot last moved
        dwell = max(0.0, ts_sec - float(lot.last_move_ts))
        dwell_times.append(dwell / 60.0)  # minutes
        
        if lot.account_path:
            path_accounts.update(lot.account_path)

        if lot.origin_type == "CASH":
            origin_cash = 1.0
        elif lot.origin_type == "UNKNOWN":
            origin_unknown = 1.0

    features["prov_max_root_amount"] = float(max_root_amt)
    features["prov_max_hops"] = float(max_hops)
    features["prov_max_fragments"] = float(max_frags)
    
    if total_frags > 0:
        features["prov_sub_threshold_fragments_ratio"] = float(sub_thresh_frags) / total_frags
    if max_root_amt > 0:
        features["prov_fragment_root_share"] = total_lot_amt / max_root_amt

    if dwell_times:
        features["prov_dwell_time_min"] = float(np.min(dwell_times))
        features["prov_dwell_time_mean"] = float(np.mean(dwell_times))

    features["prov_distinct_path_accounts"] = float(len(path_accounts))
    features["prov_origin_is_cash"] = float(origin_cash)
    features["prov_origin_is_unknown"] = float(origin_unknown)

    amt = float(txn.get('amount', 0.0))
    if (threshold * 0.8) <= amt < threshold:
        features["prov_just_below_threshold"] = 1.0

    # Fallback for unexpected NaNs to maintain contract
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features