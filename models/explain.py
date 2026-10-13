# models/explain.py
"""Translates raw features and scores into human-readable plain language reasons."""
import config

def explain(features: dict, scores: dict) -> list[str]:
    reasons = []
    
    # 1. Threshold/Rule-based
    amt = features.get("amount", 0.0)
    thresh = float(getattr(config, "ALERT_THRESHOLD_INR", 50000))
    if amt >= thresh:
        reasons.append(f"Payment amount (Rs {amt:,.2f}) exceeds regulatory alert threshold.")
    elif (thresh * 0.8) <= amt < thresh:
        reasons.append("Payment amount is suspiciously close to the reporting threshold.")

    # 2. Provenance Explanations
    if scores.get("provenance_score", 0.0) > 0.6:
        root_amt = features.get("prov_max_root_amount", 0.0)
        frags = int(features.get("prov_max_fragments", 0))
        if frags > 5 and root_amt > thresh:
            reasons.append(f"Money is 1 of ~{frags} fragments of a Rs {root_amt:,.2f} root source.")
        if features.get("prov_origin_is_cash", 0.0) > 0:
            reasons.append("Funds trace back to a cash deposit origin.")
        if features.get("prov_dwell_time_mean", 999.0) < 15.0:
            reasons.append("Funds are moving through accounts very rapidly (average dwell < 15 mins).")

    # 3. Graph Explanations
    if scores.get("graph_score", 0.0) > 0.6:
        shared_dev = int(features.get("graph_src_shared_device_links", 0))
        if shared_dev > 0:
            reasons.append(f"Sender shares a device/IP with {shared_dev} other active accounts.")
        if features.get("graph_src_community_density", 0.0) > 0.4:
            reasons.append("Account belongs to a highly dense network cluster indicative of coordination.")

    # 4. Behavior Explanations
    if scores.get("behavior_score", 0.0) > 0.6:
        if features.get("beh_new_device", 0.0) > 0:
            reasons.append("Account is transacting from a previously unseen device.")
        if features.get("beh_first_time_counterparty", 0.0) > 0:
            reasons.append("Payment directed to a first-time counterparty.")
        if features.get("beh_passthrough_ratio", 0.0) > 0.8:
            reasons.append("High pass-through ratio: funds are immediately transferred out.")

    if not reasons:
        reasons.append("Elevated risk scores across combined AI models.")
        
    return reasons