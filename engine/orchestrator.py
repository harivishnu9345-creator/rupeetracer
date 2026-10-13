"""Decision orchestrator: (risk_score, cluster_risk, segment) -> ALLOW / STEP_UP / HOLD / BLOCK / ESCALATE.

* Base band from config.ACTION_BANDS (defaults: STEP_UP >= 0.30, HOLD >= 0.55, BLOCK >= 0.80; lower bound inclusive).
* ESCALATE overrides the band when cluster_risk >= 0.70 (network-level case for the investigations team).
* Per-segment adaptation: the analyst-feedback loop (cases.py) stores a learned offset per segment
  ("<account_type>|kyc<tier>").  The offset is added to the risk score before banding and is ALWAYS
  clamped to +-OFFSET_LIMIT, so thresholds can drift only a bounded amount.
  Positive offset (confirmed fraud) = more sensitive segment; negative (false positives) = less sensitive.
* decide() returns the Decision dict used by engine.py / api.py / cases.py.
"""
from __future__ import annotations

import logging
import math

try:  # config is optional so this module can be unit-tested in isolation
    import config
except ImportError:  # pragma: no cover
    config = None

log = logging.getLogger("rupeetrace")

ACTION_ORDER = ["ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"]  # increasing severity
OPENS_CASE = ("HOLD", "BLOCK", "ESCALATE")

_DEFAULT_FLOORS = {"STEP_UP": 0.30, "HOLD": 0.55, "BLOCK": 0.80}


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _unit(v) -> float:
    """Sanitise to [0, 1]; NaN / None / garbage -> 0.0."""
    x = _num(v)
    return 0.0 if x is None else min(1.0, max(0.0, x))


def _parse_bands(raw) -> dict | None:
    """Accept {action: lower_bound}, {action: (lo, hi)} or [(action, lo, hi) | (lo, hi, action), ...]."""
    floors: dict[str, float] = {}
    try:
        entries = raw.items() if isinstance(raw, dict) else [(None, e) for e in raw]
        for key, val in entries:
            parts = [key, *(val if isinstance(val, (list, tuple)) else [val])]
            name = next((p for p in parts if isinstance(p, str) and p.upper() in ACTION_ORDER), None)
            nums = [n for n in (_num(p) for p in parts if not isinstance(p, str)) if n is not None]
            if name and nums:
                floors[name.upper()] = nums[0]
    except TypeError:
        return None
    if not all(k in floors for k in _DEFAULT_FLOORS):
        return None
    out = {k: floors[k] for k in _DEFAULT_FLOORS}
    vals = list(out.values())
    if not (0.0 < vals[0] < vals[1] < vals[2] <= 1.0):
        return None
    return out


def _load_floors() -> dict:
    raw = getattr(config, "ACTION_BANDS", None)
    if raw is not None:
        parsed = _parse_bands(raw)
        if parsed:
            return parsed
        log.warning("config.ACTION_BANDS not understood (%r); using defaults %s", raw, _DEFAULT_FLOORS)
    return dict(_DEFAULT_FLOORS)


BAND_FLOORS = _load_floors()
ESCALATE_CLUSTER_RISK = float(getattr(config, "CLUSTER_ESCALATE_THRESHOLD", 0.70))
ESCALATE_MIN_RISK = float(getattr(config, "ESCALATE_MIN_RISK", 0.0))  # optional floor on the txn's own risk
OFFSET_LIMIT = float(getattr(config, "OFFSET_LIMIT", 0.10))

INTERVENTION = {
    "ALLOW": "No action needed. The payment goes through normally.",
    "STEP_UP": "Ask the customer for extra verification (OTP or biometric) before the payment completes.",
    "HOLD": "Hold the payment and send it to an analyst. Contact the account holder to confirm before releasing it.",
    "BLOCK": "Block the payment and temporarily restrict the account. Contact the customer on a verified channel.",
    "ESCALATE": "Send the linked accounts to the fraud-investigation team as a possible mule network. "
                "Keep the funds held while they review.",
}
SLA_MINUTES = {"ALLOW": 0, "STEP_UP": 5, "HOLD": 60, "BLOCK": 15, "ESCALATE": 30}


def band_for(score) -> str:
    """Base band (no offsets, no escalation) for a risk score. Lower bound of each band is inclusive."""
    s = _unit(score)
    if s >= BAND_FLOORS["BLOCK"]:
        return "BLOCK"
    if s >= BAND_FLOORS["HOLD"]:
        return "HOLD"
    if s >= BAND_FLOORS["STEP_UP"]:
        return "STEP_UP"
    return "ALLOW"


def segment_key(account_type, kyc_tier) -> str:
    """Segment label shared with cases.py (its learned offsets are keyed by this string)."""
    at = "unknown" if account_type is None or (isinstance(account_type, float) and math.isnan(account_type)) else str(account_type)
    k = _num(kyc_tier)
    kt = "unknown" if k is None else (str(int(k)) if k == int(k) else str(k))
    return f"{at}|kyc{kt}"


def clamp_offset(v) -> float:
    x = _num(v)
    return 0.0 if x is None else max(-OFFSET_LIMIT, min(OFFSET_LIMIT, x))


def decide(risk_score, cluster_risk=0.0, account_type=None, kyc_tier=None, reasons=None,
           learned_offsets: dict | None = None, watchlist_hit: bool = False) -> dict:
    base = _unit(risk_score)
    cluster = _unit(cluster_risk)
    seg = segment_key(account_type, kyc_tier)
    offset = clamp_offset((learned_offsets or {}).get(seg, 0.0))
    adjusted = base if offset == 0 else min(1.0, max(0.0, round(base + offset, 9)))  # rounding kills float jitter at band edges

    action = band_for(adjusted)
    escalated = cluster >= ESCALATE_CLUSTER_RISK and adjusted >= ESCALATE_MIN_RISK
    if escalated:
        action = "ESCALATE"

    out_reasons = list(reasons or [])
    if watchlist_hit and not any("watchlist" in r.lower() for r in out_reasons):
        out_reasons.append("An account in this payment is on the confirmed-fraud watchlist")
    if escalated:
        out_reasons.append(f"Linked accounts form a high-risk cluster (cluster risk {cluster:.2f})")
    if action != "ALLOW" and not out_reasons:
        out_reasons.append("Overall risk score is above the alert threshold")

    return {
        "action": action,
        "risk_score": adjusted,
        "base_risk_score": round(base, 6),
        "base_action": band_for(base),
        "cluster_risk": round(cluster, 6),
        "reasons": out_reasons,
        "recommended_intervention": INTERVENTION[action],
        "sla_minutes": SLA_MINUTES[action],
        "segment": seg,
        "offset_applied": offset,
        "escalated": escalated,
        "opens_case": action in OPENS_CASE,
    }
