"""RupeeTrace shared configuration. THIS FILE IS PART OF THE SHARED CONTRACT.

Constants below are fixed by the contract. The small helper functions at the
bottom (action_for_score, is_flag_action, bucket_index, ...) are additive
conveniences and do not change any contract constant.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ----------------------------------------------------------------- contract
SEED = 42
ALERT_THRESHOLD_INR = 50_000
HOP_CAP = 8
WINDOW_HOURS = 72
N_ACCOUNTS = 5000
SIM_DAYS = 30
TRAIN_DAYS = 21
SIM_START = "2026-01-01T00:00:00Z"

_ISO = "%Y-%m-%dT%H:%M:%SZ"
_SIM_START_DT = datetime.strptime(SIM_START, _ISO).replace(tzinfo=timezone.utc)
SPLIT_TS = (_SIM_START_DT + timedelta(days=TRAIN_DAYS)).strftime(_ISO)  # 2026-01-22T00:00:00Z
SIM_END = (_SIM_START_DT + timedelta(days=SIM_DAYS)).strftime(_ISO)

AMOUNT_BUCKETS = [(0, 5000), (5000, 50000), (50000, math.inf)]

# (low inclusive, high exclusive, action). risk_score in [0, 1].
ACTION_BANDS = [
    (0.00, 0.30, "ALLOW"),
    (0.30, 0.55, "STEP_UP"),
    (0.55, 0.80, "HOLD"),
    (0.80, 1.0000001, "BLOCK"),
]
ESCALATE_CLUSTER_RISK = 0.70  # ESCALATE added (opens analyst case) when cluster_risk >= this

EXTERNAL_ACCOUNTS = [
    "EXT_CASH", "EXT_EMPLOYER", "EXT_LOAN", "EXT_BUSINESS", "EXT_UNKNOWN", "EXT_CASH_OUT",
]

# ------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_OUT = ROOT / "data" / "out"
EVAL_OUT = ROOT / "eval" / "out"
PREDICTION_MODELS = ["A_rules", "B_behavior_anomaly", "C_plus_graph", "D_plus_provenance"]

# ---------------------------------------------------------------- helpers
FLAG_ACTIONS = ("HOLD", "BLOCK", "ESCALATE")          # "flag" = alert-level action
FRICTION_ACTIONS = ("STEP_UP", "HOLD", "BLOCK")       # customer-visible friction
_ACTION_SPLIT = re.compile(r"[+,|;/\s]+")


def action_for_score(score: float) -> str:
    """Map risk_score to its band action (ESCALATE is added separately)."""
    for lo, hi, action in ACTION_BANDS:
        if lo <= score < hi:
            return action
    return "BLOCK" if score >= 1.0 else "ALLOW"


def action_tokens(action: str) -> set:
    """An action cell may be one token ('HOLD') or several joined with '+'
    ('HOLD+ESCALATE'). Return the set of tokens."""
    return {t for t in _ACTION_SPLIT.split(str(action).upper()) if t}


def is_flag_action(action: str) -> bool:
    return bool(action_tokens(action) & set(FLAG_ACTIONS))


def is_friction_action(action: str) -> bool:
    return bool(action_tokens(action) & set(FRICTION_ACTIONS))


def bucket_index(amount: float) -> int:
    for i, (lo, hi) in enumerate(AMOUNT_BUCKETS):
        if lo <= amount < hi:
            return i
    return len(AMOUNT_BUCKETS) - 1


def bucket_label(i: int) -> str:
    lo, hi = AMOUNT_BUCKETS[i]
    fmt = lambda v: "inf" if v == math.inf else (f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    return f"Rs {fmt(lo)}-{fmt(hi)}" if hi != math.inf else f"Rs {fmt(lo)}+"
