"""Column contracts, row dataclasses and validators for RupeeTrace CSVs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

import config as C

ACCOUNT_COLUMNS = [
    "account_id", "created_at", "kyc_tier", "account_type", "city",
    "primary_device_id", "primary_ip_prefix", "phone_hash", "role_gt",
]
TXN_COLUMNS = [
    "txn_id", "ts", "src_account", "dst_account", "amount", "channel", "origin_type",
    "device_id", "ip", "city", "remark", "label", "typology", "scenario_id", "root_txn_id_gt",
]
SCENARIO_COLUMNS = [
    "scenario_id", "typology", "root_txn_id", "total_amount", "num_fragments",
    "num_accounts", "start_ts", "end_ts",
]
PREDICTION_COLUMNS = ["txn_id", "risk_score", "action", "latency_ms", "reasons"]

ACCOUNT_TYPES = {"individual", "merchant", "business"}
ROLES = {"normal", "origin", "relay", "collector", "mule", "ato_victim"}
CHANNELS = {"UPI", "IMPS", "NEFT", "CASH_DEP", "CASH_WD"}
ORIGIN_TYPES = {"SALARY", "CASH", "LOAN", "BUSINESS", "UNKNOWN", "TRANSFER"}
TYPOLOGIES = {
    "none", "account_takeover", "velocity_abuse", "mule_passthrough",
    "circular", "fan_in", "fan_out", "fragmentation",
}
LAUNDERING_TYPOLOGIES = {"mule_passthrough", "circular", "fan_in", "fan_out", "fragmentation"}
ACTION_TOKENS = {"ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"}
LEAKAGE_COLUMNS = ["label", "typology", "scenario_id", "root_txn_id_gt", "role_gt"]

_TXN_RE = re.compile(r"^T\d{7}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class SchemaError(ValueError):
    pass


# ------------------------------------------------------------ row dataclasses
@dataclass(frozen=True)
class Account:
    account_id: str
    created_at: str
    kyc_tier: int
    account_type: str
    city: str
    primary_device_id: str
    primary_ip_prefix: str
    phone_hash: str
    role_gt: str = "normal"


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    ts: str
    src_account: str
    dst_account: str
    amount: float
    channel: str
    origin_type: str
    device_id: str
    ip: str
    city: str
    remark: str
    label: int = 0
    typology: str = "none"
    scenario_id: str = ""
    root_txn_id_gt: str = ""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    typology: str
    root_txn_id: str
    total_amount: float
    num_fragments: int
    num_accounts: int
    start_ts: str
    end_ts: str


@dataclass(frozen=True)
class Prediction:
    txn_id: str
    risk_score: float
    action: str
    latency_ms: float
    reasons: str  # JSON list of strings


# ------------------------------------------------------------------ loaders
def _read(path, dtype=None) -> pd.DataFrame:
    # keep_default_na=False so blank strings stay "" (never NaN)
    return pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[])


def load_accounts(path: Optional[Path] = None) -> pd.DataFrame:
    return _read(path or C.DATA_OUT / "accounts.csv")


def load_transactions(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read(path or C.DATA_OUT / "transactions.csv")
    df["label"] = df["label"].astype(int)
    df["amount"] = df["amount"].astype(float)
    return df


def load_scenarios(path: Optional[Path] = None) -> pd.DataFrame:
    return _read(path or C.DATA_OUT / "scenarios.csv")


def load_predictions(path) -> pd.DataFrame:
    return _read(path)


# --------------------------------------------------------------- validators
def _cols(df, expected, name):
    if list(df.columns) != expected:
        raise SchemaError(f"{name}: columns {list(df.columns)} != contract {expected}")


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str) == "")


def _in(s: pd.Series, allowed: set, name: str):
    bad = set(s.unique()) - allowed
    if bad:
        raise SchemaError(f"{name}: invalid values {sorted(bad)[:5]}")


def validate_accounts(df: pd.DataFrame) -> None:
    _cols(df, ACCOUNT_COLUMNS, "accounts")
    if df["account_id"].duplicated().any():
        raise SchemaError("accounts: duplicate account_id")
    _in(df["account_type"], ACCOUNT_TYPES, "accounts.account_type")
    _in(df["role_gt"], ROLES, "accounts.role_gt")
    if not df["kyc_tier"].isin([1, 2, 3]).all():
        raise SchemaError("accounts: kyc_tier must be 1-3")
    if not df["created_at"].astype(str).str.match(_ISO_RE).all():
        raise SchemaError("accounts: created_at must be ISO UTC")
    if df.drop(columns=[]).isna().any().any():
        raise SchemaError("accounts: nulls present")


def validate_transactions(df: pd.DataFrame, accounts: Optional[pd.DataFrame] = None) -> None:
    _cols(df, TXN_COLUMNS, "transactions")
    n = len(df)
    if n == 0:
        raise SchemaError("transactions: empty")
    if not df["txn_id"].astype(str).str.match(_TXN_RE).all():
        raise SchemaError("transactions: txn_id must match T0000001 pattern")
    if df["txn_id"].duplicated().any():
        raise SchemaError("transactions: duplicate txn_id")
    if not df["ts"].astype(str).str.match(_ISO_RE).all():
        raise SchemaError("transactions: ts must be ISO UTC 'YYYY-MM-DDTHH:MM:SSZ'")
    if not df["ts"].is_monotonic_increasing:
        raise SchemaError("transactions: must be sorted by ts")
    if df["ts"].iloc[0] < C.SIM_START or df["ts"].iloc[-1] >= C.SIM_END:
        raise SchemaError("transactions: ts outside simulation window")
    if not (pd.to_numeric(df["amount"]) > 0).all():
        raise SchemaError("transactions: amount must be > 0")
    _in(df["channel"], CHANNELS, "transactions.channel")
    _in(df["origin_type"], ORIGIN_TYPES, "transactions.origin_type")
    _in(df["typology"], TYPOLOGIES, "transactions.typology")
    if not df["label"].isin([0, 1]).all():
        raise SchemaError("transactions: label must be 0/1")
    if (df["src_account"] == df["dst_account"]).any():
        raise SchemaError("transactions: src == dst")
    ext_src = df["src_account"].isin(C.EXTERNAL_ACCOUNTS)
    if ((df["origin_type"] != "TRANSFER") != ext_src).any():
        raise SchemaError("transactions: origin_type != TRANSFER iff src is external")
    if df["dst_account"].isin(C.EXTERNAL_ACCOUNTS).any() and \
            not (df.loc[df["dst_account"].isin(C.EXTERNAL_ACCOUNTS), "dst_account"] == "EXT_CASH_OUT").all():
        raise SchemaError("transactions: only EXT_CASH_OUT may be an external destination")
    pos = df["label"] == 1
    if ((df["typology"] != "none") != pos).any():
        raise SchemaError("transactions: label==1 iff typology != none")
    if (_blank(df["scenario_id"]) == pos).any():
        raise SchemaError("transactions: scenario_id must be set iff label==1")
    root = ~_blank(df["root_txn_id_gt"])
    if (root & ~pos).any():
        raise SchemaError("transactions: root_txn_id_gt only allowed on label==1")
    if root.any() and not df.loc[root, "root_txn_id_gt"].isin(set(df["txn_id"])).all():
        raise SchemaError("transactions: root_txn_id_gt references unknown txn_id")
    if accounts is not None:
        known = set(accounts["account_id"]) | set(C.EXTERNAL_ACCOUNTS)
        if not (df["src_account"].isin(known).all() and df["dst_account"].isin(known).all()):
            raise SchemaError("transactions: unknown account referenced")


def validate_scenarios(df: pd.DataFrame, txns: Optional[pd.DataFrame] = None) -> None:
    _cols(df, SCENARIO_COLUMNS, "scenarios")
    if df["scenario_id"].duplicated().any():
        raise SchemaError("scenarios: duplicate scenario_id")
    _in(df["typology"], TYPOLOGIES - {"none"}, "scenarios.typology")
    if txns is not None:
        if set(df["scenario_id"]) != set(txns.loc[txns["label"] == 1, "scenario_id"]):
            raise SchemaError("scenarios: ids do not match transactions.scenario_id")
        if not df["root_txn_id"].isin(set(txns["txn_id"])).all():
            raise SchemaError("scenarios: root_txn_id unknown")


def validate_predictions(df: pd.DataFrame, expected_ids: Optional[set] = None) -> None:
    _cols(df, PREDICTION_COLUMNS, "predictions")
    if df["txn_id"].duplicated().any():
        raise SchemaError("predictions: duplicate txn_id")
    s = pd.to_numeric(df["risk_score"])
    if s.isna().any() or ((s < 0) | (s > 1)).any():
        raise SchemaError("predictions: risk_score must be in [0, 1]")
    lat = pd.to_numeric(df["latency_ms"])
    if lat.isna().any() or (lat < 0).any():
        raise SchemaError("predictions: latency_ms must be >= 0")
    for a in df["action"].unique():
        if not C.action_tokens(a) or not C.action_tokens(a) <= ACTION_TOKENS:
            raise SchemaError(f"predictions: invalid action {a!r}")
    for r in df["reasons"].unique():
        try:
            v = json.loads(r)
        except Exception as e:  # noqa: BLE001
            raise SchemaError(f"predictions: reasons not JSON: {r!r}") from e
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise SchemaError("predictions: reasons must be a JSON list of strings")
    if expected_ids is not None:
        ids = set(df["txn_id"])
        if ids - expected_ids:
            raise SchemaError("predictions: contains txn_ids outside the test split")
