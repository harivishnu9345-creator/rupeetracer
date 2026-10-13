"""Stretch: convert the IBM AMLworld synthetic AML dataset (HI-Small) to the RupeeTrace contract.

Source: "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models"
(Altman et al., NeurIPS 2023 Datasets & Benchmarks), published on Kaggle as
"IBM Transactions for Anti-Money Laundering (AML)" under a Community Data License
Agreement (the NeurIPS supplement says CDLA; confirm the exact variant and its
sharing terms on the Kaggle page before redistributing converted data).

Schema this loader expects (HI-Small_Trans.csv; the second 'Account' header is read by
pandas as 'Account.1'):
  Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency,
  Amount Paid, Payment Currency, Payment Format, Is Laundering
Optional HI-Small_Patterns.txt gives typology blocks:
  BEGIN LAUNDERING ATTEMPT - FAN-OUT: ... / <csv lines in the same layout> / END LAUNDERING ATTEMPT - FAN-OUT

Run:  python -m data.ibm_loader --trans HI-Small_Trans.csv [--patterns HI-Small_Patterns.txt] [--out data/out/ibm]
Output goes to data/out/ibm/ by default so the main synthetic CSVs are never overwritten.

Documented conversions / limits
  * time is linearly rescaled so the dataset span becomes SIM_DAYS (IBM spans 10 days for HI-Small),
    which keeps the 70/30 train/test split at SPLIT_TS; dwell times are therefore stretched ~3x
  * amounts use 'Amount Received' converted with a static, approximate FX table (not financial-grade)
  * Payment Format -> channel (ACH/Cheque/Reinvestment->NEFT, Wire/Bitcoin/Cash->IMPS, Credit Card->UPI);
    the original format is kept in `remark`
  * IBM has no device/ip/city: each account gets a constant placeholder, so device/geo features carry no signal
  * IBM typologies are mapped to the nearest contract typology; without a patterns file, laundering edges
    are grouped into scenarios by connected component and typed 'mule_passthrough'
  * self-transfers (src == dst) are dropped because the contract forbids them (count is reported)
  * not run against the real file in the build sandbox (no network access); covered by a fixture test
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
from data import schemas as S

FX_TO_INR = {  # static, approximate; amounts are only used for relative structure
    "US Dollar": 83.0, "Euro": 90.0, "Yuan": 11.5, "Shekel": 22.5, "Canadian Dollar": 61.0,
    "Australian Dollar": 54.0, "Ruble": 0.9, "UK Pound": 105.0, "Rupee": 1.0, "Yen": 0.56,
    "Mexican Peso": 4.8, "Swiss Franc": 93.0, "Brazil Real": 16.0, "Saudi Riyal": 22.1, "Bitcoin": 2_000_000.0,
}
FORMAT_TO_CHANNEL = {"ACH": "NEFT", "Cheque": "NEFT", "Reinvestment": "NEFT", "Wire": "IMPS",
                     "Bitcoin": "IMPS", "Cash": "IMPS", "Credit Card": "UPI"}
PATTERN_TO_TYPOLOGY = {"FAN-OUT": "fan_out", "FAN-IN": "fan_in", "CYCLE": "circular",
                       "SCATTER-GATHER": "fragmentation", "GATHER-SCATTER": "mule_passthrough",
                       "STACK": "mule_passthrough", "BIPARTITE": "fan_out", "RANDOM": "mule_passthrough"}
REQUIRED = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
            "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]


def _check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise S.SchemaError(f"IBM file does not match the expected schema; missing {missing}; got {list(df.columns)}")


def _key(ts, fb, fa, tb, ta, amt) -> Tuple:
    return (str(ts).strip(), int(fb), str(fa).strip(), int(tb), str(ta).strip(), round(float(amt), 2))


def parse_patterns(text: str) -> Dict[Tuple, Tuple[str, int]]:
    """map txn key -> (IBM pattern name, attempt number)."""
    out, cur, n = {}, None, 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            cur = line.split(" - ", 1)[1].split(":")[0].strip().upper()
            n += 1
        elif line.startswith("END LAUNDERING ATTEMPT"):
            cur = None
        elif cur and line.count(",") >= 10:
            f = line.split(",")
            out[_key(f[0], f[1], f[2], f[3], f[4], f[5])] = (cur, n)
    return out


def _components(edges) -> Dict[str, int]:
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    roots = {}
    return {x: roots.setdefault(find(x), len(roots) + 1) for x in list(parent)}


def convert(trans: pd.DataFrame, patterns_text: Optional[str] = None):
    """trans: raw HI-Small_Trans dataframe (read with dtype=str). Returns (accounts, txns, scenarios, report)."""
    _check_schema(trans)
    df = trans.copy()
    unknown = set(df["Receiving Currency"]) - set(FX_TO_INR)
    if unknown:
        raise S.SchemaError(f"unknown currencies {sorted(unknown)}; extend FX_TO_INR")
    df["_ts"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["_src"] = "I" + df["From Bank"].astype(int).astype(str) + "_" + df["Account"].str.strip()
    df["_dst"] = "I" + df["To Bank"].astype(int).astype(str) + "_" + df["Account.1"].str.strip()
    df["_amt"] = df["Amount Received"].astype(float) * df["Receiving Currency"].map(FX_TO_INR)
    df["_label"] = df["Is Laundering"].astype(int)
    pat = parse_patterns(patterns_text) if patterns_text else {}
    df["_key"] = [_key(*r) for r in zip(df["Timestamp"], df["From Bank"], df["Account"], df["To Bank"],
                                        df["Account.1"], df["Amount Received"])]
    n_in = len(df)
    df = df[df["_src"] != df["_dst"]].copy()
    report = dict(rows_in=n_in, self_transfers_dropped=n_in - len(df))
    df = df[df["_amt"] > 0].sort_values(["_ts"], kind="stable").reset_index(drop=True)

    t = (df["_ts"] - df["_ts"].min()).dt.total_seconds().to_numpy()
    span = max(t.max(), 1.0)
    sec = np.floor(t * ((C.SIM_DAYS * 86400 - 2) / span)).astype("int64")
    ts = pd.to_datetime(int(pd.Timestamp(C.SIM_START).timestamp()) + sec, unit="s", utc=True)
    df["_ts_out"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    pos = df["_label"] == 1
    typ = np.array(["none"] * len(df), dtype=object)
    sid = np.array([""] * len(df), dtype=object)
    if pat:
        names = [pat.get(k) for k in df.loc[pos, "_key"]]
        typ[pos.to_numpy()] = [PATTERN_TO_TYPOLOGY.get(n[0], "mule_passthrough") if n else "mule_passthrough" for n in names]
        sid[pos.to_numpy()] = [f"S{n[1]:04d}" if n else "S9999" for n in names]
    else:
        comp = _components(zip(df.loc[pos, "_src"], df.loc[pos, "_dst"]))
        typ[pos.to_numpy()] = "mule_passthrough"
        sid[pos.to_numpy()] = [f"S{comp[s]:04d}" for s in df.loc[pos, "_src"]]
    df["_typ"], df["_sid"] = typ, sid

    txns = pd.DataFrame({
        "txn_id": ["T%07d" % (i + 1) for i in range(len(df))], "ts": df["_ts_out"],
        "src_account": df["_src"], "dst_account": df["_dst"], "amount": df["_amt"].round(2),
        "channel": df["Payment Format"].map(FORMAT_TO_CHANNEL).fillna("IMPS"),
        "origin_type": "TRANSFER", "device_id": "DEV_" + df["_src"], "ip": "10.0.0.1", "city": "NA",
        "remark": "fmt:" + df["Payment Format"].astype(str), "label": df["_label"],
        "typology": df["_typ"], "scenario_id": df["_sid"], "root_txn_id_gt": ""})[S.TXN_COLUMNS]
    first = txns[txns["scenario_id"] != ""].groupby("scenario_id")["txn_id"].first()
    launder = txns["typology"].isin(S.LAUNDERING_TYPOLOGIES)
    txns["root_txn_id_gt"] = np.where(launder, txns["scenario_id"].map(first).fillna(""), "")

    ids = pd.Index(pd.unique(pd.concat([txns["src_account"], txns["dst_account"]])))
    first_seen = pd.concat([txns[["ts", "src_account"]].rename(columns={"src_account": "a"}),
                            txns[["ts", "dst_account"]].rename(columns={"dst_account": "a"})]).groupby("a")["ts"].min()
    accounts = pd.DataFrame({
        "account_id": ids, "created_at": first_seen.reindex(ids).to_numpy(), "kyc_tier": 2,
        "account_type": "individual", "city": "NA", "primary_device_id": "DEV_" + ids.to_series().to_numpy(),
        "primary_ip_prefix": "10.0.0",
        "phone_hash": [hashlib.sha1(a.encode()).hexdigest()[:12] for a in ids], "role_gt": "normal"})[S.ACCOUNT_COLUMNS]

    rows = []
    for s, g in txns[txns["label"] == 1].groupby("scenario_id", sort=True):
        accs = set(g["src_account"]) | set(g["dst_account"])
        rows.append(dict(scenario_id=s, typology=g["typology"].iloc[0], root_txn_id=g["txn_id"].iloc[0],
                         total_amount=round(float(g["amount"].iloc[0]), 2),
                         num_fragments=int((g["amount"] < C.ALERT_THRESHOLD_INR).sum()), num_accounts=len(accs),
                         start_ts=g["ts"].min(), end_ts=g["ts"].max()))
    scen = pd.DataFrame(rows, columns=S.SCENARIO_COLUMNS)
    report.update(rows_out=len(txns), accounts=len(accounts), scenarios=len(scen),
                  label_rate=float(txns["label"].mean()))
    return accounts, txns, scen, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trans", type=Path, required=True, help="HI-Small_Trans.csv")
    ap.add_argument("--patterns", type=Path, default=None, help="HI-Small_Patterns.txt (optional)")
    ap.add_argument("--out", type=Path, default=C.DATA_OUT / "ibm")
    a = ap.parse_args(argv)
    raw = pd.read_csv(a.trans, dtype=str)
    pat = a.patterns.read_text() if a.patterns else None
    acc, tx, sc, rep = convert(raw, pat)
    S.validate_accounts(acc)
    S.validate_transactions(tx, acc)
    S.validate_scenarios(sc, tx)
    a.out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(a.out / "accounts.csv", index=False)
    tx.to_csv(a.out / "transactions.csv", index=False)
    sc.to_csv(a.out / "scenarios.csv", index=False)
    print(rep)


if __name__ == "__main__":
    main()
