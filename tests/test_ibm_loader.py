import io

import pandas as pd
import pytest

import config as C
from data import ibm_loader as L, schemas as S

HEADER = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
ROWS = [
    "2022/09/01 00:20,011,A1,011,B1,1000.0,US Dollar,1000.0,US Dollar,ACH,0",
    "2022/09/02 10:00,011,B1,012,C1,900.0,US Dollar,900.0,US Dollar,Wire,1",
    "2022/09/03 11:00,012,C1,013,D1,880.0,US Dollar,880.0,US Dollar,Wire,1",
    "2022/09/04 12:00,013,D1,013,D1,5.0,Euro,5.0,Euro,Reinvestment,0",
    "2022/09/09 12:00,013,D1,011,A1,50.0,Euro,50.0,Euro,Credit Card,0",
    "2022/09/10 00:00,011,A1,011,B1,70.0,Rupee,70.0,Rupee,Cheque,0",
]
PATTERNS = ("BEGIN LAUNDERING ATTEMPT - CYCLE: Max 3 hops\n"
            "2022/09/02 10:00,011,B1,012,C1,900.0,US Dollar,900.0,US Dollar,Wire,1\n"
            "2022/09/03 11:00,012,C1,013,D1,880.0,US Dollar,880.0,US Dollar,Wire,1\n"
            "END LAUNDERING ATTEMPT - CYCLE\n")


def raw():
    return pd.read_csv(io.StringIO("\n".join([HEADER] + ROWS)), dtype=str)


def test_header_dup_account_columns():
    assert "Account.1" in raw().columns


@pytest.mark.parametrize("patterns", [None, PATTERNS])
def test_convert_valid_contract(patterns):
    acc, tx, sc, rep = L.convert(raw(), patterns)
    S.validate_accounts(acc); S.validate_transactions(tx, acc); S.validate_scenarios(sc, tx)
    assert rep["self_transfers_dropped"] == 1 and len(tx) == 5
    assert tx.ts.iloc[0] == C.SIM_START and tx.ts.iloc[-1] < C.SIM_END
    assert tx.label.sum() == 2 and sc.shape[0] == (1 if patterns else 1)
    if patterns:
        assert set(tx.loc[tx.label == 1, "typology"]) == {"circular"}
    assert tx.amount.iloc[0] == 1000 * L.FX_TO_INR["US Dollar"]


def test_schema_mismatch_and_unknown_currency():
    with pytest.raises(S.SchemaError):
        L.convert(raw().drop(columns=["Payment Format"]))
    r = raw(); r.loc[0, "Receiving Currency"] = "Zorkmid"
    with pytest.raises(S.SchemaError):
        L.convert(r)
