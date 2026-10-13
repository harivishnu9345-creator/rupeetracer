import numpy as np
import pandas as pd
import pytest

import config as C
from data import generator, schemas as S


def test_determinism(small):
    acc2, tx2, sc2 = generator.generate(n_accounts=1000, seed=42)
    acc, tx, sc = small
    pd.testing.assert_frame_equal(acc, acc2)
    pd.testing.assert_frame_equal(tx, tx2)
    pd.testing.assert_frame_equal(sc, sc2)


def test_seed_changes_output(small):
    _, tx3, _ = generator.generate(n_accounts=1000, seed=7)
    assert not tx3["amount"].head(2000).equals(small[1]["amount"].head(2000))


def test_schema_valid(small):
    acc, tx, sc = small
    S.validate_accounts(acc)
    S.validate_transactions(tx, acc)
    S.validate_scenarios(sc, tx)


def test_schema_rejects_bad_data(small):
    acc, tx, sc = small
    bad = tx.copy()
    bad.loc[0, "label"] = 1  # label without typology
    with pytest.raises(S.SchemaError):
        S.validate_transactions(bad, acc)
    with pytest.raises(S.SchemaError):
        S.validate_transactions(tx.iloc[::-1].reset_index(drop=True), acc)  # not sorted


def test_class_balance(small):
    _, tx, _ = small
    assert 0.003 < tx["label"].mean() < 0.04
    assert (tx["typology"].loc[tx.label == 1].nunique()) == len(S.TYPOLOGIES) - 1


def test_scenarios_in_both_splits(small):
    _, _, sc = small
    sc = sc.assign(test=sc["start_ts"] >= C.SPLIT_TS)
    for typ, g in sc.groupby("typology"):
        assert g["test"].any() and (~g["test"]).any(), typ
    assert not ((sc["start_ts"] < C.SPLIT_TS) & (sc["end_ts"] >= C.SPLIT_TS)).any()  # none straddle the split


def test_fragmentation_shape(small):
    _, tx, sc = small
    f = tx[(tx.typology == "fragmentation") & (tx.src_account != "EXT_CASH") & (tx.dst_account != "EXT_CASH_OUT")]
    assert f["amount"].between(2900, 9900).all()          # sub-threshold fragments
    assert (f["amount"] < C.ALERT_THRESHOLD_INR).all()
    roots = tx[(tx.typology == "fragmentation") & (tx.channel == "CASH_DEP")]
    assert roots["amount"].between(150_000, 1_050_000).all()
    assert (roots["root_txn_id_gt"] == roots["txn_id"]).all()


def test_relays_are_unflagged_and_have_own_traffic(small):
    acc, tx, _ = small
    relays = set(acc.loc[acc.role_gt == "relay", "account_id"])
    legit = tx[tx.label == 0]
    with_history = relays & (set(legit["src_account"]) | set(legit["dst_account"]))
    assert len(with_history) / len(relays) > 0.95


def test_hard_negatives_present(small):
    _, tx, _ = small
    leg = tx[tx.label == 0]
    assert (leg["amount"] >= C.ALERT_THRESHOLD_INR).any()          # legit large payments
    assert leg["remark"].isin(["wedding advance", "tuition fee", "dinner split"]).any()
    assert leg["remark"].isin(["refund", "urgent"]).any()          # text overlap with fraud


def test_shared_infra_across_relays(small):
    acc, _, _ = small
    rel = acc[acc.role_gt == "relay"]
    assert rel["primary_device_id"].str.startswith("DEV_SHR").any()


def test_no_label_leakage_in_scoring_inputs():
    from eval import rules_baseline as R
    assert not set(R.NEEDED) & set(S.LEAKAGE_COLUMNS)


def test_full_csvs_if_present():
    p = C.DATA_OUT / "transactions.csv"
    if not p.exists():
        pytest.skip("full CSVs not generated")
    acc, tx, sc = S.load_accounts(), S.load_transactions(), S.load_scenarios()
    S.validate_accounts(acc); S.validate_transactions(tx, acc); S.validate_scenarios(sc, tx)
    assert 300_000 <= len(tx) <= 500_000
    assert 0.007 <= tx["label"].mean() <= 0.015
