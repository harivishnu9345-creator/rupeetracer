# make_sample.py  (run: python make_sample.py)
import pandas as pd
t = pd.read_csv("data/out/transactions.csv")
t = t.sort_values("ts")
head = t.head(4000)
scen = t[t["scenario_id"].notna() & (t["scenario_id"].astype(str) != "")]
scen = scen.groupby("scenario_id", group_keys=False).head(150)
pd.concat([head, scen]).drop_duplicates("txn_id").sort_values("ts").to_csv("data/out/transactions_sample.csv", index=False)
pd.read_csv("eval/out/predictions_A_rules.csv").head(300).to_csv("eval/out/predictions_A_rules_sample.csv", index=False)
print("done")