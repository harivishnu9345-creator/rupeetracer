import pandas as pd, shutil, os
src = "data/out/transactions.csv"; bak = "data/out/transactions_FULL.csv"
if not os.path.exists(bak): shutil.copy(src, bak)
t = pd.read_csv(bak).sort_values("ts")
ts = pd.to_datetime(t["ts"], utc=True)
split = pd.Timestamp("2026-01-22T00:00:00Z")
def shrink(df, n):
    m = df["scenario_id"].notna() & (df["scenario_id"].astype(str).str.strip() != "")
    return pd.concat([df[m], df[~m].sample(min(n, (~m).sum()), random_state=42)]).sort_values("ts")
small = pd.concat([shrink(t[ts < split], 15000), shrink(t[ts >= split], 8000)]).sort_values("ts")
small.to_csv(src, index=False)
print("rows:", len(small))
