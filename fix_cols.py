import pandas as pd
cols = ["txn_id", "risk_score", "action", "latency_ms", "reasons"]
for n in ["B_behavior_anomaly", "C_plus_graph", "D_plus_provenance"]:
    p = f"eval/out/predictions_{n}.csv"
    d = pd.read_csv(p, keep_default_na=False)
    d["reasons"] = d["reasons"].apply(lambda x: x if str(x).startswith("[") else "[]") if "reasons" in d else "[]"
    d[cols].to_csv(p, index=False)
print("fixed")
