# Ablation (test split)

flag = HOLD/BLOCK/ESCALATE. friction = STEP_UP/HOLD/BLOCK on legitimate txns. Latencies: detection in minutes (detected scenarios only), engine in ms.

| model | label | precision | recall | f1 | fpr | friction_rate | pr_auc | scenario_recall | det_latency_median_min | det_latency_p95_min | latency_ms_p50 | latency_ms_p95 | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rules | A: Rules | 0.2897 | 0.3607 | 0.3213 | 0.0091 | 0.0091 | 0.1766 | 0.8108 | 0.0000 | 187.7558 | 0.0014 | 0.0027 | 487 | 1194 | 863 | 129547 |
