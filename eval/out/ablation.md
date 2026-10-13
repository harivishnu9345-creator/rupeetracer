# Ablation (test split)

flag = HOLD/BLOCK/ESCALATE. friction = STEP_UP/HOLD/BLOCK on legitimate txns. Latencies: detection in minutes (detected scenarios only), engine in ms.

| model | label | precision | recall | f1 | fpr | friction_rate | pr_auc | scenario_recall | det_latency_median_min | det_latency_p95_min | latency_ms_p50 | latency_ms_p95 | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rules | A: Rules | 0.8773 | 0.3600 | 0.5105 | 0.0085 | 0.0085 | 0.4244 | 0.8378 | 0.0000 | 244.3917 | 0.0029 | 0.0046 | 486 | 68 | 864 | 7932 |
| B_behavior_anomaly | B: + Behavior/Anomaly | 0.9731 | 0.8570 | 0.9114 | 0.0040 | 0.0324 | 0.9578 | 0.9459 | 0.0000 | 79.0150 | 15.4545 | 28.7775 | 1157 | 32 | 193 | 7968 |
| C_plus_graph | C: + Graph | 0.9896 | 0.7748 | 0.8691 | 0.0014 | 0.0334 | 0.9554 | 0.8108 | 0.0000 | 166.6450 | 15.4545 | 28.7775 | 1046 | 11 | 304 | 7989 |
| D_plus_provenance | D: + Provenance | 0.9920 | 0.8230 | 0.8996 | 0.0011 | 0.0239 | 0.9642 | 0.7838 | 10.1167 | 168.8067 | 15.4545 | 28.7775 | 1111 | 9 | 239 | 7991 |
