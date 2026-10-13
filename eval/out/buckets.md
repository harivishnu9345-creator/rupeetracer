# Recall / precision by amount bucket (test split)

| model | bucket | n_txn | n_pos | flagged | tp | fp | recall | precision |
|---|---|---|---|---|---|---|---|---|
| A_rules | Rs 0-5k | 7764 | 331 | 166 | 165 | 1 | 0.4985 | 0.9940 |
| A_rules | Rs 5k-50k | 1474 | 970 | 276 | 272 | 4 | 0.2804 | 0.9855 |
| A_rules | Rs 50k+ | 112 | 49 | 112 | 49 | 63 | 1.0000 | 0.4375 |
| B_behavior_anomaly | Rs 0-5k | 7764 | 331 | 280 | 267 | 13 | 0.8066 | 0.9536 |
| B_behavior_anomaly | Rs 5k-50k | 1474 | 970 | 872 | 858 | 14 | 0.8845 | 0.9839 |
| B_behavior_anomaly | Rs 50k+ | 112 | 49 | 37 | 32 | 5 | 0.6531 | 0.8649 |
| C_plus_graph | Rs 0-5k | 7764 | 331 | 258 | 253 | 5 | 0.7644 | 0.9806 |
| C_plus_graph | Rs 5k-50k | 1474 | 970 | 778 | 772 | 6 | 0.7959 | 0.9923 |
| C_plus_graph | Rs 50k+ | 112 | 49 | 21 | 21 | 0 | 0.4286 | 1.0000 |
| D_plus_provenance | Rs 0-5k | 7764 | 331 | 267 | 262 | 5 | 0.7915 | 0.9813 |
| D_plus_provenance | Rs 5k-50k | 1474 | 970 | 837 | 833 | 4 | 0.8588 | 0.9952 |
| D_plus_provenance | Rs 50k+ | 112 | 49 | 16 | 16 | 0 | 0.3265 | 1.0000 |
