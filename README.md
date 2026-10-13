# RupeeTrace

Real-time fraud and AML risk engine for UPI/IMPS-style payments with rupee-level provenance
("money lots"). Hackathon topic: *Real-Time Fraud & AML Risk Orchestration*.
Other sections (ledger, features, models, engine, dashboard) are owned by other accounts.

## Data and Evaluation

Owner: acc_1. Python 3.11+, `pip install -r requirements.txt`. Run everything from the repo root.

### Run commands

| Step | Command | Output |
|---|---|---|
| Generate data (deterministic, seed 42) | `python -m data.generator` | `data/out/{accounts,transactions,scenarios}.csv` |
| Small/fast variant | `python -m data.generator --n-accounts 1000 --out /tmp/small` | same files, elsewhere |
| Rules baseline (model A) | `python -m eval.rules_baseline` | `eval/out/predictions_A_rules.csv` |
| Metrics for all existing prediction files | `python -m eval.metrics` | `eval/out/metrics_summary.{csv,json}` |
| Ablation table + chart | `python -m eval.ablation` | `eval/out/ablation.{csv,md,html}` |
| Recall/precision by amount bucket (headline) | `python -m eval.buckets` | `eval/out/buckets.{csv,md}`, `buckets_recall.html`, `buckets_precision.html` |
| Latency summary | `python -m eval.latency` | `eval/out/latency.{csv,md,html}` |
| IBM AMLworld converter (stretch) | `python -m data.ibm_loader --trans HI-Small_Trans.csv [--patterns HI-Small_Patterns.txt]` | `data/out/ibm/*.csv` |
| Tests | `python -m pytest -q` | |

`eval.*` scripts accept `--pred-dir` (and `--out-dir`) and skip any missing `predictions_*.csv`
with a warning, so they work while B/C/D are still being built. PNG export happens
automatically if `kaleido` is installed; HTML is always written.

### What the generated data contains (default run)

* 5,000 accounts (4,000 individual, 600 merchant, 400 business), 30 days, 445,022 transactions,
  1.19 % `label=1` (5,301), 127 scenarios. Train = days 1-21 (312,931 txns), test = days 22-30 (132,091 txns, 1.02 % positive).
* Normal traffic: salary credits, UPI P2P, merchant payments, fixed-biller bills, rent, cash deposits/withdrawals,
  loans, B2B payments (some legitimately above Rs 50,000), diurnal timing.
* Hard negatives (all `label=0`): weddings, tuition (incl. round amounts just under the threshold), festival
  rushes at shops (legit fan-in), friends splitting bills, freelancers, new-phone + travel + large purchase
  (looks like account takeover), families sharing a device/IP, scam-word remarks ("refund", "urgent", "KYC update")
  on ~1.5 % of normal txns so text is not a trivial signal.
* Typologies with ground truth (txn counts): fragmentation 3,513 (11 scenarios), velocity_abuse 687, fan_in 409,
  fan_out 286, circular 193, account_takeover 129, mule_passthrough 84. Every typology has scenarios in both
  train and test, and no scenario straddles the split.
* Fragmentation: cash deposit of ~Rs 2.4-9.8 lakh at an `origin` account, split into 40-150 layer-1 transfers of
  Rs 3,000-9,900, relayed through 15-40 relay accounts over 2-4 layers (dwell times from ~2 minutes to 3 hours per hop),
  re-converging at 2-3 collectors that cash out in chunks below Rs 50,000. Relays are normal accounts with their own
  history; about 45 % of scenarios share a device and/or IP prefix across half their relays.

### Assumptions and interpretations (none change the contract)

* `num_fragments` in `scenarios.csv` = number of transactions in the scenario below `ALERT_THRESHOLD_INR`.
* `scenarios.csv.total_amount` = value entering the scenario (deposit for fragmentation/fan_out, first transfer for
  mule/circular, sum of victim transfers for takeover, sum of inbound transfers for fan_in/velocity).
* `scenarios.csv.root_txn_id` = earliest transaction of the scenario. Row-level `root_txn_id_gt` is filled only for
  laundering typologies (fragmentation, mule_passthrough, circular, fan_in, fan_out); blank for account_takeover and velocity_abuse.
* The fragmentation root deposit is itself `label=1` and `root_txn_id_gt` points to itself.
* Money entering from outside carries `origin_type != TRANSFER` and an `EXT_*` source; the only external destination is `EXT_CASH_OUT`.
* `action` may be one token (`HOLD`) or tokens joined by `+` (`HOLD+ESCALATE`, `ALLOW+ESCALATE`). All eval code
  parses tokens: flag = any of HOLD/BLOCK/ESCALATE; friction = any of STEP_UP/HOLD/BLOCK. **Model owners must keep
  to one of those two forms.**
* `config.py` additionally exposes helpers (`action_for_score`, `is_flag_action`, `bucket_index`, ...) and path
  constants; all contract constants are unchanged.
* Metrics are computed on the test split only. Missing predictions count as ALLOW (warned); predictions outside the
  test split raise an error.
* Score-level metrics use threshold 0.55 (HOLD band edge); PR-AUC uses the raw `risk_score`.
* Detection latency = minutes from a scenario's first `label=1` test txn to its first flagged txn; median/p95 are over
  detected scenarios only, so always read it together with `scenario_recall`.
* Rules baseline (A): R1 amount >= 50,000; R2 more than 10 txns in a trailing hour; R3 UTC-day outflow > Rs 1,00,000;
  R4 never-seen device (and not the primary device) with amount >= Rs 20,000; R5 multiple of Rs 500 in [45,000, 50,000).
  Each rule has a weight; `risk_score = 1 - prod(1 - w)`, so one hit = HOLD, two hits = BLOCK. It is a streaming
  engine, state is warmed on the train period, `latency_ms` is measured per transaction.
* Leakage rule: `label`, `typology`, `scenario_id`, `root_txn_id_gt`, `role_gt` are evaluation-only.
  `tests/test_eval.py` checks the rules engine gives identical output when those columns are removed.

### Known limits

* Synthetic data: the fragmentation scenario count is small (11 total, 3 in test), so scenario-level numbers are noisy;
  transaction-level numbers (1,350 test positives) are steadier. Change `BASE_COUNTS` in `data/generator.py` for more.
* The fragmentation root deposit (>= Rs 2 lakh) is above the alert threshold, so threshold rules alert at the start of
  most fragmentation scenarios. Detection latency therefore favours rules; the headline comparison is **txn-level recall
  by amount bucket** (`eval.buckets`), where fragments live in the two sub-threshold buckets.
* Relay-hop amounts can dip to ~Rs 2,980 after small skims at later hops; layer-1 fragments are always within 3,000-9,900.
* Rules baseline on the default data (test split): precision 0.29, recall 0.36, F1 0.32, FPR/friction 0.9 %,
  PR-AUC 0.18, scenario recall 0.81. Recall by bucket: 0.50 (< Rs 5k), 0.28 (Rs 5k-50k), 1.00 (>= Rs 50k, precision 0.05).
  The daily-outflow and velocity rules catch some busy relays; this baseline was not tuned to look weak.
* IBM loader (`data/ibm_loader.py`): written against the published HI-Small schema (verified from public descriptions;
  the Kaggle download was not reachable from the build sandbox, so it is tested on a fixture, not the real file). The dataset
  is on Kaggle under a Community Data License Agreement per the NeurIPS supplement; check the exact variant
  before redistributing converted data. It stretches 10 days to 30, converts currencies with a static FX table,
  and has no device/IP/city signal. Output goes to `data/out/ibm/`, never over the main CSVs.
