# models/train.py
"""Train the four risk components on the historical split and save to models/saved/."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

import config
from features.behavior import BehaviorExtractor, FEATURE_NAMES as BEH_NAMES
from features.graph import GraphExtractor, FEATURE_NAMES as GRAPH_NAMES
from features.provenance import FEATURE_NAMES as PROV_NAMES
from ledger.lots import LotLedger

def train_models():
    print("Loading historical transactions...")
    txns_path = Path("data/out/transactions.csv")
    if not txns_path.exists():
        raise FileNotFoundError(f"{txns_path} not found. Generate data first.")
    
    df = pd.read_csv(txns_path)
    df["_ts_dt"] = pd.to_datetime(df["ts"], utc=True)
    
    split_dt = pd.to_datetime(config.SPLIT_TS, utc=True)
    train_df = df[df["_ts_dt"] < split_dt].sort_values("_ts_dt").copy()
    
    print(f"Extracting features for {len(train_df)} training transactions (Chronological)...")
    ledger = LotLedger(config)
    beh_ext = BehaviorExtractor(config.ALERT_THRESHOLD_INR)
    graph_ext = GraphExtractor(config.ALERT_THRESHOLD_INR)
    
    X_beh, X_graph, X_prov, y = [], [], [], []
    
    # Chronological streaming for causal feature extraction
    for _, row in train_df.iterrows():
        txn = row.to_dict()
        
        prov_feats = ledger.process(txn)
        beh_feats = beh_ext.extract_and_update(txn)
        graph_feats = graph_ext.extract_and_update(txn)
        
        X_beh.append([beh_feats[k] for k in BEH_NAMES])
        X_graph.append([graph_feats[k] for k in GRAPH_NAMES])
        X_prov.append([prov_feats.get(k, 0.0) for k in PROV_NAMES])
        y.append(int(txn.get("label", 0)))

    X_beh = np.array(X_beh, dtype=np.float32)
    X_graph = np.array(X_graph, dtype=np.float32)
    X_prov = np.array(X_prov, dtype=np.float32)
    y = np.array(y, dtype=int)

    print("Training Behavior Model (Supervised)...")
    clf_beh = HistGradientBoostingClassifier(random_state=config.SEED, class_weight="balanced")
    clf_beh.fit(X_beh, y)

    print("Training Anomaly Model (Unsupervised Isolation Forest)...")
    clf_ano = IsolationForest(random_state=config.SEED, n_estimators=100, contamination=0.01)
    clf_ano.fit(X_beh)

    print("Training Graph Model (Supervised)...")
    clf_graph = HistGradientBoostingClassifier(random_state=config.SEED, class_weight="balanced")
    clf_graph.fit(X_graph, y)

    print("Training Provenance Model (Supervised)...")
    clf_prov = HistGradientBoostingClassifier(random_state=config.SEED, class_weight="balanced")
    clf_prov.fit(X_prov, y)

    # Save Models
    models_dir = Path(getattr(config, "MODELS_DIR", "models")) / "saved"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf_beh, models_dir / "behavior.joblib")
    joblib.dump(clf_ano, models_dir / "anomaly.joblib")
    joblib.dump(clf_graph, models_dir / "graph.joblib")
    joblib.dump(clf_prov, models_dir / "provenance.joblib")

    # Save Thresholds & Stats for normalization
    ano_scores = clf_ano.decision_function(X_beh)
    thresholds = {
        "anomaly_min": float(np.min(ano_scores)),
        "anomaly_max": float(np.max(ano_scores)),
        "feature_names": {
            "behavior": BEH_NAMES,
            "graph": GRAPH_NAMES,
            "provenance": PROV_NAMES
        }
    }
    with open(models_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Training complete. Models saved to {models_dir}")

if __name__ == "__main__":
    train_models()