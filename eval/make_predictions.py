# eval/make_predictions.py
"""Generate test-split prediction CSVs for all ablation variants."""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import config
from models.stream_scorer import StreamScorer

def get_action(risk_score: float) -> str:
    """Map fused risk score to action band."""
    if risk_score >= 0.80: return "BLOCK"
    if risk_score >= 0.55: return "HOLD"
    if risk_score >= 0.30: return "STEP_UP"
    return "ALLOW"

def main():
    print("Initializing StreamScorer...")
    scorer = StreamScorer(getattr(config, "MODELS_DIR", "models"))
    
    # Try importing Fusion, fallback if not available
    try:
        from engine.fusion import Fusion
        fusion_engine = Fusion()
        fusion_available = True
    except ImportError:
        fusion_available = False
        print("Warning: engine.fusion not found, using rule-based fallback stacker.")

    df = pd.read_csv("data/out/transactions.csv")
    df["_ts_dt"] = pd.to_datetime(df["ts"], utc=True)
    
    # We must process chronologically through the whole stream, 
    # but only save outputs for transactions >= SPLIT_TS
    split_dt = pd.to_datetime(config.SPLIT_TS, utc=True)
    
    res_b, res_c, res_d = [], [], []
    
    print(f"Scoring {len(df)} transactions (saving test split)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        txn = row.to_dict()
        out = scorer.process(txn)
        
        if row["_ts_dt"] >= split_dt:
            # Component sets mapping
            comps = {
                "behavior": out["behavior_score"],
                "anomaly": out["anomaly_score"],
                "graph": out["graph_score"],
                "provenance": out["provenance_score"]
            }

            def fuse(component_keys):
                if fusion_available:
                    # Provide only the available keys, let fusion fill missing with None/0
                    subset = {k: comps[k] for k in component_keys}
                    f = Fusion(components=component_keys) 
                    return f.score(subset)["risk_score"]
                else:
                    # Fallback logical stacker
                    base = sum(comps[k] for k in component_keys) / len(component_keys)
                    if "provenance" in component_keys:
                        # Cap provenance to never push to hold alone (mimicking fusion.py)
                        others = [comps[k] for k in component_keys if k != "provenance"]
                        m = max(others) if others else 0.0
                        cap = 0.54 + 1.5 * max(0.0, m - 0.30)
                        return min(base, cap)
                    return base

            # B: Behavior + Anomaly
            r_b = fuse(["behavior", "anomaly"])
            res_b.append({"txn_id": out["txn_id"], "action": get_action(r_b), "risk_score": r_b, "latency_ms": out["latency_ms"]})
            
            # C: + Graph
            r_c = fuse(["behavior", "anomaly", "graph"])
            res_c.append({"txn_id": out["txn_id"], "action": get_action(r_c), "risk_score": r_c, "latency_ms": out["latency_ms"]})
            
            # D: + Provenance
            r_d = fuse(["behavior", "anomaly", "graph", "provenance"])
            res_d.append({"txn_id": out["txn_id"], "action": get_action(r_d), "risk_score": r_d, "latency_ms": out["latency_ms"]})

    out_dir = Path("eval/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(res_b).to_csv(out_dir / "predictions_B_behavior_anomaly.csv", index=False)
    pd.DataFrame(res_c).to_csv(out_dir / "predictions_C_plus_graph.csv", index=False)
    pd.DataFrame(res_d).to_csv(out_dir / "predictions_D_plus_provenance.csv", index=False)
    
    print(f"Predictions saved to {out_dir}")

if __name__ == "__main__":
    main()