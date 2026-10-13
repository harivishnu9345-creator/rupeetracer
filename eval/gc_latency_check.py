# eval/gc_latency_check.py
"""Analyzes scoring latency with and without the GC/Threading optimizations."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from models.stream_scorer import StreamScorer
import config
from tqdm import tqdm

def run_scorer(enable_fix: bool, title: str):
    print(f"\n--- Running: {title} (fix={'ON' if enable_fix else 'OFF'}) ---")
    scorer = StreamScorer(getattr(config, "MODELS_DIR", "models"), enable_gc_fix=enable_fix)
    
    df = pd.read_csv("data/out/transactions.csv").head(10000) # Use 10k for speed check
    latencies = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Ignore label and eval fields
        txn = {k: v for k, v in row.to_dict().items() if k not in ["label", "typology", "scenario_id"]}
        out = scorer.process(txn)
        latencies.append(out["latency_ms"])
        
    latencies = np.array(latencies)
    print(f"p50: {np.percentile(latencies, 50):.3f} ms")
    print(f"p95: {np.percentile(latencies, 95):.3f} ms")
    print(f"p99: {np.percentile(latencies, 99):.3f} ms")
    print(f"Max: {np.max(latencies):.3f} ms")
    print(f"Spikes > 100ms: {np.sum(latencies > 100)}")
    return latencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare fix ON vs fix OFF")
    parser.add_argument("--no-fix", action="store_true", help="Run with fix disabled")
    args = parser.parse_args()

    if args.compare:
        run_scorer(enable_fix=False, title="Baseline (No GC Fix)")
        run_scorer(enable_fix=True, title="Optimized (GC Fix ON)")
    else:
        run_scorer(enable_fix=not args.no_fix, title="Single Run")