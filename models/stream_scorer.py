# models/stream_scorer.py
"""StreamScorer implementation with optimized deterministic GC/threading for latency control."""
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np

import config
from features.behavior import BehaviorExtractor, FEATURE_NAMES as BEH_NAMES
from features.graph import GraphExtractor, FEATURE_NAMES as GRAPH_NAMES
from features.provenance import FEATURE_NAMES as PROV_NAMES
from ledger.lots import LotLedger
from models.explain import explain


class StreamScorer:
    def __init__(self, models_dir: str, enable_gc_fix: bool = True):
        self.models_dir = Path(models_dir) / "saved"
        if not self.models_dir.exists():
            raise FileNotFoundError("Models not found. Train models first.")

        # Load models
        self.clf_beh = joblib.load(self.models_dir / "behavior.joblib")
        self.clf_ano = joblib.load(self.models_dir / "anomaly.joblib")
        self.clf_graph = joblib.load(self.models_dir / "graph.joblib")
        self.clf_prov = joblib.load(self.models_dir / "provenance.joblib")
        
        with open(self.models_dir / "thresholds.json") as f:
            self.thresholds = json.load(f)

        # Initialize State Extractors
        self.ledger = LotLedger(config)
        self.beh_ext = BehaviorExtractor(config.ALERT_THRESHOLD_INR)
        self.graph_ext = GraphExtractor(config.ALERT_THRESHOLD_INR)

        # GC and Latency Optimization State
        self.enable_gc_fix = enable_gc_fix
        self._txn_count = 0
        self._in_context = False

        if self.enable_gc_fix:
            gc.freeze()
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._louvain_future = None
            self._next_louvain_result = None
            
            # Monkey-patch GraphExtractor to run Louvain asynchronously but swap deterministically
            def _deterministic_refresh():
                from networkx.algorithms.community import louvain_communities
                
                # 1. Apply the computation from exactly one interval ago
                if self._next_louvain_result is not None:
                    self.graph_ext.communities, self.graph_ext.comm_stats = self._next_louvain_result
                    self._next_louvain_result = None

                # 2. Await the background thread (should be finished since it had a full interval)
                if self._louvain_future is not None:
                    self._next_louvain_result = self._louvain_future.result()

                # 3. Spawn the next computation on a static snapshot
                U = self.graph_ext.G.to_undirected()
                
                def _compute(graph):
                    if len(graph.nodes) == 0: return {}, {}
                    try:
                        comms = louvain_communities(graph, seed=config.SEED)
                    except Exception:
                        comms = [set(graph.nodes)]
                    
                    c_dict, stats = {}, {}
                    for i, comm in enumerate(comms):
                        nodes = list(comm)
                        sub_G = self.graph_ext.G.subgraph(nodes)
                        edges = sub_G.number_of_edges()
                        n = len(nodes)
                        density = (2.0 * edges) / (n * (n - 1)) if n > 1 else 0.0
                        total_amt = sum(d.get('weight', 0.0) for u, v, d in sub_G.edges(data=True))
                        sub_thresh = sum(d.get('sub_thresh_count', 0) for u, v, d in sub_G.edges(data=True))
                        
                        stats[i] = {"size": n, "density": density, "total_amount": total_amt, "sub_thresh_edges": sub_thresh}
                        for node in nodes:
                            c_dict[node] = i
                    return c_dict, stats

                self._louvain_future = self._executor.submit(_compute, U)

            self.graph_ext._refresh_communities = _deterministic_refresh

    def __del__(self):
        if self.enable_gc_fix:
            gc.enable()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)

    def process(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        
        if self.enable_gc_fix:
            if not self._in_context:
                gc.disable()
                self._in_context = True
            self._txn_count += 1
            if self._txn_count % 500 == 0:
                gc.collect(0)
            if self._txn_count % 10000 == 0:
                gc.collect(1)

        # 1. Causal Extraction
        prov_feats = self.ledger.process(txn)
        beh_feats = self.beh_ext.extract_and_update(txn)
        graph_feats = self.graph_ext.extract_and_update(txn)

        # 2. Vectorize
        x_beh = np.array([[beh_feats[k] for k in BEH_NAMES]], dtype=np.float32)
        x_graph = np.array([[graph_feats[k] for k in GRAPH_NAMES]], dtype=np.float32)
        x_prov = np.array([[prov_feats.get(k, 0.0) for k in PROV_NAMES]], dtype=np.float32)

        # 3. Model Scoring
        s_beh = float(self.clf_beh.predict_proba(x_beh)[0, 1])
        s_graph = float(self.clf_graph.predict_proba(x_graph)[0, 1])
        s_prov = float(self.clf_prov.predict_proba(x_prov)[0, 1])
        
        # Anomaly Normalization
        raw_ano = float(self.clf_ano.decision_function(x_beh)[0])
        a_min = self.thresholds["anomaly_min"]
        a_max = self.thresholds["anomaly_max"]
        s_ano = max(0.0, min(1.0, (raw_ano - a_min) / (a_max - a_min + 1e-9)))

        # 4. Packaging
        features = {**prov_feats, **beh_feats, **graph_feats, "amount": float(txn.get("amount", 0.0))}
        scores = {
            "behavior_score": round(s_beh, 4),
            "anomaly_score": round(1.0 - s_ano, 4), # Invert anomaly (lower decision is more anomalous)
            "graph_score": round(s_graph, 4),
            "provenance_score": round(s_prov, 4)
        }
        
        # Determine cluster risk based on community density and sub-threshold behavior
        c_risk = float(graph_feats.get("graph_src_community_density", 0.0)) * \
                 min(1.0, float(graph_feats.get("graph_src_comm_sub_threshold_edges", 0.0)) / 10.0)
        c_risk = round(min(1.0, c_risk), 4)

        out = {
            "txn_id": txn.get("txn_id"),
            **scores,
            "cluster_risk": c_risk,
            "reasons": explain(features, scores),
            "features": features,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3)
        }
        return out