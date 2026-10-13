"""Incremental network graph feature extractor (Strictly Causal)."""
import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np

FEATURE_NAMES = [
    "graph_src_in_degree", "graph_src_out_degree",
    "graph_dst_in_degree", "graph_dst_out_degree",
    "graph_src_distinct_counterparties",
    "graph_src_shared_device_links",
    "graph_src_community_size", "graph_src_community_density",
    "graph_src_comm_total_amount", "graph_src_comm_sub_threshold_edges"
]

REFRESH_INTERVAL = 2000

class GraphExtractor:
    def __init__(self, alert_threshold=50000):
        self.alert_threshold = alert_threshold
        self.G = nx.DiGraph()
        self.G_shared = nx.Graph()
        
        self.account_devices = {}
        self.account_ips = {}
        
        self.txn_count = 0
        self.communities = {} 
        self.comm_stats = {} 

    def extract_and_update(self, txn: dict) -> dict:
        src = str(txn.get('src_account', ''))
        dst = str(txn.get('dst_account', ''))
        amt = float(txn.get('amount', 0.0))
        device = str(txn.get('device_id', ''))
        ip = str(txn.get('ip', ''))

        feats = {k: 0.0 for k in FEATURE_NAMES}

        # --- 1. STRICTLY CAUSAL EXTRACTION ---
        for node, prefix in [(src, "graph_src_"), (dst, "graph_dst_")]:
            if node in self.G:
                feats[prefix + "in_degree"] = float(self.G.in_degree(node))
                feats[prefix + "out_degree"] = float(self.G.out_degree(node))
                if prefix == "graph_src_":
                    preds = set(self.G.predecessors(node))
                    succs = set(self.G.successors(node))
                    feats["graph_src_distinct_counterparties"] = float(len(preds | succs))
            
            if prefix == "graph_src_" and node in self.G_shared:
                feats["graph_src_shared_device_links"] = float(self.G_shared.degree[node])

        comm_id = self.communities.get(src, -1)
        if comm_id != -1 and comm_id in self.comm_stats:
            c_stats = self.comm_stats[comm_id]
            feats["graph_src_community_size"] = float(c_stats["size"])
            feats["graph_src_community_density"] = float(c_stats["density"])
            feats["graph_src_comm_total_amount"] = float(c_stats["total_amount"])
            feats["graph_src_comm_sub_threshold_edges"] = float(c_stats["sub_thresh_edges"])

        for k, v in feats.items():
            if not np.isfinite(v):
                feats[k] = 0.0

        # --- 2. CAUSAL STATE UPDATE ---
        if src and dst:
            if self.G.has_edge(src, dst):
                self.G[src][dst]['weight'] += amt
                self.G[src][dst]['count'] += 1
                if amt < self.alert_threshold:
                    self.G[src][dst]['sub_thresh_count'] = self.G[src][dst].get('sub_thresh_count', 0) + 1
            else:
                self.G.add_edge(src, dst, weight=amt, count=1, 
                                sub_thresh_count=1 if amt < self.alert_threshold else 0)

        for entity, registry in [(device, self.account_devices), (ip, self.account_ips)]:
            if entity and src:
                if entity not in registry:
                    registry[entity] = set()
                for other in registry[entity]:
                    if other != src:
                        self.G_shared.add_edge(src, other)
                registry[entity].add(src)

        self.txn_count += 1
        if self.txn_count % REFRESH_INTERVAL == 0:
            self._refresh_communities()

        return feats

    def _refresh_communities(self):
        U = self.G.to_undirected()
        if len(U.nodes) == 0:
            return

        try:
            comms = louvain_communities(U, seed=42)
        except Exception:
            comms = [set(U.nodes)]

        self.communities = {}
        self.comm_stats = {}
        
        for i, comm in enumerate(comms):
            nodes = list(comm)
            sub_G = self.G.subgraph(nodes)
            edges = sub_G.number_of_edges()
            n = len(nodes)
            # Correct undirected density math
            density = (2.0 * edges) / (n * (n - 1)) if n > 1 else 0.0

            total_amt = sum(d.get('weight', 0.0) for u, v, d in sub_G.edges(data=True))
            sub_thresh = sum(d.get('sub_thresh_count', 0) for u, v, d in sub_G.edges(data=True))

            self.comm_stats[i] = {
                "size": n,
                "density": density,
                "total_amount": total_amt,
                "sub_thresh_edges": sub_thresh
            }
            for node in nodes:
                self.communities[node] = i