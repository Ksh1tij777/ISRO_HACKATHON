"""
src/eval/topo_metrics.py
------------------------
Topological metrics for the road graph.
These go into the results table in your hackathon submission.
DO NOT edit seg_metrics.py — that is AA's file.
"""

from __future__ import annotations
import networkx as nx
import numpy as np


def compute_topo_metrics(G: nx.Graph, ablation_results: list | None = None) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics = {
        "n_nodes"          : n,
        "n_edges"          : m,
        "graph_density"    : round(nx.density(G), 6),
        "avg_degree"       : round(2*m/n, 3) if n > 0 else 0,
    }

    comps   = list(nx.connected_components(G))
    largest = max(comps, key=len) if comps else set()
    metrics["n_components"]     = len(comps)
    metrics["pct_largest_comp"] = round(100*len(largest)/n, 2) if n > 0 else 0
    metrics["n_bridges"]        = len(list(nx.bridges(G)))
    metrics["avg_path_length_m"]= round(_sample_avg_path(G), 1)

    healed = sum(1 for _,_,d in G.edges(data=True) if d.get("healed"))
    metrics["n_healed_edges"]   = healed
    metrics["healing_rate_pct"] = round(100*healed/m, 2) if m > 0 else 0

    if ablation_results:
        metrics["top5_damage_scores"] = [r["damage_score"] for r in ablation_results[:5]]
        metrics["top5_edges"]         = [list(r["edge"])   for r in ablation_results[:5]]
    else:
        metrics["top5_damage_scores"] = []
        metrics["top5_edges"]         = []

    return metrics


def _sample_avg_path(G, n_samples=300):
    rng   = np.random.default_rng(1)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0
    total, count = 0.0, 0
    for _ in range(n_samples):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            total += nx.shortest_path_length(G, u, v, weight="length_m")
            count += 1
        except nx.NetworkXNoPath:
            pass
    return total / count if count > 0 else 0.0


def print_metrics_table(metrics: dict) -> None:
    print("\n" + "="*50)
    print("  TOPOLOGICAL METRICS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print("="*50 + "\n")