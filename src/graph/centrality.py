"""
src/graph/centrality.py
-----------------------
Answers: which roads are the most critical?

Method 1 — Betweenness Centrality
    Count how many shortest paths between all city pairs pass through each road.
    High score = road carries a lot of traffic = CRITICAL.

Method 2 — Ablation
    Remove each top road temporarily, measure how badly the network degrades.
    Produces a damage_score per road (0–1, higher = more critical).
"""

from __future__ import annotations
import networkx as nx
import numpy as np

TOP_K_ABLATION = 20


def analyse(G: nx.Graph, top_k: int = TOP_K_ABLATION) -> dict:
    """
    Run centrality + ablation on road graph G.

    Returns dict with keys:
      edge_centrality   {(u,v): float}
      node_centrality   {node: float}
      top_edges         list of (u,v) sorted by centrality desc
      ablation_results  list of dicts, one per top-k edge
      critical_edges    list of (u,v) — bridges (removal disconnects graph)
    """
    print("[centrality] Edge betweenness centrality …")
    edge_bc = _edge_betweenness(G)

    print("[centrality] Node betweenness centrality …")
    node_bc = nx.betweenness_centrality(G, weight="length_m", normalized=True)

    nx.set_edge_attributes(G, {
        (u,v): edge_bc.get((u,v), edge_bc.get((v,u), 0.0))
        for u,v in G.edges()
    }, "betweenness")
    nx.set_node_attributes(G, node_bc, "betweenness")

    top_edges = sorted(
        set((min(u,v), max(u,v)) for u,v in edge_bc),
        key=lambda e: edge_bc.get(e, 0),
        reverse=True,
    )

    print(f"[centrality] Ablation on top-{top_k} edges …")
    ablation_results = _ablation(G, top_edges[:top_k], edge_bc)

    print("[centrality] Finding bridge edges …")
    critical_edges = list(nx.bridges(G))
    print(f"[centrality] Done. {len(critical_edges)} bridges found.")

    return dict(
        edge_centrality  = edge_bc,
        node_centrality  = node_bc,
        top_edges        = top_edges,
        ablation_results = ablation_results,
        critical_edges   = critical_edges,
    )


def _edge_betweenness(G: nx.Graph) -> dict:
    raw = nx.edge_betweenness_centrality(G, weight="length_m", normalized=True)
    sym = {}
    for (u,v), s in raw.items():
        sym[(u,v)] = s
        sym[(v,u)] = s
    return sym


def _ablation(G, edges_to_test, edge_bc) -> list[dict]:
    baseline_comp     = nx.number_connected_components(G)
    baseline_avg_path = _sample_avg_path(G)
    results, seen = [], set()

    for u, v in edges_to_test:
        key = (min(u,v), max(u,v))
        if key in seen:
            continue
        seen.add(key)

        edata = G.edges[u,v].copy()
        G.remove_edge(u, v)

        n_comp    = nx.number_connected_components(G)
        conn_loss = n_comp > baseline_comp
        new_avg   = _sample_avg_path(G)
        path_inc  = (100*(new_avg - baseline_avg_path)/baseline_avg_path
                     if baseline_avg_path > 0 else 0.0)

        bc     = edge_bc.get((u,v), edge_bc.get((v,u), 0.0))
        damage = (0.4 * bc
                  + 0.4 * (1.0 if conn_loss else 0.0)
                  + 0.2 * min(path_inc/100, 1.0))

        results.append(dict(
            edge                  = (u, v),
            betweenness           = round(bc, 6),
            connectivity_loss     = conn_loss,
            n_components_after    = n_comp,
            avg_path_increase_pct = round(path_inc, 2),
            damage_score          = round(damage, 4),
            length_m              = edata.get("length_m", 0),
            healed                = edata.get("healed", False),
        ))
        G.add_edge(u, v, **edata)

    results.sort(key=lambda r: r["damage_score"], reverse=True)
    return results


def _sample_avg_path(G, n_samples=200, seed=0):
    rng   = np.random.default_rng(seed)
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