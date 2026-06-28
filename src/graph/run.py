"""
src/graph/run.py
----------------
Master runner — one command runs the entire graph pipeline.

Usage:
    python -m src.graph.run --mask data/inference/jaipur/predicted_mask.tif
"""

import argparse, json, time

from src.graph.pipeline    import build_graph_from_mask
from src.graph.centrality  import analyse
from src.eval.topo_metrics import compute_topo_metrics, print_metrics_table
from dashboard.build_map   import build_dashboard

DEFAULT_MASK = "data/inference/jaipur/predicted_mask.tif"
DEFAULT_OUT  = "dashboard/jaipur_road_criticality.html"


def run(mask_path: str, output_html: str) -> None:
    t0 = time.time()
    print("\n" + "="*60)
    print("  ROUTE RESILIENCE — Graph Pipeline")
    print("="*60)
    print(f"  Mask  : {mask_path}")
    print(f"  Output: {output_html}")
    print("="*60 + "\n")

    # Step 1+2 — skeletonize + build graph
    G, pipe_meta = build_graph_from_mask(mask_path)

    # Step 3 — centrality + ablation
    results = analyse(G)

    # Step 4 — topological metrics
    metrics = compute_topo_metrics(G, results["ablation_results"])
    print_metrics_table(metrics)

    # Step 5 — Folium dashboard
    build_dashboard(G, results, metrics, output_html)

    # Step 6 — save metrics JSON for submission table
    json_path = output_html.replace(".html", "_metrics.json")
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            safe[k] = [list(e) if isinstance(e, tuple) else e for e in v]
        else:
            safe[k] = v
    with open(json_path, "w") as f:
        json.dump(safe, f, indent=2)

    print(f"[run] Metrics JSON → {json_path}")
    print(f"\n✅ Done in {time.time()-t0:.1f}s")
    print(f"   Open in browser: {output_html}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default=DEFAULT_MASK)
    parser.add_argument("--out",  default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.mask, args.out)