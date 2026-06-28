"""
src/graph/pipeline.py
---------------------
Graph half — Stage A, B, C.

Stage A  Skeletonize : thin road mask to 1-pixel-wide lines
Stage B  Build Graph : intersections=nodes, road segments=edges (with length_m)
Stage C  Heal Gaps   : bridge small breaks caused by tree/shadow occlusion

Gap healing uses Union-Find (Disjoint Set Union) for O(α·n) component
tracking — heals all valid gaps in a single pass with no iteration needed.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import rasterio
from rasterio.transform import xy as transform_xy
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
import sknw

# ── tuneable knobs ─────────────────────────────────────────────────────────────
GAP_HEAL_METRES   = 150.0
MIN_ROAD_METRES   = 10.0
DEGREES_PER_METRE = 1 / 111_320


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_mask(
    mask_path: str,
    gap_heal_metres: float = GAP_HEAL_METRES,
    min_road_metres: float = MIN_ROAD_METRES,
) -> tuple[nx.Graph, dict]:
    """
    Full pipeline: GeoTIFF mask → healed NetworkX road graph.

    Returns
    -------
    G    : nx.Graph
           nodes → x (lon), y (lat)
           edges → length_m (float), healed (bool)
    meta : dict  diagnostic info
    """
    print("[pipeline] Loading mask …")
    mask, transform, crs = _load_mask(mask_path)

    print("[pipeline] Skeletonizing …")
    skel = skeletonize(mask.astype(bool))
    print(f"[pipeline] Skeleton pixels: {skel.sum():,}")

    print("[pipeline] Building graph with sknw …")
    G = _sknw_to_graph(skel, transform)
    print(f"[pipeline] Raw graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    print(f"[pipeline] Removing short fragments < {min_road_metres} m …")
    G = _remove_short_fragments(G, min_road_metres)

    print(f"[pipeline] Healing gaps ≤ {gap_heal_metres} m …")
    n_healed = _heal_gaps(G, gap_heal_metres)

    meta = dict(
        n_nodes       = G.number_of_nodes(),
        n_edges       = G.number_of_edges(),
        n_healed      = n_healed,
        mask_path     = mask_path,
        gap_threshold = gap_heal_metres,
        crs           = str(crs),
    )
    print(f"[pipeline] Done → {meta['n_nodes']} nodes, "
          f"{meta['n_edges']} edges ({n_healed} healed).")
    return G, meta


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_mask(path):
    with rasterio.open(path) as src:
        mask = (src.read(1) > 0).astype(np.uint8)
        return mask, src.transform, src.crs


def _px_to_lonlat(row, col, tf):
    lon, lat = transform_xy(tf, row, col)
    return float(lon), float(lat)


def _dist_m(lon1, lat1, lon2, lat2):
    mid  = np.radians((lat1 + lat2) / 2)
    dlat = np.radians(lat2 - lat1) * 111_320
    dlon = np.radians(lon2 - lon1) * 111_320 * np.cos(mid)
    return float(np.sqrt(dlat**2 + dlon**2))


def _sknw_to_graph(skel, transform):
    G_raw = sknw.build_sknw(skel)
    G = nx.Graph()

    for n, data in G_raw.nodes(data=True):
        r, c = int(data['o'][0]), int(data['o'][1])
        lon, lat = _px_to_lonlat(r, c, transform)
        G.add_node(int(n), x=lon, y=lat, row=r, col=c)

    for u, v, data in G_raw.edges(data=True):
        u, v = int(u), int(v)
        if u not in G.nodes or v not in G.nodes:
            continue
        pts = data.get('pts', np.array([]))
        if len(pts) > 1:
            length = sum(
                _dist_m(*_px_to_lonlat(pts[i][0],   pts[i][1],   transform),
                        *_px_to_lonlat(pts[i+1][0], pts[i+1][1], transform))
                for i in range(len(pts) - 1)
            )
        else:
            length = _dist_m(G.nodes[u]['x'], G.nodes[u]['y'],
                             G.nodes[v]['x'], G.nodes[v]['y'])
        if not G.has_edge(u, v):
            G.add_edge(u, v, length_m=round(length, 2), healed=False)
    return G


def _remove_short_fragments(G, min_metres):
    to_remove = []
    for comp in nx.connected_components(G):
        sub   = G.subgraph(comp)
        total = sum(d.get('length_m', 0) for _, _, d in sub.edges(data=True))
        if total < min_metres:
            to_remove.extend(comp)
    G.remove_nodes_from(to_remove)
    return G


# ── Union-Find (Disjoint Set Union) ──────────────────────────────────────────

class _UF:
    """
    Union-Find with path compression + union by rank.
    Find and Union are effectively O(1) (inverse Ackermann).
    """
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank   = {n: 0  for n in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def same(self, x, y):
        return self.find(x) == self.find(y)


def _heal_gaps(G, threshold_m):
    """
    Bridge gaps using Union-Find for instant component tracking.
    Single pass — no iteration, no slow connectivity queries.
    Heals shortest gaps first.
    """
    uf = _UF(G.nodes())
    for u, v in G.edges():
        uf.union(u, v)       # seed UF with existing edges

    nodes      = list(G.nodes(data=True))
    coords     = np.array([[d['y'], d['x']] for _, d in nodes])
    ids        = [n for n, _ in nodes]
    thresh_deg = threshold_m * DEGREES_PER_METRE

    pairs = list(KDTree(coords).query_pairs(r=thresh_deg))

    # sort by gap distance — shortest gaps healed first
    pairs.sort(key=lambda p: _dist_m(
        G.nodes[ids[p[0]]]['x'], G.nodes[ids[p[0]]]['y'],
        G.nodes[ids[p[1]]]['x'], G.nodes[ids[p[1]]]['y'],
    ))

    n_healed = 0
    for i, j in pairs:
        u, v = ids[i], ids[j]
        if G.has_edge(u, v):
            continue
        if uf.same(u, v):
            continue            # already connected — skip
        d = _dist_m(G.nodes[u]['x'], G.nodes[u]['y'],
                    G.nodes[v]['x'], G.nodes[v]['y'])
        G.add_edge(u, v, length_m=round(d, 2), healed=True)
        uf.union(u, v)          # update component tracking instantly
        n_healed += 1

    return n_healed