"""
dashboard/build_map.py
----------------------
Builds the interactive Folium HTML map.
Handles large graphs (8000+ edges) by using GeoJson instead of
individual PolyLine calls — much faster rendering in browser.

Colour coding:
  🔴 Red    = critical road (top 10% betweenness)
  🟡 Orange = medium priority
  🟢 Green  = normal road
  🟣 Purple dashed = healed gap
"""

from __future__ import annotations
import os, json
import folium
from folium.plugins import FastMarkerCluster
import networkx as nx
import numpy as np

HIGH_THRESHOLD   = 0.005
MEDIUM_THRESHOLD = 0.001

# for large graphs, only render top N critical roads in detail
# rest go into a single GeoJSON layer (much faster)
MAX_INDIVIDUAL_LINES = 500


def build_dashboard(
    G: nx.Graph,
    results: dict,
    metrics: dict,
    output_path: str = "dashboard/jaipur_road_criticality.html",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_edges  = G.number_of_edges()
    is_large = n_edges > MAX_INDIVIDUAL_LINES
    print(f"[dashboard] {n_edges} edges — using {'GeoJSON' if is_large else 'PolyLine'} mode")

    m = folium.Map(location=[26.90, 75.82], zoom_start=13,
                   tiles="CartoDB positron")

    edge_bc = results.get("edge_centrality", {})
    top_set = set(
        (min(u,v), max(u,v))
        for u,v in results.get("top_edges", [])[:50]
    )
    abl_map = {
        (min(r["edge"][0], r["edge"][1]), max(r["edge"][0], r["edge"][1])): r
        for r in results.get("ablation_results", [])
    }

    if is_large:
        _add_geojson_layers(m, G, edge_bc, top_set, abl_map)
    else:
        _add_polyline_layers(m, G, edge_bc, top_set, abl_map)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_stats_panel(metrics)))
    m.get_root().html.add_child(folium.Element(_legend()))

    m.save(output_path)
    print(f"[dashboard] Saved → {output_path}")
    return output_path


def _edge_coords(G, u, v):
    return (
        [G.nodes[u].get("y",0), G.nodes[u].get("x",0)],
        [G.nodes[v].get("y",0), G.nodes[v].get("x",0)],
    )


def _edge_color(bc, is_healed, is_top):
    if is_healed:   return "#9b59b6"
    if is_top:      return "#c0392b"
    if bc >= HIGH_THRESHOLD:   return "#e74c3c"
    if bc >= MEDIUM_THRESHOLD: return "#f39c12"
    return "#27ae60"


def _add_geojson_layers(m, G, edge_bc, top_set, abl_map):
    """
    For large graphs: group edges into GeoJSON FeatureCollections.
    Browser renders these as a single draw call — much faster.
    """
    features_normal  = []
    features_medium  = []
    features_high    = []
    features_healed  = []

    for u, v, data in G.edges(data=True):
        lon_u = G.nodes[u].get("x",0); lat_u = G.nodes[u].get("y",0)
        lon_v = G.nodes[v].get("x",0); lat_v = G.nodes[v].get("y",0)
        bc        = data.get("betweenness", 0.0)
        is_healed = data.get("healed", False)
        length_m  = data.get("length_m", 0)
        ekey      = (min(u,v), max(u,v))
        is_top    = ekey in top_set
        abl       = abl_map.get(ekey, {})

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon_u, lat_u], [lon_v, lat_v]]
            },
            "properties": {
                "length_m"   : round(length_m, 1),
                "betweenness": round(bc, 6),
                "damage"     : abl.get("damage_score", "—"),
                "critical"   : "⚠️ CRITICAL" if is_top else "",
                "healed"     : "🔧 Healed" if is_healed else "",
            }
        }

        if is_healed:
            features_healed.append(feature)
        elif bc >= HIGH_THRESHOLD or is_top:
            features_high.append(feature)
        elif bc >= MEDIUM_THRESHOLD:
            features_medium.append(feature)
        else:
            features_normal.append(feature)

    def make_layer(features, name, color, weight, dash=None, show=True):
        if not features:
            return
        style = {"color": color, "weight": weight, "opacity": 0.8}
        if dash:
            style["dashArray"] = dash
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            name=name,
            style_function=lambda x, s=style: s,
            tooltip=folium.GeoJsonTooltip(
                fields=["length_m","betweenness","damage","critical","healed"],
                aliases=["Length (m)","Betweenness","Damage score","",""],
            ),
            show=show,
        ).add_to(m)

    make_layer(features_normal, "🟢 Normal roads",    "#27ae60", 1.5)
    make_layer(features_medium, "🟡 Medium priority", "#f39c12", 2.5)
    make_layer(features_high,   "🔴 Critical roads",  "#e74c3c", 4,   show=True)
    make_layer(features_healed, "🟣 Healed gaps",     "#9b59b6", 2,   dash="6 4")


def _add_polyline_layers(m, G, edge_bc, top_set, abl_map):
    """For small graphs: individual PolyLines with full tooltips."""
    fg_normal = folium.FeatureGroup(name="🟢 Normal roads",    show=True)
    fg_medium = folium.FeatureGroup(name="🟡 Medium priority", show=True)
    fg_high   = folium.FeatureGroup(name="🔴 Critical roads",  show=True)
    fg_healed = folium.FeatureGroup(name="🟣 Healed gaps",     show=True)
    fg_nodes  = folium.FeatureGroup(name="⚪ Intersections",    show=False)

    for u, v, data in G.edges(data=True):
        lat_u = G.nodes[u].get("y",0); lon_u = G.nodes[u].get("x",0)
        lat_v = G.nodes[v].get("y",0); lon_v = G.nodes[v].get("x",0)
        coords    = [[lat_u,lon_u],[lat_v,lon_v]]
        bc        = data.get("betweenness", 0.0)
        is_healed = data.get("healed", False)
        length_m  = data.get("length_m", 0)
        ekey      = (min(u,v), max(u,v))
        is_top    = ekey in top_set
        abl       = abl_map.get(ekey, {})
        tip = (f"Length: {length_m:.1f}m | BC: {bc:.5f} | "
               f"Damage: {abl.get('damage_score','—')} "
               f"{'⚠️ CRITICAL' if is_top else ''}"
               f"{'🔧 Healed' if is_healed else ''}")

        if is_healed:
            folium.PolyLine(coords, color="#9b59b6", weight=2,
                            dash_array="6 4", tooltip=tip).add_to(fg_healed)
        elif bc >= HIGH_THRESHOLD or is_top:
            folium.PolyLine(coords, color="#e74c3c",
                            weight=5 if is_top else 3,
                            tooltip=tip).add_to(fg_high)
        elif bc >= MEDIUM_THRESHOLD:
            folium.PolyLine(coords, color="#f39c12",
                            weight=2, tooltip=tip).add_to(fg_medium)
        else:
            folium.PolyLine(coords, color="#27ae60",
                            weight=1.5, tooltip=tip).add_to(fg_normal)

    for node, data in G.nodes(data=True):
        bc  = data.get("betweenness", 0)
        deg = G.degree(node)
        folium.CircleMarker(
            location=[data.get("y",0), data.get("x",0)],
            radius=3 if bc > 0.01 else 2,
            color="#c0392b" if bc > 0.01 else "#7f8c8d",
            fill=True, fill_opacity=0.7,
            tooltip=f"Node {node} | Degree: {deg} | BC: {bc:.5f}",
        ).add_to(fg_nodes)

    for fg in [fg_normal, fg_medium, fg_high, fg_healed, fg_nodes]:
        fg.add_to(m)


def _stats_panel(metrics):
    top5 = metrics.get("top5_damage_scores", [])
    top5_str = ", ".join(f"{s:.3f}" for s in top5) if top5 else "N/A"
    return f"""
    <div style="position:fixed;top:10px;left:60px;z-index:1000;
        background:rgba(255,255,255,0.93);padding:12px 16px;
        border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
        font-family:'Segoe UI',sans-serif;font-size:12px;max-width:260px;">
      <b style="font-size:14px;">🗺️ Route Resilience — Jaipur</b>
      <hr style="margin:6px 0">
      Nodes (intersections): <b>{metrics.get('n_nodes','?')}</b><br>
      Edges (road segments): <b>{metrics.get('n_edges','?')}</b><br>
      Connected components:  <b>{metrics.get('n_components','?')}</b><br>
      Largest component:     <b>{metrics.get('pct_largest_comp','?')}%</b><br>
      Avg path length:       <b>{metrics.get('avg_path_length_m','?')} m</b><br>
      <hr style="margin:6px 0">
      Healed gaps:   <b>{metrics.get('n_healed_edges','?')}</b><br>
      Bridge roads:  <b>{metrics.get('n_bridges','?')}</b><br>
      <hr style="margin:6px 0">
      <b>Top-5 damage scores:</b><br>{top5_str}
    </div>"""


def _legend():
    return """
    <div style="position:fixed;bottom:30px;left:60px;z-index:1000;
        background:rgba(255,255,255,0.93);padding:10px 14px;
        border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
        font-family:'Segoe UI',sans-serif;font-size:12px;">
      <b>Legend</b><br>
      <span style="color:#e74c3c">━━</span> Critical road<br>
      <span style="color:#f39c12">━━</span> Medium priority<br>
      <span style="color:#27ae60">━━</span> Normal road<br>
      <span style="color:#9b59b6">┅┅</span> Healed gap<br>
      <span style="color:#c0392b">●</span> Critical intersection
    </div>"""