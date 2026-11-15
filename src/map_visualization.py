"""
src/map_visualization.py
Interactive Folium mapping module for EV Smart Navigation.
Renders the city road network, charging stations, and optimized navigation routes.
"""

from typing import Dict, Any, List, Optional, Tuple
import folium
from folium import plugins
import networkx as nx

# Reference anchor coordinates for VoltCity
BASE_LAT = 12.9716
BASE_LON = 77.5946
SCALE_LAT = 0.018  # Spacing scale for Y coordinate
SCALE_LON = 0.022  # Spacing scale for X coordinate

# Traffic colors
TRAFFIC_COLORS = {
    "low": "#2ecc71",     # Green
    "medium": "#f39c12",  # Amber/Orange
    "high": "#e74c3c"     # Red
}

# Mode route overlay colors
MODE_COLORS = {
    "shortest": "#2980b9",   # Royal Blue
    "fastest": "#e67e22",    # Orange
    "energy": "#27ae60",     # Emerald Green
    "balanced": "#8e44ad"    # Purple
}


def node_to_latlon(pos: Tuple[float, float]) -> Tuple[float, float]:
    """Converts local (x, y) planar coordinates to geographic (lat, lon)."""
    x, y = pos
    lat = BASE_LAT + (y - 4.5) * SCALE_LAT
    lon = BASE_LON + (x - 4.5) * SCALE_LON
    return (lat, lon)


def create_city_folium_map(
    graph: nx.Graph,
    selected_path: Optional[List[str]] = None,
    source_node: Optional[str] = None,
    destination_node: Optional[str] = None,
    mode_name: str = "energy"
) -> folium.Map:
    """
    Constructs an interactive Folium map representing the city grid,
    EV charging stations, and the active navigation path.

    Parameters:
        graph (nx.Graph): Road network graph.
        selected_path (list, optional): List of node IDs forming the chosen route.
        source_node (str, optional): Start node ID.
        destination_node (str, optional): Destination node ID.
        mode_name (str): Navigation mode name to determine path styling.

    Returns:
        folium.Map: The rendered Folium map instance.
    """
    # 1. Initialize Folium Map centered on the city
    m = folium.Map(
        location=[BASE_LAT, BASE_LON],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 2. Draw All Base Road Network Segments
    for u, v, data in graph.edges(data=True):
        pos_u = graph.nodes[u]["pos"]
        pos_v = graph.nodes[v]["pos"]
        coords = [node_to_latlon(pos_u), node_to_latlon(pos_v)]
        traffic_lvl = data.get("traffic_level", "low").lower()
        edge_color = TRAFFIC_COLORS.get(traffic_lvl, "#95a5a6")

        tooltip_html = (
            f"<b>Road {u} &harr; {v}</b><br>"
            f"Distance: {data.get('distance_km', 0)} km<br>"
            f"Speed Limit: {data.get('speed_limit_kmh', 50)} km/h<br>"
            f"Traffic: {traffic_lvl.upper()}<br>"
            f"Slope: {data.get('slope_percent', 0.0):+.2f}%"
        )

        folium.PolyLine(
            locations=coords,
            color=edge_color,
            weight=3.0,
            opacity=0.55,
            tooltip=tooltip_html
        ).add_to(m)

    # 3. Draw Selected Route Polyline (Thick vibrant overlay)
    if selected_path and len(selected_path) > 1:
        path_coords = [node_to_latlon(graph.nodes[n]["pos"]) for n in selected_path]
        route_color = MODE_COLORS.get(mode_name.lower(), "#2980b9")

        # Outer glow
        folium.PolyLine(
            locations=path_coords,
            color="#2c3e50",
            weight=8.0,
            opacity=0.4
        ).add_to(m)

        # Main active path
        folium.PolyLine(
            locations=path_coords,
            color=route_color,
            weight=5.5,
            opacity=0.95,
            tooltip=f"Active Route ({mode_name.upper()} Mode)"
        ).add_to(m)

    # 4. Draw All Nodes & EV Charging Stations
    for node_id, data in graph.nodes(data=True):
        lat, lon = node_to_latlon(data["pos"])
        name = data.get("name", f"Intersection {node_id}")
        elevation = data.get("elevation_m", 120.0)
        has_charger = data.get("has_charging_station", False)
        charger_power = data.get("charging_capacity_kw", 0)

        # Popup information
        popup_html = (
            f"<div style='font-family:sans-serif; min-width:140px;'>"
            f"<h4 style='margin:0 0 4px 0; color:#2c3e50;'>Node {node_id}</h4>"
            f"<b>{name}</b><br>"
            f"Elevation: {elevation:.0f} m<br>"
        )
        if has_charger:
            popup_html += (
                f"<span style='color:#27ae60; font-weight:bold;'>&#9889; EV Fast Charger: {charger_power} kW</span>"
            )
        popup_html += "</div>"

        # Distinction for Source, Destination, Charging, and Regular nodes
        if source_node and node_id == source_node:
            # Source Node (Green Pin)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"<b>[START] Node {node_id}: {name}</b>", max_width=250),
                tooltip=f"Start: Node {node_id} ({name})",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(m)

        elif destination_node and node_id == destination_node:
            # Destination Node (Red Flag Pin)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"<b>[DESTINATION] Node {node_id}: {name}</b>", max_width=250),
                tooltip=f"Destination: Node {node_id} ({name})",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
            ).add_to(m)

        elif has_charger:
            # EV Charging Hub (Emerald Marker with Lightning bolt)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Node {node_id}: {name} (⚡ {charger_power}kW Charger)",
                icon=folium.Icon(color="darkgreen", icon="bolt", prefix="fa")
            ).add_to(m)

        else:
            # Regular city intersection
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#2c3e50",
                weight=1.5,
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Node {node_id}: {name} ({elevation:.0f}m)"
            ).add_to(m)

    return m
