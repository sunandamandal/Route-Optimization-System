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
    "shortest": "#3b82f6",   # Vibrant Blue
    "fastest": "#f97316",    # Vibrant Orange
    "energy": "#10b981",     # Neon Emerald Green
    "balanced": "#a855f7"    # Radiant Purple
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


def create_india_route_map(
    source_loc: Any,
    dest_loc: Any,
    route_coords: List[Tuple[float, float]],
    active_route: Any,
    selected_mode: str = "energy",
    alternative_routes: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    candidate_routes: Optional[Dict[str, Any]] = None
) -> folium.Map:
    """
    Constructs an interactive Folium map for Real-World India Navigation.
    Features:
      - Automatic bounds fitting to frame any distance (city to inter-state).
      - Display of alternative candidate routes (dashed subtle lines) when available.
      - Real GPS highway polylines with mode-specific vibrant glow for selected route.
      - Start (Green) and Destination (Red) custom pins with formatted names & coordinates.
      - Real Highway EV Fast-Charging Plaza pins along the corridor.
      - Full OpenStreetMap legal attribution.
    """
    # 1. Map Center calculation
    center_lat = (source_loc.lat + dest_loc.lat) / 2.0
    center_lon = (source_loc.lon + dest_loc.lon) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        attr="Data © OpenStreetMap contributors, ODbL 1.0",
        control_scale=True
    )

    # 2. Draw Alternative Candidate Routes (Subtle dashed polylines)
    if alternative_routes:
        for alt_mode, alt_coords in alternative_routes.items():
            if alt_mode.lower() != selected_mode.lower() and alt_coords and len(alt_coords) > 1:
                alt_r = candidate_routes.get(alt_mode) if candidate_routes else None
                if alt_r:
                    tip = (
                        f"Alternative ({alt_mode.upper()} Mode): "
                        f"{alt_r.total_distance_km:.1f} km | {alt_r.total_time_min:.0f} min | "
                        f"{alt_r.total_energy_kwh:.2f} kWh"
                    )
                else:
                    tip = f"Alternative Route ({alt_mode.upper()} Mode)"

                folium.PolyLine(
                    locations=alt_coords,
                    color="#64748b",
                    weight=3.5,
                    opacity=0.65,
                    dash_array="6, 8",
                    tooltip=tip
                ).add_to(m)

    # 3. Draw Selected Active Route Polyline (Thick vibrant glow)
    if route_coords and len(route_coords) > 1:
        route_color = MODE_COLORS.get(selected_mode.lower(), "#10b981")

        # Outer depth glow
        folium.PolyLine(
            locations=route_coords,
            color="#0f172a",
            weight=8.5,
            opacity=0.50
        ).add_to(m)

        # Vibrant active navigation path
        folium.PolyLine(
            locations=route_coords,
            color=route_color,
            weight=5.5,
            opacity=0.98,
            tooltip=(
                f"Selected {selected_mode.upper()} Route: "
                f"{active_route.total_distance_km:.1f} km | "
                f"{active_route.total_time_min:.0f} min | "
                f"{active_route.total_energy_kwh:.2f} kWh"
            )
        ).add_to(m)

    # 4. Auto-Fit Bounds to frame the entire journey with padding
    all_points: List[Tuple[float, float]] = []
    if route_coords:
        all_points.extend(route_coords)
    if alternative_routes:
        for a_coords in alternative_routes.values():
            if a_coords:
                all_points.extend(a_coords)
    if hasattr(source_loc, "lat") and hasattr(source_loc, "lon"):
        all_points.append((source_loc.lat, source_loc.lon))
    if hasattr(dest_loc, "lat") and hasattr(dest_loc, "lon"):
        all_points.append((dest_loc.lat, dest_loc.lon))

    if all_points:
        all_lats = [pt[0] for pt in all_points]
        all_lons = [pt[1] for pt in all_points]
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        lat_span = max(0.01, (max_lat - min_lat) * 0.05)
        lon_span = max(0.01, (max_lon - min_lon) * 0.05)
        m.fit_bounds(
            [[min_lat - lat_span, min_lon - lon_span], [max_lat + lat_span, max_lon + lon_span]],
            padding=(30, 30)
        )

    # 5. Source Marker (Green Pin)
    src_popup = (
        f"<div style='font-family:Segoe UI, sans-serif; min-width:190px; color:#1e293b;'>"
        f"<h4 style='margin:0 0 6px 0; color:#10b981; font-weight:700;'>🟢 START LOCATION</h4>"
        f"<b>{source_loc.display_name}</b><br>"
        f"<span style='color:#64748b;'>GPS:</span> <code>{source_loc.lat:.4f}° N, {source_loc.lon:.4f}° E</code><br>"
        f"<span style='color:#64748b;'>Elevation:</span> <b>{source_loc.elevation_m:.0f} m</b>"
        f"</div>"
    )
    folium.Marker(
        location=[source_loc.lat, source_loc.lon],
        popup=folium.Popup(src_popup, max_width=320),
        tooltip=f"Start: {source_loc.short_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    # 6. Destination Marker (Red Flag)
    dst_popup = (
        f"<div style='font-family:Segoe UI, sans-serif; min-width:190px; color:#1e293b;'>"
        f"<h4 style='margin:0 0 6px 0; color:#ef4444; font-weight:700;'>🏁 DESTINATION</h4>"
        f"<b>{dest_loc.display_name}</b><br>"
        f"<span style='color:#64748b;'>GPS:</span> <code>{dest_loc.lat:.4f}° N, {dest_loc.lon:.4f}° E</code><br>"
        f"<span style='color:#64748b;'>Elevation:</span> <b>{dest_loc.elevation_m:.0f} m</b>"
        f"</div>"
    )
    folium.Marker(
        location=[dest_loc.lat, dest_loc.lon],
        popup=folium.Popup(dst_popup, max_width=320),
        tooltip=f"Destination: {dest_loc.short_name}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
    ).add_to(m)

    # 7. Highway EV Fast Charging Stops along Route
    if active_route.charging_stops and route_coords:
        num_stops = len(active_route.charging_stops)
        coord_step = len(route_coords) // (num_stops + 1)
        for idx, stop_desc in enumerate(active_route.charging_stops, 1):
            pt_idx = min(len(route_coords) - 1, idx * coord_step)
            stop_lat, stop_lon = route_coords[pt_idx]

            charger_popup = (
                f"<div style='font-family:Segoe UI, sans-serif; min-width:190px; color:#1e293b;'>"
                f"<h4 style='margin:0 0 6px 0; color:#0284c7; font-weight:700;'>⚡ Highway Fast Charger</h4>"
                f"<b>{stop_desc}</b><br>"
                f"<span style='color:#64748b;'>Corridor Point:</span> <code>{stop_lat:.4f}° N, {stop_lon:.4f}° E</code>"
                f"</div>"
            )
            folium.Marker(
                location=[stop_lat, stop_lon],
                popup=folium.Popup(charger_popup, max_width=290),
                tooltip=f"⚡ {stop_desc}",
                icon=folium.Icon(color="darkblue", icon="bolt", prefix="fa")
            ).add_to(m)

    return m

