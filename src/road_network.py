"""
src/road_network.py
Module for representing, loading, and visualizing the city road network
for Electric Vehicle (EV) Route Optimization using NetworkX and Matplotlib.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "road_network.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "city_road_network.png"

# Color mappings for traffic levels
TRAFFIC_COLORS = {
    "low": "#27ae60",     # Green: smooth flow, low energy stop-and-go
    "medium": "#e67e22",  # Orange: moderate congestion
    "high": "#c0392b"     # Red: heavy congestion, higher idle energy drain
}


def load_road_network(file_path: Optional[str or Path] = None) -> nx.Graph:
    """
    Loads the city road network data from JSON into a NetworkX Graph.

    Parameters:
        file_path (str or Path, optional): Path to road_network.json.
                                           Defaults to 'data/road_network.json'.

    Returns:
        nx.Graph: Graph object populated with nodes (intersections) and edges (roads).
    """
    if file_path is None:
        file_path = DEFAULT_DATA_PATH
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Road network data file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Undirected graph for bidirectional urban roads
    G = nx.Graph(city_name=data.get("city_name", "VoltCity"))

    # 1. Add Nodes (Intersections & Key City Points)
    for node in data.get("nodes", []):
        node_id = node["id"]
        G.add_node(
            node_id,
            name=node["name"],
            pos=(float(node["pos"][0]), float(node["pos"][1])),
            elevation_m=float(node["elevation_m"]),
            has_charging_station=bool(node.get("has_charging_station", False)),
            charging_capacity_kw=int(node.get("charging_capacity_kw", 0))
        )

    # 2. Add Edges (Roads connecting locations)
    for road in data.get("roads", []):
        u = road["source"]
        v = road["target"]
        G.add_edge(
            u,
            v,
            distance_km=float(road["distance_km"]),
            travel_time_min=float(road["travel_time_min"]),
            traffic_level=str(road["traffic_level"]).lower(),
            elevation_change_m=float(road["elevation_change_m"]),
            slope_percent=float(road["slope_percent"]),
            speed_limit_kmh=int(road["speed_limit_kmh"])
        )

    return G


def get_network_summary(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculates summary statistics of the city road network.

    Parameters:
        G (nx.Graph): The city graph.

    Returns:
        dict: Detailed statistics including node count, road count,
              EV stations, and distance stats.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    charging_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("has_charging_station")
    ]

    total_distance_km = sum(
        attr.get("distance_km", 0.0) for _, _, attr in G.edges(data=True)
    )

    traffic_counts = {"low": 0, "medium": 0, "high": 0}
    for _, _, attr in G.edges(data=True):
        lvl = attr.get("traffic_level", "low")
        traffic_counts[lvl] = traffic_counts.get(lvl, 0) + 1

    summary = {
        "city_name": G.graph.get("city_name", "VoltCity"),
        "total_nodes": num_nodes,
        "total_roads": num_edges,
        "charging_stations_count": len(charging_nodes),
        "charging_station_nodes": charging_nodes,
        "total_network_distance_km": round(total_distance_km, 2),
        "avg_road_distance_km": round(total_distance_km / max(1, num_edges), 2),
        "traffic_breakdown": traffic_counts
    }
    return summary


def display_road_network(
    G: nx.Graph,
    save_path: Optional[str or Path] = None,
    show: bool = False,
    title: Optional[str] = None
) -> str:
    """
    Renders and visualizes the city road network graph with:
    - Node positions based on realistic 2D city coordinates
    - Visual markers distinguishing EV Charging Stations from standard intersections
    - Color-coded road edges reflecting traffic levels (low: green, medium: orange, high: red)
    - Edge labels displaying distance (km) and speed limit (km/h)

    Parameters:
        G (nx.Graph): NetworkX graph of the city.
        save_path (str or Path, optional): File path to save the generated image.
                                           Defaults to 'outputs/city_road_network.png'.
        show (bool): If True, calls plt.show() to open the interactive plot window.
        title (str, optional): Custom figure title.

    Returns:
        str: Absolute or relative path where the visualization image was saved.
    """
    if save_path is None:
        save_path = DEFAULT_OUTPUT_PATH
    else:
        save_path = Path(save_path)

    # Ensure parent output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    # Extract spatial coordinates
    pos = {node: data["pos"] for node, data in G.nodes(data=True)}

    # Separate standard nodes from EV charging stations
    regular_nodes = [
        n for n, d in G.nodes(data=True) if not d.get("has_charging_station", False)
    ]
    charging_nodes = [
        n for n, d in G.nodes(data=True) if d.get("has_charging_station", False)
    ]

    # Draw Edges partitioned by traffic level
    for traffic_lvl, color in TRAFFIC_COLORS.items():
        edges_subset = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("traffic_level") == traffic_lvl
        ]
        if edges_subset:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_subset,
                edge_color=color,
                width=3.2 if traffic_lvl == "high" else 2.5,
                alpha=0.85,
                ax=ax
            )

    # Draw Regular Nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=regular_nodes,
        node_size=650,
        node_color="#3498db",
        edgecolors="#2c3e50",
        linewidths=1.8,
        ax=ax
    )

    # Draw EV Charging Station Nodes (Highlighted in emerald green with double ring effect)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=charging_nodes,
        node_size=900,
        node_color="#2ecc71",
        edgecolors="#1b5e20",
        linewidths=2.5,
        ax=ax
    )

    # Node Labels (Node ID: A, B, C...)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=11,
        font_weight="bold",
        font_color="#ffffff",
        ax=ax
    )

    # Descriptive Subtitle Labels placed slightly above each node
    for node, (x, y) in pos.items():
        attr = G.nodes[node]
        name = attr.get("name", "")
        elev = attr.get("elevation_m", 0)
        is_cs = attr.get("has_charging_station", False)
        prefix = "[EV] " if is_cs else ""
        label_text = f"{prefix}{name}\n({int(elev)}m)"
        ax.text(
            x,
            y + 0.30,
            label_text,
            fontsize=8,
            fontweight="semibold",
            ha="center",
            va="bottom",
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f1f2f6", edgecolor="#ced6e0", alpha=0.9)
        )

    # Edge Labels (Distance & Speed limit)
    edge_labels = {
        (u, v): f"{d['distance_km']}km\n{d['speed_limit_kmh']}km/h"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#2f3542",
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#dfe4ea", alpha=0.85),
        ax=ax
    )

    # Custom Legend
    legend_elements = [
        mpatches.Patch(color=TRAFFIC_COLORS["low"], label="Traffic: Low (Flowing)"),
        mpatches.Patch(color=TRAFFIC_COLORS["medium"], label="Traffic: Medium (Normal)"),
        mpatches.Patch(color=TRAFFIC_COLORS["high"], label="Traffic: High (Congested)"),
        plt.Line2D([0], [0], marker="o", color="w", label="EV Charging Station Node",
                   markerfacecolor="#2ecc71", markeredgecolor="#1b5e20", markersize=12, markeredgewidth=2),
        plt.Line2D([0], [0], marker="o", color="w", label="Standard Intersection Node",
                   markerfacecolor="#3498db", markeredgecolor="#2c3e50", markersize=10, markeredgewidth=1.5)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#bdc3c7",
        fontsize=9,
        title="Map Legend",
        title_fontsize=10
    )

    chart_title = title or f"{G.graph.get('city_name', 'VoltCity')} - EV Smart Navigation Road Network"
    ax.set_title(
        chart_title,
        fontsize=14,
        fontweight="bold",
        pad=18,
        color="#2c3e50"
    )

    # Adjust axis bounds with margin
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.2)
    ax.set_ylim(min(all_y) - 0.8, max(all_y) + 1.2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Grid line accents
    ax.grid(color="#eceff1", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return str(save_path)


if __name__ == "__main__":
    print("Testing road network module...")
    graph = load_road_network()
    summary = get_network_summary(graph)
    print("\n--- Road Network Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    saved_img = display_road_network(graph)
    print(f"\nGraph visualization successfully saved to: {saved_img}")
