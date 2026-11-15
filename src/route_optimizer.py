"""
src/route_optimizer.py
Route Optimization Engine for Smart Electric Vehicle (EV) Navigation.
Implements Dijkstra's Algorithm and A* Search Algorithm across multiple optimization modes:
  - Mode 1: Shortest Route (Minimum distance)
  - Mode 2: Fastest Route (Minimum travel time with traffic congestion)
  - Mode 3: Energy Efficient Route (Minimum predicted kWh from ANN)
  - Mode 4: Balanced Route (Multi-objective weighted combination)
"""

import math
import heapq
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.road_network import load_road_network
from src.energy_model import EVEnergyPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUTE_PLOT_PATH = PROJECT_ROOT / "outputs" / "route_comparison.png"


@dataclass
class RouteSegment:
    """Represents a single road edge along a journey."""
    source: str
    target: str
    distance_km: float
    travel_time_min: float
    energy_kwh: float
    traffic_level: str
    slope_percent: float
    speed_limit_kmh: int


@dataclass
class RouteResult:
    """Encapsulates the complete optimized route information."""
    mode: str
    algorithm: str
    source: str
    destination: str
    path: List[str]
    total_distance_km: float
    total_time_min: float
    total_energy_kwh: float
    charging_stops: List[str]
    segments: List[RouteSegment] = field(default_factory=list)
    nodes_evaluated: int = 0

    def summary(self) -> str:
        stops_str = ", ".join(self.charging_stops) if self.charging_stops else "None"
        return (
            f"Mode: {self.mode.upper()} | Algorithm: {self.algorithm.upper()}\n"
            f"  Path              : {' -> '.join(self.path)}\n"
            f"  Total Distance    : {self.total_distance_km:.2f} km\n"
            f"  Estimated Time    : {self.total_time_min:.1f} min\n"
            f"  Energy Consumed   : {self.total_energy_kwh:.3f} kWh\n"
            f"  EV Charging Stops : {stops_str}\n"
            f"  Nodes Evaluated   : {self.nodes_evaluated}"
        )


class EVRouteOptimizer:
    """
    Intelligent EV Smart Navigation Router using NetworkX graphs and ANN energy predictions.
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        energy_predictor: Optional[EVEnergyPredictor] = None,
        vehicle_load_kg: float = 75.0,
        temperature_c: float = 20.0
    ):
        """
        Initializes the route optimizer and constructs a weighted directed graph.
        """
        self.base_graph = graph if graph is not None else load_road_network()
        self.predictor = energy_predictor if energy_predictor is not None else EVEnergyPredictor()
        self.vehicle_load_kg = vehicle_load_kg
        self.temperature_c = temperature_c

        # Build directed graph with direction-aware topography and ANN energy predictions
        self.di_graph = self._build_directed_weighted_graph()

        # Compute normalization factors for balanced multi-objective mode
        self._norm_distance, self._norm_time, self._norm_energy = self._compute_normalizers()

    def _build_directed_weighted_graph(self) -> nx.DiGraph:
        """
        Builds a directed graph where each edge (u, v) contains directional slope,
        traffic-adjusted transit times, and ANN-predicted energy consumption.
        """
        DG = nx.DiGraph(city_name=self.base_graph.graph.get("city_name", "VoltCity"))

        # Copy nodes
        for node, attr in self.base_graph.nodes(data=True):
            DG.add_node(node, **attr)

        # Traffic delay multipliers
        traffic_time_multipliers = {
            "low": 1.00,
            "medium": 1.25,
            "high": 1.65
        }
        traffic_encoding = {
            "low": 0,
            "medium": 1,
            "high": 2
        }

        # Add directed edges for both travel directions
        for u, v, attr in self.base_graph.edges(data=True):
            dist_km = float(attr["distance_km"])
            speed_kmh = float(attr.get("speed_limit_kmh", 50))
            traffic_lvl = str(attr.get("traffic_level", "low")).lower()
            traffic_enc = traffic_encoding.get(traffic_lvl, 0)
            base_time = float(attr.get("travel_time_min", (dist_km / speed_kmh) * 60.0))

            elev_u = float(DG.nodes[u].get("elevation_m", 100.0))
            elev_v = float(DG.nodes[v].get("elevation_m", 100.0))

            # Direction 1: u -> v
            elev_delta_uv = elev_v - elev_u
            slope_uv = (elev_delta_uv / (dist_km * 1000.0)) * 100.0
            actual_time_uv = base_time * traffic_time_multipliers.get(traffic_lvl, 1.0)
            energy_uv = self.predictor.predict(
                distance_km=dist_km,
                speed_kmh=speed_kmh,
                traffic_level_encoded=traffic_enc,
                slope_percent=slope_uv,
                elevation_change_m=elev_delta_uv,
                vehicle_load_kg=self.vehicle_load_kg,
                temperature_c=self.temperature_c
            )
            # Enforce non-negativity constraint for Dijkstra and A*
            energy_uv = max(0.01, round(energy_uv, 4))

            DG.add_edge(
                u, v,
                distance_km=dist_km,
                travel_time_min=round(actual_time_uv, 2),
                energy_kwh=energy_uv,
                traffic_level=traffic_lvl,
                slope_percent=round(slope_uv, 2),
                elevation_change_m=round(elev_delta_uv, 1),
                speed_limit_kmh=int(speed_kmh)
            )

            # Direction 2: v -> u (Opposite direction, inverted elevation change)
            elev_delta_vu = elev_u - elev_v
            slope_vu = (elev_delta_vu / (dist_km * 1000.0)) * 100.0
            actual_time_vu = actual_time_uv
            energy_vu = self.predictor.predict(
                distance_km=dist_km,
                speed_kmh=speed_kmh,
                traffic_level_encoded=traffic_enc,
                slope_percent=slope_vu,
                elevation_change_m=elev_delta_vu,
                vehicle_load_kg=self.vehicle_load_kg,
                temperature_c=self.temperature_c
            )
            energy_vu = max(0.01, round(energy_vu, 4))

            DG.add_edge(
                v, u,
                distance_km=dist_km,
                travel_time_min=round(actual_time_vu, 2),
                energy_kwh=energy_vu,
                traffic_level=traffic_lvl,
                slope_percent=round(slope_vu, 2),
                elevation_change_m=round(elev_delta_vu, 1),
                speed_limit_kmh=int(speed_kmh)
            )

        return DG

    def _compute_normalizers(self) -> Tuple[float, float, float]:
        """Calculates mean edge costs to normalize multi-objective units in Balanced mode."""
        distances = [d["distance_km"] for _, _, d in self.di_graph.edges(data=True)]
        times = [d["travel_time_min"] for _, _, d in self.di_graph.edges(data=True)]
        energies = [d["energy_kwh"] for _, _, d in self.di_graph.edges(data=True)]

        avg_d = sum(distances) / max(1, len(distances))
        avg_t = sum(times) / max(1, len(times))
        avg_e = sum(energies) / max(1, len(energies))
        return avg_d, avg_t, avg_e

    def _get_edge_weight(
        self,
        u: str,
        v: str,
        mode: str,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Returns the non-negative scalar weight for edge (u, v) according to routing mode.
        """
        data = self.di_graph[u][v]
        mode = mode.lower()

        if mode == "shortest":
            return data["distance_km"]
        elif mode == "fastest":
            return data["travel_time_min"]
        elif mode in ("energy", "energy_efficient"):
            return data["energy_kwh"]
        elif mode == "balanced":
            # Multi-objective normalized cost: w_e * (E / avgE) + w_t * (T / avgT) + w_d * (D / avgD)
            if weights is None:
                weights = {"energy": 0.50, "time": 0.30, "distance": 0.20}
            w_e = weights.get("energy", 0.50)
            w_t = weights.get("time", 0.30)
            w_d = weights.get("distance", 0.20)

            cost = (
                w_e * (data["energy_kwh"] / self._norm_energy) +
                w_t * (data["travel_time_min"] / self._norm_time) +
                w_d * (data["distance_km"] / self._norm_distance)
            )
            return max(0.001, cost)
        else:
            raise ValueError(f"Unknown optimization mode '{mode}'. Use 'shortest', 'fastest', 'energy', or 'balanced'.")

    # -------------------------------------------------------------------------
    # ALGORITHM 1: DIJKSTRA'S ALGORITHM
    # -------------------------------------------------------------------------
    def run_dijkstra(
        self,
        source: str,
        destination: str,
        mode: str,
        balanced_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], int]:
        """
        Executes Dijkstra's algorithm from source to destination.
        Guarantees finding the globally optimal path for non-negative edge weights.

        Returns:
            Tuple[List[str], int]: (Optimal path as node list, Count of evaluated nodes)
        """
        if source not in self.di_graph:
            raise ValueError(f"Source node '{source}' not found in city network.")
        if destination not in self.di_graph:
            raise ValueError(f"Destination node '{destination}' not found in city network.")

        if source == destination:
            return [source], 1

        # Priority queue stores tuples: (cumulative_cost, node_id)
        pq: List[Tuple[float, str]] = [(0.0, source)]
        distances: Dict[str, float] = {source: 0.0}
        predecessors: Dict[str, Optional[str]] = {source: None}
        visited = set()
        evaluated_count = 0

        while pq:
            current_cost, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)
            evaluated_count += 1

            if current_node == destination:
                break

            for neighbor in self.di_graph.neighbors(current_node):
                if neighbor in visited:
                    continue

                edge_cost = self._get_edge_weight(current_node, neighbor, mode, balanced_weights)
                new_cost = current_cost + edge_cost

                if neighbor not in distances or new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_cost, neighbor))

        if destination not in predecessors:
            raise nx.NetworkXNoPath(f"No reachable path from '{source}' to '{destination}'.")

        # Reconstruct path by backtracking
        path = []
        curr = destination
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        return path, evaluated_count

    # -------------------------------------------------------------------------
    # ALGORITHM 2: A* SEARCH ALGORITHM
    # -------------------------------------------------------------------------
    def _admissible_heuristic(
        self,
        node: str,
        goal: str,
        mode: str,
        balanced_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Computes an admissible, consistent heuristic estimate h(n) from node to goal.
        Admissibility rule: h(n) <= true remaining cost h*(n).
        """
        pos_n = self.di_graph.nodes[node]["pos"]
        pos_g = self.di_graph.nodes[goal]["pos"]

        # Planar straight-line Euclidean distance
        euclidean_dist = math.hypot(pos_n[0] - pos_g[0], pos_n[1] - pos_g[1])

        if mode == "shortest":
            # Straight line is always <= actual road distance
            return euclidean_dist

        elif mode == "fastest":
            # Maximum city speed limit is 70 km/h; time cannot be faster than straight-line at max speed
            max_speed_kmh = 70.0
            return (euclidean_dist / max_speed_kmh) * 60.0

        elif mode in ("energy", "energy_efficient"):
            # Minimum possible energy rate per km (e.g. 0.05 kWh/km)
            min_energy_per_km = 0.05
            return euclidean_dist * min_energy_per_km

        elif mode == "balanced":
            if balanced_weights is None:
                balanced_weights = {"energy": 0.50, "time": 0.30, "distance": 0.20}
            h_dist = euclidean_dist
            h_time = (euclidean_dist / 70.0) * 60.0
            h_energy = euclidean_dist * 0.05
            return (
                balanced_weights.get("energy", 0.50) * (h_energy / self._norm_energy) +
                balanced_weights.get("time", 0.30) * (h_time / self._norm_time) +
                balanced_weights.get("distance", 0.20) * (h_dist / self._norm_distance)
            )
        return 0.0

    def run_astar(
        self,
        source: str,
        destination: str,
        mode: str,
        balanced_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], int]:
        """
        Executes A* search algorithm using the admissible heuristic.
        f(n) = g(n) + h(n)
        Prunes suboptimal branches and evaluates fewer nodes than Dijkstra.
        """
        if source not in self.di_graph:
            raise ValueError(f"Source node '{source}' not found in city network.")
        if destination not in self.di_graph:
            raise ValueError(f"Destination node '{destination}' not found in city network.")

        if source == destination:
            return [source], 1

        # Priority queue stores tuples: (f_score, g_score, node_id)
        h_start = self._admissible_heuristic(source, destination, mode, balanced_weights)
        pq: List[Tuple[float, float, str]] = [(h_start, 0.0, source)]
        g_scores: Dict[str, float] = {source: 0.0}
        predecessors: Dict[str, Optional[str]] = {source: None}
        visited = set()
        evaluated_count = 0

        while pq:
            f_score, g_score, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)
            evaluated_count += 1

            if current_node == destination:
                break

            for neighbor in self.di_graph.neighbors(current_node):
                if neighbor in visited:
                    continue

                edge_cost = self._get_edge_weight(current_node, neighbor, mode, balanced_weights)
                tentative_g = g_score + edge_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_val = self._admissible_heuristic(neighbor, destination, mode, balanced_weights)
                    f_val = tentative_g + h_val
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (f_val, tentative_g, neighbor))

        if destination not in predecessors:
            raise nx.NetworkXNoPath(f"No reachable path from '{source}' to '{destination}'.")

        # Backtrack path
        path = []
        curr = destination
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        return path, evaluated_count

    # -------------------------------------------------------------------------
    # ROUTE OPTIMIZATION WRAPPER & METRICS CALCULATOR
    # -------------------------------------------------------------------------
    def find_route(
        self,
        source: str,
        destination: str,
        mode: str = "energy",
        algorithm: str = "dijkstra",
        balanced_weights: Optional[Dict[str, float]] = None
    ) -> RouteResult:
        """
        Calculates the optimal route and comprehensive trip statistics.

        Parameters:
            source (str): Starting intersection (e.g. 'A').
            destination (str): Target intersection (e.g. 'I').
            mode (str): 'shortest', 'fastest', 'energy', or 'balanced'.
            algorithm (str): 'dijkstra' or 'astar'.
            balanced_weights (dict, optional): Weights for balanced mode.

        Returns:
            RouteResult: Complete details with path, distance, time, and predicted energy.
        """
        source = str(source).upper()
        destination = str(destination).upper()
        mode = mode.lower()
        algorithm = algorithm.lower()

        if algorithm == "dijkstra":
            path, evaluated = self.run_dijkstra(source, destination, mode, balanced_weights)
        elif algorithm in ("astar", "a*", "a_star"):
            path, evaluated = self.run_astar(source, destination, mode, balanced_weights)
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'dijkstra' or 'astar'.")

        # Assemble full route physical metrics
        total_dist = 0.0
        total_time = 0.0
        total_energy = 0.0
        segments = []
        charging_stops = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.di_graph[u][v]
            seg = RouteSegment(
                source=u,
                target=v,
                distance_km=data["distance_km"],
                travel_time_min=data["travel_time_min"],
                energy_kwh=data["energy_kwh"],
                traffic_level=data["traffic_level"],
                slope_percent=data["slope_percent"],
                speed_limit_kmh=data["speed_limit_kmh"]
            )
            segments.append(seg)
            total_dist += seg.distance_km
            total_time += seg.travel_time_min
            total_energy += seg.energy_kwh

            # Check if intermediate node has EV charging
            if self.di_graph.nodes[u].get("has_charging_station", False) and u != source:
                charging_stops.append(u)

        if self.di_graph.nodes[destination].get("has_charging_station", False):
            charging_stops.append(f"{destination} (Destination)")

        return RouteResult(
            mode=mode,
            algorithm=algorithm,
            source=source,
            destination=destination,
            path=path,
            total_distance_km=round(total_dist, 2),
            total_time_min=round(total_time, 1),
            total_energy_kwh=round(total_energy, 3),
            charging_stops=charging_stops,
            segments=segments,
            nodes_evaluated=evaluated
        )

    def compare_all_modes(
        self,
        source: str,
        destination: str,
        algorithm: str = "dijkstra"
    ) -> Dict[str, RouteResult]:
        """
        Runs all four modes (Shortest, Fastest, Energy Efficient, Balanced)
        for a given source and destination, enabling direct comparison.
        """
        modes = ["shortest", "fastest", "energy", "balanced"]
        results = {}
        for m in modes:
            results[m] = self.find_route(source, destination, mode=m, algorithm=algorithm)
        return results


def visualize_route_comparison(
    optimizer: EVRouteOptimizer,
    routes: Dict[str, RouteResult],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> str:
    """
    Renders the city road graph with colored overlays of routes for different modes.
    """
    if save_path is None:
        save_path = DEFAULT_ROUTE_PLOT_PATH
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    G = optimizer.base_graph
    pos = {n: d["pos"] for n, d in G.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    # Draw base network edges in light grey
    nx.draw_networkx_edges(G, pos, edge_color="#dcdde1", width=1.5, alpha=0.7, ax=ax)

    # Style definitions for the 4 modes
    mode_styles = {
        "shortest": {"color": "#2980b9", "width": 5.0, "style": "solid", "alpha": 0.85, "label": "Shortest (Distance)"},
        "fastest":  {"color": "#e67e22", "width": 4.2, "style": "dashed", "alpha": 0.90, "label": "Fastest (Time)"},
        "energy":   {"color": "#27ae60", "width": 3.4, "style": "solid", "alpha": 0.95, "label": "Energy Efficient (kWh)"},
        "balanced": {"color": "#8e44ad", "width": 2.2, "style": "dashdot", "alpha": 0.95, "label": "Balanced (Hybrid)"}
    }

    # Overlay routes
    legend_handles = []
    for mode_name, result in routes.items():
        style = mode_styles.get(mode_name, {"color": "#333", "width": 2.0, "style": "solid", "alpha": 0.8, "label": mode_name})
        path_edges = list(zip(result.path[:-1], result.path[1:]))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color=style["color"],
            width=style["width"],
            style=style["style"],
            alpha=style["alpha"],
            ax=ax
        )
        legend_handles.append(
            mpatches.Patch(
                color=style["color"],
                label=f"{style['label']}: {result.total_dist_str if hasattr(result, 'total_dist_str') else result.total_distance_km}km | {result.total_time_min}min | {result.total_energy_kwh}kWh"
            )
        )

    # Draw Nodes
    regular_nodes = [n for n, d in G.nodes(data=True) if not d.get("has_charging_station")]
    charging_nodes = [n for n, d in G.nodes(data=True) if d.get("has_charging_station")]

    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=600, node_color="#ecf0f1", edgecolors="#2c3e50", linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=charging_nodes, node_size=800, node_color="#2ecc71", edgecolors="#1b5e20", linewidths=2.2, ax=ax)

    # Highlight Source and Destination
    sample_route = next(iter(routes.values()))
    src, dst = sample_route.source, sample_route.destination
    nx.draw_networkx_nodes(G, pos, nodelist=[src], node_size=1000, node_color="#3498db", edgecolors="#154360", linewidths=3.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[dst], node_size=1000, node_color="#e74c3c", edgecolors="#78281f", linewidths=3.0, ax=ax)

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", font_color="#2c3e50", ax=ax)

    # Subtitles for nodes
    for node, (x, y) in pos.items():
        attr = G.nodes[node]
        name = attr.get("name", "")
        elev = attr.get("elevation_m", 0)
        ax.text(x, y + 0.28, f"{name}\n({int(elev)}m)", fontsize=7.5, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.85))

    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9.5, title=f"Routes: {src} -> {dst}")
    ax.set_title(f"EV Route Optimization Engine - Mode Comparison ({src} to {dst})", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return str(save_path)
