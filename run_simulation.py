"""
run_simulation.py
Command-Line Runner and Batch Simulation Script for EV Route Optimization System.
Allows running multi-scenario simulations and terminal tests without starting the web UI.
"""

from pathlib import Path
from src.road_network import load_road_network, display_road_network, get_network_summary
from src.route_optimizer import EVRouteOptimizer, visualize_route_comparison
from src.ev_calculations import (
    EVBatteryState,
    compare_routes_battery,
    format_battery_report
)


def run_battery_simulation(
    optimizer: EVRouteOptimizer,
    source: str,
    destination: str,
    battery_capacity_kwh: float,
    current_soc_percent: float,
    min_reserve_percent: float = 15.0,
    save_map: bool = True
):
    src_name = optimizer.base_graph.nodes[source]["name"]
    dst_name = optimizer.base_graph.nodes[destination]["name"]

    print("\n" + "=" * 80)
    print(f" TRIP SIMULATION: {source} ({src_name})  --->  {destination} ({dst_name})")
    print("=" * 80)

    routes = optimizer.compare_all_modes(source, destination, algorithm="dijkstra")
    battery_state = EVBatteryState(
        capacity_kwh=battery_capacity_kwh,
        current_soc_percent=current_soc_percent,
        min_reserve_percent=min_reserve_percent
    )
    summary = compare_routes_battery(routes, battery_state)
    print(format_battery_report(summary))

    if save_map:
        map_filename = f"route_comparison_{source}_to_{destination}.png"
        save_path = Path("outputs") / map_filename
        visualize_route_comparison(optimizer, routes, save_path=save_path)
        print(f"\n  [Visual Map Saved]: outputs/{map_filename}")

    return summary


def main():
    print("=" * 80)
    print("        SMART EV ROUTE OPTIMIZATION & NAVIGATION SYSTEM")
    print("             (VoltCity Intelligent Road Grid - CLI)")
    print("=" * 80)

    graph = load_road_network()
    summary = get_network_summary(graph)
    print(f"  - Total Intersections : {summary['total_nodes']} nodes")
    print(f"  - Total Road Segments : {summary['total_roads']} edges")
    print(f"  - EV Charging Stations: {summary['charging_stations_count']} locations")

    optimizer = EVRouteOptimizer(graph=graph)

    # Scenario 1: Safe
    print("\n[Scenario 1: Normal Commute - 50% SoC]")
    run_battery_simulation(optimizer, "G", "L", 60.0, 50.0, 15.0, save_map=True)

    # Scenario 2: Caution (Reserve Breached)
    print("\n[Scenario 2: Caution Commute - 16.5% SoC]")
    run_battery_simulation(optimizer, "G", "L", 60.0, 16.5, 15.0, save_map=False)

    # Scenario 3: Infeasible (Stranded)
    print("\n[Scenario 3: Critical Commute - 1.0% SoC]")
    run_battery_simulation(optimizer, "G", "L", 60.0, 1.0, 15.0, save_map=False)


if __name__ == "__main__":
    main()
