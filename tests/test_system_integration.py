"""
tests/test_system_integration.py
Comprehensive End-to-End System Testing & Validation Suite for EV Route Optimization.

Implements and validates the 8 required test cases:
  1. Shortest route
  2. Fastest route
  3. Energy-efficient route
  4. Balanced route
  5. Invalid source or destination
  6. Unreachable destination
  7. Insufficient EV battery
  8. Different source-destination pairs

Verifies the 5 core properties:
  - Every returned route uses valid graph edges.
  - Route distance equals the sum of road distances.
  - Energy consumption is non-negative.
  - The selected route matches the requested optimization mode.
  - Battery feasibility is calculated correctly.
"""

import unittest
from pathlib import Path
import networkx as nx

from src.road_network import load_road_network
from src.route_optimizer import EVRouteOptimizer, RouteResult
from src.ev_calculations import (
    EVBatteryState,
    FeasibilityStatus,
    evaluate_route_battery,
    compare_routes_battery
)


class TestEVSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_road_network()
        cls.optimizer = EVRouteOptimizer(graph=cls.graph)

    # -------------------------------------------------------------------------
    # TEST CASE 1: SHORTEST ROUTE
    # -------------------------------------------------------------------------
    def test_case_1_shortest_route(self):
        """Test Case 1: Verify Shortest Route minimizes distance and sums road segments correctly."""
        route = self.optimizer.find_route("G", "L", mode="shortest", algorithm="dijkstra")

        # 1. Verify valid edges
        for i in range(len(route.path) - 1):
            u, v = route.path[i], route.path[i + 1]
            self.assertTrue(self.optimizer.di_graph.has_edge(u, v), f"Edge ({u}, {v}) is invalid")

        # 2. Verify route distance equals sum of road distances
        sum_dist = sum(seg.distance_km for seg in route.segments)
        self.assertAlmostEqual(route.total_distance_km, sum_dist, places=2)

        # 3. Verify it is strictly <= other modes in distance
        fastest_route = self.optimizer.find_route("G", "L", mode="fastest")
        self.assertLessEqual(route.total_distance_km, fastest_route.total_distance_km + 1e-4)

    # -------------------------------------------------------------------------
    # TEST CASE 2: FASTEST ROUTE
    # -------------------------------------------------------------------------
    def test_case_2_fastest_route(self):
        """Test Case 2: Verify Fastest Route minimizes travel time considering traffic congestion."""
        route = self.optimizer.find_route("G", "L", mode="fastest", algorithm="dijkstra")

        # 1. Verify valid edges
        for i in range(len(route.path) - 1):
            u, v = route.path[i], route.path[i + 1]
            self.assertTrue(self.optimizer.di_graph.has_edge(u, v))

        # 2. Verify travel time equals sum of segments
        sum_time = sum(seg.travel_time_min for seg in route.segments)
        self.assertAlmostEqual(route.total_time_min, sum_time, places=1)

        # 3. Verify travel time is <= shortest route travel time
        shortest_route = self.optimizer.find_route("G", "L", mode="shortest")
        self.assertLessEqual(route.total_time_min, shortest_route.total_time_min + 1e-4)

    # -------------------------------------------------------------------------
    # TEST CASE 3: ENERGY-EFFICIENT ROUTE
    # -------------------------------------------------------------------------
    def test_case_3_energy_efficient_route(self):
        """Test Case 3: Verify Energy-Efficient Route minimizes ANN-predicted battery energy."""
        route = self.optimizer.find_route("G", "L", mode="energy", algorithm="dijkstra")

        # 1. Verify valid edges
        for i in range(len(route.path) - 1):
            u, v = route.path[i], route.path[i + 1]
            self.assertTrue(self.optimizer.di_graph.has_edge(u, v))

        # 2. Verify energy equals sum of segments and is non-negative
        sum_energy = sum(seg.energy_kwh for seg in route.segments)
        self.assertAlmostEqual(route.total_energy_kwh, sum_energy, places=3)
        self.assertGreater(route.total_energy_kwh, 0.0, "Energy consumption must be positive")

        # 3. Verify energy is <= shortest route energy
        shortest_route = self.optimizer.find_route("G", "L", mode="shortest")
        self.assertLessEqual(route.total_energy_kwh, shortest_route.total_energy_kwh + 1e-4)

    # -------------------------------------------------------------------------
    # TEST CASE 4: BALANCED ROUTE
    # -------------------------------------------------------------------------
    def test_case_4_balanced_route(self):
        """Test Case 4: Verify Balanced Route multi-objective weighted optimization."""
        weights = {"energy": 0.50, "time": 0.30, "distance": 0.20}
        route = self.optimizer.find_route("G", "L", mode="balanced", balanced_weights=weights)

        # 1. Verify valid edges
        for i in range(len(route.path) - 1):
            u, v = route.path[i], route.path[i + 1]
            self.assertTrue(self.optimizer.di_graph.has_edge(u, v))

        # 2. Check metrics validity
        self.assertGreater(route.total_distance_km, 0.0)
        self.assertGreater(route.total_time_min, 0.0)
        self.assertGreater(route.total_energy_kwh, 0.0)

    # -------------------------------------------------------------------------
    # TEST CASE 5: INVALID SOURCE OR DESTINATION
    # -------------------------------------------------------------------------
    def test_case_5_invalid_source_or_destination(self):
        """Test Case 5: Verify robust error handling when source or destination is non-existent."""
        with self.assertRaises(ValueError) as ctx:
            self.optimizer.find_route("INVALID_NODE", "A", mode="energy")
        self.assertIn("not found", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            self.optimizer.find_route("A", "UNKNOWN_GOAL", mode="energy")
        self.assertIn("not found", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # TEST CASE 6: UNREACHABLE DESTINATION
    # -------------------------------------------------------------------------
    def test_case_6_unreachable_destination(self):
        """Test Case 6: Verify NetworkXNoPath is raised if no path connects source to destination."""
        # Create a disconnected island node in graph copy
        disconnected_opt = EVRouteOptimizer(graph=self.graph)
        disconnected_opt.di_graph.add_node("ISOLATED_ISLAND", pos=(10.0, 10.0), elevation_m=100.0)

        with self.assertRaises(nx.NetworkXNoPath):
            disconnected_opt.find_route("A", "ISOLATED_ISLAND", mode="energy")

    # -------------------------------------------------------------------------
    # TEST CASE 7: INSUFFICIENT EV BATTERY
    # -------------------------------------------------------------------------
    def test_case_7_insufficient_battery(self):
        """Test Case 7: Verify critical warning and non-feasibility when battery is depleted."""
        # Trip from G to L requires ~1.30 kWh
        route = self.optimizer.find_route("G", "L", mode="energy")

        # Subcase A: Critically Insufficient Battery (1% of 60 kWh = 0.60 kWh stored)
        critical_battery = EVBatteryState(capacity_kwh=60.0, current_soc_percent=1.0, min_reserve_percent=15.0)
        eval_crit = evaluate_route_battery(route.total_energy_kwh, critical_battery, route.total_distance_km, route.total_time_min)

        self.assertFalse(eval_crit.is_feasible, "Must NOT claim vehicle can reach destination!")
        self.assertFalse(eval_crit.is_safe)
        self.assertEqual(eval_crit.status, FeasibilityStatus.INFEASIBLE)
        self.assertGreater(eval_crit.critical_deficit_kwh, 0.0)
        self.assertGreater(eval_crit.charging_needed_kwh, 0.0)
        self.assertIsNotNone(eval_crit.warning_message)
        self.assertIn("CANNOT reach", eval_crit.warning_message)

        # Subcase B: Marginal Battery breaching 15% reserve buffer (16.5% SoC = 9.90 kWh stored, usable 0.90 kWh)
        marginal_battery = EVBatteryState(capacity_kwh=60.0, current_soc_percent=16.5, min_reserve_percent=15.0)
        eval_marg = evaluate_route_battery(route.total_energy_kwh, marginal_battery, route.total_distance_km, route.total_time_min)

        self.assertTrue(eval_marg.is_feasible, "Vehicle physically arrives")
        self.assertFalse(eval_marg.is_safe, "Reserve buffer is breached")
        self.assertEqual(eval_marg.status, FeasibilityStatus.RESERVE_BREACHED)
        self.assertGreater(eval_marg.energy_deficit_kwh, 0.0)
        self.assertIsNotNone(eval_marg.warning_message)
        self.assertIn("reserve", eval_marg.warning_message.lower())

    # -------------------------------------------------------------------------
    # TEST CASE 8: DIFFERENT SOURCE-DESTINATION PAIRS
    # -------------------------------------------------------------------------
    def test_case_8_different_od_pairs(self):
        """Test Case 8: Verify multiple diverse source-destination pairs across all modes."""
        od_pairs = [
            ("A", "I"),  # Downtown to Airport
            ("G", "L"),  # Harbor Bay to Uptown Plaza
            ("B", "M"),  # North Gate to Industrial Park
            ("E", "H"),  # West End to East Hill Suburbs
            ("K", "F"),  # Green Valley to South Tech Park
            ("C", "P"),  # Financial District to Metro Interchange
            ("D", "N"),  # Midtown to Residential Heights
            ("A", "G"),  # Downtown to Harbor
            ("G", "A"),  # Harbor to Downtown (Uphill return)
            ("O", "B")   # Logistics Hub to North Gate
        ]

        modes = ["shortest", "fastest", "energy", "balanced"]

        for src, dst in od_pairs:
            for mode in modes:
                route = self.optimizer.find_route(src, dst, mode=mode, algorithm="dijkstra")

                # Verify start and finish
                self.assertEqual(route.source, src)
                self.assertEqual(route.destination, dst)
                self.assertEqual(route.path[0], src)
                self.assertEqual(route.path[-1], dst)

                # Verify every segment is a real edge in the city graph
                for i in range(len(route.path) - 1):
                    u, v = route.path[i], route.path[i + 1]
                    self.assertTrue(self.optimizer.di_graph.has_edge(u, v))

                # Verify distance equals sum of road distances
                computed_dist = sum(self.optimizer.di_graph[route.path[i]][route.path[i + 1]]["distance_km"] for i in range(len(route.path) - 1))
                self.assertAlmostEqual(route.total_distance_km, computed_dist, places=2)

                # Verify energy is non-negative
                self.assertGreater(route.total_energy_kwh, 0.0)


if __name__ == "__main__":
    unittest.main()
