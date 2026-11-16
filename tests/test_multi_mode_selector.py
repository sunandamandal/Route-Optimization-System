"""
tests/test_multi_mode_selector.py
Unit and Integration Tests for the Modular Multi-Mode Route Selector.
"""

import unittest
from pathlib import Path
from src.india_road_service import (
    MultiModeRouteSelector,
    ModeSelectionResult,
    get_india_route_service
)
from src.india_geocoding import geocode_indian_location
from src.route_optimizer import EVRouteOptimizer, RouteResult, RouteSegment
from src.road_network import load_road_network
from src.ev_calculations import EVBatteryState, compare_routes_battery, FeasibilityStatus


class TestMultiModeRouteSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = MultiModeRouteSelector()
        cls.service = get_india_route_service()

        # Create 3 distinct mock candidates for rigorous mathematical selector testing
        cls.mock_candidates = [
            {
                "title": "Corridor A (Expressway - High Speed, Longer Distance)",
                "distance_km": 210.0,
                "duration_min": 140.0,     # Fastest travel time
                "energy_kwh": 31.50,       # High speed -> higher aerodynamic drag
                "path_nodes": ["Start", "ExpWay-1", "Goal"],
                "charging_stops": ["Charger 1"],
                "segments": [
                    RouteSegment("Start", "Goal", 210.0, 140.0, 31.50, "low", 0.1, 95)
                ],
                "coordinates": [(22.5, 88.3), (23.5, 87.3)],
                "turn_instructions": []
            },
            {
                "title": "Corridor B (State Highway - Shorter Distance, Lower Speed)",
                "distance_km": 185.0,      # Shortest physical road distance
                "duration_min": 180.0,
                "energy_kwh": 26.20,
                "path_nodes": ["Start", "StateHwy-1", "Goal"],
                "charging_stops": [],
                "segments": [
                    RouteSegment("Start", "Goal", 185.0, 180.0, 26.20, "medium", 0.2, 60)
                ],
                "coordinates": [(22.5, 88.3), (23.5, 87.3)],
                "turn_instructions": []
            },
            {
                "title": "Corridor C (Eco-Cruise Highway - Moderated Speed, Flatter Elevation)",
                "distance_km": 195.0,
                "duration_min": 165.0,
                "energy_kwh": 22.80,       # Minimum ANN-predicted energy consumption
                "path_nodes": ["Start", "Eco-1", "Goal"],
                "charging_stops": [],
                "segments": [
                    RouteSegment("Start", "Goal", 195.0, 165.0, 22.80, "low", 0.05, 72)
                ],
                "coordinates": [(22.5, 88.3), (23.5, 87.3)],
                "turn_instructions": []
            }
        ]

    def test_mode_1_shortest_route_selection(self):
        """Mode 1: Shortest Route must select the candidate with minimum physical road distance."""
        res = self.selector.select_shortest(
            self.mock_candidates,
            algorithm="astar",
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        self.assertEqual(res.mode, "shortest")
        self.assertEqual(res.route.total_distance_km, 185.0)
        self.assertIn("Corridor B", res.selection_explanation)
        self.assertIn("lowest actual road distance of 185.00 km", res.selection_explanation)
        # Check honest optimality scope (no false claim of global continental optimality)
        self.assertIn("Candidate Set Optimum", res.optimality_scope)
        self.assertEqual(res.candidates_count, 3)

    def test_mode_2_fastest_route_selection(self):
        """Mode 2: Fastest Route must select the candidate with minimum road travel time."""
        res = self.selector.select_fastest(
            self.mock_candidates,
            algorithm="astar",
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        self.assertEqual(res.mode, "fastest")
        self.assertEqual(res.route.total_time_min, 140.0)
        self.assertIn("Corridor A", res.selection_explanation)
        self.assertIn("lowest estimated road transit time of 140.0 min", res.selection_explanation)
        self.assertIn("Candidate Set Optimum", res.optimality_scope)

    def test_mode_3_energy_efficient_route_selection(self):
        """Mode 3: Energy Efficient Route must select the candidate with lowest ANN-predicted kWh."""
        res = self.selector.select_energy_efficient(
            self.mock_candidates,
            algorithm="astar",
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        self.assertEqual(res.mode, "energy")
        self.assertEqual(res.route.total_energy_kwh, 22.80)
        self.assertIn("Corridor C", res.selection_explanation)
        self.assertIn("lowest ANN-predicted energy consumption of 22.800 kWh", res.selection_explanation)
        # Verify it reports savings vs fastest (31.50 - 22.80 = 8.70 kWh)
        self.assertIn("8.700 kWh", res.selection_explanation)

    def test_mode_4_balanced_route_selection(self):
        """Mode 4: Balanced Route must minimize normalized weighted multi-objective score."""
        weights = {"energy": 0.60, "time": 0.20, "distance": 0.20}
        res = self.selector.select_balanced(
            self.mock_candidates,
            weights=weights,
            algorithm="astar",
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        self.assertEqual(res.mode, "balanced")
        self.assertIsNotNone(res.route)
        self.assertIn("lowest multi-objective score", res.selection_explanation)
        self.assertIn("Energy: 0.60", res.selection_explanation)

    def test_select_all_modes_returns_four_complete_results(self):
        """Verify select_all_modes returns all 4 modes properly populated."""
        all_modes = self.selector.select_all_modes(
            self.mock_candidates,
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        self.assertEqual(set(all_modes.keys()), {"shortest", "fastest", "energy", "balanced"})
        for m, result in all_modes.items():
            self.assertIsInstance(result, ModeSelectionResult)
            self.assertGreater(result.route.total_distance_km, 0.0)
            self.assertGreater(result.route.total_time_min, 0.0)
            self.assertGreater(result.route.total_energy_kwh, 0.0)
            self.assertTrue(len(result.selection_explanation) > 0)
            self.assertTrue(len(result.optimality_scope) > 0)

    def test_insufficient_battery_handling_on_candidates(self):
        """Verify handling of insufficient battery across candidate routes."""
        all_modes = self.selector.select_all_modes(
            self.mock_candidates,
            source_name="Kolkata",
            dest_name="Durgapur"
        )
        routes_dict = {m: res.route for m, res in all_modes.items()}

        # Battery with only 15 kWh total capacity (all routes need >= 22.8 kWh)
        battery_empty = EVBatteryState(capacity_kwh=15.0, current_soc_percent=50.0, min_reserve_percent=15.0)
        summary = compare_routes_battery(routes_dict, battery_empty)

        for mode_key, ev_eval in summary.evaluations.items():
            self.assertEqual(ev_eval.status, FeasibilityStatus.INFEASIBLE)
            self.assertFalse(ev_eval.is_feasible)
            self.assertTrue(ev_eval.charging_needed_kwh > 0.0)

    def test_preservation_of_dijkstra_and_astar_in_simulated_graph(self):
        """Verify Dijkstra and A* remain fully functional and optimal on the simulated VoltCity graph."""
        graph = load_road_network()
        optimizer = EVRouteOptimizer(graph=graph)

        # 1. Dijkstra on shortest mode
        dijkstra_path, d_evals = optimizer.run_dijkstra("G", "L", mode="shortest")
        # 2. A* on shortest mode
        astar_path, a_evals = optimizer.run_astar("G", "L", mode="shortest")

        self.assertEqual(dijkstra_path[0], "G")
        self.assertEqual(dijkstra_path[-1], "L")
        self.assertEqual(dijkstra_path, astar_path)
        # A* should evaluate fewer or equal nodes than Dijkstra
        self.assertLessEqual(a_evals, d_evals)


if __name__ == "__main__":
    unittest.main()
