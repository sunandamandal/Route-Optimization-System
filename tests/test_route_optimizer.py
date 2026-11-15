"""
tests/test_route_optimizer.py
Unit tests for the EV Route Optimization Engine (Dijkstra and A* Search).
"""

import unittest
from pathlib import Path
import networkx as nx

from src.route_optimizer import EVRouteOptimizer, RouteResult, visualize_route_comparison


class TestEVRouteOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = EVRouteOptimizer()

    def test_non_negative_edge_weights(self):
        """Verify that all edge weights across all modes are strictly positive."""
        for u, v, d in self.optimizer.di_graph.edges(data=True):
            self.assertGreater(d["distance_km"], 0, f"Negative distance on ({u}, {v})")
            self.assertGreater(d["travel_time_min"], 0, f"Negative time on ({u}, {v})")
            self.assertGreater(d["energy_kwh"], 0, f"Negative energy on ({u}, {v})")

    def test_shortest_route_validity(self):
        """Verify shortest route returns a connected, valid path in the graph."""
        res = self.optimizer.find_route("A", "I", mode="shortest", algorithm="dijkstra")
        self.assertIsInstance(res, RouteResult)
        self.assertEqual(res.source, "A")
        self.assertEqual(res.destination, "I")
        self.assertEqual(res.path[0], "A")
        self.assertEqual(res.path[-1], "I")

        # Check path continuity
        for i in range(len(res.path) - 1):
            u, v = res.path[i], res.path[i + 1]
            self.assertTrue(self.optimizer.di_graph.has_edge(u, v))

    def test_fastest_route_validity(self):
        """Verify fastest route calculation and non-zero metrics."""
        res = self.optimizer.find_route("A", "I", mode="fastest", algorithm="dijkstra")
        self.assertGreater(res.total_time_min, 0)
        self.assertGreater(res.total_distance_km, 0)
        self.assertGreater(res.total_energy_kwh, 0)

    def test_energy_efficient_route(self):
        """Verify energy-efficient mode produces optimal energy consumption."""
        res_energy = self.optimizer.find_route("A", "I", mode="energy", algorithm="dijkstra")
        res_dist = self.optimizer.find_route("A", "I", mode="shortest", algorithm="dijkstra")
        self.assertLessEqual(
            res_energy.total_energy_kwh,
            res_dist.total_energy_kwh + 1e-4,
            "Energy route should have <= energy consumption than shortest distance route."
        )

    def test_balanced_mode(self):
        """Verify balanced multi-objective mode computes valid metrics."""
        res = self.optimizer.find_route("B", "M", mode="balanced", algorithm="dijkstra")
        self.assertEqual(res.path[0], "B")
        self.assertEqual(res.path[-1], "M")
        self.assertGreater(res.total_distance_km, 0)

    def test_astar_search_algorithm(self):
        """Verify A* search produces identical optimal path cost to Dijkstra."""
        res_dijkstra = self.optimizer.find_route("A", "I", mode="shortest", algorithm="dijkstra")
        res_astar = self.optimizer.find_route("A", "I", mode="shortest", algorithm="astar")

        self.assertEqual(res_dijkstra.path, res_astar.path)
        self.assertAlmostEqual(res_dijkstra.total_distance_km, res_astar.total_distance_km, places=2)

    def test_same_source_and_destination(self):
        """Verify routing from node to itself returns path with cost 0."""
        res = self.optimizer.find_route("A", "A", mode="energy")
        self.assertEqual(res.path, ["A"])
        self.assertEqual(res.total_distance_km, 0.0)
        self.assertEqual(res.total_time_min, 0.0)
        self.assertEqual(res.total_energy_kwh, 0.0)

    def test_invalid_node_raises_error(self):
        """Verify invalid source or destination raises ValueError."""
        with self.assertRaises(ValueError):
            self.optimizer.find_route("NON_EXISTENT", "A", mode="shortest")
        with self.assertRaises(ValueError):
            self.optimizer.find_route("A", "INVALID_NODE", mode="shortest")

    def test_all_modes_comparison(self):
        """Verify compare_all_modes returns results for all four modes."""
        all_routes = self.optimizer.compare_all_modes("A", "I")
        self.assertIn("shortest", all_routes)
        self.assertIn("fastest", all_routes)
        self.assertIn("energy", all_routes)
        self.assertIn("balanced", all_routes)

    def test_visualize_route_comparison(self):
        """Verify route comparison map can be rendered and saved."""
        all_routes = self.optimizer.compare_all_modes("A", "I")
        out_img = Path(__file__).resolve().parent.parent / "outputs" / "test_route_plot.png"
        res_path = visualize_route_comparison(self.optimizer, all_routes, save_path=out_img)
        self.assertTrue(Path(res_path).exists())
        if Path(res_path).exists():
            Path(res_path).unlink()


if __name__ == "__main__":
    unittest.main()
