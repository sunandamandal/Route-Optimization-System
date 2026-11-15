"""
tests/test_road_network.py
Unit tests for the road network graph representation and loader.
"""

import unittest
from pathlib import Path
import networkx as nx

from src.road_network import load_road_network, get_network_summary, display_road_network


class TestRoadNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_road_network()

    def test_minimum_nodes_count(self):
        """Verify at least 15 city nodes are present."""
        self.assertGreaterEqual(
            self.graph.number_of_nodes(),
            15,
            f"Expected >= 15 nodes, got {self.graph.number_of_nodes()}"
        )

    def test_minimum_roads_count(self):
        """Verify at least 25 roads (edges) are present."""
        self.assertGreaterEqual(
            self.graph.number_of_edges(),
            25,
            f"Expected >= 25 edges, got {self.graph.number_of_edges()}"
        )

    def test_node_attributes(self):
        """Verify all nodes have required attributes (name, pos, elevation_m)."""
        for node_id, data in self.graph.nodes(data=True):
            self.assertIn("name", data, f"Node {node_id} missing 'name'")
            self.assertIn("pos", data, f"Node {node_id} missing 'pos'")
            self.assertIn("elevation_m", data, f"Node {node_id} missing 'elevation_m'")
            self.assertIsInstance(data["pos"], tuple)
            self.assertEqual(len(data["pos"]), 2)

    def test_edge_attributes(self):
        """Verify all edges have required attributes: distance, travel time, traffic, elevation/slope, speed limit."""
        required_attrs = [
            "distance_km",
            "travel_time_min",
            "traffic_level",
            "elevation_change_m",
            "slope_percent",
            "speed_limit_kmh"
        ]
        for u, v, data in self.graph.edges(data=True):
            for attr in required_attrs:
                self.assertIn(attr, data, f"Edge ({u}, {v}) missing '{attr}'")

            self.assertIn(data["traffic_level"], ["low", "medium", "high"])
            self.assertGreater(data["distance_km"], 0)
            self.assertGreater(data["travel_time_min"], 0)
            self.assertGreater(data["speed_limit_kmh"], 0)

    def test_ev_charging_stations(self):
        """Verify EV charging stations exist at key nodes."""
        charging_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("has_charging_station")]
        self.assertGreaterEqual(len(charging_nodes), 1)

    def test_graph_connectivity(self):
        """Verify the fictional city network is fully connected (no isolated islands)."""
        self.assertTrue(
            nx.is_connected(self.graph),
            "The city road network must be fully connected for EV routing."
        )

    def test_display_generation(self):
        """Verify visualization image can be generated and saved."""
        output_file = Path(__file__).resolve().parent.parent / "outputs" / "test_network.png"
        result_path = display_road_network(self.graph, save_path=output_file)
        self.assertTrue(Path(result_path).exists())
        # Clean up test artifact
        if Path(result_path).exists():
            Path(result_path).unlink()


if __name__ == "__main__":
    unittest.main()
