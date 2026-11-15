"""
tests/test_map_visualization.py
Unit tests for interactive Folium mapping and geographic projections.
"""

import unittest
from pathlib import Path
import folium
from src.road_network import load_road_network
from src.map_visualization import create_city_folium_map, node_to_latlon


class TestMapVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_road_network()

    def test_node_to_latlon_conversion(self):
        """Verify coordinate translation maps to realistic coordinates."""
        lat, lon = node_to_latlon((2.0, 5.0))
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertGreater(lat, 10.0)
        self.assertGreater(lon, 70.0)

    def test_create_folium_map_without_route(self):
        """Verify base map generation with all nodes and edges."""
        f_map = create_city_folium_map(self.graph)
        self.assertIsInstance(f_map, folium.Map)
        html_str = f_map._repr_html_()
        self.assertIn("folium", html_str.lower())

    def test_create_folium_map_with_selected_route(self):
        """Verify map generation with source, destination, and highlighted path."""
        path = ["A", "D", "O", "J", "I"]
        f_map = create_city_folium_map(
            graph=self.graph,
            selected_path=path,
            source_node="A",
            destination_node="I",
            mode_name="energy"
        )
        self.assertIsInstance(f_map, folium.Map)
        html_str = f_map._repr_html_()
        self.assertTrue(len(html_str) > 1000)


if __name__ == "__main__":
    unittest.main()
