"""
tests/test_map_visualization.py
Unit tests for interactive Folium mapping and geographic projections.
"""

import unittest
from pathlib import Path
import folium
from src.road_network import load_road_network
from src.map_visualization import create_city_folium_map, create_india_route_map, node_to_latlon
from src.route_optimizer import RouteResult
from src.india_geocoding import GeocodedLocation


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

    def test_create_india_route_map_with_alternatives(self):
        """Verify India-wide road map generation with active and alternative routes."""
        source = GeocodedLocation(
            query="Kolkata",
            display_name="Kolkata, West Bengal, India",
            short_name="Kolkata",
            lat=22.5726,
            lon=88.3639,
            state="West Bengal",
            country="India",
            elevation_m=9.0,
            source="offline_index",
            is_valid=True
        )
        dest = GeocodedLocation(
            query="Durgapur",
            display_name="Durgapur, West Bengal, India",
            short_name="Durgapur",
            lat=23.5204,
            lon=87.3119,
            state="West Bengal",
            country="India",
            elevation_m=65.0,
            source="offline_index",
            is_valid=True
        )
        active_route = RouteResult(
            mode="energy",
            algorithm="astar",
            source="Kolkata",
            destination="Durgapur",
            path=["Kolkata", "Durgapur"],
            total_distance_km=170.5,
            total_time_min=185.0,
            total_energy_kwh=28.4,
            charging_stops=["NH19 EV Plaza (60 kW DC)"],
            segments=[]
        )
        fastest_route = RouteResult(
            mode="fastest",
            algorithm="astar",
            source="Kolkata",
            destination="Durgapur",
            path=["Kolkata", "Durgapur"],
            total_distance_km=172.0,
            total_time_min=170.0,
            total_energy_kwh=31.5,
            charging_stops=[],
            segments=[]
        )
        route_coords = [(22.5726, 88.3639), (23.0, 87.8), (23.5204, 87.3119)]
        alt_routes = {
            "fastest": [(22.5726, 88.3639), (23.1, 87.7), (23.5204, 87.3119)],
            "shortest": [(22.5726, 88.3639), (22.9, 87.9), (23.5204, 87.3119)]
        }
        candidate_routes = {"fastest": fastest_route, "energy": active_route}

        f_map = create_india_route_map(
            source_loc=source,
            dest_loc=dest,
            route_coords=route_coords,
            active_route=active_route,
            selected_mode="energy",
            alternative_routes=alt_routes,
            candidate_routes=candidate_routes
        )

        self.assertIsInstance(f_map, folium.Map)
        html_str = f_map._repr_html_()
        self.assertIn("folium", html_str.lower())
        self.assertIn("openstreetmap", html_str.lower())
        self.assertIn("Kolkata", html_str)
        self.assertIn("Durgapur", html_str)
        self.assertTrue(len(html_str) > 2000)


if __name__ == "__main__":
    unittest.main()
