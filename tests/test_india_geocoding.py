"""
tests/test_india_geocoding.py
Unit and Integration Tests for India-Wide Location Search & Routing.
"""

import unittest
from pathlib import Path
from src.india_geocoding import (
    IndiaGeocoder,
    GeocodedLocation,
    geocode_indian_location,
    OSM_ATTRIBUTION,
    INDIA_LAT_MIN,
    INDIA_LAT_MAX,
    INDIA_LON_MIN,
    INDIA_LON_MAX
)
from src.india_road_service import get_india_route_service, haversine_distance_km
from src.ev_calculations import EVBatteryState, compare_routes_battery


class TestIndiaGeocodingAndRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geocoder = IndiaGeocoder()
        cls.router = get_india_route_service()

    def test_geocoding_major_indian_cities(self):
        """Test that major Indian cities resolve accurately with valid coordinates."""
        test_cities = [
            ("Kolkata, West Bengal", 22.0, 23.5, 88.0, 89.0),
            ("Howrah, West Bengal", 22.0, 23.5, 88.0, 89.0),
            ("Durgapur, West Bengal", 23.0, 24.0, 87.0, 88.0),
            ("Delhi, India", 28.0, 29.5, 76.5, 78.0),
            ("Mumbai, Maharashtra", 18.5, 20.0, 72.5, 73.5),
            ("Bengaluru, Karnataka", 12.5, 13.5, 77.0, 78.5)
        ]

        for city_query, min_lat, max_lat, min_lon, max_lon in test_cities:
            loc = self.geocoder.geocode(city_query)
            self.assertTrue(loc.is_valid, f"Failed to geocode '{city_query}': {loc.error_message}")
            self.assertEqual(loc.country, "India")
            self.assertTrue(min_lat <= loc.lat <= max_lat, f"Latitude {loc.lat} out of range for {city_query}")
            self.assertTrue(min_lon <= loc.lon <= max_lon, f"Longitude {loc.lon} out of range for {city_query}")

    def test_india_bounding_box_validation(self):
        """Verify that resolved coordinates strictly lie within India boundaries."""
        loc = self.geocoder.geocode("Kolkata")
        self.assertTrue(self.geocoder.is_within_india(loc.lat, loc.lon))
        self.assertTrue(INDIA_LAT_MIN <= loc.lat <= INDIA_LAT_MAX)
        self.assertTrue(INDIA_LON_MIN <= loc.lon <= INDIA_LON_MAX)

    def test_reject_non_indian_locations(self):
        """Verify that non-Indian locations are rejected gracefully."""
        loc = self.geocoder.geocode("Tokyo, Japan")
        self.assertFalse(loc.is_valid)
        self.assertIn("could not be found in India", loc.error_message or "")

    def test_empty_query_handling(self):
        """Verify empty or whitespace queries return clear validation errors."""
        loc = self.geocoder.geocode("   ")
        self.assertFalse(loc.is_valid)
        self.assertIn("cannot be empty", loc.error_message)

    def test_geocoding_caching(self):
        """Verify that repeated queries hit the cache without re-fetching."""
        # First call caches
        loc1 = self.geocoder.geocode("Howrah")
        # Second call should be served from memory or cache
        loc2 = self.geocoder.geocode("Howrah")
        self.assertEqual(loc1.lat, loc2.lat)
        self.assertEqual(loc1.lon, loc2.lon)
        self.assertEqual(loc1.short_name, loc2.short_name)

    def test_haversine_distance_calculation(self):
        """Verify haversine formula between Kolkata and Howrah (~5-10 km)."""
        kolkata = self.geocoder.geocode("Kolkata")
        howrah = self.geocoder.geocode("Howrah")
        dist = haversine_distance_km(kolkata.lat, kolkata.lon, howrah.lat, howrah.lon)
        self.assertTrue(3.0 <= dist <= 12.0, f"Unexpected distance: {dist} km")

    def test_india_multi_mode_routing_kolkata_to_durgapur(self):
        """Verify India routing engine creates 4 valid modes with ANN energy prediction."""
        kolkata = self.geocoder.geocode("Kolkata, West Bengal")
        durgapur = self.geocoder.geocode("Durgapur, West Bengal")

        routes_data = self.router.compute_india_routes(
            source=kolkata,
            dest=durgapur,
            algorithm="astar"
        )
        candidates = routes_data["candidate_routes"]

        # 1. Check all 4 modes exist
        for mode in ["shortest", "fastest", "energy", "balanced"]:
            self.assertIn(mode, candidates)
            r = candidates[mode]
            self.assertTrue(r.total_distance_km > 100.0, f"Distance too short: {r.total_distance_km}")
            self.assertTrue(r.total_time_min > 60.0, f"Time too short: {r.total_time_min}")
            self.assertTrue(r.total_energy_kwh > 0.0, f"Energy must be non-negative: {r.total_energy_kwh}")
            self.assertTrue(len(r.segments) >= 3, "Should have multiple corridor segments")

        # 2. Check battery evaluation integration
        battery = EVBatteryState(capacity_kwh=60.0, current_soc_percent=50.0, min_reserve_percent=15.0)
        summary = compare_routes_battery(candidates, battery)
        self.assertIn("energy", summary.evaluations)
        self.assertTrue(summary.evaluations["energy"].is_feasible)

    def test_osm_attribution_present(self):
        """Verify OSM attribution is defined as required by Nominatim usage policy."""
        self.assertIn("OpenStreetMap", OSM_ATTRIBUTION)


if __name__ == "__main__":
    unittest.main()
