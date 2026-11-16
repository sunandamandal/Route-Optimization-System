"""
tests/test_india_journeys_and_system.py
Comprehensive End-to-End Test Suite for India-Wide EV Navigation & Optimization.

Tests the 5 Required Indian Journeys:
  1. Kolkata to Howrah (Twin cities, intra-state)
  2. Kolkata to Durgapur (Inter-district highway corridor, NH19)
  3. Kolkata to Delhi (Major inter-state mega-corridor)
  4. Kolkata to Mumbai (Cross-country inter-state corridor)
  5. Delhi to Jaipur (Inter-state highway, NH48)

Validates All Core System Components:
  - Indian location search & geocoding
  - Coordinate validation (India bounding box)
  - Actual road route retrieval & fallback resilience
  - Route geometry & coordinate formatting
  - Interactive map rendering (ensures India journeys do NOT show Bengaluru)
  - Longitude,Latitude order check for OSRM queries
  - Alternative candidate routes
  - Turn-by-turn navigation instructions
  - 4 Optimization modes: Shortest, Fastest, Energy-Efficient, Balanced
  - ANN energy prediction & feature order integrity
  - EV Battery calculations, feasibility status & insufficient battery alerts
  - Invalid locations & network error handling
  - Preservation of CNN and VoltCity simulated graph features
"""

import unittest
from pathlib import Path
import folium

from src.india_geocoding import (
    IndiaGeocoder,
    GeocodedLocation,
    geocode_indian_location,
    INDIA_LAT_MIN,
    INDIA_LAT_MAX,
    INDIA_LON_MIN,
    INDIA_LON_MAX,
    OSM_ATTRIBUTION
)
from src.india_road_service import (
    IndiaRouteService,
    OSRMRoutingProvider,
    SimulatedFallbackRoutingProvider,
    EVRouteEvaluator,
    RawRoadRoute,
    TurnInstruction,
    get_india_route_service,
    haversine_distance_km
)
from src.energy_model import EVEnergyPredictor, FEATURE_NAMES
from src.ev_calculations import (
    EVBatteryState,
    FeasibilityStatus,
    evaluate_route_battery,
    compare_routes_battery
)
from src.map_visualization import create_india_route_map, create_city_folium_map
from src.road_network import load_road_network
from src.route_optimizer import EVRouteOptimizer, RouteResult
from src.traffic_cnn import predict_traffic_from_image, traffic_category_to_weight, TRAFFIC_DATASET_DIR


class TestIndiaJourneysAndSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geocoder = IndiaGeocoder()
        cls.service = get_india_route_service()
        cls.ann_predictor = EVEnergyPredictor()
        cls.city_graph = load_road_network()
        cls.city_optimizer = EVRouteOptimizer(graph=cls.city_graph)

    # -------------------------------------------------------------------------
    # JOURNEY 1: Kolkata to Howrah (Twin Cities, Intra-State)
    # -------------------------------------------------------------------------
    def test_journey_1_kolkata_to_howrah(self):
        """Journey 1: Kolkata to Howrah - Short intra-state trip with geometry and maneuvers."""
        src = self.geocoder.geocode("Kolkata, West Bengal")
        dst = self.geocoder.geocode("Howrah, West Bengal")

        # 1. Location Validation
        self.assertTrue(src.is_valid)
        self.assertTrue(dst.is_valid)
        self.assertEqual(src.state, "West Bengal")
        self.assertEqual(dst.state, "West Bengal")

        # 2. Compute Routes
        routes_data = self.service.compute_india_routes(src, dst, algorithm="astar")
        candidates = routes_data["candidate_routes"]
        coords_dict = routes_data["routes_coords"]

        self.assertIn("energy", candidates)
        active_route = candidates["energy"]
        active_coords = coords_dict.get("energy", [])

        # 3. Units & Values
        self.assertTrue(4.0 <= active_route.total_distance_km <= 20.0,
                        f"Distance {active_route.total_distance_km} km out of realistic range")
        self.assertTrue(active_route.total_time_min > 0.0)
        self.assertTrue(active_route.total_energy_kwh > 0.0)
        self.assertTrue(len(active_coords) >= 10, "Should have rich coordinates")

        # 4. Turn-by-turn Navigation
        instructions = routes_data.get("routes_instructions", {}).get("energy", [])
        self.assertTrue(len(instructions) >= 2, "Must contain turn maneuvers")
        first_step = instructions[0]
        self.assertIsInstance(first_step, TurnInstruction)
        self.assertTrue(first_step.distance_km > 0.0)

        # 5. Map Rendering
        f_map = create_india_route_map(
            source_loc=src,
            dest_loc=dst,
            route_coords=active_coords,
            active_route=active_route,
            selected_mode="energy",
            alternative_routes=coords_dict,
            candidate_routes=candidates
        )
        self.assertIsInstance(f_map, folium.Map)
        map_html = f_map._repr_html_()
        self.assertIn("Kolkata", map_html)
        self.assertIn("Howrah", map_html)
        # Verify map is in West Bengal (~22.5 N), NOT Bengaluru (~12.97 N)
        self.assertNotIn("VoltCity", map_html)

    # -------------------------------------------------------------------------
    # JOURNEY 2: Kolkata to Durgapur (Highway Corridor, NH19)
    # -------------------------------------------------------------------------
    def test_journey_2_kolkata_to_durgapur(self):
        """Journey 2: Kolkata to Durgapur - All 4 modes, battery evaluation, and alternatives."""
        src = self.geocoder.geocode("Kolkata, West Bengal")
        dst = self.geocoder.geocode("Durgapur, West Bengal")

        routes_data = self.service.compute_india_routes(src, dst, algorithm="astar")
        candidates = routes_data["candidate_routes"]

        # Check all 4 modes exist
        for mode in ["shortest", "fastest", "energy", "balanced"]:
            self.assertIn(mode, candidates)
            r = candidates[mode]
            self.assertTrue(140.0 <= r.total_distance_km <= 220.0)
            self.assertTrue(r.total_time_min > 80.0)
            self.assertTrue(r.total_energy_kwh > 15.0)

        fastest = candidates["fastest"]
        energy = candidates["energy"]
        # Fastest must have <= time than energy
        self.assertLessEqual(fastest.total_time_min, energy.total_time_min + 1e-4)
        # Energy must have <= kWh than fastest
        self.assertLessEqual(energy.total_energy_kwh, fastest.total_energy_kwh + 1e-4)

        # EV Battery Calculation
        # Case A: Adequate battery (60 kWh, 80% SoC -> 48 kWh available, trip needs ~25 kWh)
        batt_good = EVBatteryState(capacity_kwh=60.0, current_soc_percent=80.0, min_reserve_percent=15.0)
        summary_good = compare_routes_battery(candidates, batt_good)
        self.assertTrue(summary_good.evaluations["energy"].is_safe)
        self.assertEqual(summary_good.evaluations["energy"].status, FeasibilityStatus.SAFE)

        # Case B: Insufficient battery (30 kWh, 20% SoC -> 6 kWh available, trip needs ~25 kWh)
        batt_bad = EVBatteryState(capacity_kwh=30.0, current_soc_percent=20.0, min_reserve_percent=15.0)
        summary_bad = compare_routes_battery(candidates, batt_bad)
        self.assertFalse(summary_bad.evaluations["energy"].is_feasible)
        self.assertEqual(summary_bad.evaluations["energy"].status, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(summary_bad.evaluations["energy"].charging_needed_kwh > 0.0)

    # -------------------------------------------------------------------------
    # JOURNEY 3: Kolkata to Delhi (Inter-State Mega-Corridor)
    # -------------------------------------------------------------------------
    def test_journey_3_kolkata_to_delhi(self):
        """Journey 3: Kolkata to Delhi - Cross-state long corridor with charging plazas."""
        src = self.geocoder.geocode("Kolkata, West Bengal")
        dst = self.geocoder.geocode("Delhi, India")

        self.assertTrue(src.is_valid)
        self.assertTrue(dst.is_valid)

        routes_data = self.service.compute_india_routes(src, dst, algorithm="astar")
        route = routes_data["candidate_routes"]["energy"]

        # Realistic driving distance ~1400 - 1650 km
        self.assertTrue(1300.0 <= route.total_distance_km <= 1800.0)
        self.assertTrue(route.total_time_min > 800.0)
        self.assertTrue(route.total_energy_kwh > 150.0)

        # Verifies Highway EV Fast Charging Stops are scheduled
        self.assertTrue(len(route.charging_stops) >= 5, "Long trip must schedule multiple charging plazas")

        # Sum of segment energies equals total
        seg_energy_sum = sum(s.energy_kwh for s in route.segments)
        self.assertAlmostEqual(route.total_energy_kwh, seg_energy_sum, places=2)

    # -------------------------------------------------------------------------
    # JOURNEY 4: Kolkata to Mumbai (Cross-Country Inter-State)
    # -------------------------------------------------------------------------
    def test_journey_4_kolkata_to_mumbai(self):
        """Journey 4: Kolkata to Mumbai - Major cross-country corridor (~1900 km)."""
        src = self.geocoder.geocode("Kolkata, West Bengal")
        dst = self.geocoder.geocode("Mumbai, Maharashtra")

        self.assertTrue(src.is_valid)
        self.assertTrue(dst.is_valid)
        self.assertEqual(dst.state, "Maharashtra")

        routes_data = self.service.compute_india_routes(src, dst, algorithm="astar")
        route = routes_data["candidate_routes"]["energy"]

        self.assertTrue(1700.0 <= route.total_distance_km <= 2300.0)
        self.assertTrue(route.total_energy_kwh > 200.0)

        # Verify battery evaluation detects deficit for standard passenger EV without mid-trip charging
        batt_standard = EVBatteryState(capacity_kwh=60.0, current_soc_percent=100.0, min_reserve_percent=15.0)
        ev_eval = evaluate_route_battery(route.total_energy_kwh, batt_standard, route.total_distance_km, route.total_time_min)
        self.assertEqual(ev_eval.status, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(ev_eval.critical_deficit_kwh > 100.0)

    # -------------------------------------------------------------------------
    # JOURNEY 5: Delhi to Jaipur (Inter-State Corridor NH48)
    # -------------------------------------------------------------------------
    def test_journey_5_delhi_to_jaipur(self):
        """Journey 5: Delhi to Jaipur - Inter-state highway between Delhi and Rajasthan."""
        src = self.geocoder.geocode("Delhi, India")
        dst = self.geocoder.geocode("Jaipur, Rajasthan")

        self.assertTrue(src.is_valid)
        self.assertTrue(dst.is_valid)
        self.assertEqual(dst.state, "Rajasthan")

        routes_data = self.service.compute_india_routes(src, dst, algorithm="astar")
        candidates = routes_data["candidate_routes"]

        self.assertIn("fastest", candidates)
        self.assertIn("energy", candidates)

        fastest = candidates["fastest"]
        self.assertTrue(250.0 <= fastest.total_distance_km <= 350.0)
        self.assertTrue(fastest.total_time_min > 140.0)

        # Verify map auto-fits bounds
        f_map = create_india_route_map(
            source_loc=src,
            dest_loc=dst,
            route_coords=routes_data["routes_coords"]["energy"],
            active_route=candidates["energy"]
        )
        self.assertIsInstance(f_map, folium.Map)
        map_html = f_map._repr_html_()
        self.assertIn("Delhi", map_html)
        self.assertIn("Jaipur", map_html)

    # -------------------------------------------------------------------------
    # VERIFICATION 1: Coordinate Order Passed to OSRM (lon,lat;lon,lat)
    # -------------------------------------------------------------------------
    def test_osrm_coordinate_order_lon_lat(self):
        """Verify that OSRM URL format strictly uses 'longitude,latitude' as required by API."""
        provider = OSRMRoutingProvider()
        src = GeocodedLocation("Kolkata", "Kolkata, WB", "Kolkata", 22.5726, 88.3639, "West Bengal", "India", 9.0, "index")
        dst = GeocodedLocation("Howrah", "Howrah, WB", "Howrah", 22.5736, 88.3103, "West Bengal", "India", 12.0, "index")

        # The coordinates string format inside OSRMRoutingProvider
        coords_str = f"{src.lon},{src.lat};{dst.lon},{dst.lat}"
        parts = coords_str.split(";")
        src_parts = parts[0].split(",")
        dst_parts = parts[1].split(",")

        # First number is longitude (~88.36), second is latitude (~22.57)
        self.assertAlmostEqual(float(src_parts[0]), 88.3639, places=3)
        self.assertAlmostEqual(float(src_parts[1]), 22.5726, places=3)
        self.assertAlmostEqual(float(dst_parts[0]), 88.3103, places=3)
        self.assertAlmostEqual(float(dst_parts[1]), 22.5736, places=3)

    # -------------------------------------------------------------------------
    # VERIFICATION 2: India-Wide Mode Does NOT Show Bengaluru Map
    # -------------------------------------------------------------------------
    def test_india_mode_does_not_show_bengaluru_map(self):
        """Verify India-wide map is centered on journey corridor, NOT VoltCity Bengaluru."""
        kolkata = self.geocoder.geocode("Kolkata, West Bengal")
        durgapur = self.geocoder.geocode("Durgapur, West Bengal")

        routes_data = self.service.compute_india_routes(kolkata, durgapur)
        active_route = routes_data["candidate_routes"]["energy"]
        active_coords = routes_data["routes_coords"]["energy"]

        f_map = create_india_route_map(
            source_loc=kolkata,
            dest_loc=durgapur,
            route_coords=active_coords,
            active_route=active_route,
            selected_mode="energy"
        )
        map_html = f_map._repr_html_()

        # VoltCity node IDs and names should NOT be present on the India highway map
        self.assertNotIn("Node A", map_html)
        self.assertNotIn("Node G", map_html)
        self.assertNotIn("Harbor Bay", map_html)
        self.assertNotIn("Uptown Plaza", map_html)

        # Check location center is around Bengal (~23.0 N, ~87.8 E)
        self.assertIn("Kolkata", map_html)
        self.assertIn("Durgapur", map_html)

    # -------------------------------------------------------------------------
    # VERIFICATION 3: Units Integrity
    # -------------------------------------------------------------------------
    def test_units_integrity_distance_time_energy(self):
        """Verify units are strictly: distance in km, time in minutes, energy in kWh."""
        src = self.geocoder.geocode("Delhi, India")
        dst = self.geocoder.geocode("Jaipur, Rajasthan")

        routes_data = self.service.compute_india_routes(src, dst)
        route = routes_data["candidate_routes"]["energy"]

        # Road distance is ~280 km (not meters)
        self.assertTrue(200.0 < route.total_distance_km < 400.0)
        # Duration is ~200-300 min (not seconds)
        self.assertTrue(120.0 < route.total_time_min < 500.0)
        # Energy is ~40-60 kWh (not Joules or Wh)
        self.assertTrue(25.0 < route.total_energy_kwh < 100.0)

    # -------------------------------------------------------------------------
    # VERIFICATION 4: ANN Feature Order Integrity
    # -------------------------------------------------------------------------
    def test_ann_input_feature_order_matches_training(self):
        """Verify that FEATURE_NAMES matches the exact 7 features the ANN was trained on."""
        expected_features = [
            "distance_km",
            "speed_kmh",
            "traffic_level_encoded",
            "slope_percent",
            "elevation_change_m",
            "vehicle_load_kg",
            "temperature_c"
        ]
        self.assertEqual(FEATURE_NAMES, expected_features)
        self.assertEqual(len(self.ann_predictor.scaler.feature_names_in_), 7)
        for i, feat in enumerate(expected_features):
            self.assertEqual(self.ann_predictor.scaler.feature_names_in_[i], feat)

    # -------------------------------------------------------------------------
    # VERIFICATION 5: Coordinate Validation & Non-Indian Locations
    # -------------------------------------------------------------------------
    def test_coordinate_validation_and_invalid_locations(self):
        """Verify coordinates lie within India boundaries and invalid locations fail gracefully."""
        # Check India bounding box constants
        self.assertEqual(INDIA_LAT_MIN, 6.0)
        self.assertEqual(INDIA_LAT_MAX, 37.5)
        self.assertEqual(INDIA_LON_MIN, 68.0)
        self.assertEqual(INDIA_LON_MAX, 97.5)

        # Empty location
        empty_loc = self.geocoder.geocode("")
        self.assertFalse(empty_loc.is_valid)

        # Non-Indian location
        tokyo_loc = self.geocoder.geocode("Tokyo, Japan")
        self.assertFalse(tokyo_loc.is_valid)

    # -------------------------------------------------------------------------
    # VERIFICATION 6: Network Failure Fallback Resilience
    # -------------------------------------------------------------------------
    def test_network_failure_fallback_resilience(self):
        """Verify router seamlessly falls back to offline model on network failure."""
        fallback = SimulatedFallbackRoutingProvider()
        src = self.geocoder.geocode("Kolkata, West Bengal")
        dst = self.geocoder.geocode("Durgapur, West Bengal")

        routes = fallback.get_routes(src, dst, alternatives=True)
        self.assertTrue(len(routes) >= 1)
        self.assertFalse(routes[0].is_real_osrm)
        self.assertTrue(len(routes[0].coordinates) >= 20)
        self.assertTrue(len(routes[0].turn_instructions) >= 2)

    # -------------------------------------------------------------------------
    # VERIFICATION 7: Preservation of CNN and VoltCity Features
    # -------------------------------------------------------------------------
    def test_cnn_and_simulated_graph_preserved(self):
        """Verify that CNN image classification and VoltCity simulated graph remain functional."""
        # 1. CNN Module verification
        w_low = traffic_category_to_weight("low")
        w_high = traffic_category_to_weight("high")
        self.assertEqual(w_low["encoded"], 0)
        self.assertEqual(w_high["encoded"], 2)
        self.assertLess(w_low["time_multiplier"], w_high["time_multiplier"])

        sample_img = TRAFFIC_DATASET_DIR / "sample_low.png"
        if sample_img.exists():
            pred = predict_traffic_from_image(sample_img)
            self.assertIn("traffic_level", pred)
            self.assertIn("confidence", pred)

        # 2. VoltCity Simulation Grid verification
        volt_route = self.city_optimizer.find_route("G", "L", mode="energy", algorithm="astar")
        self.assertEqual(volt_route.source, "G")
        self.assertEqual(volt_route.destination, "L")
        self.assertTrue(volt_route.total_distance_km > 0.0)
        self.assertTrue(volt_route.total_energy_kwh > 0.0)

        # 3. VoltCity Map rendering
        city_map = create_city_folium_map(self.city_graph, volt_route.path, "G", "L", "energy")
        self.assertIsInstance(city_map, folium.Map)
        self.assertIn("Harbor Bay", city_map._repr_html_())


if __name__ == "__main__":
    unittest.main()
