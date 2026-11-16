"""
tests/test_india_real_road_routing.py
Comprehensive Unit and Integration Tests for Real-Road India Navigation & OSRM Engine.
"""

import unittest
from pathlib import Path
from src.india_geocoding import geocode_indian_location
from src.india_road_service import (
    BaseRoutingProvider,
    OSRMRoutingProvider,
    SimulatedFallbackRoutingProvider,
    EVRouteEvaluator,
    IndiaRouteService,
    RawRoadRoute,
    TurnInstruction,
    get_india_route_service,
    haversine_distance_km
)
from src.ev_calculations import EVBatteryState, compare_routes_battery


class TestIndiaRealRoadRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.osrm_provider = OSRMRoutingProvider(timeout_sec=12.0)
        cls.fallback_provider = SimulatedFallbackRoutingProvider()
        cls.service = get_india_route_service()

    def test_provider_interface_conformance(self):
        """Verify that both OSRM and Fallback providers inherit from BaseRoutingProvider."""
        self.assertTrue(issubclass(OSRMRoutingProvider, BaseRoutingProvider))
        self.assertTrue(issubclass(SimulatedFallbackRoutingProvider, BaseRoutingProvider))

    def test_osrm_real_route_and_geometry_kolkata_to_howrah(self):
        """Verify OSRM fetches real road geometry and turn-by-turn steps between Kolkata and Howrah."""
        kolkata = geocode_indian_location("Kolkata, West Bengal")
        howrah = geocode_indian_location("Howrah, West Bengal")

        routes = self.osrm_provider.get_routes(kolkata, howrah, alternatives=True)
        self.assertTrue(len(routes) >= 1)

        primary = routes[0]
        self.assertIsInstance(primary, RawRoadRoute)
        self.assertTrue(primary.is_real_osrm)
        self.assertTrue(4.0 <= primary.distance_km <= 15.0, f"Distance {primary.distance_km} km out of realistic range")
        self.assertTrue(len(primary.coordinates) >= 20, "Should have rich GPS polyline")
        self.assertTrue(len(primary.turn_instructions) >= 3, "Should have turn-by-turn steps")

        # Verify turn-by-turn instruction structure
        step1 = primary.turn_instructions[0]
        self.assertIsInstance(step1, TurnInstruction)
        self.assertEqual(step1.step_number, 1)
        self.assertTrue(len(step1.instruction) > 0)

    def test_interstate_routing_delhi_to_jaipur(self):
        """Verify real highway routing across state borders (Delhi to Rajasthan NH48)."""
        delhi = geocode_indian_location("Delhi, India")
        jaipur = geocode_indian_location("Jaipur, Rajasthan")

        routes = self.osrm_provider.get_routes(delhi, jaipur, alternatives=True)
        self.assertTrue(len(routes) >= 1)

        primary = routes[0]
        # Real driving distance via NH48 is ~280 - 320 km
        self.assertTrue(260.0 <= primary.distance_km <= 340.0, f"Distance {primary.distance_km} unexpected")
        self.assertTrue(primary.duration_min > 150.0)
        self.assertTrue(len(primary.coordinates) > 100)

    def test_interstate_routing_bengaluru_to_chennai(self):
        """Verify inter-state highway routing between Karnataka and Tamil Nadu."""
        blr = geocode_indian_location("Bengaluru, Karnataka")
        chn = geocode_indian_location("Chennai, Tamil Nadu")

        routes = self.osrm_provider.get_routes(blr, chn, alternatives=True)
        self.assertTrue(len(routes) >= 1)
        primary = routes[0]
        # Real driving distance is ~320 - 360 km
        self.assertTrue(290.0 <= primary.distance_km <= 380.0)

    def test_scientific_separation_fastest_vs_energy_optimal(self):
        """
        Verify that OSRM fastest route (time-optimized) is NOT falsely labeled
        as the energy-optimal route, and that ANN energy evaluation distinguishes them.
        """
        kolkata = geocode_indian_location("Kolkata, West Bengal")
        durgapur = geocode_indian_location("Durgapur, West Bengal")

        result = self.service.compute_india_routes(kolkata, durgapur, algorithm="astar")
        candidates = result["candidate_routes"]

        self.assertIn("fastest", candidates)
        self.assertIn("energy", candidates)

        fastest = candidates["fastest"]
        energy = candidates["energy"]

        # Fastest route minimizes travel time
        self.assertTrue(fastest.total_time_min <= energy.total_time_min,
                        f"Fastest ({fastest.total_time_min}) should be <= Energy ({energy.total_time_min})")

        # Energy-efficient route minimizes ANN-predicted kWh
        self.assertTrue(energy.total_energy_kwh <= fastest.total_energy_kwh,
                        f"Energy ({energy.total_energy_kwh} kWh) should be <= Fastest ({fastest.total_energy_kwh} kWh)")

        # Verify energy savings
        savings = fastest.total_energy_kwh - energy.total_energy_kwh
        self.assertTrue(savings >= 0.0)

    def test_fallback_provider_resilience(self):
        """Verify fallback provider produces valid candidate routes with geometry and steps."""
        src = geocode_indian_location("Kolkata, West Bengal")
        dst = geocode_indian_location("Durgapur, West Bengal")

        fb_routes = self.fallback_provider.get_routes(src, dst, alternatives=True)
        self.assertTrue(len(fb_routes) >= 1)
        self.assertFalse(fb_routes[0].is_real_osrm)
        self.assertTrue(fb_routes[0].distance_km > 100.0)
        self.assertTrue(len(fb_routes[0].coordinates) >= 15)

    def test_battery_feasibility_on_real_route(self):
        """Verify EV battery calculations and charging recommendations on a real 190 km trip."""
        kolkata = geocode_indian_location("Kolkata, West Bengal")
        durgapur = geocode_indian_location("Durgapur, West Bengal")

        result = self.service.compute_india_routes(kolkata, durgapur)
        candidates = result["candidate_routes"]

        # Case A: Adequate battery (75 kWh, 90% SoC -> ~67.5 kWh)
        battery_safe = EVBatteryState(capacity_kwh=75.0, current_soc_percent=90.0, min_reserve_percent=15.0)
        summary_safe = compare_routes_battery(candidates, battery_safe)
        self.assertTrue(summary_safe.evaluations["energy"].is_safe)

        # Case B: Low battery (40 kWh, 20% SoC -> 8 kWh available, trip needs ~22 kWh)
        battery_low = EVBatteryState(capacity_kwh=40.0, current_soc_percent=20.0, min_reserve_percent=15.0)
        summary_low = compare_routes_battery(candidates, battery_low)
        self.assertFalse(summary_low.evaluations["energy"].is_feasible)
        self.assertTrue(summary_low.evaluations["energy"].charging_needed_kwh > 0.0)

    def test_ann_feature_extraction_and_provenance(self):
        """
        Verify that all 7 required ANN input features are explicitly extracted
        and labeled with their provenance status (no silently invented values).
        """
        evaluator = self.service.evaluator
        feat_set = evaluator.extract_ann_segment_features(
            distance_km=15.2,
            duration_min=12.5,
            source_elevation_m=9.0,
            dest_elevation_m=75.0,
            seg_idx=0,
            total_segments=5,
            vehicle_load_kg=80.0,
            temperature_c=25.0
        )

        # 1. Verify all 7 features are present
        required_features = [
            "distance_km", "speed_kmh", "traffic_level_encoded",
            "slope_percent", "elevation_change_m", "vehicle_load_kg", "temperature_c"
        ]
        for f in required_features:
            self.assertIn(f, feat_set.feature_provenance)
            prov = feat_set.feature_provenance[f]
            self.assertIn("value", prov)
            self.assertIn("status", prov)
            self.assertIn("source", prov)

        # 2. Check provenance labels
        self.assertEqual(feat_set.feature_provenance["distance_km"]["status"], "DIRECT_MEASURED")
        self.assertEqual(feat_set.feature_provenance["speed_kmh"]["status"], "DERIVED_MEASURED")
        self.assertEqual(feat_set.feature_provenance["traffic_level_encoded"]["status"], "ESTIMATED_TRAFFIC")
        self.assertEqual(feat_set.feature_provenance["elevation_change_m"]["status"], "ESTIMATED_TOPOGRAPHY")
        self.assertEqual(feat_set.feature_provenance["slope_percent"]["status"], "COMPUTED_FORMULA")
        self.assertEqual(feat_set.feature_provenance["vehicle_load_kg"]["status"], "USER_SPECIFIED")
        self.assertEqual(feat_set.feature_provenance["temperature_c"]["status"], "USER_SPECIFIED")

        # 3. Verify predicted energy is strictly positive and bounded
        self.assertTrue(feat_set.predicted_kwh > 0.0)

    def test_ann_total_route_energy_equals_sum_of_segments(self):
        """Verify total predicted route energy is exactly the sum of its individual ANN segment predictions."""
        kolkata = geocode_indian_location("Kolkata, West Bengal")
        durgapur = geocode_indian_location("Durgapur, West Bengal")

        result = self.service.compute_india_routes(kolkata, durgapur)
        for mode_key, route in result["candidate_routes"].items():
            segment_energy_sum = sum(seg.energy_kwh for seg in route.segments)
            self.assertAlmostEqual(route.total_energy_kwh, segment_energy_sum, places=2)

            segment_dist_sum = sum(seg.distance_km for seg in route.segments)
            self.assertAlmostEqual(route.total_distance_km, segment_dist_sum, places=2)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
