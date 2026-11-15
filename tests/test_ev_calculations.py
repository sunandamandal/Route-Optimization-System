"""
tests/test_ev_calculations.py
Unit tests for EV battery calculations, feasibility logic, and energy comparisons.
"""

import unittest
from src.ev_calculations import (
    EVBatteryState,
    FeasibilityStatus,
    calculate_available_battery_energy,
    evaluate_route_battery,
    compare_routes_battery,
    format_battery_report
)


class TestEVCalculations(unittest.TestCase):
    def setUp(self):
        # Standard EV battery: 60.0 kWh pack, 50% SoC, 15% reserve
        # Current energy = 30.0 kWh, Reserve = 9.0 kWh, Usable = 21.0 kWh
        self.battery = EVBatteryState(
            capacity_kwh=60.0,
            current_soc_percent=50.0,
            min_reserve_percent=15.0
        )

    def test_battery_energy_properties(self):
        """Verify nominal, current, reserve, and usable energy calculations."""
        self.assertEqual(self.battery.capacity_kwh, 60.0)
        self.assertEqual(self.battery.current_energy_kwh, 30.0)
        self.assertEqual(self.battery.reserve_energy_kwh, 9.0)
        self.assertEqual(self.battery.usable_energy_kwh, 21.0)
        self.assertEqual(self.battery.usable_soc_percent, 35.0)

    def test_calculate_available_battery_energy_function(self):
        """Verify functional wrapper calculate_available_battery_energy."""
        res = calculate_available_battery_energy(75.0, 80.0, 20.0)
        self.assertEqual(res["capacity_kwh"], 75.0)
        self.assertEqual(res["current_energy_kwh"], 60.0)
        self.assertEqual(res["reserve_energy_kwh"], 15.0)
        self.assertEqual(res["usable_energy_kwh"], 45.0)
        self.assertEqual(res["usable_soc_percent"], 60.0)

    def test_invalid_battery_inputs_raise_value_error(self):
        """Verify boundary input validation."""
        with self.assertRaises(ValueError):
            EVBatteryState(capacity_kwh=-10.0, current_soc_percent=50.0)
        with self.assertRaises(ValueError):
            EVBatteryState(capacity_kwh=60.0, current_soc_percent=105.0)
        with self.assertRaises(ValueError):
            EVBatteryState(capacity_kwh=60.0, current_soc_percent=-5.0)
        with self.assertRaises(ValueError):
            EVBatteryState(capacity_kwh=60.0, current_soc_percent=50.0, min_reserve_percent=100.0)

    def test_safe_route_evaluation(self):
        """Verify route evaluation when energy consumed is well within usable capacity."""
        eval_safe = evaluate_route_battery(
            energy_consumed_kwh=10.0,
            battery_state=self.battery,
            distance_km=50.0,
            travel_time_min=45.0,
            mode_name="energy"
        )
        self.assertTrue(eval_safe.is_feasible)
        self.assertTrue(eval_safe.is_safe)
        self.assertEqual(eval_safe.status, FeasibilityStatus.SAFE)
        self.assertAlmostEqual(eval_safe.remaining_energy_kwh, 20.0, places=2)
        self.assertAlmostEqual(eval_safe.remaining_soc_percent, 33.33, places=1)
        self.assertEqual(eval_safe.energy_deficit_kwh, 0.0)
        self.assertIsNone(eval_safe.warning_message)

    def test_reserve_breached_route_evaluation(self):
        """Verify caution state when route consumes usable energy and dips into reserve buffer."""
        # Consumes 25 kWh (usable is 21 kWh, total available is 30 kWh)
        eval_breach = evaluate_route_battery(
            energy_consumed_kwh=25.0,
            battery_state=self.battery,
            distance_km=120.0,
            travel_time_min=100.0,
            mode_name="fastest"
        )
        self.assertTrue(eval_breach.is_feasible)
        self.assertFalse(eval_breach.is_safe)
        self.assertEqual(eval_breach.status, FeasibilityStatus.RESERVE_BREACHED)
        self.assertAlmostEqual(eval_breach.remaining_energy_kwh, 5.0, places=2)
        self.assertAlmostEqual(eval_breach.remaining_soc_percent, 8.33, places=1)
        self.assertAlmostEqual(eval_breach.energy_deficit_kwh, 4.0, places=2)
        self.assertGreater(eval_breach.charging_needed_kwh, 4.0)
        self.assertIsNotNone(eval_breach.warning_message)
        self.assertIn("reserve", eval_breach.warning_message.lower())

    def test_infeasible_route_strictly_cannot_reach(self):
        """Verify strict non-reachability when energy consumed exceeds total available energy."""
        # Consumes 35 kWh (total available is only 30 kWh)
        eval_infeasible = evaluate_route_battery(
            energy_consumed_kwh=35.0,
            battery_state=self.battery,
            distance_km=180.0,
            travel_time_min=150.0,
            mode_name="shortest"
        )
        # CRITICAL RULE: Must NOT claim vehicle reaches destination!
        self.assertFalse(eval_infeasible.is_feasible)
        self.assertFalse(eval_infeasible.is_safe)
        self.assertEqual(eval_infeasible.status, FeasibilityStatus.INFEASIBLE)
        self.assertLess(eval_infeasible.remaining_soc_percent, 0.0)
        self.assertEqual(eval_infeasible.critical_deficit_kwh, 5.0)
        self.assertIsNotNone(eval_infeasible.warning_message)
        self.assertIn("CANNOT reach", eval_infeasible.warning_message)

    def test_compare_routes_and_savings(self):
        """Verify comparative energy analysis and percentage savings calculations."""
        candidate_routes = {
            "shortest": {"energy_kwh": 12.0, "distance_km": 60.0, "time_min": 55.0},
            "fastest":  {"energy_kwh": 11.0, "distance_km": 65.0, "time_min": 45.0},
            "energy":   {"energy_kwh": 9.0,  "distance_km": 63.0, "time_min": 50.0},
            "balanced": {"energy_kwh": 9.5,  "distance_km": 62.0, "time_min": 48.0}
        }
        summary = compare_routes_battery(candidate_routes, self.battery)

        # Savings vs Shortest: 12.0 - 9.0 = 3.0 kWh (25.0%)
        self.assertAlmostEqual(summary.savings_vs_shortest_kwh, 3.0, places=2)
        self.assertAlmostEqual(summary.savings_vs_shortest_percent, 25.0, places=1)

        # Savings vs Fastest: 11.0 - 9.0 = 2.0 kWh (18.18%)
        self.assertAlmostEqual(summary.savings_vs_fastest_kwh, 2.0, places=2)
        self.assertAlmostEqual(summary.savings_vs_fastest_percent, 18.18, places=1)

        self.assertEqual(summary.recommended_mode, "energy")

    def test_format_battery_report(self):
        """Verify report string formatting."""
        candidate_routes = {
            "shortest": {"energy_kwh": 12.0, "distance_km": 60.0, "time_min": 55.0},
            "fastest":  {"energy_kwh": 11.0, "distance_km": 65.0, "time_min": 45.0},
            "energy":   {"energy_kwh": 9.0,  "distance_km": 63.0, "time_min": 50.0}
        }
        summary = compare_routes_battery(candidate_routes, self.battery)
        report_text = format_battery_report(summary)
        self.assertIn("EV BATTERY & ROUTE FEASIBILITY REPORT", report_text)
        self.assertIn("60.0 kWh", report_text)
        self.assertIn("Energy Savings vs. Shortest Route", report_text)


if __name__ == "__main__":
    unittest.main()
