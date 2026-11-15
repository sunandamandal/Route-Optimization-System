"""
tests/test_ann_model.py
Unit tests for the trained ANN energy model and inference functions.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.energy_model import EVEnergyPredictor, predict_energy_consumption


class TestEVEnergyPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = EVEnergyPredictor()

    def test_model_and_scaler_files_exist(self):
        """Verify model and scaler joblib files exist in models/."""
        model_path = Path(__file__).resolve().parent.parent / "models" / "ann_energy_model.joblib"
        scaler_path = Path(__file__).resolve().parent.parent / "models" / "scaler.joblib"
        self.assertTrue(model_path.exists(), "ann_energy_model.joblib is missing.")
        self.assertTrue(scaler_path.exists(), "scaler.joblib is missing.")

    def test_single_prediction_output(self):
        """Verify single prediction returns a positive float."""
        pred = predict_energy_consumption(
            distance_km=10.0,
            speed_kmh=50.0,
            traffic_level_encoded=1,
            slope_percent=0.5
        )
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

    def test_elevation_energy_relationship(self):
        """Verify uphill travel predicts higher energy than downhill travel."""
        uphill = predict_energy_consumption(
            distance_km=8.0, speed_kmh=45.0, traffic_level_encoded=0, slope_percent=3.0
        )
        downhill = predict_energy_consumption(
            distance_km=8.0, speed_kmh=45.0, traffic_level_encoded=0, slope_percent=-3.0
        )
        self.assertGreater(
            uphill, downhill,
            f"Uphill ({uphill} kWh) should consume more than downhill ({downhill} kWh)"
        )

    def test_traffic_energy_relationship(self):
        """Verify high traffic predicts higher energy than low traffic."""
        high_traffic = predict_energy_consumption(
            distance_km=6.0, speed_kmh=25.0, traffic_level_encoded=2, slope_percent=0.0
        )
        low_traffic = predict_energy_consumption(
            distance_km=6.0, speed_kmh=50.0, traffic_level_encoded=0, slope_percent=0.0
        )
        self.assertGreater(
            high_traffic, low_traffic,
            f"High traffic ({high_traffic} kWh) should consume more than low traffic ({low_traffic} kWh)"
        )

    def test_temperature_heater_relationship(self):
        """Verify freezing temperature predicts higher energy due to cabin/battery heating."""
        freezing = predict_energy_consumption(
            distance_km=10.0, speed_kmh=40.0, traffic_level_encoded=1, slope_percent=0.0, temperature_c=-5.0
        )
        mild = predict_energy_consumption(
            distance_km=10.0, speed_kmh=40.0, traffic_level_encoded=1, slope_percent=0.0, temperature_c=21.0
        )
        self.assertGreater(
            freezing, mild,
            f"Freezing temp ({freezing} kWh) should consume more than mild temp ({mild} kWh)"
        )

    def test_batch_prediction(self):
        """Verify batch prediction returns an array matching input rows."""
        df_batch = pd.DataFrame([
            {"distance_km": 5.0, "speed_kmh": 40.0, "traffic_level_encoded": 0, "slope_percent": 0.0, "elevation_change_m": 0.0, "vehicle_load_kg": 50.0, "temperature_c": 20.0},
            {"distance_km": 15.0, "speed_kmh": 60.0, "traffic_level_encoded": 1, "slope_percent": 1.2, "elevation_change_m": 180.0, "vehicle_load_kg": 120.0, "temperature_c": 18.0},
        ])
        preds = self.predictor.predict_batch(df_batch)
        self.assertEqual(len(preds), 2)
        self.assertGreater(preds[1], preds[0])


if __name__ == "__main__":
    unittest.main()
