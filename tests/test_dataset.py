"""
tests/test_dataset.py
Unit tests for EV energy consumption dataset validation.
"""

import unittest
from pathlib import Path
import pandas as pd


class TestEVDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = Path(__file__).resolve().parent.parent / "data" / "ev_energy_dataset.csv"
        cls.df = pd.read_csv(cls.dataset_path)

    def test_file_exists(self):
        """Verify the dataset file exists at data/ev_energy_dataset.csv."""
        self.assertTrue(self.dataset_path.exists(), "Dataset file does not exist.")

    def test_minimum_sample_size(self):
        """Verify at least 1000 simulated records are generated."""
        self.assertGreaterEqual(
            len(self.df),
            1000,
            f"Expected at least 1000 rows, found {len(self.df)}"
        )

    def test_no_missing_values(self):
        """Verify the dataset contains zero null or NaN values."""
        null_count = self.df.isnull().sum().sum()
        self.assertEqual(null_count, 0, f"Found {null_count} missing values.")

    def test_required_columns_present(self):
        """Verify all mandatory features are in the dataset."""
        required = [
            "distance_km",
            "speed_kmh",
            "traffic_level_encoded",
            "slope_percent",
            "vehicle_load_kg",
            "temperature_c",
            "energy_consumed_kwh"
        ]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")

    def test_target_value_constraints(self):
        """Verify energy consumption is strictly positive."""
        self.assertTrue((self.df["energy_consumed_kwh"] > 0).all())

    def test_traffic_encoding_range(self):
        """Verify traffic level encoded values are in {0, 1, 2}."""
        unique_vals = set(self.df["traffic_level_encoded"].unique())
        self.assertTrue(unique_vals.issubset({0, 1, 2}))

    def test_correlation_direction(self):
        """Verify distance and slope have strong positive correlation with energy consumption."""
        corr = self.df[["distance_km", "slope_percent", "energy_consumed_kwh"]].corr()["energy_consumed_kwh"]
        self.assertGreater(corr["distance_km"], 0.5, "Distance should strongly correlate positively with energy.")
        self.assertGreater(corr["slope_percent"], 0.4, "Slope should strongly correlate positively with energy.")


if __name__ == "__main__":
    unittest.main()
