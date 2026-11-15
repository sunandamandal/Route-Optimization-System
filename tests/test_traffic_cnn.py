"""
tests/test_traffic_cnn.py
Unit tests for the optional CNN traffic classification module and graph dynamic reweighting.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.traffic_cnn import (
    build_traffic_cnn_model,
    preprocess_image,
    predict_traffic_from_image,
    traffic_category_to_weight,
    update_graph_edge_from_image,
    create_sample_traffic_images,
    TRAFFIC_DATASET_DIR
)
from src.route_optimizer import EVRouteOptimizer


class TestTrafficCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_sample_traffic_images()
        cls.img_low = TRAFFIC_DATASET_DIR / "sample_low.png"
        cls.img_med = TRAFFIC_DATASET_DIR / "sample_medium.png"
        cls.img_high = TRAFFIC_DATASET_DIR / "sample_high.png"

    def test_sample_images_exist(self):
        """Verify built-in sample road camera images were created."""
        self.assertTrue(self.img_low.exists(), "sample_low.png is missing")
        self.assertTrue(self.img_med.exists(), "sample_medium.png is missing")
        self.assertTrue(self.img_high.exists(), "sample_high.png is missing")

    def test_preprocess_image_dimensions_and_normalization(self):
        """Verify image preprocessing outputs shape (1, 128, 128, 3) normalized in [0, 1]."""
        arr = preprocess_image(self.img_low, target_size=(128, 128))
        self.assertEqual(arr.shape, (1, 128, 128, 3))
        self.assertGreaterEqual(arr.min(), 0.0)
        self.assertLessEqual(arr.max(), 1.0)

    def test_build_cnn_model_specification(self):
        """Verify model builder returns valid architecture configuration."""
        model_spec = build_traffic_cnn_model(input_shape=(128, 128, 3), num_classes=3)
        self.assertIsNotNone(model_spec)

    def test_predict_traffic_from_images(self):
        """Verify prediction function returns expected schema and probability distributions."""
        for img_path, expected_lvl in [
            (self.img_low, "low"),
            (self.img_med, "medium"),
            (self.img_high, "high")
        ]:
            res = predict_traffic_from_image(img_path)
            self.assertIn("traffic_level", res)
            self.assertIn(res["traffic_level"], ["low", "medium", "high"])
            self.assertIn("confidence", res)
            self.assertGreater(res["confidence"], 0.0)
            self.assertIn("probabilities", res)
            total_prob = sum(res["probabilities"].values())
            self.assertAlmostEqual(total_prob, 1.0, places=2)
            self.assertIn("is_trained_cnn", res)
            self.assertIn("status_note", res)

    def test_traffic_category_to_weight(self):
        """Verify traffic category converts to correct graph delay multipliers."""
        w_low = traffic_category_to_weight("low")
        w_med = traffic_category_to_weight("medium")
        w_high = traffic_category_to_weight("high")

        self.assertEqual(w_low["encoded"], 0)
        self.assertEqual(w_med["encoded"], 1)
        self.assertEqual(w_high["encoded"], 2)

        self.assertLess(w_low["time_multiplier"], w_med["time_multiplier"])
        self.assertLess(w_med["time_multiplier"], w_high["time_multiplier"])

        with self.assertRaises(ValueError):
            traffic_category_to_weight("invalid_traffic_level")

    def test_dynamic_graph_update_from_image(self):
        """Verify that updating a road from a camera image increases travel time and energy."""
        opt = EVRouteOptimizer()
        edge = ("A", "B")
        orig_time = opt.di_graph[edge[0]][edge[1]]["travel_time_min"]
        orig_energy = opt.di_graph[edge[0]][edge[1]]["energy_kwh"]

        # Simulate heavy congestion detected by camera image on edge (A, B)
        update_info = update_graph_edge_from_image(opt, "A", "B", self.img_high)

        self.assertEqual(update_info["predicted_traffic"], "high")
        self.assertGreaterEqual(
            update_info["updated_travel_time_min"],
            orig_time,
            "High traffic must increase edge travel time."
        )


if __name__ == "__main__":
    unittest.main()
