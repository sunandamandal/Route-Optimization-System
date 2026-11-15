"""
src/energy_model.py
Modular inference module for Electric Vehicle Energy Consumption prediction.
Provides the EVEnergyPredictor class and helper functions for road-network integration.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ann_energy_model.joblib"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"

FEATURE_NAMES = [
    "distance_km",
    "speed_kmh",
    "traffic_level_encoded",
    "slope_percent",
    "elevation_change_m",
    "vehicle_load_kg",
    "temperature_c"
]


class EVEnergyPredictor:
    """
    Wrapper for loading the trained MLPRegressor ANN model and StandardScaler,
    providing streamlined energy prediction for EV routing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        """Loads trained weights and feature scalers using joblib."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained ANN model not found at {self.model_path}. "
                "Please run train_model.py first."
            )
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Trained Scaler not found at {self.scaler_path}. "
                "Please run train_model.py first."
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(
        self,
        distance_km: float,
        speed_kmh: float,
        traffic_level_encoded: int,
        slope_percent: Optional[float] = None,
        elevation_change_m: Optional[float] = None,
        vehicle_load_kg: float = 75.0,
        temperature_c: float = 20.0
    ) -> float:
        """
        Predicts energy consumption in kWh for a single road segment.

        Parameters:
            distance_km (float): Distance in kilometers.
            speed_kmh (float): Average speed in km/h.
            traffic_level_encoded (int): 0 for Low, 1 for Medium, 2 for High.
            slope_percent (float, optional): Gradient percentage. Calculated if None.
            elevation_change_m (float, optional): Elevation difference in meters. Calculated if None.
            vehicle_load_kg (float): Added payload/passenger weight in kg (default: 75.0 kg).
            temperature_c (float): Ambient temperature in Celsius (default: 20.0°C).

        Returns:
            float: Predicted energy consumption in kilowatt-hours (kWh).
        """
        # Topography reconciliation
        if slope_percent is None and elevation_change_m is not None:
            slope_percent = (elevation_change_m / (distance_km * 1000.0)) * 100.0
        elif elevation_change_m is None and slope_percent is not None:
            elevation_change_m = (distance_km * 1000.0) * (slope_percent / 100.0)
        elif slope_percent is None and elevation_change_m is None:
            slope_percent = 0.0
            elevation_change_m = 0.0

        # Construct input DataFrame preserving feature names
        sample_df = pd.DataFrame([{
            "distance_km": float(distance_km),
            "speed_kmh": float(speed_kmh),
            "traffic_level_encoded": int(traffic_level_encoded),
            "slope_percent": float(slope_percent),
            "elevation_change_m": float(elevation_change_m),
            "vehicle_load_kg": float(vehicle_load_kg),
            "temperature_c": float(temperature_c)
        }])[FEATURE_NAMES]

        # Apply standard scaling (fitted strictly on training set)
        sample_scaled = self.scaler.transform(sample_df)

        # ANN prediction
        predicted_kwh = float(self.model.predict(sample_scaled)[0])

        # Physical boundary: minimum baseline auxiliary power consumption
        min_floor = 0.04 * float(distance_km)
        return max(min_floor, round(predicted_kwh, 4))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts energy consumption for a batch of road segments.

        Parameters:
            df (pd.DataFrame): DataFrame containing the required feature columns.

        Returns:
            np.ndarray: Array of predicted energy consumption in kWh.
        """
        # Ensure all required features are present
        X = df[FEATURE_NAMES].copy()
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Apply physical floor
        floors = 0.04 * X["distance_km"].values
        return np.maximum(floors, np.round(predictions, 4))


# Global cached predictor instance
_PREDICTOR_INSTANCE: Optional[EVEnergyPredictor] = None


def predict_energy_consumption(
    distance_km: float,
    speed_kmh: float,
    traffic_level_encoded: int,
    slope_percent: Optional[float] = None,
    elevation_change_m: Optional[float] = None,
    vehicle_load_kg: float = 75.0,
    temperature_c: float = 20.0,
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> float:
    """
    Convenience function that accepts road conditions and returns
    predicted energy consumption in kWh.
    """
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None or model_path or scaler_path:
        _PREDICTOR_INSTANCE = EVEnergyPredictor(model_path=model_path, scaler_path=scaler_path)

    return _PREDICTOR_INSTANCE.predict(
        distance_km=distance_km,
        speed_kmh=speed_kmh,
        traffic_level_encoded=traffic_level_encoded,
        slope_percent=slope_percent,
        elevation_change_m=elevation_change_m,
        vehicle_load_kg=vehicle_load_kg,
        temperature_c=temperature_c
    )
