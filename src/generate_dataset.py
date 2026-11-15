"""
src/generate_dataset.py
Physics-grounded dataset generator for Electric Vehicle (EV) Energy Consumption.
Generates realistic, reproducible road-trip data for training Artificial Neural Networks (ANN).
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ev_energy_dataset.csv"


def calculate_physics_ev_energy(
    distance_km: np.ndarray,
    speed_kmh: np.ndarray,
    traffic_level: np.ndarray,
    slope_percent: np.ndarray,
    vehicle_load_kg: np.ndarray,
    temperature_c: np.ndarray,
    curb_weight_kg: float = 1750.0,
    regen_efficiency: float = 0.65,
    motor_efficiency: float = 0.90,
    random_seed: int = 42
) -> np.ndarray:
    """
    Computes realistic EV energy consumption (kWh) based on vehicle dynamics:
    1. Rolling Resistance: Crr * m * g * distance
    2. Aerodynamic Drag: 0.5 * rho * Cd * A * v^2 * distance
    3. Gravitational Potential Energy: m * g * delta_h (with regen on downhills)
    4. Traffic Congestion Factor: Stop-and-go acceleration losses and idling
    5. Auxiliary & HVAC Energy: Battery heating/cooling and cabin climate control based on ambient temperature
    """
    rng = np.random.default_rng(random_seed)
    n = len(distance_km)

    # Physical constants
    g = 9.81             # Gravity (m/s^2)
    rho_air = 1.225      # Air density (kg/m^3)
    c_rr = 0.0105        # Rolling resistance coefficient
    cd = 0.24            # Aerodynamic drag coefficient (typical modern EV)
    frontal_area = 2.25  # Frontal area (m^2)

    # Total vehicle mass (curb weight + payload)
    total_mass_kg = curb_weight_kg + vehicle_load_kg

    # Convert units
    dist_m = distance_km * 1000.0
    speed_ms = speed_kmh / 3.6
    time_hours = distance_km / np.maximum(speed_kmh, 5.0)  # Avoid division by zero

    # 1. Rolling Resistance Energy (Joules -> kWh)
    # E_roll = C_rr * m * g * distance
    energy_roll_j = c_rr * total_mass_kg * g * dist_m
    energy_roll_kwh = (energy_roll_j / 3.6e6) / motor_efficiency

    # 2. Aerodynamic Drag Energy (Joules -> kWh)
    # E_aero = 0.5 * rho * Cd * A * v^2 * distance
    energy_aero_j = 0.5 * rho_air * cd * frontal_area * (speed_ms ** 2) * dist_m
    energy_aero_kwh = (energy_aero_j / 3.6e6) / motor_efficiency

    # 3. Gravitational Potential Energy (Uphill vs Downhill)
    # delta_h = distance_m * (slope_percent / 100)
    elevation_delta_m = dist_m * (slope_percent / 100.0)
    energy_grav_j = total_mass_kg * g * elevation_delta_m
    energy_grav_kwh = np.where(
        energy_grav_j >= 0,
        (energy_grav_j / 3.6e6) / motor_efficiency,      # Uphill consumes extra motor power
        (energy_grav_j / 3.6e6) * regen_efficiency       # Downhill recovers energy via regenerative braking
    )

    # 4. Traffic Congestion Factor (Stop-and-go acceleration losses)
    # 0 = Low (smooth), 1 = Medium (+12%), 2 = High (+28%)
    traffic_multipliers = np.array([1.00, 1.14, 1.30])
    traffic_factor = traffic_multipliers[traffic_level]

    # Driving energy before HVAC
    traction_energy_kwh = (energy_roll_kwh + energy_aero_kwh) * traffic_factor + energy_grav_kwh

    # 5. Auxiliary and HVAC Energy (Cabin comfort & Battery thermal management)
    # Optimal battery & human comfort zone is ~21°C - 23°C
    # Heating required when T < 18°C (PTC / Heat Pump)
    # Cooling (AC) required when T > 24°C
    base_aux_kw = 0.40  # Computers, sensors, headlamps, display screens
    hvac_heating_kw = np.maximum(0.0, 1.6 + 0.10 * (18.0 - temperature_c))
    hvac_cooling_kw = np.maximum(0.0, 1.0 + 0.08 * (temperature_c - 24.0))

    aux_power_kw = base_aux_kw + np.where(
        temperature_c < 18.0,
        hvac_heating_kw,
        np.where(temperature_c > 24.0, hvac_cooling_kw, 0.0)
    )
    aux_energy_kwh = aux_power_kw * time_hours

    # 6. Combined Total Energy with slight driver behavior noise (+-2%)
    driver_variability = rng.normal(1.0, 0.02, size=n)
    total_energy_kwh = (traction_energy_kwh + aux_energy_kwh) * driver_variability

    # Minimum energy floor (even on steep downhill with full regen, systems consume minimal baseline)
    total_energy_kwh = np.maximum(total_energy_kwh, 0.05 * distance_km)

    return np.round(total_energy_kwh, 3)


def generate_ev_energy_dataset(
    num_samples: int = 1500,
    random_seed: int = 42,
    output_path: Optional[str or Path] = None
) -> pd.DataFrame:
    """
    Generates a realistic, synthetic EV energy consumption dataset.

    Parameters:
        num_samples (int): Total number of trip records to simulate (default: 1500).
        random_seed (int): Seed for reproducibility.
        output_path (str or Path, optional): File path to save CSV.

    Returns:
        pd.DataFrame: Generated dataset containing features and target variable.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Distance in kilometers: mixture of short urban trips and regional trips (1.0 to 45.0 km)
    distance_km = np.round(rng.gamma(shape=3.5, scale=3.5, size=num_samples), 2)
    distance_km = np.clip(distance_km, 1.2, 45.0)

    # 2. Traffic Level: 0 = Low, 1 = Medium, 2 = High
    # Probabilities: 40% Low, 35% Medium, 25% High
    traffic_level_encoded = rng.choice([0, 1, 2], size=num_samples, p=[0.40, 0.35, 0.25])

    # 3. Average Speed (km/h): conditioned realistically by traffic level
    base_speed = rng.normal(loc=55.0, scale=14.0, size=num_samples)
    speed_reductions = np.array([0.0, 15.0, 28.0])
    speed_kmh = np.round(base_speed - speed_reductions[traffic_level_encoded], 1)
    speed_kmh = np.clip(speed_kmh, 15.0, 110.0)

    # 4. Road Slope (%) and Elevation Change (m)
    # Range -5.0% to +5.0% typical for urban/suburban topography
    slope_percent = np.round(rng.normal(loc=0.0, scale=1.8, size=num_samples), 2)
    slope_percent = np.clip(slope_percent, -5.5, 5.5)
    elevation_change_m = np.round((distance_km * 1000.0) * (slope_percent / 100.0), 1)

    # 5. Vehicle Load (kg): Passengers and luggage (0 to 400 kg)
    # Discrete passenger weights + cargo
    passenger_counts = rng.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.45, 0.30, 0.15, 0.07, 0.03])
    avg_person_kg = 75.0
    cargo_kg = rng.uniform(0.0, 50.0, size=num_samples)
    vehicle_load_kg = np.round((passenger_counts - 1) * avg_person_kg + cargo_kg, 1)

    # 6. Ambient Temperature (°C): -8°C to +38°C (Seasonal variations)
    temperature_c = np.round(rng.normal(loc=19.0, scale=10.0, size=num_samples), 1)
    temperature_c = np.clip(temperature_c, -8.0, 38.0)

    # 7. Target: Energy Consumption in kWh
    energy_consumed_kwh = calculate_physics_ev_energy(
        distance_km=distance_km,
        speed_kmh=speed_kmh,
        traffic_level=traffic_level_encoded,
        slope_percent=slope_percent,
        vehicle_load_kg=vehicle_load_kg,
        temperature_c=temperature_c,
        random_seed=random_seed
    )

    # Human-readable traffic label
    traffic_map = {0: "Low", 1: "Medium", 2: "High"}
    traffic_name = [traffic_map[lvl] for lvl in traffic_level_encoded]

    # Create DataFrame
    df = pd.DataFrame({
        "distance_km": distance_km,
        "speed_kmh": speed_kmh,
        "traffic_level_encoded": traffic_level_encoded,
        "traffic_level": traffic_name,
        "slope_percent": slope_percent,
        "elevation_change_m": elevation_change_m,
        "vehicle_load_kg": vehicle_load_kg,
        "temperature_c": temperature_c,
        "energy_consumed_kwh": energy_consumed_kwh
    })

    if output_path is None:
        output_path = DEFAULT_DATASET_PATH
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def inspect_dataset(csv_path: Optional[str or Path] = None) -> pd.DataFrame:
    """
    Loads, inspects, and displays statistical summaries of the EV dataset.

    Parameters:
        csv_path (str or Path, optional): Path to ev_energy_dataset.csv.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if csv_path is None:
        csv_path = DEFAULT_DATASET_PATH
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=" * 75)
    print(" EV ENERGY CONSUMPTION DATASET INSPECTION")
    print(f" Source: {csv_path}")
    print("=" * 75)

    # 1. Dataset Shape and Missing Values
    print(f"\n[1] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[2] Missing Values Check:")
    null_counts = df.isnull().sum()
    print(null_counts)
    total_missing = null_counts.sum()
    print(f"--> Total missing/null values: {total_missing} ({'Clean, 0% missing' if total_missing == 0 else 'Attention needed'})")

    # 2. Data Types
    print("\n[3] Column Data Types:")
    print(df.dtypes)

    # 3. First 5 Rows Preview
    print("\n[4] Sample Records (First 5 Rows):")
    print(df.head().to_string(index=False))

    # 4. Statistical Summary
    numeric_cols = [
        "distance_km", "speed_kmh", "traffic_level_encoded",
        "slope_percent", "elevation_change_m", "vehicle_load_kg",
        "temperature_c", "energy_consumed_kwh"
    ]
    print("\n[5] Descriptive Statistics (Numerical Features):")
    stats = df[numeric_cols].describe().round(2)
    print(stats.to_string())

    # 5. Correlation with Target (energy_consumed_kwh)
    print("\n[6] Feature Correlations with Target (energy_consumed_kwh):")
    correlations = df[numeric_cols].corr()["energy_consumed_kwh"].sort_values(ascending=False).round(3)
    for col, corr in correlations.items():
        if col != "energy_consumed_kwh":
            print(f"  - {col:<22}: {corr:+.3f}")

    print("\n" + "=" * 75)
    return df


if __name__ == "__main__":
    print("Generating simulated EV energy dataset...")
    generated_df = generate_ev_energy_dataset(num_samples=1500, random_seed=42)
    print(f"Successfully generated {len(generated_df)} records.")
    inspect_dataset()
