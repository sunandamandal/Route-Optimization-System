"""
train_model.py
Artificial Neural Network (ANN) Training and Evaluation Pipeline for EV Energy Prediction.
Uses Scikit-learn's MLPRegressor, StandardScaler, and Joblib.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Relative paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "ev_energy_dataset.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "ann_energy_model.joblib"
SCALER_SAVE_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
METRICS_SAVE_PATH = PROJECT_ROOT / "outputs" / "ann_evaluation_metrics.json"
PLOT_SAVE_PATH = PROJECT_ROOT / "outputs" / "ann_actual_vs_predicted.png"

FEATURE_NAMES = [
    "distance_km",
    "speed_kmh",
    "traffic_level_encoded",
    "slope_percent",
    "elevation_change_m",
    "vehicle_load_kg",
    "temperature_c"
]
TARGET_NAME = "energy_consumed_kwh"


def load_and_preprocess_data(
    dataset_path: Path,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Loads dataset, separates features & target, applies train/test split,
    and scales features strictly on the training set to prevent data leakage.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} records from '{dataset_path.name}'.")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"Train set: {len(X_train)} samples ({(1 - test_size) * 100:.0f}%) | "
          f"Test set: {len(X_test)} samples ({test_size * 100:.0f}%)")

    # Feature Scaling (Fit on X_train ONLY to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_ann_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42
) -> MLPRegressor:
    """
    Builds and trains an Artificial Neural Network using Scikit-Learn's MLPRegressor.

    Architecture:
      - Input Layer   : 7 features
      - Hidden Layer 1: 64 neurons (ReLU activation)
      - Hidden Layer 2: 32 neurons (ReLU activation)
      - Output Layer  : 1 neuron (Linear activation for regression)
      - Optimizer     : Adam (adaptive moment estimation)
    """
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.003,
        alpha=0.0001,          # L2 regularization to prevent overfitting
        batch_size=32,
        max_iter=600,
        random_state=random_state,
        verbose=False
    )

    print("\nTraining Artificial Neural Network (MLPRegressor)...")
    ann.fit(X_train_scaled, y_train)
    print(f"Convergence achieved after {ann.n_iter_} iterations.")
    return ann


def evaluate_model(
    model: MLPRegressor,
    X_test_scaled: np.ndarray,
    y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates the model against the unseen test set using MAE, MSE, RMSE, and R2.
    """
    y_pred = model.predict(X_test_scaled)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "Mean_Absolute_Error_MAE_kWh": round(mae, 4),
        "Mean_Squared_Error_MSE_kWh2": round(mse, 4),
        "Root_Mean_Squared_Error_RMSE_kWh": round(rmse, 4),
        "R_Squared_Score_R2": round(r2, 4),
        "Test_Samples_Count": len(y_test)
    }

    print("\n" + "=" * 55)
    print(" ANN MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 55)
    print(f"  Mean Absolute Error (MAE) : {metrics['Mean_Absolute_Error_MAE_kWh']:.4f} kWh")
    print(f"  Mean Squared Error (MSE)  : {metrics['Mean_Squared_Error_MSE_kWh2']:.4f} kWh²")
    print(f"  Root Mean Sq. Error (RMSE): {metrics['Root_Mean_Squared_Error_RMSE_kWh']:.4f} kWh")
    print(f"  R-Squared Score (R²)      : {metrics['R_Squared_Score_R2']:.4f} ({metrics['R_Squared_Score_R2']*100:.2f}% accuracy)")
    print("=" * 55)

    return metrics, y_pred


def plot_and_save_results(
    y_test: pd.Series,
    y_pred: np.ndarray,
    loss_curve: list,
    metrics: Dict[str, float],
    save_path: Path
):
    """
    Plots Actual vs Predicted Energy Consumption, prediction error residuals,
    and the training loss curve.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")

    # 1. Actual vs Predicted Scatter Plot
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#2980b9", edgecolors="#1b4f72", s=45, label="Test Samples")

    # Ideal prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2.2, label="Ideal Line (y = x)")

    ax1.set_title(f"Actual vs Predicted Energy (R² = {metrics['R_Squared_Score_R2']:.4f})", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Actual Energy Consumed (kWh)", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("Predicted Energy (kWh)", fontsize=10, fontweight="semibold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(color="#e0e0e0", linestyle="--", linewidth=0.6)

    # 2. Residual Distribution (Errors = Actual - Predicted)
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=25, color="#27ae60", edgecolor="#145a32", alpha=0.75, density=True)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2.0, label="Zero Error")

    ax2.set_title(f"Residual Error Distribution (MAE = {metrics['Mean_Absolute_Error_MAE_kWh']:.3f} kWh)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Error (Actual - Predicted) [kWh]", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Density", fontsize=10, fontweight="semibold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(color="#e0e0e0", linestyle="--", linewidth=0.6)

    # 3. Training Loss Curve (Adam Optimizer convergence)
    ax3 = axes[2]
    ax3.set_facecolor("#ffffff")
    ax3.plot(loss_curve, color="#8e44ad", linewidth=2.0, label="Training Loss")
    ax3.set_title("ANN Training Loss Curve (Adam)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Iterations (Epochs)", fontsize=10, fontweight="semibold")
    ax3.set_ylabel("Loss (MSE)", fontsize=10, fontweight="semibold")
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(color="#e0e0e0", linestyle="--", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation plot to: '{save_path.name}'.")


def save_artifacts(
    model: MLPRegressor,
    scaler: StandardScaler,
    metrics: Dict[str, float]
):
    """Saves model and scaler with joblib, and metrics as JSON."""
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saved trained ANN model to : '{MODEL_SAVE_PATH.name}'")

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Saved StandardScaler to   : '{SCALER_SAVE_PATH.name}'")

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics summary to  : '{METRICS_SAVE_PATH.name}'")


def main():
    print("=" * 65)
    print(" EV ROUTE OPTIMIZATION - ARTIFICIAL NEURAL NETWORK (ANN)")
    print(" Energy Consumption Prediction Pipeline")
    print("=" * 65)

    # 1. Load data and split with zero data leakage
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(
        DATASET_PATH, test_size=0.2, random_state=42
    )

    # 2. Train ANN (MLPRegressor)
    model = train_ann_model(X_train_scaled, y_train, random_state=42)

    # 3. Evaluate Model
    metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)

    # 4. Save Artifacts (model, scaler, metrics)
    save_artifacts(model, scaler, metrics)

    # 5. Generate and save publication-quality visualization
    plot_and_save_results(y_test, y_pred, model.loss_curve_, metrics, PLOT_SAVE_PATH)

    # 6. Demonstration of Inference Function
    print("\n" + "=" * 65)
    print(" PREDICTION DEMONSTRATION ON TEST SCENARIOS")
    print("=" * 65)
    from src.energy_model import predict_energy_consumption

    test_cases = [
        {"name": "Flat City Commute (Low Traffic)", "dist": 5.0, "speed": 45.0, "traffic": 0, "slope": 0.0, "elev": 0.0, "load": 75.0, "temp": 21.0},
        {"name": "Steep Uphill Climb (+3.5%)", "dist": 5.0, "speed": 45.0, "traffic": 0, "slope": 3.5, "elev": 175.0, "load": 75.0, "temp": 21.0},
        {"name": "Steep Downhill Descent (-3.5%)", "dist": 5.0, "speed": 45.0, "traffic": 0, "slope": -3.5, "elev": -175.0, "load": 75.0, "temp": 21.0},
        {"name": "Heavy Congestion & Freezing Temp (-5°C)", "dist": 5.0, "speed": 22.0, "traffic": 2, "slope": 0.0, "elev": 0.0, "load": 150.0, "temp": -5.0},
    ]

    for tc in test_cases:
        pred_kwh = predict_energy_consumption(
            distance_km=tc["dist"],
            speed_kmh=tc["speed"],
            traffic_level_encoded=tc["traffic"],
            slope_percent=tc["slope"],
            elevation_change_m=tc["elev"],
            vehicle_load_kg=tc["load"],
            temperature_c=tc["temp"],
            model_path=MODEL_SAVE_PATH,
            scaler_path=SCALER_SAVE_PATH
        )
        print(f" Scenario: {tc['name']:<42} -> Predicted Energy: {pred_kwh:.3f} kWh")

    print("\nANN training and evaluation pipeline executed successfully.")


if __name__ == "__main__":
    main()
