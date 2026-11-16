# 🚗 Route Optimization System
### AI-Powered Smart Navigation for Electric Vehicles (EVs)

<p align="center">
  <strong>An Intelligent EV Route Optimization System using ANN, CNN, Graph Algorithms and Battery Feasibility Analysis.</strong>
</p>

---

## 📌 Project Overview

The **Route Optimization System (Smart Navigation)** is an Artificial Intelligence and Machine Learning project designed to identify efficient routes for Electric Vehicles (EVs).

Unlike traditional navigation systems that primarily focus on distance or travel time, this project considers multiple EV-specific factors, including:

- Road distance
- Estimated travel time
- Traffic conditions
- Road elevation and slope
- Predicted battery energy consumption
- Current battery percentage
- Minimum battery reserve

The system uses an Artificial Neural Network (ANN) to predict EV energy consumption and graph search algorithms to calculate optimized routes.

An optional Convolutional Neural Network (CNN) module is also included for image-based traffic classification.

The application is developed using Python and Streamlit, with interactive map visualization through Folium.

## 🎯 Project Objectives

1. Develop an intelligent route optimization system for Electric Vehicles.
2. Predict energy consumption using an Artificial Neural Network.
3. Represent road networks using graph data structures.
4. Implement Dijkstra's and A* search algorithms.
5. Compare shortest, fastest, energy-efficient and balanced routes.
6. Calculate battery consumption and remaining State of Charge (SoC).
7. Evaluate battery feasibility before starting a journey.
8. Explore CNN-based traffic image classification.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🗺️ Road Network | Graph-based road representation using NetworkX |
| 🧠 ANN Prediction | Predicts EV energy consumption in kWh |
| 👁️ CNN Traffic Detection | Classifies traffic images into Low, Medium and High |
| 📍 Shortest Route | Finds the minimum-distance route |
| ⚡ Fastest Route | Finds the minimum-time route |
| 🔋 Energy-Efficient Route | Minimizes predicted energy consumption |
| ⚖️ Balanced Route | Combines distance, time and energy costs |
| 🔌 Battery Management | Calculates available and remaining battery energy |
| 🛡️ Battery Feasibility | Checks battery reserve and insufficient-energy conditions |
| 📊 Route Comparison | Compares route distance, time and energy |
| 🌐 Interactive Dashboard | Displays results through Streamlit and Folium |

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.13 | Main programming language |
| NumPy | Numerical calculations |
| Pandas | Dataset processing |
| Scikit-learn | ANN model and preprocessing |
| TensorFlow / Keras | Optional CNN implementation |
| NetworkX | Graph and road network modeling |
| Streamlit | Web application interface |
| Folium | Interactive map visualization |
| Matplotlib | Data visualization |
| Joblib | Model and scaler serialization |
| Pytest | Automated testing |

## 🧠 Artificial Intelligence Components

### 1. Artificial Neural Network (ANN)

The project uses Scikit-learn's `MLPRegressor` to predict EV energy consumption.

**Input features:**
- Distance (km)
- Average speed (km/h)
- Traffic level
- Slope percentage
- Elevation change (m)
- Vehicle load (kg)
- Temperature (°C)

**Output:**

Predicted energy consumption in kilowatt-hours (kWh).

**Model architecture:**

- Input layer: 7 features
- Hidden layer 1: 64 neurons
- Hidden layer 2: 32 neurons
- Output layer: 1 neuron
- Activation: ReLU in hidden layers

The dataset uses simulated, physics-inspired EV trip records.

### ANN Evaluation Results

The existing model documentation reports the following test-set metrics:

| Evaluation Metric | Result |
|---|---:|
| MAE | 0.1338 kWh |
| MSE | 0.0374 kWh² |
| RMSE | 0.1933 kWh |
| R² Score | 0.9859 |

**Note:** These results are based on the project's simulated dataset and should not be interpreted as verified real-world EV prediction accuracy.

### 2. Convolutional Neural Network (CNN)

The project includes an optional CNN-based traffic classification module.

Its purpose is to process road images and classify traffic into:

- Low Traffic
- Medium Traffic
- High Traffic

The predicted traffic category can be converted into a traffic weight for road-network updates.

## 🗺️ Route Optimization Algorithms

The project implements two graph search algorithms.

### Dijkstra's Algorithm

Dijkstra finds the minimum-cost path between a source and destination when edge weights are non-negative.

### A* Search Algorithm

A* uses the following evaluation function:

`f(n) = g(n) + h(n)`

Where:

- `g(n)` = Actual cost from the starting node.
- `h(n)` = Estimated remaining cost to the destination.
- `f(n)` = Estimated total cost.

### Four Optimization Modes

| Mode | Optimization Goal |
|---|---|
| Shortest | Minimum road distance |
| Fastest | Minimum estimated travel time |
| Energy Efficient | Minimum predicted energy |
| Balanced | Weighted combination of distance, time and energy |

## 🔋 EV Battery Management

The system evaluates battery feasibility before completing a journey.

**Available battery energy:**

`Current Energy = Battery Capacity × (Current SoC / 100)`

**Reserve energy:**

`Reserve Energy = Battery Capacity × (Reserve SoC / 100)`

**Usable energy:**

`Usable Energy = Current Energy - Reserve Energy`

### Battery Feasibility Status

| Status | Meaning |
|---|---|
| SAFE | Sufficient energy while maintaining the reserve |
| RESERVE BREACH | Vehicle can reach the destination but falls below the reserve |
| INFEASIBLE | Available battery energy is insufficient |

## 🏗️ System Architecture

The system follows this workflow:

1. User enters source and destination nodes.
2. Road network is loaded as a graph.
3. Road and driving features are extracted.
4. ANN predicts energy consumption.
5. Dijkstra or A* calculates the optimal route.
6. Battery feasibility is evaluated.
7. Results are displayed in the Streamlit dashboard.
8. Folium visualizes the selected route.

## 📂 Project Structure

```text
PROJECT TESTING/
│
├── app.py
├── train_model.py
├── run_simulation.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── ev_energy_dataset.csv
│   └── road_network.json
│
├── models/
│   ├── ann_energy_model.joblib
│   └── scaler.joblib
│
├── src/
│   ├── road_network.py
│   ├── route_optimizer.py
│   ├── energy_model.py
│   ├── ev_calculations.py
│   ├── map_visualization.py
│   ├── traffic_cnn.py
│   └── generate_dataset.py
│
├── tests/
│   ├── test_ann_model.py
│   ├── test_dataset.py
│   ├── test_ev_calculations.py
│   ├── test_map_visualization.py
│   ├── test_road_network.py
│   ├── test_route_optimizer.py
│   ├── test_system_integration.py
│   └── test_traffic_cnn.py
│
├── outputs/
└── docs/
```

*This is a representative structure; actual filenames and folders may vary.*

## ⚙️ Installation and Setup

### Prerequisites

- Windows 10/11
- Python 3.10 or higher
- Visual Studio Code
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sunandamandal/Route-Optimization-System.git
```

### Step 2: Navigate to the Project

```bash
cd Route-Optimization-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser:

`http://localhost:8501`

### Run with One Command on Your Windows Computer

```powershell
& "D:\UNIVERSITY RESOURCE\AI\PROJECT TESTING\.venv\Scripts\python.exe" -m streamlit run "D:\UNIVERSITY RESOURCE\AI\PROJECT TESTING\app.py"
```

## 🧪 Model Training and Testing

### Train the ANN Model

```bash
python train_model.py
```

### Run Automated Tests

```bash
python -m pytest tests/ -v
```

### Run Command-Line Simulation

```bash
python run_simulation.py
```

### Previous Test Results

The project previously completed:

- Total tests: 55
- Passed: 55
- Failed: 0

Execution time: 8.45 seconds.

These are results from the previous test execution; run the tests again after making changes.

## 📊 Sample Demonstration

The following is an example from the simulated VoltCity road network.

| Route Mode | Distance | Time | Energy |
|---|---:|---:|---:|
| Shortest | 10.80 km | 19.4 min | 1.320 kWh |
| Fastest | 11.20 km | 17.9 min | 1.299 kWh |
| Energy Efficient | 11.20 km | 17.9 min | 1.299 kWh |
| Balanced | 11.20 km | 17.9 min | 1.299 kWh |

These are illustrative project outputs, not live Indian road-navigation results.

## ⚠️ Current Limitations

1. The existing documented road network is a simulated city network called VoltCity.
2. Traffic profiles are static or image-simulated rather than live traffic feeds.
3. ANN predictions are trained and evaluated on simulated data.
4. CNN performance depends on the availability and quality of trained image data.
5. Real-world battery behavior may differ due to weather, vehicle condition and driving style.
6. Real-time India-wide road navigation requires integration with external mapping and routing services.

## 🚀 Future Enhancements

- India-wide road navigation using OpenStreetMap.
- Location search using geocoding services.
- Real road routing through OSRM or another routing provider.
- Real-time traffic integration.
- Improved energy predictions using real EV driving data.
- EV charging station integration.
- Weather-based energy estimation.
- Mobile-friendly navigation interface.
- Real-time vehicle battery monitoring.

## 🎓 Academic Project Information

**Project Title:** Route Optimization System (Smart Navigation for Electric Vehicles)

**Subject:** Artificial Intelligence

**Project Domain:** Artificial Intelligence and Machine Learning

**Primary Concepts:**
- Artificial Neural Networks
- Convolutional Neural Networks
- Graph Theory
- Dijkstra's Algorithm
- A* Search
- EV Energy Prediction
- Battery Feasibility Analysis

## 👨‍💻 Author

**Sunanda Mandal**

B.Tech – Computer Science and Engineering (Artificial Intelligence and Machine Learning)

GitHub: [@sunandamandal](https://github.com/sunandamandal)


## 📜 License

This project is developed for academic and educational purposes.

A formal open-source license can be added if the project is intended for public reuse.

## 🚀 Live Demo

The Route Optimization System is deployed on Render and can be accessed here:

🔗 **[Open Live Application](https://route-optimization-system-db1i.onrender.com)**

Explore the application to test EV route optimization and view the available features.
---

<p align="center">
  <strong>Built with Python, Artificial Intelligence and Machine Learning ❤️</strong>
</p>

<p align="center">
  <em>Smarter Routes. Better Energy Efficiency. Sustainable Mobility.</em>
</p>

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/cba78cd7-7a07-4a94-b23e-f8d540b2f5d3" />
