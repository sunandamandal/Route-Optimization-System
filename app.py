"""
app.py
Route Optimization System (Smart Navigation for Electric Vehicles)
Professional Streamlit Web Application Dashboard

Features:
  - Sidebar inputs for Source/Destination, Battery State, Optimization Mode, and Balanced Weights
  - Interactive City Road Network Folium Map with route highlighting and charging stations
  - Real-time battery feasibility status, warnings, and en-route charging requirements
  - Comparison table and comparative charts (Energy, Travel Time, Distance) across 4 routing modes
  - Highlighting of energy savings and technical project methodology for college project demonstrations
"""

import sys
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional, Union
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to sys.path so modules import reliably
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.road_network import load_road_network
from src.route_optimizer import EVRouteOptimizer, RouteResult
from src.ev_calculations import (
    EVBatteryState,
    FeasibilityStatus,
    calculate_available_battery_energy,
    compare_routes_battery,
    evaluate_route_battery
)
from src.map_visualization import create_city_folium_map

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Smart Navigation | VoltCity",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a252f;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #7f8c8d;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
    }
    .status-badge-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_optimizer() -> Tuple[EVRouteOptimizer, Any]:
    """Loads and caches the NetworkX road graph and ANN-powered optimizer."""
    graph = load_road_network()
    optimizer = EVRouteOptimizer(graph=graph)
    return optimizer, graph


optimizer, city_graph = get_cached_optimizer()

# Prepare readable node choices: "A: Downtown Central", etc.
node_options = [
    f"{nid}: {city_graph.nodes[nid]['name']}" + (" ⚡" if city_graph.nodes[nid].get("has_charging_station") else "")
    for nid in sorted(city_graph.nodes())
]
node_id_map = {opt: opt.split(":")[0] for opt in node_options}


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electric-car.png", width=64)
    st.title("Navigation Controls")

    st.subheader("📍 Journey Plan")
    source_sel = st.selectbox("Source Location (Start)", options=node_options, index=6)  # Default: G (Harbor Bay)
    dest_sel = st.selectbox("Destination Location (Goal)", options=node_options, index=11)  # Default: L (Uptown Plaza)

    source_node = node_id_map[source_sel]
    dest_node = node_id_map[dest_sel]

    st.divider()
    st.subheader("🔋 Battery Specifications")
    battery_cap = st.number_input("Pack Capacity (kWh)", min_value=15.0, max_value=150.0, value=60.0, step=5.0)
    current_soc = st.slider("Current Battery Level (%)", min_value=1.0, max_value=100.0, value=45.0, step=1.0)
    min_reserve = st.slider("Minimum Reserve Level (%)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

    st.divider()
    st.subheader("⚙️ Optimization Strategy")
    mode_display_map = {
        "⚡ Energy Efficient": "energy",
        "⏱️ Fastest Route": "fastest",
        "📏 Shortest Distance": "shortest",
        "⚖️ Balanced (Multi-Objective)": "balanced"
    }
    selected_mode_label = st.radio(
        "Routing Objective Mode",
        options=list(mode_display_map.keys()),
        index=0
    )
    selected_mode = mode_display_map[selected_mode_label]

    # Balanced Mode Weights
    balanced_weights = {"energy": 0.50, "time": 0.30, "distance": 0.20}
    if selected_mode == "balanced":
        with st.expander("⚖️ Tune Balanced Mode Weights", expanded=True):
            w_e = st.slider("Energy Weight (kWh)", 0.0, 1.0, 0.50, 0.05)
            w_t = st.slider("Travel Time Weight", 0.0, 1.0, 0.30, 0.05)
            w_d = st.slider("Distance Weight", 0.0, 1.0, 0.20, 0.05)
            total_w = max(0.001, w_e + w_t + w_d)
            balanced_weights = {
                "energy": w_e / total_w,
                "time": w_t / total_w,
                "distance": w_d / total_w
            }
            st.caption(f"Normalized weights: Energy={balanced_weights['energy']:.2f}, Time={balanced_weights['time']:.2f}, Dist={balanced_weights['distance']:.2f}")

    algorithm_sel = st.selectbox("Search Algorithm", options=["Dijkstra's Algorithm", "A* Search Algorithm"], index=1)
    algo_key = "astar" if "A*" in algorithm_sel else "dijkstra"

    with st.expander("🌡️ Vehicle & Environment Settings"):
        load_kg = st.slider("Added Passenger/Cargo Load (kg)", 0.0, 350.0, 75.0, 25.0)
        temp_c = st.slider("Ambient Temperature (°C)", -10.0, 40.0, 20.0, 1.0)

    # Sync optimizer state if environmental variables change
    optimizer.vehicle_load_kg = load_kg
    optimizer.temperature_c = temp_c


# -----------------------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Smart Navigation & EV Route Optimization System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Topographical & Traffic-Aware Routing with Artificial Neural Network (ANN) Energy Modeling</div>',
    unsafe_allow_html=True
)

# Handle Identical Source & Destination Error
if source_node == dest_node:
    st.warning("⚠️ Source and Destination are identical! Please select two distinct intersections to calculate a route.")
    st.stop()

# -----------------------------------------------------------------------------
# ROUTE COMPUTATION & EVALUATION
# -----------------------------------------------------------------------------
try:
    # 1. Compute Candidate Routes across all 4 modes
    candidate_routes = optimizer.compare_all_modes(source_node, dest_node, algorithm=algo_key)

    # 2. Build EV Battery State and evaluate all candidate routes
    battery_state = EVBatteryState(
        capacity_kwh=battery_cap,
        current_soc_percent=current_soc,
        min_reserve_percent=min_reserve
    )
    comparison_summary = compare_routes_battery(candidate_routes, battery_state)

    # 3. Active Selected Route Details
    active_route: RouteResult = candidate_routes[selected_mode]
    active_eval = comparison_summary.evaluations[selected_mode]

except Exception as err:
    st.error(f"Routing Error: Unable to compute path. Details: {err}")
    st.stop()


# -----------------------------------------------------------------------------
# TOP METRICS ROW
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.metric(
        label="Total Distance",
        value=f"{active_route.total_distance_km:.2f} km",
        help="Physical road distance along the selected route."
    )

with m_col2:
    st.metric(
        label="Est. Travel Time",
        value=f"{active_route.total_time_min:.1f} min",
        help="Estimated transit time considering road speed limits and traffic congestion."
    )

with m_col3:
    st.metric(
        label="Predicted Energy",
        value=f"{active_route.total_energy_kwh:.3f} kWh",
        delta=f"-{comparison_summary.savings_vs_shortest_kwh:.3f} kWh vs Shortest" if selected_mode == "energy" and comparison_summary.savings_vs_shortest_kwh > 0 else None,
        delta_color="normal",
        help="Battery energy consumption predicted by the trained Artificial Neural Network."
    )

with m_col4:
    st.metric(
        label="Battery Remaining",
        value=f"{active_eval.remaining_soc_percent:.1f}%",
        delta=f"{active_eval.remaining_energy_kwh:.2f} kWh",
        delta_color="off" if active_eval.is_safe else "inverse",
        help="Projected State of Charge and remaining energy in kWh upon reaching destination."
    )

with m_col5:
    st.markdown("<p style='font-size:0.875rem; font-weight:600; color:#555; margin-bottom:4px;'>FEASIBILITY STATUS</p>", unsafe_allow_html=True)
    if active_eval.status == FeasibilityStatus.SAFE:
        st.markdown('<span class="status-badge-safe">✅ SAFE TO TRAVEL</span>', unsafe_allow_html=True)
    elif active_eval.status == FeasibilityStatus.RESERVE_BREACHED:
        st.markdown('<span class="status-badge-warning">⚠️ RESERVE BREACH</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-danger">⛔ INFEASIBLE</span>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DYNAMIC WARNING & CHARGING ADVISORY BANNER
# -----------------------------------------------------------------------------
if active_eval.status == FeasibilityStatus.INFEASIBLE:
    st.error(f"""
    🚨 **CRITICAL BATTERY WARNING: The vehicle CANNOT reach the destination!**
    - The planned route requires **{active_eval.energy_consumed_kwh:.2f} kWh**, but the battery only contains **{battery_state.current_energy_kwh:.2f} kWh**.
    - The vehicle will become stranded approximately **{active_eval.critical_deficit_kwh:.2f} kWh** short of the destination.
    - **Required Action**: Charge at least **{active_eval.charging_needed_kwh:.2f} kWh** (~{active_eval.est_charge_time_50kw_min:.0f} min at 50 kW or ~{active_eval.est_charge_time_150kw_min:.0f} min at 150 kW) before proceeding.
    """)
elif active_eval.status == FeasibilityStatus.RESERVE_BREACHED:
    st.warning(f"""
    ⚠️ **RESERVE PENETRATION ADVISORY**:
    - The vehicle can physically reach the destination, but will penetrate the **{battery_state.min_reserve_percent:.0f}%** safety reserve buffer.
    - Arrival battery level will be **{active_eval.remaining_soc_percent:.1f}%** ({active_eval.remaining_energy_kwh:.2f} kWh remaining).
    - Recommended to top up **{active_eval.charging_needed_kwh:.2f} kWh** at an en-route fast charger.
    """)
else:
    st.success(
        f"✅ **Route is fully verified and safe.** The vehicle will reach the destination with **{active_eval.remaining_soc_percent:.1f}% SoC** ({active_eval.remaining_energy_kwh:.2f} kWh), comfortably above the {battery_state.min_reserve_percent:.0f}% reserve threshold."
    )


# -----------------------------------------------------------------------------
# INTERACTIVE FOLIUM MAP & ROUTE BREADCRUMBS
# -----------------------------------------------------------------------------
col_map, col_details = st.columns([1.75, 1.0])

with col_map:
    st.subheader("🗺️ Interactive City Road Network & Navigation Map")
    folium_map = create_city_folium_map(
        graph=city_graph,
        selected_path=active_route.path,
        source_node=source_node,
        destination_node=dest_node,
        mode_name=selected_mode
    )
    # Render Folium map cleanly via HTML component
    st.components.v1.html(folium_map._repr_html_(), height=500, scrolling=False)
    st.caption("🟢 Green Pin: Start | 🔴 Red Flag: Destination | ⚡ Green Bolt: DC Fast Charging Station | Colored Lines: Traffic-Aware City Roads")

with col_details:
    st.subheader("🚗 Turn-by-Turn Navigation")
    st.markdown(f"**Selected Mode:** `{selected_mode.upper()}` | **Algorithm:** `{algo_key.upper()}`")
    st.markdown(f"**Path Sequence:** `{' ➔ '.join(active_route.path)}`")

    # Breadcrumb segments breakdown
    with st.container(height=380):
        for i, seg in enumerate(active_route.segments, 1):
            src_lbl = city_graph.nodes[seg.source]["name"]
            tgt_lbl = city_graph.nodes[seg.target]["name"]
            has_cs = city_graph.nodes[seg.target].get("has_charging_station", False)
            cs_badge = " ⚡ [EV Charger Available]" if has_cs else ""

            st.markdown(f"""
            **{i}. {seg.source} &rarr; {seg.target}**: {src_lbl} to {tgt_lbl}{cs_badge}
            - 📏 Dist: `{seg.distance_km} km` | ⏱️ Time: `{seg.travel_time_min} min` | ⚡ Energy: `{seg.energy_kwh:.3f} kWh`
            - 🚦 Traffic: `{seg.traffic_level.upper()}` | ⛰️ Slope: `{seg.slope_percent:+.2f}%`
            ---
            """)

    # Intermediate Charging Hubs
    if active_route.charging_stops:
        st.info(f"⚡ **Charging Stations Along Route**: {', '.join(active_route.charging_stops)}")


# -----------------------------------------------------------------------------
# ROUTE COMPARISON SECTION
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📊 Candidate Route Comparison Table")

table_rows = []
for mode_key, r in candidate_routes.items():
    ev_eval = comparison_summary.evaluations[mode_key]
    status_str = "✅ Safe" if ev_eval.is_safe else ("⚠️ Caution" if ev_eval.is_feasible else "⛔ Infeasible")

    table_rows.append({
        "Mode": mode_key.capitalize(),
        "Algorithm": r.algorithm.upper(),
        "Distance (km)": r.total_distance_km,
        "Travel Time (min)": r.total_time_min,
        "Energy Consumed (kWh)": r.total_energy_kwh,
        "Remaining Battery (%)": f"{ev_eval.remaining_soc_percent:.1f}%",
        "Remaining Energy (kWh)": f"{ev_eval.remaining_energy_kwh:.2f} kWh",
        "Charging Needed (kWh)": f"{ev_eval.charging_needed_kwh:.2f} kWh" if ev_eval.charging_needed_kwh > 0 else "0.0 kWh",
        "Feasibility": status_str,
        "Route Path": " ➔ ".join(r.path)
    })

comparison_df = pd.DataFrame(table_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Highlight Energy Savings
if comparison_summary.savings_vs_shortest_kwh > 0 or comparison_summary.savings_vs_fastest_kwh > 0:
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.info(f"💡 **Energy Savings vs. Shortest Route**: **+{comparison_summary.savings_vs_shortest_kwh:.3f} kWh** ({comparison_summary.savings_vs_shortest_percent:.2f}% reduction)")
    with s_col2:
        st.info(f"💡 **Energy Savings vs. Fastest Route**: **+{comparison_summary.savings_vs_fastest_kwh:.3f} kWh** ({comparison_summary.savings_vs_fastest_percent:.2f}% reduction)")


# -----------------------------------------------------------------------------
# COMPARATIVE CHARTS
# -----------------------------------------------------------------------------
st.subheader("📈 Multi-Mode Performance Metrics")
c_col1, c_col2, c_col3 = st.columns(3)

# Data for charts
chart_data = pd.DataFrame({
    "Mode": [r.mode.capitalize() for r in candidate_routes.values()],
    "Energy (kWh)": [r.total_energy_kwh for r in candidate_routes.values()],
    "Time (min)": [r.total_time_min for r in candidate_routes.values()],
    "Distance (km)": [r.total_distance_km for r in candidate_routes.values()],
}).set_index("Mode")

with c_col1:
    st.markdown("**🔋 Predicted Energy Consumption (kWh)**")
    st.bar_chart(chart_data["Energy (kWh)"], color="#27ae60")

with c_col2:
    st.markdown("**⏱️ Estimated Travel Time (min)**")
    st.bar_chart(chart_data["Time (min)"], color="#e67e22")

with c_col3:
    st.markdown("**📏 Total Physical Distance (km)**")
    st.bar_chart(chart_data["Distance (km)"], color="#2980b9")


# -----------------------------------------------------------------------------
# TECHNICAL DETAILS & PROJECT METHODOLOGY
# -----------------------------------------------------------------------------
with st.expander("🎓 AI & System Architecture Details (For College Project Viva/Demo)"):
    st.markdown(r"""
    ### System Architecture & Working Principles
    1. **City Road Graph Modeling**:
       - Modeled with **NetworkX** containing 16 intersections and 30 bidirectional road segments.
       - Each edge stores physical distance, speed limit, dynamic traffic congestion level, and directional topographical slope ($% = \Delta h / \text{dist}$).
    2. **Artificial Neural Network (ANN) Energy Predictor**:
       - Architecture: Multi-Layer Perceptron (`MLPRegressor`) trained on 1,500 physics-grounded simulated road trips.
       - Input features: `distance_km`, `speed_kmh`, `traffic_level_encoded`, `slope_percent`, `elevation_change_m`, `vehicle_load_kg`, `temperature_c`.
       - Performance: **$R^2 = 98.59\%$**, Mean Absolute Error = **$0.1338\text{ kWh}$**.
    3. **Route Optimization Engines**:
       - **Dijkstra's Algorithm**: Min-heap implementation finding guaranteed global optimum across non-negative cost graphs.
       - **A* Search Algorithm**: Heuristic exploration using admissible straight-line Euclidean and maximum-velocity heuristics ($f(n) = g(n) + h(n)$), pruning redundant branches.
    4. **Battery Feasibility Constraints**:
       - Computes usable vs. reserved chemical energy ($E_{\text{usable}} = \max(0, E_{\text{current}} - E_{\text{reserve}})$).
       - Strictly flags trips as `INFEASIBLE` if battery energy is exhausted before reaching the destination.
    """)

st.caption("Route Optimization System (Smart Navigation for Electric Vehicles) • University AI Project")
