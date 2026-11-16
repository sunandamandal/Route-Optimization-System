"""
app.py
Route Optimization System (Smart Navigation for Electric Vehicles)
Professional Streamlit Web Application Dashboard with India-Wide Real Road Routing.

Features:
  - India-Wide Location Search: Enter any Indian city, town, locality, or address
  - OpenStreetMap Nominatim Geocoding with territorial validation, rate limiting & caching
  - Real Road Navigation using OpenStreetMap OSRM driving engine & inter-state highway corridors
  - Turn-by-Turn Navigation instructions & maneuver directions
  - Artificial Neural Network (ANN) Energy Model predicting kWh for all real road segments
  - Explicit distinction between OSRM Fastest Route and ANN Energy-Optimal Route
  - All 4 Route Optimization Modes: Shortest, Fastest, Energy-Efficient, and Balanced
  - Search Algorithms: Dijkstra's Algorithm and A* Search Algorithm
  - EV Battery State of Charge (SoC), Usable Energy, Reserve Buffer, and Feasibility Warnings
  - Interactive Folium Map with real road polyline, start/destination pins, highway charging plazas
  - Alternative route overlays as dashed secondary polylines
  - Comparison table, energy savings callouts, and multi-mode performance charts
  - Full dark professional dashboard theme
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
from src.map_visualization import create_city_folium_map, create_india_route_map
from src.india_geocoding import get_india_geocoder, GeocodedLocation, OSM_ATTRIBUTION
from src.india_road_service import get_india_route_service, TurnInstruction


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Smart Navigation | India-Wide",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DARK THEME CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }

    /* ── Main header ── */
    .ev-main-header {
        font-size: 2.0rem;
        font-weight: 800;
        color: #e6edf3;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .ev-sub-header {
        font-size: 0.95rem;
        color: #8b949e;
        margin-top: 4px;
        margin-bottom: 1.2rem;
    }

    /* ── Info banner (OSRM notice) ── */
    .ev-banner {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 16px;
        color: #c9d1d9;
        font-size: 0.875rem;
        margin-bottom: 14px;
        line-height: 1.6;
    }

    /* ── Feasibility status badges ── */
    .badge-safe {
        background: linear-gradient(135deg, #064e3b, #065f46);
        color: #34d399;
        font-weight: 700;
        padding: 7px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.88rem;
        border: 1px solid #059669;
        letter-spacing: 0.03em;
    }
    .badge-warn {
        background: linear-gradient(135deg, #451a03, #78350f);
        color: #fbbf24;
        font-weight: 700;
        padding: 7px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.88rem;
        border: 1px solid #d97706;
        letter-spacing: 0.03em;
    }
    .badge-danger {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        color: #f87171;
        font-weight: 700;
        padding: 7px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.88rem;
        border: 1px solid #dc2626;
        letter-spacing: 0.03em;
    }

    /* ── Location display cards ── */
    .loc-card-src {
        background: #0d2318;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.84rem;
        border-left: 3px solid #10b981;
        color: #d1fae5;
        line-height: 1.6;
    }
    .loc-card-dst {
        background: #200f0f;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.84rem;
        border-left: 3px solid #ef4444;
        color: #fee2e2;
        line-height: 1.6;
    }
    .loc-card-src code, .loc-card-dst code {
        font-family: "Courier New", monospace;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* ── Mode pill indicator ── */
    .mode-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .mode-energy  { background:#064e3b; color:#34d399; border:1px solid #059669; }
    .mode-fastest { background:#431407; color:#fb923c; border:1px solid #ea580c; }
    .mode-shortest{ background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb; }
    .mode-balanced{ background:#2e1065; color:#c084fc; border:1px solid #9333ea; }

    /* ── Section dividers ── */
    .ev-section-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 18px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #21262d;
    }

    /* ── Sidebar text colours ── */
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar label, .stSidebar p, .stSidebar span {
        color: #c9d1d9 !important;
    }

    /* ── Dark metric cards ── */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; }
    [data-testid="stMetricDelta"] { color: #34d399 !important; }

    /* ── Dark dataframe ── */
    [data-testid="stDataFrame"] { background: #161b22; border-radius: 8px; }

    /* ── Route comparison section ── */
    .cmp-header {
        color: #38bdf8;
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* ── EV logo container ── */
    .ev-logo-container {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 6px 0;
    }
    .ev-logo-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: -0.01em;
    }
    .ev-logo-sub {
        font-size: 0.72rem;
        color: #6e7681;
    }

    /* ── Attribution footer ── */
    .osm-attr {
        font-size: 0.72rem;
        color: #484f58;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def fmt_time(minutes: float) -> str:
    """Format minutes as 'Xh Ymin' when >= 60, else 'Y.Z min'."""
    if minutes >= 60:
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"{h}h {m:02d}min"
    return f"{minutes:.1f} min"


def mode_pill(mode: str) -> str:
    css_class = f"mode-{mode.lower()}"
    labels = {
        "energy": "⚡ ENERGY",
        "fastest": "⏱️ FASTEST",
        "shortest": "📏 SHORTEST",
        "balanced": "⚖️ BALANCED"
    }
    return f'<span class="mode-pill {css_class}">{labels.get(mode.lower(), mode.upper())}</span>'


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
geocoder = get_india_geocoder()
india_router = get_india_route_service()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # ── Branding (inline SVG — no broken external image) ──
    st.markdown("""
    <div class="ev-logo-container">
        <svg xmlns="http://www.w3.org/2000/svg" width="52" height="38" viewBox="0 0 52 38">
            <rect x="2" y="12" width="48" height="20" rx="4" fill="#1f4068" stroke="#38bdf8" stroke-width="1.5"/>
            <rect x="6" y="8" width="28" height="14" rx="3" fill="#162032"/>
            <circle cx="11" cy="34" r="4" fill="#38bdf8"/>
            <circle cx="39" cy="34" r="4" fill="#38bdf8"/>
            <circle cx="11" cy="34" r="2" fill="#0d1117"/>
            <circle cx="39" cy="34" r="2" fill="#0d1117"/>
            <polygon points="20,10 16,19 22,19 18,27 28,16 22,16 26,10" fill="#fbbf24"/>
            <rect x="46" y="20" width="4" height="6" rx="1" fill="#38bdf8"/>
        </svg>
        <div>
            <div class="ev-logo-title">EV Navigator</div>
            <div class="ev-logo-sub">India-Wide Smart Routing</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── NAVIGATION MODE TOGGLE ──
    st.markdown('<p class="ev-section-title">🗺️ Routing Mode</p>', unsafe_allow_html=True)
    routing_mode = st.radio(
        "Select Map Mode",
        options=["🌍 India-Wide (Real Roads)", "🏙️ VoltCity Simulation (Demo)"],
        index=0,
        label_visibility="collapsed"
    )
    is_voltcity_benchmark = "VoltCity" in routing_mode

    st.divider()

    # ── INDIA-WIDE LOCATION SEARCH ──
    if not is_voltcity_benchmark:
        st.markdown('<p class="ev-section-title">📍 Journey Plan</p>', unsafe_allow_html=True)

        preset_options = {
            "Custom Location": ("", ""),
            "Kolkata ➔ Durgapur (WB)": ("Kolkata, West Bengal", "Durgapur, West Bengal"),
            "Kolkata ➔ Howrah (WB)": ("Kolkata, West Bengal", "Howrah, West Bengal"),
            "Delhi ➔ Jaipur (NH48)": ("Delhi, India", "Jaipur, Rajasthan"),
            "Bengaluru ➔ Chennai (Expressway)": ("Bengaluru, Karnataka", "Chennai, Tamil Nadu"),
            "Mumbai ➔ Pune (Expressway)": ("Mumbai, Maharashtra", "Pune, Maharashtra"),
            "Hyderabad ➔ Vijayawada (NH65)": ("Hyderabad, Telangana", "Vijayawada, Andhra Pradesh"),
            "Ahmedabad ➔ Surat (NH48)": ("Ahmedabad, Gujarat", "Surat, Gujarat"),
        }

        selected_preset = st.selectbox(
            "⚡ Quick Corridor",
            options=list(preset_options.keys()),
            index=1,
            help="Pick a preset or choose 'Custom Location' to type any Indian place."
        )

        default_src, default_dst = preset_options[selected_preset]
        if selected_preset == "Custom Location":
            default_src = st.session_state.get("last_src", "")
            default_dst = st.session_state.get("last_dst", "")

        source_query = st.text_input(
            "🟢 Source (Start)",
            value=default_src,
            placeholder="e.g. Kolkata, West Bengal",
            help="Any Indian city, town, locality or street address"
        )
        dest_query = st.text_input(
            "🏁 Destination (Goal)",
            value=default_dst,
            placeholder="e.g. Durgapur, West Bengal",
            help="Any Indian city, town, locality or street address"
        )

        st.session_state["last_src"] = source_query
        st.session_state["last_dst"] = dest_query

        # ── Geocode both inputs ──
        source_geo: Optional[GeocodedLocation] = None
        dest_geo: Optional[GeocodedLocation] = None

        if source_query.strip():
            with st.spinner("📡 Locating source via OpenStreetMap…"):
                source_geo = geocoder.geocode(source_query)
        if dest_query.strip():
            with st.spinner("📡 Locating destination via OpenStreetMap…"):
                dest_geo = geocoder.geocode(dest_query)

        # ── Show resolved location cards ──
        if source_geo and source_geo.is_valid:
            st.markdown(f"""
            <div class="loc-card-src">
                <b>🟢 {source_geo.short_name}</b><br>
                <small>{source_geo.state} &nbsp;|&nbsp; <code>{source_geo.lat:.4f}°N, {source_geo.lon:.4f}°E</code></small>
            </div>
            """, unsafe_allow_html=True)
        elif source_geo and not source_geo.is_valid:
            st.error(f"❌ Source: {source_geo.error_message}")

        if dest_geo and dest_geo.is_valid:
            st.markdown(f"""
            <div class="loc-card-dst">
                <b>🏁 {dest_geo.short_name}</b><br>
                <small>{dest_geo.state} &nbsp;|&nbsp; <code>{dest_geo.lat:.4f}°N, {dest_geo.lon:.4f}°E</code></small>
            </div>
            """, unsafe_allow_html=True)
        elif dest_geo and not dest_geo.is_valid:
            st.error(f"❌ Destination: {dest_geo.error_message}")

        st.markdown(f'<p class="osm-attr">🌐 {OSM_ATTRIBUTION}</p>', unsafe_allow_html=True)

    else:
        source_geo = None
        dest_geo = None
        st.info("🏙️ Running VoltCity Simulation Grid (Bengaluru Benchmark).")

    st.divider()

    # ── BATTERY SPECIFICATIONS ──
    st.markdown('<p class="ev-section-title">🔋 Battery Specifications</p>', unsafe_allow_html=True)
    battery_cap = st.number_input(
        "Pack Capacity (kWh)",
        min_value=15.0, max_value=150.0, value=60.0, step=5.0,
        help="Total battery pack energy in kilowatt-hours."
    )
    current_soc = st.slider(
        "Current Battery Level (%)",
        min_value=1.0, max_value=100.0, value=45.0, step=1.0,
        help="State of Charge before the journey starts."
    )
    min_reserve = st.slider(
        "Minimum Reserve (%)",
        min_value=5.0, max_value=30.0, value=15.0, step=1.0,
        help="Minimum SoC buffer to keep at destination for safety."
    )

    # Live battery stats
    current_kwh = battery_cap * current_soc / 100.0
    reserve_kwh = battery_cap * min_reserve / 100.0
    usable_kwh = max(0.0, current_kwh - reserve_kwh)
    batt_col1, batt_col2 = st.columns(2)
    batt_col1.metric("Current Energy", f"{current_kwh:.1f} kWh")
    batt_col2.metric("Usable Energy", f"{usable_kwh:.1f} kWh")

    st.divider()

    # ── OPTIMIZATION STRATEGY ──
    st.markdown('<p class="ev-section-title">⚙️ Optimization Mode</p>', unsafe_allow_html=True)
    mode_display_map = {
        "⚡ Energy Efficient (ANN Optimized)": "energy",
        "⏱️ Fastest Route (Min Transit Time)": "fastest",
        "📏 Shortest Distance (Min Road km)": "shortest",
        "⚖️ Balanced (Multi-Objective)": "balanced"
    }
    selected_mode_label = st.radio(
        "Routing Objective",
        options=list(mode_display_map.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_mode = mode_display_map[selected_mode_label]

    # Balanced Mode Weights
    balanced_weights = {"energy": 0.50, "time": 0.30, "distance": 0.20}
    if selected_mode == "balanced":
        with st.expander("⚖️ Tune Balanced Weights", expanded=True):
            w_e = st.slider("Energy Weight", 0.0, 1.0, 0.50, 0.05)
            w_t = st.slider("Travel Time Weight", 0.0, 1.0, 0.30, 0.05)
            w_d = st.slider("Distance Weight", 0.0, 1.0, 0.20, 0.05)
            total_w = max(0.001, w_e + w_t + w_d)
            balanced_weights = {
                "energy": w_e / total_w,
                "time": w_t / total_w,
                "distance": w_d / total_w
            }
            st.caption(
                f"Normalized — E:{balanced_weights['energy']:.2f} "
                f"T:{balanced_weights['time']:.2f} "
                f"D:{balanced_weights['distance']:.2f}"
            )

    st.divider()

    # ── ALGORITHM & ENVIRONMENT ──
    st.markdown('<p class="ev-section-title">🔬 Advanced Settings</p>', unsafe_allow_html=True)
    algorithm_sel = st.selectbox(
        "Search Algorithm",
        options=["A* Search Algorithm", "Dijkstra's Algorithm"],
        index=0,
        help="A* uses Haversine heuristics; Dijkstra exhaustively explores."
    )
    algo_key = "astar" if "A*" in algorithm_sel else "dijkstra"

    with st.expander("🌡️ Vehicle & Environment"):
        load_kg = st.slider("Passenger/Cargo Load (kg)", 0.0, 350.0, 75.0, 25.0)
        temp_c = st.slider("Ambient Temperature (°C)", -10.0, 40.0, 20.0, 1.0)

    # Sync vehicle parameters with the optimizer
    optimizer.vehicle_load_kg = load_kg
    optimizer.temperature_c = temp_c


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

# ── Title Row ──
st.markdown(
    '<div class="ev-main-header">⚡ EV Route Optimization System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="ev-sub-header">Smart Navigation for Electric Vehicles — '
    'India-Wide Real Roads · OpenStreetMap OSRM · ANN Energy Prediction</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# ROUTE COMPUTATION
# -----------------------------------------------------------------------------
india_result = None

if not is_voltcity_benchmark:
    # Validate both geocoded locations
    src_ok = source_geo is not None and source_geo.is_valid
    dst_ok = dest_geo is not None and dest_geo.is_valid

    if not source_query.strip() or not dest_query.strip():
        st.info("👆 Enter a **Source** and **Destination** in the left sidebar to calculate your route.")
        st.stop()

    if not src_ok or not dst_ok:
        st.warning("⚠️ One or both locations could not be resolved. Please check and try again.")
        st.stop()

    if abs(source_geo.lat - dest_geo.lat) < 0.001 and abs(source_geo.lon - dest_geo.lon) < 0.001:
        st.warning("⚠️ Source and destination are virtually identical. Please enter two distinct locations.")
        st.stop()

    with st.spinner("🛣️ Fetching real road routes from OpenStreetMap OSRM and running ANN evaluation…"):
        try:
            india_result = india_router.compute_india_routes(
                source=source_geo,
                dest=dest_geo,
                algorithm=algo_key,
                balanced_weights=balanced_weights,
                vehicle_load_kg=load_kg,
                temperature_c=temp_c
            )
        except Exception as err:
            st.error(f"🚫 **Routing Error:** {err}")
            st.caption("Check your internet connection or try a different location pair.")
            st.stop()

    candidate_routes: Dict[str, RouteResult] = india_result["candidate_routes"]
    routes_coords: Dict[str, List[Tuple[float, float]]] = india_result["routes_coords"]
    routes_instructions: Dict[str, List[TurnInstruction]] = india_result.get("routes_instructions", {})
    provider_name = india_result.get("provider_name", "OpenStreetMap OSRM")

    battery_state = EVBatteryState(
        capacity_kwh=battery_cap,
        current_soc_percent=current_soc,
        min_reserve_percent=min_reserve
    )
    comparison_summary = compare_routes_battery(candidate_routes, battery_state)
    active_route: RouteResult = candidate_routes[selected_mode]
    active_eval = comparison_summary.evaluations[selected_mode]
    active_instructions = routes_instructions.get(selected_mode, [])

else:
    # VoltCity benchmark grid
    with st.spinner("🏙️ Running VoltCity Simulation Grid…"):
        src_node, dst_node = "G", "L"
        candidate_routes = optimizer.compare_all_modes(src_node, dst_node, algorithm=algo_key)
        battery_state = EVBatteryState(
            capacity_kwh=battery_cap,
            current_soc_percent=current_soc,
            min_reserve_percent=min_reserve
        )
        comparison_summary = compare_routes_battery(candidate_routes, battery_state)
        active_route = candidate_routes[selected_mode]
        active_eval = comparison_summary.evaluations[selected_mode]
        active_instructions = []
        routes_coords = {}
        provider_name = "VoltCity Simulation Grid"


# -----------------------------------------------------------------------------
# SCIENTIFIC ROUTING NOTICE + SELECTION RATIONALE
# -----------------------------------------------------------------------------
st.markdown("""
<div class="ev-banner">
    <b>💡 Energy-Aware Navigation:</b>
    OpenStreetMap OSRM natively optimizes for <b>minimum transit time</b> (posted speed limits).
    Our <b>Artificial Neural Network (ANN)</b> separately models aerodynamic drag
    (P<sub>aero</sub> ∝ v³), vehicle payload, and topographic gradients to minimise
    true kWh consumption — so the <b>Energy-Efficient route</b> is never falsely equated
    with the fastest route.
</div>
""", unsafe_allow_html=True)

if not is_voltcity_benchmark and india_result and "mode_selections" in india_result:
    sel_res = india_result["mode_selections"].get(selected_mode)
    if sel_res and sel_res.selection_explanation:
        with st.expander(f"🎯 Selection Rationale — {selected_mode.upper()} Mode", expanded=False):
            st.info(
                f"**{sel_res.selection_explanation}**\n\n"
                f"*Scope:* {sel_res.optimality_scope}"
            )


# -----------------------------------------------------------------------------
# TOP METRICS ROW (5 cards)
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.metric(
        label="📏 Total Distance",
        value=f"{active_route.total_distance_km:.2f} km",
        help="Physical road distance along the selected route."
    )

with m_col2:
    st.metric(
        label="⏱️ Travel Time",
        value=fmt_time(active_route.total_time_min),
        help="Estimated transit time considering road speed limits."
    )

with m_col3:
    energy_delta = None
    if selected_mode == "energy" and comparison_summary.savings_vs_shortest_kwh > 0:
        energy_delta = f"-{comparison_summary.savings_vs_shortest_kwh:.3f} kWh vs Shortest"
    st.metric(
        label="🔋 ANN Energy",
        value=f"{active_route.total_energy_kwh:.3f} kWh",
        delta=energy_delta,
        delta_color="normal",
        help="Battery energy consumption predicted by the trained Artificial Neural Network."
    )

with m_col4:
    batt_delta_color = "off" if active_eval.is_safe else "inverse"
    st.metric(
        label="🔌 Battery Remaining",
        value=f"{active_eval.remaining_soc_percent:.1f} %",
        delta=f"{active_eval.remaining_energy_kwh:.2f} kWh",
        delta_color=batt_delta_color,
        help="Projected State of Charge upon reaching the destination."
    )

with m_col5:
    st.markdown(
        "<p style='font-size:0.75rem; font-weight:600; color:#8b949e; "
        "margin-bottom:6px; text-transform:uppercase; letter-spacing:0.05em;'>"
        "Feasibility</p>",
        unsafe_allow_html=True
    )
    if active_eval.status == FeasibilityStatus.SAFE:
        st.markdown('<span class="badge-safe">✅ SAFE TO TRAVEL</span>', unsafe_allow_html=True)
    elif active_eval.status == FeasibilityStatus.RESERVE_BREACHED:
        st.markdown('<span class="badge-warn">⚠️ RESERVE BREACH</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-danger">⛔ INFEASIBLE</span>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# ADVISORY BANNER
# -----------------------------------------------------------------------------
if active_eval.status == FeasibilityStatus.INFEASIBLE:
    st.error(
        f"🚨 **CRITICAL BATTERY WARNING — Vehicle CANNOT reach destination!**\n\n"
        f"- Route requires **{active_eval.energy_consumed_kwh:.2f} kWh** but battery holds only "
        f"**{battery_state.current_energy_kwh:.2f} kWh**.\n"
        f"- Deficit: **{active_eval.critical_deficit_kwh:.2f} kWh** short.\n"
        f"- **Charge at least {active_eval.charging_needed_kwh:.2f} kWh** before departing "
        f"(~{active_eval.est_charge_time_50kw_min:.0f} min at 50 kW "
        f"| ~{active_eval.est_charge_time_150kw_min:.0f} min at 150 kW)."
    )
elif active_eval.status == FeasibilityStatus.RESERVE_BREACHED:
    st.warning(
        f"⚠️ **Reserve Penetration Advisory** — The vehicle can reach the destination but will "
        f"breach the **{battery_state.min_reserve_percent:.0f}%** safety buffer.\n\n"
        f"- Arrival SoC: **{active_eval.remaining_soc_percent:.1f}%** "
        f"({active_eval.remaining_energy_kwh:.2f} kWh).\n"
        f"- Recommend topping up **{active_eval.charging_needed_kwh:.2f} kWh** en route."
    )
else:
    st.success(
        f"✅ **Route verified — Safe to travel.** Arriving with "
        f"**{active_eval.remaining_soc_percent:.1f}% SoC** "
        f"({active_eval.remaining_energy_kwh:.2f} kWh) — "
        f"above the {battery_state.min_reserve_percent:.0f}% reserve threshold."
    )


# -----------------------------------------------------------------------------
# MAP + TURN-BY-TURN  (60/40 split)
# -----------------------------------------------------------------------------
col_map, col_nav = st.columns([1.65, 1.0])

with col_map:
    st.markdown(
        f'<div class="ev-section-title">🗺️ Interactive Road Map '
        f'{mode_pill(selected_mode)}</div>',
        unsafe_allow_html=True
    )

    if not is_voltcity_benchmark:
        active_coords = routes_coords.get(selected_mode, [])
        # Build alternative routes dict (all modes except selected)
        alt_routes = {
            m: coords
            for m, coords in routes_coords.items()
            if m != selected_mode and coords
        }
        folium_map = create_india_route_map(
            source_loc=source_geo,
            dest_loc=dest_geo,
            route_coords=active_coords,
            active_route=active_route,
            selected_mode=selected_mode,
            alternative_routes=alt_routes if alt_routes else None,
            candidate_routes=candidate_routes
        )
    else:
        folium_map = create_city_folium_map(
            graph=city_graph,
            selected_path=active_route.path,
            source_node="G",
            destination_node="L",
            mode_name=selected_mode
        )

    st.components.v1.html(folium_map._repr_html_(), height=530, scrolling=False)

    # Map legend and attribution
    st.caption(
        "🟢 Green Pin: Start &nbsp;|&nbsp; "
        "🔴 Red Flag: Destination &nbsp;|&nbsp; "
        "⚡ Blue Bolt: EV Highway Charger &nbsp;|&nbsp; "
        "— Dashed lines: Alternative Routes &nbsp;|&nbsp; "
        f"Engine: {provider_name}"
    )
    st.markdown(f'<p class="osm-attr">🌐 {OSM_ATTRIBUTION}</p>', unsafe_allow_html=True)

with col_nav:
    st.markdown(
        f'<div class="ev-section-title">🚗 Navigation Details '
        f'<small style="color:#8b949e; font-weight:400;">({algo_key.upper()})</small></div>',
        unsafe_allow_html=True
    )

    tab_turns, tab_segments, tab_features = st.tabs([
        "🛣️ Turn-by-Turn",
        "⛰️ Road Segments",
        "🔬 ANN Features"
    ])

    with tab_turns:
        with st.container(height=390):
            if active_instructions:
                for step in active_instructions:
                    if step.maneuver_type == "depart":
                        icon = "🟢"
                    elif step.maneuver_type == "arrive":
                        icon = "🏁"
                    else:
                        icon = "↳"
                    st.markdown(
                        f"**{icon} Step {step.step_number}:** {step.instruction}  \n"
                        f"<small>📏 `{step.distance_km:.2f} km` &nbsp;|&nbsp; "
                        f"⏱️ `{step.duration_min:.1f} min` &nbsp;|&nbsp; "
                        f"🛣️ *{step.road_name}*</small>\n\n---",
                        unsafe_allow_html=True
                    )
            elif active_route.path:
                for i, node in enumerate(active_route.path, 1):
                    st.markdown(f"**Step {i}:** Node `{node}`")
            else:
                st.info("Turn-by-turn directions available for India-wide real-road mode.")

    with tab_segments:
        with st.container(height=390):
            if active_route.segments:
                for i, seg in enumerate(active_route.segments, 1):
                    st.markdown(
                        f"**{i}. {seg.source} → {seg.target}**\n"
                        f"- 📏 `{seg.distance_km:.2f} km` &nbsp; "
                        f"⏱️ `{seg.travel_time_min:.1f} min` &nbsp; "
                        f"⚡ `{seg.energy_kwh:.3f} kWh`\n"
                        f"- 🚦 `{seg.traffic_level.upper()}` &nbsp; "
                        f"⛰️ `{seg.slope_percent:+.2f}%` &nbsp; "
                        f"🏎️ `{seg.speed_limit_kmh} km/h`\n\n---"
                    )
            else:
                st.info("Detailed road segments available for India-wide real-road mode.")

    with tab_features:
        with st.container(height=390):
            st.caption("🔍 **ANN Input Feature Transparency** (MLPRegressor + StandardScaler)")
            first_seg = active_route.segments[0] if active_route.segments else None
            prov = getattr(first_seg, "feature_provenance", None) if first_seg else None

            if prov:
                prov_rows = []
                for fname, finfo in prov.items():
                    prov_rows.append({
                        "Feature": fname,
                        "Sample Value": f"{finfo.get('value')} {finfo.get('unit', '')}".strip(),
                        "Provenance": finfo.get("label", finfo.get("status", "")),
                        "Source": finfo.get("source", "")
                    })
                st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)
                st.markdown("""
**Legend:**
- 🟢 Directly Measured — from OSM geometry
- 🔵 Derived — from OSRM duration (v = d/t)
- 🟠 Estimated Topography — regional DEM
- 🟡 Estimated Traffic — speed/freeflow ratio
- 🟣 Computed Formula — civil engineering grade
- ⚪ User Specified — load & temperature controls
""")
            else:
                st.info("ANN feature provenance is available for India-wide real-road segments.")

    # Charging stops advisory
    if active_route.charging_stops:
        st.info(
            "⚡ **En-Route Charging Stops:**\n" +
            "\n".join(f"- {s}" for s in active_route.charging_stops)
        )


# -----------------------------------------------------------------------------
# ROUTE COMPARISON TABLE
# -----------------------------------------------------------------------------
st.markdown('<div class="ev-section-title">📊 Candidate Route Comparison</div>', unsafe_allow_html=True)

table_rows = []
for mode_key, r in candidate_routes.items():
    ev_eval = comparison_summary.evaluations[mode_key]
    if ev_eval.is_safe:
        status_str = "✅ Safe"
    elif ev_eval.is_feasible:
        status_str = "⚠️ Caution"
    else:
        status_str = "⛔ Infeasible"

    strategy_map = {
        "fastest": "Min Transit Time (OSRM)",
        "energy": "Min kWh (ANN Optimized)",
        "shortest": "Min Road Distance",
        "balanced": "Multi-Objective Weighted"
    }

    table_rows.append({
        "Mode": mode_key.capitalize(),
        "Algorithm": r.algorithm.upper(),
        "Distance (km)": round(r.total_distance_km, 2),
        "Time": fmt_time(r.total_time_min),
        "Energy (kWh)": round(r.total_energy_kwh, 3),
        "Remaining SoC": f"{ev_eval.remaining_soc_percent:.1f}%",
        "Remaining kWh": f"{ev_eval.remaining_energy_kwh:.2f}",
        "Charge Needed": f"{ev_eval.charging_needed_kwh:.2f} kWh" if ev_eval.charging_needed_kwh > 0 else "—",
        "Feasibility": status_str,
        "Strategy": strategy_map.get(mode_key, "—")
    })

comparison_df = pd.DataFrame(table_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Energy savings callouts
if comparison_summary.savings_vs_shortest_kwh > 0 or comparison_summary.savings_vs_fastest_kwh > 0:
    sav_col1, sav_col2 = st.columns(2)
    with sav_col1:
        st.info(
            f"💡 **vs. Shortest Route:** "
            f"+{comparison_summary.savings_vs_shortest_kwh:.3f} kWh saved "
            f"({comparison_summary.savings_vs_shortest_percent:.2f}% reduction)"
        )
    with sav_col2:
        st.info(
            f"💡 **vs. Fastest Route:** "
            f"+{comparison_summary.savings_vs_fastest_kwh:.3f} kWh saved "
            f"({comparison_summary.savings_vs_fastest_percent:.2f}% reduction)"
        )


# -----------------------------------------------------------------------------
# PERFORMANCE CHARTS
# -----------------------------------------------------------------------------
st.markdown('<div class="ev-section-title">📈 Multi-Mode Performance Metrics</div>', unsafe_allow_html=True)

chart_data = pd.DataFrame({
    "Mode": [r.mode.capitalize() for r in candidate_routes.values()],
    "Energy (kWh)": [r.total_energy_kwh for r in candidate_routes.values()],
    "Time (min)": [r.total_time_min for r in candidate_routes.values()],
    "Distance (km)": [r.total_distance_km for r in candidate_routes.values()],
}).set_index("Mode")

c_col1, c_col2, c_col3 = st.columns(3)
with c_col1:
    st.markdown("**🔋 Predicted Energy (kWh)**")
    st.bar_chart(chart_data["Energy (kWh)"], color="#10b981")
with c_col2:
    st.markdown("**⏱️ Travel Time (min)**")
    st.bar_chart(chart_data["Time (min)"], color="#f97316")
with c_col3:
    st.markdown("**📏 Road Distance (km)**")
    st.bar_chart(chart_data["Distance (km)"], color="#3b82f6")


# -----------------------------------------------------------------------------
# TECHNICAL DETAILS (VIVA / DEMO)
# -----------------------------------------------------------------------------
with st.expander("🎓 AI & System Architecture (College Project Reference)"):
    st.markdown(r"""
### System Architecture & Working Principles

1. **India-Wide OpenStreetMap Geocoding**
   - Nominatim geocoder with bounding-box validation:
     $6.0^\circ\text{N} \le \text{lat} \le 37.5^\circ\text{N}$,
     $68.0^\circ\text{E} \le \text{lon} \le 97.5^\circ\text{E}$.
   - Two-tier caching (memory + `data/geocoding_cache.json`) with ≥1 s rate limiting.

2. **Real Highway Network via OSRM**
   - Retrieves curvatures, distances, transit times and turn-by-turn directions across states.
   - `BaseRoutingProvider` interface allows swapping routing engines.

3. **ANN vs OSRM — Scientific Separation**
   - OSRM minimises transit time by posted speed limits.
   - ANN (`MLPRegressor`) models non-linear drag ($P_{\text{aero}} \propto v^3$),
     payload, and gradient — the Energy-Efficient route is never equated with fastest.

4. **Route Optimization Engines**
   - **Dijkstra's Algorithm**: Min-heap with guaranteed global optimum.
   - **A\* Search**: Admissible Haversine heuristic ($f(n) = g(n) + h(n)$).

5. **EV Battery Feasibility**
   - Usable energy: $E_{\text{usable}} = \max(0, E_{\text{current}} - E_{\text{reserve}})$.
   - Three states: SAFE · RESERVE_BREACHED · INFEASIBLE.
   - Charging time estimates for 50 kW and 150 kW DC fast chargers.

6. **CNN Traffic Classifier** (optional module)
   - Accepts road-camera images → classifies Low / Medium / High traffic.
   - Fallback heuristic pixel-density analyser when model weights are absent.
""")


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    f"Route Optimization System (Smart Navigation for Electric Vehicles) · "
    f"University AI/ML Project · "
    f"{OSM_ATTRIBUTION}"
)
