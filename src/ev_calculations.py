"""
src/ev_calculations.py
Electric Vehicle (EV) Battery State, Feasibility, and Consumption Analysis Module.

Provides reusable classes and functions to:
  1. Calculate available and usable battery energy.
  2. Compute post-trip state-of-charge (SoC) and remaining kWh.
  3. Perform battery feasibility and safety constraint evaluations.
  4. Estimate charging deficit and required charging energy/time.
  5. Compare energy consumption and percentage savings across routing modes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union


class FeasibilityStatus(Enum):
    """Classification of trip feasibility based on battery state."""
    SAFE = "SAFE"                              # Reaches destination with reserve intact (SoC >= reserve)
    RESERVE_BREACHED = "RESERVE_BREACHED"      # Reaches destination, but dips into reserve buffer (0% < SoC < reserve)
    INFEASIBLE = "INFEASIBLE"                  # Cannot reach destination; vehicle will be stranded (SoC <= 0%)


@dataclass
class EVBatteryState:
    """
    Represents the operational state of an Electric Vehicle battery pack.

    Parameters:
        capacity_kwh (float): Total nominal battery pack capacity in kilowatt-hours (kWh).
        current_soc_percent (float): Current State of Charge as a percentage (0% to 100%).
        min_reserve_percent (float): Minimum reserve threshold percentage to preserve battery health (default: 15.0%).
    """
    capacity_kwh: float
    current_soc_percent: float
    min_reserve_percent: float = 15.0

    def __post_init__(self):
        if self.capacity_kwh <= 0:
            raise ValueError(f"Battery capacity must be strictly positive, got {self.capacity_kwh} kWh.")
        if not (0.0 <= self.current_soc_percent <= 100.0):
            raise ValueError(f"Current SoC must be between 0% and 100%, got {self.current_soc_percent}%.")
        if not (0.0 <= self.min_reserve_percent < 100.0):
            raise ValueError(f"Reserve percentage must be between 0% and 99%, got {self.min_reserve_percent}%.")

    @property
    def current_energy_kwh(self) -> float:
        """Total current chemical energy stored in battery (kWh)."""
        return round(self.capacity_kwh * (self.current_soc_percent / 100.0), 3)

    @property
    def reserve_energy_kwh(self) -> float:
        """Energy reserved to prevent deep-discharge damage (kWh)."""
        return round(self.capacity_kwh * (self.min_reserve_percent / 100.0), 3)

    @property
    def usable_energy_kwh(self) -> float:
        """Usable energy available for travel before encroaching into the reserve buffer (kWh)."""
        return round(max(0.0, self.current_energy_kwh - self.reserve_energy_kwh), 3)

    @property
    def usable_soc_percent(self) -> float:
        """Usable State of Charge percentage before hitting the reserve limit (%)."""
        return round(max(0.0, self.current_soc_percent - self.min_reserve_percent), 2)


@dataclass
class RouteBatteryEvaluation:
    """
    Detailed battery feasibility evaluation for a single candidate navigation route.
    """
    mode_name: str
    distance_km: float
    travel_time_min: float
    energy_consumed_kwh: float
    initial_soc_percent: float
    remaining_energy_kwh: float
    remaining_soc_percent: float
    status: FeasibilityStatus
    is_feasible: bool            # True if vehicle physically reaches destination (SoC > 0)
    is_safe: bool                # True if vehicle reaches destination with reserve intact (SoC >= reserve)
    energy_deficit_kwh: float    # Additional kWh needed to maintain reserve buffer upon arrival
    critical_deficit_kwh: float  # Deficit below 0 kWh (absolute physical shortfall)
    charging_needed_kwh: float   # Recommended charge energy including 90% charging efficiency
    warning_message: Optional[str]
    est_charge_time_50kw_min: float
    est_charge_time_150kw_min: float


@dataclass
class RouteComparisonSummary:
    """
    Side-by-side comparison of candidate routes showing energy consumption and relative savings.
    """
    evaluations: Dict[str, RouteBatteryEvaluation]
    battery_state: EVBatteryState
    savings_vs_shortest_kwh: float
    savings_vs_shortest_percent: float
    savings_vs_fastest_kwh: float
    savings_vs_fastest_percent: float
    recommended_mode: str
    recommendation_reason: str


def calculate_available_battery_energy(
    battery_capacity_kwh: float,
    current_soc_percent: float,
    min_reserve_percent: float = 15.0
) -> Dict[str, float]:
    """
    Computes available, reserve, and usable battery energy in kWh and percentages.

    Formulas:
      - Current Energy (kWh) = Capacity * (Current_SoC / 100)
      - Reserve Energy (kWh) = Capacity * (Reserve_SoC / 100)
      - Usable Energy (kWh)  = max(0, Current Energy - Reserve Energy)
      - Usable SoC (%)       = max(0, Current_SoC - Reserve_SoC)

    Returns:
        dict: Breakdown of battery state metrics.
    """
    state = EVBatteryState(
        capacity_kwh=battery_capacity_kwh,
        current_soc_percent=current_soc_percent,
        min_reserve_percent=min_reserve_percent
    )
    return {
        "capacity_kwh": state.capacity_kwh,
        "current_soc_percent": state.current_soc_percent,
        "current_energy_kwh": state.current_energy_kwh,
        "min_reserve_percent": state.min_reserve_percent,
        "reserve_energy_kwh": state.reserve_energy_kwh,
        "usable_energy_kwh": state.usable_energy_kwh,
        "usable_soc_percent": state.usable_soc_percent
    }


def evaluate_route_battery(
    energy_consumed_kwh: float,
    battery_state: EVBatteryState,
    distance_km: float = 0.0,
    travel_time_min: float = 0.0,
    mode_name: str = "Candidate Route",
    charging_efficiency: float = 0.90
) -> RouteBatteryEvaluation:
    """
    Evaluates whether an EV can complete a specific route given its current battery state.

    Formulas:
      - Remaining Energy (kWh) = Current Energy - Energy Consumed
      - Remaining SoC (%)       = (Remaining Energy / Capacity) * 100
      - Energy Deficit (kWh)   = max(0, Energy Consumed - Usable Energy)
      - Critical Deficit (kWh) = max(0, Energy Consumed - Current Energy)

    Guarantees:
      - Strict truthfulness: never claims the vehicle reaches destination if Energy Consumed > Current Energy.
    """
    current_energy = battery_state.current_energy_kwh
    usable_energy = battery_state.usable_energy_kwh
    capacity = battery_state.capacity_kwh

    # Calculate post-trip energy balance
    remaining_energy = current_energy - energy_consumed_kwh
    remaining_soc = (remaining_energy / capacity) * 100.0

    # Deficits
    energy_deficit = max(0.0, round(energy_consumed_kwh - usable_energy, 3))
    critical_deficit = max(0.0, round(energy_consumed_kwh - current_energy, 3))

    # Grid charging required (compensates for standard ~90% AC/DC conversion loss)
    charging_needed = round(energy_deficit / charging_efficiency, 2) if energy_deficit > 0 else 0.0

    # Charging time estimates (minutes): Energy (kWh) / Power (kW) * 60
    charge_time_50kw = round((charging_needed / 50.0) * 60.0, 1) if charging_needed > 0 else 0.0
    charge_time_150kw = round((charging_needed / 150.0) * 60.0, 1) if charging_needed > 0 else 0.0

    # Feasibility and safety classification
    if energy_consumed_kwh > current_energy:
        # Cannot physically reach destination
        status = FeasibilityStatus.INFEASIBLE
        is_feasible = False
        is_safe = False
        # Calculate estimated distance before battery depletion
        avg_rate_kwh_per_km = energy_consumed_kwh / max(0.1, distance_km)
        stranded_dist_km = round(current_energy / max(0.01, avg_rate_kwh_per_km), 1)
        warning = (
            f"CRITICAL WARNING: The EV CANNOT reach destination! "
            f"Trip requires {energy_consumed_kwh:.2f} kWh, but only {current_energy:.2f} kWh is available. "
            f"Battery will be fully depleted ~{stranded_dist_km} km into the trip (shortfall: {critical_deficit:.2f} kWh). "
            f"You MUST charge at least {charging_needed:.2f} kWh before or during the journey."
        )

    elif energy_consumed_kwh > usable_energy:
        # Reaches destination, but breaches the reserve threshold
        status = FeasibilityStatus.RESERVE_BREACHED
        is_feasible = True
        is_safe = False
        warning = (
            f"CAUTION: Route is physically reachable, but penetrates the {battery_state.min_reserve_percent:.0f}% reserve buffer. "
            f"Estimated arrival SoC is {remaining_soc:.1f}% ({remaining_energy:.2f} kWh), "
            f"which is below the {battery_state.min_reserve_percent:.0f}% reserve limit ({battery_state.reserve_energy_kwh:.2f} kWh). "
            f"Recommended to charge ~{charging_needed:.2f} kWh ({charge_time_50kw:.0f} min at 50kW) en route."
        )

    else:
        # Completely safe
        status = FeasibilityStatus.SAFE
        is_feasible = True
        is_safe = True
        warning = None

    return RouteBatteryEvaluation(
        mode_name=mode_name,
        distance_km=round(distance_km, 2),
        travel_time_min=round(travel_time_min, 1),
        energy_consumed_kwh=round(energy_consumed_kwh, 3),
        initial_soc_percent=round(battery_state.current_soc_percent, 1),
        remaining_energy_kwh=round(remaining_energy, 3),
        remaining_soc_percent=round(remaining_soc, 1),
        status=status,
        is_feasible=is_feasible,
        is_safe=is_safe,
        energy_deficit_kwh=energy_deficit,
        critical_deficit_kwh=critical_deficit,
        charging_needed_kwh=charging_needed,
        warning_message=warning,
        est_charge_time_50kw_min=charge_time_50kw,
        est_charge_time_150kw_min=charge_time_150kw
    )


def compare_routes_battery(
    routes_dict: Dict[str, Any],
    battery_state: EVBatteryState
) -> RouteComparisonSummary:
    """
    Evaluates battery metrics for candidate routing modes (Shortest, Fastest, Energy, Balanced)
    and computes energy savings between routes.

    Formulas:
      - Energy Savings (kWh) = Baseline Energy - Energy-Efficient Energy
      - Percentage Savings  = (Energy Savings / Baseline Energy) * 100
    """
    evaluations: Dict[str, RouteBatteryEvaluation] = {}

    for mode_name, route_obj in routes_dict.items():
        # Handle both RouteResult objects and dictionary inputs
        if hasattr(route_obj, "total_energy_kwh"):
            energy = route_obj.total_energy_kwh
            dist = route_obj.total_distance_km
            time = route_obj.total_time_min
        else:
            energy = float(route_obj["energy_kwh"])
            dist = float(route_obj.get("distance_km", 0.0))
            time = float(route_obj.get("time_min", 0.0))

        evaluations[mode_name] = evaluate_route_battery(
            energy_consumed_kwh=energy,
            battery_state=battery_state,
            distance_km=dist,
            travel_time_min=time,
            mode_name=mode_name
        )

    # Compute energy savings: Energy Efficient vs Shortest and vs Fastest
    energy_eff_eval = evaluations.get("energy") or evaluations.get("energy_efficient")
    shortest_eval = evaluations.get("shortest")
    fastest_eval = evaluations.get("fastest")

    # Savings vs Shortest
    if energy_eff_eval and shortest_eval and shortest_eval.energy_consumed_kwh > 0:
        savings_vs_short_kwh = max(0.0, shortest_eval.energy_consumed_kwh - energy_eff_eval.energy_consumed_kwh)
        savings_vs_short_pct = (savings_vs_short_kwh / shortest_eval.energy_consumed_kwh) * 100.0
    else:
        savings_vs_short_kwh = 0.0
        savings_vs_short_pct = 0.0

    # Savings vs Fastest
    if energy_eff_eval and fastest_eval and fastest_eval.energy_consumed_kwh > 0:
        savings_vs_fast_kwh = max(0.0, fastest_eval.energy_consumed_kwh - energy_eff_eval.energy_consumed_kwh)
        savings_vs_fast_pct = (savings_vs_fast_kwh / fastest_eval.energy_consumed_kwh) * 100.0
    else:
        savings_vs_fast_kwh = 0.0
        savings_vs_fast_pct = 0.0

    # Determine recommended mode
    # Preference: Energy if safe -> Balanced if safe -> Fastest if safe -> Shortest if safe
    # If none safe, recommend mode with lowest energy deficit
    safe_modes = [k for k, v in evaluations.items() if v.is_safe]
    feasible_modes = [k for k, v in evaluations.items() if v.is_feasible]

    if "energy" in safe_modes:
        recommended = "energy"
        reason = "Safest route with maximum battery conservation and reserve preservation."
    elif "balanced" in safe_modes:
        recommended = "balanced"
        reason = "Safe route providing the best compromise between transit time and battery conservation."
    elif safe_modes:
        recommended = safe_modes[0]
        reason = f"Mode '{recommended}' satisfies the safety reserve buffer."
    elif "energy" in feasible_modes:
        recommended = "energy"
        reason = "Encroaches on reserve buffer, but reaches destination with the smallest battery deficit."
    elif feasible_modes:
        # Minimum energy among feasible, preferring 'energy' mode on tie
        min_e = min(evaluations[k].energy_consumed_kwh for k in feasible_modes)
        candidate_min = [k for k in feasible_modes if abs(evaluations[k].energy_consumed_kwh - min_e) < 1e-5]
        recommended = "energy" if "energy" in candidate_min else candidate_min[0]
        reason = f"Reaches destination with minimal reserve penetration ({recommended})."
    else:
        # All infeasible: recommend energy mode to minimize charging deficit
        min_e = min(evaluations[k].energy_consumed_kwh for k in evaluations.keys())
        candidate_min = [k for k in evaluations.keys() if abs(evaluations[k].energy_consumed_kwh - min_e) < 1e-5]
        recommended = "energy" if "energy" in candidate_min else candidate_min[0]
        reason = "Trip is INFEASIBLE under current battery. Recommend charging before departure."

    return RouteComparisonSummary(
        evaluations=evaluations,
        battery_state=battery_state,
        savings_vs_shortest_kwh=round(savings_vs_short_kwh, 3),
        savings_vs_shortest_percent=round(savings_vs_short_pct, 2),
        savings_vs_fastest_kwh=round(savings_vs_fast_kwh, 3),
        savings_vs_fastest_percent=round(savings_vs_fast_pct, 2),
        recommended_mode=recommended,
        recommendation_reason=reason
    )


def format_battery_report(summary: RouteComparisonSummary) -> str:
    """
    Renders an ASCII analysis report of the battery state, route feasibilities,
    and comparative savings.
    """
    bs = summary.battery_state
    lines = []
    lines.append("=" * 80)
    lines.append("                 EV BATTERY & ROUTE FEASIBILITY REPORT")
    lines.append("=" * 80)
    lines.append(" 1. VEHICLE BATTERY SPECIFICATION & CURRENT STATUS:")
    lines.append(f"    - Pack Capacity        : {bs.capacity_kwh:.1f} kWh")
    lines.append(f"    - Current SoC          : {bs.current_soc_percent:.1f}% ({bs.current_energy_kwh:.2f} kWh stored)")
    lines.append(f"    - Safety Reserve Limit : {bs.min_reserve_percent:.1f}% ({bs.reserve_energy_kwh:.2f} kWh reserved)")
    lines.append(f"    - Usable Energy        : {bs.usable_energy_kwh:.2f} kWh ({bs.usable_soc_percent:.1f}% usable SoC)")
    lines.append("-" * 80)

    lines.append(" 2. CANDIDATE ROUTE CONSUMPTION & BATTERY BALANCE:")
    header = (
        f" {'Mode':<14} | {'Dist':<7} | {'Time':<8} | {'Energy':<10} | "
        f"{'End SoC':<9} | {'End kWh':<9} | {'Feasibility Status'}"
    )
    lines.append(header)
    lines.append(" " + "-" * 78)

    for mode_name, ev in summary.evaluations.items():
        status_label = {
            FeasibilityStatus.SAFE: "[OK] SAFE",
            FeasibilityStatus.RESERVE_BREACHED: "[!] RESERVE BREACH",
            FeasibilityStatus.INFEASIBLE: "[X] INFEASIBLE"
        }.get(ev.status, str(ev.status.value))

        row = (
            f" {ev.mode_name.capitalize():<14} | {ev.distance_km:>5.1f}km | {ev.travel_time_min:>6.1f}m | "
            f"{ev.energy_consumed_kwh:>7.3f}kWh | {ev.remaining_soc_percent:>7.1f}% | "
            f"{ev.remaining_energy_kwh:>7.2f}kWh | {status_label}"
        )
        lines.append(row)
    lines.append("-" * 80)

    lines.append(" 3. ENERGY SAVINGS ANALYSIS:")
    lines.append(
        f"    - Energy Savings vs. Shortest Route : {summary.savings_vs_shortest_kwh:+.3f} kWh "
        f"({summary.savings_vs_shortest_percent:.2f}% savings)"
    )
    lines.append(
        f"    - Energy Savings vs. Fastest Route  : {summary.savings_vs_fastest_kwh:+.3f} kWh "
        f"({summary.savings_vs_fastest_percent:.2f}% savings)"
    )
    lines.append("-" * 80)

    lines.append(" 4. WARNINGS & CHARGING REQUIREMENTS:")
    warnings_found = False
    for mode_name, ev in summary.evaluations.items():
        if ev.warning_message:
            warnings_found = True
            lines.append(f"    * [{ev.mode_name.upper()}]: {ev.warning_message}")
            if ev.charging_needed_kwh > 0:
                lines.append(
                    f"      --> Required Charge : {ev.charging_needed_kwh:.2f} kWh "
                    f"({ev.est_charge_time_50kw_min:.0f} min @ 50kW | {ev.est_charge_time_150kw_min:.0f} min @ 150kW)"
                )
    if not warnings_found:
        lines.append("    * All candidate routes are completely SAFE. No en-route charging required.")
    lines.append("-" * 80)

    lines.append(f" 5. SYSTEM RECOMMENDATION: Mode '{summary.recommended_mode.upper()}'")
    lines.append(f"    Reason: {summary.recommendation_reason}")
    lines.append("=" * 80)

    return "\n".join(lines)
