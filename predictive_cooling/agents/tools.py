"""Python tools the agents may call (through the G2 gateway). Each returns plain JSON-able data.

The tools wrap the pipeline steps:
    read_sensors / inspect_data_quality  -> step 1-2 (sensor log, cleaning rules)
    summarize_state / forecast_hold      -> step 3   (trained forecaster)
    run_mpc / check_safety               -> step 4-5 (81-plan MPC, safety filter)
A CaseWindow holds only readings at or before the case time, so no tool can look into the future.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .. import config as C
from ..controllers import live_state
from ..forecast import Forecaster
from ..mpc import MPC
from ..preprocess import STUCK_READINGS, VALID_RANGE
from ..safety import SafetyFilter

REQUIRED = ["supply_air_c", "it_power_kw", "cooling_kw", "outdoor_temp_c"]
STEP = pd.Timedelta(minutes=C.STEP_MINUTES)


@dataclass
class AgentContext:
    """Everything the tools need: the sensor log and the trained step-3/4/5 components."""
    log: pd.DataFrame
    forecaster: Forecaster
    margins: list
    fallback_sp: float

    def __post_init__(self):
        self.mpc = MPC(self.forecaster, self.margins)
        self.safety = SafetyFilter(self.fallback_sp)

    @classmethod
    def from_bundle(cls, bundle):
        return cls(bundle["raw"], bundle["forecaster"], list(bundle["margins"]), bundle["fixed_sp"])


def default_context():
    from .. import pipeline
    return AgentContext.from_bundle(pipeline.load())


@dataclass
class CaseWindow:
    log: pd.DataFrame
    at: pd.Timestamp
    hours: float
    fault_racks: tuple = ()
    frame: pd.DataFrame = field(init=False)
    clean: pd.DataFrame | None = field(init=False, default=None)

    def __post_init__(self):
        self.at = pd.Timestamp(self.at)
        start = self.at - pd.Timedelta(hours=self.hours)
        grid = pd.date_range(start, self.at, freq=STEP)
        frame = self.log.loc[start:self.at].reindex(grid).copy()      # at or before `at` only
        frame.loc[self.at, "crah_setpoint_c"] = np.nan                # this decision is not made yet
        for rack in self.fault_racks:                                  # demo: sensors go silent
            frame.loc[self.at - 2 * STEP:self.at, f"inlet_{rack}_c"] = np.nan
        self.frame = frame

    @property
    def setpoint_in_use(self):
        value = self.frame["crah_setpoint_c"].get(self.at - STEP, np.nan)
        return None if pd.isna(value) else float(value)


def _num(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits)


# ---- S2 inspect --------------------------------------------------------------------------
def read_sensors(win: CaseWindow):
    latest = win.frame.loc[win.at]
    return {
        "case_time": f"{win.at:%Y-%m-%d %H:%M}", "readings_in_window": int(win.frame["it_power_kw"].notna().sum()),
        "expected_readings": len(win.frame), "setpoint_in_use_c": win.setpoint_in_use,
        "latest": {c: _num(latest[c]) for c in C.INLET_COLS + REQUIRED + ["pue"]},
    }


def inspect_data_quality(win: CaseWindow):
    frame = win.frame.copy()
    errors, warnings = [], []
    if frame.loc[win.at, C.INLET_COLS + REQUIRED].isna().all():
        errors.append("NO_READING_AT_CASE_TIME")
    if frame.loc[win.at - STEP, C.INLET_COLS + REQUIRED].isna().all():
        errors.append("NO_PREVIOUS_READING")
    if win.setpoint_in_use is None:
        errors.append("SETPOINT_IN_USE_UNKNOWN")
    for col, (low, high) in VALID_RANGE.items():
        if col not in frame or col == "crah_setpoint_c":
            continue
        bad = (frame[col] < low) | (frame[col] > high)
        if bad.any():
            warnings.append(f"IMPOSSIBLE_VALUE_REMOVED:{col}")
            frame.loc[bad, col] = np.nan
    stuck = set()
    for col in C.INLET_COLS:
        tail = frame[col].iloc[-STUCK_READINGS:]
        if tail.notna().all() and tail.nunique() == 1:
            errors.append(f"STUCK_SENSOR:{col[6:8]}")
            stuck.add(col)
    trusted = frame.copy()                       # holds are only ever copied from a trusted reading
    if not any(e.startswith("NO_") for e in errors):
        for col in C.INLET_COLS + REQUIRED:
            if col in stuck:
                continue
            for t in (win.at - STEP, win.at):
                if pd.notna(trusted.loc[t, col]):
                    continue
                earlier = trusted.loc[t - STEP, col] if t - STEP in trusted.index else np.nan
                if pd.notna(earlier):
                    frame.loc[t, col] = earlier
                    warnings.append(f"HELD_LAST_VALUE:{col}@{t:%H:%M}")
                else:
                    code = f"MISSING_SENSOR:{col[6:8]}" if col.startswith("inlet_") else f"MISSING_VALUE:{col}"
                    if code not in errors:
                        errors.append(code)
    frame["max_inlet_c"] = frame[C.INLET_COLS].max(axis=1, skipna=False)
    win.clean = frame
    return {"ready": not errors, "errors": errors, "warnings": warnings}


# ---- S3 analyze & forecast -----------------------------------------------------------------
def _reading(win, t):
    row = win.clean.loc[t].to_dict()
    row["timestamp"] = t
    return row


def _state(win, t, setpoint_prev):
    return live_state(_reading(win, t - STEP), _reading(win, t), setpoint_prev)


def current_state(win):
    return _state(win, win.at, win.setpoint_in_use)


def summarize_state(win: CaseWindow):
    now = win.clean.loc[win.at]
    inlets = now[C.INLET_COLS].astype(float)
    hottest = inlets.idxmax()
    hour_ago = win.clean["max_inlet_c"].get(win.at - 4 * STEP, np.nan)
    return {
        "hottest_rack": hottest[6:8], "hottest_inlet_c": _num(inlets.max(), 2),
        "headroom_to_limit_c": _num(C.INLET_LIMIT - inlets.max(), 2), "mean_inlet_c": _num(inlets.mean(), 2),
        "hottest_inlet_change_last_hour_c": _num(inlets.max() - hour_ago, 2),
        "it_power_kw": _num(now["it_power_kw"], 1), "cooling_kw": _num(now["cooling_kw"], 1),
        "pue": _num(now["facility_kw"] / now["it_power_kw"], 3) if pd.notna(now.get("facility_kw")) else None,
        "outdoor_temp_c": _num(now["outdoor_temp_c"], 1), "setpoint_in_use_c": win.setpoint_in_use,
        "weekday_gpu_batch_window": bool(win.at.day_of_week < 5 and 0 <= win.at.hour < 5),
    }


def recent_forecast_error(win: CaseWindow, ctx: AgentContext):
    """Feedback: how far the 15-minute forecast was off during the last hour (mean, °C)."""
    errors = []
    for k in range(1, 5):
        t = win.at - k * STEP
        if t - 2 * STEP < win.clean.index[0]:
            continue
        sp_prev, sp_used = win.clean.loc[t - STEP, "crah_setpoint_c"], win.clean.loc[t, "crah_setpoint_c"]
        state = _state(win, t, sp_prev)
        actual = win.clean.loc[t + STEP, "max_inlet_c"]
        if state.isna().any() or pd.isna(sp_used) or pd.isna(actual):
            continue
        predicted = ctx.forecaster.predict_plans(state, np.array([[sp_used] * C.HORIZON]))["max_inlet_c"][0, 0]
        errors.append(actual - predicted)
    bias = float(np.mean(errors)) if errors else 0.0
    return float(np.clip(bias, -0.5, 2.0)), len(errors)


def forecast_hold(win: CaseWindow, ctx: AgentContext):
    state = current_state(win)
    bias, n = recent_forecast_error(win, ctx)
    sp = win.setpoint_in_use
    pred = ctx.forecaster.predict_plans(state, np.array([[sp] * C.HORIZON]))
    inlet = pred["max_inlet_c"][0] + bias
    guarded = inlet + np.asarray(ctx.margins)
    return {
        "hold_setpoint_c": sp, "horizon_minutes": [15 * (k + 1) for k in range(C.HORIZON)],
        "predicted_hottest_inlet_c": [_num(v, 2) for v in inlet],
        "with_safety_margin_c": [_num(v, 2) for v in guarded],
        "predicted_energy_next_hour_kwh": _num((pred["cooling_kw"][0] + pred["it_power_kw"][0]).sum() * C.DT_H, 1),
        "recent_forecast_error_c": _num(bias, 2), "recent_errors_used": n,
        "holding_is_safe": bool((guarded <= C.INLET_LIMIT).all()),
        "unfamiliar_inputs": ctx.forecaster.out_of_range(state),
    }


# ---- S4 evaluate ---------------------------------------------------------------------------
def mpc_decision(win: CaseWindow, ctx: AgentContext):
    state = current_state(win)
    bias, _ = recent_forecast_error(win, ctx)
    first, reason, table = ctx.mpc.decide(state, win.setpoint_in_use, bias)
    return first, reason, table, state


def run_mpc(win: CaseWindow, ctx: AgentContext):
    first, reason, table, _ = mpc_decision(win, ctx)
    chosen = table[table["chosen"]].iloc[0]
    hold = table[table["moves"] == 0].iloc[0]
    alternatives = table[table["safe"]].nsmallest(5, "cost")
    return {
        "plans_tried": len(table), "safe_plans": int(table["safe"].sum()), "reason": reason,
        "chosen_plan_c": [float(chosen[f"sp_{k}"]) for k in range(1, C.HORIZON + 1)], "first_step_c": first,
        "chosen_energy_next_hour_kwh": _num(chosen["energy_kwh"], 2), "hold_energy_next_hour_kwh": _num(hold["energy_kwh"], 2),
        "chosen_peak_with_margin_c": _num(chosen["peak_guarded_c"], 2), "hold_is_safe": bool(hold["safe"]),
        "cheapest_safe_plans": [{"plan": r["plan"], "energy_kwh": _num(r["energy_kwh"], 2),
                                 "peak_with_margin_c": _num(r["peak_guarded_c"], 2)} for _, r in alternatives.iterrows()],
    }


def check_safety(win: CaseWindow, ctx: AgentContext, setpoint: float):
    state = current_state(win)
    final, fired = ctx.safety.check(float(setpoint), win.setpoint_in_use, _reading(win, win.at),
                                    ctx.forecaster.out_of_range(state))
    return {"proposed_c": float(setpoint), "final_c": final, "rules_fired": fired, "changed": final != float(setpoint)}


def expected_recommendation(win: CaseWindow, ctx: AgentContext):
    """What the verified tools say should be proposed (used by the offline advisor and by G3)."""
    first, reason, table, _ = mpc_decision(win, ctx)
    safety = check_safety(win, ctx, first)
    return safety["final_c"], reason, table, safety


def fallback_step(win: CaseWindow, ctx: AgentContext):
    """When data is not ready, safety rule R1 moves the room toward the proven fallback setpoint."""
    current = win.setpoint_in_use if win.setpoint_in_use is not None else ctx.fallback_sp
    return float(min(max(ctx.fallback_sp, current - 1), current + 1))
