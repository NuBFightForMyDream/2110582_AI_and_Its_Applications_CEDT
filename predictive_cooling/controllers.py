"""The three ways to run the room that we compare.

* FixedSetpoint  — one constant supply-air setpoint (how many rooms are run today)
* RuleBased      — react to the hottest inlet: too warm -> colder, comfortably cool -> warmer
* AIController   — forecaster + MPC (step 3-4) behind the safety filter (step 5)
Every controller sees the same sensor readings and returns the next setpoint.
"""
import math

import numpy as np
import pandas as pd

from . import config as C
from .forecast import STATE_FEATURES
from .mpc import MPC
from .safety import SafetyFilter


class FixedSetpoint:
    kind = "fixed"

    def __init__(self, setpoint):
        self.setpoint = float(setpoint)
        self.name = "Fixed setpoint"
        self.initial_sp = self.setpoint

    def decide(self, prev, reading, current_sp):
        return self.setpoint, {"mode": "Fixed"}


class RuleBased:
    kind = "rule"

    def __init__(self, upper=26.0, lower=24.5, step_up=0.5, step_down=1.0):
        self.upper, self.lower, self.step_up, self.step_down = upper, lower, step_up, step_down
        self.name = "Rule-based"
        self.initial_sp = 20.0

    def decide(self, prev, reading, current_sp):
        inlets = [reading[c] for c in C.INLET_COLS if not math.isnan(reading[c])]
        if not inlets:
            return current_sp, {"mode": "Rule"}
        hottest = max(inlets)
        sp = current_sp
        if hottest > self.upper:
            sp -= self.step_down
        elif hottest < self.lower:
            sp += self.step_up
        return float(np.clip(sp, C.SP_MIN, C.SP_MAX)), {"mode": "Rule"}

    def describe(self):
        return (f"hottest inlet > {self.upper:g} °C → setpoint −{self.step_down:g} °C; "
                f"< {self.lower:g} °C → +{self.step_up:g} °C")


def live_state(prev, reading, current_sp) -> pd.Series:
    """The same features as forecast.state_features, built from the two latest live readings."""
    def hottest(r):
        values = [r[c] for c in C.INLET_COLS if not math.isnan(r[c])]
        return max(values) if values else math.nan

    def average(r):
        values = [r[c] for c in C.INLET_COLS if not math.isnan(r[c])]
        return sum(values) / len(values) if values else math.nan

    ts = reading["timestamp"]
    hour = ts.hour + ts.minute / 60
    now_hot = hottest(reading)
    state = {
        "max_inlet_c": now_hot, "mean_inlet_c": average(reading),
        "inlet_trend_c": now_hot - hottest(prev) if prev else 0.0,
        "supply_air_c": reading["supply_air_c"], "setpoint_prev_c": current_sp,
        "it_power_kw": reading["it_power_kw"],
        "it_trend_kw": reading["it_power_kw"] - prev["it_power_kw"] if prev else 0.0,
        "cooling_kw": reading["cooling_kw"], "outdoor_temp_c": reading["outdoor_temp_c"],
        "hour": hour, "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekend": float(ts.day_of_week >= 5),
    }
    return pd.Series(state)[STATE_FEATURES]


class AIController:
    kind = "ai"

    BIAS_SMOOTHING = 0.3          # how fast the feedback correction follows new forecast errors
    BIAS_RANGE = (-0.5, 2.0)      # never trust a correction that would make the room look much cooler

    def __init__(self, mpc: MPC, safety: SafetyFilter, use_safety=True):
        self.mpc, self.safety, self.use_safety = mpc, safety, use_safety
        self.name = "AI + MPC" if use_safety else "AI + MPC (no safety filter)"
        self.initial_sp = 20.0
        self.reset()

    def reset(self):
        self.bias, self.expected_next = 0.0, None

    def _learn_from_error(self, state):
        """Feedback: compare last step's forecast with the measured hottest inlet."""
        if self.expected_next is not None and not math.isnan(state["max_inlet_c"]):
            error = state["max_inlet_c"] - self.expected_next
            self.bias = (1 - self.BIAS_SMOOTHING) * self.bias + self.BIAS_SMOOTHING * error
            self.bias = float(np.clip(self.bias, *self.BIAS_RANGE))
        self.expected_next = None

    def decide(self, prev, reading, current_sp):
        state = live_state(prev, reading, current_sp)
        self._learn_from_error(state)
        if state.isna().any():                      # a whole sensor group is gone: nothing to plan with
            proposed, reason = current_sp, "no_state"
        else:
            proposed, reason, table = self.mpc.decide(state, current_sp, self.bias)
            chosen = table[table["chosen"]].iloc[0]
            self.expected_next = float(chosen["nominal_1"])
        info = {"proposed": proposed, "reason": reason, "bias": self.bias, "state": state}
        if not self.use_safety:
            final = float(np.clip(proposed, max(C.SP_MIN, current_sp - 1), min(C.SP_MAX, current_sp + 1)))
            return final, info | {"mode": "AI", "rules": ""}
        unfamiliar = self.mpc.forecaster.out_of_range(state) if not state.isna().any() else []
        final, fired = self.safety.check(proposed, current_sp, reading, unfamiliar)
        if {"R1_missing_data", "R2_unfamiliar_conditions"} & set(fired):
            mode = "Fallback"
        elif "R3_too_hot_now" in fired:
            mode = "Emergency"
        else:
            mode = "AI"
        if final != proposed:
            self.expected_next = None               # the plan we forecast was not the one applied
        return final, info | {"mode": mode, "rules": ",".join(fired)}
