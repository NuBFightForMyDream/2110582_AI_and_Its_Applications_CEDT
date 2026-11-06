"""Step 4 — Model Predictive Control: try 81 plans, keep the safe ones, pick the cheapest.

Every 15 minutes:
1. Build all plans for the next hour: each 15-min step can lower / keep / raise the setpoint by
   1 °C -> 3 x 3 x 3 x 3 = 81 plans.
2. Ask the forecaster what each plan does to the hottest inlet and to cooling + IT power.
3. A plan is SAFE if, at every step, predicted hottest inlet + safety margin <= 27 °C.
4. Among safe plans pick the lowest predicted energy (a tiny cost per setpoint change avoids
   needless wiggling). Apply only the FIRST step, then re-plan 15 minutes later with fresh data.
5. Feedback: 15 minutes later we compare the forecast with what the sensors really show and
   carry that error ("bias") into the next forecasts. This keeps the plan honest if the room
   drifts away from the data the model was trained on.
"""
import itertools

import numpy as np
import pandas as pd

from . import config as C
from .forecast import Forecaster

PLANS = np.array(list(itertools.product(C.MOVES, repeat=C.HORIZON)))      # (81, 4) moves in °C


class MPC:
    def __init__(self, forecaster: Forecaster, margins, limit=C.INLET_LIMIT, move_cost_kwh=C.MOVE_COST_KWH):
        self.forecaster = forecaster
        self.margins = np.asarray(margins, dtype=float)
        self.limit = limit
        self.move_cost_kwh = move_cost_kwh

    def evaluate_plans(self, state: pd.Series, current_sp: float, bias: float = 0.0) -> pd.DataFrame:
        setpoints = current_sp + np.cumsum(PLANS, axis=1)
        within = ((setpoints >= C.SP_MIN) & (setpoints <= C.SP_MAX)).all(axis=1)
        pred = self.forecaster.predict_plans(state, setpoints)
        nominal, cool, it = pred["max_inlet_c"], pred["cooling_kw"], pred["it_power_kw"]
        temp = nominal + bias
        guarded = temp + self.margins[None, :]
        energy = (cool + it).sum(axis=1) * C.DT_H
        moves = np.abs(PLANS).sum(axis=1)
        table = pd.DataFrame({
            "plan": [" ".join(f"{s:g}" for s in row) for row in setpoints],
            "first_sp": setpoints[:, 0], "energy_kwh": energy, "cooling_kwh": cool.sum(axis=1) * C.DT_H,
            "it_kwh": it.sum(axis=1) * C.DT_H, "peak_inlet_c": temp.max(axis=1), "peak_guarded_c": guarded.max(axis=1),
            "moves": moves, "within_limits": within, "nominal_1": nominal[:, 0],
        })
        table["safe"] = within & (table["peak_guarded_c"] <= self.limit)
        table["cost"] = table["energy_kwh"] + self.move_cost_kwh * moves
        for k in range(C.HORIZON):
            table[f"sp_{k + 1}"] = setpoints[:, k]
            table[f"inlet_{k + 1}"] = temp[:, k]
        return table

    def decide(self, state: pd.Series, current_sp: float, bias: float = 0.0):
        table = self.evaluate_plans(state, current_sp, bias)
        safe = table[table["safe"]]
        if len(safe):
            best = safe["cost"].idxmin()
            reason = "cheapest_safe_plan"
        else:   # nothing is predicted safe: take the plan that keeps the racks coolest
            candidates = table[table["within_limits"]]
            best = candidates["peak_guarded_c"].idxmin()
            reason = "no_safe_plan_max_cooling"
        table["chosen"] = table.index == best
        return float(table.loc[best, "first_sp"]), reason, table
