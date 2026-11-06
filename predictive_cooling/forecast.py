"""Step 3 — predict the next hour from sensors + a candidate setpoint plan.

Question the model answers:
    "If we use setpoints [s1, s2, s3, s4] for the next 4 x 15 min, what will the hottest
     rack inlet, the cooling power and the IT power be at +15, +30, +45 and +60 minutes?"

Three forecasters are compared on the same chronological test weeks:
* Persistence      — "the next hour looks like right now" (the baseline the proposal asks for)
* Linear (Ridge)   — straight-line relationships only
* Gradient boosting — non-linear trees; physically sensible monotonic constraints
                      (warmer setpoint -> warmer racks, less chiller power, more server-fan power)
One model is trained per target and per horizon (3 targets x 4 horizons).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

STATE_FEATURES = ["max_inlet_c", "mean_inlet_c", "inlet_trend_c", "supply_air_c", "setpoint_prev_c",
                  "it_power_kw", "it_trend_kw", "cooling_kw", "outdoor_temp_c", "hour", "hour_sin", "hour_cos", "weekend"]
TARGETS = {"max_inlet_c": "Hottest rack inlet (°C)", "cooling_kw": "Cooling power (kW)", "it_power_kw": "IT power (kW)"}
# +1: target rises when the feature rises, -1: target falls, 0: learn freely.
MONOTONIC = {
    "max_inlet_c": {"max_inlet_c": 1, "mean_inlet_c": 1, "supply_air_c": 1, "setpoint_prev_c": 1, "plan": 1},
    "cooling_kw": {"supply_air_c": -1, "setpoint_prev_c": -1, "it_power_kw": 1, "outdoor_temp_c": 1, "plan": -1},
    "it_power_kw": {"plan": 1},
}
FEATURE_LABELS = {
    "max_inlet_c": "Hottest inlet now", "mean_inlet_c": "Average inlet now", "inlet_trend_c": "Inlet change last 15 min",
    "supply_air_c": "Supply air now", "setpoint_prev_c": "Setpoint in use", "it_power_kw": "IT power now",
    "it_trend_kw": "IT power change last 15 min", "cooling_kw": "Cooling power now", "outdoor_temp_c": "Outdoor temperature", "hour": "Time of day",
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)", "weekend": "Weekend",
    **{f"sp_{k}": f"Planned setpoint +{15 * (k - 1)} min" for k in range(1, C.HORIZON + 1)},
}


def plan_cols(h):
    return [f"sp_{k}" for k in range(1, h + 1)]


def state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a clean sensor log into one feature row per timestamp."""
    hour = np.asarray(frame.index.hour + frame.index.minute / 60, dtype=float)
    out = pd.DataFrame(index=frame.index)
    out["max_inlet_c"] = frame["max_inlet_c"]
    out["mean_inlet_c"] = frame["mean_inlet_c"]
    out["inlet_trend_c"] = frame["max_inlet_c"].diff()
    out["supply_air_c"] = frame["supply_air_c"]
    out["setpoint_prev_c"] = frame["crah_setpoint_c"].shift(1)
    out["it_power_kw"] = frame["it_power_kw"]
    out["it_trend_kw"] = frame["it_power_kw"].diff()
    out["cooling_kw"] = frame["cooling_kw"]
    out["outdoor_temp_c"] = frame["outdoor_temp_c"]
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekend"] = np.asarray(frame.index.day_of_week >= 5, dtype=float)
    return out


def training_table(frame: pd.DataFrame, h: int):
    """Features at time t + the setpoints actually used at t..t+h-1  ->  targets at t+h."""
    x = state_features(frame)
    for k in range(1, h + 1):
        x[f"sp_{k}"] = frame["crah_setpoint_c"].shift(-(k - 1))
    y = pd.DataFrame({t: frame[t].shift(-h) for t in TARGETS}, index=frame.index)
    ok = x.notna().all(axis=1) & y.notna().all(axis=1)
    return x[ok], y[ok]


class Forecaster:
    KINDS = {"persistence": "Persistence (no AI)", "linear": "Linear regression", "gbm": "Gradient boosting"}

    def __init__(self, kind="gbm"):
        self.kind = kind
        self.models = {}
        self.ranges = {}

    def _make(self, target, columns):
        if self.kind == "linear":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        rules = MONOTONIC[target]
        cst = [rules.get("plan" if c.startswith("sp_") else c, 0) for c in columns]
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=25,
                                             l2_regularization=1.0, monotonic_cst=cst, random_state=0)

    def fit(self, frame):
        for h in range(1, C.HORIZON + 1):
            x, y = training_table(frame, h)
            if h == 1:
                self.ranges = {c: (float(x[c].min()), float(x[c].max())) for c in STATE_FEATURES}
            if self.kind == "persistence":
                continue
            for target in TARGETS:
                self.models[target, h] = self._make(target, list(x.columns)).fit(x.to_numpy(), y[target].to_numpy())
        return self

    def predict_table(self, x: pd.DataFrame, h: int) -> pd.DataFrame:
        """Predictions for rows that already contain sp_1..sp_h."""
        if self.kind == "persistence":
            return pd.DataFrame({t: x[t].to_numpy() for t in TARGETS}, index=x.index)
        values = x[STATE_FEATURES + plan_cols(h)].to_numpy(dtype=float)
        return pd.DataFrame({t: self.models[t, h].predict(values) for t in TARGETS}, index=x.index)

    def predict_plans(self, state: pd.Series, setpoints: np.ndarray) -> dict:
        """state: one row of state_features; setpoints: (n_plans, HORIZON). Returns target -> (n_plans, HORIZON)."""
        n = len(setpoints)
        now = state[STATE_FEATURES].to_numpy(dtype=float)
        if self.kind == "persistence":
            return {t: np.full((n, C.HORIZON), float(state[t])) for t in TARGETS}
        base = np.repeat(now[None, :], n, axis=0)
        out = {t: np.empty((n, C.HORIZON)) for t in TARGETS}
        for h in range(1, C.HORIZON + 1):
            x = np.hstack([base, setpoints[:, :h]])
            for t in TARGETS:
                out[t][:, h - 1] = self.models[t, h].predict(x)
        return out

    def out_of_range(self, state: pd.Series, tolerance=0.1):
        """Features that are far outside what the model saw in training (the model would be guessing)."""
        flags = []
        for c in ["max_inlet_c", "it_power_kw", "it_trend_kw", "outdoor_temp_c", "supply_air_c"]:
            low, high = self.ranges[c]
            pad = tolerance * (high - low)
            if not low - pad <= state[c] <= high + pad:
                flags.append(c)
        return flags


def evaluate(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model and horizon: MAE for temperature, NRMSE (% of mean) for power."""
    rows = []
    for h in range(1, C.HORIZON + 1):
        x, y = training_table(frame, h)
        for name, model in models.items():
            pred = model.predict_table(x, h)
            row = {"model": Forecaster.KINDS[name], "kind": name, "horizon_min": 15 * h,
                   "inlet_mae_c": float((pred["max_inlet_c"] - y["max_inlet_c"]).abs().mean())}
            for t in ["cooling_kw", "it_power_kw"]:
                rmse = float(np.sqrt(((pred[t] - y[t]) ** 2).mean()))
                row[f"{t.split('_')[0]}_nrmse_pct"] = 100 * rmse / float(y[t].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def safety_margins(model: Forecaster, frame: pd.DataFrame, q=C.MARGIN_QUANTILE) -> list:
    """Per horizon: the model under-predicts the hottest inlet by less than this in 99 % of cases (°C)."""
    margins = []
    for h in range(1, C.HORIZON + 1):
        x, y = training_table(frame, h)
        under = y["max_inlet_c"] - model.predict_table(x, h)["max_inlet_c"]
        margins.append(max(0.0, float(under.quantile(q))))
    return margins
