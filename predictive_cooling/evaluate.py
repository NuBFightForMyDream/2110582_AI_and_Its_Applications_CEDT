"""Closed-loop tests: every controller runs the plant under identical conditions.

Same workload, same weather, same starting state and the same sensor noise for every controller.
The plant (not the AI's own forecaster) decides energy use and temperatures.
"""
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

from . import config as C
from .dataset import make_exogenous
from .forecast import STATE_FEATURES
from .plant import Plant, PlantParams

TRUE_COLS = ["it_power_kw", "cooling_kw", "chiller_kw", "crah_fan_kw", "pump_tower_kw", "other_kw",
             "facility_kw", "server_fan_kw", "outdoor_temp_c", "it_load_pct", "supply_air_c", "max_inlet_c"]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    story: str
    params: PlantParams = PlantParams()
    surges: tuple = ()
    sensor_faults: tuple = ()          # (start, hours, racks)
    highlight: tuple = field(default=())


def scenarios():
    return {
        "normal": Scenario("normal", "Normal operation",
                           "The two test weeks exactly as forecast: nightly GPU batch, office hours, hot April weather."),
        "sensor_failure": Scenario(
            "sensor_failure", "Sensor failure",
            "The B5/B6 inlet sensors (the hottest racks) stop reporting during the 01:00 GPU batch, "
            "and the whole B row goes silent one afternoon.",
            sensor_faults=(("2026-04-21 00:30", 4, ("B5", "B6")), ("2026-04-28 13:00", 3, tuple(f"B{i}" for i in range(1, 7)))),
            highlight=(("2026-04-21 00:30", 4), ("2026-04-28 13:00", 3))),
        "plant_change": Scenario(
            "plant_change", "Room changed after training",
            "Blanking panels removed (+20 % hot-air recirculation at the GPU racks), chiller 10 % less efficient "
            "and faster room air: the forecaster was trained on a different room.",
            params=PlantParams().mismatch()),
        "load_surge": Scenario(
            "load_surge", "Unplanned load surge",
            "Three surprise jobs push the GPU racks to full power at times the history never showed.",
            surges=(("2026-04-22 10:00", 2, 0.45), ("2026-04-25 20:00", 3, 0.5), ("2026-04-30 16:00", 2, 0.45)),
            highlight=(("2026-04-22 10:00", 2), ("2026-04-25 20:00", 3), ("2026-04-30 16:00", 2))),
    }


def _window(start, days, surges=()):
    exo = make_exogenous(surges=surges)
    begin = exo.index.get_indexer([pd.Timestamp(start)])[0]
    return exo.iloc[begin:begin + days * C.STEPS_PER_DAY]


def run(controller, scenario: Scenario, start=C.VAL_END, days=14, seed=C.TEST_SEED) -> pd.DataFrame:
    exo = _window(start, days, scenario.surges)
    utils = exo[[f"util_{r}" for r in C.RACKS]].to_numpy()
    outdoor = exo["outdoor_c"].to_numpy()
    faults = [(pd.Timestamp(s), pd.Timestamp(s) + pd.Timedelta(hours=h), racks) for s, h, racks in scenario.sensor_faults]
    plant = Plant(scenario.params, seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    sp = controller.initial_sp
    rows, prev = [], None
    for t, ts in enumerate(exo.index):
        truth, reading = plant.reset(sp, utils[t], outdoor[t]) if t == 0 else plant.step(sp, utils[t], outdoor[t])
        reading["timestamp"] = ts
        for begin, end, racks in faults:
            if begin <= ts < end:
                for rack in racks:
                    reading[f"inlet_{rack}_c"] = np.nan
        started = time.perf_counter()
        new_sp, info = controller.decide(prev, reading, sp)
        elapsed_ms = 1000 * (time.perf_counter() - started)
        row = {"timestamp": ts, "setpoint_in_use": sp, "command": new_sp, "mode": info.get("mode"),
               "rules": info.get("rules", ""), "proposed": info.get("proposed", np.nan), "bias_c": info.get("bias", np.nan),
               "decision_ms": elapsed_ms,
               "measured_max_inlet_c": np.nanmax([reading[c] for c in C.INLET_COLS]) if not all(
                   np.isnan(reading[c]) for c in C.INLET_COLS) else np.nan,
               "sensors_missing": int(sum(np.isnan(reading[c]) for c in C.INLET_COLS))}
        row.update({c: truth[c] for c in TRUE_COLS})
        row.update({c: truth[c] for c in C.INLET_COLS})
        if "state" in info:
            row.update({f"x_{c}": info["state"][c] for c in STATE_FEATURES})
        rows.append(row)
        prev, sp = reading, new_sp
    return pd.DataFrame(rows).set_index("timestamp")


def summarize(log: pd.DataFrame, name: str, uses_ai: bool) -> dict:
    energy = lambda col: float(log[col].sum() * C.DT_H)  # noqa: E731
    over = log["max_inlet_c"] - C.INLET_LIMIT
    hours = len(log) * C.DT_H
    overhead = C.AI_SERVER_KW * hours if uses_ai else 0.0
    return {
        "controller": name,
        "facility_kwh": energy("facility_kw") + overhead, "cooling_kwh": energy("cooling_kw"),
        "it_kwh": energy("it_power_kw"), "other_kwh": energy("other_kw"), "ai_overhead_kwh": overhead,
        "pue": (energy("facility_kw") + overhead) / energy("it_power_kw"),
        "violation_minutes": int((over > 0).sum() * C.STEP_MINUTES),
        "degree_hours_over": float(over.clip(lower=0).sum() * C.DT_H),
        "hottest_inlet_c": float(log["max_inlet_c"].max()), "mean_setpoint_c": float(log["setpoint_in_use"].mean()),
        "setpoint_changes": int((log["command"].diff().abs() > 0).sum()),
        "fallback_steps": int((log["mode"] == "Fallback").sum()), "emergency_steps": int((log["mode"] == "Emergency").sum()),
        "decision_ms_mean": float(log["decision_ms"].mean()), "decision_ms_max": float(log["decision_ms"].max()),
    }
