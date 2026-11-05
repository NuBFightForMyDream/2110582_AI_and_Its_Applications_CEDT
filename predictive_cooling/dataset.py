"""Step 1 — mock sensor history (what a BMS/DCIM would export every 15 minutes).

`python -m predictive_cooling.dataset` writes data/sensor_history.csv:
13 weeks x 96 readings/day of rack inlet temperatures, power meters, CRAH status and weather.

How the history is made:
1. Workload: office-hour demand, a nightly GPU training batch at 01:00 on weekdays,
   an afternoon reporting batch, plus random unplanned jobs.
2. Weather: hot humid climate, daily cycle, slow warming toward April, afternoon storms.
3. The plant is run by a "historical operator": a simple rule plus occasional setpoint
   step tests (so the data shows what happens at many different setpoints).
4. Real logs are messy, so we add missing values, glitch spikes, a stuck sensor and logger gaps.
"""
import numpy as np
import pandas as pd

from . import config as C
from .plant import Plant, PlantParams

GPU_IDLE = 0.18                     # GPU racks are nearly idle outside their jobs
GPU = [C.RACKS.index(r) for r in ("B4", "B5", "B6")]
COMPUTE = [C.RACKS.index(r) for r in C.ROWS["A"]]


def make_exogenous(start=C.START, days=C.DAYS, seed=C.DATASET_SEED, surges=()):
    """Per-rack CPU utilisation (0-1) and outdoor temperature on the 15-minute grid.

    `surges` adds unplanned events: iterable of (timestamp, hours, extra_util) on the GPU racks.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * C.STEPS_PER_DAY, freq=f"{C.STEP_MINUTES}min")
    n = len(idx)
    hour = np.asarray(idx.hour + idx.minute / 60, dtype=float)
    weekend = np.asarray(idx.day_of_week >= 5)
    day = np.arange(n) // C.STEPS_PER_DAY

    office = 0.36 + 0.24 * np.clip(np.sin(np.pi * (hour - 7) / 14), 0, None)
    office = office * np.where(weekend, 0.8, 1.0)
    common = _ar1(rng, n, 0.97, 0.012)
    util = np.repeat(office[:, None], len(C.RACKS), axis=1) + common[:, None]
    util += np.stack([_ar1(rng, n, 0.9, 0.02) for _ in C.RACKS], axis=1)
    util[:, GPU] = GPU_IDLE + 0.5 * (util[:, GPU] - 0.36) + 0.03 * rng.standard_normal((n, len(GPU)))

    batch = (~weekend) & (hour >= 1) & (hour < 5)                 # nightly GPU training job
    util[np.ix_(batch, GPU)] = 0.93 + 0.02 * rng.standard_normal((int(batch.sum()), len(GPU)))
    report = (~weekend) & (hour >= 13) & (hour < 15)              # afternoon reporting job
    util[np.ix_(report, COMPUTE)] += 0.18

    for d in range(days):                                          # random unplanned jobs
        for _ in range(rng.poisson(0.5)):
            begin = d * C.STEPS_PER_DAY + rng.integers(0, C.STEPS_PER_DAY)
            length = rng.integers(4, 13)
            racks = GPU if rng.random() < 0.5 else COMPUTE
            util[begin:begin + length, racks] += rng.uniform(0.15, 0.35)
    for when, hours, extra in surges:
        begin = idx.get_indexer([pd.Timestamp(when)])[0]
        util[begin:begin + int(hours * 4), GPU] += extra
    util = np.clip(util, 0.05, 1.0)

    season = 27.6 + 3.0 * day / max(days - 1, 1)
    daily = _ar1(rng, days, 0.7, 0.8)[day]
    outdoor = season + daily + 3.6 * np.cos(2 * np.pi * (hour - 15) / 24) + 0.2 * rng.standard_normal(n)
    for d in range(days):                                          # afternoon thunderstorms
        if rng.random() < 0.08 + 0.25 * d / days:
            at = d * C.STEPS_PER_DAY + 60 + rng.integers(0, 12)    # around 15:00-18:00
            k = np.arange(n - at)
            outdoor[at:] -= 4.5 * np.exp(-k / 8.0)
    frame = pd.DataFrame(util, index=idx, columns=[f"util_{r}" for r in C.RACKS])
    frame["outdoor_c"] = outdoor
    frame.index.name = "timestamp"
    return frame


def _ar1(rng, n, phi, sd):
    x = np.zeros(n)
    noise = rng.standard_normal(n) * sd
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


class HistoricalOperator:
    """How the room was run before the AI: a deadband rule plus occasional step tests."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.rule_sp = 20.0
        self.test_offset, self.test_left = 0.0, 0

    def decide(self, reading):
        hot = reading["max_inlet_c"]
        if hot > 26.0:
            self.rule_sp -= 1.0
        elif hot < 24.5:
            self.rule_sp += 0.5
        self.rule_sp = float(np.clip(self.rule_sp, 18.0, 23.0))
        if self.test_left == 0 and self.rng.random() < 1 / 20:
            self.test_offset = float(self.rng.choice([-3, -2, -1, 1, 2]))
            self.test_left = int(self.rng.integers(3, 13))
        if self.test_left:
            self.test_left -= 1
            if hot > 27.3:
                self.test_left = 0
        offset = self.test_offset if self.test_left else 0.0
        return float(np.clip(round(self.rule_sp + offset), C.SP_MIN, C.SP_MAX))


def simulate_history(exo, params=PlantParams(), seed=C.DATASET_SEED):
    plant = Plant(params, seed=seed)
    operator = HistoricalOperator(seed + 1)
    utils = exo[[f"util_{r}" for r in C.RACKS]].to_numpy()
    outdoor = exo["outdoor_c"].to_numpy()
    setpoint = 20.0
    _, reading = plant.reset(setpoint, utils[0], outdoor[0])
    rows = []
    for t in range(len(exo)):
        if t:
            _, reading = plant.step(setpoint, utils[t], outdoor[t])
        setpoint = operator.decide(reading)
        rows.append({**reading, "crah_setpoint_c": setpoint})
    frame = pd.DataFrame(rows, index=exo.index)
    frame["pue"] = frame["facility_kw"] / frame["it_power_kw"]
    return frame


COLUMNS = (["outdoor_temp_c", "it_load_pct", "it_power_kw"] + C.INLET_COLS +
           ["supply_air_c", "return_air_c", "crah_setpoint_c", "crah_fan_pct",
            "chiller_kw", "crah_fan_kw", "pump_tower_kw", "cooling_kw", "facility_kw", "pue"])


def add_log_faults(frame, seed=C.DATASET_SEED):
    """Make the log look like a real export: gaps, glitches, a stuck sensor, missing cells."""
    rng = np.random.default_rng(seed + 2)
    out = frame[COLUMNS].copy()
    n = len(out)
    sensor_cols = [c for c in COLUMNS if c != "crah_setpoint_c"]
    for col in sensor_cols:                                         # ~0.3 % missing cells
        out.loc[out.index[rng.random(n) < 0.003], col] = np.nan
    for _ in range(14):                                             # glitch spikes (bad packets)
        at = out.index[rng.integers(0, n)]
        out.loc[at, rng.choice(C.INLET_COLS)] = float(rng.choice([85.0, -40.0, 0.0]))
    stuck_at = int(rng.integers(20, 50)) * C.STEPS_PER_DAY + 40     # inlet_A3 freezes for 8 h
    out.iloc[stuck_at:stuck_at + 32, out.columns.get_loc("inlet_A3_c")] = out["inlet_A3_c"].iloc[stuck_at]
    drop = []
    for day, steps in [(int(rng.integers(5, 30)), 6), (int(rng.integers(35, 60)), 12)]:   # logger outages
        begin = day * C.STEPS_PER_DAY + int(rng.integers(0, 80))
        drop.extend(out.index[begin:begin + steps])
    out = out.drop(index=drop)
    return out.round(3)


def build(path=C.DATA_PATH):
    exo = make_exogenous()
    history = simulate_history(exo)
    log = add_log_faults(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index_label="timestamp")
    return log


def load(path=C.DATA_PATH):
    if not path.exists():
        build(path)
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


if __name__ == "__main__":
    data = build()
    print(f"Wrote {len(data):,} rows x {data.shape[1]} sensor columns to {C.DATA_PATH.relative_to(C.ROOT)}")
