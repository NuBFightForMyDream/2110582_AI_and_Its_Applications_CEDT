"""Step 2 — clean the sensor log before anything learns from it.

Four checks, each counted in a small quality report:
1. logger gaps   -> put every reading back on the 15-minute grid
2. impossible    -> values outside the physical range become missing
3. stuck sensor  -> the same exact value for 6+ readings becomes missing
4. fill          -> short holes (<= 1 hour) are interpolated; longer holes stay missing
"""
import numpy as np
import pandas as pd

from . import config as C

VALID_RANGE = {
    **{col: (5.0, 45.0) for col in C.INLET_COLS},
    "supply_air_c": (5.0, 40.0), "return_air_c": (5.0, 60.0), "outdoor_temp_c": (0.0, 50.0),
    "it_load_pct": (0.0, 100.0), "it_power_kw": (0.0, 500.0), "crah_fan_pct": (0.0, 100.0),
    "chiller_kw": (0.0, 300.0), "crah_fan_kw": (0.0, 100.0), "pump_tower_kw": (0.0, 100.0),
    "cooling_kw": (0.0, 400.0), "facility_kw": (0.0, 1000.0), "pue": (1.0, 3.0),
    "crah_setpoint_c": (C.SP_MIN, C.SP_MAX),
}
STUCK_READINGS = 6
FILL_LIMIT = 4   # readings (= 1 hour)


def clean(raw: pd.DataFrame):
    """Return (clean frame on the full grid, quality report frame)."""
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq=f"{C.STEP_MINUTES}min")
    frame = raw.reindex(grid)
    frame.index.name = "timestamp"
    gap_rows = int(len(grid) - len(raw))
    report = {col: {"missing": int(raw[col].isna().sum()), "impossible": 0, "stuck": 0} for col in raw.columns}

    for col, (low, high) in VALID_RANGE.items():
        bad = (frame[col] < low) | (frame[col] > high)
        report[col]["impossible"] = int(bad.sum())
        frame.loc[bad, col] = np.nan

    for col in C.INLET_COLS:
        same = frame[col].diff().eq(0)
        run_id = (~same).cumsum()
        run_len = same.groupby(run_id).transform("sum")
        stuck = run_len >= STUCK_READINGS - 1          # the whole run, including its first reading
        report[col]["stuck"] = int(stuck.sum())
        frame.loc[stuck, col] = np.nan

    before = frame.isna()
    frame = frame.interpolate(method="time", limit=FILL_LIMIT, limit_area="inside")
    for col in raw.columns:
        report[col]["filled"] = int((before[col] & frame[col].notna()).sum())
        report[col]["still_missing"] = int(frame[col].isna().sum())

    frame["max_inlet_c"] = frame[C.INLET_COLS].max(axis=1, skipna=False)
    frame["mean_inlet_c"] = frame[C.INLET_COLS].mean(axis=1, skipna=False)
    inlets = frame[C.INLET_COLS].to_numpy()
    hottest = np.array(C.RACKS, dtype=object)[np.nanargmax(np.nan_to_num(inlets, nan=-np.inf), axis=1)]
    frame["hottest_rack"] = np.where(frame["max_inlet_c"].notna(), hottest, None)
    quality = pd.DataFrame(report).T
    quality.index.name = "sensor"
    quality.attrs["gap_rows"] = gap_rows
    return frame, quality


def split(frame):
    """Chronological split: train (weeks 1-9), validation (10-11), test (12-13)."""
    train = frame.loc[:pd.Timestamp(C.TRAIN_END) - pd.Timedelta(minutes=1)]
    val = frame.loc[C.TRAIN_END:pd.Timestamp(C.VAL_END) - pd.Timedelta(minutes=1)]
    test = frame.loc[C.VAL_END:]
    return train, val, test
