"""Fast checks for every pipeline step (no full 2-week MPC runs; those live in the pipeline)."""
import math

import numpy as np
import pandas as pd
import pytest

from predictive_cooling import config as C
from predictive_cooling import dataset, forecast, preprocess
from predictive_cooling.controllers import AIController, FixedSetpoint, RuleBased, live_state
from predictive_cooling.evaluate import Scenario, run, summarize
from predictive_cooling.mpc import MPC, PLANS
from predictive_cooling.plant import Plant, PlantParams
from predictive_cooling.safety import SafetyFilter


@pytest.fixture(scope="module")
def data():
    raw = dataset.load()
    clean, quality = preprocess.clean(raw)
    train, val, test = preprocess.split(clean)
    return raw, clean, quality, train, val, test


@pytest.fixture(scope="module")
def gbm(data):
    _, _, _, train, _, _ = data
    return forecast.Forecaster("gbm").fit(train)


# 6 · plant ---------------------------------------------------------------------------------
def test_plant_physics_trade_off():
    util = np.full(12, 0.5)
    cool, warm = Plant(PlantParams(), seed=0), Plant(PlantParams(), seed=0)
    t_cool, _ = cool.reset(18, util, 30)
    t_warm, _ = warm.reset(24, util, 30)
    assert t_warm["max_inlet_c"] > t_cool["max_inlet_c"]                 # warmer air -> warmer racks
    assert t_warm["chiller_kw"] < t_cool["chiller_kw"]                   # ... but a cheaper chiller
    assert t_warm["server_fan_kw"] > t_cool["server_fan_kw"]             # ... and busier server fans
    assert t_cool["facility_kw"] == pytest.approx(t_cool["it_power_kw"] + t_cool["cooling_kw"] + t_cool["other_kw"])
    assert max(t_cool[c] for c in C.INLET_COLS) == t_cool["inlet_B6_c"]   # the GPU rack is the hot spot


def test_plant_has_thermal_lag():
    plant = Plant(PlantParams(), seed=0)
    util = np.full(12, 0.3)
    start, _ = plant.reset(20, util, 30)
    first, _ = plant.step(20, np.full(12, 0.9), 30)
    for _ in range(12):
        settled, _ = plant.step(20, np.full(12, 0.9), 30)
    assert start["max_inlet_c"] < first["max_inlet_c"] < settled["max_inlet_c"]


# 1 · dataset -------------------------------------------------------------------------------
def test_dataset_shape_and_columns(data):
    raw = data[0]
    assert set(dataset.COLUMNS) == set(raw.columns)
    assert len(raw) > 0.99 * C.DAYS * C.STEPS_PER_DAY           # a few logger gaps were removed on purpose
    assert raw["crah_setpoint_c"].nunique() >= 6                # step tests give the model many setpoints to learn from
    assert raw["it_power_kw"].between(50, 150).mean() > 0.99


def test_exogenous_is_reproducible_and_surges_only_add_load():
    a, b = dataset.make_exogenous(days=7), dataset.make_exogenous(days=7)
    pd.testing.assert_frame_equal(a, b)
    surged = dataset.make_exogenous(days=7, surges=[("2026-02-03 10:00", 2, 0.4)])
    diff = surged.filter(like="util_B6").to_numpy() - a.filter(like="util_B6").to_numpy()
    assert (diff >= 0).all() and diff.max() > 0.1
    assert np.allclose(surged["outdoor_c"], a["outdoor_c"])


# 2 · cleaning ------------------------------------------------------------------------------
def test_cleaning_removes_every_kind_of_fault(data):
    raw, clean, quality, *_ = data
    assert len(clean) == C.DAYS * C.STEPS_PER_DAY                # back on the full grid
    assert quality.attrs["gap_rows"] == len(clean) - len(raw)
    assert quality.loc["inlet_A3_c", "stuck"] >= 30              # the 8-hour frozen sensor
    assert quality["impossible"].sum() >= 10
    assert clean[C.INLET_COLS].max().max() < 40 and clean[C.INLET_COLS].min().min() > 10


def test_chronological_split(data):
    *_, train, val, test = data
    assert train.index.max() < val.index.min() and val.index.max() < test.index.min()
    assert len(test) == 14 * C.STEPS_PER_DAY


# 3 · forecast ------------------------------------------------------------------------------
def test_training_table_alignment(data):
    _, clean, *_ = data
    x, y = forecast.training_table(clean, 2)
    t = x.index[100]
    assert x.loc[t, "sp_1"] == clean.loc[t, "crah_setpoint_c"]
    assert x.loc[t, "sp_2"] == clean.loc[t + pd.Timedelta(minutes=15), "crah_setpoint_c"]
    assert y.loc[t, "max_inlet_c"] == clean.loc[t + pd.Timedelta(minutes=30), "max_inlet_c"]


def test_ml_beats_persistence_and_meets_targets(data, gbm):
    *_, test = data
    models = {"persistence": forecast.Forecaster("persistence").fit(data[3]), "gbm": gbm}
    acc = forecast.evaluate(models, test).set_index(["kind", "horizon_min"])
    for h in (15, 30, 45, 60):
        assert acc.loc[("gbm", h), "inlet_mae_c"] < acc.loc[("persistence", h), "inlet_mae_c"]
    assert acc.loc[("gbm", 15), "inlet_mae_c"] <= 1.0                     # proposal target
    assert acc.xs("gbm")[["cooling_nrmse_pct", "it_nrmse_pct"]].max().max() <= 10.0


def test_forecast_respects_physics(data, gbm):
    *_, test = data
    state = forecast.state_features(test).dropna().iloc[200]
    plans = np.array([[sp] * C.HORIZON for sp in (19.0, 21.0, 23.0)])
    pred = gbm.predict_plans(state, plans)
    assert (np.diff(pred["max_inlet_c"][:, -1]) >= 0).all()          # warmer setpoint -> not cooler racks
    assert (np.diff(pred["cooling_kw"][:, -1]) <= 0).all()           # warmer setpoint -> not more cooling power


# 4 · MPC -----------------------------------------------------------------------------------
def test_81_plans_and_constraint(data, gbm):
    assert PLANS.shape == (81, C.HORIZON)
    *_, test = data
    state = forecast.state_features(test).dropna().iloc[300]
    mpc = MPC(gbm, [0.5] * C.HORIZON)
    sp, reason, table = mpc.decide(state, float(state["setpoint_prev_c"]))
    assert len(table) == 81 and table["chosen"].sum() == 1
    assert abs(sp - state["setpoint_prev_c"]) <= 1
    chosen = table[table["chosen"]].iloc[0]
    if reason == "cheapest_safe_plan":
        assert chosen["peak_guarded_c"] <= C.INLET_LIMIT
        assert chosen["cost"] == table[table["safe"]]["cost"].min()


def test_bias_makes_mpc_more_careful(data, gbm):
    *_, test = data
    state = forecast.state_features(test).dropna().iloc[300]
    mpc = MPC(gbm, [0.5] * C.HORIZON)
    normal = mpc.evaluate_plans(state, 21.0)["safe"].sum()
    warned = mpc.evaluate_plans(state, 21.0, bias=1.5)["safe"].sum()
    assert warned <= normal


# 5 · safety --------------------------------------------------------------------------------
def _reading(value=24.0, **override):
    reading = {c: value for c in C.INLET_COLS}
    reading.update(override)
    return reading


def test_safety_rules():
    safety = SafetyFilter(fallback_sp=20)
    assert safety.check(22, 21, _reading()) == (22.0, [])
    sp, fired = safety.check(23, 21, _reading())
    assert sp == 22 and fired == ["R4_rate_or_range_limit"]
    sp, fired = safety.check(23, 22, _reading(inlet_B6_c=math.nan))
    assert sp == 21 and fired[0] == "R1_missing_data"             # heads to the fallback, 1 °C per step
    sp, fired = safety.check(22, 22, _reading(inlet_B6_c=26.8))
    assert sp == 20 and "R3_too_hot_now" in fired                  # emergency: cool by 2 °C at once
    sp, fired = safety.check(23, 22, _reading(), unfamiliar=["outdoor_temp_c"])
    assert sp == 21 and fired[0] == "R2_unfamiliar_conditions"
    assert safety.check(30, 24, _reading())[0] == C.SP_MAX


def test_live_state_matches_training_features(data):
    _, clean, *_ = data
    frame = clean.loc["2026-04-21 00:00":"2026-04-21 02:00"]
    expected = forecast.state_features(frame).iloc[-1]
    prev = {**frame.iloc[-2].to_dict(), "timestamp": frame.index[-2]}
    cur = {**frame.iloc[-1].to_dict(), "timestamp": frame.index[-1]}
    got = live_state(prev, cur, frame["crah_setpoint_c"].iloc[-2])
    pd.testing.assert_series_equal(got, expected[forecast.STATE_FEATURES], check_names=False, atol=1e-9)


# closed loop -------------------------------------------------------------------------------
def test_closed_loop_is_fair_and_safe(data, gbm):
    _, _, _, _, val, _ = data
    margins = forecast.safety_margins(gbm, val)
    scenario = Scenario("short", "Short", "1 day")
    fixed = run(FixedSetpoint(20), scenario, days=1)
    rule = run(RuleBased(), scenario, days=1)
    ai = run(AIController(MPC(gbm, margins), SafetyFilter(20)), scenario, days=1)
    assert np.allclose(fixed["it_load_pct"], ai["it_load_pct"])           # identical workload
    assert np.allclose(fixed["outdoor_temp_c"], rule["outdoor_temp_c"])   # identical weather
    s = summarize(ai, "ai", True)
    assert s["violation_minutes"] == 0
    assert s["mean_setpoint_c"] > summarize(fixed, "fixed", False)["mean_setpoint_c"]
    assert s["decision_ms_max"] < 10_000                                   # proposal: < 10 s per decision
    assert (ai["command"].diff().abs().dropna() <= 2).all()


def test_sensor_failure_falls_back(data, gbm):
    _, _, _, _, val, _ = data
    scenario = Scenario("f", "Fault", "B5/B6 offline", sensor_faults=(("2026-04-20 01:00", 2, ("B5", "B6")),))
    log = run(AIController(MPC(gbm, forecast.safety_margins(gbm, val)), SafetyFilter(20)), scenario, days=1)
    window = log.loc["2026-04-20 01:00":"2026-04-20 02:45"]
    assert (window["mode"] == "Fallback").all()
    assert log["max_inlet_c"].max() <= C.INLET_LIMIT
