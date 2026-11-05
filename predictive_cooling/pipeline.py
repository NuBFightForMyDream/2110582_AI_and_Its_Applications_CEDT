"""Run the whole study end to end and save what the dashboard shows.

    python -m predictive_cooling.pipeline          # about 1-3 minutes

1 sensors -> 2 clean -> 3 train + score forecasters -> tune baselines on validation weeks
-> 4/5 run Fixed, Rule-based and AI + MPC (+ safety) on the plant (6) in four test scenarios.
"""
from concurrent.futures import ProcessPoolExecutor
import itertools
import os
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from . import config as C
from . import dataset, forecast, preprocess
from .controllers import AIController, FixedSetpoint, RuleBased
from .evaluate import run, scenarios, summarize
from .mpc import MPC
from .safety import SafetyFilter

VERSION = 1
BUNDLE = C.ARTIFACT_DIR / "bundle.pkl"
CONTROLLER_ORDER = ["fixed", "rule", "ai", "ai_unsafe"]


def _simulate(job):
    controller, scenario_key, start = job
    with threadpool_limits(1):
        return run(controller, scenarios()[scenario_key], start=start)


def _run_all(jobs, workers):
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(_simulate, jobs))
    return [_simulate(job) for job in jobs]


def tune_fixed(workers):
    """Highest constant setpoint with zero overheating in the validation weeks."""
    candidates = [float(sp) for sp in range(int(C.SP_MAX), int(C.SP_MIN) - 1, -1)]
    logs = _run_all([(FixedSetpoint(sp), "normal", C.TRAIN_END) for sp in candidates], workers)
    table = pd.DataFrame([summarize(log, f"{sp:g} °C", False) | {"setpoint": sp} for sp, log in zip(candidates, logs)])
    safe = table[table["violation_minutes"] == 0]
    return float(safe["setpoint"].max()), table


def tune_rule(workers):
    """Deadband rule with the lowest energy and zero overheating in the validation weeks."""
    grid = [dict(upper=u, lower=u - gap, step_up=up) for u, gap, up in
            itertools.product([25.0, 25.5, 26.0, 26.5], [1.0, 1.5, 2.0], [0.5, 1.0])]
    logs = _run_all([(RuleBased(**g), "normal", C.TRAIN_END) for g in grid], workers)
    table = pd.DataFrame([summarize(log, RuleBased(**g).describe(), False) | g for g, log in zip(grid, logs)])
    safe = table[table["violation_minutes"] == 0].sort_values("facility_kwh")
    best = safe.iloc[0]
    return {k: float(best[k]) for k in ["upper", "lower", "step_up"]}, table


def _importance(model, frame):
    out = {}
    for h in (1, C.HORIZON):
        x, y = forecast.training_table(frame, h)
        cols = forecast.STATE_FEATURES + forecast.plan_cols(h)
        result = permutation_importance(model.models["max_inlet_c", h], x[cols].to_numpy(), y["max_inlet_c"].to_numpy(),
                                        scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
        out[h] = pd.DataFrame({"feature": [forecast.FEATURE_LABELS[c] for c in cols],
                               "importance": result.importances_mean}).sort_values("importance", ascending=False)
    return out


def _prediction_samples(models, frame):
    samples = {}
    for h in range(1, C.HORIZON + 1):
        x, y = forecast.training_table(frame, h)
        table = pd.DataFrame(index=x.index + pd.Timedelta(minutes=C.STEP_MINUTES * h))
        table.index.name = "target_time"
        for target in forecast.TARGETS:
            table[f"actual_{target}"] = y[target].to_numpy()
        for name, model in models.items():
            pred = model.predict_table(x, h)
            for target in forecast.TARGETS:
                table[f"{name}_{target}"] = pred[target].to_numpy()
        samples[h] = table
    return samples


def build(workers=None, log=print):
    started = time.time()
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    log("1/5 Loading the sensor history (generating it if missing)")
    raw = dataset.load()
    log("2/5 Cleaning sensor data")
    clean, quality = preprocess.clean(raw)
    train, val, test = preprocess.split(clean)
    log("3/5 Training forecasters (persistence, linear, gradient boosting)")
    models = {kind: forecast.Forecaster(kind).fit(train) for kind in forecast.Forecaster.KINDS}
    accuracy = forecast.evaluate(models, test)
    gbm = models["gbm"]
    margins = forecast.safety_margins(gbm, val)
    importance = _importance(gbm, test)
    samples = _prediction_samples(models, test)

    log("4/5 Tuning the Fixed and Rule-based baselines on the validation weeks")
    fixed_sp, fixed_table = tune_fixed(workers)
    rule_params, rule_table = tune_rule(workers)

    log("5/5 Closed-loop tests on the plant: 4 scenarios x 3 controllers (+ AI without safety filter)")
    mpc = MPC(gbm, margins)
    controllers = {"fixed": FixedSetpoint(fixed_sp), "rule": RuleBased(**rule_params),
                   "ai": AIController(mpc, SafetyFilter(fixed_sp)),
                   "ai_unsafe": AIController(mpc, SafetyFilter(fixed_sp), use_safety=False)}
    keys = [(s, c) for s in scenarios() for c in ["fixed", "rule", "ai"]] + [("sensor_failure", "ai_unsafe")]
    logs = dict(zip(keys, _run_all([(controllers[c], s, C.VAL_END) for s, c in keys], workers)))
    summary = pd.DataFrame([summarize(logs[s, c], controllers[c].name, controllers[c].kind == "ai") | {"scenario": s, "key": c}
                            for s, c in keys])

    bundle = dict(
        version=VERSION, sklearn=sklearn.__version__, built_seconds=round(time.time() - started, 1),
        raw=raw, clean=clean, quality=quality, accuracy=accuracy, margins=margins, importance=importance,
        samples=samples, forecaster=gbm, fixed_sp=fixed_sp, fixed_table=fixed_table, rule_params=rule_params,
        rule_table=rule_table, rule_text=RuleBased(**rule_params).describe(), logs=logs, summary=summary,
        controller_names={k: v.name for k, v in controllers.items()},
    )
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(BUNDLE, "wb") as fh:
        pickle.dump(bundle, fh)
    accuracy.round(3).to_csv(C.ARTIFACT_DIR / "forecast_accuracy.csv", index=False)
    summary.round(3).to_csv(C.ARTIFACT_DIR / "closed_loop_results.csv", index=False)
    quality.to_csv(C.ARTIFACT_DIR / "data_quality.csv")
    log(f"Done in {bundle['built_seconds']} s -> {BUNDLE.relative_to(C.ROOT)}")
    return bundle


def load(rebuild_if_stale=True, log=print):
    if BUNDLE.exists():
        with open(BUNDLE, "rb") as fh:
            bundle = pickle.load(fh)
        if bundle.get("version") == VERSION and bundle.get("sklearn") == sklearn.__version__:
            return bundle
    if not rebuild_if_stale:
        return None
    return build(log=log)


def headline(bundle, scenario="normal"):
    """The numbers people ask first: savings vs each baseline and overheating."""
    table = bundle["summary"].set_index(["scenario", "key"])
    ai = table.loc[(scenario, "ai")]
    out = {"ai": ai}
    days = len(bundle["logs"][scenario, "ai"]) / C.STEPS_PER_DAY
    for base in ["fixed", "rule"]:
        ref = table.loc[(scenario, base)]
        saved = ref["facility_kwh"] - ai["facility_kwh"]
        out[base] = dict(
            ref=ref, cooling_pct=100 * (1 - ai["cooling_kwh"] / ref["cooling_kwh"]),
            facility_pct=100 * saved / ref["facility_kwh"], facility_kwh=saved,
            yearly_kwh=saved / days * 365, yearly_thb=saved / days * 365 * C.ELECTRICITY_THB_PER_KWH,
        )
    return out


if __name__ == "__main__":
    result = build()
    pd.set_option("display.width", 200)
    cols = ["scenario", "controller", "facility_kwh", "cooling_kwh", "pue", "violation_minutes", "hottest_inlet_c",
            "mean_setpoint_c", "fallback_steps", "decision_ms_mean"]
    print(result["summary"][cols].round(2).to_string(index=False))
    print(result["accuracy"].round(3).to_string(index=False))
    print("margins", np.round(result["margins"], 2), "fixed", result["fixed_sp"], "rule", result["rule_params"])
