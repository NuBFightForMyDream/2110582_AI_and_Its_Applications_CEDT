"""Step 6 — the data-center plant model (the "judge").

A small grey-box model of one server room: 12 racks, two CRAH units, a chiller, pumps and a
cooling tower. It generates the mock sensor history and later scores every controller.
The forecaster never sees these equations; it only learns from the sensor log.

Main physical effects (all deliberately simple):
* Rack inlet = supply air + recirculated hot air; GPU racks far from the CRAH get the most.
* Inlet temperature reacts with a ~25 minute lag, so a sudden load jump arrives "late".
* Warmer supply air -> higher chiller COP (less cooling power) ...
* ... but inlets above 26 °C make server fans spin faster (more IT power).
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from . import config as C


@dataclass(frozen=True)
class PlantParams:
    capacity_kw: tuple = (9, 9, 9, 10, 10, 10, 10, 10, 10, 16, 18, 18)
    recirc_c: tuple = (1.0, 1.3, 1.6, 2.0, 2.4, 2.8, 1.2, 1.6, 2.0, 3.6, 4.8, 6.0)
    idle_fraction: float = 0.35          # server power at 0 % CPU, as a share of full power
    inlet_tau_min: float = 25.0          # thermal lag of the room air / racks
    supply_tau_min: float = 8.0          # CRAH coil follows the setpoint quickly
    cop_ref: float = 3.6                 # chiller COP at 18 °C supply air and 30 °C outdoor
    cop_gain_per_c: float = 0.04         # +4 % COP per °C warmer supply air
    cop_loss_per_c_outdoor: float = 0.02 # -2 % COP per °C hotter outdoor air
    crah_fan_rated_kw: float = 14.0      # both CRAH fans at 100 % speed
    pump_tower_fraction: float = 0.07    # pumps + tower fan per kW of heat removed
    envelope_kw_per_c: float = 0.4       # heat leaking in through walls
    server_fan_fraction: float = 0.04    # server fans at normal speed, share of rack power
    server_fan_knee_c: float = 26.0      # inlet temperature where server fans start to ramp
    server_fan_gain: float = 0.10        # fan speed +10 % per °C above the knee (power ~ speed^3)
    ups_loss_fraction: float = 0.05
    lighting_kw: float = 2.0
    noise_temp_c: float = 0.15           # sensor noise
    noise_power: float = 0.01            # meter noise (relative)

    def mismatch(self):
        """A plant that changed after training: blanking panels removed + a tired chiller."""
        recirc = tuple(r * 1.2 if i >= 9 else r for i, r in enumerate(self.recirc_c))
        return replace(self, recirc_c=recirc, cop_ref=self.cop_ref * 0.9, inlet_tau_min=20.0)


class Plant:
    def __init__(self, params: PlantParams = PlantParams(), seed: int = 0):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.cap = np.array(params.capacity_kw, dtype=float)
        self.recirc = np.array(params.recirc_c, dtype=float)
        self.a_inlet = 1 - math.exp(-C.STEP_MINUTES / params.inlet_tau_min)
        self.a_supply = 1 - math.exp(-C.STEP_MINUTES / params.supply_tau_min)
        self.supply = None
        self.inlet = None

    def _target_inlet(self, util):
        return self.supply + 0.5 + self.recirc * (0.3 + 0.7 * util)

    def reset(self, setpoint, util, outdoor):
        """Start in steady state at `setpoint`; returns the first reading."""
        self.supply = float(setpoint)
        self.inlet = self._target_inlet(np.asarray(util, dtype=float))
        return self._read(util, outdoor, setpoint)

    def step(self, setpoint, util, outdoor):
        """Hold `setpoint` for one 15-minute interval with the new load/weather; return the reading."""
        util = np.asarray(util, dtype=float)
        self.supply += self.a_supply * (setpoint - self.supply)
        self.inlet = self.inlet + self.a_inlet * (self._target_inlet(util) - self.inlet)
        return self._read(util, outdoor, setpoint)

    def _read(self, util, outdoor, setpoint):
        p = self.p
        util = np.asarray(util, dtype=float)
        server_kw = self.cap * (p.idle_fraction + (1 - p.idle_fraction) * util)
        fan_speed = 1 + p.server_fan_gain * np.clip(self.inlet - p.server_fan_knee_c, 0, None)
        server_fan_kw = p.server_fan_fraction * self.cap * fan_speed ** 3
        it_kw = float(server_kw.sum() + server_fan_kw.sum())
        design_it = float(self.cap.sum())
        crah_speed = min(1.0, max(0.4, 0.4 + 0.6 * it_kw / design_it))
        heat_kw = it_kw + p.envelope_kw_per_c * (outdoor - 24.0)
        cop = p.cop_ref * (1 + p.cop_gain_per_c * (self.supply - 18.0)) * (1 - p.cop_loss_per_c_outdoor * (outdoor - 30.0))
        cop = min(8.0, max(1.5, cop))
        chiller_kw = heat_kw / cop
        crah_fan_kw = p.crah_fan_rated_kw * crah_speed ** 3
        pump_tower_kw = p.pump_tower_fraction * heat_kw * (1 + 0.02 * (outdoor - 30.0))
        cooling_kw = chiller_kw + crah_fan_kw + pump_tower_kw
        other_kw = p.ups_loss_fraction * it_kw + p.lighting_kw
        facility_kw = it_kw + cooling_kw + other_kw
        return_air = self.supply + it_kw / (15.0 * crah_speed)
        truth = dict(
            outdoor_temp_c=float(outdoor), it_load_pct=float(100 * util.mean()), it_power_kw=it_kw,
            **{col: float(t) for col, t in zip(C.INLET_COLS, self.inlet)},
            supply_air_c=float(self.supply), return_air_c=float(return_air), crah_setpoint_c=float(setpoint),
            crah_fan_pct=100 * crah_speed, chiller_kw=chiller_kw, crah_fan_kw=crah_fan_kw,
            pump_tower_kw=pump_tower_kw, cooling_kw=cooling_kw, other_kw=other_kw, facility_kw=facility_kw,
            server_fan_kw=float(server_fan_kw.sum()), max_inlet_c=float(self.inlet.max()),
        )
        sensors = self._noisy(truth)
        return truth, sensors

    def _noisy(self, truth):
        p, rng = self.p, self.rng
        out = dict(truth)
        for col in C.INLET_COLS + ["supply_air_c", "return_air_c", "outdoor_temp_c"]:
            out[col] = truth[col] + rng.normal(0, p.noise_temp_c)
        for col in ["it_power_kw", "chiller_kw", "crah_fan_kw", "pump_tower_kw", "other_kw"]:
            out[col] = truth[col] * (1 + rng.normal(0, p.noise_power))
        out["cooling_kw"] = out["chiller_kw"] + out["crah_fan_kw"] + out["pump_tower_kw"]
        out["facility_kw"] = out["it_power_kw"] + out["cooling_kw"] + out["other_kw"]
        out["max_inlet_c"] = max(out[c] for c in C.INLET_COLS)
        del out["server_fan_kw"]  # not metered separately in a real room
        return out
