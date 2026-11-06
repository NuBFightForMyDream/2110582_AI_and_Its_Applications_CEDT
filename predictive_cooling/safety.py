"""Step 5 — safety filter between the AI and the cooling equipment.

The MPC only *proposes*. Before a setpoint is sent, four rules are checked in order:

R1 Missing data   any rack sensor is missing -> ignore the AI, use the fallback setpoint
R2 Unfamiliar     sensors far outside the training data -> ignore the AI, use the fallback setpoint
R3 Too hot now    measured hottest inlet within 0.3 °C of the limit -> cool by 2 °C now
R4 Limits         keep the setpoint in 16-24 °C; normal changes at most 1 °C per 15 minutes

The fallback setpoint is the operator's proven fixed setting (the "Fixed setpoint" baseline),
so losing the AI simply returns the room to how it was run before.
"""
import math

from . import config as C

RULES = {
    "R1_missing_data": "Missing sensor data → fallback setpoint",
    "R2_unfamiliar_conditions": "Conditions outside training data → fallback setpoint",
    "R3_too_hot_now": "Measured inlet near limit → cool by 2 °C now",
    "R4_rate_or_range_limit": "Change limited to ±1 °C per step / 16-24 °C",
}


class SafetyFilter:
    def __init__(self, fallback_sp: float, limit=C.INLET_LIMIT, emergency_margin=C.EMERGENCY_MARGIN):
        self.fallback_sp = fallback_sp
        self.limit = limit
        self.emergency_margin = emergency_margin

    def check(self, proposed_sp, current_sp, reading, unfamiliar=()):
        """Return (setpoint to send, list of rule ids that fired)."""
        fired = []
        target = proposed_sp
        inlets = [reading[c] for c in C.INLET_COLS]
        if any(v is None or math.isnan(v) for v in inlets):
            fired.append("R1_missing_data")
            target = self.fallback_sp
        elif unfamiliar:
            fired.append("R2_unfamiliar_conditions")
            target = self.fallback_sp
        hottest = max((v for v in inlets if v is not None and not math.isnan(v)), default=math.nan)
        fastest_drop = 1.0
        if hottest >= self.limit - self.emergency_margin:
            fired.append("R3_too_hot_now")
            target = min(target, current_sp - 2)
            fastest_drop = 2.0
        limited = min(max(target, current_sp - fastest_drop), current_sp + 1)
        limited = min(max(limited, C.SP_MIN), C.SP_MAX)
        if limited != target:
            fired.append("R4_rate_or_range_limit")
        return float(limited), fired
