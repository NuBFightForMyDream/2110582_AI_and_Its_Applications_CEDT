"""Fixed skill registry: which role owns which procedure, and which tools that procedure may call.

A skill is a versioned instruction file plus a tool allowlist. Loading a skill can never add tools;
G2 checks every call against this table.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    id: str
    owner: str
    name: str
    file: str
    tools: tuple[str, ...]
    version: str = "0.3.0"

    def instructions(self) -> str:
        return (Path(__file__).parent / "skill_specs" / self.file).read_text()


SKILLS = {
    "S1": Skill("S1", "supervisor", "Plan Case", "s1-plan.md", ("call_data_analyst", "call_cooling_advisor", "finish_case")),
    "S2": Skill("S2", "data_analyst", "Inspect Sensor Data", "s2-inspect.md", ("read_sensors", "inspect_data_quality")),
    "S3": Skill("S3", "data_analyst", "Analyze & Forecast", "s3-analyze.md", ("summarize_state", "forecast_hold", "finish_analysis")),
    "S4": Skill("S4", "cooling_advisor", "Evaluate Setpoint Plans", "s4-evaluate.md", ("run_mpc", "check_safety", "propose_setpoint")),
    "S5": Skill("S5", "supervisor", "Compose Recommendation", "s5-compose.md", ()),
}

ROLES = {
    "supervisor": "Supervisor (parent)",
    "data_analyst": "Data Analyst (child)",
    "cooling_advisor": "Cooling Advisor (child)",
}
