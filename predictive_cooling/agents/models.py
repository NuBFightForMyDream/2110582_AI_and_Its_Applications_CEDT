"""Typed contracts for one agent case (G1 validates the request before anything runs)."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import config as C


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CaseRequest(StrictModel):
    """One question about one moment in the room's sensor log."""
    at: datetime
    question: str = Field(default="What supply-air setpoint should we use for the next hour, and why?",
                          min_length=1, max_length=1000)
    mode: Literal["recommend", "analyze"] = "recommend"
    agent_mode: Literal["offline", "gemini"] = "offline"
    sensor_fault: tuple[str, ...] = Field(default=(), description="Demo: these racks' inlet sensors stop reporting "
                                                                   "for the last 45 minutes before `at`.")

    @field_validator("at")
    @classmethod
    def on_grid(cls, value: datetime):
        if value.tzinfo is not None or value.minute % C.STEP_MINUTES or value.second or value.microsecond:
            raise ValueError("`at` must be a local timestamp on the 15-minute grid")
        return value

    @field_validator("sensor_fault")
    @classmethod
    def known_racks(cls, value):
        if len(set(value)) != len(value) or not set(value) <= set(C.RACKS):
            raise ValueError("sensor_fault must list distinct rack ids such as 'B6'")
        return value


class Policy(StrictModel):
    version: str = "predictive-cooling-v1"
    inlet_limit_c: float = Field(default=C.INLET_LIMIT, ge=15, le=35)
    history_hours: float = Field(default=2.0, ge=1.5, le=6)
    tool_budget: int = Field(default=20, ge=0, le=100)
    model_call_budget: int = Field(default=14, ge=0, le=40)
    deadline_seconds: float = Field(default=120, gt=0, le=600)


class Event(StrictModel):
    step: int
    actor: str
    action: str
    status: str
    skill_id: str | None = None
    detail: str = ""
    elapsed_ms: float


class Check(StrictModel):
    name: str
    passed: bool
    detail: str = ""


Status = Literal["completed", "insufficient_data", "failed", "budget_exceeded"]
Decision = Literal["change", "no_change", "max_cooling", "fallback", "analysis_only", "unavailable"]


class CaseResult(BaseModel):
    case_id: str
    at: datetime
    mode: str
    agent_mode: str
    workflow_mode: str = "offline_langgraph_roles"
    status: Status = "failed"
    decision: Decision = "unavailable"
    setpoint_in_use: float | None = None
    recommended_setpoint: float | None = None
    plan: list[float] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)       # tool outputs, recomputed by G3 before release
    checks: list[Check] = Field(default_factory=list)  # G3 verification list
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str = ""
    agent_notes: dict[str, str] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    tool_calls: int = 0
    model_calls: int = 0
    model_tokens: int = 0
    elapsed_ms: float = 0.0
    execution: dict = Field(default_factory=dict)
