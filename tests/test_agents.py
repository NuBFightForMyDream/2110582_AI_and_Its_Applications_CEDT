"""Agent layer: offline roles, Gemini parent/child loops (scripted, no paid API) and the G1/G2/G3 guards."""
import pandas as pd
import pytest
from pydantic import ValidationError

from predictive_cooling import dataset, forecast, preprocess
from predictive_cooling.agents import AgentContext, CaseRequest, Policy, run_case
from predictive_cooling.agents.gemini import Call, Turn
from predictive_cooling.agents.skills import SKILLS

AT = "2026-04-21 00:30"     # just before the weekday 01:00 GPU batch


@pytest.fixture(scope="module")
def ctx():
    raw = dataset.load()
    clean, _ = preprocess.clean(raw)
    train, val, _ = preprocess.split(clean)
    gbm = forecast.Forecaster("gbm").fit(train)
    return AgentContext(raw, gbm, forecast.safety_margins(gbm, val), fallback_sp=20.0)


def actors(result):
    return [e.actor for e in result.events]


def test_offline_recommend_runs_all_three_agents_and_g3(ctx):
    result = run_case(CaseRequest(at=AT), ctx)
    assert result.status == "completed" and result.decision in ("change", "no_change", "max_cooling")
    assert {"G1", "supervisor", "data_analyst", "cooling_advisor", "G2", "G3"} <= set(actors(result))
    assert {e.skill_id for e in result.events if e.action == "skill_loaded"} == set(SKILLS)
    assert all(c.passed for c in result.checks)
    assert abs(result.recommended_setpoint - result.setpoint_in_use) <= 1
    assert result.evidence["mpc"]["plans_tried"] == 81
    assert "nothing was sent to equipment" in result.explanation


def test_precools_before_the_gpu_batch(ctx):
    result = run_case(CaseRequest(at=AT), ctx)
    assert result.evidence["forecast_hold"]["holding_is_safe"] is False
    assert result.recommended_setpoint < result.setpoint_in_use


def test_analyze_mode_skips_the_cooling_advisor(ctx):
    result = run_case(CaseRequest(at="2026-04-21 14:00", mode="analyze"), ctx)
    assert result.status == "completed" and result.decision == "analysis_only"
    assert "cooling_advisor" not in actors(result) and result.recommended_setpoint is None


def test_missing_sensors_withhold_the_ai_and_fall_back(ctx):
    result = run_case(CaseRequest(at=AT, sensor_fault=("B5", "B6")), ctx)
    assert result.status == "insufficient_data" and result.decision == "fallback"
    assert "MISSING_SENSOR:B6" in result.reason_codes
    assert "cooling_advisor" not in actors(result)
    assert result.recommended_setpoint == max(20.0, result.setpoint_in_use - 1)


def test_tools_cannot_see_the_future(ctx):
    from predictive_cooling.agents.tools import CaseWindow
    win = CaseWindow(ctx.log, pd.Timestamp(AT), 2.0)
    assert win.frame.index.max() == pd.Timestamp(AT)
    assert pd.isna(win.frame.loc[pd.Timestamp(AT), "crah_setpoint_c"])     # the decision being made is hidden


def test_g1_rejects_bad_requests(ctx):
    with pytest.raises(ValidationError):
        CaseRequest(at="2026-04-21 00:31")
    with pytest.raises(ValidationError):
        CaseRequest(at=AT, sensor_fault=("Z9",))
    result = run_case(CaseRequest(at="2026-02-02 00:15"), ctx)               # not enough history yet
    assert result.status == "failed" and result.reason_codes == ["CASE_TIME_OUTSIDE_SENSOR_LOG"]


def test_g2_shared_tool_budget(ctx):
    result = run_case(CaseRequest(at=AT), ctx, policy=Policy(tool_budget=3))
    assert result.status == "budget_exceeded" and result.recommended_setpoint is None


# ---- Gemini mode with a scripted provider (exercises the real loops without an API key) --------
class ScriptedProvider:
    def __init__(self, ctx, proposal=None):
        self.index = {}
        self.script = {
            "supervisor": [Call("call_data_analyst"), Call("call_cooling_advisor"), Call("finish_case", {"note": "Done."})],
            "data_analyst": [Call("read_sensors"), Call("inspect_data_quality"), Call("summarize_state"),
                             Call("forecast_hold"), Call("finish_analysis", {"note": "Batch at 01:00 ahead."})],
            "cooling_advisor": [Call("run_mpc"), Call("check_safety", {"setpoint": 21.0}),
                                Call("propose_setpoint", {"setpoint": proposal if proposal is not None else 21.0,
                                                          "note": "Pre-cool."})],
        }

    def turn(self, role, instructions, history, tools, timeout):
        i = self.index.get(role, 0)
        self.index[role] = i + 1
        return Turn([self.script[role][i]], tokens=10)


def install(monkeypatch, provider):
    monkeypatch.setattr("predictive_cooling.agents.gemini.provider_factory", lambda: provider)


def test_gemini_parent_child_loops_with_scripted_provider(ctx, monkeypatch):
    expected = run_case(CaseRequest(at=AT), ctx).recommended_setpoint
    provider = ScriptedProvider(ctx, proposal=expected)
    provider.script["cooling_advisor"][1] = Call("check_safety", {"setpoint": expected})
    install(monkeypatch, provider)
    result = run_case(CaseRequest(at=AT, agent_mode="gemini"), ctx)
    assert result.status == "completed", result.reason_codes
    assert result.workflow_mode == "gemini_parent_child_tool_calling"
    assert result.recommended_setpoint == expected
    assert result.model_calls == 11 and result.model_tokens == 110
    assert set(result.agent_notes) == {"supervisor", "data_analyst", "cooling_advisor"}


def test_g3_rejects_a_plausible_but_wrong_model_proposal(ctx, monkeypatch):
    expected = run_case(CaseRequest(at=AT), ctx).recommended_setpoint
    install(monkeypatch, ScriptedProvider(ctx, proposal=expected + 1))       # e.g. "warmer saves energy"
    result = run_case(CaseRequest(at=AT, agent_mode="gemini"), ctx)
    assert result.status == "failed" and result.reason_codes == ["PROPOSAL_EVIDENCE_MISMATCH"]
    assert result.recommended_setpoint is None


def test_model_cannot_skip_the_data_analyst(ctx, monkeypatch):
    provider = ScriptedProvider(ctx)
    provider.script["supervisor"] = [Call("call_cooling_advisor")]
    install(monkeypatch, provider)
    result = run_case(CaseRequest(at=AT, agent_mode="gemini"), ctx)
    assert result.reason_codes == ["TOOL_NOT_ALLOWED"] and result.recommended_setpoint is None


def test_nested_model_calls_share_one_budget(ctx, monkeypatch):
    install(monkeypatch, ScriptedProvider(ctx))
    result = run_case(CaseRequest(at=AT, agent_mode="gemini"), ctx, policy=Policy(model_call_budget=3))
    assert result.status == "budget_exceeded" and result.reason_codes == ["MODEL_BUDGET_EXCEEDED"]


def test_gemini_without_key_fails_closed(ctx, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = run_case(CaseRequest(at=AT, agent_mode="gemini"), ctx)
    assert result.status == "failed" and result.reason_codes == ["GEMINI_NOT_CONFIGURED"]
    assert result.recommended_setpoint is None
