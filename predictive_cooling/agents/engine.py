"""LangGraph orchestration of the three agents, the G2 tool gateway and G3 independent verification.

    G1 request  ->  Supervisor (S1)  ->  Data Analyst (S2, S3)  ->  Cooling Advisor (S4)
                ->  G3 re-read + recompute  ->  Supervisor (S5) composes the released answer

Offline mode runs each role's procedure directly in a graph node (no LLM).
Gemini mode replaces the three role nodes with real parent/child function-calling loops
(see gemini.py); both paths share the same tools, G2 gateway, G3 and S5.
"""
import hashlib
import json
import logging
import time
from typing import TypedDict
from uuid import uuid4

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from .. import config as C
from . import tools as T
from .models import CaseRequest, CaseResult, Check, Event, Policy
from .skills import SKILLS

LOG = logging.getLogger("predictive_cooling.agents")


class GuardError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class Runtime:
    """Shared state of one case: request, evidence window, budgets and the event trace."""

    def __init__(self, request: CaseRequest, ctx: T.AgentContext, policy: Policy, case_id: str):
        self.request, self.ctx, self.policy = request, ctx, policy
        self.started = time.monotonic()
        self.result = CaseResult(case_id=case_id, at=request.at, mode=request.mode, agent_mode=request.agent_mode)
        self.window: T.CaseWindow | None = None
        self.ready = False
        self.proposal: float | None = None
        self.proposal_made = False

    # -- trace ------------------------------------------------------------------------------
    def event(self, actor, action, status="passed", skill=None, detail=""):
        event = Event(step=len(self.result.events) + 1, actor=actor, action=action, status=status, skill_id=skill,
                      detail=detail, elapsed_ms=round((time.monotonic() - self.started) * 1000, 2))
        self.result.events.append(event)
        LOG.info(json.dumps(event.model_dump()))

    def check_deadline(self):
        if time.monotonic() - self.started >= self.policy.deadline_seconds:
            raise GuardError("DEADLINE_EXCEEDED")

    def activate(self, skill_id, actor):
        self.check_deadline()
        skill = SKILLS[skill_id]
        if skill.owner != actor:
            raise GuardError("SKILL_OWNER_MISMATCH")
        self.event(actor, "skill_loaded", skill=skill_id, detail=f"{skill.name} · v{skill.version}")
        return skill.instructions()

    # -- G2: every tool call and every delegation goes through here -------------------------
    def call(self, tool, actor, skill_id, function):
        self.check_deadline()
        skill = SKILLS[skill_id]
        if skill.owner != actor or tool not in skill.tools:
            self.event("G2", tool, "rejected", skill_id, "TOOL_NOT_ALLOWED")
            raise GuardError("TOOL_NOT_ALLOWED")
        if self.result.tool_calls >= self.policy.tool_budget:
            self.event("G2", tool, "rejected", skill_id, "TOOL_BUDGET_EXCEEDED")
            raise GuardError("TOOL_BUDGET_EXCEEDED")
        self.result.tool_calls += 1          # reserved before running, nested delegations included
        self.event("G2", tool, "authorized", skill_id, f"{actor} · window ends {self.request.at:%d %b %H:%M}")
        try:
            value = function()
            self.check_deadline()
        except GuardError:
            raise
        except Exception as exc:
            self.event(actor, tool, "failed", skill_id, type(exc).__name__)
            raise
        self.event(actor, tool, "completed", skill_id, _summary(tool, value))
        return value


def _summary(tool, value):
    if not isinstance(value, dict):
        return ""
    if tool == "inspect_data_quality":
        return "ready" if value["ready"] else "not ready: " + ", ".join(value["errors"][:4])
    if tool == "summarize_state":
        return f"hottest {value['hottest_rack']} {value['hottest_inlet_c']} °C, headroom {value['headroom_to_limit_c']} °C"
    if tool == "forecast_hold":
        return f"hold {value['hold_setpoint_c']:g} °C → peak {max(value['with_safety_margin_c']):.2f} °C incl. margin"
    if tool == "run_mpc":
        return f"{value['safe_plans']}/{value['plans_tried']} safe; cheapest {value['chosen_plan_c']}"
    if tool == "check_safety":
        return f"{value['proposed_c']:g} → {value['final_c']:g} °C" + (f" ({', '.join(value['rules_fired'])})" if value["rules_fired"] else "")
    if tool == "propose_setpoint":
        return f"proposed {value['proposed_setpoint_c']}"
    return ""


# -- shared tool implementations (used by both offline and Gemini roles) ---------------------
def tool_impl(rt: Runtime, name: str, args=None):
    win, ctx = rt.window, rt.ctx
    if name == "read_sensors":
        return T.read_sensors(win)
    if name == "inspect_data_quality":
        out = T.inspect_data_quality(win)
        rt.result.evidence["data_quality"] = out
        return out
    if win.clean is None or rt.result.evidence.get("data_quality", {}).get("ready") is not True:
        raise GuardError("DATA_NOT_INSPECTED")
    if name == "summarize_state":
        out = rt.result.evidence["state"] = T.summarize_state(win)
        return out
    if name == "forecast_hold":
        out = rt.result.evidence["forecast_hold"] = T.forecast_hold(win, ctx)
        return out
    if name == "run_mpc":
        out = rt.result.evidence["mpc"] = T.run_mpc(win, ctx)
        return out
    if name == "check_safety":
        out = rt.result.evidence["safety"] = T.check_safety(win, ctx, args["setpoint"])
        return out
    raise GuardError("UNKNOWN_TOOL")


# -- offline role procedures -------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    runtime: Runtime


def supervisor_plan(state):
    rt = state["runtime"]
    rt.activate("S1", "supervisor")
    route = "Data Analyst" + (" → Cooling Advisor if data is ready" if rt.request.mode == "recommend" else " only (analyze)")
    rt.event("supervisor", "plan", detail=route)
    return {}


def data_analyst(state):
    rt = state["runtime"]

    def work():
        rt.activate("S2", "data_analyst")
        rt.call("read_sensors", "data_analyst", "S2", lambda: tool_impl(rt, "read_sensors"))
        quality = rt.call("inspect_data_quality", "data_analyst", "S2", lambda: tool_impl(rt, "inspect_data_quality"))
        rt.activate("S3", "data_analyst")
        if quality["ready"]:
            rt.call("summarize_state", "data_analyst", "S3", lambda: tool_impl(rt, "summarize_state"))
            hold = rt.call("forecast_hold", "data_analyst", "S3", lambda: tool_impl(rt, "forecast_hold"))
            note = f"Data ready. Holding {hold['hold_setpoint_c']:g} °C looks {'safe' if hold['holding_is_safe'] else 'unsafe'} for the next hour."
        else:
            note = "Data not ready: " + ", ".join(quality["errors"])
        rt.call("finish_analysis", "data_analyst", "S3", lambda: {"ready": quality["ready"]})
        rt.result.agent_notes["data_analyst"] = note
        return quality["ready"]

    rt.ready = rt.call("call_data_analyst", "supervisor", "S1", work)
    rt.event("supervisor", "receive_analysis", "completed" if rt.ready else "blocked")
    return {}


def cooling_advisor(state):
    rt = state["runtime"]

    def work():
        rt.activate("S4", "cooling_advisor")
        mpc = rt.call("run_mpc", "cooling_advisor", "S4", lambda: tool_impl(rt, "run_mpc"))
        safety = rt.call("check_safety", "cooling_advisor", "S4",
                         lambda: tool_impl(rt, "check_safety", {"setpoint": mpc["first_step_c"]}))
        rt.call("propose_setpoint", "cooling_advisor", "S4", lambda: propose(rt, safety["final_c"]))
        rt.result.agent_notes["cooling_advisor"] = (f"Cheapest safe plan {mpc['chosen_plan_c']} of {mpc['safe_plans']} safe plans; "
                                                    f"safety filter returned {safety['final_c']:g} °C.")

    rt.call("call_cooling_advisor", "supervisor", "S1", work)
    rt.event("supervisor", "receive_proposal", "completed")
    return {}


def propose(rt: Runtime, setpoint):
    rt.proposal = None if setpoint is None else float(setpoint)
    rt.proposal_made = True
    return {"proposed_setpoint_c": rt.proposal, "requires_independent_g3_validation": True}


# -- G3: re-read, recompute, compare -----------------------------------------------------------
def validate(state):
    rt = state["runtime"]
    rt.check_deadline()
    checks = rt.result.checks
    fresh = T.CaseWindow(rt.ctx.log, pd.Timestamp(rt.request.at), rt.policy.history_hours, rt.request.sensor_fault)
    quality = T.inspect_data_quality(fresh)               # authoritative re-read, not the agent's copy
    rt.result.setpoint_in_use = fresh.setpoint_in_use
    checks.append(Check(name="Sensor data re-read and re-inspected", passed=quality["ready"],
                        detail="ready" if quality["ready"] else ", ".join(quality["errors"])))
    if quality["ready"] != rt.ready:
        raise GuardError("EVIDENCE_CHANGED")
    if not quality["ready"]:
        rt.result.status, rt.result.decision = "insufficient_data", "fallback"
        rt.result.reason_codes = quality["errors"]
        rt.result.recommended_setpoint = T.fallback_step(fresh, rt.ctx)
        rt.result.evidence["data_quality"] = quality
        rt.event("G3", "validate_evidence", "blocked", detail="AI withheld; safety rule R1 fallback applies")
        return {}
    state_now = T.summarize_state(fresh)
    reported = rt.result.evidence.get("state")
    same = reported is not None and all(reported[k] == state_now[k] for k in ("hottest_rack", "hottest_inlet_c", "it_power_kw"))
    checks.append(Check(name="Analyst's numbers match recomputation", passed=same,
                        detail=f"hottest {state_now['hottest_rack']} {state_now['hottest_inlet_c']} °C"))
    if not same:
        raise GuardError("ANALYSIS_EVIDENCE_MISMATCH")
    rt.result.evidence["state"] = state_now
    rt.result.evidence["forecast_hold"] = T.forecast_hold(fresh, rt.ctx)
    if rt.request.mode == "analyze":
        rt.result.status, rt.result.decision = "completed", "analysis_only"
        rt.event("G3", "validate_analysis", detail="Analysis only; no setpoint released")
        return {}
    expected, reason, table, safety = T.expected_recommendation(fresh, rt.ctx)
    rt.result.evidence["mpc"] = T.run_mpc(fresh, rt.ctx)
    rt.result.evidence["safety"] = safety
    rt.result.evidence["_plans"] = table
    checks.append(Check(name="A proposal was made", passed=rt.proposal_made))
    match = rt.proposal_made and rt.proposal is not None and abs(rt.proposal - expected) < 1e-9
    checks.append(Check(name="Proposal = recomputed MPC + safety filter", passed=match,
                        detail=f"proposed {rt.proposal}, recomputed {expected:g} °C"))
    if not match:
        raise GuardError("PROPOSAL_EVIDENCE_MISMATCH")
    in_range = C.SP_MIN <= expected <= C.SP_MAX and abs(expected - fresh.setpoint_in_use) <= 2
    checks.append(Check(name="Within 16–24 °C and rate limit", passed=in_range))
    if not in_range:
        raise GuardError("SETPOINT_OUT_OF_POLICY")
    chosen = table[table["chosen"]].iloc[0]
    safe_plan = bool(chosen["safe"])
    checks.append(Check(name="Chosen plan forecast ≤ 27 °C incl. margin", passed=safe_plan,
                        detail=f"peak {chosen['peak_guarded_c']:.2f} °C"))
    rt.result.plan = [float(chosen[f"sp_{k}"]) for k in range(1, C.HORIZON + 1)]
    rt.result.recommended_setpoint = expected
    rt.result.status = "completed"
    if reason == "no_safe_plan_max_cooling":
        rt.result.decision = "max_cooling"
    else:
        rt.result.decision = "no_change" if expected == fresh.setpoint_in_use else "change"
    rt.event("G3", "validate_recommendation", detail=f"approved {expected:g} °C")
    return {}


# -- S5: deterministic statement ---------------------------------------------------------------
def supervisor_finalize(state):
    rt = state["runtime"]
    rt.activate("S5", "supervisor")
    r, ev = rt.result, rt.result.evidence
    mock = "Mock 12-rack room and simplified plant model; nothing was sent to equipment."
    if r.status == "insufficient_data":
        r.explanation = (f"No AI recommendation: {', '.join(r.reason_codes)}. Safety rule R1 applies — return toward the proven "
                         f"fallback setpoint {rt.ctx.fallback_sp:g} °C ({r.recommended_setpoint:g} °C this step, max 1 °C per 15 min). {mock}")
    elif r.status == "completed":
        s, hold = ev["state"], ev["forecast_hold"]
        risk = (f"Holding {hold['hold_setpoint_c']:g} °C for the next hour is forecast to peak at "
                f"{max(hold['with_safety_margin_c']):.1f} °C incl. safety margin ({'safe' if hold['holding_is_safe'] else 'too close to 27 °C'}).")
        now = (f"Hottest rack {s['hottest_rack']} is {s['hottest_inlet_c']:.1f} °C ({s['headroom_to_limit_c']:.1f} °C below the limit), "
               f"IT {s['it_power_kw']:.0f} kW, cooling {s['cooling_kw']:.0f} kW.")
        if r.mode == "analyze":
            r.explanation = f"{now} {risk} {mock}"
        else:
            m, safety = ev["mpc"], ev["safety"]
            head = {"change": f"Change the supply-air setpoint from {r.setpoint_in_use:g} °C to {r.recommended_setpoint:g} °C now.",
                    "no_change": f"Keep the supply-air setpoint at {r.recommended_setpoint:g} °C.",
                    "max_cooling": f"No plan is forecast to stay safe: cool to {r.recommended_setpoint:g} °C now (maximum cooling)."}[r.decision]
            saving = m["hold_energy_next_hour_kwh"] - m["chosen_energy_next_hour_kwh"]
            if abs(saving) < 0.05:
                versus = "the same as holding"
            elif saving > 0:
                versus = f"{saving:.1f} kWh less than holding"
            else:
                versus = f"{-saving:.1f} kWh more than holding" + ("" if m["hold_is_safe"] else ", the price of staying below the limit")
            plan = " → ".join(f"{v:g}" for v in r.plan)
            rules = ", ".join(safety["rules_fired"]) or "no rule had to step in"
            r.explanation = (f"{head} {now} {risk} Of {m['plans_tried']} plans, {m['safe_plans']} stay safe; the cheapest is "
                             f"{plan} °C, forecast to use {m['chosen_energy_next_hour_kwh']:.1f} kWh next hour "
                             f"({versus}). Safety filter: {rules}. G3 re-read the sensors and recomputed "
                             f"this independently. {mock}")
    else:
        r.explanation = "No recommendation was released: " + ", ".join(r.reason_codes)
    rt.event("supervisor", "compose", "completed", "S5")
    return {}


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("supervisor_plan", supervisor_plan)
    builder.add_node("data_analyst", data_analyst)
    builder.add_node("cooling_advisor", cooling_advisor)
    builder.add_node("gemini_agents", lambda s: __import__("predictive_cooling.agents.gemini", fromlist=["x"]).run_agents(s["runtime"]))
    builder.add_node("guardrail_g3", validate)
    builder.add_node("supervisor_finalize", supervisor_finalize)
    builder.add_conditional_edges(START, lambda s: "gemini_agents" if s["runtime"].request.agent_mode == "gemini" else "supervisor_plan")
    builder.add_edge("supervisor_plan", "data_analyst")
    builder.add_conditional_edges("data_analyst", lambda s: "cooling_advisor" if s["runtime"].ready and s["runtime"].request.mode == "recommend"
                                  else "guardrail_g3")
    builder.add_edge("cooling_advisor", "guardrail_g3")
    builder.add_edge("gemini_agents", "guardrail_g3")
    builder.add_edge("guardrail_g3", "supervisor_finalize")
    builder.add_edge("supervisor_finalize", END)
    return builder.compile()


GRAPH = build_graph()


def run_case(request: CaseRequest, ctx: T.AgentContext | None = None, policy: Policy | None = None,
             case_id: str | None = None) -> CaseResult:
    ctx = ctx or T.default_context()
    policy = policy or Policy()
    rt = Runtime(request, ctx, policy, case_id or uuid4().hex[:12])
    rt.result.execution = {"policy": policy.model_dump(), "request": request.model_dump(mode="json"),
                           "skills": {sid: {"version": s.version, "sha256": hashlib.sha256(s.instructions().encode()).hexdigest()[:16]}
                                      for sid, s in SKILLS.items()}}
    try:
        at = pd.Timestamp(request.at)                                  # G1: scope and request checks
        first, last = ctx.log.index.min(), ctx.log.index.max()
        if not first + pd.Timedelta(hours=policy.history_hours) <= at <= last:
            raise GuardError("CASE_TIME_OUTSIDE_SENSOR_LOG")
        rt.window = T.CaseWindow(ctx.log, at, policy.history_hours, request.sensor_fault)
        rt.event("G1", "validate_request", detail=f"{request.mode} · {request.agent_mode} · room zone-a · "
                                                   f"tools see {policy.history_hours:g} h up to {at:%d %b %H:%M}")
        GRAPH.invoke({"runtime": rt}, config={"recursion_limit": 16})
    except GuardError as exc:
        r = rt.result
        r.status = "budget_exceeded" if exc.code.endswith("BUDGET_EXCEEDED") or exc.code == "DEADLINE_EXCEEDED" else "failed"
        r.decision, r.recommended_setpoint, r.plan = "unavailable", None, []
        r.reason_codes = [exc.code]
        r.explanation = "No recommendation was released: " + exc.code
        rt.event("workflow", "stop", "blocked", detail=exc.code)
    except Exception as exc:                                           # provider/config failures never look like success
        r = rt.result
        r.status, r.decision, r.recommended_setpoint, r.plan = "failed", "unavailable", None, []
        r.reason_codes = ["GEMINI_NOT_CONFIGURED" if str(exc) == "GEMINI_NOT_CONFIGURED" else "WORKFLOW_ERROR"]
        r.explanation = "The workflow failed; no recommendation was released (" + r.reason_codes[0] + ")."
        rt.event("workflow", "stop", "failed", detail=type(exc).__name__)
    rt.result.elapsed_ms = round((time.monotonic() - rt.started) * 1000, 2)
    return rt.result


def plans_table(result: CaseResult):
    """The 81-plan table G3 verified (for charts); not part of the JSON result."""
    return result.evidence.get("_plans")


def to_json(result: CaseResult) -> dict:
    data = result.model_dump(mode="json", exclude={"evidence"})
    data["evidence"] = {k: v for k, v in result.evidence.items() if not k.startswith("_")}
    return json.loads(json.dumps(data, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
