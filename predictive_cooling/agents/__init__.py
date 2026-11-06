"""Agent layer: Supervisor + Data Analyst + Cooling Advisor (LangGraph), guarded by G1/G2/G3.

The agents decide *what to check and in which order* and explain the result; the Python tools
(sensor log, cleaning, forecaster, 81-plan MPC, safety filter) compute every number.
"""
from .engine import plans_table, run_case, to_json
from .models import CaseRequest, CaseResult, Policy
from .tools import AgentContext, default_context

__all__ = ["AgentContext", "CaseRequest", "CaseResult", "Policy", "default_context", "plans_table", "run_case", "to_json"]
