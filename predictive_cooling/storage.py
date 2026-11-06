"""Optional case history in Supabase.

Off unless both SUPABASE_URL and SUPABASE_KEY are set. Use a server-side *secret* key (Supabase "secret" /
service-role key): the Streamlit script runs on the server, so the key never reaches the browser, and the table has
RLS on with no public policies (see supabase/migrations/001_agent_cases.sql). Storage failures never break a case.
"""
import os

TABLE = "agent_cases"


def enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


def _client():
    from supabase import create_client          # imported lazily: the package is only needed when enabled
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def case_row(request, result) -> dict:
    """One flat row per case; the full result (minus raw tool outputs) goes in `result` as JSON."""
    return {
        "case_id": result.case_id,
        "case_at": result.at.isoformat(),
        "question": request.question,
        "task": request.mode,
        "agent_mode": request.agent_mode,
        "sensor_fault": list(request.sensor_fault),
        "status": result.status,
        "decision": result.decision,
        "setpoint_in_use": result.setpoint_in_use,
        "recommended_setpoint": result.recommended_setpoint,
        "explanation": result.explanation,
        "tool_calls": result.tool_calls,
        "model_calls": result.model_calls,
        "result": result.model_dump(mode="json", exclude={"evidence"}),
    }


def save_case(request, result, client=None) -> str | None:
    """Insert one case. Returns an error message instead of raising."""
    if client is None and not enabled():
        return None
    try:
        (client or _client()).table(TABLE).insert(case_row(request, result)).execute()
        return None
    except Exception as exc:                       # network, auth, missing table…
        return f"{type(exc).__name__}: {exc}"


def recent_cases(limit=20, client=None) -> list[dict]:
    if client is None and not enabled():
        return []
    columns = ("created_at,case_at,question,task,agent_mode,status,decision,setpoint_in_use,recommended_setpoint,"
               "tool_calls,model_calls")
    response = (client or _client()).table(TABLE).select(columns).order("created_at", desc=True).limit(limit).execute()
    return response.data or []
