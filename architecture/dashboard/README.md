# Dashboard

[Architecture index](../../architecture.md)

One Streamlit page answers one question: which cooling setpoint uses the least total data-center electricity (IT equipment + cooling system) without exceeding the rack-inlet limit. FastAPI is called through `gateway.py` with bearer authentication; the UI never runs the workflow itself.

The page lists simulated situations only (synthetic test fixtures are hidden). Selecting a situation runs one offline `/analysis` case per browser session and shows the recommended setpoint, kWh saved versus the current setting, the hottest inlet against the limit, an IT-versus-cooling explanation and a stacked energy chart per setpoint. An optional form runs a new SustainDC simulation with default workload, capacity and candidate setpoints.

Supabase mode adds email/password sign-in. Imports, Gemini mode, case history, reviews and monitoring are API-only. The pre-simplification eight-page dashboard is preserved in `archive/dashboard-before-simplification-2026-10-31.tar.gz`. No physical control button exists.
