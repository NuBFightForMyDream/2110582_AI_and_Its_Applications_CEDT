# Implementation Status — 0.2

[Architecture index](../architecture.md) · [Run](../README.md)

| Area | Implemented and locally exercised | Remaining |
|---|---|---|
| Simulation | Pinned upstream SustainDC power/temperature submodel; isolated process; same-seed comparisons; two reproducible studies | Full dynamic environment, calibration and controlled site validation |
| Agents | LangGraph; offline procedures; actual bounded Gemini parent/child function-calling implementation | Live provider acceptance and task-quality evaluation with an API key |
| Skills | Five versioned local procedures, fixed owner/tool bindings, prompt hashes and load events | Broader task-specific skill library and evaluations |
| G1 | Typed requests, local bearer auth, owner/zone catalog scope; implemented Supabase JWT verification/memberships | Live Supabase Auth/RLS validation with a configured project |
| G2 | Allowlisted tools, phase gating, validated function arguments, shared call budgets/deadline; bounded provider timeout; simulator hard process timeout | Distributed rate limiting, job queue, cancellation service and operational abuse controls |
| G3 | Independent evidence read, matched snapshots/fingerprints/boundaries, arithmetic, temperature and ranking checks | Site-approved policy, applicability validation and causal savings verification |
| Imports | Bounded CSV schema/units declaration, timezone/gap checks, source hash, immutable catalog and raw local artifact | BMS adapter, tag mapping, verified sensor calibration and lifecycle administration |
| Persistence | Owner-scoped SQLite; idempotent cases/reviews; implemented user-JWT Supabase case/review adapter and migration | Live remote persistence check, shared scenario/object storage, tamper-evident audit and disaster recovery |
| UI | One page: situation picker, recommended setpoint, whole-data-center energy chart and optional new simulation via the API client | External account onboarding and multi-user deployment evaluation |
| Monitoring | Structured local events; owner-scoped recent cases, latency, usage and guard outcomes | Live equipment monitoring, alerts, external uptime checks, full retention/aggregation |
| Deployment | Working loopback FastAPI + Streamlit launcher with private generated token | Public hosting, managed secrets/TLS, persistent volumes and hosted acceptance tests |

Default offline mode needs no model key. Gemini and Supabase credentials were absent during implementation; do not describe either live integration as tested. Offline mode is deterministic role orchestration, not autonomous LLM reasoning. Model notes never replace Python numeric evidence.
