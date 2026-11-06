# API Surface

[Backend](README.md)

| Method | Route | Purpose |
|---|---|---|
| GET | /health | Public liveness/version, no credentials |
| GET | /settings | Boolean readiness, current owner/zones; never secret values |
| GET | /catalog | Authorized public/precomputed and owner-created scenarios |
| GET | /scenarios/{id} | Full authorized evidence/provenance |
| POST | /simulations | Bounded typed SustainDC request; isolated execution, save scenario |
| POST | /imports | Validate observation CSV, preserve artifact, save analysis-only scenario |
| POST | /analysis | Idempotent guarded workflow; Idempotency-Key required |
| GET | /cases | Latest 20 cases for current owner |
| GET | /cases/{id} | Full case, status, evidence, execution metadata and trace |
| GET | /cases/{id}/events | Owner-protected event trace |
| GET | /cases/{id}/reviews | Persisted review records |
| POST | /cases/{id}/decision | Record acknowledge / mark_for_review / dismiss |
| GET | /monitoring | Owner-scoped recent-case aggregates and guard events |

All routes except health require Authorization: Bearer. Generated local token is internal to the launcher/UI backend environment. In Supabase mode the dashboard forwards the signed-in user's access token. Invalid/expired sessions require signing in again; automatic refresh is not implemented.

Statuses distinguish completed, insufficient_data, not_supported, no_feasible_option, budget_exceeded and failed. Failed/budget cases release no selected option. Review writes are idempotent audit intent; equipment_command is always false.

`models.py` is the authoritative implemented schema; older planning handoff examples in contracts.md describe the earlier target. Do not treat those draft types as live API classes.
