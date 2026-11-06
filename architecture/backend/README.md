# Backend and Storage

[Architecture index](../../architecture.md) · [API routes](api.md)

FastAPI is the only dashboard entry to workflows, catalogs, simulation and persistence. Pydantic models forbid unknown fields and nonfinite numbers. The dashboard uses `gateway.py` over authenticated HTTP; it does not invoke the engine or database directly.

Local token mode verifies a generated private bearer token and assigns one local operator/zone-a. Supabase mode verifies the user JWT with Auth, reads approved cooling_memberships and passes the same user JWT to row-scoped case/review queries. No service-role key is required. The Supabase mode needs credentials and migration 002; it has not been exercised against a live project.

`storage.py` reserves a case before execution using owner + idempotency key + request hash. An exact completed retry returns the same case. A changed request with the same key, or an interrupted reservation, returns a conflict. Finalized SQLite cases cannot be overwritten through finish(). Interrupted cases are visible and require a new submission key; there is no recovery queue yet.

SQLite and Supabase store the complete result, trace and separate review actions. Both stores filter by owner. Scenarios and imported artifacts remain on local persistent disk in both modes. They are owner scoped, hash named and immutable through the API. Supabase records are user-owned, not cryptographically tamper-evident audit records.

Process-local rate limits bound analysis/import/simulation requests. The request-body limit is 2.5 MB. Simulation uses a child process and timeout. There are no arbitrary shell, SQL, filesystem-read or equipment-command endpoints.
