# Deployment

[Architecture index](../../architecture.md) · [Local runbook](local-runbook.md)

The tested deployment is local Python 3.12: Streamlit on loopback 8501 and FastAPI on loopback 8000. One launcher starts both and supplies the local API token. SQLite, owner catalogs, CSV artifacts and the pinned upstream source live under ignored `.local/`.

| Component | Current stack | Where it runs now |
|---|---|---|
| UI | Streamlit + Plotly | Local Python process |
| API / G1 | FastAPI + Pydantic | Local Python process |
| Roles / guards | LangGraph + Python + optional google-genai | API process |
| Physics | NumPy/pandas + pinned SustainDC files | Isolated child process |
| Cases / reviews | SQLite; optional Supabase user JWT/RLS | Local disk; remote adapter awaits credentials |
| Catalog / artifacts | Validated JSON + hashed CSV | Local persistent disk |

External Gemini and Supabase integrations are implemented but not live verified. No cloud resources have been created. No free-tier availability or billing is guaranteed. Hosting accounts, TLS/secrets, persistence and an end-to-end hosted acceptance check are still required before public release.

For a later hosted demo, separate the Streamlit service from FastAPI and configure COOLING_API_URL plus authenticated access. Keep API secrets in backend service secrets. The backend requires durable storage for imports/catalogs, or implement remote object/catalog storage first. If workers scale horizontally, replace process-local rate limiting and synchronize catalogs. Precomputed studies avoid requiring the live simulator on a constrained web host. Do not run this Python simulator/worker as a Vercel frontend function.

Vercel is optional only if a separate frontend is built; it is not necessary for this Python workspace. Docker is not required and no container image or cloud deployment is claimed tested.
