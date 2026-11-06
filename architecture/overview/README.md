# System Overview

[Architecture index](../../architecture.md)

```mermaid
flowchart LR
    UI["Streamlit / Plotly"] -->|"Bearer-authenticated HTTP"| API["FastAPI / Pydantic · G1"]
    API --> C["Owner-scoped evidence catalog"]
    API --> W["LangGraph workflow"]
    W --> A["Offline roles OR Gemini parent/child loops"]
    A --> G2["G2 · tool gateway"]
    G2 --> PY["Python data-quality / energy tools"]
    PY --> C
    W --> G3["G3 · independent recomputation"]
    G3 --> DB["SQLite OR user-JWT Supabase cases/reviews"]
    DB --> UI
    API --> SIM["Isolated pinned SustainDC submodel"]
    SIM --> C
    CSV["Bounded CSV import"] --> API
```

Simulation/import is a separate explicit user operation. Analysis agents use existing evidence and cannot start arbitrary programs, query arbitrary SQL or command equipment. Source provenance is stored separately from execution metadata (request, policy, model, skill hashes).

The local default is fully usable without external accounts. Optional Gemini/Supabase clients are implemented but await credentials for live validation. See [status](../prototype-status.md).
