# Review and Remaining Work

[Architecture index](../../architecture.md) · [Status matrix](../prototype-status.md)

The local application, simulator, guards, persistence, API and UI have executable tests. Genuine Gemini parent/child function calling and Supabase Auth/RLS clients are implemented. Their live acceptance remains unverified because no account credentials are configured. Scripted LLM and SDK HTTP mocks do not substitute for provider quality evaluation or a live RLS test.

Before a research claim: define a tuned rule baseline, validate thermal observability/energy accounting, calibrate model applicability, evaluate across representative traces and report uncertainty. Current constant-setpoint studies often favor the lowest setpoint; this is evidence about this model configuration, not an AI-control success.

Before a shared deployment: test Auth/RLS with two real users, use managed secrets/TLS, provide durable catalog/artifact storage, account for session expiry, move rate limits/jobs to shared infrastructure and define backup/retention procedures. Supabase user-owned rows are not tamper-evident audit records.

Before a real data center trial: build an authorized read-only BMS adapter, validate tags/units/sensors and obtain site-engineering policy. A later physical execution integration would need a separately designed, tested and explicitly authorized control system. None is implemented here.

No RAG/knowledge layer, vector database, GPU optimizer, physical actuator, external notification or cloud deployment has been added.
