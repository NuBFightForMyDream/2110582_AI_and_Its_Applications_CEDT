# Supervisor Agent

> Earlier design detail. The implemented v0.2 runtime/contracts are documented in [runtime.md](runtime.md); type/tool examples below may describe the broader target.


[Architecture index](../../architecture.md)

The parent coordinates one case and returns the final explanation.

## Skills

- [S1 — Plan Analysis](skills/s1-plan-analysis.md)
- [S5 — Compose Recommendation](skills/s5-compose-recommendation.md)

## Role and handoffs

Receive the user question, selected dataset/zone/window, and server-created scope. Produce a typed plan for `analyze` or `recommend`. Dispatch to Data Analyst first. If its data is sufficient, dispatch to Cooling Advisor only when alternatives are requested. All handoffs pass validated objects or stable references rather than the full conversation.

The parent can ask the user for missing business intent. Missing telemetry or an unavailable experiment is returned as a data limitation, not invented or silently supplied by the model.

## Tools and phase boundaries

Only dispatch tools are exposed during planning: `call_data_analyst` and `call_cooling_advisor`. The composition phase has no tools and uses the G3-validated result. The finalizer attaches numerical fields from that result. G3 response checks run again before release if the explanation changes claims or references.

## Completion

Return a supported conclusion or a typed limitation. Keep the existing setting when it is the best supported option. No physical equipment action is available. The model, tool, and retry budgets cover the full parent/child case.
