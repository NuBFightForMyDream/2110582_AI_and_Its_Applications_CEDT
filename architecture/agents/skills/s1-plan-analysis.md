> Design reference from the earlier planning phase. See [current runtime bindings](README.md); the running application loads cooling_advisor/skill_specs/, not this draft.

# S1 — Plan Analysis

[Architecture index](../../../architecture.md)

This is a draft procedure to bind to its owner during implementation.

Version: `0.1-draft`. Status: specification only. Owner: **Supervisor**.

## Trigger and input

A new scoped case or a bounded clarification.

Required input: User question; selected dataset, zone and window; trusted context; policy reference.

## Procedure

1. Classify the request as analyze or recommend.
2. Check required selections and distinguish missing user intent from missing evidence.
3. Create a scoped task and dispatch Data Analyst; only request alternatives after sufficient data.

## Tool access

`call_data_analyst`, `call_cooling_advisor` through the dispatch guard.

## Output and completion

`TaskPlan` and scoped `AnalysisTask`/`AdvisorTask` handoffs.

Stop or fallback: Missing user intent: request clarification. Invalid scope: reject. Missing evidence: return the specialist limitation without inventing it.

## Checks and trace

Schema, scope and limits are enforced by Python. Record this skill ID/version, case ID, tool-result IDs and outcome. Validate its positive path, missing-evidence path, and unauthorized-input path before implementation acceptance.

[Skill catalog](README.md) · [Agent ownership](../README.md) · [Tool contracts](../../backend/tools.md)
