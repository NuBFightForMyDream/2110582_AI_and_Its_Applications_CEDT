> Design reference from the earlier planning phase. See [current runtime bindings](README.md); the running application loads cooling_advisor/skill_specs/, not this draft.

# S5 — Compose Recommendation

[Architecture index](../../../architecture.md)

This is a draft procedure to bind to its owner during implementation.

Version: `0.1-draft`. Status: specification only. Owner: **Supervisor**.

## Trigger and input

A validated analysis or recommendation is ready for presentation.

Required input: G3-validated facts and comparison, approved evidence IDs, source kind and limitations.

## Procedure

1. Explain supported findings and the selected option or limitation.
2. Reference only approved evidence and describe simulation results as simulation results.
3. Return a structured explanation; the finalizer inserts trusted numerical fields and rechecks references.

## Tool access

None during composition.

## Output and completion

`Explanation` plus the finalizer-built `AnalysisResponse`.

Stop or fallback: Invalid explanation: bounded repair or a deterministic validated summary. No valid comparison: explain the typed limitation.

## Checks and trace

Schema, scope and limits are enforced by Python. Record this skill ID/version, case ID, tool-result IDs and outcome. Validate its positive path, missing-evidence path, and unauthorized-input path before implementation acceptance.

[Skill catalog](README.md) · [Agent ownership](../README.md) · [Tool contracts](../../backend/tools.md)
