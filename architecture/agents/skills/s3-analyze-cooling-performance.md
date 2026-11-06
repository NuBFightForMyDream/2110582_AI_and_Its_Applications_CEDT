> Design reference from the earlier planning phase. See [current runtime bindings](README.md); the running application loads cooling_advisor/skill_specs/, not this draft.

# S3 — Analyze Cooling Performance

[Architecture index](../../../architecture.md)

This is a draft procedure to bind to its owner during implementation.

Version: `0.1-draft`. Status: specification only. Owner: **Data Analyst**.

## Trigger and input

S2 has passed all checks required for the requested analysis.

Required input: Validated telemetry reference, quality report, baseline definition and metric scope.

## Procedure

1. Select a documented, comparable baseline.
2. Calculate energy using sample semantics and valid intervals; calculate only observable temperature metrics.
3. Report differences, provenance and uncertainty; label unverified causal explanations as hypotheses.

## Tool access

`compute_baseline`; a bounded `read_telemetry` only if its inputs are missing, all through G2.

## Output and completion

`DataAnalysisResult` containing facts, baseline, quality, hypotheses, limitations and evidence IDs.

Stop or fallback: No comparable baseline: report that limitation. Incomplete required coverage: insufficient_data. Current-state facts may still be returned if supported.

## Checks and trace

Schema, scope and limits are enforced by Python. Record this skill ID/version, case ID, tool-result IDs and outcome. Validate its positive path, missing-evidence path, and unauthorized-input path before implementation acceptance.

[Skill catalog](README.md) · [Agent ownership](../README.md) · [Tool contracts](../../backend/tools.md)
