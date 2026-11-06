> Design reference from the earlier planning phase. See [current runtime bindings](README.md); the running application loads cooling_advisor/skill_specs/, not this draft.

# S4 — Evaluate Cooling Options

[Architecture index](../../../architecture.md)

This is a draft procedure to bind to its owner during implementation.

Version: `0.1-draft`. Status: specification only. Owner: **Cooling Advisor**.

## Trigger and input

The case requests alternatives and required data passed S2/S3.

Required input: Analysis result reference; scenario/snapshot IDs; baseline; evaluation horizon; policy version.

## Procedure

1. Retrieve experiments matching the scenario, state, horizon and energy boundary.
2. Compare recorded alternatives, including the baseline; use Python for arithmetic and candidate ordering.
3. Submit candidate and evidence IDs for independent G3 validation.

## Tool access

`lookup_experiments`, `compare_candidates` through G2.

## Output and completion

`CandidateResults`; G3 produces `ValidationResult` before a recommendation is released.

Stop or fallback: No matching experiments: not_supported. No feasible candidate: no_feasible_option. Baseline remains best: completed with no_change.

## Checks and trace

Schema, scope and limits are enforced by Python. Record this skill ID/version, case ID, tool-result IDs and outcome. Validate its positive path, missing-evidence path, and unauthorized-input path before implementation acceptance.

[Skill catalog](README.md) · [Agent ownership](../README.md) · [Tool contracts](../../backend/tools.md)
