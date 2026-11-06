> Design reference from the earlier planning phase. See [current runtime bindings](README.md); the running application loads cooling_advisor/skill_specs/, not this draft.

# S2 — Inspect Cooling Data

[Architecture index](../../../architecture.md)

This is a draft procedure to bind to its owner during implementation.

Version: `0.1-draft`. Status: specification only. Owner: **Data Analyst**.

## Trigger and input

Before numerical analysis of a selected window.

Required input: Dataset and metric references; authorized zone/window; required units and observability.

## Procedure

1. Read a bounded window and obtain source/provenance references.
2. Check coverage, timestamp meaning, units, gaps, invalid values and temperature observability.
3. Return a quality report with required and optional missing fields.

## Tool access

`read_telemetry`, `inspect_data_quality` through G2.

## Output and completion

A tool-backed quality report within `DataAnalysisResult`.

Stop or fallback: Required fields, energy boundary, or time interpretation missing: insufficient_data. Optional fields may be omitted only with an explicit limitation.

## Checks and trace

Schema, scope and limits are enforced by Python. Record this skill ID/version, case ID, tool-result IDs and outcome. Validate its positive path, missing-evidence path, and unauthorized-input path before implementation acceptance.

[Skill catalog](README.md) · [Agent ownership](../README.md) · [Tool contracts](../../backend/tools.md)
