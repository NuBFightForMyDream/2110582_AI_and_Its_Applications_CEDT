# Cooling Advisor Agent

> Earlier design detail. The implemented v0.2 runtime/contracts are documented in [runtime.md](runtime.md); type/tool examples below may describe the broader target.


[Architecture index](../../architecture.md)

The second specialist compares supported alternatives after data inspection.

## Skill

- [S4 — Evaluate Cooling Options](skills/s4-evaluate-cooling-options.md)

## Input and tools

Receive a sufficient `DataAnalysisResult`, scenario/snapshot references, baseline ID, horizon, and policy version. Use `lookup_experiments` and `compare_candidates` through G2.

## Procedure

Retrieve compatible precomputed experiments. Compare recorded outcomes and submit candidate IDs and tool-result references to G3. Ranking arithmetic and feasibility checks run in Python. The supervisor receives the validated comparison after G3, then composes the explanation.

## Output and limits

Return `CandidateResults`, `not_supported`, or `no_feasible_option`. Do not extrapolate to untested settings, substitute a different initial state, or equate a simulated benefit with measured site savings. A live `run_simulation` tool requires the later worker design and is absent from the MVP registry.
