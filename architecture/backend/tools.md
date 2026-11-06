# Tool Registry

> Earlier design detail. The implemented v0.2 runtime/contracts are documented in [api.md](api.md); type/tool examples below may describe the broader target.


[Architecture index](../../architecture.md)

This registry connects each skill to actual future Python function boundaries.

## Tool allowlists

| Tool | Owner/phase | Input references | Output |
|---|---|---|---|
| `call_data_analyst` | Supervisor planning | `AnalysisTask` | `DataAnalysisResult` |
| `call_cooling_advisor` | Supervisor planning | `AdvisorTask` | `CandidateResults` followed by G3 |
| `read_telemetry` | Data Analyst S2/S3 | dataset, zone, window, approved metrics | bounded samples, manifest, evidence reference |
| `inspect_data_quality` | Data Analyst S2 | authorized telemetry reference, required metrics | units/time/coverage/observability report |
| `compute_baseline` | Data Analyst S3 | telemetry and approved baseline references | Python-computed metrics/differences and units |
| `lookup_experiments` | Cooling Advisor S4 | scenario, snapshot, horizon, baseline, policy | compatible experiment IDs or not_supported |
| `compare_candidates` | Cooling Advisor S4 | authorized experiment IDs, baseline, policy | calculated energy/temperature comparison with result ID |

Compose Recommendation (S5) has no tools. A live simulation tool is not registered in the MVP.

## Execution wrapper

```text
Agent selects a registered tool and typed arguments
  -> G2 checks agent + skill + tool + data scope
  -> reserve shared budget and establish timeout
  -> resolve references using the scoped repository
  -> execute Python calculation/query
  -> validate output schema and persist/reference its evidence
  -> return a bounded structured result and record the event
```

Instructions and schemas are application-owned. Neither a skill file nor a model-generated payload can register a new callable. Queries are constructed by Python, with explicit metric/window/row bounds. Large arrays stay outside the LLM context; return summaries and immutable references.

## Evidence and error semantics

Every result identifies `tool_result_id`, source/evidence references, units, computation/version metadata, and limitations. G3 recomputes the final ranking from authoritative evidence. It does not accept LLM-provided energies as measurements.

Numerical algorithms and the distinction between historical and experiment baselines are defined in [Analysis](../analysis/README.md).

Expected errors use typed codes such as `ACCESS_DENIED`, `INVALID_ARGUMENT`, `MISSING_DATA`, `UNSUPPORTED_SCENARIO`, `BUDGET_EXCEEDED`, `TIMEOUT` and `UPSTREAM_UNAVAILABLE`. Preserve error type when routing; retries must not convert a rejection into success or bypass the same scope checks.
