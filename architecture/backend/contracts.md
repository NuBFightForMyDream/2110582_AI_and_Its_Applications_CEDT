# Contracts and Case Lifecycle

> Earlier design detail. The implemented v0.2 runtime/contracts are documented in [api.md](api.md); type/tool examples below may describe the broader target.


[Architecture index](../../architecture.md)

These are proposed schema contracts, not implemented models.

## Request and execution context

```text
AnalysisRequest
  mode: analyze | recommend
  question, dataset_id, zone_id, time_window
  scenario_id / baseline_id when required

RunContext (server-created authorization/configuration)
  case_id, trace_id, user_id, authorized_scope, policy_version
  deadline, budget_limits

UsageLedger (server-owned counters, not model-editable)
  model_calls, tool_calls, tokens_when_reported, retries, elapsed_time
```

Resolve default scenario/baseline references only from documented server configuration. Missing ambiguous intent requires clarification. Missing measurements or comparison evidence produces a typed limitation rather than a model-created default.

## Parent and child handoffs

| Contract | Minimum fields |
|---|---|
| `TaskPlan` | mode, scoped task references, required skills, stop conditions |
| `AnalysisTask` | dataset/zone/window, required metrics, baseline reference |
| `InspectionTask` | scoped telemetry selection, required metrics, expected units and sample semantics |
| `DataQualityReport` | ready/insufficient status, coverage, units/time checks, observable metrics and missing fields |
| `DataAnalysisResult` | ready/insufficient status, quality report, facts with units, baseline, hypotheses, missing fields, evidence IDs |
| `AdvisorTask` | sufficient analysis reference, scenario/snapshot, baseline, horizon, policy version |
| `CandidateResults` | found/unsupported/no-feasible status, candidate IDs, comparison-tool result IDs, limitations |
| `ValidationResult` | passed, reason codes, accepted candidate IDs, trusted metrics, policy and evidence versions |
| `Explanation` | supported claim/evidence references, limitations, recommendation wording |

Python/Pydantic validates each boundary. Pass immutable evidence references and the minimum relevant data. Returned IDs are re-authorized and re-read; they do not grant access to another case or zone.

## Lifecycle and response

```text
Case execution_status: created -> running -> terminal
AnalysisResponse.status:
  completed | clarification_required | insufficient_data | not_supported |
  no_feasible_option | budget_exceeded | failed
AnalysisResponse.decision: no_change | recommend_change | unavailable
Other fields:
  case_id, source_kinds, scenario_id, policy_version
  validated_facts, baseline, selected_candidate, evidence_ids
  explanation, limitations, reason_codes, clarification_questions
```

`completed` describes a finished supported analysis, including `no_change`. It does not claim measured site savings. Non-completed results have no executable recommendation. The finalizer owns numeric fields and units; it validates explanation references before release.

`clarification_required` is terminal for the current request and includes questions about missing user intent. A clarified submission uses a new idempotency key and can reference the previous case. Missing measurements remain `insufficient_data`, not a question that the model fills in itself.

Each fact and candidate retains its own `source_kind` and evidence provenance. `source_kinds` summarizes the sources in the case; mixing measured and simulated evidence does not establish a calibrated counterfactual.

## HTTP, retries and idempotency

- G1 failures use 401/403/422/429 as appropriate and do not invoke agents. Request-level events have a request ID even when no case is created.
- Accepted cases return their typed domain outcome; unexpected service failure returns a structured error and a case ID when one was persisted. Client retry logic must inspect both HTTP status and domain outcome.
- Require an idempotency key and canonical request hash for `POST /analysis`. Same owner/key/hash returns the existing case; a different payload using the same key returns 409. Concurrent attempts reserve the case atomically.
- Bound provider retries, call timeouts and graph steps within one global deadline. Invalid model output has a bounded repair path. An explanation failure can use a deterministic summary only when its underlying result has already passed G3.
- Persist terminal status and result before reporting a stored successful case. If persistence fails, return a storage error rather than silently claiming a completed persisted run.

## Operator decisions

`acknowledge`, `mark_for_review`, and `dismiss` are authorized records tied to a case and candidate ID. They neither alter experiment evidence nor create an equipment command. An operational approval workflow is a separate later feature.
