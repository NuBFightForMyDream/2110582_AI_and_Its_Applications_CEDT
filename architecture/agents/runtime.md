# Agent Runtime

[Agents](README.md)

`engine.py` builds LangGraph and owns Runtime, event logging, G2 and G3. The START route selects either the offline node sequence or `gemini_agents`. Both paths converge at `guardrail_g3` and `supervisor_finalize`.

`agentic.py` supplies GeminiProvider, role-specific loops and strict Pydantic function-argument contracts. Each role sees only phase-available functions. Parent dispatch is itself a guarded tool call; nested work consumes the same request budget. The default maximum is 20 tools, 14 model calls and 120 seconds. The provider has one HTTP attempt per turn and a timeout capped by the remaining case deadline. Exactly one function call per response is accepted; malformed/parallel/unknown calls stop the case.

The model can request calculations and propose an experiment ID. It cannot submit its own energy or temperature values as evidence. G3 recomputes all compatible experiments, checks limits and verifies the proposal equals the deterministic minimum-total-kWh feasible option. A wrong proposal produces a failed case with no released selection.

The original provider model Content is retained in history, including thought signatures. Function responses use user Content with matching function ID/name. Model tokens, call count and resolved model versions are recorded. User questions and imported labels are untrusted context. No credentials enter model payloads.

Scripted-provider tests exercise these loops without a real API. Live provider behavior requires an account key and evaluation; the UI marks configuration separately from verification. See the [Google SDK manual function calling documentation](https://github.com/googleapis/python-genai#function-calling).
