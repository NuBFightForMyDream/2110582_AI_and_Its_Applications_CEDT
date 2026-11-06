# Runtime Skills

[Agents](../README.md) · [Runtime](../runtime.md)

Five versioned skill procedures are implemented in `cooling_advisor/skill_specs/`; the fixed registry is `cooling_advisor/skills.py`. The backend loads these trusted files, checks the owner and tool allowlist, records load events, and stores instruction hashes in each case. The model cannot load an arbitrary file or install additional skills.

| Skill | Owner | Runtime instruction |
|---|---|---|
| S1 Plan | Supervisor | [s1-plan.md](../../../cooling_advisor/skill_specs/s1-plan.md) |
| S2 Inspect | Data Analyst | [s2-inspect.md](../../../cooling_advisor/skill_specs/s2-inspect.md) |
| S3 Analyze | Data Analyst | [s3-analyze.md](../../../cooling_advisor/skill_specs/s3-analyze.md) |
| S4 Evaluate | Cooling Advisor | [s4-evaluate.md](../../../cooling_advisor/skill_specs/s4-evaluate.md) |
| S5 Compose | Supervisor | [s5-compose.md](../../../cooling_advisor/skill_specs/s5-compose.md) |

The earlier detailed planning files in this folder remain design references. Runtime authorization comes from Python and the current registry, not from prose in those drafts. Adding a new runtime skill requires a trusted instruction, registry binding, typed tool contracts, guard checks and behavioral validation. These are application-agent procedures, not installed Codex personal skills.
