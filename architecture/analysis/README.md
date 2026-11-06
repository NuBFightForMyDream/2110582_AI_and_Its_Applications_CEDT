# Numerical Analysis

[Architecture index](../../architecture.md)

`analysis.py` owns data quality, energy metrics and matched comparisons. For each valid interval, energy is `interval_average_kW × interval_hours`. Sum cooling and IT separately, then add to total electricity. Component energies use the same integration. Percent reduction is `(baseline_total - option_total) / baseline_total × 100`; a zero baseline has no defined percentage.

Quality requires contiguous integer step order, complete declared horizon and a maximum inlet temperature in every interval. Pydantic rejects nonfinite values and negative power. Candidate feasibility additionally requires setpoint bounds and maximum-inlet threshold from the scenario's versioned study policy.

Matching requires identical scenario, snapshot, zone, horizon, source kind, electrical boundary and comparison fingerprint. An imported observational baseline has comparison disabled. Include baseline in all supported comparisons. Rank feasible evidence by total kWh and experiment ID for a deterministic tie-break. If baseline wins, return no_change. If none is feasible, return no_feasible_option. Unmatched alternatives return not_supported.

G3 independently rereads and recomputes this evidence after agent execution. Model-generated numbers and notes cannot enter these calculations. Source provenance and the exact policy/request/skill hashes used are retained in the case.
