# Monitoring

[Architecture index](../../architecture.md)

Runtime events record timestamp, case, sequence, actor, skill, action, outcome, detail and elapsed milliseconds. Python emits structured JSON logging and stores events inside the case. Each result also records tool/model counts, observed model tokens, overall latency, source provenance, policy, request, skill hashes and returned model versions.

The Monitoring API/dashboard aggregates only the authenticated owner's latest 20 cases. It reports completed/blocked cases, tool/model calls, token usage, runtime and G1/G2/G3 rejection/block events. Case history provides individual traces and review records.

This is application observability, not live BMS monitoring or an external uptime service. API authentication failures/rate limits are HTTP outcomes rather than persisted cases; server access logs capture status codes. There are no configured notifications, continuous polling, long-term aggregations, retention jobs, billing estimates or equipment alarms. Token counts are usage observations, not a guaranteed currency cost ceiling.
