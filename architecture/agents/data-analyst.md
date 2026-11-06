# Data Analyst Agent

> Earlier design detail. The implemented v0.2 runtime/contracts are documented in [runtime.md](runtime.md); type/tool examples below may describe the broader target.


[Architecture index](../../architecture.md)

The first specialist establishes what the available data actually supports.

## Skills

- [S2 — Inspect Cooling Data](skills/s2-inspect-cooling-data.md)
- [S3 — Analyze Cooling Performance](skills/s3-analyze-cooling-performance.md)

## Input and tools

Receive `AnalysisTask`, dataset/zone/window references, and baseline requirements. Use `read_telemetry`, `inspect_data_quality`, and `compute_baseline` through G2. Tools resolve references under server-enforced scope and return evidence IDs.

## Procedure

Run S2 before S3. If required timestamps, units, coverage, or temperature observability fail, return `insufficient_data`. Otherwise calculate the defined energy and temperature metrics and compare only compatible baseline windows.

## Output

Return `DataAnalysisResult`: quality status, calculated facts, baseline and scope, unresolved hypotheses, missing fields, and evidence IDs. Separate observed changes from possible causes. Tool-computed values remain authoritative; the model does not substitute new numbers.
