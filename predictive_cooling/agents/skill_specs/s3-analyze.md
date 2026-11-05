# S3 — Analyze & Forecast
Owner: data_analyst. Version: 0.3.0.

After a clean inspection call `summarize_state` (hottest rack, headroom to 27 °C, IT and cooling
power, PUE, last-hour trend) and then `forecast_hold`: the trained gradient-boosting forecaster
predicts the hottest inlet and power for the next 60 minutes if the current setpoint is held,
including the safety margin and the correction learned from the last hour's forecast errors.

Finish with `finish_analysis` and a short note. Say whether holding the setpoint looks safe and
what drives the risk (for example the 01:00 GPU batch). Use only numbers returned by the tools.
The data is a mock 12-rack room; do not describe it as a measured facility.
