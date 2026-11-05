# S4 — Evaluate Setpoint Plans
Owner: cooling_advisor. Version: 0.3.0.

Call `run_mpc`: it tries all 81 plans for the next hour (each 15-minute step: -1, 0 or +1 °C),
rejects any plan whose forecast hottest inlet + safety margin exceeds 27 °C and ranks the rest by
predicted cooling + IT energy. Then call `check_safety` with the first step of the cheapest safe
plan. Finally call `propose_setpoint` with the setpoint the safety filter returned.

Propose exactly what the tools support; never round, nudge or invent a value. If no plan is safe,
the MPC already returns the plan that cools the most — propose that. G3 independently re-reads the
sensors, re-runs the forecaster, the MPC and the safety filter, and rejects any other proposal.
There is no tool that changes equipment; a proposal is a recommendation.
