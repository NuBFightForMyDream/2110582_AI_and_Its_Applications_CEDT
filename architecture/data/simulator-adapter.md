# Pinned SustainDC Adapter

[Data](README.md)

- Repository: https://github.com/HewlettPackard/dc-rl
- Commit: `a92b4755aca560e34a98d14028dda629eb968482`.
- Code used: `envs/datacenter.py`, `utils/dc_config_reader.py`, `utils/dc_config.json`.
- Workload: two upstream Alibaba traces or upstream Google trace (`cpu_load`).
- Weather: upstream EPW TMY files for Phoenix, New York or Seattle; dry-bulb temperature column.

`simulation.py` loads only the power/temperature submodel. It does not execute the full Gym/RL environment, battery or workload scheduler. Each candidate uses the same configured capacity, trace window, cooling sizing and seed. Hourly values are held over four 15-minute operating intervals. No thermal inertia is modeled and no executable transition sequence is inferred.

Two adapter adjustments are recorded in provenance: rack CPU configurations follow source JSON order rather than thread completion order; electrical accounting includes CRAC fan and CW/CT pump electricity omitted by upstream's narrower dc_total_power field. Electrical energy sums CPU, IT fan, CRAC fan, tower fan, compressor and pumps; thermal cooling load is not electrical power.

Temperature is the maximum of `rackwise_inlet_temp`, not the maximum rack outlet or an average room temperature. The source configuration and trace bytes are SHA-256 hashed. Git revision and tracked-file cleanliness are checked before executing. Candidate model objects use common random seeds in a separate process so the web server RNG is unaffected. A hard timeout ends failed jobs without catalog import.

The two bundled studies show lower setpoints dominating total modeled electricity. Report this limitation directly. No reinforcement-learning gain, measured site savings or safe deployment claim follows. The adapter is suitable for a reproducible decision-support demonstration, not yet a validated facility digital twin. Upstream license attribution is in `third_party/sustaindc/LICENSE`.
