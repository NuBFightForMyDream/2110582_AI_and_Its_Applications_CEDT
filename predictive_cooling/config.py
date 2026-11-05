"""Every number the pipeline shares lives here, so the whole study is easy to audit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sensor_history.csv"
ARTIFACT_DIR = ROOT / "artifacts"

# Time grid: one control decision every 15 minutes, looking 60 minutes ahead.
STEP_MINUTES = 15
DT_H = STEP_MINUTES / 60
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
HORIZON = 4                      # 4 x 15 min = 60 min look-ahead
MOVES = (-1.0, 0.0, 1.0)         # lower / keep / raise the setpoint by 1 °C -> 3^4 = 81 plans

# Operating limits.
SP_MIN, SP_MAX = 16.0, 24.0      # allowed CRAH supply-air setpoint (°C)
INLET_LIMIT = 27.0               # ASHRAE recommended maximum rack inlet (°C)
EMERGENCY_MARGIN = 0.3           # measured inlet this close to the limit forces extra cooling

# Mock history: 13 weeks of a hot-climate (Bangkok-like) site, Mon 2 Feb 2026 onward.
START = "2026-02-02"
DAYS = 91
TRAIN_END = "2026-04-06"         # weeks 1-9 train the forecaster
VAL_END = "2026-04-20"           # weeks 10-11 tune baselines and safety margins; weeks 12-13 are the test
DATASET_SEED = 7

# Floor layout: two rows of six racks sharing one cold aisle; CRAH units sit at the left wall.
ROWS = {"A": ["A1", "A2", "A3", "A4", "A5", "A6"], "B": ["B1", "B2", "B3", "B4", "B5", "B6"]}
RACKS = ROWS["A"] + ROWS["B"]
RACK_ROLE = {**{r: "Compute" for r in ROWS["A"]}, "B1": "Storage", "B2": "Storage", "B3": "Web",
             "B4": "GPU", "B5": "GPU", "B6": "GPU"}
INLET_COLS = [f"inlet_{r}_c" for r in RACKS]

ELECTRICITY_THB_PER_KWH = 4.0    # same assumption as the proposal (section 8.4)
AI_SERVER_KW = 0.2               # power of the machine that runs the forecaster + MPC

# MPC tuning.
MARGIN_QUANTILE = 0.99           # safety margin = 99th percentile of forecast under-prediction (validation weeks)
MOVE_COST_KWH = 0.15             # small penalty per 1 °C setpoint change, avoids needless wiggling
TEST_SEED = 11                   # sensor-noise seed shared by every controller in the closed-loop tests
