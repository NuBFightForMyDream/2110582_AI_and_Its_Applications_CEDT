# S2 — Inspect Sensor Data
Owner: data_analyst. Version: 0.3.0.

Call `read_sensors`, then `inspect_data_quality`. Tools only see readings at or before the case
time (the last 2 hours); nothing from the future is available.

The data is ready only if the reading at the case time and the one before it exist, all 12 rack
inlet sensors report (a single missing reading may be held from 15 minutes earlier), no sensor is
stuck (same exact value 6+ times), and supply air, IT power, cooling power and outdoor temperature
are present. Impossible values (e.g. 85 °C) are removed, not trusted.

If any check fails, finish the analysis and report the error codes. Never infer a rack's inlet
from its neighbours or from the average.
