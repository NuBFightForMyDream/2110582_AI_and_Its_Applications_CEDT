"""Predictive Cooling dashboard — one tab per pipeline step.

    streamlit run app.py          (or: python -m predictive_cooling)
"""
import os

# One thread per math library, set before numpy / scikit-learn load. Otherwise OpenMP/BLAS start one thread per
# host core; on Streamlit Cloud (many host cores, ~1 core of quota) that oversubscription used ~10x more CPU per
# rerun and got the app throttled. Measured locally: 3.5 s → 0.3 s CPU per rerun, and faster wall time.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from predictive_cooling import config as C
from predictive_cooling import forecast, pipeline, storage
from predictive_cooling.evaluate import scenarios
from predictive_cooling.mpc import MPC
from predictive_cooling.agents import AgentContext, CaseRequest, plans_table, run_case
from predictive_cooling.agents.skills import ROLES, SKILLS
from predictive_cooling.safety import RULES
from threadpoolctl import threadpool_limits

# The env vars above only help if numpy/sklearn were not loaded yet, and Streamlit runs every rerun in a new thread
# (OpenMP thread counts are per thread), so also cap the already-loaded pools on every run.
threadpool_limits(limits=1)

st.set_page_config(page_title="Predictive Cooling", page_icon=":material/ac_unit:", layout="wide")

# Deployed keys live in Streamlit secrets (.streamlit/secrets.toml locally, the Secrets box on Streamlit Cloud).
# The agent and storage code read environment variables, so copy the known keys across once.
for _name in ("GEMINI_API_KEY", "GEMINI_MODEL", "SUPABASE_URL", "SUPABASE_KEY"):
    try:
        if not os.getenv(_name) and _name in st.secrets:
            os.environ[_name] = str(st.secrets[_name])
    except Exception:                                   # no secrets file: environment variables only
        break

# ---------- look & feel ---------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CTRL = {"fixed": "#1baf7a", "rule": "#eb6834", "ai": "#2a78d6", "ai_unsafe": "#eda100"}
MODEL = {"persistence": MUTED, "linear": "#eb6834", "gbm": "#2a78d6"}
QTY = {"temp": "#6250d6", "it": "#e87ba4", "cooling": "#eda100"}
LIMIT = "#d03b3b"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

st.markdown(f"""<style>
.block-container {{padding-top: 1.6rem; max-width: 1280px;}}
.note {{color:{INK2}; font-size:13.5px;}}
</style>""", unsafe_allow_html=True)

# Mermaid class styles shared by every diagram: blue = ML/AI step, red = guard/safety, green = outcome.
MERMAID_STYLE = """
    classDef step fill:#ffffff,stroke:#c3c2b7,color:#0b0b0b
    classDef ai fill:#e8f1fc,stroke:#2a78d6,color:#0b0b0b,stroke-width:2px
    classDef guard fill:#fbeeee,stroke:#d03b3b,color:#0b0b0b
    classDef ok fill:#e6f6ef,stroke:#1baf7a,color:#0b0b0b
    classDef agent fill:#f1effc,stroke:#6250d6,color:#0b0b0b
"""


def diagram(body, alt):
    """A Mermaid diagram; flowcharts get the shared colour classes (other diagram types reject classDef)."""
    body = body.strip()
    st.mermaid_chart(body + "\n" + MERMAID_STYLE if body.startswith("flowchart") else body, alt=alt)


def in_short(*points, title="In short"):
    """A bordered box with the plain-language message of a tab."""
    with st.container(border=True):
        st.markdown(f":material/lightbulb: **{title}**")
        st.markdown("\n".join(f"- {p}" for p in points))


def style(fig, height=320, title=None, legend=True, hover="x unified"):
    subtitles = [a for a in fig.layout.annotations if a.xref == "paper" and a.yref == "paper" and a.yanchor == "bottom"]
    for a in subtitles:
        a.update(font=dict(size=13, color=INK2))
    top = (46 if title else 12) + (22 if subtitles and not title else 0) + (26 if legend and title and not subtitles else 0)
    bottom_legend = bool(subtitles) and legend
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=15, color=INK), y=0.98, yanchor="top") if title else None,
        margin=dict(l=8, r=8, t=top, b=64 if bottom_legend else 8), plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2, size=12.5), hovermode=hover, showlegend=legend, barcornerradius=4,
        legend=dict(orientation="h", yanchor="top" if bottom_legend else "bottom", y=-0.11 if bottom_legend else 1.02, x=0,
                    title=None, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, showline=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def limit_line(fig, row=None, col=None, label="27 °C limit"):
    kw = dict(row=row, col=col) if row else {}
    fig.add_hline(y=C.INLET_LIMIT, line=dict(color=LIMIT, width=1.5), annotation_text=label,
                  annotation_position="top left", annotation_font=dict(color=LIMIT, size=11), **kw)


def note_metric(col, label, value, note, help=None):
    """A metric whose small grey line is a note, not an up/down change."""
    try:
        col.metric(label, value, note, delta_color="off", delta_arrow="off", help=help)
    except TypeError:                                   # Streamlit versions without delta_arrow
        col.metric(label, value, note, delta_color="off", help=help)


# ---------- data ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle():
    bundle = pipeline.load(rebuild_if_stale=False)
    if bundle is None:
        with st.status("First run: generating data, training models and running the plant tests (1-3 min)…",
                       expanded=True) as status:
            bundle = pipeline.build(workers=4 if sys.platform == "linux" else 1, log=st.write)
            status.update(label="Ready", state="complete")
    return bundle


B = get_bundle()
SC = scenarios()
NAMES = B["controller_names"]
LOGS = B["logs"]
CLEAN, RAW = B["clean"], B["raw"]
TEST_DAYS = sorted({d for d in LOGS["normal", "ai"].index.date})
HEAD = pipeline.headline(B)

# ---------- header --------------------------------------------------------------------------
st.title("Predictive cooling for a data center", icon=":material/ac_unit:")
st.markdown(
    "<p class='note'>Server rooms waste electricity by cooling more than they need. This project <b>forecasts the next "
    "hour</b> and picks the warmest cooling setpoint that still keeps every rack below <b>27 °C</b>. Three AI agents "
    "explain the decision to an operator. All data is a <b>mock</b> 12-rack room from a physics model; nothing controls "
    "real equipment.</p>", unsafe_allow_html=True)

ai = HEAD["ai"]


def signed(x, digits=1):
    return f"{x:+.{digits}f} %".replace("-", "−")


# ---- the idea in 30 seconds: problem → method → result ----
with st.container(horizontal=True, gap="small"):
    for icon, head, body in [
        (":material/thermostat:", "The problem",
         "Cooling runs at a safe but cold setpoint all day. Colder than needed = wasted electricity."),
        (":material/insights:", "The idea",
         "Every 15 min: forecast the next hour, test **81 setpoint plans**, keep only the safe ones, pick the cheapest."),
        (":material/verified_user:", "The result",
         f"**{HEAD['rule']['cooling_pct']:.1f} % less cooling energy** than a tuned rule, "
         f"**{ai['violation_minutes']:.0f} minutes** above 27 °C."),
    ]:
        with st.container(border=True):
            st.markdown(f"{icon} **{head}**")
            st.caption(body)

with st.container(border=True):
    st.markdown(":material/account_tree: **How it works** · the 15-minute control loop (blue = machine learning, red = safety)")
    diagram("""
flowchart LR
    S["① Sensors<br/>12 rack inlets, power,<br/>weather · every 15 min"]:::step --> CL["② Clean<br/>gaps, glitches,<br/>stuck sensors"]:::step
    CL --> F["③ Forecast (ML)<br/>next 60 min for<br/>any setpoint plan"]:::ai
    F --> M["④ MPC<br/>try 81 plans → keep safe<br/>→ pick cheapest"]:::ai
    M --> SF["⑤ Safety filter<br/>4 rules check<br/>every command"]:::guard
    SF --> P["⑥ Room<br/>apply the setpoint,<br/>repeat in 15 min ↺"]:::ok
""", alt="Control loop: sensors, clean, forecast, MPC, safety filter, room, then repeat every 15 minutes")
    with st.expander("Agent layer: who answers the operator's question", icon=":material/smart_toy:"):
        diagram("""
flowchart LR
    Q(["Operator question"]):::step --> G1["G1 check request"]:::guard
    G1 --> SUP["Supervisor<br/>plans the case"]:::agent
    SUP --> DA["Data Analyst<br/>reads sensors + forecast<br/>(steps ①–③)"]:::agent
    DA --> CA["Cooling Advisor<br/>81-plan MPC + safety<br/>(steps ④–⑤)"]:::agent
    CA --> G3["G3 recompute<br/>everything, reject<br/>if it does not match"]:::guard
    G3 --> A(["Verified answer<br/>(recommend only)"]):::ok
""", alt="Agent layer: operator question, G1, Supervisor, Data Analyst, Cooling Advisor, G3 verification, verified answer")
        st.caption("Every tool call passes guard G2 (allowed tools for that skill, budget, no peeking past the chosen time).")

k = st.columns(5)
for col, label, value, note, icon, hlp in [
    (k[0], "Cooling energy", signed(-HEAD['rule']['cooling_pct']), "vs tuned rule", ":material/eco:",
     "AI vs the tuned rule-based controller, two test weeks, normal operation"),
    (k[1], "Cooling energy", signed(-HEAD['fixed']['cooling_pct']), f"vs fixed {B['fixed_sp']:.0f} °C", ":material/eco:",
     "AI vs the fixed setpoint, two test weeks, normal operation"),
    (k[2], "Whole facility", signed(-HEAD['rule']['facility_pct'], 2),
     f"≈ {HEAD['rule']['yearly_thb'] / 1000:,.0f}k THB / year", ":material/savings:",
     f"vs tuned rule. Includes the AI server ({C.AI_SERVER_KW:g} kW, 24/7). {C.ELECTRICITY_THB_PER_KWH:g} THB/kWh."),
    (k[3], "Minutes > 27 °C", f"{ai['violation_minutes']:.0f}", f"hottest {ai['hottest_inlet_c']:.1f} °C",
     ":material/shield:", "Minutes any rack inlet was above the 27 °C limit with the AI in control"),
    (k[4], "Decision time", f"{ai['decision_ms_mean']:.0f} ms", "target < 10 s", ":material/speed:", None),
]:
    col.metric(label, value, note, delta_color="off", delta_arrow="off", help=hlp, border=True, icon=icon)

agent_tab, *tabs = st.tabs([":material/smart_toy: Ask the agents", ":material/sensors: 1 · Sensors",
                            ":material/cleaning_services: 2 · Clean data", ":material/insights: 3 · Forecast",
                            ":material/route: 4 · MPC decision", ":material/shield: 5 · Safety",
                            ":material/bolt: 6 · Results"])


# ---------- ① sensors -----------------------------------------------------------------------
def floor_map(values: dict, title):
    """values: rack -> inlet °C (may be NaN). Rows A (top) and B (bottom) share the cold aisle."""
    z = [[values.get(r, np.nan) for r in C.ROWS["A"]], [np.nan] * 6, [values.get(r, np.nan) for r in C.ROWS["B"]]]
    scale = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8a89"], [1.0, "#a82e2e"]]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(6)), y=["Row A", "Cold aisle", "Row B"],
                               zmin=18, zmax=28, colorscale=scale, xgap=4, ygap=4, hoverinfo="skip",
                               colorbar=dict(title=dict(text="Inlet °C"), thickness=10, len=0.9)))
    for row_i, row in [(0, "A"), (2, "B")]:
        for col_i, rack in enumerate(C.ROWS[row]):
            v = values.get(rack, np.nan)
            text = f"<b>{rack}</b><br>{'—' if np.isnan(v) else f'{v:.1f} °C'}<br><span style='font-size:10px'>{C.RACK_ROLE[rack]}</span>"
            light = np.isnan(v) or 20.5 <= v <= 25.5
            fig.add_annotation(x=col_i, y=row_i, text=text, showarrow=False,
                               font=dict(color=INK if light else "white", size=12))
    fig.add_annotation(x=2.5, y=1, text="cold aisle  ←  cold air from the CRAH units on the left wall", showarrow=False,
                       font=dict(color=INK2, size=12))
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False)
    fig.update_xaxes(showgrid=False, showline=False, showticklabels=False)
    return style(fig, height=300, title=title, legend=False, hover=False)


with tabs[0]:
    st.subheader("What the room tells us every 15 minutes", icon=":material/sensors:")
    in_short("12 racks in two rows face one **cold aisle**; cold air comes from the CRAH units on the left wall.",
             "Each rack's **inlet temperature** must stay at or below **27 °C**. Red on the map = close to the limit.",
             "Rack **B6** (far end, GPU) is always the hottest, so it decides how warm the whole room can run.")
    st.markdown("<p class='note'>13 weeks of history (2 Feb – 3 May 2026), 96 readings a day. Pick a moment to see the "
                "room.</p>", unsafe_allow_html=True)
    left, right = st.columns([1, 2.2])
    with left:
        day = st.date_input("Day", value=TEST_DAYS[1], min_value=CLEAN.index.min().date(), max_value=CLEAN.index.max().date())
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
        clock = st.select_slider("Time", options=times, value="01:15")
    ts = pd.Timestamp(f"{day} {clock}")
    reading = CLEAN.loc[ts] if ts in CLEAN.index else None
    with right:
        if reading is None or pd.isna(reading["max_inlet_c"]):
            st.warning("No complete reading at this time (logger gap). Try another time.")
        else:
            show(floor_map({r: reading[f"inlet_{r}_c"] for r in C.RACKS}, f"Rack inlet temperatures · {ts:%a %d %b %H:%M}"))
    if reading is not None and not pd.isna(reading["max_inlet_c"]):
        m = st.columns(6)
        m[0].metric("IT power", f"{reading['it_power_kw']:.0f} kW")
        m[1].metric("Cooling power", f"{reading['cooling_kw']:.0f} kW")
        m[2].metric("PUE", f"{reading['pue']:.2f}", help="Facility power ÷ IT power. 1.0 would be perfect.")
        m[3].metric("Outdoor", f"{reading['outdoor_temp_c']:.1f} °C")
        m[4].metric("Supply-air setpoint", f"{reading['crah_setpoint_c']:.0f} °C")
        note_metric(m[5], "Hottest inlet", f"{reading['max_inlet_c']:.1f} °C", f"rack {reading['hottest_rack']}")

    day_frame = CLEAN.loc[str(day)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.33, 0.25],
                        subplot_titles=("Temperature (°C)", "Power (kW)", "Outdoor (°C)"))
    fig.add_scatter(x=day_frame.index, y=day_frame["max_inlet_c"], name="Hottest rack inlet", line=dict(color=QTY["temp"], width=2), row=1, col=1)
    fig.add_scatter(x=day_frame.index, y=day_frame["crah_setpoint_c"], name="Supply-air setpoint", line=dict(color=INK2, width=2, shape="hv"), row=1, col=1)
    limit_line(fig, 1, 1)
    fig.add_scatter(x=day_frame.index, y=day_frame["it_power_kw"], name="IT power", line=dict(color=QTY["it"], width=2), row=2, col=1)
    fig.add_scatter(x=day_frame.index, y=day_frame["cooling_kw"], name="Cooling power", line=dict(color=QTY["cooling"], width=2), row=2, col=1)
    fig.add_scatter(x=day_frame.index, y=day_frame["outdoor_temp_c"], name="Outdoor air", line=dict(color=MUTED, width=2), row=3, col=1)
    fig.add_vline(x=ts, line=dict(color=AXIS, width=1))
    show(style(fig, height=520, title=f"The whole day · {pd.Timestamp(day):%A %d %B}"))
    st.caption("The 01:00–05:00 jump on weekdays is the nightly GPU training batch; the afternoon bump is office load "
               "plus a reporting job. The old operator rule (and occasional setpoint step tests) chose the setpoint.")

    with st.expander("Sensor dictionary and raw file"):
        dictionary = pd.DataFrame([
            ("timestamp", "", "Reading time, every 15 minutes"),
            ("outdoor_temp_c", "°C", "Outdoor air temperature (weather station)"),
            ("it_load_pct", "%", "Average CPU/GPU utilisation of all racks"),
            ("it_power_kw", "kW", "Power drawn by servers incl. their fans (UPS output meter)"),
            ("inlet_A1_c … inlet_B6_c", "°C", "Air temperature at the front (inlet) of each rack — the safety limit applies here"),
            ("supply_air_c", "°C", "Air temperature leaving the CRAH units"),
            ("return_air_c", "°C", "Hot-aisle air returning to the CRAH units"),
            ("crah_setpoint_c", "°C", "Supply-air setpoint commanded at this time — the one knob the AI controls"),
            ("crah_fan_pct", "%", "CRAH fan speed (automatic, follows IT load)"),
            ("chiller_kw / crah_fan_kw / pump_tower_kw", "kW", "Cooling sub-meters"),
            ("cooling_kw", "kW", "Total cooling power"),
            ("facility_kw", "kW", "Whole facility: IT + cooling + UPS losses + lighting"),
            ("pue", "", "facility_kw ÷ it_power_kw"),
        ], columns=["Column", "Unit", "Meaning"])
        st.dataframe(dictionary, hide_index=True, width="stretch")
        st.download_button("Download sensor_history.csv", C.DATA_PATH.read_bytes(), "sensor_history.csv", "text/csv")
        st.dataframe(RAW.head(200), width="stretch", height=260)


# ---------- ② clean -------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Real logs are messy — fix them before training", icon=":material/cleaning_services:")
    in_short("Sensors drop out, send impossible values (85 °C) or get stuck. Training on that teaches the model nonsense.",
             "Four rules clean the data; then it is split **by time** so the model is tested on weeks it never saw.")
    diagram("""
flowchart LR
    R["Raw log"]:::step --> A["1 · back on the<br/>15-min grid"]:::step --> B["2 · impossible<br/>values → missing"]:::guard
    B --> S["3 · stuck sensor<br/>(6+ repeats) → missing"]:::guard --> F["4 · fill holes<br/>≤ 1 hour"]:::step
    F --> T["Train<br/>weeks 1–9"]:::ai
    F --> V["Validate<br/>weeks 10–11"]:::step
    F --> X["Test<br/>weeks 12–13"]:::ok
""", alt="Cleaning rules: grid, impossible values, stuck sensors, fill short holes, then a time-based train, validation and test split")
    q = B["quality"]
    m = st.columns(5)
    m[0].metric("Readings on the grid", f"{len(CLEAN):,}")
    m[1].metric("Logger gaps", f"{q.attrs.get('gap_rows', 0)} rows", help="Missing timestamps put back on the 15-min grid")
    m[2].metric("Impossible values", f"{int(q['impossible'].sum())}", help="e.g. 85 °C or −40 °C from a bad packet")
    m[3].metric("Stuck-sensor readings", f"{int(q['stuck'].sum())}", help="Same exact value 6+ times in a row")
    m[4].metric("Short holes filled", f"{int(q['filled'].sum())}", help="Interpolated when the hole is ≤ 1 hour")
    st.markdown("<p class='note'>Rules: <b>1</b> put readings back on the 15-minute grid · <b>2</b> values outside the "
                "physical range become missing · <b>3</b> a sensor repeating the exact same value 6+ times is stuck → missing · "
                "<b>4</b> holes up to 1 hour are interpolated, longer ones stay missing and those rows are not used for training.</p>",
                unsafe_allow_html=True)
    issues = q[(q[["impossible", "stuck", "missing"]].sum(axis=1)) > 0].sort_values("stuck", ascending=False)
    sensor = st.selectbox("Inspect a sensor", list(issues.index), index=0)
    bad = RAW[sensor].isna() | (RAW[sensor] != CLEAN[sensor].reindex(RAW.index))
    worst_day = (bad.groupby(RAW.index.date).sum().idxmax())
    raw_day, clean_day = RAW.loc[str(worst_day), sensor], CLEAN.loc[str(worst_day), sensor]
    fig = go.Figure()
    fig.add_scatter(x=raw_day.index, y=raw_day, name="Raw log", mode="lines+markers", line=dict(color=MUTED, width=1.5), marker=dict(size=4))
    fig.add_scatter(x=clean_day.index, y=clean_day, name="Cleaned", line=dict(color=QTY["temp"], width=2))
    removed = raw_day[bad.loc[str(worst_day)] & raw_day.notna()]
    fig.add_scatter(x=removed.index, y=removed, name="Removed", mode="markers", marker=dict(color=LIMIT, size=9, symbol="x"))
    low, high = clean_day.min(), clean_day.max()
    fig.update_yaxes(range=[low - 2, high + 2] if pd.notna(low) else None)
    show(style(fig, height=320, title=f"{sensor} on {pd.Timestamp(worst_day):%a %d %b} (the day with most problems)"))
    st.caption("Glitch values far outside the axis are removed too; the axis is zoomed to the real signal.")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.dataframe(issues.astype(int), width="stretch", height=280)
    with c2:
        split = pd.DataFrame([("Train the forecaster", C.START, C.TRAIN_END, "weeks 1-9"),
                              ("Validation: tune baselines + safety margins", C.TRAIN_END, C.VAL_END, "weeks 10-11"),
                              ("Test: score everything", C.VAL_END, str(CLEAN.index.max().date() + pd.Timedelta(days=1)), "weeks 12-13")],
                             columns=["Part", "Start", "End", "Weeks"])
        fig2 = go.Figure()
        start = pd.Timestamp(C.START)
        for i, row in split.iterrows():
            s, e = pd.Timestamp(row["Start"]), pd.Timestamp(row["End"])
            fig2.add_bar(y=["Data"], x=[(e - s).days], base=[(s - start).days], orientation="h", width=0.45,
                         marker_color=[MUTED, "#86b6ef", "#2a78d6"][i], name=f"{row['Part']} ({row['Weeks']})",
                         hovertemplate=f"{row['Part']}<br>{s:%d %b} – {(e - pd.Timedelta(days=1)):%d %b}<extra></extra>")
        fig2.update_layout(barmode="overlay")
        fig2.update_xaxes(title="days since 2 Feb 2026")
        fig2.update_yaxes(showticklabels=False)
        style(fig2, height=280, title="Chronological split (never shuffle time series)", hover="closest")
        fig2.update_layout(legend=dict(orientation="v", y=-0.45, yanchor="top"), margin=dict(b=90, t=46))
        show(fig2)


# ---------- ③ forecast ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Can the model see the next hour?", icon=":material/insights:")
    _acc = B["accuracy"]
    gbm_mae_60 = _acc[(_acc["kind"] == "gbm") & (_acc["horizon_min"] == 60)]["inlet_mae_c"].iloc[0]
    in_short("The model answers: *if we use this setpoint plan, how hot will the hottest rack be, and how much power will we "
             "use, 15–60 minutes from now?*",
             f"**Gradient boosting** errs by about **{gbm_mae_60:.2f} °C** "
             "one hour ahead; just copying the current value (persistence) is several times worse.",
             "Its edge: it learned the **01:00 GPU batch** from history, so it sees the heat coming before it arrives.")
    diagram("""
flowchart LR
    I["Now: sensors, time of day,<br/>weekday/weekend"]:::step --> M["Forecaster (ML)<br/>12 small models"]:::ai
    P["A setpoint plan<br/>for the next 4 × 15 min"]:::step --> M
    M --> O1["Hottest rack inlet"]:::ok
    M --> O2["Cooling power"]:::ok
    M --> O3["IT power"]:::ok
""", alt="Forecaster inputs (current sensors and a setpoint plan) and outputs (hottest inlet, cooling power, IT power)")
    st.markdown("<p class='note'>Inputs: current sensors (hottest/average inlet, supply air, IT and cooling power, outdoor "
                "temperature, time of day, weekend) <b>+ the setpoint plan for the next 4 × 15 min</b>. Outputs at +15/30/45/60 min: "
                "hottest inlet, cooling power, IT power. Scored on the 2 test weeks the model never saw.</p>", unsafe_allow_html=True)
    acc = B["accuracy"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for kind, label in forecast.Forecaster.KINDS.items():
            part = acc[acc["kind"] == kind]
            fig.add_scatter(x=part["horizon_min"], y=part["inlet_mae_c"], name=label, mode="lines+markers",
                            line=dict(color=MODEL[kind], width=2, dash="dot" if kind == "persistence" else None),
                            marker=dict(size=9, line=dict(color=SURFACE, width=2)))
            last = part.iloc[-1]
            fig.add_annotation(x=last["horizon_min"], y=last["inlet_mae_c"], text=f"{last['inlet_mae_c']:.2f}", xshift=22,
                               showarrow=False, font=dict(color=INK2, size=11))
        fig.add_hline(y=1.0, line=dict(color=AXIS, width=1), annotation_text="proposal target ≤ 1 °C at 15 min",
                      annotation_position="top left", annotation_font=dict(size=11, color=INK2))
        fig.update_xaxes(title="forecast horizon (minutes ahead)", tickvals=[15, 30, 45, 60])
        fig.update_yaxes(title="mean absolute error (°C)", rangemode="tozero")
        show(style(fig, height=340, title="Hottest-inlet error grows with horizon — less for ML"))
    with c2:
        table = acc.pivot(index="model", columns="horizon_min", values="inlet_mae_c").reindex(list(forecast.Forecaster.KINDS.values()))
        table.columns = [f"{c} min" for c in table.columns]
        st.markdown("**Hottest inlet, MAE (°C)** — target ≤ 1 °C at 15 min")
        st.dataframe(table.style.format("{:.2f}"), width="stretch")
        power = acc[acc["horizon_min"].isin([15, 60])].pivot(index="model", columns="horizon_min",
                                                               values=["cooling_nrmse_pct", "it_nrmse_pct"])
        power = power.reindex(list(forecast.Forecaster.KINDS.values()))
        power.columns = [f"{'Cooling' if a.startswith('cooling') else 'IT'} {h} min" for a, h in power.columns]
        st.markdown("**Power, NRMSE (% of mean)** — target ≤ 10 %")
        st.dataframe(power.style.format("{:.1f}"), width="stretch")

    horizon = st.segmented_control("Forecast made this far ahead", [15, 30, 45, 60], default=60, format_func=lambda m: f"{m} min")
    h = (horizon or 60) // 15
    sample = B["samples"][h]
    test_day = st.selectbox("Test day", TEST_DAYS, index=1, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="fc_day")
    part = sample.loc[str(test_day)]
    fig = go.Figure()
    fig.add_scatter(x=part.index, y=part["actual_max_inlet_c"], name="Actual", line=dict(color=INK, width=2))
    fig.add_scatter(x=part.index, y=part["persistence_max_inlet_c"], name="Persistence", line=dict(color=MUTED, width=2, dash="dot"))
    fig.add_scatter(x=part.index, y=part["gbm_max_inlet_c"], name="Gradient boosting (ML)", line=dict(color=MODEL["gbm"], width=2))
    fig.update_yaxes(title="hottest inlet (°C)")
    show(style(fig, height=330, title=f"Forecast made {h * 15} min earlier vs what happened"))
    st.caption("Persistence is always late: it only notices the 01:00 GPU batch after it has started. "
               "The ML model learned from history that the batch starts at 01:00 on weekdays.")

    c1, c2 = st.columns([1, 1.1])
    with c1:
        imp = B["importance"][1 if h == 1 else C.HORIZON].head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=MODEL["gbm"], width=0.55,
                               hovertemplate="%{y}: %{x:.3f} °C<extra></extra>"))
        fig.update_xaxes(title="error increase when shuffled (°C)")
        show(style(fig, height=330, title=f"What the {'15' if h == 1 else '60'}-min model looks at", legend=False, hover="closest"))
    with c2:
        st.markdown("**Try it: what if we used this plan?**")
        test_frame = CLEAN.loc[C.VAL_END:]
        states = forecast.state_features(test_frame).dropna()
        when = st.select_slider("Moment (test weeks)", options=list(states.index), value=states.index[states.index.get_indexer([pd.Timestamp(f'{TEST_DAYS[1]} 00:30')], method='nearest')[0]],
                                format_func=lambda t: f"{t:%a %d %b %H:%M}")
        state = states.loc[when]
        now_sp = float(state["setpoint_prev_c"])
        cols = st.columns(4)
        plan = [cols[i].number_input(f"+{15 * i} min", min_value=C.SP_MIN, max_value=C.SP_MAX, value=now_sp, step=1.0,
                                     format="%.0f", key=f"wi{i}")
                for i in range(4)]
        pred = B["forecaster"].predict_plans(state, np.array([plan]))
        times = [when + pd.Timedelta(minutes=15 * (i + 1)) for i in range(4)]
        fig = go.Figure()
        fig.add_scatter(x=[when] + times, y=[state["max_inlet_c"]] + list(pred["max_inlet_c"][0]), name="Predicted hottest inlet",
                        mode="lines+markers", line=dict(color=MODEL["gbm"], width=2), marker=dict(size=8))
        fig.add_scatter(x=[when] + times, y=[state["max_inlet_c"]] + list(pred["max_inlet_c"][0] + np.array(B["margins"])),
                        name="+ safety margin", mode="lines", line=dict(color=MODEL["gbm"], width=1, dash="dot"))
        fig.add_scatter(x=[when] + times, y=plan + plan[-1:], name="Setpoint plan", line=dict(color=INK2, width=2, shape="hv"))
        limit_line(fig)
        energy = float((pred["cooling_kw"][0] + pred["it_power_kw"][0]).sum() * C.DT_H)
        show(style(fig, height=290, title=f"Predicted energy next hour: {energy:,.1f} kWh"))


# ---------- ④ MPC ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("How the controller chooses a setpoint", icon=":material/route:")
    in_short("Think of it like GPS: try every route for the next hour, drop the ones that overheat, take the cheapest.",
             "Only the **first step** is applied. 15 minutes later it plans again with fresh data.")
    diagram("""
flowchart LR
    N["Setpoint now"]:::step --> C["Each 15 min:<br/>−1, 0 or +1 °C"]:::step
    C --> P["4 steps → 3⁴ =<br/>81 plans"]:::step
    P --> F["Forecast every plan"]:::ai
    F --> D{"peak + safety margin<br/>≤ 27 °C?"}:::guard
    D -- no --> X["Rejected"]:::guard
    D -- yes --> E["Cheapest safe plan<br/>(least kWh)"]:::ok
    E --> A["Apply only<br/>its first step"]:::ok
""", alt="MPC: 3 choices per step over 4 steps make 81 plans; forecast each, reject unsafe, choose cheapest, apply first step")
    st.markdown("<p class='note'>Each 15-minute step can go <b>−1, 0 or +1 °C</b>; four steps make <b>3⁴ = 81 plans</b>. "
                "The forecaster scores each plan, plans whose predicted hottest inlet + safety margin crosses 27 °C are "
                "thrown out, and the cheapest remaining plan wins. Only its first step is applied; 15 minutes later the "
                "controller looks again with fresh sensor data (and corrects itself for its last forecast error).</p>",
                unsafe_allow_html=True)
    ai_log = LOGS["normal", "ai"]
    c1, c2 = st.columns([1, 2])
    with c1:
        d = st.selectbox("Test day", TEST_DAYS, index=1, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="mpc_day")
        clock = st.select_slider("Decision time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)],
                                 value="00:30", key="mpc_time")
    t = pd.Timestamp(f"{d} {clock}")
    row = ai_log.loc[t]
    state = pd.Series({c: row[f"x_{c}"] for c in forecast.STATE_FEATURES})
    mpc = MPC(B["forecaster"], B["margins"])
    plans = mpc.evaluate_plans(state, row["setpoint_in_use"], row["bias_c"])
    safe = plans[plans["safe"]]
    best = safe["cost"].idxmin() if len(safe) else plans[plans["within_limits"]]["peak_guarded_c"].idxmin()
    chosen = plans.loc[best]
    with c1:
        st.metric("Setpoint in use", f"{row['setpoint_in_use']:.0f} °C")
        st.metric("Hottest inlet now", f"{row['measured_max_inlet_c']:.1f} °C")
        st.metric("Safe plans", f"{len(safe)} of 81")
        note_metric(st, "Chosen plan", chosen["plan"].replace(" ", " → ") + " °C", f"apply {chosen['first_sp']:.0f} °C now")
        sent = row["command"]
        if sent != chosen["first_sp"]:
            st.info(f"The safety filter changed it to {sent:.0f} °C ({row['rules']}).")
        st.caption(f"Feedback correction in use: {row['bias_c']:+.2f} °C · margins "
                   + ", ".join(f"{m:.2f}" for m in B["margins"]) + " °C at +15…60 min")
    with c2:
        fig = go.Figure()
        for flag, name, color in [(False, "Too hot (rejected)", AXIS), (True, "Safe", MODEL["gbm"])]:
            part = plans[(plans["safe"] == flag) & plans["within_limits"]]
            fig.add_scatter(x=part["peak_guarded_c"], y=part["energy_kwh"], mode="markers", name=name,
                            marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
                            customdata=part[["plan"]].to_numpy(),
                            hovertemplate="plan %{customdata[0]} °C<br>peak %{x:.2f} °C (with margin)<br>%{y:.1f} kWh<extra></extra>")
        fig.add_scatter(x=[chosen["peak_guarded_c"]], y=[chosen["energy_kwh"]], mode="markers+text", name="Chosen",
                        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.5)),
                        text=[f"  chosen: {chosen['plan']}"], textposition="middle right", textfont=dict(color=INK, size=12),
                        hoverinfo="skip")
        fig.add_vline(x=C.INLET_LIMIT, line=dict(color=LIMIT, width=1.5), annotation_text="27 °C limit",
                      annotation_font=dict(color=LIMIT, size=11))
        fig.update_xaxes(title="predicted peak hottest inlet in the next hour, incl. safety margin (°C)")
        fig.update_yaxes(title="predicted energy next hour (kWh)")
        show(style(fig, height=360, title=f"81 plans at {t:%a %d %b %H:%M}", hover="closest"))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted hottest inlet for every plan (°C)", "Setpoint of the chosen plan (°C)"))
    future = [t + pd.Timedelta(minutes=15 * (i + 1)) for i in range(C.HORIZON)]
    for i, p in plans[plans["within_limits"]].iterrows():
        if i == best:
            continue
        fig.add_scatter(x=[t] + future, y=[row["measured_max_inlet_c"]] + [p[f"inlet_{k}"] for k in range(1, 5)], mode="lines",
                        line=dict(color=AXIS, width=1), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=[t] + future, y=[row["measured_max_inlet_c"]] + [chosen[f"inlet_{k}"] for k in range(1, 5)], name="Chosen plan",
                    mode="lines+markers", line=dict(color=MODEL["gbm"], width=2.5), marker=dict(size=8), row=1, col=1)
    limit_line(fig, 1, 1)
    sps = [chosen[f"sp_{k}"] for k in range(1, 5)]
    sps = sps + sps[-1:]                       # step k is held from t+15(k-1) to t+15k
    fig.add_scatter(x=[t] + future, y=sps, name="Chosen setpoints", line=dict(color=INK2, width=2, shape="hv"), row=1, col=2)
    fig.update_yaxes(range=[min(sps) - 2, max(sps) + 2], dtick=1, row=1, col=2)
    show(style(fig, height=300, legend=False))

    day_log = ai_log.loc[str(d)]
    fig = go.Figure()
    fig.add_scatter(x=day_log.index, y=day_log["max_inlet_c"], name="Hottest inlet (true)", line=dict(color=QTY["temp"], width=2))
    fig.add_scatter(x=day_log.index, y=day_log["command"], name="AI setpoint", line=dict(color=CTRL["ai"], width=2, shape="hv"))
    fig.add_scatter(x=day_log.index, y=day_log["it_load_pct"] / 10 + 10, name="IT load (scaled: 10 + load/10)",
                    line=dict(color=AXIS, width=1.5), visible="legendonly")
    limit_line(fig)
    fig.add_vline(x=t, line=dict(color=INK2, width=1))
    show(style(fig, height=300, title="The AI's whole day: it pre-cools before 01:00 and runs warmer when racks are quiet"))


# ---------- ⑤ safety ------------------------------------------------------------------------
with tabs[4]:
    st.subheader("What if something goes wrong?", icon=":material/shield:")
    in_short("The AI only **proposes**. Four simple rules can override it before anything is sent.",
             f"When in doubt, fall back to the old proven setpoint (**{B['fixed_sp']:.0f} °C**) — never worse than before.")
    diagram("""
flowchart LR
    P["MPC proposal"]:::ai --> R1{"R1 · sensor<br/>missing?"}:::guard
    R1 -- yes --> FB["Fallback setpoint"]:::ok
    R1 -- no --> R2{"R2 · conditions never<br/>seen in training?"}:::guard
    R2 -- yes --> FB
    R2 -- no --> R3{"R3 · rack already<br/>near 27 °C?"}:::guard
    R3 -- yes --> CO["Cool 2 °C now"]:::ok
    R3 -- no --> R4["R4 · max ±1 °C per step,<br/>stay within 16–24 °C"]:::guard
    R4 --> SEND["Send command"]:::ok
""", alt="Safety filter: R1 missing sensor and R2 unfamiliar conditions go to fallback, R3 near limit cools 2 degrees, R4 limits rate and range")
    st.markdown("<p class='note'>The MPC only <i>proposes</i>. Four rules check every command before it is sent. "
                f"Fallback = the proven fixed setpoint of <b>{B['fixed_sp']:.0f} °C</b>, so losing the AI returns the room to how "
                "it was run before. Each fault scenario below runs all controllers through the same two test weeks.</p>",
                unsafe_allow_html=True)
    fired = {}
    for key in SC:
        rules = LOGS[key, "ai"]["rules"].fillna("").str.split(",").explode()
        fired[key] = rules[rules != ""].value_counts()
    cols = st.columns(4)
    for col, (rule, text) in zip(cols, RULES.items()):
        with col.container(border=True):
            st.markdown(f"**{rule.split('_')[0]}** · {text}")
            st.caption(" · ".join(f"{SC[s].title}: {int(fired[s].get(rule, 0))}" for s in SC))
    pick = st.segmented_control("Scenario", ["sensor_failure", "plant_change", "load_surge", "normal"], default="sensor_failure",
                                format_func=lambda s: SC[s].title) or "sensor_failure"
    scen = SC[pick]
    st.markdown(f"<p class='note'><b>{scen.title}.</b> {scen.story}</p>", unsafe_allow_html=True)
    keys = ["fixed", "rule", "ai"] + (["ai_unsafe"] if (pick, "ai_unsafe") in LOGS else [])
    table = B["summary"].set_index(["scenario", "key"]).loc[pick].loc[keys]
    show_cols = {"controller": "Controller", "violation_minutes": "Minutes > 27 °C", "hottest_inlet_c": "Hottest inlet (°C)",
                 "degree_hours_over": "°C·h over limit", "fallback_steps": "Fallback steps", "emergency_steps": "Emergency steps",
                 "cooling_kwh": "Cooling kWh"}
    st.dataframe(table[list(show_cols)].rename(columns=show_cols).style.format(
        {"Hottest inlet (°C)": "{:.2f}", "°C·h over limit": "{:.2f}", "Cooling kWh": "{:,.0f}"}), hide_index=True, width="stretch")
    if scen.highlight:
        center, hours = pd.Timestamp(scen.highlight[0][0]), scen.highlight[0][1]
    else:
        worst = LOGS[pick, "ai"]["max_inlet_c"].idxmax()
        center, hours = worst - pd.Timedelta(hours=2), 2
    lo, hi = center - pd.Timedelta(hours=5), center + pd.Timedelta(hours=hours + 5)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("True hottest inlet (°C)", "Setpoint sent (°C)"))
    for key in keys:
        part = LOGS[pick, key].loc[lo:hi]
        dash = "dash" if key == "ai_unsafe" else None
        fig.add_scatter(x=part.index, y=part["max_inlet_c"], name=NAMES[key], line=dict(color=CTRL[key], width=2, dash=dash), row=1, col=1)
        fig.add_scatter(x=part.index, y=part["command"], name=NAMES[key], showlegend=False,
                        line=dict(color=CTRL[key], width=2, shape="hv", dash=dash), row=2, col=1)
    for start, length in scen.highlight:
        s = pd.Timestamp(start)
        if lo <= s <= hi:
            fig.add_vrect(x0=s, x1=s + pd.Timedelta(hours=length), fillcolor="#f0efec", opacity=0.8, line_width=0, layer="below",
                          annotation_text="fault", annotation_position="top left", annotation_font=dict(size=11, color=INK2))
    limit_line(fig, 1, 1)
    show(style(fig, height=480, title=f"Zoom on the first event · {center:%a %d %b}"))
    takeaways = {
        "sensor_failure": "The rule-based controller trusts the remaining (cooler) sensors and lets B5/B6 overheat. The AI "
                          "without a safety filter does the same. With the filter, missing data → fallback setpoint → no overheating.",
        "plant_change": "The room no longer matches the training data. The fixed setpoint that used to be safe now overheats for "
                        "hours. The AI notices its forecasts are too low and corrects itself, but a few short exceedances remain at "
                        "the batch start — a cue to retrain on new data.",
        "load_surge": "Nobody can forecast a job that has never happened before. The AI runs closer to the limit, so it "
                      "is hit harder in the first 15 minutes; the 'too hot now' rule cools by 2 °C immediately.",
        "normal": "In normal operation no rule except the rate limit and an occasional 'too hot now' needs to step in.",
    }
    st.info(takeaways[pick])


# ---------- ⑥ results -----------------------------------------------------------------------
with tabs[5]:
    st.subheader("Did it save energy — without overheating?", icon=":material/bolt:")
    in_short("Three controllers run the same simulated weeks: **fixed** setpoint, a tuned **rule**, and the **AI**.",
             "Green = fixed, orange = rule, blue = AI. Lower energy is better; minutes above 27 °C must stay at 0.")
    pick = st.segmented_control("Scenario ", list(SC), default="normal", format_func=lambda s: SC[s].title, key="res_sc") or "normal"
    table = B["summary"].set_index(["scenario", "key"]).loc[pick].loc[["fixed", "rule", "ai"]]
    hours = len(LOGS[pick, "ai"]) * C.DT_H
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cooling energy (kWh)", "Minutes above 27 °C"), horizontal_spacing=0.18)
        names = [NAMES[k] for k in table.index]
        fig.add_bar(y=names, x=table["cooling_kwh"], orientation="h", marker_color=[CTRL[k] for k in table.index], width=0.5,
                    text=[f"{v:,.0f}" for v in table["cooling_kwh"]], textposition="outside", cliponaxis=False, row=1, col=1)
        fig.add_bar(y=names, x=table["violation_minutes"], orientation="h", marker_color=[CTRL[k] for k in table.index], width=0.5,
                    text=[f"{v:,.0f}" for v in table["violation_minutes"]], textposition="outside", cliponaxis=False, row=1, col=2)
        fig.update_xaxes(range=[0, table["cooling_kwh"].max() * 1.25], row=1, col=1)
        fig.update_xaxes(range=[0, max(60, table["violation_minutes"].max() * 1.3)], row=1, col=2)
        fig.update_yaxes(autorange="reversed")
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        show(style(fig, height=250, legend=False, hover="closest"))
        st.caption(f"Over {hours / 24:.0f} days. Savings look small on a zero-based axis because most cooling energy is needed "
                   "whatever the setpoint; the AI only trims the part the setpoint can influence.")
    with c2:
        avg = pd.DataFrame({NAMES[k]: [table.loc[k, "it_kwh"] / hours, table.loc[k, "cooling_kwh"] / hours,
                                       (table.loc[k, "other_kwh"] + table.loc[k, "ai_overhead_kwh"]) / hours,
                                       table.loc[k, "facility_kwh"] / hours, table.loc[k, "pue"]] for k in table.index},
                           index=["IT (kW)", "Cooling (kW)", "Other incl. AI server (kW)", "Total (kW)", "PUE"])
        st.markdown("**Average power, like proposal §8.4**")
        st.dataframe(avg.style.format("{:.3f}", subset=pd.IndexSlice[["PUE"], :]).format("{:.1f}", subset=pd.IndexSlice[avg.index[:-1], :]),
                     width="stretch")
    if pick == "normal":
        f, r = HEAD["fixed"], HEAD["rule"]
        st.markdown(
            f"**Normal operation, 14 test days.** Compared with the tuned rule-based controller the AI uses "
            f"**{r['cooling_pct']:.1f} % less cooling energy** and **{r['facility_pct']:.2f} % less total energy** "
            f"(≈ {r['yearly_kwh']:,.0f} kWh ≈ {r['yearly_thb']:,.0f} THB per year). Against the fixed {B['fixed_sp']:.0f} °C setpoint: "
            f"**{f['cooling_pct']:.1f} %** cooling, **{f['facility_pct']:.2f} %** total (≈ {f['yearly_thb']:,.0f} THB per year). "
            f"Zero minutes above 27 °C for every controller.")
        st.warning("Honest reading: the proposal targeted ≥ 10 % cooling savings vs a tuned rule. In this mock room the AI gets "
                   f"{r['cooling_pct']:.1f} %. Why: (1) rack B6 is a hot spot that caps how warm the room can run; (2) above 26 °C "
                   "inlet the server fans speed up, so warmer is not always cheaper — the model knows this and stops; "
                   "(3) the setpoint is the only knob. Fixing the B6 airflow or adding knobs (chilled-water temperature, fan "
                   "speed) are the next levers.")
    d = st.selectbox("Day to compare", TEST_DAYS, index=1, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="res_day")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Setpoint (°C)", "True hottest inlet (°C)"))
    for key in ["fixed", "rule", "ai"]:
        part = LOGS[pick, key].loc[str(d)]
        fig.add_scatter(x=part.index, y=part["setpoint_in_use"], name=NAMES[key], line=dict(color=CTRL[key], width=2, shape="hv"), row=1, col=1)
        fig.add_scatter(x=part.index, y=part["max_inlet_c"], name=NAMES[key], showlegend=False, line=dict(color=CTRL[key], width=2), row=2, col=1)
    limit_line(fig, 2, 1)
    show(style(fig, height=470, title="Same day, three controllers: the AI rides closer to the limit, but not over it"))
    with st.expander("All scenarios, all numbers"):
        cols = ["scenario", "controller", "facility_kwh", "cooling_kwh", "it_kwh", "pue", "violation_minutes", "hottest_inlet_c",
                "mean_setpoint_c", "setpoint_changes", "fallback_steps", "emergency_steps", "decision_ms_mean"]
        st.dataframe(B["summary"][cols].round(2), hide_index=True, width="stretch")
        st.markdown(f"Baselines were tuned on the validation weeks: fixed = highest setpoint with no overheating "
                    f"(**{B['fixed_sp']:.0f} °C**); rule-based = lowest-energy safe deadband ({B['rule_text']}).")

# ---------- 🤖 agents -----------------------------------------------------------------------
LANES = ["G1", "supervisor", "G2", "data_analyst", "cooling_advisor", "G3", "workflow"]
LANE_NAME = {"G1": "G1 request", "G2": "G2 gateway", "G3": "G3 verify", "workflow": "Workflow", **ROLES}
STATUS_COLOR = {"passed": "#1baf7a", "completed": "#2a78d6", "authorized": "#86b6ef", "blocked": LIMIT,
                "rejected": LIMIT, "failed": LIMIT}


@st.cache_resource(show_spinner=False)
def agent_context():
    return AgentContext.from_bundle(B)


def trace_chart(result):
    ev = pd.DataFrame([e.model_dump() for e in result.events])
    ev["lane"] = ev["actor"].map(lambda a: LANE_NAME.get(a, a))
    fig = go.Figure()
    for status, color in STATUS_COLOR.items():
        part = ev[ev["status"] == status]
        if part.empty:
            continue
        fig.add_scatter(x=part["step"], y=part["lane"], mode="markers", name=status,
                        marker=dict(size=13, color=color, symbol="x" if color == LIMIT else "circle",
                                    line=dict(color=SURFACE, width=2)),
                        customdata=part[["action", "skill_id", "detail"]].fillna("").to_numpy(),
                        hovertemplate="step %{x} · %{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    fig.add_scatter(x=ev["step"], y=ev["lane"], mode="lines", line=dict(color=GRID, width=1), showlegend=False, hoverinfo="skip")
    fig.data = fig.data[-1:] + fig.data[:-1]
    fig.update_yaxes(categoryorder="array", categoryarray=[LANE_NAME[x] for x in reversed(LANES)])
    fig.update_xaxes(title="step", dtick=2)
    return style(fig, height=330, title="Who did what, in order (hover a dot)", hover="closest")


@st.fragment
def agent_case_panel():
    """Controls, run and results of one case. A fragment: Run case reruns only this part, not all 7 tabs."""
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        case_day = st.selectbox("Day", TEST_DAYS, index=1, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="ag_day")
        case_time = st.select_slider("Time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)],
                                     value="00:30", key="ag_time")
    with c2:
        mode = st.segmented_control("Task", ["recommend", "analyze"], default="recommend",
                                    format_func={"recommend": "Recommend a setpoint", "analyze": "Analyze only"}.get) or "recommend"
        fault = st.multiselect("Simulate failed sensors", C.RACKS, default=[], help="These racks stop reporting for the last 45 min")
    with c3:
        has_key = bool(os.getenv("GEMINI_API_KEY"))
        agent_mode = st.segmented_control("Agent mode", ["offline", "gemini"], default="offline",
                                          format_func={"offline": "Offline roles", "gemini": "Gemini"}.get) or "offline"
        st.caption("Offline = fixed role procedures, no LLM, instant. Gemini = real function calling, about 11 model "
                   "calls (30–60 s) "
                   + ("(API key found)." if has_key else "— set GEMINI_API_KEY to enable; without it the case fails closed."))
    question = st.text_input("Operator question", "What supply-air setpoint should we use for the next hour, and why?")
    request = CaseRequest(at=pd.Timestamp(f"{case_day} {case_time}").to_pydatetime(), question=question, mode=mode,
                          agent_mode=agent_mode, sensor_fault=tuple(fault))
    key = request.model_dump_json()
    if st.button("Run case", type="primary") or st.session_state.get("case_key") != key and "case" not in st.session_state:
        started = time.monotonic()
        steps = st.status("Agents working…" + (" Gemini usually needs 30–60 s." if agent_mode == "gemini" else ""),
                          expanded=agent_mode == "gemini")

        def progress(event):
            """Show each trace event as it happens, so a slow model call never looks like a frozen page."""
            lane = LANE_NAME.get(event.actor, event.actor)
            icon = (":material/cancel:" if event.status in ("blocked", "rejected", "failed")
                    else ":material/check_circle:")
            steps.markdown(f"{icon} {event.elapsed_ms / 1000:.0f} s · **{lane}** · {event.action}"
                           + (f" · :gray[{event.detail[:90]}]" if event.detail else ""))
            steps.update(label=f"Agents working… {time.monotonic() - started:.0f} s · {lane}: {event.action}")

        st.session_state.case = run_case(request, agent_context(), on_event=progress)
        st.session_state.case_key = key
        done = st.session_state.case
        steps.update(label=f"{done.status.replace('_', ' ').capitalize()} in {time.monotonic() - started:.1f} s"
                           f" · {done.model_calls} model calls", expanded=False,
                     state="complete" if done.status == "completed" else "error")
        error = storage.save_case(request, st.session_state.case)
        if error:
            st.toast(f"Case not saved to Supabase ({error})", icon=":material/cloud_off:")
        elif storage.enabled():
            st.toast("Case saved to history", icon=":material/cloud_done:")
    result = st.session_state.get("case")
    if result is not None:
        if st.session_state.get("case_key") != key:
            st.caption("Inputs changed — press **Run case** to update. Showing the previous case.")
        headline = {"change": f"Set {result.recommended_setpoint:g} °C" if result.recommended_setpoint is not None else "",
                    "no_change": f"Keep {result.recommended_setpoint:g} °C" if result.recommended_setpoint is not None else "",
                    "max_cooling": f"Cool to {result.recommended_setpoint:g} °C" if result.recommended_setpoint is not None else "",
                    "fallback": f"Fallback → {result.recommended_setpoint:g} °C" if result.recommended_setpoint is not None else "",
                    "analysis_only": "Analysis only", "unavailable": "No answer released"}[result.decision]
        m = st.columns(5)
        m[0].metric("Agents' answer", headline)
        m[1].metric("Status", result.status.replace("_", " "))
        m[2].metric("Setpoint in use", f"{result.setpoint_in_use:g} °C" if result.setpoint_in_use is not None else "—")
        m[3].metric("Tool calls (G2)", f"{result.tool_calls}", help="Includes the two delegations")
        m[4].metric("Model calls", f"{result.model_calls}", help="0 in offline mode")
        (st.success if result.status == "completed" else st.warning)(result.explanation)
        left, right = st.columns([1.5, 1])
        with left:
            show(trace_chart(result))
        with right:
            st.markdown("**G3 independent verification**")
            if result.checks:
                for check in result.checks:
                    st.markdown(f"{'✅' if check.passed else '❌'} {check.name}" + (f"  \n<span class='note'>{check.detail}</span>"
                                                                                   if check.detail else ""), unsafe_allow_html=True)
            else:
                st.caption("G3 did not run: " + ", ".join(result.reason_codes))
            if result.agent_notes:
                st.markdown("**Agent notes** (explanation only, not evidence)")
                for role, note in result.agent_notes.items():
                    st.markdown(f"- *{ROLES.get(role, role)}*: {note}")
        ev = result.evidence
        if "forecast_hold" in ev:
            hold = ev["forecast_hold"]
            at = pd.Timestamp(result.at)
            times = [at] + [at + pd.Timedelta(minutes=m) for m in hold["horizon_minutes"]]
            now_inlet = ev["state"]["hottest_inlet_c"]
            fig = go.Figure()
            fig.add_scatter(x=times, y=[now_inlet] + hold["with_safety_margin_c"], name=f"Hold {hold['hold_setpoint_c']:g} °C (+margin)",
                            line=dict(color=MUTED, width=2, dash="dot"), mode="lines+markers")
            table = plans_table(result)
            if table is not None:
                chosen = table[table["chosen"]].iloc[0]
                margins = np.array(B["margins"])
                fig.add_scatter(x=times, y=[now_inlet] + [chosen[f"inlet_{k}"] + margins[k - 1] for k in range(1, 5)],
                                name=f"Chosen plan {chosen['plan']} (+margin)", line=dict(color=CTRL["ai"], width=2.5),
                                mode="lines+markers")
            limit_line(fig)
            fig.update_yaxes(title="hottest inlet (°C)")
            show(style(fig, height=300, title="Data Analyst's forecast vs the Cooling Advisor's plan"))
        with st.expander("Skills, tool allowlists and full trace"):
            st.dataframe(pd.DataFrame([{"Skill": sid, "Owner": ROLES[s.owner], "Procedure": s.name, "Allowed tools": ", ".join(s.tools) or "—"}
                                       for sid, s in SKILLS.items()]), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame([e.model_dump() for e in result.events]), hide_index=True, width="stretch", height=300)
            st.json({k: v for k, v in ev.items() if not k.startswith("_")}, expanded=False)

    with st.expander("Case history (Supabase)", icon=":material/history:"):
        if not storage.enabled():
            st.caption("Off. Set SUPABASE_URL and SUPABASE_KEY in Streamlit secrets and run "
                       "supabase/migrations/001_agent_cases.sql to keep a history of every case.")
        else:
            try:
                history = pd.DataFrame(storage.recent_cases())
                if history.empty:
                    st.caption("No cases saved yet.")
                else:
                    st.dataframe(history, hide_index=True, width="stretch")
            except Exception as exc:
                st.warning(f"Could not read the history: {type(exc).__name__}: {exc}")


with agent_tab:
    st.subheader("Ask the agents about a moment in the room", icon=":material/smart_toy:")
    in_short("Pick a moment and press **Run case**. Three agents split the work like a small team.",
             "Agents never compute numbers themselves — they call checked tools, and **G3 recomputes everything** before "
             "an answer is released.")
    diagram("""
sequenceDiagram
    participant O as Operator
    participant S as Supervisor
    participant D as Data Analyst
    participant C as Cooling Advisor
    participant G as G3 verify
    O->>S: question + moment (checked by G1)
    S->>D: inspect sensors, forecast next hour
    D-->>S: data OK? hold-setpoint forecast
    S->>C: find the best safe setpoint
    C-->>S: proposal (81-plan MPC + safety)
    S->>G: re-read sensors, recompute all
    G-->>O: verified answer (or fallback)
""", alt="Sequence: operator asks Supervisor, Supervisor delegates to Data Analyst then Cooling Advisor, G3 verifies before answering")
    st.markdown("<p class='note'>The <b>Supervisor</b> plans the case and delegates. The <b>Data Analyst</b> inspects the "
                "sensors (S2), summarises the room and forecasts the next hour (S3). The <b>Cooling Advisor</b> runs the 81-plan "
                "MPC and the safety filter and proposes a setpoint (S4). <b>G2</b> checks every tool call against the skill's "
                "allowlist and budget; <b>G3</b> re-reads the sensors and recomputes everything before anything is released; the "
                "Supervisor then writes the verified answer (S5). Tools only see the 2 hours up to the chosen moment.</p>",
                unsafe_allow_html=True)
    agent_case_panel()

st.caption("Mock data from a simplified physics model (see predictive_cooling/plant.py). Results show the method, not the "
           "savings of a real facility. The tool recommends; it never controls equipment.")
