#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np


# In[7]:


# -------- CONFIG: paths you likely already have --------
FULL_TRIPS = "citibike_2022_1months.csv" 
WEATHER    = "laguardia_weather_2022_1months.csv" 
SAMPLE     = "bike_sample.csv"
TOP_ST     = "top_stations.csv"
DAILY      = "daily_trips.csv"

# -------- 1) Sample trips to keep under 25 MB (seed=32) --------
usecols = [
    "ride_id","started_at","ended_at",
    "start_station_name","end_station_name",
    "start_lat","start_lng","end_lat","end_lng",
    "member_casual"
]
df = pd.read_csv(FULL_TRIPS, usecols=usecols, parse_dates=["started_at","ended_at"], low_memory=False)

# start with 20% and shrink if needed
frac = 0.20
while True:
    sample = df.sample(frac=frac, random_state=32)
    sample.to_csv(SAMPLE, index=False)
    mb = os.path.getsize(SAMPLE) / (1024**2)
    print(f"Trial frac={frac:.2f} -> {mb:.2f} MB")
    if mb <= 25 or frac <= 0.05:
        break
    frac *= 0.8  # shrink if still too large

print(f" Saved {SAMPLE} (~{mb:.2f} MB)")

# -------- 2) Top stations for bar chart --------
top_stations = (
    sample.groupby("start_station_name", as_index=False)
          .size()
          .rename(columns={"size":"trip_count"})
          .sort_values("trip_count", ascending=False)
)
top_stations.to_csv(TOP_ST, index=False)
print(f"Saved {TOP_ST}")

# -------- 3) Merge daily trips with weather for dual-axis --------
daily_trips = (
    sample.assign(date=sample["started_at"].dt.date)
          .groupby("date", as_index=False)["ride_id"].count()
          .rename(columns={"ride_id":"trip_count"})
)
daily_trips["date"] = pd.to_datetime(daily_trips["date"])

# read LaGuardia (NOAA daily long) and pivot
w = pd.read_csv(WEATHER)
w.columns = [c.strip().lower() for c in w.columns]
w["date"] = pd.to_datetime(w["date"].astype(str).str[:10], errors="coerce")

weather_daily = (w.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
                   .reset_index())
# convert tenths to actual units
for col in ["TMAX","TMIN","PRCP"]:
    if col in weather_daily.columns:
        weather_daily[col] = weather_daily[col] / 10.0

if {"TMAX","TMIN"} <= set(weather_daily.columns):
    weather_daily["TAVG_C"] = (weather_daily["TMAX"] + weather_daily["TMIN"])/2
elif "TMAX" in weather_daily.columns:
    weather_daily["TAVG_C"] = weather_daily["TMAX"]
else:
    weather_daily["TAVG_C"] = np.nan
weather_daily["TAVG_F"] = weather_daily["TAVG_C"] * 9/5 + 32

daily = daily_trips.merge(weather_daily, on="date", how="left")
daily.to_csv(DAILY, index=False)
print(f" Saved {DAILY} (with temperature)")


# In[9]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="NYC Citi Bike Dashboard – Part 2", layout="wide")

# --------------------
# Sidebar navigation
# --------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to page:",
    [
        "Intro",
        "Trips vs Temperature",
        "Top Stations",
        "Kepler Map",
        "Extra: Hour x Weekday",
        "Recommendations"
    ]
)

# --------------------
# Data loader (cached)
# --------------------
@st.cache_data
def load_data():
    trips_path = "bike_sample.csv"
    tops_path  = "top_stations.csv"
    daily_path = "/Users/evancarr/Desktop/daily_trips.csv"

    if not all(os.path.exists(p) for p in [trips_path, tops_path, daily_path]):
        st.error("Missing one of: bike_sample.csv, top_stations.csv, daily_trips.csv. Run the notebook step first.")
        st.stop()

    trips = pd.read_csv(trips_path, parse_dates=["started_at","ended_at"])
    top_stations = pd.read_csv(tops_path)
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    return trips, top_stations, daily

trips, top_stations, daily = load_data()

# --------------------
# Optional global filters
# --------------------
with st.sidebar.expander("Filters", expanded=False):
    # date range for daily chart
    dmin, dmax = daily["date"].min().date(), daily["date"].max().date()
    dsel = st.date_input("Date range (daily)", (dmin, dmax), min_value=dmin, max_value=dmax)

    # user type filter for extra chart
    user_types = sorted(trips["member_casual"].dropna().unique().tolist())
    selected_types = st.multiselect("User type", user_types, default=user_types)

daily_f = daily[(daily["date"] >= pd.to_datetime(dsel[0])) & (daily["date"] <= pd.to_datetime(dsel[1]))]

# --------------------
# PAGES
# --------------------
if page == "Intro":
    st.title("NYC Citi Bike – Interactive Dashboard (Part 2)")
    st.markdown("""
**Purpose.** Explore station popularity, seasonality vs temperature, and origin–destination flows to support **supply & rebalancing decisions**.

**What’s inside:**
- **Trips vs Temperature** *(dual-axis)* — demand seasonality vs weather.
- **Top Stations** *(bar)* — busiest origins.
- **Kepler Map** — station‑to‑station flows (arcs).
- **Extra** — hourly/weekday usage heatmap.
- **Recommendations** — actions to improve bike/dock availability.

_Data note: using a random sample (seed=32) under 25 MB for easy sharing & deployment._
""")

elif page == "Trips vs Temperature":
    st.header("Daily Trips vs Temperature (LaGuardia)")
    # choose available temp column
    temp_col = None
    for c in ["TAVG_F", "TAVG_C", "TMAX"]:
        if c in daily_f.columns:
            temp_col = c
            break

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # temperature on left
    fig.add_trace(go.Scatter(x=daily_f["date"], y=daily_f[temp_col],
                             name=f"Temperature ({temp_col})", mode="lines"),
                  secondary_y=False)
    # trips on right
    fig.add_trace(go.Scatter(x=daily_f["date"], y=daily_f["trip_count"],
                             name="Trips", mode="lines"),
                  secondary_y=True)

    fig.update_layout(
        title="Daily Trips vs Temperature",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_yaxes(title_text=f"Temperature ({temp_col})", secondary_y=False)
    fig.update_yaxes(title_text="Trips", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Interpretation.** Trips increase with warmer temps and dip in colder periods.  
Watch shoulder seasons and spikes on unseasonably warm days to tune **rebalancing and staffing**.
""")

elif page == "Top Stations":
    st.header("Top 20 Start Stations")
    fig = px.bar(
        top_stations.head(20),
        x="start_station_name",
        y="trip_count",
        color="trip_count",
        color_continuous_scale="Blues",
        title="Most Popular Start Stations"
    )
    fig.update_layout(xaxis_tickangle=45, xaxis_title="", yaxis_title="Trips")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Interpretation.** These stations consistently show **high demand**.  
Consider **higher dock counts**, **AM placement** and **PM collection**, and **incentives** to rebalance user flows.
""")

elif page == "Kepler Map":
    st.header("Station Flows (Kepler.gl)")
    html_file = "nyc_trip_map.html"
    if os.path.exists(html_file):
        st.components.v1.html(open(html_file, "r", encoding="utf-8").read(), height=620)
        st.caption("Tip: Use the filter panel inside the map to focus on high trip_count corridors.")
    else:
        st.warning(f"Add {html_file} to this folder to display the Kepler map.")
    st.markdown("""
**Interpretation.** Thick arcs mark **major corridors** (Midtown and bridge approaches).  
These are candidates for **capacity boosts** and **tighter truck loops**.
""")

elif page == "Extra: Hour x Weekday":
    st.header("Usage by Hour & Weekday (Filtered by user type)")
    df = trips[trips["member_casual"].isin(selected_types)].copy()
    df["hour"] = df["started_at"].dt.hour
    df["weekday"] = df["started_at"].dt.day_name()
    # order weekdays
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df["weekday"] = pd.Categorical(df["weekday"], categories=order, ordered=True)

    heat = (df.groupby(["weekday","hour"], as_index=False)["ride_id"].count()
              .rename(columns={"ride_id":"trips"}))

    fig = px.density_heatmap(heat, x="hour", y="weekday", z="trips",
                             color_continuous_scale="Viridis",
                             title="Trips by Hour & Weekday")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**Insight.** **AM/PM commute peaks** on weekdays; broader midday peaks on weekends.  
Schedule rebalancing and staffing to match these patterns.
""")

elif page == "Recommendations":
    st.header("Recommendations")
    st.markdown("""
- **Seasonal scaling:** Reduce active fleet **~20–30% Nov–Apr**, keep service robust at commute hubs.
- **Waterfront capacity:** Add/expand docks near parks & promenades; pilot **pop‑up docks** on summer weekends.
- **Rebalancing:** Guarantee **dock availability** at top stations via **AM placement** and **PM returns**; plan **hourly truck loops** for peaks.
- **Predictive ops:** Adjust for **weather** (heat/cold/rain) using recent demand + forecast signals.
- **Incentives:** Offer **credits** for ending trips at underutilized nearby docks to self‑rebalance.
""")


# In[30]:


# run this in the folder that contains your notebook
get_ipython().system('jupyter nbconvert --to script "2_7_Presenting_Dashboard.ipynb" --output app2')


# In[ ]:




