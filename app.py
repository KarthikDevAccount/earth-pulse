import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Earth Pulse | Global Climate Intelligence", page_icon="🌍", layout="wide")

NASA_GISTEMP_CSV = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
NASA_GISTEMP_PAGE = "https://data.giss.nasa.gov/gistemp/"
NASA_POWER_DOCS = "https://power.larc.nasa.gov/docs/services/api/temporal/monthly/"
NASA_SEALEVEL = "https://sealevel.nasa.gov/"
NASA_EARTHDATA = "https://www.earthdata.nasa.gov/"

st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(120,170,190,.25); border-radius: 18px; background: linear-gradient(115deg, rgba(13,45,67,.95), rgba(12,25,44,.96)); margin-bottom: 1rem;}
.kicker {letter-spacing: .16em; text-transform: uppercase; color: #78c7d4; font-size: .72rem; font-weight: 700;}
.hero h1 {font-size: clamp(2rem,4vw,3.1rem); margin: .3rem 0 .25rem; color: #f4fbff;}
.hero p {color: #bfd2df; margin: 0;}
.source-note {font-size: .82rem; color: #9fb4c3;}
[data-testid="stMetric"] {background: rgba(34,60,78,.35); border: 1px solid rgba(130,170,190,.18); padding: 14px 16px; border-radius: 14px;}
</style>
<div class="hero">
  <div class="kicker">NASA data · Global climate · 24-month outlook</div>
  <h1>🌍 Earth Pulse</h1>
  <p>Global climate monitoring and transparent, experimental forecasting — observations are separated from model estimates.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_gistemp():
    response = requests.get(NASA_GISTEMP_CSV, timeout=40, headers={"User-Agent": "EarthPulseDashboard/1.0"})
    response.raise_for_status()
    raw = response.text
    # NASA's table contains a short metadata preamble; locate the actual CSV header.
    lines = raw.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.strip().startswith("Year,Jan,Feb")), None)
    if header_idx is None:
        raise ValueError("Could not identify the expected NASA GISTEMP CSV header.")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [str(c).strip() for c in df.columns]
    month_cols = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] if m in df.columns]
    rows = []
    for _, row in df.iterrows():
        try:
            year = int(row["Year"])
        except (ValueError, TypeError):
            continue
        for month_num, month in enumerate(month_cols, start=1):
            val = pd.to_numeric(row[month], errors="coerce")
            if pd.notna(val) and abs(float(val)) < 99:
                rows.append({"date": pd.Timestamp(year=year, month=month_num, day=1), "anomaly": float(val)})
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    if out.empty:
        raise ValueError("NASA GISTEMP returned no usable monthly observations.")
    return out.reset_index(drop=True), response.headers.get("Last-Modified", "Not supplied by server")

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_power_monthly(lat, lon, start_year, end_year):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {"parameters":"T2M,PRECTOTCORR,RH2M", "community":"AG", "longitude":lon,
              "latitude":lat, "format":"JSON", "start":start_year, "end":end_year}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    p = payload.get("properties", {}).get("parameter", {})
    frames = []
    for key, label, unit in [("T2M","Air temperature","°C"),("PRECTOTCORR","Precipitation","mm/day"),("RH2M","Relative humidity","%")]:
        if key not in p:
            continue
        s = pd.Series(p[key], dtype="float64")
        s = s[s < 900]
        f = s.rename("value").reset_index()
        f.columns = ["yyyymm","value"]
        f["date"] = pd.to_datetime(f["yyyymm"].astype(str), format="%Y%m")
        f["parameter"] = label
        f["unit"] = unit
        frames.append(f[["date","value","parameter","unit"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_features(dates, origin=None):
    # Use a shared calendar origin so validation and future timestamps remain
    # correctly positioned beyond the training period.
    if origin is None:
        origin = min(dates)
    t = np.array([(d.year-origin.year)*12 + (d.month-origin.month) for d in dates], dtype=float)
    month = np.array([d.month for d in dates], dtype=float)
    return np.column_stack([t, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                            np.sin(4*np.pi*month/12), np.cos(4*np.pi*month/12)])

@st.cache_data(show_spinner=False)
def forecast_temperature(df, horizon=24):
    series = df.sort_values("date").reset_index(drop=True)
    # Hold out the most recent 24 months for honest chronological evaluation.
    if len(series) < 120:
        raise ValueError("At least 10 years of monthly observations are needed for this forecast.")
    test_n = min(24, max(12, len(series)//10))
    train, test = series.iloc[:-test_n], series.iloc[-test_n:]
    origin = series["date"].iloc[0]
    x_train = make_features(train["date"].tolist(), origin)
    x_test = make_features(test["date"].tolist(), origin)
    candidates = {
        "Ridge trend + seasonality": Ridge(alpha=2.0),
        "Random forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=2.0, random_state=42)
    }
    scores, fitted = [], {}
    for name, model in candidates.items():
        model.fit(x_train, train["anomaly"].values)
        pred = model.predict(x_test)
        scores.append({"Model":name, "MAE (°C)":mean_absolute_error(test["anomaly"],pred),
                       "RMSE (°C)":mean_squared_error(test["anomaly"],pred)**0.5,
                       "R²":r2_score(test["anomaly"],pred)})
        fitted[name] = model
    score_df = pd.DataFrame(scores).sort_values("MAE (°C)")
    best_name = score_df.iloc[0]["Model"]
    # Refit selected model on all available observations.
    best_model = candidates[best_name]
    best_model.fit(make_features(series["date"].tolist(), origin), series["anomaly"].values)
    future_dates = pd.date_range(series["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    point = best_model.predict(make_features(future_dates.tolist(), origin))
    # Empirical residual interval from chronological holdout. Not a formal probabilistic interval.
    test_pred = fitted[best_name].predict(x_test)
    residual = test["anomaly"].values - test_pred
    spread = float(np.nanstd(residual, ddof=1)) if len(residual) > 1 else 0.1
    # Widen interval with horizon to reflect increasing forecast uncertainty.
    widen = spread * np.sqrt(1 + np.arange(1, horizon+1)/12)
    forecast = pd.DataFrame({"date":future_dates, "prediction":point, "lower":point-1.96*widen, "upper":point+1.96*widen})
    history = series.copy()
    return forecast, score_df, best_name, history, test

with st.sidebar:
    st.header("Dashboard controls")
    st.caption("Global coverage · NASA GISTEMP baseline")
    horizon = st.select_slider("Forecast horizon", options=[6,12,18,24], value=24, format_func=lambda x:f"{x} months")
    st.divider()
    st.markdown("**Optional NASA POWER explorer**")
    lat = st.slider("Latitude", -60.0, 70.0, 0.0, 1.0)
    lon = st.slider("Longitude", -180.0, 180.0, 0.0, 1.0)
    st.caption("POWER values are for the selected coordinate, not a global average.")
    if st.button("Refresh NASA observations", use_container_width=True):
        load_gistemp.clear(); load_power_monthly.clear(); forecast_temperature.clear()
        st.rerun()

try:
    with st.spinner("Retrieving NASA GISTEMP observations…"):
        climate, last_modified = load_gistemp()
    forecast, scores, best_model, history, holdout = forecast_temperature(climate, horizon)
except Exception as exc:
    st.error(f"NASA data could not be loaded or processed: {exc}")
    st.markdown(f"You can still access the source directly: [{NASA_GISTEMP_PAGE}]({NASA_GISTEMP_PAGE})")
    st.stop()

latest = history.iloc[-1]
latest_date = latest["date"]
latest_val = latest["anomaly"]
prev_year = history[history["date"] == latest_date - pd.DateOffset(years=1)]
yoy = latest_val - float(prev_year.iloc[0]["anomaly"]) if not prev_year.empty else np.nan
recent_12 = history.tail(12)["anomaly"].mean()
prior_12 = history.iloc[-24:-12]["anomaly"].mean() if len(history) >= 24 else np.nan
trend = recent_12 - prior_12 if pd.notna(prior_12) else np.nan

st.caption(f"Latest available NASA monthly observation: **{latest_date:%B %Y}** · Source file last-modified: {last_modified} · Data retrieval: {datetime.now().astimezone():%d %b %Y, %H:%M %Z}")
if latest_date < pd.Timestamp(datetime.now().year, datetime.now().month, 1) - pd.offsets.MonthBegin(1):
    st.info("The newest NASA monthly observation is not the current month. The dashboard uses the latest published observation and displays its actual date.")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Latest global temperature anomaly", f"{latest_val:+.2f} °C", f"{yoy:+.2f} °C vs same month last year" if pd.notna(yoy) else None)
m2.metric("Latest 12-month mean anomaly", f"{recent_12:+.2f} °C")
m3.metric("Change vs previous 12-month mean", f"{trend:+.2f} °C" if pd.notna(trend) else "—")
m4.metric("Forecast horizon", f"{horizon} months", f"Selected model: {best_model}")

st.subheader("Global surface temperature anomaly")
st.caption("NASA GISTEMP LOTI monthly anomaly, °C relative to the 1951–1980 baseline. This is an anomaly, not the absolute global temperature.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history["anomaly"], name="NASA observation", mode="lines", line=dict(color="#64c9d8", width=1.8)))
fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper"], name="Forecast upper interval", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower"], name="Approx. 95% residual band", mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(245,166,78,.18)"))
fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["prediction"], name="Model forecast", mode="lines", line=dict(color="#f5a64e", width=2.5, dash="dash")))
fig.add_vline(x=latest_date, line_dash="dot", line_color="rgba(220,230,240,.6)")
fig.update_layout(height=470, template="plotly_dark", margin=dict(l=10,r=10,t=25,b=10), hovermode="x unified", legend=dict(orientation="h", y=1.12), yaxis_title="Temperature anomaly (°C)", xaxis_title="Month")
st.plotly_chart(fig, use_container_width=True)

left,right = st.columns([1.1, .9])
with left:
    st.subheader("Forecast summary")
    end = forecast.iloc[-1]
    delta = float(end["prediction"] - latest_val)
    st.metric(f"Model estimate · {end['date']:%B %Y}", f"{end['prediction']:+.2f} °C anomaly", f"{delta:+.2f} °C vs latest observation")
    st.caption("The endpoint estimate is a statistical extrapolation of the temperature-anomaly series, not an official NASA forecast. The shaded band is an approximate residual-based range and does not include all climate-model uncertainties.")
    st.dataframe(forecast.assign(Month=forecast["date"].dt.strftime("%b %Y"))[ ["Month","prediction","lower","upper"] ].rename(columns={"prediction":"Estimate °C","lower":"Lower °C","upper":"Upper °C"}).round(3), use_container_width=True, hide_index=True)
with right:
    st.subheader("Model validation")
    st.caption("Chronological holdout: the latest 12–24 months are excluded from training.")
    st.dataframe(scores.round(3), use_container_width=True, hide_index=True)
    st.markdown(f"**Selected model:** {best_model}")
    st.markdown("Model selection is based on lowest holdout MAE. Performance can change as new observations arrive.")

st.subheader("NASA POWER: local climate explorer")
st.caption("Explore NASA POWER monthly meteorological estimates for a selected point. This section is intentionally labelled point-level and must not be interpreted as global climate statistics.")
try:
    pwr = load_power_monthly(lat, lon, 2001, min(datetime.now().year-1, 2025))
    if not pwr.empty:
        selected = st.selectbox("POWER variable", sorted(pwr["parameter"].unique()))
        subset = pwr[pwr["parameter"] == selected]
        pf = go.Figure(go.Scatter(x=subset["date"], y=subset["value"], mode="lines", line=dict(color="#8ed081", width=1.7), name=selected))
        pf.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=20,b=10), yaxis_title=subset["unit"].iloc[0], xaxis_title="Month")
        st.plotly_chart(pf, use_container_width=True)
    else:
        st.warning("NASA POWER did not return values for this coordinate and period.")
except Exception as exc:
    st.warning(f"NASA POWER explorer is temporarily unavailable: {exc}")

st.subheader("What the current data can and cannot tell us")
col1,col2 = st.columns(2)
with col1:
    st.markdown("**Supported in this release**")
    st.markdown("- Global monthly land–ocean temperature anomaly from NASA GISTEMP.\n- A reproducible 24-month statistical baseline forecast.\n- Holdout accuracy metrics and approximate residual range.\n- NASA POWER point-level temperature, precipitation and humidity explorer.")
with col2:
    st.markdown("**Not claimed in this release**")
    st.markdown("- No claim of predicting individual extreme weather events.\n- No global CO₂, sea-level or sea-ice forecast until their NASA data pipelines are integrated and validated.\n- No claim that the statistical forecast replaces physical climate-model ensembles.\n- No guarantee that model uncertainty bands capture every source of uncertainty.")

st.divider()
st.markdown("**Data provenance**")
st.markdown(f"- [NASA GISS Surface Temperature Analysis (GISTEMP v4)]({NASA_GISTEMP_PAGE}) — global monthly land–ocean temperature index.\n- [NASA POWER Monthly API documentation]({NASA_POWER_DOCS}) — meteorological time series at a selected point.\n- [NASA Sea Level Change Portal]({NASA_SEALEVEL}) — sea-level science and data tools.\n- [NASA Earthdata]({NASA_EARTHDATA}) — access to NASA Earth observation datasets.")
st.caption("Earth Pulse is an independent educational analysis dashboard. NASA does not operate, endorse, or certify this application or its model forecasts. Always cite the underlying NASA dataset and access date in research or publications.")
