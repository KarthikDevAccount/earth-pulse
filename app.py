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
    params = {"parameters":"T2M,PRECTOTCORR,RH2M,GWETROOT", "community":"AG", "longitude":lon,
              "latitude":lat, "format":"JSON", "start":start_year, "end":end_year}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    p = payload.get("properties", {}).get("parameter", {})
    frames = []
    for key, label, unit in [("T2M","Air temperature","°C"),("PRECTOTCORR","Precipitation","mm/day"),("RH2M","Relative humidity","%"),("GWETROOT","Root-zone soil wetness","fraction")]:
        if key not in p:
            continue
        s = pd.Series(p[key], dtype="float64")
        s = s[s < 900]
        f = s.rename("value").reset_index()
        f.columns = ["yyyymm","value"]
        # POWER JSON keys are usually YYYYMM but can be decoded as YYYYMM.0
        # by pandas/numpy in some versions. Keep only the first six digits.
        date_keys = f["yyyymm"].astype(str).str.extract(r"(\d{6})", expand=False)
        f["date"] = pd.to_datetime(date_keys, format="%Y%m", errors="coerce")
        f = f.dropna(subset=["date"])
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
    pwr = load_power_monthly(lat, lon, 2001, datetime.now().year-1)
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

# -----------------------------------------------------------------------------
# Earth Pulse 2.0 — additional climate indicators
# These integrations are deliberately source-labelled. NOAA/NSIDC products are
# not described as NASA products, even though they complement NASA observations.
# -----------------------------------------------------------------------------

def _numeric_forecast(frame, date_col="date", value_col="value", horizon=24, min_points=36):
    """Chronological holdout model comparison for a regularly sampled series."""
    data = frame[[date_col, value_col]].dropna().sort_values(date_col).drop_duplicates(date_col).reset_index(drop=True)
    if len(data) < min_points:
        raise ValueError(f"Need at least {min_points} valid observations; found {len(data)}.")
    dates = pd.to_datetime(data[date_col])
    vals = data[value_col].astype(float).to_numpy()
    # Infer monthly cadence where possible; otherwise forecast yearly observations.
    monthly = len(dates) > 1 and (dates.diff().dropna().dt.days.median() < 45)
    step = 1 if monthly else 12
    test_n = min(max(6, min(24, len(data)//5)), max(1, len(data)//3))
    if len(data) - test_n < 24:
        test_n = max(3, len(data)//5)
    origin = dates.iloc[0]

    def features(ds):
        ds = pd.to_datetime(ds)
        t = np.array([(d.year-origin.year)*12 + d.month-origin.month for d in ds], dtype=float)
        m = np.array([d.month for d in ds], dtype=float)
        return np.column_stack([t, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                                np.sin(4*np.pi*m/12), np.cos(4*np.pi*m/12)])

    split = len(data)-test_n
    xtr, xte = features(dates.iloc[:split]), features(dates.iloc[split:])
    candidates = {
        "Ridge trend + seasonality": Ridge(alpha=2.0),
        "Random forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, l2_regularization=1.5, random_state=42),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(xtr, vals[:split])
        pred = model.predict(xte)
        rows.append({"Model":name, "MAE":mean_absolute_error(vals[split:], pred),
                     "RMSE":mean_squared_error(vals[split:], pred)**.5,
                     "R²":r2_score(vals[split:], pred) if len(vals[split:]) > 1 else np.nan})
        fitted[name] = model
    scores = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    winner = str(scores.iloc[0]["Model"])
    model = candidates[winner]
    model.fit(features(dates), vals)
    if monthly:
        future_dates = pd.date_range(dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    else:
        future_dates = pd.date_range(dates.iloc[-1] + pd.DateOffset(years=1), periods=max(1, int(np.ceil(horizon/12))), freq="YS")
    predicted = model.predict(features(future_dates))
    holdout_resid = vals[split:] - fitted[winner].predict(xte)
    sigma = float(np.nanstd(holdout_resid, ddof=1)) if len(holdout_resid) > 1 else float(np.nanstd(vals))
    sigma = max(sigma, 1e-8)
    lead = np.arange(1, len(future_dates)+1, dtype=float)
    widen = sigma*np.sqrt(1 + lead/(12 if monthly else 1))
    future = pd.DataFrame({"date":future_dates, "prediction":predicted,
                           "lower":predicted-1.96*widen, "upper":predicted+1.96*widen})
    return future, scores, winner, data


@st.cache_data(ttl=12*60*60, show_spinner=False)
def load_noaa_global_co2():
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv"
    r = requests.get(url, timeout=35, headers={"User-Agent":"EarthPulseDashboard/2.0"})
    r.raise_for_status()
    lines = [line for line in r.text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # NOAA global monthly file convention: year, month, decimal date, average, interpolated, trend, days.
    year_col = next(c for c in df.columns if c in ("year",))
    month_col = next(c for c in df.columns if c in ("month",))
    val_col = next(c for c in df.columns if c in ("average", "interpolated", "trend"))
    out = df[[year_col, month_col, val_col]].copy()
    out.columns = ["year", "month", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["value"] > 0]
    out["date"] = pd.to_datetime(dict(year=out.year.astype(int), month=out.month.astype(int), day=1))
    return out[["date", "value"]].sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_nsidc_monthly_ice():
    base = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/"
    parts = []
    for month in range(1, 13):
        url = f"{base}N_{month:02d}_extent_v4.0.csv"
        r = requests.get(url, timeout=30, headers={"User-Agent":"EarthPulseDashboard/2.0"})
        r.raise_for_status()
        # Files contain a short comment/header preamble; detect the data header.
        lines = r.text.splitlines()
        start = next((i for i, line in enumerate(lines) if "year" in line.lower() and "," in line), 0)
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), comment="#", skipinitialspace=True)
        df.columns = [str(c).strip().lower() for c in df.columns]
        year_col = next((c for c in df.columns if c == "year" or "year" in c), None)
        extent_col = next((c for c in df.columns if "extent" in c), None)
        if year_col is None or extent_col is None:
            # Some archive versions have no header row; use their documented first columns.
            df = pd.read_csv(io.StringIO(r.text), comment="#", header=None)
            if df.shape[1] < 2:
                continue
            df = df.iloc[:, :2]
            df.columns = ["year", "extent"]
            year_col, extent_col = "year", "extent"
        part = pd.DataFrame({"year":pd.to_numeric(df[year_col], errors="coerce"),
                             "value":pd.to_numeric(df[extent_col], errors="coerce")})
        part = part.dropna()
        part["date"] = pd.to_datetime(part["year"].astype(int).astype(str) + f"-{month:02d}-01")
        parts.append(part[["date", "value"]])
    if not parts:
        raise ValueError("NSIDC monthly extent files did not contain parseable data.")
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_global_sea_level():
    # NOAA STAR's altimetry table links to this NASA/JPL-derived global series.
    url = "https://www.star.nesdis.noaa.gov/sod/lsa/SeaLevelRise/slr/slr_sla_gbl_keep_ref_90.csv"
    r = requests.get(url, timeout=35, headers={"User-Agent":"EarthPulseDashboard/2.0"})
    r.raise_for_status()
    text = r.text
    try:
        df = pd.read_csv(io.StringIO(text), comment="#", sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(text), comment="#", delim_whitespace=True)
    # The file has changed formats over time; identify decimal-year/date and GMSL columns.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("Sea-level source format was not recognised.")
    cols = list(numeric.columns)
    year_col = next((c for c in cols if "year" in str(c).lower() or "time" in str(c).lower()), cols[0])
    value_col = next((c for c in cols if any(k in str(c).lower() for k in ("gmsl", "sea", "height", "sla")) and c != year_col), None)
    if value_col is None:
        value_col = cols[1] if len(cols) > 1 else None
    if value_col is None:
        raise ValueError("Sea-level values column was not found in source file.")
    x = numeric[[year_col, value_col]].dropna()
    x.columns = ["decimal_year", "value"]
    # Convert decimal year to a monthly timestamp; source values are millimetres in this product.
    x["date"] = pd.to_datetime(x.decimal_year.astype(float).map(lambda v: f"{int(v)}-{max(1,min(12,int((v-int(v))*12)+1)):02d}-01"))
    x = x.groupby("date", as_index=False)["value"].mean().sort_values("date")
    if len(x) < 24:
        raise ValueError("Sea-level source returned too few usable records.")
    return x[["date", "value"]]


@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_power_global_region(parameter, start_year, end_year, lat=0.0, lon=0.0):
    """Fetch a NASA POWER monthly point series.

    The POWER regional endpoint does not support a whole-world bounding box.
    This intentionally uses the selected dashboard coordinate and is labelled
    point-level in the UI rather than misrepresenting it as a global mean.
    """
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {"parameters":parameter, "community":"AG", "latitude":lat,
              "longitude":lon, "format":"JSON", "start":start_year, "end":end_year}
    r = requests.get(url, params=params, timeout=60, headers={"User-Agent":"EarthPulseDashboard/2.1"})
    r.raise_for_status()
    payload = r.json()
    series = payload.get("properties", {}).get("parameter", {}).get(parameter, {})
    if not series:
        raise ValueError(f"NASA POWER returned no values for {parameter} at ({lat}, {lon}).")
    out = pd.Series(series, dtype="float64").rename("value").reset_index()
    out.columns = ["yyyymm", "value"]
    date_keys = out["yyyymm"].astype(str).str.extract(r"(\d{6})", expand=False)
    out["date"] = pd.to_datetime(date_keys, format="%Y%m", errors="coerce")
    out = out.dropna(subset=["date"])
    out = out[out.value < 900]
    return out[["date", "value"]].sort_values("date").reset_index(drop=True)


st.divider()
st.header("Global climate indicators · Earth Pulse 2.0")
st.caption("Additional indicators are independently sourced and carry their own update schedules and limitations. A forecast is a statistical estimate, not an official agency projection.")
co2_tab, sea_tab, ice_tab, water_tab = st.tabs(["Atmospheric CO₂", "Global sea level", "Arctic sea ice", "Precipitation & soil moisture"])

with co2_tab:
    st.subheader("Atmospheric carbon dioxide")
    st.caption("NOAA Global Monitoring Laboratory global marine surface monthly mean CO₂. This is a complementary non-NASA observing dataset; it is not an OCO-2 satellite retrieval.")
    try:
        with st.spinner("Loading global CO₂ observations…"):
            co2 = load_noaa_global_co2()
        co2_forecast, co2_scores, co2_model, co2_hist = _numeric_forecast(co2, horizon=horizon, min_points=120)
        latest_co2 = co2_hist.iloc[-1]
        c1,c2,c3 = st.columns(3)
        c1.metric("Latest global monthly CO₂", f"{latest_co2.value:.1f} ppm", latest_co2.date.strftime("%B %Y"))
        c2.metric("Selected forecast model", co2_model)
        c3.metric("Forecast endpoint", f"{co2_forecast.iloc[-1].prediction:.1f} ppm", co2_forecast.iloc[-1].date.strftime("%B %Y"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=co2_hist.date,y=co2_hist.value,name="Observed CO₂",line=dict(color="#64c9d8",width=2)))
        f.add_trace(go.Scatter(x=co2_forecast.date,y=co2_forecast.upper,name="Upper residual range",line=dict(width=0),showlegend=False))
        f.add_trace(go.Scatter(x=co2_forecast.date,y=co2_forecast.lower,name="Approx. residual range",line=dict(width=0),fill="tonexty",fillcolor="rgba(245,166,78,.18)"))
        f.add_trace(go.Scatter(x=co2_forecast.date,y=co2_forecast.prediction,name="Experimental forecast",line=dict(color="#f5a64e",dash="dash",width=2.5)))
        f.update_layout(template="plotly_dark",height=390,hovermode="x unified",yaxis_title="CO₂ (ppm)",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(f,use_container_width=True)
        st.markdown("**Chronological holdout model comparison**")
        st.dataframe(co2_scores.round(3),hide_index=True,use_container_width=True)
        st.caption("CO₂ has a strong long-term trend and seasonal cycle. The model is a short-horizon statistical extrapolation and does not account for future emissions scenarios.")
    except Exception as exc:
        st.warning(f"CO₂ feed is temporarily unavailable or its format changed: {exc}")
        st.link_button("Open NOAA global CO₂ source", "https://gml.noaa.gov/ccgg/trends/gl_data.html")

with sea_tab:
    st.subheader("Global mean sea level")
    st.caption("Satellite-altimetry series distributed by NOAA STAR's Laboratory for Satellite Altimetry; the linked global series is NASA/JPL-derived. Coverage and processing notes apply.")
    try:
        with st.spinner("Loading satellite sea-level observations…"):
            sea = load_global_sea_level()
        sea_forecast, sea_scores, sea_model, sea_hist = _numeric_forecast(sea, horizon=horizon, min_points=24)
        s1,s2,s3 = st.columns(3)
        s1.metric("Latest source value", f"{sea_hist.iloc[-1].value:.1f}", sea_hist.iloc[-1].date.strftime("%B %Y"))
        s2.metric("Selected forecast model", sea_model)
        s3.metric("Forecast endpoint", f"{sea_forecast.iloc[-1].prediction:.1f}", sea_forecast.iloc[-1].date.strftime("%B %Y"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=sea_hist.date,y=sea_hist.value,name="Altimetry observation",line=dict(color="#64c9d8",width=2)))
        f.add_trace(go.Scatter(x=sea_forecast.date,y=sea_forecast.upper,name="Upper residual range",line=dict(width=0),showlegend=False))
        f.add_trace(go.Scatter(x=sea_forecast.date,y=sea_forecast.lower,name="Approx. residual range",line=dict(width=0),fill="tonexty",fillcolor="rgba(245,166,78,.18)"))
        f.add_trace(go.Scatter(x=sea_forecast.date,y=sea_forecast.prediction,name="Experimental forecast",line=dict(color="#f5a64e",dash="dash",width=2.5)))
        f.update_layout(template="plotly_dark",height=390,hovermode="x unified",yaxis_title="Source units (verify source metadata)",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(f,use_container_width=True)
        st.dataframe(sea_scores.round(3),hide_index=True,use_container_width=True)
        st.caption("The source product's baseline and units must be retained when interpreting absolute values. The dashboard forecasts the supplied series; it does not estimate local coastal flood risk.")
    except Exception as exc:
        st.warning(f"Sea-level feed is temporarily unavailable or its format changed: {exc}")
        st.link_button("Open global sea-level source and downloads", "https://www.star.nesdis.noaa.gov/sod/lsa/SeaLevelRise/LSA_SLR_timeseries_global.php")

with ice_tab:
    st.subheader("Arctic sea-ice extent")
    st.caption("NOAA at NSIDC Sea Ice Index Version 4 monthly extent. NSIDC is a US data centre; this is not labelled as a NASA-only observation.")
    try:
        with st.spinner("Loading NSIDC monthly sea-ice data…"):
            ice = load_nsidc_monthly_ice()
        ice_forecast, ice_scores, ice_model, ice_hist = _numeric_forecast(ice, horizon=horizon, min_points=120)
        i1,i2,i3 = st.columns(3)
        i1.metric("Latest monthly extent", f"{ice_hist.iloc[-1].value:.2f} million km²", ice_hist.iloc[-1].date.strftime("%B %Y"))
        i2.metric("Selected forecast model", ice_model)
        i3.metric("Forecast endpoint", f"{ice_forecast.iloc[-1].prediction:.2f} million km²", ice_forecast.iloc[-1].date.strftime("%B %Y"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=ice_hist.date,y=ice_hist.value,name="NSIDC observation",line=dict(color="#64c9d8",width=1.8)))
        f.add_trace(go.Scatter(x=ice_forecast.date,y=ice_forecast.upper,name="Upper residual range",line=dict(width=0),showlegend=False))
        f.add_trace(go.Scatter(x=ice_forecast.date,y=ice_forecast.lower,name="Approx. residual range",line=dict(width=0),fill="tonexty",fillcolor="rgba(245,166,78,.18)"))
        f.add_trace(go.Scatter(x=ice_forecast.date,y=ice_forecast.prediction,name="Experimental forecast",line=dict(color="#f5a64e",dash="dash",width=2.5)))
        f.update_layout(template="plotly_dark",height=390,hovermode="x unified",yaxis_title="Extent (million km²)",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(f,use_container_width=True)
        st.dataframe(ice_scores.round(3),hide_index=True,use_container_width=True)
        st.caption("Sea ice is strongly seasonal. The forecast is an experimental statistical continuation and should not be interpreted as a seasonal ice outlook from an operational forecast centre.")
    except Exception as exc:
        st.warning(f"NSIDC sea-ice feed is temporarily unavailable or its format changed: {exc}")
        st.link_button("Open NSIDC Sea Ice Index archive", "https://nsidc.org/data/seaice_index/data-and-image-archive")

with water_tab:
    st.subheader("Global water-cycle indicators")
    st.caption(f"NASA POWER monthly estimates at the selected dashboard coordinate ({lat:.1f}°, {lon:.1f}°). This is a point-level series, not a global average. The regional API does not accept a whole-world bounding box.")
    variable = st.selectbox("Select NASA POWER parameter", ["PRECTOTCORR", "GWETROOT"], format_func=lambda x: {"PRECTOTCORR":"Precipitation (mm/day)","GWETROOT":"Root-zone soil wetness (fraction)"}[x])
    try:
        with st.spinner("Loading NASA POWER point observations…"):
            water = load_power_global_region(variable, 2001, datetime.now().year-1, lat, lon)
        water_forecast, water_scores, water_model, water_hist = _numeric_forecast(water, horizon=horizon, min_points=60)
        w1,w2,w3 = st.columns(3)
        w1.metric("Latest regional value", f"{water_hist.iloc[-1].value:.2f}", water_hist.iloc[-1].date.strftime("%B %Y"))
        w2.metric("Selected forecast model", water_model)
        w3.metric("Forecast endpoint", f"{water_forecast.iloc[-1].prediction:.2f}", water_forecast.iloc[-1].date.strftime("%B %Y"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=water_hist.date,y=water_hist.value,name="NASA POWER point observation",line=dict(color="#64c9d8",width=1.8)))
        f.add_trace(go.Scatter(x=water_forecast.date,y=water_forecast.upper,name="Upper residual range",line=dict(width=0),showlegend=False))
        f.add_trace(go.Scatter(x=water_forecast.date,y=water_forecast.lower,name="Approx. residual range",line=dict(width=0),fill="tonexty",fillcolor="rgba(245,166,78,.18)"))
        f.add_trace(go.Scatter(x=water_forecast.date,y=water_forecast.prediction,name="Experimental forecast",line=dict(color="#f5a64e",dash="dash",width=2.5)))
        unit = "mm/day" if variable == "PRECTOTCORR" else "fraction"
        f.update_layout(template="plotly_dark",height=390,hovermode="x unified",yaxis_title=unit,margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(f,use_container_width=True)
        st.dataframe(water_scores.round(3),hide_index=True,use_container_width=True)
        st.caption("NASA POWER monthly meteorological fields are model/reanalysis-based products with documented spatial and temporal limits. Regional averaging can conceal substantial local and land–ocean differences.")
    except Exception as exc:
        st.warning(f"NASA POWER regional feed is temporarily unavailable or this parameter is unsupported: {exc}")
        st.link_button("Open NASA POWER monthly API documentation", NASA_POWER_DOCS)

st.divider()
st.subheader("Data provenance for Earth Pulse 2.0")
st.markdown("- **Temperature:** NASA GISTEMP v4 global land–ocean temperature index.\n- **CO₂:** NOAA Global Monitoring Laboratory global marine surface monthly mean (complementary source; not NASA OCO-2).\n- **Sea level:** NOAA STAR satellite-altimetry time series, NASA/JPL-derived global product.\n- **Arctic sea ice:** NOAA at NSIDC Sea Ice Index Version 4.\n- **Water cycle:** NASA POWER monthly point meteorological service.")
st.caption("Forecasts are experimental statistical estimates. They are not official NASA, NOAA or NSIDC forecasts. Data availability, source revisions, missing values and changes in source-file formats can affect the dashboard.")
