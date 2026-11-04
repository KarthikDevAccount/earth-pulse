# Earth Pulse 2.0 — Global Climate Intelligence

Earth Pulse is a public-deployment-ready Streamlit dashboard for global climate observations and transparent, experimental short-horizon statistical forecasts.

## Included indicators

1. **Global surface temperature anomaly** — NASA GISTEMP v4 monthly land–ocean temperature index, relative to its 1951–1980 baseline.
2. **Atmospheric CO₂** — NOAA Global Monitoring Laboratory global marine surface monthly mean. This is explicitly a complementary NOAA observing dataset, not an OCO-2 satellite product.
3. **Global mean sea level** — satellite-altimetry time series distributed by NOAA STAR's Laboratory for Satellite Altimetry and derived from NASA/JPL product work. The source's measurement coverage, reference and units must be retained when interpreting it.
4. **Arctic sea-ice extent** — NOAA at NSIDC Sea Ice Index Version 4 monthly data.
5. **Precipitation and root-zone soil wetness** — NASA POWER regional monthly service for a broad 60°S–70°N latitude belt and all longitudes. This is a regional aggregate, not a full polar-to-polar global gridded mean.
6. **NASA POWER point explorer** — selected-coordinate meteorological series; not a global statistic.

The app handles each feed independently. If one upstream service is unavailable or changes its file format, that module shows a source-specific warning while the other modules continue to load.

## Forecast methodology and limitations

- Ridge regression, Random Forest and HistGradientBoosting are compared using a chronological holdout.
- Model selection uses lowest holdout MAE. MAE, RMSE and R² are displayed where calculable.
- The selected estimator is refit on all available data for the forward forecast.
- Shaded ranges are approximate residual-based ranges widened with forecast lead. They are **not calibrated 95% prediction intervals**, and they do not represent all climate-system, emissions-scenario, measurement or structural uncertainties.
- These are statistical extrapolations, not official NASA, NOAA or NSIDC forecasts, and not substitutes for physical climate-model ensembles.
- Monthly climate variables have strong seasonality. The model includes annual and semiannual harmonic features, but this is a baseline method and can miss regime shifts and long-term forcing changes.
- Forecast accuracy is not guaranteed. Use the displayed chronological holdout metrics and source dates when interpreting results.

## Data freshness

The dashboard displays the latest timestamp present in each source file. Different datasets have different publication schedules. NASA GISTEMP typically publishes monthly data with a delay; NASA POWER meteorological monthly products may be available only through the previous year. The app does not label older observations as current-month measurements.

## Run locally for development

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy/update on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `Dockerfile` to the GitHub repository connected to the Streamlit app.
2. Open the app's Streamlit Community Cloud settings and verify the repository, branch and main file (`app.py`).
3. Reboot the app after the GitHub commit. Streamlit Cloud installs packages from `requirements.txt` during deployment.
4. Check the app logs if an upstream data provider is unavailable or has changed its format.

No API key is required for the public CSV/API endpoints used in this project. Hosting-provider account and plan limits apply.

## Data sources

- NASA GISTEMP: https://data.giss.nasa.gov/gistemp/
- NASA POWER monthly API: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- NOAA Global Monitoring Laboratory CO₂ trends: https://gml.noaa.gov/ccgg/trends/gl_data.html
- NOAA STAR global mean sea-level time series: https://www.star.nesdis.noaa.gov/sod/lsa/SeaLevelRise/LSA_SLR_timeseries_global.php
- NOAA at NSIDC Sea Ice Index archive: https://nsidc.org/data/seaice_index/data-and-image-archive

## Project files

- `app.py` — Streamlit application, data ingestion, charts and forecasts.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container deployment option.
- `.gitignore` — excludes local virtual environments and generated files.

Earth Pulse is an independent educational analysis dashboard. NASA, NOAA and NSIDC do not operate, endorse or certify this application or its model forecasts. Cite the underlying dataset and access date in research or publications.
