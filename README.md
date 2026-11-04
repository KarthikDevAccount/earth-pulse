# Earth Pulse — Global Climate Intelligence

A public-deployment-ready Streamlit dashboard using NASA GISTEMP global monthly land–ocean temperature anomalies and NASA POWER point-level meteorological data. It includes a transparent experimental 24-month statistical forecast and chronological holdout evaluation.

## Important scope and scientific limitations

- NASA GISTEMP is updated about monthly, generally around the 10th, using the previous month's observations. The app displays the latest observation actually present in the NASA source file; it never relabels a prior month as the current month.
- The forecast is a statistical extrapolation, not an official NASA forecast and not a replacement for physical climate-model ensembles.
- The uncertainty band is an approximate residual-based interval from a chronological holdout, not a calibrated climate projection interval.
- NASA POWER's selected-coordinate chart is point-level. It is not presented as a global average.
- Global CO2, sea-level, Arctic sea-ice, soil-moisture and global precipitation pipelines are not represented as live or forecasted in this first release. They require dedicated dataset ingestion, harmonisation, validation and uncertainty treatment.

## Run locally for development

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Publish as a public website (Streamlit Community Cloud)

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create an app, select the repository and branch, and set the main file to `app.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` URL.
5. In the app settings, ensure the app is public if you want anyone with the URL to access it.

No NASA API key is required for the GISTEMP CSV or NASA POWER public API used here. Hosting-provider account approval and current plan limits apply.

## Data endpoints

- GISTEMP v4 CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
- GISTEMP overview: https://data.giss.nasa.gov/gistemp/
- NASA POWER monthly API docs: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/

## Model approach

Three regressors are compared using a chronological holdout: Ridge regression, Random Forest and HistGradientBoosting. Inputs encode time and annual/semiannual seasonality. The selected model minimises holdout MAE, then is refit on all available observations. This is intentionally a baseline; future versions should compare against persistence, linear trend, seasonal-naive baselines and official multi-model climate projections, and use rolling-origin evaluation.

## Suggested next scientific integrations

1. NASA OCO-2 / OCO-2 GEOS monthly XCO2 for atmospheric carbon dioxide, with product-specific coverage and bias caveats.
2. NASA Sea Level Change / satellite altimetry data for global mean sea level.
3. NASA/NSIDC sea-ice concentration and extent products for Arctic sea ice.
4. NASA GISTEMP gridded products for spatial anomaly maps and regional summaries.
5. A reproducible ETL job, database, dataset versioning and monitoring before labelling any additional parameter live.
