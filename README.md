# Chuang Lab @ Taipei Medical University

**Department of Molecular Parasitology and Tropical Diseases**

We model the relationship between dengue fever epidemiology and climate change by integrating geospatial climate projections, reported infection cases, and socioeconomic pathway scenarios. We apply machine learning and deep learning methods to predict the spatiotemporal spread of dengue fever under future climate conditions.

[Snakemake](https://snakemake.readthedocs.io/en/stable/) is used for workflow management and data provenance tracking across all modules.

---

## Modules

### (A) Climate Projection Module

Processes and harmonizes multi-source climate projection data into standardized inputs for downstream modelling.

**Data sources:**
- **AR6** — CMIP6 model ensemble (CCMC-ESM2, BCC-CSM2-MR, GFDL-ESM4, CanESM5, ACCESS-CM2, etc.) under IPCC AR6 scenarios: historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
- **GWL** — Global Warming Level datasets at 1.5 °C, 2.0 °C, 3.0 °C, and 4.0 °C above pre-industrial baseline
- **TREAD** — Taiwan Regional Extreme Analysis Data; high-resolution downscaled climate data for Taiwan and Kaohsiung

**Climate variables:** precipitation, T_min, T_max, T_avg, relative humidity, wind speed, sunshine hours

**Pipeline steps** (CSV → NetCDF → TSV):
1. Convert annual CSVs to NetCDF and merge across years
2. Merge across models/scenarios per climate variable
3. Merge across climate variables into a single dataset
4. Slice by date range and spatial resolution
5. Export to TSV for the ML module

**Output:** Merged NetCDF files (e.g. `AR6_all.nc`, `TREAD_all.nc`) and TSV tables covering 2003–2023

---

### (B) Dengue Infection Module

Aggregates and standardises dengue case data from global and regional sources at fine spatiotemporal resolution.

**Data sources:**
- **OpenDengue** — Global case counts (WHO regions SEARO, WPRO, EMRO), 2000–2025, admin-0/1/2 levels, weekly and monthly resolution
- **Indonesia** — Province-level monthly infection and death counts from Ministry of Health Excel reports
- **Taiwan (Tainan)** — Sub-district (BSA) level case data joined with demographic and shapefile data

**Pipeline steps (OpenDengue):**
1. Extract records for Indo-Pacific WHO regions
2. Filter to 10 Southeast Asian countries, 2011–2018
3. Standardise admin-1 region names (spell-check and harmonisation)
4. Impute missing regional data from national aggregates
5. Spatial join with FAO GAUL boundaries → GeoParquet output
6. Moran's I spatial autocorrelation analysis

**Additional pipelines:** Indonesia data cleaning and wavelet analysis; Taiwan/Tainan BSA-level spatial joins

**Output:** Standardised GeoParquet and CSV tables with columns for country, province, district, date range, case counts, spatial resolution, and temporal resolution

---

### (C) Machine Learning Module

Trains and evaluates spatiotemporal models for dengue prediction using outputs from the climate and infection modules.

**Models:**

| Model | Scope | Notes |
|---|---|---|
| **XGBoost** | National & regional (Indonesia) | Classifier and regressor variants; K-Fold and walk-forward validation |
| **ST-GNN** | Regional (Indonesia) | Spatiotemporal graph neural network using a region adjacency graph |

**Feature engineering:** temporal lags, rolling statistics, seasonal decomposition, optional land-use/LULC variables

**Hyperparameter tuning:** Optuna (Bayesian optimisation, 40–50 trials per study)

**Experiment tracking:** Weights & Biases (online/offline modes)

**Model interpretation:** SHAP feature importance and Integrated Gradients attribution for ST-GNN

**Computational requirements:** CUDA 12.1, 32 GB+ RAM, GPU recommended (A100/V100)

**Output:** Trained model checkpoints (`best.pt`), metric JSON files, feature importance plots, and attribution heatmaps

---

## Repository Structure

```
├── climate-projection-module/
│   ├── workflow/
│   │   ├── Snakefile
│   │   ├── config.yaml
│   │   ├── rules/          # AR6.smk, GWL.smk, TREAD.smk
│   │   └── scripts/
│   └── main/data/          # raw/, interim/, processed/
│
├── dengue-infection-module/
│   ├── workflow/
│   │   ├── Snakefile
│   │   └── scripts/        # OpenDengue/, Indonesia/, Taiwan/
│   └── main/
│       ├── raw/            # OpenDengue/, IN_DENGUE/, TW_DENGUE/
│       ├── interim/
│       └── external/       # FAO GAUL shapefiles, Natural Earth
│
└── machine-learning-module/
    ├── workflow/
    │   ├── Snakefile
    │   ├── config.yaml
    │   └── scripts/        # XGBoost, ST-GNN, integrated gradients
    └── main/
        ├── data/external/  # Indonesia & Taiwan shapefiles
        └── reports/
```

## Key Dependencies

| Library | Purpose |
|---|---|
| Snakemake | Workflow management |
| xarray / netCDF4 | Climate data processing |
| Dask / dask-cuda | Parallel and GPU-accelerated computation |
| GeoPandas / Shapely | Geospatial data handling |
| XGBoost | Gradient boosting models |
| PyTorch / PyTorch Geometric | Deep learning and graph neural networks |
| RAPIDS (cuDF, cuML) | GPU-accelerated data processing |
| Optuna | Hyperparameter optimisation |
| SHAP | Model interpretability |
| Weights & Biases | Experiment tracking |
