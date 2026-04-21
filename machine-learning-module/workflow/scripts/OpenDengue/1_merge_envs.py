import pandas as pd
from pathlib import Path
import json

SEA_DIR = Path("/home/patwuch/Documents/projects/Chuang-Lab-TMU/machine-learning-module/main/data/raw/SEA")
OUT_DIR = Path("/home/patwuch/Documents/projects/Chuang-Lab-TMU/machine-learning-module/main/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOIN_KEYS = ["admin", "name", "year_month"]


def to_monthly(df):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    return df


def expand_annual_to_monthly(df, value_cols):
    """Repeat annual rows for each month of the year."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["Date"]).dt.year
    df = df.merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    df["year_month"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df[["admin", "name", "year_month"] + value_cols]


def histogram_to_pct(hist):
    """Convert a pixel-count histogram dict/string to fractional percentages."""
    if isinstance(hist, str):
        hist = json.loads(hist)
    if not isinstance(hist, dict) or not hist or sum(hist.values()) == 0:
        return {}
    total = sum(hist.values())
    return {int(k): v / total for k, v in hist.items()}


# ---------------------------------------------------------------------------
print("Processing ERA5_LAND...")
era5 = to_monthly(pd.read_parquet(SEA_DIR / "ERA5_LAND_2011-01-01_to_2018-12-31.parquet"))
print(f"  ERA5_LAND schema:\n  {era5.columns.tolist()}")
era5_cols = [c for c in era5.columns if c.endswith(("_sum", "_mean"))]
era_monthly = era5.groupby(JOIN_KEYS)[era5_cols].mean().reset_index()
print(f"  ERA5 shape: {era_monthly.shape}")

# ---------------------------------------------------------------------------
# 2. CHIRPS
# ---------------------------------------------------------------------------
print("Processing CHIRPS...")
chirps = to_monthly(pd.read_parquet(SEA_DIR / "CHIRPS_2011-01-01_to_2018-12-31.parquet"))
print(f"  CHIRPS schema:\n  {chirps.columns.tolist()}")
chirps_monthly = chirps.groupby(JOIN_KEYS)["precipitation_sum"].mean().reset_index()
print(f"  CHIRPS shape: {chirps_monthly.shape}")

# ---------------------------------------------------------------------------
# 3. MODIS_LST
# ---------------------------------------------------------------------------
print("Processing MODIS_LST...")
lst = to_monthly(pd.read_parquet(SEA_DIR / "MODIS_LST_2011-01-01_to_2018-12-31.parquet"))
print(f"  MODIS_LST schema:\n  {lst.columns.tolist()}")
lst_monthly = lst.groupby(JOIN_KEYS)[["LST_Day_1km_mean", "LST_Night_1km_mean"]].mean().reset_index()
print(f"  MODIS_LST shape: {lst_monthly.shape}")

# ---------------------------------------------------------------------------
# 4. MODIS_NDVI_EVI
# ---------------------------------------------------------------------------
print("Processing MODIS_NDVI_EVI...")
ndvi = to_monthly(pd.read_parquet(SEA_DIR / "MODIS_NDVI_EVI_2011-01-01_to_2018-12-31.parquet"))
print(f"  MODIS_NDVI_EVI schema:\n  {ndvi.columns.tolist()}")
ndvi_monthly = ndvi.groupby(JOIN_KEYS)[["NDVI_mean", "EVI_mean"]].mean().reset_index()
print(f"  MODIS_NDVI_EVI shape: {ndvi_monthly.shape}")

# ---------------------------------------------------------------------------
# 5. MODIS_LULC (annual → monthly, histogram → % per class)
# ---------------------------------------------------------------------------
print("Processing MODIS_LULC...")
lulc = pd.read_parquet(SEA_DIR / "MODIS_LULC_2011-01-01_to_2018-12-31.parquet")
print(f"  MODIS_LULC schema:\n  {lulc.columns.tolist()}")

# Expand histogram → pct columns, using int keys from the start
hist_expanded = (
    lulc["LC_Type1_histogram"]
    .apply(histogram_to_pct)
    .apply(pd.Series)
)

# Sort columns numerically, rename, then fillna(0) to catch ALL missing values
hist_expanded = hist_expanded[sorted(hist_expanded.columns)]
hist_expanded.columns = [f"LC_Type1_pct_class{c}" for c in hist_expanded.columns]
hist_expanded = hist_expanded.fillna(0)

lulc_hist = pd.concat([lulc[["admin", "name", "Date"]], hist_expanded], axis=1)

lulc_monthly = expand_annual_to_monthly(lulc_hist, list(hist_expanded.columns))
print(f"  MODIS_LULC shape: {lulc_monthly.shape}")
print(f"  LC classes found: {list(hist_expanded.columns)}")
print(f"  Null check: {lulc_monthly[hist_expanded.columns].isnull().sum().sum()} nulls")

# ---------------------------------------------------------------------------
# 6. WorldPop (annual → monthly)
# ---------------------------------------------------------------------------
print("Processing WorldPop...")
pop = pd.read_parquet(SEA_DIR / "WorldPop_2011-01-01_to_2018-12-31.parquet")
print(f"  WorldPop schema:\n  {pop.columns.tolist()}")
pop_monthly = expand_annual_to_monthly(pop, ["population_sum"])
print(f"  WorldPop shape: {pop_monthly.shape}")

# ---------------------------------------------------------------------------
# 7. Merge all
# ---------------------------------------------------------------------------
print("\nMerging all datasets...")
combined = chirps_monthly.copy()

for ds_name, df in [
    ("ERA5_LAND",      era_monthly),
    ("MODIS_LST",      lst_monthly),
    ("MODIS_NDVI_EVI", ndvi_monthly),
    ("MODIS_LULC",     lulc_monthly),
    ("WorldPop",       pop_monthly),
]:
    combined = combined.merge(df, on=JOIN_KEYS, how="outer")
    print(f"  after {ds_name}: {combined.shape}")

# ---------------------------------------------------------------------------
# 8. Finalise
# ---------------------------------------------------------------------------
combined.rename(columns={"year_month": "Date"}, inplace=True)
id_cols = ["admin", "name", "Date"]
data_cols = [c for c in combined.columns if c not in id_cols]
combined = combined[id_cols + data_cols]

out_csv = OUT_DIR / "SEA_env_combined_monthly_2011-2018.csv"
combined.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(f"Final shape: {combined.shape}")
print(f"Columns:\n{combined.columns.tolist()}")
print(f"Null check (LULC cols): {combined[[c for c in combined.columns if 'LC_Type1' in c]].isnull().sum().sum()} nulls")