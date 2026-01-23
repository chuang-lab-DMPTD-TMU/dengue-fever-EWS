import xarray as xr
import pandas as pd
from pathlib import Path
import dask

# -------------------------------
# 1️⃣ Open NetCDF with Dask chunks
# -------------------------------
# Why: This prevents loading the full dataset into RAM.
# Only small chunks are read into memory at a time.
# Chunking along 'time' is usually most effective for climate projections.
ds = xr.open_dataset(snakemake.input[0], chunks={"time": 10})

# Optional: limit chunk size to avoid spikes during operations
dask.config.set({"array.chunk-size": "500MB"})

# Print info for debugging
print(ds.time.min().values, ds.time.max().values)
print(ds.time)
print(ds.time.dtype)
print(ds.time.attrs)

# -------------------------------
# 2️⃣ Load config parameters
# -------------------------------
start_date = pd.to_datetime(snakemake.config["start_date"])
end_date = pd.to_datetime(snakemake.config["end_date"])
spatial_resolution = snakemake.config["spatial_resolution"]
temporal_resolution = snakemake.config["temporal_resolution"]
exclude_models = snakemake.config.get("exclude_models", [])

# -------------------------------
# 3️⃣ Temporal slicing (lazy)
# -------------------------------
# .sel() is lazy with Dask, so only the sliced chunks are loaded into memory
ds_sliced = ds.sel(time=slice(start_date, end_date))
print(ds_sliced.time)
print(ds_sliced.time.dtype)
print(ds_sliced.time.min().values, ds_sliced.time.max().values)

# -------------------------------
# 4️⃣ Spatial resolution adjustments
# -------------------------------
# .coarsen() with Dask is lazy; memory-efficient averaging
if spatial_resolution == "1deg":
    ds_sliced = ds_sliced.coarsen(lat=2, lon=2, boundary="trim").mean()
elif spatial_resolution == "pinpoint":
    # Take global mean; lazy with Dask
    ds_sliced = ds_sliced.mean(dim=["lat", "lon"], keep_attrs=True)

# -------------------------------
# 5️⃣ Temporal resolution adjustments
# -------------------------------
# Resampling is lazy with Dask; no memory spike
if temporal_resolution == "monthly":
    ds_sliced = ds_sliced.resample(time="1M").mean()
elif temporal_resolution == "daily":
    pass
else:
    raise ValueError(f"Unknown temporal_resolution: {temporal_resolution}")

# -------------------------------
# 6️⃣ Exclude models if needed
# -------------------------------
# .where() is lazy with Dask; avoids loading the full dataset
if exclude_models:
    ds_sliced = ds_sliced.where(~ds_sliced.model.isin(exclude_models), drop=True)

# -------------------------------
# 7️⃣ Save output (compute lazily in chunks)
# -------------------------------
# Writing with .to_netcdf() triggers computation, but Dask handles it in chunks
output_path = Path(snakemake.output[0])
output_path.parent.mkdir(parents=True, exist_ok=True)
ds_sliced.to_netcdf(output_path)

print("Slicing done!")
