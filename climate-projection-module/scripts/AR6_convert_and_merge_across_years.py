import pandas as pd
import xarray as xr
import dask
from pathlib import Path

# --------------------
# Snakemake inputs
# --------------------
csv_files = snakemake.input
output_file = snakemake.output[0]
clim_factor = snakemake.wildcards.clim_factor
ssp = snakemake.wildcards.ar6

if not csv_files:
    raise ValueError(f"No CSV files found for {clim_factor}/{ssp}")

all_da = []

# Use threads (good for netCDF + pandas)
dask.config.set(scheduler="threads")

for f in csv_files:
    print("Reading:", f)
    df = pd.read_csv(f)

    if df.empty:
        raise ValueError(f"Empty CSV: {f}")

    # --------------------
    # Clean & validate
    # --------------------
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {f}")

    time_cols = [
        c for c in df.columns
        if c not in ["LAT", "LON"] and c.isdigit()
    ]

    valid_time_cols = []
    for c in time_cols:
        try:
            pd.to_datetime(c, format="%Y%m%d")
            valid_time_cols.append(c)
        except ValueError:
            pass

    if not valid_time_cols:
        raise ValueError(f"No valid date columns in {f}")

    # --------------------
    # Melt to long (still pandas → cheap)
    # --------------------
    df_long = df.melt(
        id_vars=["LAT", "LON"],
        value_vars=valid_time_cols,
        var_name="time",
        value_name=clim_factor
    )

    df_long["time"] = pd.to_datetime(df_long["time"], format="%Y%m%d")

    # IMPORTANT: sort BEFORE xarray
    df_long = df_long.sort_values(["time", "LAT", "LON"])

    # Extract model name
    model = Path(f).name.split("_")[-2]

    # --------------------
    # Convert to xarray (still NumPy-backed)
    # --------------------
    da = (
        df_long
        .set_index(["time", "LAT", "LON"])
        .to_xarray()[clim_factor]
    )

    # Promote model & ssp to dimensions
    da = da.expand_dims(
        model=[model],
        ssp=[ssp]
    )

    all_da.append(da)

# --------------------
# Combine safely
# --------------------
ds = xr.combine_by_coords(
    all_da,
    combine_attrs="override"
)

# --------------------
# Chunk ONCE, at the end
# --------------------
ds = ds.chunk({
    "time": 365,    # or 365 if daily
    "LAT": -1,      # keep spatial contiguous
    "LON": -1,
})

# --------------------
# Write output
# --------------------
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

ds.to_netcdf(
    output_file,
    engine="netcdf4",
    compute=True
)
