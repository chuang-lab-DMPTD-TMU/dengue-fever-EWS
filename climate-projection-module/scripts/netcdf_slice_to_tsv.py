import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np

# Load dataset
ds = xr.open_dataset(snakemake.input[0])

# Track whether any fill values were replaced
fill_replaced = False

# Replace fill values with NaN for all variables
for var in ds.data_vars:
    fill_value = ds[var].attrs.get("_FillValue") or ds[var].attrs.get("missing_value")
    if fill_value is not None:
        ds[var] = ds[var].where(ds[var] != fill_value, np.nan)
        fill_replaced = True

# Convert to tabular form
df = ds.to_dataframe().reset_index()

# Ensure output directory exists
output_path = Path(snakemake.output[0])
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save as TSV, using "NA" for missing values
df.to_csv(output_path, sep="\t", index=False, na_rep="NA")

# Print statements
if fill_replaced:
    print(f"Conversion complete! Fill values replaced with NaN and TSV saved at {output_path}")
else:
    print(f"No fill values detected. TSV saved at {output_path}")
