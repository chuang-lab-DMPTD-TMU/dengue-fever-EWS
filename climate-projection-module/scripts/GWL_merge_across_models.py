# Access Snakemake variables
snakemake_input_files = snakemake.input
snakemake_output_file = snakemake.output[0]

print("Input files:", snakemake_input_files)

import xarray as xr

ds_list = [xr.open_dataset(f) for f in snakemake_input_files]
if not ds_list:
    raise ValueError("No NetCDF files found to concatenate! Check previous rules and paths.")

combined = xr.concat(ds_list, dim="gwl")
combined.to_netcdf(snakemake_output_file)
