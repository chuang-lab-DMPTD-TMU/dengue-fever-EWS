import xarray as xr
import dask
import os

# Set Dask config globally
dask.config.set({"array.slicing.split_large_chunks": True})

snakemake_input_files = snakemake.input
snakemake_output_file = snakemake.output[0]

def preprocess(ds, filepath):
    # Extract model name from filename
    model_name = os.path.basename(filepath).split('_')[0]
    
    # Keep 'ssp' if it exists, otherwise ignore
    coords_to_keep = {'lat', 'lon', 'time', 'model'}
    if 'ssp' in ds.coords:
        coords_to_keep.add('ssp')
    
    # Drop unnecessary coordinates
    to_drop = [v for v in ds.coords if v not in coords_to_keep]
    ds = ds.drop_vars(to_drop)
    
    # Expand dataset to include model dimension if not present
    if 'model' not in ds.coords:
        ds = ds.expand_dims(model=[model_name])
    
    # Sort by time to ensure proper ordering
    ds = ds.sortby('time')
    
    return ds

# Open and preprocess datasets with explicit chunking
datasets = [
    preprocess(xr.open_dataset(f, chunks={'time': 500}), f)
    for f in snakemake_input_files
]

# Concatenate along 'model', preserving coordinates minimally
combined = xr.concat(
    datasets,
    dim='model',
    data_vars='minimal',    # only concat shared variables
    coords='all',           # keep coordinates per dataset
    compat='override',      # allow variables to differ
    join='outer'            # union of all coordinates
)


# Double-check that time is strictly increasing across historical -> future
combined = combined.sortby('time')

# Use compression for efficiency
encoding = {var: {'zlib': True, 'complevel': 1} for var in combined.data_vars}

# Write to NetCDF
combined.to_netcdf(
    snakemake_output_file,
    engine='netcdf4',
    encoding=encoding,
    compute=True
)
