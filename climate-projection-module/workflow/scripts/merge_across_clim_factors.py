import xarray as xr
from pathlib import Path

FILL_VALUE = -99.9

ds_list = [
    xr.open_dataset(f, chunks="auto")  # Setting dask chunks to auto to prevent OOM errors
    for f in snakemake.input
]

masked = []
for ds in ds_list:
    ds = ds.where(ds != FILL_VALUE)
    masked.append(ds)

merged = xr.merge(masked)

Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)

merged.to_netcdf(snakemake.output[0])
