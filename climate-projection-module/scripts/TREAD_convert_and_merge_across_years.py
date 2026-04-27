import pandas as pd
import numpy as np
from netCDF4 import Dataset, date2num
from pathlib import Path

csv_files = snakemake.input
out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

first = True

# Choose a reference date earlier than your earliest data
time_units = "days since 1850-01-01 00:00:00"

for f in csv_files:
    df = pd.read_csv(f)
    print(f"Processing file: {f}")
    df.columns = df.columns.str.strip()
    time_cols = [c for c in df.columns if c not in ["LON", "LAT"]]
    valid_time_cols = [c for c in time_cols if c.isdigit()]
    if not valid_time_cols:
        continue
    times = pd.to_datetime(valid_time_cols, format="%Y%m%d")

    lat = np.sort(df["LAT"].unique())
    lon = np.sort(df["LON"].unique())
    data = np.full((len(times), len(lat), len(lon)), np.nan, dtype=np.float32)
    for i, tcol in enumerate(valid_time_cols):
        grid = df.pivot_table(values=tcol, index="LAT", columns="LON").reindex(index=lat, columns=lon)
        data[i] = grid.values

    # Convert times to numeric values (days since reference)
    times_num = date2num(times.to_pydatetime(), units=time_units)

    if first:
        # Create file and define dimensions
        nc = Dataset(out_path, 'w', format='NETCDF4')
        nc.createDimension('time', None)  # unlimited
        nc.createDimension('lat', len(lat))
        nc.createDimension('lon', len(lon))

        times_var = nc.createVariable('time', 'f8', ('time',))
        times_var.units = time_units
        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lon_var = nc.createVariable('lon', 'f4', ('lon',))
        data_var = nc.createVariable(snakemake.wildcards.clim_factor, 'f4', ('time', 'lat', 'lon'), zlib=True)

        lat_var[:] = lat
        lon_var[:] = lon
        times_var[:] = times_num
        data_var[:] = data

        first = False
    else:
        # Append along time
        nc = Dataset(out_path, 'a')
        t_len = len(nc.dimensions['time'])
        nc.variables['time'][t_len:t_len+len(times)] = times_num
        nc.variables[snakemake.wildcards.clim_factor][t_len:t_len+len(times), :, :] = data

    nc.close()
