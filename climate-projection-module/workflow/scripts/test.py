import xarray as xr
import numpy as np

def summarize_dataset(ds, fill_candidates=[-99.9, -9999]):
    print("=== Dataset Summary ===\n")
    
    # Dataset-level attributes
    print("Dataset attributes:")
    if ds.attrs:
        for k, v in ds.attrs.items():
            print(f"  {k}: {v}")
    else:
        print("  (None)")
    
    print("\nCoordinates:")
    for coord in ds.coords:
        print(f"\nCoordinate: {coord}")
        print(f"  Values: min={ds[coord].min().values}, max={ds[coord].max().values}")
        if ds[coord].attrs:
            for k, v in ds[coord].attrs.items():
                print(f"  {k}: {v}")
        else:
            print("  (No attributes)")

    print("\nVariables:")
    for var in ds.data_vars:
        values = ds[var].values
        masked = np.isnan(values)
        n_masked = masked.sum()
        n_fill = np.isin(values, fill_candidates).sum()
        
        print(f"\nVariable: {var}")
        print(f"  Total elements: {values.size}")
        print(f"  Masked (NaN) count: {n_masked}")
        print(f"  Fill candidate count: {n_fill}")
        
        # Variable attributes
        if ds[var].attrs:
            print("  Attributes:")
            for k, v in ds[var].attrs.items():
                print(f"    {k}: {v}")
        else:
            print("  (No attributes)")
        
        # Encoding info
        if ds[var].encoding:
            print("  Encoding:")
            for k, v in ds[var].encoding.items():
                print(f"    {k}: {v}")
        else:
            print("  (No encoding info)")



# Example usage
ds = xr.open_dataset("/home/patwuch/Documents/projects/Chuang_Lab_TMU/ssp_rcp/work/data/processed/NetCDF/AR6_NetCDF/historical/ssp245.nc")
summarize_dataset(ds)
for coord in ds.coords:
    print(coord, ds[coord].values)
