import geopandas as gpd 
import pandas as pd

filtered_sea = gpd.read_parquet("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/external/geoparquet/gaul_2024_sea_filtered.parquet")

# This will show you every row where 'admin' is actually null
null_rows = filtered_sea[filtered_sea['admin'].isna()]
print(null_rows)

# This check if you have any non-string objects (like None or numbers) 
# that might be hiding in a column you expect to be all strings
print(filtered_sea['admin'].apply(type).unique())

# Reset index to ensure clean alignment
gdf_reset = filtered_sea.reset_index(drop=True)

for admin_value in gdf_reset['admin'].unique():
    # Use .loc to be explicit about row-level indexing
    subset_gdf = gdf_reset.loc[gdf_reset['admin'] == admin_value]
    
    if not subset_gdf.empty:
        filename = f"{str(admin_value).replace(' ', '_')}.parquet"
        subset_gdf.to_parquet(filename)