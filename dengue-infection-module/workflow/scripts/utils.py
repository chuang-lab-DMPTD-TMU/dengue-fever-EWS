# Import basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os 
from pathlib import Path
import sys
from tqdm import tqdm  # Import tqdm for progress bars

def notebook_find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()  # fallback

def find_underrepresented_ids(df, id_column='ID_2'):
    """
    Finds and returns underrepresented IDs based on the most common count of occurrences.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the ID column.
    id_column (str): The name of the column containing IDs. Default is 'ID_2'.

    Returns:
    list: Underrepresented IDs.
    """
    # Count the number of occurrences of each ID
    id_counts = df[id_column].value_counts().sort_index()

    # Determine the expected count (most common count)
    expected_count = id_counts.mode()[0]

    # Find IDs with counts less than the expected
    underrepresented_ids = id_counts[id_counts < expected_count]

    # Convert index of underrepresented IDs to a list
    underrepresented_ids_list = underrepresented_ids.index.tolist()

    # Print expected count and underrepresented IDs
    print("Expected count per {}:".format(id_column), expected_count)
    print("IDs with less than expected count:")
    print(underrepresented_ids)

    return underrepresented_ids_list

def fill_missing_polygons(dataframe, shapefile, missing_ids, stat_cols):
    """
    Fill missing polygons in the dataset using the nearest available polygon's data.

    Parameters:
        dataframe (pd.DataFrame): The main DataFrame with 'ID_2' and 'Date'.
        shapefile (gpd.GeoDataFrame): Shapefile GeoDataFrame with 'ID_2' and geometries.
        missing_ids (set or list): Set or list of missing 'ID_2' values.
        stat_cols (list): List of statistic columns to be considered (optional, for reference).

    Returns:
        pd.DataFrame: The updated DataFrame with rows for missing IDs filled in.
    """
    
    # Ensure CRS is projected
    shapefile = shapefile.to_crs(epsg=3857)

    # Compute centroids
    shapefile["centroid"] = shapefile.geometry.centroid

    # Get non-missing polygons
    non_missing_polygons = shapefile[~shapefile["ID_2"].isin(missing_ids)].copy()

    # Map nearest polygon for each missing ID_2
    nearest_mapping = {}

    print("Finding nearest polygons for missing ID_2 values...")
    for ID in tqdm(missing_ids, desc="Computing nearest polygons"):
        missing_centroid = shapefile[shapefile["ID_2"] == ID].centroid.iloc[0]
        non_missing_polygons["distance"] = non_missing_polygons["centroid"].distance(missing_centroid)
        nearest_mapping[ID] = non_missing_polygons.loc[non_missing_polygons["distance"].idxmin(), "ID_2"]

    # Unique dates in the dataset
    unique_dates = dataframe["Date"].unique()

    # Prepare to store new rows
    missing_rows = []
    total_rows = 0

    print("\nDuplicating rows for missing values...")
    for missing_ID in tqdm(missing_ids, desc="Processing missing ID_2"):
        nearest_ID = nearest_mapping[missing_ID]
        nearest_rows = dataframe[dataframe["ID_2"] == nearest_ID]

        for date in tqdm(unique_dates, desc=f"Assigning dates for ID_2={missing_ID}", leave=False):
            if not ((dataframe["ID_2"] == missing_ID) & (dataframe["Date"] == date)).any():
                match = nearest_rows[nearest_rows["Date"] == date]
                if not match.empty:
                    new_row = match.iloc[0].copy()
                    new_row["ID_2"] = missing_ID
                    missing_rows.append(new_row)
                    total_rows += 1
        tqdm.write(f"✅ Finished ID_2={missing_ID}, total rows processed: {total_rows}")

    # Create DataFrame and append to original
    missing_df = pd.DataFrame(missing_rows)
    updated_df = pd.concat([dataframe, missing_df], ignore_index=True)

    print(f"\n✅ Process completed! {total_rows} rows added.")
    return updated_df
