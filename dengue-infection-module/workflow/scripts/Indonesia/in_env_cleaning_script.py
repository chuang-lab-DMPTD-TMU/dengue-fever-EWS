import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os
from pathlib import Path
import sys
from tqdm import tqdm

# Define project root based on notebook location
def notebook_find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()  # fallback

PROJECT_ROOT = notebook_find_project_root(Path.cwd())
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
IN_DIR = PROJECT_ROOT / "data" / "processed" / "INDONESIA"

# Loop through files and process CSV files
for file in os.listdir(INTERIM_DIR):
    if file.endswith('.csv'):
        # Create the variable name based on the file name (without extension)
        dataframe_name = file.split('.')[0].lower()

        # Load the CSV file into a DataFrame
        file_path = os.path.join(INTERIM_DIR, file)
        df = pd.read_csv(file_path)

        # Assign the DataFrame to the variable name in globals
        globals()[dataframe_name] = df

        # Print out the name of the dataframe and check the dataframe content
        print(f'DataFrame created: {dataframe_name}')

'''Creating Daily, ID_2 level statisics.
ENSO, DMI, and LULC data are not loaded at the daily level.'''
# First drop the 'Unnamed: 0' column from the era_chirps_id_1993_2023 DataFrame
env_daily = era_chirps_id_1993_2023.drop(columns=['Unnamed: 0'])
# Remove redundant suffixes from column names
columns_to_check = [
    'temperature_2m_MEAN', 'temperature_2m_min_MEAN',
    'temperature_2m_max_MEAN', 'potential_evaporation_sum_MEAN',
    'total_evaporation_sum_MEAN', 'precipitation_MEAN'
]
# Create a dictionary for renaming
rename_dict = {col: col.replace('_MEAN', '') for col in env_daily.columns if col in columns_to_check and '_MEAN' in col}
env_daily.rename(columns=rename_dict, inplace=True)

# # Change unit of temperature statistics from Kelvin to Celsius (if appicable)
# env_daily['temperature_2m'] = env_daily['temperature_2m'].abs() - 273.15
# env_daily['temperature_2m_min'] =  env_daily['temperature_2m_min'].abs() - 273.15
# env_daily['temperature_2m_max'] =  env_daily['temperature_2m_max'].abs() - 273.15

# Change unit of evaporation statistics from m3 to ml
env_daily['potential_evaporation_sum'] = env_daily['potential_evaporation_sum'].abs() * 1000
env_daily['total_evaporation_sum'] = env_daily['total_evaporation_sum'].abs() * 1000

# Make sure the 'Date' column is in datetime format
env_daily['Date'] = pd.to_datetime(env_daily['Date'])
# Create a 'YearMonth' column for merging
env_daily['YearMonth'] = env_daily['Date'].dt.to_period('M')
# Convert YearMonth to timestamp for merging
env_daily['YearMonth'] = env_daily['YearMonth'].dt.to_timestamp()
# Save full dataframe
env_daily.to_csv(IN_DIR / 'daily_env_id_1993_2023.csv', index=False)
# Filter the dataframe for the years 2010 to 2023
filtered_daily = env_daily[(env_daily['YearMonth'].dt.year >= 2010) & (env_daily['YearMonth'].dt.year <= 2023)].reset_index(drop=True)
filtered_daily.to_csv(IN_DIR / 'daily_env_id_2010_2023.csv', index=False)


'''This section creates monthly statisitcs by ID_2, which includes ENSO, DMI, and LULC data.'''
df = env_daily.copy()
# Drop 'Doy' and replace 'YearMonth' with first day of month
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()
# Define relevant column sets
temperature_cols = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']
evap_precip_cols = ['potential_evaporation_sum', 'total_evaporation_sum', 'precipitation']
keep_cols = ['YearMonth', 'ID_2', 'Region'] + temperature_cols + evap_precip_cols
# Filter necessary columns
df = df[keep_cols]

# Group by YearMonth and ID_2
monthly_id = df.groupby(['YearMonth', 'ID_2'], as_index=False).agg({
    **{col: 'mean' for col in temperature_cols},
    **{col: 'sum' for col in evap_precip_cols},
    'Region': 'first'  # Assuming region does not change within ID_2
})

# Calculate 30-year average for anomaly calculation
start_year_30yr_avg = 1993
end_year_30yr_avg = 2022 # 1993 to 2022 is 30 years
monthly_id['Month'] = monthly_id['YearMonth'].dt.month
print(monthly_id.columns)
for col in temperature_cols + evap_precip_cols:
    # Calculate 30-year monthly average for each ID_2
    monthly_30yr_avg = monthly_id[(monthly_id['YearMonth'].dt.year >= start_year_30yr_avg) &
                                  (monthly_id['YearMonth'].dt.year <= end_year_30yr_avg)].groupby(['ID_2', 'Month'])[col].mean().reset_index()
    monthly_30yr_avg.rename(columns={col: f'{col}_30yr_avg'}, inplace=True)

    # Merge 30-year average back to the main DataFrame
    monthly_id = monthly_id.merge(monthly_30yr_avg, on=['ID_2', 'Month'], how='left')

    # Calculate anomaly
    monthly_id[f'{col}_ANOM'] = monthly_id[col] - monthly_id[f'{col}_30yr_avg']

    # Drop the temporary 30yr_avg column
    monthly_id.drop(columns=[f'{col}_30yr_avg'], inplace=True)
monthly_id.drop(columns=['Month'], inplace=True)

# First, melt dmi dataframe (convert from wide to long format)
dmi_long = dmi.melt(id_vars='Year', var_name='Month', value_name='DMI')
dmi_long['Month'] = dmi_long['Month'].astype(int)  # ensure Month is integer
dmi_long['YearMonth'] = pd.to_datetime(dmi_long['Year'].astype(str) + '-' + dmi_long['Month'].astype(str).str.zfill(2))

# Melt dmi_east dataframe
dmi_east_long = dmi_east.melt(id_vars='Year', var_name='Month', value_name='DMI_East')
dmi_east_long['Month'] = dmi_east_long['Month'].astype(int)
dmi_east_long['YearMonth'] = pd.to_datetime(dmi_east_long['Year'].astype(str) + '-' + dmi_east_long['Month'].astype(str).str.zfill(2))

# Create a matching 'YearMonth' column in enso dataframe
enso_sst['YearMonth'] = pd.to_datetime(enso_sst['Year'].astype(str) + '-' + enso_sst['Month'].astype(str).str.zfill(2))
# Columns to add from enso
enso_columns_to_add = ['YearMonth', 'NINO1+2', 'ANOM1+2', 'NINO3', 'ANOM3', 'NINO4', 'ANOM4', 'NINO3.4', 'ANOM3.4']
# Merge enso dataframes
monthly_id = monthly_id.merge(enso_sst[enso_columns_to_add], on='YearMonth', how='left')
# Merge DMI
monthly_id = monthly_id.merge(dmi_long[['YearMonth', 'DMI']], on='YearMonth', how='left')
# Merge DMI East
monthly_id = monthly_id.merge(dmi_east_long[['YearMonth', 'DMI_East']], on='YearMonth', how='left')

# Compute evaporative_stress_index and aridity_index
monthly_id['evaporative_stress_index'] = (
    monthly_id['total_evaporation_sum'] / monthly_id['potential_evaporation_sum'])
monthly_id['aridity_index'] = (
    monthly_id['precipitation'] / monthly_id['potential_evaporation_sum'])

# Add lulc class columns
# 1. Create a mapping of ID_2 to Region from dengue_by_ID_2
id2_region_map = env_daily[['ID_2', 'Region']].drop_duplicates()
# 2. Add 'Region' column to lulc_final
lulc_200_full = pd.merge(lulc_200_full, id2_region_map, on='ID_2', how='left')
# Define LULC class columns
class_columns = [
    'Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95',
    'Class_30', 'Class_20', 'Class_10', 'Class_90', 'Class_80']
# Compute True_Area_* columns based on class proportions and Class_sum
for col in class_columns:
    lulc_200_full[f'True_Area_{col}'] = lulc_200_full[col] * lulc_200_full['Class_sum']
# Select only the necessary columns from lulc_200_full
true_area_cols = [f'True_Area_{col}' for col in class_columns]
all_lulc_columns = ['ID_2', 'Class_sum', 'Region'] + class_columns + true_area_cols

# Merge into monthly_id
monthly_id = pd.merge(monthly_id, lulc_200_full[all_lulc_columns], on=['ID_2','Region'], how='left')
monthly_id.to_csv(IN_DIR / 'monthly_env_id_1993_2023.csv', index=False)
# Filter the dataframe for the years 2010 to 2023
filtered_monthly_id = monthly_id[(monthly_id['YearMonth'].dt.year >= 2010) &
                                  (monthly_id['YearMonth'].dt.year <= 2023)].reset_index(drop=True)
filtered_monthly_id.to_csv(IN_DIR / 'monthly_env_id_2010_2023.csv', index=False)


'''This section creates monthly statisitcs by region, which includes ENSO, DMI, and LULC data.'''

# Reset df variable to env_daily
df = env_daily.copy()
# Group by 'Region' and 'YearMonth', then calculate the mean for relevant columns
monthly_env_mean = df.groupby(['Region', 'YearMonth'])[
    ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']
].mean().reset_index()

# Group by 'Region' and 'YearMonth', then calculate the sum for relevant columns
monthly_env_sum = df.groupby(['Region', 'YearMonth'])[
    ['precipitation','potential_evaporation_sum', 'total_evaporation_sum']
].sum().reset_index()

# Merge the mean and sum dataframes
monthly_env_region = pd.merge(monthly_env_mean, monthly_env_sum, on=['Region', 'YearMonth'], how='inner')

# Anomaly calculation for monthly_env_region
# Add 'Month' column to monthly_env_data before anomaly calculation
monthly_env_region['Month'] = monthly_env_region['YearMonth'].dt.month 
for col in temperature_cols + evap_precip_cols:
    # Calculate 30-year monthly average for each Region
    # Ensure 'Month' is available in monthly_env_data before this step
    monthly_30yr_avg = monthly_env_region[(monthly_env_region['YearMonth'].dt.year >= start_year_30yr_avg) &
                                         (monthly_env_region['YearMonth'].dt.year <= end_year_30yr_avg)].groupby(['Region', 'Month'])[col].mean().reset_index()
    monthly_30yr_avg.rename(columns={col: f'{col}_30yr_avg'}, inplace=True)

    # Merge 30-year average back to the main DataFrame
    monthly_env_region = monthly_env_region.merge(monthly_30yr_avg, on=['Region', 'Month'], how='left')

    # Calculate anomaly
    monthly_env_region[f'{col}_ANOM'] = monthly_env_region[col] - monthly_env_region[f'{col}_30yr_avg']

    # Drop the temporary 30yr_avg column
    monthly_env_region.drop(columns=[f'{col}_30yr_avg'], inplace=True)
# Drop 'Month' after all calculations
monthly_env_region.drop(columns=['Month'], inplace=True)

# Define the list of columns to aggregate
area_columns = [
    'True_Area_Class_70', 'True_Area_Class_60', 'True_Area_Class_50',
    'True_Area_Class_40', 'True_Area_Class_95', 'True_Area_Class_30',
    'True_Area_Class_20', 'True_Area_Class_10', 'True_Area_Class_90',
    'True_Area_Class_80', 'Class_sum'
]

# We sum the area columns to get total area per region for each class
lulc_agg_by_region = lulc_200_full[all_lulc_columns].groupby('Region')[area_columns].sum().reset_index()

# Since LULC data is static (not time-series in this context), it's merged only on 'Region'
monthly_region = pd.merge(
    monthly_env_region,
    lulc_agg_by_region,
    on='Region',
    how='inner' # Use 'inner' to ensure only matching regions are kept
)

true_area_class_columns = [
    'True_Area_Class_70', 'True_Area_Class_60', 'True_Area_Class_50',
    'True_Area_Class_40', 'True_Area_Class_95', 'True_Area_Class_30',
    'True_Area_Class_20', 'True_Area_Class_10', 'True_Area_Class_90',
    'True_Area_Class_80']
# Create new percentage columns
for col in true_area_class_columns:
    # Extract the class number from the column name (e.g., '70' from 'True_Area_Class_70')
    class_num = col.replace('True_Area_Class_', '')
    new_col_name = f'Class_{class_num}'
    # Calculate the percentage. Handle potential division by zero by checking if 'Class_sum' is 0
    # If Class_sum is 0, the percentage for that row will be 0.
    monthly_region[new_col_name] = monthly_region.apply(
        lambda row: row[col] / row['Class_sum'] if row['Class_sum'] != 0 else 0,
        axis=1
    )

# Recompute evaporative_stress_index and aridity_index
monthly_region['evaporative_stress_index'] = (
    monthly_region['total_evaporation_sum'] / monthly_region['potential_evaporation_sum'])
monthly_region['aridity_index'] = (
    monthly_region['precipitation'] / monthly_region['potential_evaporation_sum'])
monthly_region.to_csv(IN_DIR / 'monthly_env_region_1993_2023.csv', index=False)

# Filter the dataframe
filtered_monthly_region = monthly_region[(monthly_region['YearMonth'].dt.year >= 2010) & (monthly_region['YearMonth'].dt.year <= 2023)].reset_index(drop=True)
filtered_monthly_region.to_csv(IN_DIR / 'monthly_env_region_2010_2023.csv', index=False)


'''National level monthly statistics. +LULC'''

# Reset df variable to env_daily
df = env_daily.copy()
monthly_env_mean = df.groupby(['YearMonth'])[
    ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']
].mean().reset_index()
# Group by 'Region' and 'YearMonth', then calculate the sum for relevant columns
monthly_env_sum = df.groupby(['YearMonth'])[
    ['precipitation','potential_evaporation_sum', 'total_evaporation_sum']
].sum().reset_index()
# Merge the mean and sum dataframes
monthly_env_data = pd.merge(monthly_env_mean, monthly_env_sum, on=['YearMonth'], how='inner')


# Sum true area for each class
true_area_totals = lulc_200_full[true_area_cols].sum()
# Total area across all classes
total_area = true_area_totals.sum()
# Build proportion columns named Class_{i}
class_proportions = {
    f"Class_{col.split('_')[-1]}": (true_area_totals[col] / total_area).round(6)
    for col in true_area_cols
}
# Combine all into one row
summary_row = {
    **class_proportions,
    **true_area_totals.to_dict(),
    "Class_sum": total_area
}
# Create DataFrame
lulc_200_national = pd.DataFrame([summary_row])


# Columns to add from enso
enso_columns_to_add = ['YearMonth', 'NINO1+2', 'ANOM1+2', 'NINO3', 'ANOM3', 'NINO4', 'ANOM4', 'NINO3.4', 'ANOM3.4']
# Merge enso dataframes
monthly_national = monthly_env_data.merge(enso_sst[enso_columns_to_add], on='YearMonth', how='left')
# Merge DMI
monthly_national = monthly_national.merge(dmi_long[['YearMonth', 'DMI']], on='YearMonth', how='left')
# Merge DMI East
monthly_national = monthly_national.merge(dmi_east_long[['YearMonth', 'DMI_East']], on='YearMonth', how='left')

# Anomaly calculation for monthly_env_national
# Add 'Month' column to monthly_env_data before anomaly calculation
monthly_national['Month'] = monthly_national['YearMonth'].dt.month # <--- Add this line here

for col in temperature_cols + evap_precip_cols:
    # Calculate 30-year monthly average for national level
    # Ensure 'Month' is available in monthly_env_data before this step
    monthly_30yr_avg = monthly_national[(monthly_national['YearMonth'].dt.year >= start_year_30yr_avg) &
                                         (monthly_national['YearMonth'].dt.year <= end_year_30yr_avg)].groupby(['Month'])[col].mean().reset_index()
    monthly_30yr_avg.rename(columns={col: f'{col}_30yr_avg'}, inplace=True)

    # Merge 30-year average back to the main DataFrame
    monthly_national = monthly_national.merge(monthly_30yr_avg, on=['Month'], how='left')

    # Calculate anomaly
    monthly_national[f'{col}_ANOM'] = monthly_national[col] - monthly_national[f'{col}_30yr_avg']

    # Drop the temporary 30yr_avg column
    monthly_national.drop(columns=[f'{col}_30yr_avg'], inplace=True)
monthly_national.drop(columns=['Month'], inplace=True) # Drop 'Month' after all calculations

# Recompute evaporative_stress_index and aridity_index
monthly_national['evaporative_stress_index'] = (
    monthly_national['total_evaporation_sum'] / monthly_national['potential_evaporation_sum'])
monthly_national['aridity_index'] = (
    monthly_national['precipitation'] / monthly_national['potential_evaporation_sum'])

# Repeat summary_df to match the number of rows in merged_df
lulc_replicated = pd.concat([lulc_200_national] * len(monthly_national), ignore_index=True)
# Concatenate along columns
monthly_national = pd.concat([monthly_national.reset_index(drop=True), lulc_replicated], axis=1)

monthly_national.to_csv(IN_DIR / 'monthly_env_national_1993_2023.csv', index=False)

# Filter the dataframe
filtered_national = monthly_national[(monthly_national['YearMonth'].dt.year >= 2010) & (monthly_national['YearMonth'].dt.year <= 2023)].reset_index(drop=True)
filtered_national.to_csv(IN_DIR / 'monthly_env_national_2010_2023.csv', index=False)