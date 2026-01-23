import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import xgboost
from xgboost import XGBClassifier # Changed to XGBClassifier
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # New classification metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import cudf
import cupy as cp
import optuna
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gc
import psutil
import pynvml


# Define project root based on notebook location (assuming this part is correct for your setup)
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve() # fallback

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "xgboost"
SQLITE_DIR = PROJECT_ROOT / "sqlite"
TABLES_DIR = REPORTS_DIR / "tables"

# Create Map Visualisation of Regional Accuracies
import geopandas as gpd
# Read in Indonesia shapefile
in_shp = EXTERNAL_DIR / "in_shp_new" / "in_shp_new"
# Read in Indonesia shapefile
in_shp = gpd.read_file(in_shp)

print(in_shp.head())

# Create averaged statistics per ID to visualise


# Load the data once outside the train function for efficiency
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" / "monthly_dengue_env_id_class_log.csv")

df['Risk_Category'] = df['Risk_Category'].replace({
    'Zero': 0,
    'Low': 1,
    'High': 2}).infer_objects(copy=False)
df['Risk_Category'] = df['Risk_Category'].astype('int32')
num_classes = df['Risk_Category'].nunique()
print("-" * 50)
# Create a list of regions to iterate over
regions_to_model = df['Region_Group'].unique()


df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime
df['Incidence_Rate_lag1'] = df.groupby('ID_2')['Incidence_Rate'].shift(1)

# Define variable categories
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum',
    'evaporative_stress_index', 'aridity_index',
    'temperature_2m_ANOM', 'temperature_2m_min_ANOM', 'temperature_2m_max_ANOM',
    'potential_evaporation_sum_ANOM', 'total_evaporation_sum_ANOM', 'precipitation_ANOM'
]

land_use_vars = [
    'Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95',
    'Class_30', 'Class_20', 'Class_10', 'Class_90', 'Class_80'
]

epidemic_vars = ['Incidence_Rate_lag1']

climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']

# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
USE_LANDUSE_FEATURES = True  # Set to False to exclude land use features
variable_columns = []
for var in env_vars + climate_vars + epidemic_vars:
    if var in df.columns:
        variable_columns.append(var)
if USE_LANDUSE_FEATURES:
    for var in land_use_vars:
        if var in df.columns:
            variable_columns.append(var)
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                variable_columns.append(lagged_var)


# Select relevant columns (metadata, variables, target)
target = 'Risk_Category'
metadata_columns = ['YearMonth', 'ID_2', 'Region_Group','Incidence_Rate']
# Final feature list excluding metadata and target
variable_columns = [
    col for col in variable_columns
    if col not in [metadata_columns, target]
]

print("Starting mapping with the following columns:")

print("--- Target Column ---")
print([target])
print("-" * 50)
print("--- Metadata Columns ---")
print(metadata_columns)
print("-" * 50)
print("--- Variable Columns ---")
print(variable_columns)

# Identify all columns in the DataFrame
all_columns = df.columns.tolist()

# Determine columns that are not variable columns (these will be kept with 'first')
other_columns = [col for col in all_columns if col not in variable_columns and col not in ['ID_2', 'YearMonth']]

# Define aggregation: mean for variable columns, first for other columns
agg_dict = {col: 'mean' for col in variable_columns}
agg_dict.update({col: 'first' for col in other_columns})

# Group by ID_2 and YearMonth and aggregate
new_df = df.groupby(['ID_2', 'YearMonth'], as_index=False).agg(agg_dict)

print("New DataFrame with averaged variables and preserved metadata:")
print(new_df.head())



# for model in MODELS_DIR:
#     model = xgboost.Booster()
#     model.load_model(str(model_path))