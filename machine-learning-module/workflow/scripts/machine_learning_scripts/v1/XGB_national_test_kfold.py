import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path
import sys
import re
import shap
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cudf
import cupy as cp
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay

# Define project root
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" # Changed to REPORTS_ROOT for consistency
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data - region reassignments are no longer relevant for national analysis but kept for consistency with source data
df = pd.read_csv(PROCESSED_DIR / "monthly_dengue_env_id.csv")
# --- REGION REASSIGNMENT (kept for data loading consistency, but 'Region' column won't be used for splitting) ---
df['Region'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
print("--- DataFrame loaded ---")
print(f"Total entries: {len(df)}")
print("-" * 50)

# Read hyperparameters for the national model
# Expects a CSV with a 'Region' column where one row is 'National'
hyperparams_filename = "xgb_national_kfold_hyperparameters_lulc_0727.csv" # Original filename
hyperparams_filepath = TABLES_DIR / hyperparams_filename
hyperparams_df = pd.read_csv(hyperparams_filepath)

# Determine training procedure, version stamp, and LULC usage from filename
train_procedure = ""
version_stamp = "" # Default
use_lulc = False

if "walk" in hyperparams_filename:
    train_procedure = "walk"
elif "kfold" in hyperparams_filename:
    train_procedure = "kfold"


match_version = re.search(r'(\d{4})', hyperparams_filename)
if match_version:
    version_stamp = match_version.group(1)

if "_lulc" in hyperparams_filename:
    use_lulc = True

print(f"Hyperparameters loaded from: {hyperparams_filename}")
print(f"Detected Training Procedure: {train_procedure.upper()} Validation")
print(f"Detected Version Stamp: {version_stamp}")
print(f"Uses Land Use Features: {use_lulc}")
print("-" * 50)
# Read hyperparameters for the national model

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

climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']
target = 'Incidence_Rate'

# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
features = []
# Add base features if present
for var in env_vars + climate_vars: # Always include env and climate base vars
    if var in df.columns:
        features.append(var)

if use_lulc: # Only add land use vars if _lulc is in the filename
    for var in land_use_vars:
        if var in df.columns:
            features.append(var)

# Add lag features if present
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                features.append(lagged_var)

# Final feature list excluding metadata and target
actual_feature_columns = [
    col for col in features
    if col not in ['YearMonth', 'ID_2', 'Year', target]
]

# Print the final list of features that will be used
print("\n--- Final list of features for the model ---")
print(actual_feature_columns)
print(f"Total features: {len(actual_feature_columns)}")
print("-" * 50)

# Add Year column
df['Year'] = df['YearMonth'].astype(str).str[:4].astype(int)

all_shap_plots = []
all_pdp_plots = []  
national_summary_data = {}

print("\n=== Processing National Data ===")

# Filter and drop NaNs for the entire national dataset
df_processed = df.dropna(subset=actual_feature_columns + [target])

# Split data into training/validation (before 2023) and test (2023) sets
df_train_val = df_processed[df_processed['Year'] < 2023]
df_test = df_processed[df_processed['Year'] == 2023]

# Concatenate for full dataset analysis
df_full = pd.concat([df_train_val, df_test], ignore_index=True)

# Extract hyperparameters for the national model
params = hyperparams_df.iloc[0]  # Assuming the first row contains national hyperparameters
national_hyperparams = {
    'gamma': params['gamma'],
    'n_estimators': int(params['n_estimators']),
    'max_depth': int(params['max_depth']),
    'reg_alpha': params['reg_alpha'],
    'subsample': params['subsample'],
    'reg_lambda': params['reg_lambda'],
    'learning_rate': params['learning_rate'],
    'colsample_bytree': params['colsample_bytree'],
    'min_child_weight': int(params['min_child_weight'])
}

# Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
X_train = cudf.DataFrame(df_train_val[actual_feature_columns])
y_train = cudf.DataFrame(df_train_val[[target]])
X_test = cudf.DataFrame(df_test[actual_feature_columns])
y_test = cudf.DataFrame(df_test[[target]])
X_full = cudf.DataFrame(df_full[actual_feature_columns])
y_full = cudf.DataFrame(df_full[[target]])

# Select a subset of X_train for SHAP background sampling
background_sample = df_train_val[actual_feature_columns].sample(n=100, random_state=42)

# Initialize and train the XGBoost Regressor model
model = xgboost.XGBRegressor(
    objective='reg:squarederror', # Objective for regression tasks
    tree_method='hist',           # Use histogram-based tree method for faster training
    device='cuda',                # Specify GPU device for training
    random_state=64,              # For reproducibility
    **national_hyperparams        # Unpack national hyperparameters
)
model.fit(X_train, y_train)

# Initialize dictionary to store metrics for the national model
current_national_metrics = {}

for label, X_gpu, y_gpu in [
    ("Train/Val", X_train, y_train),
    ("Test", X_test, y_test),
    ("Full", X_full, y_full)
]:
    preds = model.predict(X_gpu)
    preds = np.maximum(0, cp.asnumpy(preds))  # CuPy -> NumPy
    y_true = y_gpu.to_numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    current_national_metrics[f'{label} RMSE'] = rmse
    current_national_metrics[f'{label} MAE'] = mae
    current_national_metrics[f'{label} R2'] = r2

    # Prepare values for SHAP tree explainer -> 
    # takes strictly numpy arrays or pandas DataFrames
    X_pd = X_gpu.to_pandas()
    explainer = shap.TreeExplainer(model, data=background_sample, feature_perturbation="auto")
    shap_vals = explainer.shap_values(X_pd.values)
    shap_vals_np = cp.asnumpy(shap_vals)   # For plotting

    all_shap_plots.append((shap_vals_np, X_pd, f"National - {label}", rmse, mae, r2))

    if label == "Test":
        mean_abs_shap = np.abs(shap_vals_np).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=X_pd.columns)
        top_5_features = feature_importance.nlargest(5).index.tolist()
        print(f"Top 5 predictors for National (Test Set): {top_5_features}")

        for feature in top_5_features:
            all_pdp_plots.append((model, X_pd, feature, "National"))

# Store all calculated metrics and Incidence Rate statistics for the national model
national_summary_data['National'] = current_national_metrics
national_summary_data['National']['IR Min'] = df_processed[target].min()
national_summary_data['National']['IR Max'] = df_processed[target].max()
national_summary_data['National']['IR 25th Quantile'] = df_processed[target].quantile(0.25)
national_summary_data['National']['IR 50th Quantile'] = df_processed[target].quantile(0.50)
national_summary_data['National']['IR 75th Quantile'] = df_processed[target].quantile(0.75)

# Generate SHAP plots and save them to a PDF file
lulc_suffix = "lulc" if use_lulc else ""
pdf_output_filename = f"xgb_national_{train_procedure}_plots_{lulc_suffix}_{version_stamp}.pdf"
pdf_path = FIGURES_DIR / pdf_output_filename

with PdfPages(pdf_path) as pdf:
    # SHAP summary plots
    for shap_vals, X_pd, title, rmse, mae, r2 in all_shap_plots:
        fig, ax = plt.subplots(figsize=(10, 8 + len(actual_feature_columns) * 0.25))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_pd, show=False)
        ax.set_title(f"{title} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    # PDP plots using sklearn
    for model, X_pd, feature, title in all_pdp_plots:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X_pd,
                [feature],
                ax=ax,
                kind='average',
                grid_resolution=100
            )
            ax.set_title(f'{title} - Sklearn PDP - {feature}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Failed sklearn PDP for {title} - {feature}: {e}")


print(f"National SHAP and PDP plots saved to '{pdf_path}'")

# --- Generate National Summary Table ---
print("\n--- National Summary Table ---")

# Convert the collected summary data into a pandas DataFrame
summary_df = pd.DataFrame.from_dict(national_summary_data, orient='index').T

# Save the DataFrame to a CSV file
csv_output_filename = f"xgb_national_{train_procedure}_test_{version_stamp}{lulc_suffix}.csv"
csv_filename = TABLES_DIR / csv_output_filename
summary_df.to_csv(csv_filename, index=True, float_format="%.2f")

print(f"National summary table saved to '{csv_filename}'")
print("-" * 50)