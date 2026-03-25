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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import r2_score
import cudf
import cupy as cp
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
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True) # Ensure TABLES_DIR exists

# Load data and region-specific hyperparameters
df = pd.read_csv(PROCESSED_DIR / "monthly_dengue_env_id.csv")
# --- REGION REASSIGNMENT ---
df['Region'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
print("--- DataFrame after Region modification ---")
print(df['Region'].value_counts())
print("-" * 50)

# Read hyperparameters for the national model
# Expects a CSV with a 'Region' column where one row is 'National'
hyperparams_filename = "xgb_region_kfold_hyperparameters_lulc__0726.csv" # Original filename
hyperparams_filepath = TABLES_DIR / hyperparams_filename
hyperparams_df = pd.read_csv(hyperparams_filepath, index_col='Region')

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

# Store all SHAP info for plotting
all_shap_plots = []
all_pdp_plots = []
region_summary_data = {} # Dictionary to store metrics and IR stats for each region

# Iterate through each region to train models, calculate metrics, and generate SHAP plots
for region_name, params in hyperparams_df.iterrows():
    print(f"\n=== Processing Region: {region_name} ===")
    # Filter data for the current region
    df_region = df[df['Region'] == region_name].copy()

    # Split data into training/validation (before 2023) and test (2023) sets
    df_train_val = df_region[df_region['Year'] < 2023].dropna(subset=actual_feature_columns + [target])
    df_test = df_region[df_region['Year'] == 2023].dropna(subset=actual_feature_columns + [target])

    # Concatenate for full dataset analysis
    df_full = pd.concat([df_train_val, df_test], ignore_index=True)

    # Skip region if insufficient data for training or testing
    if df_train_val.empty or df_test.empty:
        print(f"Skipping {region_name}: insufficient data for training or testing.")
        continue

    # Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
    X_train = cudf.DataFrame(df_train_val[actual_feature_columns])
    y_train = cudf.DataFrame(df_train_val[[target]])
    X_test = cudf.DataFrame(df_test[actual_feature_columns])
    y_test = cudf.DataFrame(df_test[[target]])
    X_full = cudf.DataFrame(df_full[actual_feature_columns])
    y_full = cudf.DataFrame(df_full[[target]])

    # Extract hyperparameters for the current region
    hyperparams = {
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

    # Initialize and train the XGBoost Regressor model
    model = xgboost.XGBRegressor(
        objective='reg:squarederror', # Objective for regression tasks
        tree_method='hist',           # Use histogram-based tree method for faster training
        device='cuda',                # Specify GPU device for training
        random_state=64,              # For reproducibility
        **hyperparams                 # Unpack region-specific hyperparameters
    )
    model.fit(X_train, y_train)

    # Initialize dictionary to store metrics for the current region
    current_region_metrics = {}

    # Evaluate model performance and generate SHAP values for Train/Val, Test, and Full datasets
    for label, X_gpu, y_gpu in [
        ("Train/Val", X_train, y_train),
        ("Test", X_test, y_test),
        ("Full", X_full, y_full)
    ]:
        try:
            # Make predictions and ensure they are non-negative
            preds = model.predict(X_gpu)
            preds = np.maximum(0, cp.asnumpy(preds)) # Convert CuPy array to NumPy and ensure non-negative
            y_true = y_gpu.to_numpy().flatten() # Convert cuDF Series to NumPy array

            # Calculate RMSE, MAE, and R-squared
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds) # Calculate R-squared

            # Store metrics for the current dataset split
            current_region_metrics[f'{label} RMSE'] = rmse
            current_region_metrics[f'{label} MAE'] = mae
            current_region_metrics[f'{label} R2'] = r2 # Store R-squared

            # Convert cuDF DataFrame to pandas DataFrame for SHAP
            X_pd = X_gpu.to_pandas()
            # Initialize SHAP explainer and calculate SHAP values
            # Using feature_perturbation="interventional" for TreeExplainer
            explainer = shap.TreeExplainer(model, X_pd, feature_perturbation="interventional")
            shap_vals = explainer(X_pd)

            # Store SHAP values and related info for later plotting
            all_shap_plots.append((shap_vals.values, X_pd, f"{region_name} - {label}", rmse, mae, r2))

            # --- Identify Top 5 Predictors for Test Set and prepare for PDP ---
            if label == "Test":
                # Calculate mean absolute SHAP values for the test set
                mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
                # Create a pandas Series for easy sorting
                feature_importance = pd.Series(mean_abs_shap, index=X_pd.columns)
                # Get the top 5 features
                top_5_features = feature_importance.nlargest(5).index.tolist()
                print(f"Top 5 predictors for {region_name} (Test Set): {top_5_features}")

                # Store info for PDP plots
                for feature in top_5_features:
                    all_pdp_plots.append((model, X_pd, feature, region_name))

        except Exception as e:
            print(f"Failed SHAP or metric calculation for {region_name} - {label}: {e}")

    # Store all calculated metrics and Incidence Rate statistics for the current region
    region_summary_data[region_name] = current_region_metrics
    region_summary_data[region_name]['IR Min'] = df_region[target].min()
    region_summary_data[region_name]['IR Max'] = df_region[target].max()
    region_summary_data[region_name]['IR 25th Quantile'] = df_region[target].quantile(0.25)
    region_summary_data[region_name]['IR 50th Quantile'] = df_region[target].quantile(0.50)
    region_summary_data[region_name]['IR 75th Quantile'] = df_region[target].quantile(0.75)

# --- Dynamically create filenames for PDF and CSV ---
lulc_suffix = "_lulc" if use_lulc else ""
pdf_filename = f"xgb_region_{train_procedure}_plots{lulc_suffix}_{version_stamp}.pdf"
csv_filename = f"xgb_region_{train_procedure}_test{lulc_suffix}_{version_stamp}.csv"

pdf_path = FIGURES_DIR / pdf_filename
csv_path = TABLES_DIR / csv_filename

# Generate SHAP and PDP plots and save all to a single PDF
print(f"\nSaving plots to: {pdf_path}")
with PdfPages(pdf_path) as pdf:
    # SHAP summary plots
    for shap_vals, X_df, title, rmse, mae, r2 in all_shap_plots:
        fig, ax = plt.subplots(figsize=(10, 8 + len(actual_feature_columns) * 0.25))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_df, show=False)
        ax.set_title(f"{title} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # PDP plots for top 5 predictors (Test set) of each region
    for model, X_df, feature, region_name in all_pdp_plots:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=X_df,
                features=[feature],
                ax=ax
            )
            ax.set_title(f'{region_name} - PDP - {feature}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Failed scikit-learn PDP for {region_name} - {feature}: {e}")


# --- Generate Region-wise Summary Table ---
print("\n--- Region-wise Summary Table ---")

# Convert the collected summary data into a pandas DataFrame
summary_df = pd.DataFrame.from_dict(region_summary_data, orient='index').T

# Save the DataFrame to a CSV file
print(f"Region-wise summary table saved to '{csv_path}'")
summary_df.to_csv(csv_path, index=True, float_format="%.2f")

print("-" * 50)