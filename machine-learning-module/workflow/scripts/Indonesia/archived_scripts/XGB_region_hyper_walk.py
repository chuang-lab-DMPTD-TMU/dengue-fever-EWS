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
from matplotlib.backends.backend_pdf import PdfPages
import shap
import wandb
import xgboost
from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import cudf
import cupy as cp
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold # Import KFold

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"


# --- CONFIGURATION ---
# Set this to your desired config string, e.g. "KF0728"
config_input = "WF0729"

# Set WANDB settings to online/offline
os.environ["WANDB_MODE"] = "online"

# --- Run sweep for the entire nation ---
n_runs_for_region_sweep = 50 # Number of runs for the regional sweep
all_best_hypers = [] # This will store the single best result for each region


# --- Normalize and validate input ---
config_input = config_input.strip().lower()
if len(config_input) != 6:
    raise ValueError("Invalid config format. Use 6 characters like 'kt0726' or 'wf2031'.")

# Parse components
validation_flag = config_input[0]
landuse_flag = config_input[1]
version_digits = config_input[2:]

# Determine validation strategy
if validation_flag == 'k':
    validation_strategy = 'kfold'
elif validation_flag == 'w':
    validation_strategy = 'walk'
else:
    raise ValueError("Invalid validation flag. Use 'K' for kfold or 'W' for walk_forward.")

# Determine land use flag
if landuse_flag == 't':
    USE_LANDUSE_FEATURES = True
    landuse_suffix = "lulc"
elif landuse_flag == 'f':
    USE_LANDUSE_FEATURES = False
    landuse_suffix = ""
else:
    raise ValueError("Land use flag must be 'T' or 'F'.")

# Validate version digits
if not version_digits.isdigit():
    raise ValueError("Version must be 4 digits.")
VERSION_STAMP = version_digits
version_suffix = f"{VERSION_STAMP}"


# --- Logging ---
print(f"Using validation strategy: {validation_strategy} in version {VERSION_STAMP}")
print(f"Land Use Features Included: {USE_LANDUSE_FEATURES}")
print(f"Validation flag: {validation_flag}, Landuse flag: {landuse_flag}, Version digits: {version_digits}")


# Define the shared hyperparameters with the desired sweep ranges
shared_hyperparameters = {
    'n_estimators': {
        'distribution': 'int_uniform',
        'min': 50,
        'max': 7000
    },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 5
    },
    'subsample': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.5
    },
    'colsample_bytree': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.5
    },
    'min_child_weight': {'distribution': 'int_uniform', 'min': 10, 'max': 50},
    'gamma': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 10},
    'reg_alpha': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 10},
    'reg_lambda': {'distribution': 'log_uniform_values', 'min': 1, 'max': 100}
}
# Define the base sweep configuration
base_sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'global_overall_rmse',
        'goal': 'minimize'
    },
    'parameters': {
        **shared_hyperparameters, # Unpack shared hyperparameters
        'validation_strategy': {'value': validation_strategy} # Pass the strategy to the train function config
    }
}

# Add strategy-specific parameters
if validation_strategy == 'kfold':
    base_sweep_config['parameters']['n_splits'] = {'value': 5}
    validation_suffix = "kfold"
elif validation_strategy == 'walk':
    base_sweep_config['parameters']['initial_train_months'] = {'value': 60}
    base_sweep_config['parameters']['test_window'] = {'value': 6}
    validation_suffix = "walk"
else:
    raise ValueError("Invalid validation_strategy. Choose 'kfold' or 'walk'.")

sweep_config = base_sweep_config
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
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"

# Load the data once outside the train function for efficiency in a sweep
df = pd.read_csv(PROCESSED_DIR /"monthly_dengue_env_id.csv")

# --- REGION REASSIGNMENT ---
df['Region_Group'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
unique_regions = df['Region_Group'].unique()
print("--- DataFrame after Region_Group creation ---")
print(df['Region_Group'].value_counts())
print("-" * 50)

df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime


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
# Add base environmental and climate features
for var in env_vars + climate_vars:
    if var in df.columns:
        features.append(var)

# NEW: Conditionally add land use features
if USE_LANDUSE_FEATURES:
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


# --- Refactored training function for a single run (K-fold and Walk-Forward version) ---
def train(df_input, feature_columns, target_column):
    """
    Trains an XGBoost Regressor model using either K-fold cross-validation or
    walk-forward validation and logs metrics and artifacts to Weights & Biases.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing all features and target.
        feature_columns (list): A list of column names to be used as features.
        target_column (str): The name of the target variable column.
    """
    with wandb.init() as run:
        config = run.config

        print(f"\n--- Starting run with config: {config} ---")
        current_strategy = config.validation_strategy # Get strategy from config

        # --- Data Preprocessing ---
        df_local = df_input.copy()
        df_local['YearMonth'] = pd.to_datetime(df_local['YearMonth'])
        # Crucial: Sort by YearMonth first, then ID_2 for consistent time-based slicing
        df_local = df_local.sort_values(by=['YearMonth', 'ID_2']).reset_index(drop=True)

        actual_feature_columns = [col for col in feature_columns if col not in ['YearMonth', 'ID_2', target_column]]

        # Drop NaNs based on the columns actually used for model training
        df_processed = df_local.dropna(subset=actual_feature_columns + [target_column])

        if df_processed.empty:
            print("DataFrame is empty after dropping NaNs. Cannot proceed with training.")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf'), "error": "Empty DataFrame after NaN drop"})
            return

        # Convert to cuDF Dataframes for GPU acceleration
        X_gpu = cudf.DataFrame(df_processed[actual_feature_columns])
        y_gpu = cudf.DataFrame(df_processed[[target_column]])

        all_preds = []
        all_true = []
        fold_metrics = []
        model = None

        if current_strategy == 'kfold':
            # --- K-Fold Validation Setup ---
            n_splits = config.n_splits # Number of folds for K-fold CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=64) # Shuffle for KFold

            print(f"Performing {n_splits}-fold cross-validation...")

            # Iterate through K-fold splits
            for i, (train_index, test_index) in enumerate(kf.split(X_gpu.to_pandas())):
                train_index_gpu = cp.array(train_index)
                test_index_gpu = cp.array(test_index)

                X_train_fold_gpu = X_gpu.iloc[train_index_gpu]
                y_train_fold_gpu = y_gpu.iloc[train_index_gpu]
                X_test_fold_gpu = X_gpu.iloc[test_index_gpu]
                y_test_fold_gpu = y_gpu.iloc[test_index_gpu]

                if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
                    print(f"      Warning: Fold {i+1} has empty train or test split. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBRegressor(
                    objective='reg:squarederror', tree_method="hist", device="cuda",
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    max_depth=config.max_depth,
                    subsample=config.subsample,
                    colsample_bytree=config.colsample_bytree,
                    random_state=64,
                    n_jobs=-1,
                    min_child_weight=config.min_child_weight,
                    gamma=config.gamma,
                    reg_alpha=config.reg_alpha,
                    reg_lambda=config.reg_lambda
                )

                try:
                    model.fit(X_train_fold_gpu, y_train_fold_gpu)
                except Exception as e:
                    print(f"      Error during model fit in Fold {i+1}: {e}")
                    run.log({f"fold_{i+1}_error": str(e)}, step=i+1)
                    continue

                # --- Prediction ---
                predictions_gpu = model.predict(X_test_fold_gpu)
                predictions = cp.asnumpy(predictions_gpu)
                predictions = np.maximum(0, predictions)

                # --- Evaluate and Log Fold Metrics ---
                y_test_fold_np = y_test_fold_gpu.to_numpy().flatten()
                fold_rmse = np.sqrt(mean_squared_error(y_test_fold_np, predictions))
                fold_mae = mean_absolute_error(y_test_fold_np, predictions)
                fold_metrics.append({'rmse': fold_rmse, 'mae': fold_mae})
                print(f"      Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

                run.log({
                    f"fold_{i+1}_rmse": fold_rmse,
                    f"fold_{i+1}_mae": fold_mae,
                    "current_fold": i + 1,
                    "fold_rmse_history": fold_rmse,
                    "fold_mae_history": fold_mae
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_test_fold_np)

        elif current_strategy == 'walk':
            # --- Walk-Forward Validation Setup ---
            initial_train_months = config.initial_train_months
            test_window = config.test_window

            unique_time_periods = df_processed['YearMonth'].unique()
            n_time_periods = len(unique_time_periods)

            n_splits = (n_time_periods - initial_train_months) // test_window

            if n_splits <= 0:
                print("Not enough unique time periods for specified initial_train_months and test_window.")
                print(f"Total time periods: {n_time_periods}, Initial train: {initial_train_months}, Test window: {test_window}")
                run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf'), "error": "Insufficient data for splits"})
                return

            print(f"Performing {n_splits} global walk-forward splits based on time periods...")

            for i in range(n_splits):
                train_end_period_idx = initial_train_months + i * test_window
                test_start_period_idx = train_end_period_idx
                test_end_period_idx = test_start_period_idx + test_window

                if test_end_period_idx > n_time_periods:
                    print(f"      Warning: Test end period index {test_end_period_idx} exceeds total time periods {n_time_periods}. Ending walk-forward.")
                    break

                train_end_time = unique_time_periods[train_end_period_idx - 1]
                test_start_time = unique_time_periods[test_start_period_idx]
                test_end_time = unique_time_periods[test_end_period_idx - 1]

                train_df_period = df_processed[df_processed['YearMonth'] <= train_end_time]
                test_df_period = df_processed[
                    (df_processed['YearMonth'] >= test_start_time) &
                    (df_processed['YearMonth'] <= test_end_time)
                ]

                X_train_fold_gpu = cudf.DataFrame(train_df_period[actual_feature_columns])
                y_train_fold_gpu = cudf.DataFrame(train_df_period[[target_column]])
                X_test_fold_gpu = cudf.DataFrame(test_df_period[actual_feature_columns])
                y_test_fold_gpu = cudf.DataFrame(test_df_period[[target_column]])

                if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
                    print(f"      Warning: Fold {i+1} has empty train or test split after time-based slicing. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBRegressor(
                    objective='reg:squarederror', tree_method="hist", device="cuda",
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    max_depth=config.max_depth,
                    subsample=config.subsample,
                    colsample_bytree=config.colsample_bytree,
                    random_state=64,
                    n_jobs=-1,
                    min_child_weight=config.min_child_weight,
                    gamma=config.gamma,
                    reg_alpha=config.reg_alpha,
                    reg_lambda=config.reg_lambda
                )

                try:
                    model.fit(X_train_fold_gpu, y_train_fold_gpu)
                except Exception as e:
                    print(f"      Error during model fit in Fold {i+1}: {e}")
                    run.log({f"fold_{i+1}_error": str(e)}, step=i+1)
                    continue

                # --- Prediction ---
                predictions_gpu = model.predict(X_test_fold_gpu)
                predictions = cp.asnumpy(predictions_gpu)
                predictions = np.maximum(0, predictions)

                # --- Evaluate and Log Fold Metrics ---
                y_test_fold_np = y_test_fold_gpu.to_numpy().flatten()
                fold_rmse = np.sqrt(mean_squared_error(y_test_fold_np, predictions))
                fold_mae = mean_absolute_error(y_test_fold_np, predictions)
                fold_metrics.append({'rmse': fold_rmse, 'mae': fold_mae})
                print(f"      Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

                run.log({
                    f"fold_{i+1}_rmse": fold_rmse,
                    f"fold_{i+1}_mae": fold_mae,
                    "current_fold": i + 1,
                    "fold_rmse_history": fold_rmse,
                    "fold_mae_history": fold_mae
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_test_fold_np)
        else:
            raise ValueError("Invalid validation_strategy specified. Choose 'kfold' or 'walk_forward'.")

        # --- Calculate and Log Global Overall Metrics (AFTER the loop) ---
        if all_true:
            global_overall_rmse = np.sqrt(mean_squared_error(all_true, all_preds))
            global_overall_mae = mean_absolute_error(all_true, all_preds)

            run.log({
                "global_overall_rmse": global_overall_rmse,
                "global_overall_mae": global_overall_mae,
                "mean_fold_rmse": np.mean([f['rmse'] for f in fold_metrics]),
                "mean_fold_mae": np.mean([f['mae'] for f in fold_metrics]),
                "n_splits_completed": len(fold_metrics)
            })
            print(f"\n--- GLOBAL Overall {current_strategy.replace('_', '-').upper()} Evaluation: RMSE={global_overall_rmse:.2f}, MAE={global_overall_mae:.2f} ---")
        else:
            print(f"\n--- No successful predictions made during {current_strategy.replace('_', '-')}. Setting metrics to infinity. ---")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf')})

        # --- Model Artifact Saving ---
        if model is not None:
            try:
                model_filename = "xgboost_model.json"
                model.save_model(model_filename)

                artifact = wandb.Artifact(
                    name=f"dengue-xgboost-model_{run.id}",
                    type="model",
                    description=f"XGBoost model trained during sweep run ({current_strategy})",
                    metadata=dict(config)
                )
                artifact.add_file(model_filename)
                run.log_artifact(artifact)
                print(f"Model saved as artifact: {model_filename}")
                os.remove(model_filename)
                print(f"Removed local model file: {model_filename}")

            except Exception as e:
                print(f"Error saving or logging model artifact: {e}")
                run.log({"model_save_error": str(e)})
        else:
            print("No model object found to save as artifact for this run.")
            run.log({"model_save_status": "No model trained"})


# --- REGION SWEEP AND CSV AGGREGATION ---
for region in unique_regions:
    print(f"\n--- Starting sweep for Region: {region} with {validation_strategy.replace('_', '-').upper()} validation and data version '{VERSION_STAMP}' ---")
    print(f"Using Land Use Features: {USE_LANDUSE_FEATURES}")

    # Filter data for the current region
    df_region = df[df['Region_Group'] == region].copy()

    # Data for hyperparameter tuning (all data before 2023 for validation)
    df_train_val_region = df_region[df_region['YearMonth'].dt.year < 2023].copy()
    # Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
    df_test_region = df_region[df_region['YearMonth'].dt.year == 2023].copy()

    print(f"Shape of df_train_val for {region}: {df_train_val_region.shape}")
    print(f"Shape of df_final_test for {region}: {df_test_region.shape}")

    if df_train_val_region.empty:
        print(f"No training/validation data for region {region}. Skipping sweep for this region.")
        continue

    # Initialize the sweep with a project name specific to the region, strategy, and data version
    project_name = f"dengue-indonesia-xgb-{region.replace(' ', '-')}-{validation_suffix}-{landuse_suffix}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID for {region}: {sweep_id}")

    # Run the sweep agent for the current region
    wandb.agent(sweep_id, function=lambda: train(df_train_val_region.copy(), feature_columns=features, target_column=target), count=n_runs_for_region_sweep)
    print(f"Sweep for Region {region} completed.")

    # --- Retrieve and Log Best Hyperparameters for the Region ---
    try:
        api = wandb.Api()
        # Find the project for the current region
        current_project = api.project(project_name)
        # Get all runs from the current sweep, sorted by the metric (global_overall_rmse)
        runs = api.runs(f"{current_project.entity}/{project_name}", filters={"sweep": sweep_id})
        best_run = None
        min_rmse = float('inf')
        for run in runs:
            if "global_overall_rmse" in run.summary and run.summary["global_overall_rmse"] < min_rmse:
                min_rmse = run.summary["global_overall_rmse"]
                best_run = run
        if best_run:
            best_config = best_run.config
            best_metrics = best_run.summary
            # Prepare data for the table
            row = {
                'Region': region,
                'validation_flag': validation_flag,
                'landuse_flag': landuse_flag,
                'version_digits': version_digits
            }
            row.update(best_config)
            row['best_rmse'] = best_metrics.get('global_overall_rmse', np.nan)
            row['best_mae'] = best_metrics.get('global_overall_mae', np.nan)
            all_best_hypers.append(row)
            print(f"Best hyperparameters for {region}: {best_config}")
            print(f"Best RMSE for {region}: {best_metrics.get('global_overall_rmse', np.nan)}")
            print(f"Best MAE for {region}: {best_metrics.get('global_overall_mae', np.nan)}")
        else:
            print(f"No successful runs found for region {region} in sweep {sweep_id}.")
            row = {
                'Region': region,
                'validation_flag': validation_flag,
                'landuse_flag': landuse_flag,
                'version_digits': version_digits,
                'best_rmse': np.nan,
                'best_mae': np.nan
            }
            for param in sweep_config['parameters'].keys():
                row[param] = np.nan
            all_best_hypers.append(row)
    except Exception as e:
        print(f"Error retrieving best hyperparameters for {region}: {e}")
        row = {
            'Region': region,
            'validation_flag': validation_flag,
            'landuse_flag': landuse_flag,
            'version_digits': version_digits,
            'best_rmse': np.nan,
            'best_mae': np.nan
        }
        for param in sweep_config['parameters'].keys():
            row[param] = np.nan
        all_best_hypers.append(row)

# --- Final Aggregation and CSV Saving ---
if all_best_hypers:
    best_hypers_df = pd.DataFrame(all_best_hypers)
    # Compose filename with all flags for traceability
    best_hypers_csv_path = TABLES_DIR / f"xgb_region_{validation_suffix}_hyperparameters_{landuse_suffix}_{version_digits}.csv"
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best hyperparameters to {best_hypers_csv_path}")
else:
    print("\nNo best hyperparameters found to save or log.")



# --- FINAL TRAINING, TESTING, AND PLOTTING ---

# Use the dynamically generated best hyperparameters CSV from above
best_hypers_csv_path = TABLES_DIR / f"xgb_region_{validation_suffix}_hyperparameters_{landuse_suffix}_{version_digits}.csv"
hyperparams_df = pd.read_csv(best_hypers_csv_path, index_col='Region', dtype={'version_digits': str})

# Inherit flags from the CSV (assume all rows have the same flags)
first_row = hyperparams_df.iloc[0]
validation_flag = str(first_row['validation_flag'])
landuse_flag = str(first_row['landuse_flag'])
version_digits = str(first_row['version_digits'])
validation_strategy = 'kfold' if validation_flag == 'k' else 'walk'
USE_LANDUSE_FEATURES = True if landuse_flag == 't' else False
version_suffix = version_digits
landuse_suffix = "lulc" if USE_LANDUSE_FEATURES else ""
train_procedure = validation_strategy

print(f"Hyperparameters loaded from: {best_hypers_csv_path}")
print(f"Detected Training Procedure: {train_procedure.upper()} Validation")
print(f"Detected Version Stamp: {version_suffix}")
print(f"Uses Land Use Features: {USE_LANDUSE_FEATURES}")
print("-" * 50)

# Add Year column
if 'Year' not in df.columns:
    df['Year'] = df['YearMonth'].astype(str).str[:4].astype(int)

# Compile feature list
features = []
for var in env_vars + climate_vars:
    if var in df.columns:
        features.append(var)
if USE_LANDUSE_FEATURES:
    for var in land_use_vars:
        if var in df.columns:
            features.append(var)
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                features.append(lagged_var)
actual_feature_columns = [col for col in features if col not in ['YearMonth', 'ID_2', 'Year', target]]

# Store all SHAP info for plotting
all_shap_plots = []
all_pdp_plots = []
region_summary_data = {}

for region_name, params in hyperparams_df.iterrows():
    print(f"\n=== Processing Region: {region_name} ===")
    df_region = df[df['Region'] == region_name].copy()
    df_train_val = df_region[df_region['Year'] < 2023].dropna(subset=actual_feature_columns + [target])
    df_test = df_region[df_region['Year'] == 2023].dropna(subset=actual_feature_columns + [target])
    df_full = pd.concat([df_train_val, df_test], ignore_index=True)
    if df_train_val.empty or df_test.empty:
        print(f"Skipping {region_name}: insufficient data for training or testing.")
        continue
    X_train = cudf.DataFrame(df_train_val[actual_feature_columns])
    y_train = cudf.DataFrame(df_train_val[[target]])
    X_test = cudf.DataFrame(df_test[actual_feature_columns])
    y_test = cudf.DataFrame(df_test[[target]])
    X_full = cudf.DataFrame(df_full[actual_feature_columns])
    y_full = cudf.DataFrame(df_full[[target]])
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
    model = xgboost.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        random_state=64,
        **hyperparams
    )
    model.fit(X_train, y_train)
    current_region_metrics = {}
    for label, X_gpu, y_gpu in [
        ("Train/Val", X_train, y_train),
        ("Test", X_test, y_test),
        ("Full", X_full, y_full)
    ]:
        try:
            preds = model.predict(X_gpu)
            preds = np.maximum(0, cp.asnumpy(preds))
            y_true = y_gpu.to_numpy().flatten()
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            current_region_metrics[f'{label} RMSE'] = rmse
            current_region_metrics[f'{label} MAE'] = mae
            current_region_metrics[f'{label} R2'] = r2
            X_pd = X_gpu.to_pandas()
            explainer = shap.TreeExplainer(model, X_pd, feature_perturbation="interventional")
            shap_vals = explainer(X_pd)
            all_shap_plots.append((shap_vals.values, X_pd, f"{region_name} - {label}", rmse, mae, r2))
            if label == "Test":
                mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
                feature_importance = pd.Series(mean_abs_shap, index=X_pd.columns)
                top_5_features = feature_importance.nlargest(5).index.tolist()
                print(f"Top 5 predictors for {region_name} (Test Set): {top_5_features}")
                for feature in top_5_features:
                    all_pdp_plots.append((model, X_pd, feature, region_name))
        except Exception as e:
            print(f"Failed SHAP or metric calculation for {region_name} - {label}: {e}")
    region_summary_data[region_name] = current_region_metrics
    region_summary_data[region_name]['IR Min'] = df_region[target].min()
    region_summary_data[region_name]['IR Max'] = df_region[target].max()
    region_summary_data[region_name]['IR 25th Quantile'] = df_region[target].quantile(0.25)
    region_summary_data[region_name]['IR 50th Quantile'] = df_region[target].quantile(0.50)
    region_summary_data[region_name]['IR 75th Quantile'] = df_region[target].quantile(0.75)

lulc_suffix = "_lulc" if USE_LANDUSE_FEATURES else ""
pdf_filename = f"xgb_region_{train_procedure}_plots{lulc_suffix}_{version_suffix}.pdf"
csv_filename = f"xgb_region_{train_procedure}_test{lulc_suffix}_{version_suffix}.csv"
pdf_path = FIGURES_DIR / pdf_filename
csv_path = TABLES_DIR / csv_filename
print(f"\nSaving plots to: {pdf_path}")
with PdfPages(pdf_path) as pdf:
    for shap_vals, X_df, title, rmse, mae, r2 in all_shap_plots:
        fig, ax = plt.subplots(figsize=(10, 8 + len(actual_feature_columns) * 0.25))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_df, show=False)
        ax.set_title(f"{title} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
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
print("\n--- Region-wise Summary Table ---")
summary_df = pd.DataFrame.from_dict(region_summary_data, orient='index').T
print(f"Region-wise summary table saved to '{csv_path}'")
summary_df.to_csv(csv_path, index=True, float_format="%.2f")
print("-" * 50)