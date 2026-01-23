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
import wandb
import xgboost
from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cudf
import cupy as cp
from sklearn.model_selection import KFold # Import KFold

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KT0727"  

# Normalize and validate input
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
version_suffix = f"_{VERSION_STAMP}" 

# --- Logging ---
print(f"Using validation strategy: {validation_strategy} in version {VERSION_STAMP}")
print(f"Land Use Features Included: {USE_LANDUSE_FEATURES}")


# Define the shared hyperparameters with the desired sweep ranges
shared_hyperparameters = {
    'n_estimators': {
        'distribution': 'int_uniform',
        'min': 50,
        'max': 600
    },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 0.001,
        'max': 0.1
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 3,
        'max': 6
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
    'min_child_weight': {'distribution': 'int_uniform', 'min': 1, 'max': 30},
    'gamma': {'distribution': 'log_uniform_values', 'min': 1e-2, 'max': 15},
    'reg_alpha': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 15},
    'reg_lambda': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 15}
}

# Define the base sweep configuration
base_sweep_config = {
    'method': 'random', # CHANGED: From 'bayes' to 'random'
    'metric': {
        'name': 'global_overall_rmse',
        'goal': 'minimize'
    },
    'parameters': {
        **shared_hyperparameters, 
        'validation_strategy': {'value': validation_strategy} 
    }
}

# Add validation method-specific parameters
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

# --- REGION REASSIGNMENT (Keep this for consistency, but it won't be used for grouping) ---
df['Region_Group'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
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
                    objective='reg:squaredererror', tree_method="hist", device="cuda",
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


# Set WANDB settings
os.environ["WANDB_MODE"] = "online"

# --- Run sweep for the entire nation ---
# Keep n_runs_for_national_sweep the same or adjust based on how many random trials you want
n_runs_for_national_sweep = 100 
all_best_hypers = [] # This will store the single best result for the nation

print(f"\n--- Starting NATIONAL sweep with {validation_strategy.replace('_', '-').upper()} validation and data version '{VERSION_STAMP}' ---")

# Data for hyperparameter tuning (all data before 2023 for validation)
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy()
# Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
df_test_national = df[df['YearMonth'].dt.year == 2023].copy()


print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

if df_train_val_national.empty:
    print(f"No training/validation data for national sweep. Cannot proceed.")
else:
    # Initialize the sweep with a project name specific to the national scope, strategy, and data version
    project_name = f"dengue-indonesia-xgb-national-random-{validation_suffix}-{landuse_suffix}" # Project name updated
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID for National: {sweep_id}")

    # Run the sweep agent for the entire nation
    wandb.agent(sweep_id, function=lambda: train(df_train_val_national.copy(), feature_columns=features, target_column=target), count=n_runs_for_national_sweep)
    print(f"National sweep completed.")

    # --- Retrieve and Log Best Hyperparameters for the Nation ---
    try:
        api = wandb.Api()
        # Find the project for the national run
        current_project = api.project(project_name)
        
        # Get all runs from the current sweep, sorted by the metric (global_overall_rmse)
        runs = api.runs(f"{current_project.entity}/{project_name}",
                         filters={"sweep": sweep_id})
        
        best_run = None
        min_rmse = float('inf')

        for run in runs:
            if "global_overall_rmse" in run.summary and run.summary["global_overall_rmse"] < min_rmse:
                min_rmse = run.summary["global_overall_rmse"]
                best_run = run
        
        if best_run:
            best_config = best_run.config
            best_metrics = best_run.summary

            # Prepare data for the table (now a single row for the nation)
            row = {'Region': 'National'} # Label it as 'National'
            row.update(best_config)
            row['best_rmse'] = best_metrics.get('global_overall_rmse', np.nan)
            row['best_mae'] = best_metrics.get('global_overall_mae', np.nan)
            all_best_hypers.append(row)
            print(f"Best hyperparameters for National: {best_config}")
            print(f"Best RMSE for National: {best_metrics.get('global_overall_rmse', np.nan)}")
            print(f"Best MAE for National: {best_metrics.get('global_overall_mae', np.nan)}")

        else:
            print(f"No successful runs found for the national sweep {sweep_id}.")
            row = {'Region': 'National', 'best_rmse': np.nan, 'best_mae': np.nan}
            # Fill in config parameters with NaN for runs where no best was found
            for param in sweep_config['parameters'].keys():
                row[param] = np.nan
            all_best_hypers.append(row)

    except Exception as e:
        print(f"Error retrieving best hyperparameters for National: {e}")
        row = {'Region': 'National', 'best_rmse': np.nan, 'best_mae': np.nan}
        for param in sweep_config['parameters'].keys():
            row[param] = np.nan
        all_best_hypers.append(row)

# --- Final Aggregation and CSV Saving ---
if all_best_hypers:
    best_hypers_df = pd.DataFrame(all_best_hypers)
    # Changed output filename to reflect national scope and random search
    best_hypers_csv_path = TABLES_DIR / f"xgb_national_random_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv" 
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
else:
    print("\nNo best national hyperparameters found to save or log.")