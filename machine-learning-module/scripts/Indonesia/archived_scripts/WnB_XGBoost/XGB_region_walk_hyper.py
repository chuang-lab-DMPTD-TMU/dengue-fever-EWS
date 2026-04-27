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
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# Load the data once outside the train function for efficiency in a sweep
df = pd.read_csv(PROCESSED_DIR /"monthly_dengue_env_id.csv")

# --- REGION REASSIGNMENT ---
df['Region_Group'] = df['Region'].replace({'Maluku Islands': 'Maluku_Papua', 'Papua': 'Maluku_Papua'})
print("--- DataFrame after Region_Group creation ---")
print(df['Region_Group'].value_counts())
print("-" * 50)

df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime

# Define variables based on new names
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max', # Add min/max for dummy data if needed
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum', 'evaporative_stress_index', 'aridity_index',
    'temperature_2m_ANOM', 'temperature_2m_min_ANOM',
    'temperature_2m_max_ANOM', 'potential_evaporation_sum_ANOM',
    'total_evaporation_sum_ANOM', 'precipitation_ANOM'
]

climate_vars = [
    'ANOM1+2','ANOM3','ANOM4',
    'ANOM3.4', 'DMI', 'DMI_East',
]

target = 'Incidence_Rate'

# Crucial: Sort by time FIRST, then by ID_2 to ensure correct slicing for all ID_2s in a time period
# This is fundamental for walk-forward validation on panel data.
df = df.sort_values(['YearMonth', 'ID_2'])

print("--- DataFrame after initial sorting (df.sort_values(['YearMonth', 'ID_2'])) ---")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("-" * 50)


# Create lag features - Group by ID_2 to ensure lags are per-area
for var in env_vars:
    for lag in [1, 2, 3]:
        df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

for var in climate_vars:
    for lag in [1, 3, 6]:
        df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

print("--- DataFrame after lag feature creation ---")
print(df.head()) # Observe the new _lag columns and their NaNs
print(df.info())
print(df.isnull().sum()) # Notice increased NaNs due to lags
print("-" * 50)

# Original features
features = [col for col in env_vars + climate_vars  if col in df.columns]
for var in env_vars:
    for lag in [1, 2, 3]:
        if f'{var}_lag{lag}' in df.columns:
            features.append(f'{var}_lag{lag}')
for var in climate_vars:
    for lag in [1, 3, 6]:
        if f'{var}_lag{lag}' in df.columns:
            features.append(f'{var}_lag{lag}')

# Print the final list of features that will be used
print("\n--- Final list of features for the model ---")
print(features)
print(f"Total features: {len(features)}")
print("-" * 50)


# --- Refactored training function for a single run ---
def train(df_input, feature_columns, target_column):
    """
    Trains an XGBoost Regressor model using walk-forward validation and logs
    metrics and artifacts to Weights & Biases.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing all features and target.
        feature_columns (list): A list of column names to be used as features.
        target_column (str): The name of the target variable column.
    """
    # The project name now comes from the sweep agent's context, which is set per-region
    with wandb.init() as run:
        config = run.config

        print(f"\n--- Starting run with config: {config} ---")

        # --- Data Preprocessing (ensure 'YearMonth' is datetime and sort) ---
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
        df_processed_index = df_processed[['YearMonth', 'ID_2']].reset_index(drop=True)


        # --- Walk-Forward Validation Setup ---
        initial_train_months = config.initial_train_months # This is now the number of *distinct YearMonth periods*
        test_window = config.test_window # This is the number of *distinct YearMonth periods* to test

        # Determine unique YearMonth periods to define the time-based splits
        unique_time_periods = df_processed['YearMonth'].unique()
        n_time_periods = len(unique_time_periods)

        # Calculate the number of splits based on time periods
        n_splits = (n_time_periods - initial_train_months) // test_window

        if n_splits <= 0:
            print("Not enough unique time periods for specified initial_train_months and test_window.")
            print(f"Total time periods: {n_time_periods}, Initial train: {initial_train_months}, Test window: {test_window}")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf'), "error": "Insufficient data for splits"})
            return

        print(f"Performing {n_splits} global walk-forward splits based on time periods...")

        all_preds = []
        all_true = []
        fold_metrics = []

        model = None

        # Iterate through global time-based splits
        for i in range(n_splits):
            # Define the training and testing time period ranges
            train_end_period_idx = initial_train_months + i * test_window
            test_start_period_idx = train_end_period_idx
            test_end_period_idx = test_start_period_idx + test_window

            if test_end_period_idx > n_time_periods:
                print(f"     Warning: Test end period index {test_end_period_idx} exceeds total time periods {n_time_periods}. Ending walk-forward.")
                break

            # Get the actual datetime values for these periods
            train_end_time = unique_time_periods[train_end_period_idx - 1] # -1 because unique_time_periods is 0-indexed
            test_start_time = unique_time_periods[test_start_period_idx]
            test_end_time = unique_time_periods[test_end_period_idx - 1]

            # Slice the data based on these time periods
            # Use 'df_processed' (pandas) for time-based filtering, then convert to cuDF
            train_df_period = df_processed[df_processed['YearMonth'] <= train_end_time]
            test_df_period = df_processed[
                (df_processed['YearMonth'] >= test_start_time) &
                (df_processed['YearMonth'] <= test_end_time)
            ]

            # Convert to cuDF for training/prediction
            X_train_fold_gpu = cudf.DataFrame(train_df_period[actual_feature_columns])
            y_train_fold_gpu = cudf.DataFrame(train_df_period[[target_column]])
            X_test_fold_gpu = cudf.DataFrame(test_df_period[actual_feature_columns])
            y_test_fold_gpu = cudf.DataFrame(test_df_period[[target_column]])

            if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
                print(f"     Warning: Fold {i+1} has empty train or test split after time-based slicing. Skipping.")
                continue

            # --- Model Training ---
            model = xgboost.XGBRegressor(
                objective='reg:squarederror', tree_method="hist", device="cuda",
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                random_state=42,
                n_jobs=-1, # Use all available cores (CPU cores, but XGBoost will use GPU)
                min_child_weight=config.min_child_weight,
                gamma=config.gamma,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda
            )

            try:
                model.fit(X_train_fold_gpu, y_train_fold_gpu)
            except Exception as e:
                print(f"     Error during model fit in Fold {i+1}: {e}")
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
            print(f"     Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

            run.log({
                f"fold_{i+1}_rmse": fold_rmse,
                f"fold_{i+1}_mae": fold_mae,
                "current_fold": i + 1,
                "fold_rmse_history": fold_rmse,
                "fold_mae_history": fold_mae
            }, step=i+1)
            all_preds.extend(predictions)
            all_true.extend(y_test_fold_np)

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
            print(f"\n--- GLOBAL Overall Walk-Forward Evaluation: RMSE={global_overall_rmse:.2f}, MAE={global_overall_mae:.2f} ---")
        else:
            print("\n--- No successful predictions made during walk-forward. Setting metrics to infinity. ---")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf')})

        # --- Model Artifact Saving ---
        if model is not None:
            try:
                model_filename = "xgboost_model.json"
                model.save_model(model_filename)

                artifact = wandb.Artifact(
                    name=f"dengue-xgboost-model_{run.id}",
                    type="model",
                    description="XGBoost model trained during sweep run",
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


# Define the sweep configuration directly in Python
sweep_config = {
    'method': 'bayes',        # grid, random, or bayes
    'metric': {
        'name': 'global_overall_rmse',
        'goal': 'minimize'
    },
    'parameters': {
        # Fixed parameters (or parameters you don't want to sweep)
        'initial_train_months': {'value': 60},
        'test_window': {'value': 6},

        # Hyperparameters to sweep
        'n_estimators': {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 300
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.3
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 7
        },
        'subsample': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
        'colsample_bytree': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
        'min_child_weight': {'distribution': 'int_uniform', 'min': 1, 'max': 10},
        'gamma': {'distribution': 'uniform', 'min': 0, 'max': 5},
        'reg_alpha': {'distribution': 'uniform', 'min': 0, 'max': 1},
        'reg_lambda': {'distribution': 'uniform', 'min': 0, 'max': 1}
    }
}

# Set WANDB settings
os.environ["WANDB_MODE"] = "online"

# --- Iterate through regions and run sweeps ---
unique_regions = df['Region_Group'].unique()
n_runs_per_sweep = 50 # Number of runs for each regional sweep
all_best_hypers = []

for region in unique_regions:
    print(f"\n--- Starting sweep for Region: {region} ---")

    # Filter data for the current region
    df_region = df[df['Region_Group'] == region].copy()

    # Data for hyperparameter tuning (walk-forward validation) - all data before 2023
    df_train_val_region = df_region[df_region['YearMonth'].dt.year < 2023].copy()
    # Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
    df_test_region = df_region[df_region['YearMonth'].dt.year == 2023].copy()

    print(f"Shape of df_train_val for {region}: {df_train_val_region.shape}")
    print(f"Shape of df_final_test for {region}: {df_test_region.shape}")

    if df_train_val_region.empty:
        print(f"No training/validation data for region {region}. Skipping sweep for this region.")
        continue

    # Initialize the sweep with a project name specific to the region
    project_name = f"dengue-indonesia-xgb-{region.replace(' ', '_')}-walk"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID for {region}: {sweep_id}")

    # Run the sweep agent for the current region
    wandb.agent(sweep_id, function=lambda: train(df_train_val_region.copy(), feature_columns=features, target_column=target), count=n_runs_per_sweep)
    print(f"Sweep for Region {region} completed.")

    # --- Retrieve and Log Best Hyperparameters for the Region ---
    try:
        api = wandb.Api()
        # Find the project for the current region
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

            # Prepare data for the table
            row = {'Region': region}
            row.update(best_config)
            row['best_rmse'] = best_metrics.get('global_overall_rmse', np.nan)
            row['best_mae'] = best_metrics.get('global_overall_mae', np.nan)
            all_best_hypers.append(row)
            print(f"Best hyperparameters for {region}: {best_config}")
            print(f"Best RMSE for {region}: {best_metrics.get('global_overall_rmse', np.nan)}")
            print(f"Best MAE for {region}: {best_metrics.get('global_overall_mae', np.nan)}")

        else:
            print(f"No successful runs found for region {region} in sweep {sweep_id}.")
            row = {'Region': region, 'best_rmse': np.nan, 'best_mae': np.nan}
            # Fill in config parameters with NaN for runs where no best was found
            for param in sweep_config['parameters'].keys():
                row[param] = np.nan
            all_best_hypers.append(row)

    except Exception as e:
        print(f"Error retrieving best hyperparameters for {region}: {e}")
        row = {'Region': region, 'best_rmse': np.nan, 'best_mae': np.nan}
        for param in sweep_config['parameters'].keys():
            row[param] = np.nan
        all_best_hypers.append(row)

# --- Final Aggregation and CSV Saving ---
if all_best_hypers:
    best_hypers_df = pd.DataFrame(all_best_hypers)
    best_hypers_csv_path = TABLES_DIR / "xgb_region_walk_hyperparameters"
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best hyperparameters to {best_hypers_csv_path}")
else:
    print("\nNo best hyperparameters found to save or log.")