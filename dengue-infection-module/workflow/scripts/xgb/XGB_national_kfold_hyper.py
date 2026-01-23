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
from sklearn.model_selection import KFold # Import KFold
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

# lulc_vars = ['Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95', 'Class_30',
#               'Class_20', 'Class_10', 'Class_90', 'Class_80']

target = 'Incidence_Rate'

# Crucial: Sort by time FIRST, then by ID_2 to ensure correct slicing for all ID_2s in a time period
# This is fundamental for walk-forward validation on panel data.
# This sorting is less critical for KFold, but keeping it doesn't hurt.
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
features = [col for col in env_vars + climate_vars if col in df.columns]
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


# --- Split data into training/validation and final test sets ---
# Assuming 'YearMonth' is in YYYYMM format (e.g., 202301)
# Create a 'Year' column if not already present, or extract from 'YearMonth'
if 'Year' not in df.columns:
    df['Year'] = df['YearMonth'].astype(str).str[:4].astype(int)

df_train_val = df[df['Year'] < 2023].copy()
df_test = df[df['Year'] == 2023].copy()

print(f"\n--- Data Split Information ---")
print(f"Shape of training/validation data (excluding 2023): {df_train_val.shape}")
print(f"Min YearMonth in train/val: {df_train_val['YearMonth'].min()}, Max YearMonth in train/val: {df_train_val['YearMonth'].max()}")
print(f"Shape of 2023 test data: {df_test.shape}")
print(f"Min YearMonth in test: {df_test['YearMonth'].min()}, Max YearMonth in test: {df_test['YearMonth'].max()}")
print("-" * 50)


# --- Refactored training function for a single run using KFold ---
def train(df_input, feature_columns, target_column):
    """
    Trains an XGBoost Regressor model using K-Fold cross-validation and logs
    metrics and artifacts to Weights & Biases.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing all features and target.
        feature_columns (list): A list of column names to be used as features.
        target_column (str): The name of the target variable column.
    """
    # Use the desired project name directly here
    with wandb.init(project="dengue-indonesia-xgb-national-kfold") as run:
        config = run.config

        print(f"\n--- Starting run with config: {config} ---")

        df_local = df_input.copy()

        actual_feature_columns = [col for col in feature_columns if col not in ['YearMonth', 'ID_2', 'Year', target_column]] # Exclude 'Year' as a feature if present

        # Drop NaNs based on the columns actually used for model training
        df_processed = df_local.dropna(subset=actual_feature_columns + [target_column])

        if df_processed.empty:
            print("DataFrame is empty after dropping NaNs. Cannot proceed with training.")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf'), "error": "Empty DataFrame after NaN drop"})
            return

        # Convert to cuDF DataFrames for GPU acceleration
        # We will convert the full df_processed to cuDF once for efficiency
        X_gpu_full = cudf.DataFrame(df_processed[actual_feature_columns])
        y_gpu_full = cudf.DataFrame(df_processed[[target_column]])

        # --- K-Fold Validation Setup ---
        n_splits = config.n_splits # K-fold parameter
        random_state = config.random_state # For reproducibility of splits

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        print(f"Performing {n_splits}-fold randomized cross-validation...")

        all_preds = []
        all_true = []
        fold_metrics = []

        model = None # Initialize model outside loop

        # Iterate through K-Fold splits
        for i, (train_index, test_index) in enumerate(kf.split(X_gpu_full)):
            print(f"   Processing Fold {i+1}/{n_splits}")

            # Get GPU data for current fold
            X_train_fold_gpu = X_gpu_full.iloc[train_index]
            y_train_fold_gpu = y_gpu_full.iloc[train_index]
            X_test_fold_gpu = X_gpu_full.iloc[test_index]
            y_test_fold_gpu = y_gpu_full.iloc[test_index]

            if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
                print(f"    Warning: Fold {i+1} has empty train or test split. Skipping.")
                run.log({f"fold_{i+1}_error": "Empty train/test split"}, step=i+1)
                continue

            # --- Model Training ---
            model = xgboost.XGBRegressor(
                objective='reg:squarederror', tree_method="hist", device="cuda",
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                random_state=42, # Still use a consistent random state for model initialization
                n_jobs=-1, # Use all available cores (CPU cores, but XGBoost will use GPU)
                min_child_weight=config.min_child_weight,
                gamma=config.gamma,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda
            )

            try:
                model.fit(X_train_fold_gpu, y_train_fold_gpu)
            except Exception as e:
                print(f"    Error during model fit in Fold {i+1}: {e}")
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
            print(f"    Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

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
            print(f"\n--- GLOBAL Overall K-Fold Evaluation: RMSE={global_overall_rmse:.2f}, MAE={global_overall_mae:.2f} ---")
        else:
            print("\n--- No successful predictions made during K-fold. Setting metrics to infinity. ---")
            run.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf')})

        # --- Model Artifact Saving ---
        # Save the model from the *last* fold or retrain on full data if desired.
        # For KFold, it's common to retrain on full data or save the last model if you only need one.
        # Here, we'll save the model from the last completed fold.
        if model is not None:
            try:
                model_filename = "xgboost_model.json"
                model.save_model(model_filename)

                artifact = wandb.Artifact(
                    name=f"dengue-xgboost-model_{run.id}",
                    type="model",
                    description="XGBoost model trained during sweep run (last fold of KFold)",
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


# %%
# Define the sweep configuration directly in Python
sweep_config = {
    'method': 'bayes',        # grid, random, or bayes
    'metric': {
        'name': 'global_overall_rmse',
        'goal': 'minimize'
    },
    'parameters': {
        # New K-fold parameters
        'n_splits': {'value': 5}, # Typically 5 or 10 folds
        'random_state': {'value': 42}, # For reproducibility of splits

        # Hyperparameters to sweep
        'n_estimators': {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 300
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.2
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

# %%
# --- Step A: Initialize the sweep (requires brief internet connection) ---
n_runs = 100
print("Initializing sweep... (requires brief internet connection)")
# Use the desired project name directly here for consistency
project_name = "dengue-indonesia-xgb-national-kfold"
sweep_id = wandb.sweep(sweep_config, project=project_name)
print(f"Sweep ID: {sweep_id}")

# --- Step B: Set WANDB settings ---
os.environ["WANDB_MODE"] = "online" # Keep this if you want online by default
os.environ["WANDB_SILENT"] = "true" # Commented out for verbose logging during debugging

# --- Step C: Run the sweep agent ---
print(f"Starting sweep agent for {n_runs} runs...")

# Pass the pre-loaded df_train_val to avoid reloading it inside each train() call,
# but ensure train() receives a copy to prevent accidental modification.
wandb.agent(sweep_id, function=lambda: train(df_train_val.copy(), feature_columns=features, target_column=target), count=n_runs)
print("\nSweep completed.")

# --- Step D: Export best hyperparameters to CSV ---
print("Exporting best hyperparameters to CSV...")
api = wandb.Api()
# Use the same project name variable defined earlier
runs = api.runs(f"{api.default_entity}/{project_name}", filters={"sweep": sweep_id})

# Filter out runs that might not have the metric (e.g., failed runs or runs that didn't complete training)
successful_runs = [run for run in runs if "global_overall_rmse" in run.summary]

if successful_runs:
    # Sort runs by the metric defined in sweep_config's 'goal'
    metric_name = sweep_config['metric']['name']
    goal = sweep_config['metric']['goal']

    if goal == 'minimize':
        best_run = min(successful_runs, key=lambda run: run.summary[metric_name])
    else: # maximize
        best_run = max(successful_runs, key=lambda run: run.summary[metric_name])

    print(f"\nBest run found (ID: {best_run.id}) with {metric_name}: {best_run.summary[metric_name]:.4f}")

    best_hyperparameters = best_run.config

    # Convert the dictionary of hyperparameters to a pandas DataFrame for easy saving
    df_best_hp = pd.DataFrame([best_hyperparameters])

    # Define the output path for the CSV, using TABLES_DIR
    # Naming the file to clearly indicate it's from the K-Fold sweep
    csv_filename = TABLES_DIR / f"xgb_national_kfold_hyperparameters.csv"
    TABLES_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    df_best_hp.to_csv(csv_filename, index=False)

    print(f"Best hyperparameters saved to: {csv_filename}")
else:
    print("No successful runs found in the sweep to determine best hyperparameters.")
print("-" * 50)