# %%
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
from tqdm import tqdm
import shap # Import the SHAP library
import wandb
import xgboost
from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define project root based on notebook location
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()  # fallback

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# %% [markdown]
# Creating lag variables for machine learning (by region)

# %%
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" /"monthly_dengue_env_region.csv")


# Define variables based on new names
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum', 'evaporative_stress_index', 'aridity_index'
]

climate_vars = [
    'ANOM1+2','ANOM3','ANOM4',
    'ANOM3.4', 'DMI', 'DMI_East',
]

lulc_vars = ['Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95', 'Class_30',
        'Class_20', 'Class_10', 'Class_90', 'Class_80']

target_var = 'Incidence_Rate'

# Sort by region and time
df = df.sort_values(['Region', 'YearMonth'])

# Create lag features
# Environmental: 1, 2, 3 month lags
for var in env_vars:
    for lag in [1, 2, 3]:
        df[f'{var}_lag{lag}'] = df.groupby('Region')[var].shift(lag)

# Climate indices: 1, 3, 6 month lags
for var in climate_vars:
    for lag in [1, 3, 6]:
        df[f'{var}_lag{lag}'] = df.groupby('Region')[var].shift(lag)

# Original features
feat_list = env_vars + climate_vars + lulc_vars

# Lag features for environmental variables (1, 2, 3 months)
for var in env_vars:
    for lag in [1, 2, 3]:
        feat_list.append(f'{var}_lag{lag}')

# Lag features for climate variables (1, 3, 6 months)
for var in climate_vars:
    for lag in [1, 3, 6]:
        feat_list.append(f'{var}_lag{lag}')

# %%
# --- Define the training function for a single run ---
# Pass df as an argument
def train(df, features_to_use=None): # Add features_to_use as an argument, with a default of None
    # wandb.init() will automatically pick up the config from the sweep controller
    # This creates a new run for each iteration of the sweep
    with wandb.init(project="dengue-indonesia-forecasting") as run:
        # Access hyperparameters from wandb.config
        config = run.config

        print(f"\n--- Starting run with config: {config} ---")

        # --- Load and Preprocess Data
        # Ensure 'YearMonth' is a datetime object and sort by it for time series processing
        # Important: Sort by YearMonth first, then Region to ensure consistent time ordering
        # before one-hot encoding, especially if data for different regions might not be perfectly aligned
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        df = df.sort_values(by=['YearMonth', 'Region']).reset_index(drop=True)

        # --- Handle 'Region' column ---
        # If 'Region' is in features_to_use, it will be handled as a regular numerical feature
        # If not, it will be excluded from feature columns.
        # We are explicitly NOT one-hot encoding 'Region' based on your request.

        # Define your features (X) and target (y)
        target_column = 'Incidence_Rate'

        # Determine the feature columns
        if features_to_use is not None:
            # Use the specified list of features, ensuring 'Region' is excluded if it was implicitly included
            # Filter out 'YearMonth' and target_column if they were somehow included in features_to_use
            feature_columns = [col for col in features_to_use if col not in ['YearMonth', target_column, 'Region']]
        else:
            # Original behavior: use all columns except 'YearMonth' and 'Total_Infection' and 'Region'
            feature_columns = [col for col in df.columns if col not in ['YearMonth', target_column, 'Region']]
            
        # If 'Region' needs to be used as a numerical feature, ensure it's not removed by the above logic.
        # But based on your request "I don't want to use region as a training feature", it's excluded.
        
        print(f"Features selected for training: {feature_columns}")

        # Final check for NaNs after all preprocessing steps
        df_processed = df.dropna(subset=feature_columns + [target_column])

        # Walk-Forward Validation Setup (on the entire combined dataset)
        initial_train_months = config.initial_train_months # Get from config
        test_window = config.test_window # Get from config

        n_samples = len(df_processed)
        n_splits = (n_samples - initial_train_months) // test_window

        if n_splits <= 0:
            print("Not enough data for specified initial_train_months and test_window.")
            print(f"Total samples: {n_samples}, Initial train: {initial_train_months}, Test window: {test_window}")
            run.finish()
            return

        print(f"Performing {n_splits} walk-forward splits on combined data...")

        all_preds = []
        all_true = []
        fold_metrics = []
        
        # Lists to store SHAP values for aggregation or further analysis
        all_shap_values = []
        all_expected_values = []
        all_X_test_folds = [] # <--- NEW: Store X_test_fold for SHAP plotting

        for i in range(n_splits):
            train_end_idx = initial_train_months + i * test_window
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_window

            if test_end_idx > n_samples:
                print(f"    Warning: Test end index {test_end_idx} exceeds total samples {n_samples}. Ending walk-forward.")
                break # Stop if we run out of data for a full test window

            X_train_fold = df_processed.iloc[:train_end_idx][feature_columns]
            y_train_fold = df_processed.iloc[:train_end_idx][target_column]
            X_test_fold = df_processed.iloc[test_start_idx:test_end_idx][feature_columns]
            y_test_fold = df_processed.iloc[test_start_idx:test_end_idx][target_column]

            # --- Model Training ---
            model = xgboost.XGBRegressor(
                objective='reg:squarederror', tree_method="hist", device="cuda", # device="cuda" is key for GPU
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                random_state=42,
                n_jobs=-1 # Use all available cores
            )

            model.fit(X_train_fold, y_train_fold)

            # --- Prediction ---
            predictions = model.predict(X_test_fold)
            predictions = np.maximum(0, predictions).astype(int)

            # --- Calculate SHAP values with GPU acceleration ---
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_fold)
            expected_value = explainer.expected_value
            
            all_shap_values.append(shap_values)
            all_expected_values.append(expected_value)
            all_X_test_folds.append(X_test_fold) # <--- NEW: Append the X_test_fold here

            # --- Evaluate and Log Fold Metrics ---
            fold_rmse = np.sqrt(mean_squared_error(y_test_fold, predictions))
            fold_mae = mean_absolute_error(y_test_fold, predictions)

            fold_metrics.append({'rmse': fold_rmse, 'mae': fold_mae})
            print(f"    Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")
            # Log metrics for the *current* fold
            wandb.log({
                f"fold_{i+1}_rmse": fold_rmse, # Log RMSE for this specific fold
                f"fold_{i+1}_mae": fold_mae,   # Log MAE for this specific fold
                "current_fold": i + 1,       # Log the current fold number as a step
                "fold_rmse_history": fold_rmse, # Log general fold RMSE to see trend
                "fold_mae_history": fold_mae    # Log general fold MAE to see trend
            }, step=i+1) # Use step to track progress over folds
            all_preds.extend(predictions)
            all_true.extend(y_test_fold)

        # --- Calculate and Log Global Overall Metrics ---
        if all_true:
            global_overall_rmse = np.sqrt(mean_squared_error(all_true, all_preds))
            global_overall_mae = mean_absolute_error(all_true, all_preds)

            wandb.log({
                "global_overall_rmse": global_overall_rmse,
                "global_overall_mae": global_overall_mae,
                "mean_fold_rmse": np.mean([f['rmse'] for f in fold_metrics]),
                "mean_fold_mae": np.mean([f['mae'] for f in fold_metrics]),
                "n_splits": n_splits
            })

            print(f"\n--- GLOBAL Overall Walk-Forward Evaluation: RMSE={global_overall_rmse:.2f}, MAE={global_overall_mae:.2f} ---")
        else:
            print("\n--- No successful predictions made during walk-forward. ---")
            wandb.log({"global_overall_rmse": float('inf'), "global_overall_mae": float('inf')})
            
        # --- Aggregate and Log SHAP values (Optional, but useful) ---
        if all_shap_values and all_X_test_folds: # <--- Ensure both lists have data
            # Concatenate all SHAP values from folds for a global view
            concatenated_shap_values = np.vstack(all_shap_values)
            # <--- NEW: Concatenate all X_test_folds to get all feature values for plotting
            concatenated_X_test_folds = pd.concat(all_X_test_folds, axis=0) 
            
            # Save raw SHAP data and X_test for future aggregation
            np.save("shap_values.npy", concatenated_shap_values)
            concatenated_X_test_folds.to_csv("X_test_for_shap.csv", index=False)

            # Log as a W&B artifact
            artifact = wandb.Artifact("shap_data", type="dataset")
            artifact.add_file("shap_values.npy")
            artifact.add_file("X_test_for_shap.csv")
            run.log_artifact(artifact)

            try:
                # Pass the actual feature values for correct coloring
                # Ensure feature names are correctly aligned, usually by passing the DataFrame
                shap.summary_plot(concatenated_shap_values, concatenated_X_test_folds, show=False)
                wandb.log({"shap_summary_plot": wandb.Image(plt)}) # Log the plot
                plt.close() # Close the plot to prevent it from displaying in notebooks
            except Exception as e:
                print(f"Could not generate SHAP summary plot: {e}")

            # You can also log individual feature importances based on mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(concatenated_shap_values), axis=0)
            feature_importances_shap = pd.DataFrame({
                'feature': concatenated_X_test_folds.columns, # Use columns from the concatenated df
                'importance': mean_abs_shap
            }).sort_values(by='importance', ascending=False)
            
            print("\n--- SHAP Feature Importances (Mean Absolute SHAP Value) ---")
            print(feature_importances_shap)
            wandb.log({"shap_feature_importances": wandb.Table(dataframe=feature_importances_shap)})

# %%
# Jupyter Notebook Cell 2: Define Sweep Configuration
# Define the sweep configuration directly in Python
sweep_config = {
    'method': 'bayes',     # grid, random, or bayes
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
sweep_id = wandb.sweep(sweep_config, project="dengue-indonesia-forecasting")
print(f"Sweep ID: {sweep_id}")

# --- Step B: Set WANDB settings ---
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_SILENT"] = "true"

# --- Step C: Run the sweep agent ---
print(f"Starting sweep agent for {n_runs} runs...")

wandb.agent(sweep_id, function=lambda: train(df.copy(), features_to_use=feat_list), count=n_runs)
print("\nSweep completed.")

# --- Step D: Post-Sweep Aggregation of SHAP Values ---
print("\nAggregating SHAP values from completed runs...")

# Replace these with your W&B entity and project
ENTITY = "patwuch"
PROJECT = "dengue-indonesia-forecasting"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"sweep": sweep_id, "state": "finished"})

all_shap = []
all_X = []

for run in runs:
    try:
        artifact = run.use_artifact("shap_data:latest")
        artifact_dir = artifact.download()
        shap_values = np.load(os.path.join(artifact_dir, "shap_values.npy"))
        X_test = pd.read_csv(os.path.join(artifact_dir, "X_test_for_shap.csv"))

        all_shap.append(shap_values)
        all_X.append(X_test)

    except Exception as e:
        print(f"Skipping run {run.name} due to error: {e}")

if all_shap and all_X:
    print("Generating global SHAP summary plot...")
    global_shap = np.vstack(all_shap)
    global_X = pd.concat(all_X, axis=0)

    shap.summary_plot(global_shap, global_X, show=True)  # Set show=False to save instead of display
else:
    print("No SHAP data found across runs. Global SHAP summary plot will not be created.")

# %%



