import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import shap
import wandb
import cuml
from cuml import ensemble
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import cudf
import cupy as cp
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold # Import KFold

# print(cuml.__version__)
# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KF0729"

# Set WANDB settings to online/offline
os.environ["WANDB_MODE"] = "online"

# --- Run sweep for the entire nation ---
n_runs_for_national_sweep = 100 # Number of runs for the national sweep
all_best_hypers = [] # This will store the single best result for the nation

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
version_suffix = f"{VERSION_STAMP}"

# --- Logging ---
print(f"Using validation strategy: {validation_strategy} in version {VERSION_STAMP}")
print(f"Land Use Features Included: {USE_LANDUSE_FEATURES}")


# Define the shared hyperparameters with the desired sweep ranges
shared_hyperparameters = {
    'n_estimators': {
        'distribution': 'int_uniform',
        'min': 50,
        'max': 400
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 7
    },
    'max_features': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 1.0
    },
    'min_samples_leaf': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 20
    },
    'bootstrap': {
        'values': [True, False]
    }
}

# Define the base sweep configuration
base_sweep_config = {
    'method': 'bayes',
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
    Trains an RandomForest Regressor model using either K-fold cross-validation or
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
                    print(f"       Warning: Fold {i+1} has empty train or test split. Skipping.")
                    continue

                # --- Model Training ---
                model = ensemble.RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    max_features=config.max_features,
                    min_samples_leaf=config.min_samples_leaf,
                    bootstrap=config.bootstrap,
                    random_state=64
                )

                try:
                    model.fit(X_train_fold_gpu, y_train_fold_gpu)
                except Exception as e:
                    print(f"       Error during model fit in Fold {i+1}: {e}")
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
                print(f"       Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

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
                    print(f"       Warning: Test end period index {test_end_period_idx} exceeds total time periods {n_time_periods}. Ending walk-forward.")
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
                    print(f"       Warning: Fold {i+1} has empty train or test split after time-based slicing. Skipping.")
                    continue

                # --- Model Training ---
                model = ensemble.RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    max_features=config.max_features,
                    min_samples_leaf=config.min_samples_leaf,
                    bootstrap=config.bootstrap,
                    random_state=64
                )

                try:
                    model.fit(X_train_fold_gpu, y_train_fold_gpu)
                except Exception as e:
                    print(f"       Error during model fit in Fold {i+1}: {e}")
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
                print(f"       Fold {i+1}/{n_splits}: RMSE={fold_rmse:.2f}, MAE={fold_mae:.2f}")

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
                # Save model using pickle (cuML models)
                
                model_filename = "rf_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)

                artifact = wandb.Artifact(
                    name=f"dengue-rf-model_{run.id}",
                    type="model",
                    description=f"Random Forest model trained during sweep run ({current_strategy})",
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


print(f"\n--- Starting NATIONAL sweep with {validation_strategy.replace('_', '-').upper()} validation and data version '{VERSION_STAMP}' ---")

# Data for hyperparameter tuning (all data before 2023 for validation)
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=actual_feature_columns + [target])
# Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=actual_feature_columns + [target])


print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

# if df_train_val_national.empty:
#     print(f"No training/validation data for national sweep. Cannot proceed.")
# else:
#     # Initialize the sweep with a project name specific to the national scope, strategy, and data version
#     project_name = f"dengue-indonesia-rf-national-{validation_suffix}-{landuse_suffix}"
#     sweep_id = wandb.sweep(sweep_config, project=project_name)
#     print(f"Sweep ID for National: {sweep_id}")

#     # Run the sweep agent for the entire nation
#     wandb.agent(sweep_id, function=lambda: train(df_train_val_national.copy(), feature_columns=features, target_column=target), count=n_runs_for_national_sweep)
#     print(f"National sweep completed.")

#     # --- Retrieve and Log Best Hyperparameters for the Nation ---
#     try:
#         api = wandb.Api()
#         # Find the project for the national run
#         current_project = api.project(project_name)
        
#         # Get all runs from the current sweep, sorted by the metric (global_overall_rmse)
#         runs = api.runs(f"{current_project.entity}/{project_name}",
#                          filters={"sweep": sweep_id})
        
#         best_run = None
#         min_rmse = float('inf')

#         for run in runs:
#             if "global_overall_rmse" in run.summary and run.summary["global_overall_rmse"] < min_rmse:
#                 min_rmse = run.summary["global_overall_rmse"]
#                 best_run = run
        
#         if best_run:
#             best_config = best_run.config
#             best_metrics = best_run.summary

#             # Prepare data for the table (now a single row for the nation)
#             row = {'Region': 'National'} # Label it as 'National'
#             row.update(best_config)
#             row['best_rmse'] = best_metrics.get('global_overall_rmse', np.nan)
#             row['best_mae'] = best_metrics.get('global_overall_mae', np.nan)
#             all_best_hypers.append(row)
#             print(f"Best hyperparameters for National: {best_config}")
#             print(f"Best RMSE for National: {best_metrics.get('global_overall_rmse', np.nan)}")
#             print(f"Best MAE for National: {best_metrics.get('global_overall_mae', np.nan)}")

#         else:
#             print(f"No successful runs found for the national sweep {sweep_id}.")
#             row = {'Region': 'National', 'best_rmse': np.nan, 'best_mae': np.nan}
#             # Fill in config parameters with NaN for runs where no best was found
#             for param in sweep_config['parameters'].keys():
#                 row[param] = np.nan
#             all_best_hypers.append(row)

#     except Exception as e:
#         print(f"Error retrieving best hyperparameters for National: {e}")
#         row = {'Region': 'National', 'best_rmse': np.nan, 'best_mae': np.nan}
#         for param in sweep_config['parameters'].keys():
#             row[param] = np.nan
#         all_best_hypers.append(row)

# # --- Final Aggregation and CSV Saving ---
# if all_best_hypers:
#     best_hypers_df = pd.DataFrame(all_best_hypers)
#     best_hypers_csv_path = TABLES_DIR / f"rf_national_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv"
#     best_hypers_df.to_csv(best_hypers_csv_path, index=False)
#     print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
# else:
#     print("\nNo best national hyperparameters found to save or log.")

print("\n--- NATIONAL Sweep and Hyperparameter Retrieval Completed ---")
print("\n--- Begin RandomForest_national_test_kfold.py ---")

best_hypers_csv_path = TABLES_DIR / f"rf_national_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv"
# Read hyperparameters for the national model
hyperparams_df = pd.read_csv(best_hypers_csv_path)
print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

all_shap_plots = []
all_pdp_plots = []
national_summary_data = {}

# Concatenate for full dataset analysis
df_full = pd.concat([df_train_val_national, df_test_national], ignore_index=True)


# Extract hyperparameters for the national model
params = hyperparams_df.iloc[0]  # Assuming the first row contains national hyperparameters
national_hyperparams = {
    'n_estimators': int(params['n_estimators']),
    'max_depth': int(params['max_depth']),
    'max_features': params['max_features'],
    'min_samples_leaf': int(params['min_samples_leaf']),
    'bootstrap': params['bootstrap']
}

# Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
X_train = df_train_val_national[actual_feature_columns]
y_train = df_train_val_national[[target]]
X_test = df_test_national[actual_feature_columns]
y_test = df_test_national[[target]]
X_full = df_full[actual_feature_columns]
y_full = df_full[[target]]

# Select a subset of X_train for SHAP background sampling
background_sample = df_train_val_national[actual_feature_columns].sample(n=100, random_state=42)

# Initialize and train the SKLEARN Random Forest Regressor model
model = RandomForestRegressor(
    random_state=64,
    **national_hyperparams
)
model.fit(X_train, y_train.values.flatten())

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
    X_pd = X_gpu
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
national_summary_data['National']['IR Min'] = df_full[target].min()
national_summary_data['National']['IR Max'] = df_full[target].max()
national_summary_data['National']['IR 25th Quantile'] = df_full[target].quantile(0.25)
national_summary_data['National']['IR 50th Quantile'] = df_full[target].quantile(0.50)
national_summary_data['National']['IR 75th Quantile'] = df_full[target].quantile(0.75)


# Generate PDP plots and save them to a PDF file
pdf_output_filename = f"rf_national_{validation_strategy}_plots_{landuse_suffix}_{version_digits}.pdf"
pdf_path = FIGURES_DIR / pdf_output_filename

with PdfPages(pdf_path) as pdf:
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

print(f"National PDP plots saved to '{pdf_path}'")

# --- Generate National Summary Table ---
print("\n--- National Summary Table ---")

# Convert the collected summary data into a pandas DataFrame
summary_df = pd.DataFrame.from_dict(national_summary_data, orient='index').T

# Save the DataFrame to a CSV file
csv_output_filename = f"rf_national_{validation_strategy}_test_{landuse_suffix}_{version_digits}.csv"
csv_filename = TABLES_DIR / csv_output_filename
summary_df.to_csv(csv_filename, index=True, float_format="%.2f")

print(f"National summary table saved to '{csv_filename}'")
print("-" * 50)
