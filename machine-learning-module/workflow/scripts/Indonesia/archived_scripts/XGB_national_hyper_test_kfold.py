import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import cudf
import cupy as cp
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold # Import KFold

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KT0731"

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
    'name': 'global_overall_accuracy',
    'goal': 'maximize'
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
df = pd.read_csv(PROCESSED_DIR /"monthly_dengue_env_id_with_class.csv")

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
target = 'IR_class_log'

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

# Data for hyperparameter tuning (all data before 2023 for validation)
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=actual_feature_columns + [target])
# Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=actual_feature_columns + [target])

print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

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
        # Ensure target is integer-encoded for classification
        y_gpu = cudf.DataFrame(df_processed[[target_column]].astype(int))

        all_preds = []
        all_true = []
        fold_metrics = []
        acc_scores = []
        model = None

        if current_strategy == 'kfold':
            # --- K-Fold Validation Setup ---
            n_splits = config.n_splits # Number of folds for K-fold CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=64) # Shuffle for KFold

            print(f"Performing {n_splits}-fold cross-validation...")

            # Iterate through K-fold splits
            for i, (train_index, val_index) in enumerate(kf.split(X_gpu.to_pandas())):
                # Use .take for cudf DataFrames instead of iloc with cupy arrays
                X_train_fold_gpu = X_gpu.take(train_index)
                y_train_fold_gpu = y_gpu.take(train_index)
                X_val_fold_gpu = X_gpu.take(val_index)
                y_val_fold_gpu = y_gpu.take(val_index)

                if X_train_fold_gpu.empty or X_val_fold_gpu.empty:
                    print(f"      Warning: Fold {i+1} has empty train or validation split. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBClassifier(
                    objective='multi:softprob', num_class=4, tree_method="hist", device="cuda",
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
                    model.fit(
                        X_train_fold_gpu, y_train_fold_gpu,
                        eval_set=[(X_val_fold_gpu, y_val_fold_gpu)],
                        verbose=False
                    )
                except Exception as e:
                    print(f"      Error during model fit in Fold {i+1}: {e}")
                    run.log({f"fold_{i+1}_error": str(e)}, step=i+1)
                    continue

                # Log best_iteration and best_score if available
                if hasattr(model, 'best_iteration') and hasattr(model, 'best_score'):
                    run.log({
                        f"fold_{i+1}_best_iteration": model.best_iteration,
                        f"fold_{i+1}_best_score": model.best_score
                    }, step=i+1)

                # --- Prediction ---
                predictions_gpu = model.predict(X_val_fold_gpu)
                predictions = cp.asnumpy(predictions_gpu)
                predictions = np.maximum(0, predictions)

                # --- Evaluate and Log Fold Metrics ---
                y_val_fold_np = y_val_fold_gpu.to_numpy().flatten().astype(int)
                fold_acc = accuracy_score(y_val_fold_np, predictions)
                fold_f1 = f1_score(y_val_fold_np, predictions, average='weighted')
                fold_metrics.append({'acc': fold_acc, 'f1': fold_f1})
                acc_scores.append(fold_acc)
                print(f"      Fold {i+1}/{n_splits}: ACC={fold_acc:.3f}, F1={fold_f1:.3f}")

                run.log({
                    f"fold_{i+1}_accuracy": fold_acc,
                    f"fold_{i+1}_f1": fold_f1,
                    "current_fold": i + 1,
                    "fold_acc_history": fold_acc,
                    "fold_f1_history": fold_f1
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_val_fold_np)

        elif current_strategy == 'walk':
            # --- Walk-Forward Validation Setup ---
            initial_train_months = config.initial_train_months
            val_window = config.test_window

            unique_time_periods = df_processed['YearMonth'].unique()
            n_time_periods = len(unique_time_periods)

            n_splits = (n_time_periods - initial_train_months) // val_window

            if n_splits <= 0:
                print("Not enough unique time periods for specified initial_train_months and validation window.")
                print(f"Total time periods: {n_time_periods}, Initial train: {initial_train_months}, Validation window: {val_window}")
                run.log({"error": "Insufficient data for splits"})
                return

            print(f"Performing {n_splits} global walk-forward splits based on time periods...")

            for i in range(n_splits):
                train_end_period_idx = initial_train_months + i * val_window
                val_start_period_idx = train_end_period_idx
                val_end_period_idx = val_start_period_idx + val_window

                if val_end_period_idx > n_time_periods:
                    print(f"      Warning: Validation end period index {val_end_period_idx} exceeds total time periods {n_time_periods}. Ending walk-forward.")
                    break

                train_end_time = unique_time_periods[train_end_period_idx - 1]
                val_start_time = unique_time_periods[val_start_period_idx]
                val_end_time = unique_time_periods[val_end_period_idx - 1]

                train_df_period = df_processed[df_processed['YearMonth'] <= train_end_time]
                val_df_period = df_processed[
                    (df_processed['YearMonth'] >= val_start_time) &
                    (df_processed['YearMonth'] <= val_end_time)
                ]

                X_train_fold_gpu = cudf.DataFrame(train_df_period[actual_feature_columns])
                y_train_fold_gpu = cudf.DataFrame(train_df_period[[target_column]])
                X_val_fold_gpu = cudf.DataFrame(val_df_period[actual_feature_columns])
                y_val_fold_gpu = cudf.DataFrame(val_df_period[[target_column]])

                if X_train_fold_gpu.empty or X_val_fold_gpu.empty:
                    print(f"      Warning: Fold {i+1} has empty train or validation split after time-based slicing. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBClassifier(
                    objective='multi:softprob', num_class=4, tree_method="hist", device="cuda",
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
                    model.fit(
                        X_train_fold_gpu, y_train_fold_gpu,
                        eval_set=[(X_val_fold_gpu, y_val_fold_gpu)],
                        verbose=False
                    )
                except Exception as e:
                    print(f"      Error during model fit in Fold {i+1}: {e}")
                    run.log({f"fold_{i+1}_error": str(e)}, step=i+1)
                    continue

                # Log best_iteration and best_score if available
                if hasattr(model, 'best_iteration') and hasattr(model, 'best_score'):
                    run.log({
                        f"fold_{i+1}_best_iteration": model.best_iteration,
                        f"fold_{i+1}_best_score": model.best_score
                    }, step=i+1)

                # --- Prediction ---
                predictions_gpu = model.predict(X_val_fold_gpu)
                predictions = cp.asnumpy(predictions_gpu)
                predictions = np.maximum(0, predictions)

                # --- Evaluate and Log Fold Metrics ---
                y_val_fold_np = y_val_fold_gpu.to_numpy().flatten().astype(int)
                fold_acc = accuracy_score(y_val_fold_np, predictions)
                fold_f1 = f1_score(y_val_fold_np, predictions, average='weighted')
                fold_metrics.append({'acc': fold_acc, 'f1': fold_f1})
                acc_scores.append(fold_acc)
                print(f"      Fold {i+1}/{n_splits}: ACC={fold_acc:.3f}, F1={fold_f1:.3f}")

                run.log({
                    f"fold_{i+1}_accuracy": fold_acc,
                    f"fold_{i+1}_f1": fold_f1,
                    "current_fold": i + 1,
                    "fold_acc_history": fold_acc,
                    "fold_f1_history": fold_f1
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_val_fold_np)
        else:
            raise ValueError("Invalid validation_strategy specified. Choose 'kfold' or 'walk_forward'.")

        # --- Calculate and Log Global Overall Metrics (AFTER the loop) ---
        if all_true:
            global_overall_accuracy = accuracy_score(np.array(all_true).astype(int), np.array(all_preds).astype(int))
            global_overall_f1 = f1_score(np.array(all_true).astype(int), np.array(all_preds).astype(int), average='weighted')

            run.log({
                "global_overall_accuracy": global_overall_accuracy,
                "global_overall_f1": global_overall_f1,
                "mean_fold_accuracy": np.mean([f['acc'] for f in fold_metrics]),
                "mean_fold_f1": np.mean([f['f1'] for f in fold_metrics]),
                "n_splits_completed": len(fold_metrics)
            })
            print(f"\n--- GLOBAL Overall {current_strategy.replace('_', '-').upper()} Evaluation: ACC={global_overall_accuracy:.3f}, F1={global_overall_f1:.3f} ---")

            # Save best hyperparameters and metrics for this run
            best_hyper = dict(config)
            best_hyper['global_overall_accuracy'] = global_overall_accuracy
            best_hyper['global_overall_f1'] = global_overall_f1
            best_hyper['mean_fold_accuracy'] = np.mean([f['acc'] for f in fold_metrics])
            best_hyper['mean_fold_f1'] = np.mean([f['f1'] for f in fold_metrics])
            best_hyper['n_splits_completed'] = len(fold_metrics)
            # Use global variable
            global all_best_hypers
            all_best_hypers.append(best_hyper)
        else:
            print(f"\n--- No successful predictions made during {current_strategy.replace('_', '-')}. Setting accuracy and F1 to zero. ---")
            run.log({"global_overall_accuracy": 0.0, "global_overall_f1": 0.0})

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
            print("Validation data for national sweep. Cannot proceed.")

# --- Final Aggregation and CSV Saving ---

# After all runs, select the best hyperparameters (highest global_overall_accuracy) and save only that
if all_best_hypers:
    best_idx = np.argmax([h['global_overall_accuracy'] for h in all_best_hypers])
    best_hyper = all_best_hypers[best_idx]
    best_hypers_df = pd.DataFrame([best_hyper])
    best_hypers_csv_path = TABLES_DIR / f"xgb_national_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv"
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
else:
    print("\nNo best national hyperparameters found to save or log.")

print("\n--- NATIONAL Sweep and Hyperparameter Retrieval Completed ---")
print("\n--- Begin XGB_national_test_kfold.py ---")

# # Read hyperparameters for the national model
# best_hypers_csv_path = TABLES_DIR / f"xgb_national_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv"
# hyperparams_df = pd.read_csv(best_hypers_csv_path)
# print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

# all_shap_plots = []
# all_pdp_plots = []  
# national_summary_data = {}

# # Concatenate for full dataset analysis
# df_full = pd.concat([df_train_val_national, df_test_national], ignore_index=True)

# # Extract hyperparameters for the national model
# params = hyperparams_df.iloc[0]  # Assuming the first row contains national hyperparameters
# national_hyperparams = {
#     'gamma': params['gamma'],
#     'n_estimators': int(params['n_estimators']),
#     'max_depth': int(params['max_depth']),
#     'reg_alpha': params['reg_alpha'],
#     'subsample': params['subsample'],
#     'reg_lambda': params['reg_lambda'],
#     'learning_rate': params['learning_rate'],
#     'colsample_bytree': params['colsample_bytree'],
#     'min_child_weight': int(params['min_child_weight']),
#     'num_class': 4
# }

# X_train = cudf.DataFrame(df_train_val_national[actual_feature_columns])
# y_train = cudf.DataFrame(df_train_val_national[[target]].astype(int))
# X_test = cudf.DataFrame(df_test_national[actual_feature_columns])
# y_test = cudf.DataFrame(df_test_national[[target]].astype(int))
# X_full = cudf.DataFrame(df_full[actual_feature_columns])
# y_full = cudf.DataFrame(df_full[[target]].astype(int))

# # Initialize and train the XGBoost Classifier model
# model = xgboost.XGBClassifier(
#     objective='multi:softprob', # Objective for multiclass classification
#     tree_method='hist',         # Use histogram-based tree method for faster training
#     device='cuda',              # Specify GPU device for training
#     random_state=64,            # For reproducibility
#     **national_hyperparams      # Unpack national hyperparameters
# )
# model.fit(X_train, y_train.values.flatten().astype(int))

# # Initialize dictionary to store metrics for the national model
# current_national_metrics = {}



# for label, X_gpu, y_gpu in [
#     ("Train/Val", X_train, y_train),
#     ("Test", X_test, y_test),
#     ("Full", X_full, y_full)
# ]:
#     preds = model.predict(X_gpu)
#     y_true = y_gpu.to_numpy().flatten().astype(int)

#     acc = accuracy_score(y_true, preds)
#     cm = confusion_matrix(y_true, preds)
#     report = classification_report(y_true, preds, output_dict=True)

#     current_national_metrics[f'{label} Accuracy'] = acc
#     current_national_metrics[f'{label} ConfusionMatrix'] = cm.tolist()
#     current_national_metrics[f'{label} ClassificationReport'] = report

# # Store all calculated metrics and IR_class_log statistics for the national model
# national_summary_data['National'] = current_national_metrics
# national_summary_data['National']['Class Min'] = df_full[target].min()
# national_summary_data['National']['Class Max'] = df_full[target].max()
# national_summary_data['National']['Class 25th Quantile'] = df_full[target].quantile(0.25)
# national_summary_data['National']['Class 50th Quantile'] = df_full[target].quantile(0.50)
# national_summary_data['National']['Class 75th Quantile'] = df_full[target].quantile(0.75)

# # --- Generate National Summary Table ---
# print("\n--- National Summary Table ---")

# # Convert the collected summary data into a pandas DataFrame
# summary_df = pd.DataFrame.from_dict(national_summary_data, orient='index').T

# # Save the DataFrame to a CSV file
# csv_output_filename = f"xgb_national_{validation_strategy}_test_{landuse_suffix}_{version_digits}.csv"
# csv_filename = TABLES_DIR / csv_output_filename
# summary_df.to_csv(csv_filename, index=True)

# print(f"National summary table saved to '{csv_filename}'")
# print("-" * 50)