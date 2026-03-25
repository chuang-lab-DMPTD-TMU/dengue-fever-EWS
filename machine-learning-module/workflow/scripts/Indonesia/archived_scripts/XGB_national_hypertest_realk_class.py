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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "WF0729"

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



# Add validation method-specific parameters
if validation_strategy == 'kfold':
    shared_hyperparameters['n_splits'] = {'value': 5, 'type': 'fixed'}
elif validation_strategy == 'walk':
    shared_hyperparameters['initial_train_months'] = {'value': 60, 'type': 'fixed'}
    shared_hyperparameters['val_window'] = {'value': 6, 'type': 'fixed'}
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
# Ensure IR_class_log is string before encoding (for robust mapping)
df['IR_class_log'] = df['IR_class_log'].astype(str)
le = LabelEncoder()
df['IR_class_encoded'] = le.fit_transform(df['IR_class_log'])

# Print the mapping from original class to integer
print("\n--- Class label encoding mapping (original -> integer) ---")
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(class_mapping)
print("-" * 50)

# Define prediction target as integer label for class
y_target = 'IR_class_encoded'
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
X_features = [
    col for col in features
    if col not in ['YearMonth', 'ID_2', 'Year', y_target]
]

# ENFORCE: All feature columns must be numeric for GPU XGBoost
non_numeric_cols = df[X_features].select_dtypes(include=['object', 'string']).columns.tolist()
if non_numeric_cols:
    print(f"Warning: The following feature columns are not numeric and will be converted: {non_numeric_cols}")
    for col in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
# After conversion, check for any remaining non-numeric columns
still_non_numeric = df[X_features].select_dtypes(include=['object', 'string']).columns.tolist()
if still_non_numeric:
    print(f"Error: The following columns could not be converted to numeric and will be dropped: {still_non_numeric}")
    X_features = [col for col in X_features if col not in still_non_numeric]

# Drop NaNs based on the columns actually used for model training
df_full = df.dropna(subset=X_features + [y_target])

# Print the final list of features that will be used
print("\n--- Final list of features for the model ---")
print(df_full.columns)
print(f"Total features: {len(df_full.columns)}")
print("-" * 50)

# Data for hyperparameter tuning (all data before 2023 for validation)
df_train_val_national = df_full[df_full['YearMonth'].dt.year < 2023].copy()
# Data for final, unseen test - only 2023 data (not used in sweep, but good for context)
df_test_national = df_full[df_full['YearMonth'].dt.year == 2023].copy()


print(f"Shape of df_train_val_national: {df_train_val_national.shape}")
print(f"Shape of df_test_national: {df_test_national.shape}")

# --- Refactored training function for a single run (K-fold and Walk-Forward version) ---
def train(df_input, feature_columns, target_column):
    """
    Trains an XGBoost classifier model using either K-fold cross-validation or
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

        # Convert to cuDF Dataframes for GPU acceleration
        X_gpu = cudf.DataFrame(df_input[feature_columns])
        y_gpu = cudf.DataFrame(df_input[[target_column]])

        all_preds = []
        all_true = []
        fold_metrics = []
        model = None

        # --- K-Fold Cross-Validation Setup ---
        if current_strategy == 'kfold':
            n_splits = config.n_splits # Number of folds for K-fold CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=64) # Shuffle for KFold

            print(f"Performing {n_splits}-fold cross-validation...")

            # Iterate through K-fold splits
            for i, (train_index, val_index) in enumerate(kf.split(X_gpu.to_pandas())):
                train_index_gpu = cp.array(train_index)
                val_index_gpu = cp.array(val_index)

                X_train_fold_gpu = X_gpu.iloc[train_index_gpu]
                y_train_fold_gpu = y_gpu.iloc[train_index_gpu]
                X_val_fold_gpu = X_gpu.iloc[val_index_gpu]
                y_val_fold_gpu = y_gpu.iloc[val_index_gpu]

                if X_train_fold_gpu.empty or X_val_fold_gpu.empty:
                    print(f"Warning: Fold {i+1} has empty train or val split. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBClassifier(
                    objective='multi:softprob',
                    num_class=4,
                    tree_method="hist", device="cuda",
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
                    print(f"Error during model fit in Fold {i+1}: {e}")
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
                predictions = np.maximum(0, predictions).astype(int)

                # --- Evaluate and Log Fold Metrics ---
                y_val_fold_np = y_val_fold_gpu.to_numpy().flatten()
                
                # Calculate classification metrics
                fold_accuracy = accuracy_score(y_val_fold_np, predictions)
                fold_f1 = f1_score(y_val_fold_np, predictions, average='macro')
                fold_precision = precision_score(y_val_fold_np, predictions, average='macro', zero_division=0)
                fold_recall = recall_score(y_val_fold_np, predictions, average='macro', zero_division=0)

                fold_metrics.append({
                    'accuracy': fold_accuracy, 
                    'f1_macro': fold_f1, 
                    'precision_macro': fold_precision,
                    'recall_macro': fold_recall
                })
                
                print(f"Fold {i+1}/{n_splits}: Accuracy={fold_accuracy:.4f}, F1-macro={fold_f1:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}")

                run.log({
                    f"fold_{i+1}_accuracy": fold_accuracy,
                    f"fold_{i+1}_f1_macro": fold_f1,
                    f"fold_{i+1}_precision_macro": fold_precision,
                    f"fold_{i+1}_recall_macro": fold_recall,
                    "current_fold": i + 1,
                    "fold_accuracy_history": fold_accuracy,
                    "fold_f1_history": fold_f1,
                    "fold_precision_history": fold_precision,
                    "fold_recall_history": fold_recall
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_val_fold_np)

        elif current_strategy == 'walk':
            # --- Walk-Forward Validation Setup ---
            initial_train_months = config.initial_train_months
            val_window = config.val_window

            unique_time_periods = df_input['YearMonth'].unique()
            n_time_periods = len(unique_time_periods)

            n_splits = (n_time_periods - initial_train_months) // val_window

            if n_splits <= 0:
                print("Not enough unique time periods for specified initial_train_months and val_window.")
                print(f"Total time periods: {n_time_periods}, Initial train: {initial_train_months}, Val window: {val_window}")
                run.log({"error": "Insufficient data for splits"})
                return

            print(f"Performing {n_splits} global walk-forward splits based on time periods...")

            for i in range(n_splits):
                train_end_period_idx = initial_train_months + i * val_window
                val_start_period_idx = train_end_period_idx
                val_end_period_idx = val_start_period_idx + val_window

                if val_end_period_idx > n_time_periods:
                    print(f"Warning: Val end period index {val_end_period_idx} exceeds total time periods {n_time_periods}. Ending walk-forward.")
                    break

                train_end_time = unique_time_periods[train_end_period_idx - 1]
                val_start_time = unique_time_periods[val_start_period_idx]
                val_end_time = unique_time_periods[val_end_period_idx - 1]

                train_df_period = df_input[df_input['YearMonth'] <= train_end_time]
                val_df_period = df_input[
                    (df_input['YearMonth'] >= val_start_time) &
                    (df_input['YearMonth'] <= val_end_time)
                ]

                X_train_fold_gpu = cudf.DataFrame(train_df_period[feature_columns])
                y_train_fold_gpu = cudf.DataFrame(train_df_period[[target_column]])
                X_val_fold_gpu = cudf.DataFrame(val_df_period[feature_columns])
                y_val_fold_gpu = cudf.DataFrame(val_df_period[[target_column]])

                if X_train_fold_gpu.empty or X_val_fold_gpu.empty:
                    print(f"Warning: Fold {i+1} has empty train or val split after time-based slicing. Skipping.")
                    continue

                # --- Model Training ---
                model = xgboost.XGBClassifier(
                    objective='multi:softprob',
                    num_class=4,
                    tree_method="hist", device="cuda",
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
                    print(f"Error during model fit in Fold {i+1}: {e}")
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
                predictions = np.maximum(0, predictions).astype(int)

                # --- Evaluate and Log Fold Metrics ---
                y_val_fold_np = y_val_fold_gpu.to_numpy().flatten()
                
                # Calculate classification metrics
                fold_accuracy = accuracy_score(y_val_fold_np, predictions)
                fold_f1 = f1_score(y_val_fold_np, predictions, average='macro')
                fold_precision = precision_score(y_val_fold_np, predictions, average='macro', zero_division=0)
                fold_recall = recall_score(y_val_fold_np, predictions, average='macro', zero_division=0)
                
                fold_metrics.append({
                    'accuracy': fold_accuracy, 
                    'f1_macro': fold_f1, 
                    'precision_macro': fold_precision,
                    'recall_macro': fold_recall
                })
                
                print(f"Fold {i+1}/{n_splits}: Accuracy={fold_accuracy:.4f}, F1-macro={fold_f1:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}")

                run.log({
                    f"fold_{i+1}_accuracy": fold_accuracy,
                    f"fold_{i+1}_f1_macro": fold_f1,
                    f"fold_{i+1}_precision_macro": fold_precision,
                    f"fold_{i+1}_recall_macro": fold_recall,
                    "current_fold": i + 1,
                    "fold_accuracy_history": fold_accuracy,
                    "fold_f1_history": fold_f1,
                    "fold_precision_history": fold_precision,
                    "fold_recall_history": fold_recall
                }, step=i+1)
                all_preds.extend(predictions)
                all_true.extend(y_val_fold_np)
        else:
            raise ValueError("Invalid validation_strategy specified. Choose 'kfold' or 'walk_forward'.")

        # --- Calculate and Log Global Overall Metrics (AFTER the loop) ---
        if all_true:
            global_overall_accuracy = accuracy_score(all_true, all_preds)
            global_overall_f1 = f1_score(all_true, all_preds, average='macro')
            global_overall_precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
            global_overall_recall = recall_score(all_true, all_preds, average='macro', zero_division=0)

            run.log({
                "global_overall_accuracy": global_overall_accuracy,
                "global_overall_f1_macro": global_overall_f1,
                "global_overall_precision_macro": global_overall_precision,
                "global_overall_recall_macro": global_overall_recall,
                "mean_fold_accuracy": np.mean([f['accuracy'] for f in fold_metrics]),
                "mean_fold_f1_macro": np.mean([f['f1_macro'] for f in fold_metrics]),
                "mean_fold_precision_macro": np.mean([f['precision_macro'] for f in fold_metrics]),
                "mean_fold_recall_macro": np.mean([f['recall_macro'] for f in fold_metrics]),
                "n_splits_completed": len(fold_metrics)
            })
            print(f"\n--- GLOBAL Overall {current_strategy.replace('_', '-').upper()} Evaluation: Accuracy={global_overall_accuracy:.4f}, F1-macro={global_overall_f1:.4f} ---")
        else:
            print(f"\n--- No successful predictions made during {current_strategy.replace('_', '-')}. Setting metrics to infinity. ---")
            run.log({"global_overall_accuracy": 0, "global_overall_f1_macro": 0, "global_overall_precision_macro": 0, "global_overall_recall_macro": 0})

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


print(f"\n--- Starting NATIONAL classifier sweep with {validation_strategy.replace('_', '-').upper()} validation and data version '{VERSION_STAMP}' ---")


if df_train_val_national.empty:
    print(f"No training/validation data for national sweep. Cannot proceed.")
else:
    # Initialize the sweep with a project name specific to the national scope, strategy, and data version
    project_name = f"dengue-indonesia-xgb-national-class-{validation_suffix}-{landuse_suffix}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID for National: {sweep_id}")

    # Run the sweep agent for the entire nation
    wandb.agent(sweep_id, function=lambda: train(df_train_val_national.copy(), feature_columns=X_features, target_column=y_target), count=n_runs_for_national_sweep)
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
    best_hypers_csv_path = TABLES_DIR / f"xgb_national_{validation_suffix}_hyperparameters_{landuse_suffix}{version_suffix}.csv"
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
else:
    print("\nNo best national hyperparameters found to save or log.")

print("\n--- NATIONAL Sweep and Hyperparameter Retrieval Completed ---")
print("\n--- Begin XGB_national_test_kfold.py ---")
# Read hyperparameters for the national model
hyperparams_df = pd.read_csv(best_hypers_csv_path)
print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

all_shap_plots = []
all_pdp_plots = []
national_summary_data = {}

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
X_train_val = cudf.DataFrame(df_train_val_national[X_features])
y_train_val = cudf.DataFrame(df_train_val_national[[y_target]])
X_test = cudf.DataFrame(df_test_national[X_features])
y_test = cudf.DataFrame(df_test_national[[y_target]])
X_full = cudf.DataFrame(df_full[X_features])
y_full = cudf.DataFrame(df_full[[y_target]])

# Select a subset of X_train_val for SHAP background sampling
background_sample = df_train_val_national[X_features].sample(n=100, random_state=42)

# Initialize and train the XGBoost Classifier model
model = xgboost.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    tree_method='hist',
    device='cuda',
    random_state=64,
    **national_hyperparams
)
model.fit(X_train_val, y_train_val)

# Initialize dictionary to store metrics for the national model
current_national_metrics = {}

for label, X_gpu, y_gpu in [
    ("Train/Val", X_train_val, y_train_val),
    ("Test", X_test, y_test),
    ("Full", X_full, y_full)
]:
    preds_gpu = model.predict(X_gpu)
    preds = cp.asnumpy(preds_gpu)
    y_true = y_gpu.to_numpy().flatten()
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    precision = precision_score(y_true, preds, average='macro', zero_division=0)
    recall = recall_score(y_true, preds, average='macro', zero_division=0)

    current_national_metrics[f'{label} Accuracy'] = accuracy
    current_national_metrics[f'{label} F1-macro'] = f1
    current_national_metrics[f'{label} Precision-macro'] = precision
    current_national_metrics[f'{label} Recall-macro'] = recall
    
    # Print the metrics
    print(f"Metrics for {label} set: Accuracy={accuracy:.4f}, F1-macro={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    # Prepare values for SHAP tree explainer -> 
    # takes strictly numpy arrays or pandas DataFrames
    X_pd = X_gpu.to_pandas()
    explainer = shap.TreeExplainer(model, data=background_sample, feature_perturbation="auto")
    shap_vals = explainer.shap_values(X_pd.values)
    
    # SHAP returns a list of arrays for multi-class, so we need to average them
    if isinstance(shap_vals, list):
        shap_vals_np = np.stack(shap_vals, axis=-1).mean(axis=-1)
    else:
        shap_vals_np = shap_vals
    
    all_shap_plots.append((shap_vals_np, X_pd, f"National - {label}", accuracy, f1, precision, recall))

    if label == "Test":
        mean_abs_shap = np.abs(shap_vals_np).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=X_pd.columns)
        top_5_features = feature_importance.nlargest(5).index.tolist()
        print(f"Top 5 predictors for National (Test Set): {top_5_features}")

        for feature in top_5_features:
            all_pdp_plots.append((model, X_pd, feature, "National"))

# Store all calculated metrics and class distribution for the national model
national_summary_data['National'] = current_national_metrics
# For a classification task, it's more relevant to log the class distribution
national_summary_data['National']['Train/Val Class Distribution'] = dict(df_train_val_national[y_target].value_counts().sort_index())
national_summary_data['National']['Test Class Distribution'] = dict(df_test_national[y_target].value_counts().sort_index())
national_summary_data['National']['Full Class Distribution'] = dict(df_full[y_target].value_counts().sort_index())

# Generate SHAP plots and save them to a PDF file
pdf_output_filename = f"xgb_national_class_{validation_strategy}_plots_{landuse_suffix}_{version_digits}.pdf"
pdf_path = FIGURES_DIR / pdf_output_filename

with PdfPages(pdf_path) as pdf:
    # SHAP summary plots
    for shap_vals_np, X_pd, title, accuracy, f1, precision, recall in all_shap_plots:
        fig, ax = plt.subplots(figsize=(10, 8 + len(X_features) * 0.25))
        plt.sca(ax)
        shap.summary_plot(shap_vals_np, X_pd, show=False)
        ax.set_title(f"{title} | Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
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
csv_output_filename = f"xgb_national_class_{validation_strategy}_test_{landuse_suffix}_{version_digits}.csv"
csv_filename = TABLES_DIR / csv_output_filename
summary_df.to_csv(csv_filename, index=True)

print(f"National summary table saved to '{csv_filename}'")
print("-" * 50)