import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import xgboost
from xgboost import XGBClassifier # Changed to XGBClassifier
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import cudf
import cupy as cp
import optuna
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gc
import psutil
import pynvml

pynvml.nvmlInit()
def log_memory(tag=""):
    """Log system RAM + GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024**2

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_used_mb = gpu_mem.used / 1024**2

    print(f"[{tag}] RAM: {ram_mb:.2f} MB | GPU: {gpu_used_mb:.2f} MB")
# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "WT1014"

# --- Run sweep for the entire nation ---
n_trials_for_study = 40 # Number of trials for the study






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
all_study_name = f"xgbC-regions-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"

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
SQLITE_DIR = PROJECT_ROOT / "sqlite"
MODELS_DIR = PROJECT_ROOT / "models" / "xgboost"
TABLES_DIR = REPORTS_DIR / "tables"



# Load the data once outside the train function for efficiency
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" / "monthly_dengue_env_id_class_log.csv")

df['Risk_Category'] = df['Risk_Category'].replace({
    'Zero': 0,
    'Low': 1,
    'High': 2}).infer_objects(copy=False)
df['Risk_Category'] = df['Risk_Category'].astype('int32')
num_classes = df['Risk_Category'].nunique()
print("-" * 50)
# Create a list of regions to iterate over
regions_to_model = df['Region_Group'].unique()


df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime
df['Incidence_Rate_lag1'] = df.groupby('ID_2')['Incidence_Rate'].shift(1)

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

epidemic_vars = ['Incidence_Rate_lag1']

climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']

# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
variable_columns = []
for var in env_vars + climate_vars + epidemic_vars:
    if var in df.columns:
        variable_columns.append(var)
if USE_LANDUSE_FEATURES:
    for var in land_use_vars:
        if var in df.columns:
            variable_columns.append(var)
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                variable_columns.append(lagged_var)


# Select relevant columns (metadata, variables, target)
target = 'Risk_Category'
metadata_columns = ['YearMonth', 'ID_2', 'Region_Group','Incidence_Rate']
# Final feature list excluding metadata and target
variable_columns = [
    col for col in variable_columns
    if col not in [metadata_columns, target]
]

# Use a clearly writable folder under 'reports'
region_stats_dir = REPORTS_DIR / "region_env_stats"
region_stats_dir.mkdir(parents=True, exist_ok=True)

from tqdm import tqdm

# Loop over each region
for region_name in tqdm(df['Region_Group'].unique(), desc="Saving region stats"):
    df_region = df[df['Region_Group'] == region_name]
    if df_region.empty:
        continue

    # Compute descriptive stats
    env_stats = df_region[env_vars].describe(percentiles=[0.25, 0.5, 0.75]).T

    # Round all numeric values to 4 decimals
    env_stats = env_stats.round(4)

    # Add region name column
    env_stats.insert(0, "Region", region_name)

    # Save to CSV safely
    output_path = region_stats_dir / f"{region_name}_env_stats.csv"
    try:
        env_stats.to_csv(output_path, index=True)
    except PermissionError as e:
        print(f"⚠️ Skipped {region_name}: {e}")

print(f"[OK] All stats saved under: {region_stats_dir.resolve()}")

# Data for hyperparameter tuning
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=variable_columns + [target])
# Data for final, unseen test
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=variable_columns + [target])

# Convert to cudf DataFrames for GPU processing
cudf_train_val_national = cudf.DataFrame(df_train_val_national)[variable_columns + metadata_columns + [target]]
cudf_test_national = cudf.DataFrame(df_test_national)[variable_columns + metadata_columns + [target]]

print("Starting training with the following columns:")

print("--- Target Column ---")
print([target])
print("-" * 50)
print("--- Metadata Columns ---")
print(metadata_columns)
print("-" * 50)
print("--- Variable Columns ---")
print(variable_columns)

def calculate_sample_weights(y):
    """
    Calculate sample weights to handle class imbalance.
    Weights are inversely proportional to class frequencies.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    num_classes = len(unique_classes)
    
    weights = {}
    for i, cls in enumerate(unique_classes):
        weights[cls] = total_samples / (num_classes * counts[i])
    
    # Create an array of weights corresponding to the y array
    sample_weights = np.array([weights[cls] for cls in y])
    return sample_weights


# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================

import cudf
import gc

# def get_splits_gpu(df, validation_flag, train_window=None, test_window=None):
#     """
#     Generates train/test splits on the GPU, correctly handling multiple rows per time step,
#     using only cuDF operations.

#     Returns a list of tuples: (fold_number, train_idx, test_idx)
#     where train_idx and test_idx are cudf.RangeIndex objects.
#     """
#     if validation_flag != 'w':
#         raise ValueError(f"Unsupported validation flag: {validation_flag}")

#     if test_window is None:
#         raise ValueError("test_window must be provided for walk-forward validation")

#     # Sort and get unique time steps, reset index to a standard RangeIndex
#     df_sorted = df.sort_values('YearMonth').reset_index(drop=True)
#     unique_time_steps = df_sorted['YearMonth'].unique()
#     n_time_steps = len(unique_time_steps)
    
#     splits = []
    
#     initial_train_window = 36 if train_window is None else train_window
    
#     # Calculate the number of folds
#     num_folds = (n_time_steps - initial_train_window) // test_window
    
#     for fold_num in range(num_folds):
#         # Calculate the start and end indices for the current fold based on unique time steps
#         test_start_idx_time = initial_train_window + fold_num * test_window
#         test_end_idx_time = test_start_idx_time + test_window
        
#         test_start_time = unique_time_steps[test_start_idx_time]
#         test_end_time = unique_time_steps[test_end_idx_time]
        
#         # Expanding window logic
#         if train_window is None:
#             train_start_time = unique_time_steps[0]
#         # Rolling window logic
#         else:
#             train_start_idx_time = test_start_idx_time - train_window
#             if train_start_idx_time < 0:
#                 print("Warning: Skipping fold, not enough data for train window.")
#                 continue
#             train_start_time = unique_time_steps[train_start_idx_time]

#         # Use cuDF's `searchsorted` to find the indices of the full dataframe
#         # that correspond to the start and end of the time windows.
#         train_start_row_idx = df_sorted['YearMonth'].searchsorted(train_start_time, side='left')
#         train_end_row_idx = df_sorted['YearMonth'].searchsorted(test_start_time, side='left')
        
#         test_start_row_idx = df_sorted['YearMonth'].searchsorted(test_start_time, side='left')
#         test_end_row_idx = df_sorted['YearMonth'].searchsorted(test_end_time, side='left')

#         # Create cuDF RangeIndex objects from these row indices
#         train_idx = cudf.RangeIndex(train_start_row_idx, train_end_row_idx)
#         test_idx = cudf.RangeIndex(test_start_row_idx, test_end_row_idx)
        
#         # Ensure splits are not empty
#         if not train_idx.empty and not test_idx.empty:
#             splits.append((fold_num, train_idx, test_idx))

#     return splits

# def objective(trial, X_gpu, y_gpu, splits, num_classes=2):
#     y_gpu = y_gpu.squeeze()

#     params = {
#         'objective': 'binary:logistic' if num_classes == 2 else 'multi:softprob',
#         'num_class': num_classes if num_classes > 2 else None,
#         'tree_method': 'hist',
#         'device': 'cuda',
#         'random_state': 64,
#         'n_jobs': -1,
#         'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 2, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
#         'gamma': trial.suggest_float('gamma', 0.1, 10, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
#         'eval_metric': 'logloss' if num_classes == 2 else 'mlogloss'
#     }
    
#     num_boost_round = trial.suggest_int('n_estimators', 50, 3000)

#     all_preds, all_true = [], []

#     for fold_num, train_idx, test_idx in splits:
#         X_train, y_train = X_gpu.iloc[train_idx], y_gpu.iloc[train_idx]
#         X_val, y_val = X_gpu.iloc[test_idx], y_gpu.iloc[test_idx]

#         if X_train.empty or X_val.empty:
#             continue

#         log_memory(f"Before fold {fold_num}")

#         sample_weights = calculate_sample_weights(y_train.to_numpy())
#         dtrain = xgboost.DMatrix(X_train, label=y_train.to_numpy(), weight=sample_weights)
#         dval = xgboost.DMatrix(X_val, label=y_val.to_numpy())

#         booster = xgboost.train(
#             params=params,
#             dtrain=dtrain,
#             num_boost_round=num_boost_round,
#             evals=[(dval, 'val')],
#             verbose_eval=False
#         )

#         preds_proba = booster.predict(dval)
#         preds = (preds_proba > 0.5).astype(int) if num_classes == 2 else np.argmax(preds_proba, axis=1)

#         all_preds.append(preds)
#         all_true.append(y_val.to_numpy().flatten())

#         # cleanup
#         del booster, dtrain, dval
#         gc.collect()

#         log_memory(f"After fold {fold_num}")

#     if all_true:
#         acc = accuracy_score(np.concatenate(all_true), np.concatenate(all_preds))
#         log_memory("End of trial")
#         return acc
#     else:
#         return 0.0
    


# all_best_hypers = []

# for region in regions_to_model:
#     print(f"\n--- Starting modeling for Region: {region} ---")
    
#     # Filter the data for the current region
#     cudf_train_val_region = cudf_train_val_national[cudf_train_val_national['Region_Group'] == region].copy()
#     X_cudf = cudf_train_val_region[variable_columns]
#     y_cudf = cudf_train_val_region[target].astype('int32')
    
#     # Generate the splits. These are numpy arrays of indices.
#     splits = get_splits_gpu(
#         df=cudf_train_val_region,
#         validation_flag=validation_flag,
#         test_window=12
#     )
#     print(f"Number of splits: {len(splits)}")
#     for fold_num, train_idx, test_idx in splits:
#         print(f"Fold {fold_num}: Train {len(train_idx)}, Test {len(test_idx)}")

#     # Inside the region loop
#     region_study_name = f"xgbC-{region}-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
#     print(f"Name of study: {region_study_name}")
    
#     # Construct the full path to the .db file in the MODELS_DIR
#     db_path = SQLITE_DIR / f"{region_study_name}.db"
    
#     study = optuna.create_study(
#         study_name=region_study_name,
#         direction='maximize',
#         # Use the absolute path for the storage URL
#         storage=f'sqlite:///{db_path.resolve()}', 
#         load_if_exists=True)
    
#     print(f"Optuna study created/loaded: {db_path.resolve()}")
    
#     print(f"Starting {n_trials_for_study} trials...")
    
#     # Run the optimization
#     study.optimize(
#         lambda trial: objective(
#             trial,
#             X_gpu=X_cudf,
#             y_gpu=y_cudf,
#             splits=splits,
#             num_classes=num_classes
#         ),
#         n_trials=n_trials_for_study,
#         n_jobs=-1
#     )

#     print(f"\n{region} study completed.")
    
#     # --- Retrieve and Log Best Hyperparameters for the Nation ---
#     print(f"\n--- Best Hyperparameters of {region} Found by Optuna ---")
#     best_params = study.best_params
#     best_value = study.best_value
#     print(f"Best Accuracy: {best_value:.4f}")
#     print("Best hyperparameters:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")

#     # Prepare data for CSV saving
#     current_region_params = {
#     'Region': region,  # Use the current region variable
#     'best_accuracy': best_value,
#     **best_params
#     }
#     all_best_hypers.append(current_region_params)

# if all_best_hypers:
#     best_hypers_df = pd.DataFrame(all_best_hypers)
#     # Use a consistent file name for the combined CSV
#     final_hypers_csv_path = TABLES_DIR / f"{all_study_name}_params.csv"
#     best_hypers_df.to_csv(final_hypers_csv_path, index=False)
#     print(f"\nSaved all regional hyperparameters to {final_hypers_csv_path}")
# else:
#     print("\nNo regional hyperparameters found to save.")

# print("\n--- ALL REGIONS Study and Hyperparameter Retrieval Completed ---")


# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================

print("\n--- Begin Final Model Training and Evaluation ---")

import pandas as pd
import cudf
import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_log_error
import warnings



'''Below is the script for training on the whole training set
and evaluating on the unseen test set, including SHAP analysis.
Comment out to use.'''


# --- Paths and setup (assuming these are already defined in your script) ---
# --- Helper function to get top features ---
def get_top_features(explainer, shap_values, num_features=5):
    """
    Identifies the top N features based on mean absolute SHAP value.
    """
    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    return mean_abs_shap.index.tolist()[:num_features]

def get_shap_array(shap_obj):
    """
    Convert shap.Explanation object, list of Explanation, or ndarray to a numpy array.
    Handles binary, multi-class, and regression.
    """
    # If it's a list (multi-class), process each element
    if isinstance(shap_obj, list):
        shap_arrays = []
        for sv in shap_obj:
            if hasattr(sv, "values"):
                shap_arrays.append(sv.values)
            else:  # already ndarray
                shap_arrays.append(sv)
        # If binary classification, return positive class (class 1)
        if len(shap_arrays) == 2:
            return shap_arrays[1]
        else:  # multi-class: average absolute across classes
            return np.mean([np.abs(sv) for sv in shap_arrays], axis=0)
    else:  # single Explanation or ndarray
        if hasattr(shap_obj, "values"):
            return shap_obj.values
        else:
            return shap_obj
        

# --- Get unique regions and read hyperparameters ---
hyperparams_csv_path = TABLES_DIR / f"{all_study_name}_params.csv"
hyperparams_df = pd.read_csv(hyperparams_csv_path)
print(f"Extracted hyperparameters for all regions from: {hyperparams_csv_path}")
excluded_feature = epidemic_vars
batch_size_shap = 5000  # batch size for SHAP on GPU

all_results = []

for region in regions_to_model:
    region_study_name = f"xgbC-{region}-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
    print(f"\n{'='*50}\nProcessing Region: {region}\n{'='*50}")

    cudf_train_val_region = cudf_train_val_national[cudf_train_val_national['Region_Group'] == region].copy()
    cudf_test_region = cudf_test_national[cudf_test_national['Region_Group'] == region].copy()
    
    if cudf_train_val_region.empty or cudf_test_region.empty:
        print(f"Skipping region '{region}' due to insufficient data.")
        continue

    # Convert to Pandas for easier SHAP slicing
    # Even though SHAP uses GPUTreeExplainer, it does not take cudf or cupy directly
    X_train_val = cudf_train_val_region[variable_columns].to_pandas()
    y_train_val = cudf_train_val_region[target].to_pandas().values.flatten()
    X_test = cudf_test_region[variable_columns].to_pandas()
    y_test = cudf_test_region[target].to_pandas().values.flatten()

    model_path = MODELS_DIR / f"{region_study_name}.json"
    region_dir = os.path.join(FIGURES_DIR, region)
    beeswarm_dir = os.path.join(region_dir, "beeswarm_plots")
    dependence_dir = os.path.join(region_dir, "dependence_plots")
    os.makedirs(beeswarm_dir, exist_ok=True)
    os.makedirs(dependence_dir, exist_ok=True)

    # ----------------------------
    # --- Load or Train Model ---
    # ----------------------------
    if model_path.exists():
        print(f"Found pretrained model for '{region}'. Loading...")
        model = xgboost.Booster()
        model.load_model(str(model_path))
    else:
        print(f"No pretrained model found for '{region}'. Training new model...")
        region_params_row = hyperparams_df[hyperparams_df['Region'] == region].iloc[0]
        region_hyperparams = {
            'gamma': region_params_row['gamma'],
            'max_depth': int(region_params_row['max_depth']),
            'reg_alpha': region_params_row['reg_alpha'],
            'subsample': region_params_row['subsample'],
            'reg_lambda': region_params_row['reg_lambda'],
            'learning_rate': region_params_row['learning_rate'],
            'colsample_bytree': region_params_row['colsample_bytree'],
            'min_child_weight': int(region_params_row['min_child_weight']),
            'device': 'cuda'
        }
        if num_classes > 2:
            region_hyperparams['objective'] = 'multi:softprob'
            region_hyperparams['num_class'] = num_classes
        else:
            region_hyperparams['objective'] = 'binary:logistic'

        num_round = int(region_params_row['n_estimators'])
        Dtrain = xgboost.DMatrix(X_train_val, label=y_train_val)
        model = xgboost.train(region_hyperparams, Dtrain, num_boost_round=num_round)
        model.save_model(str(model_path))
        del Dtrain
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"Saved new model for {region}.")

    # =========================================================================
    # --- Evaluate Train/Val and Test Sets ---
    # =========================================================================
    datasets = {
        "Train_Val": (X_train_val, y_train_val),
        "Test": (X_test, y_test)
    }

    for set_name, (X_set, y_set) in datasets.items():
        print(f"\n--- Evaluating {set_name} Set ---")

        Dset = xgboost.DMatrix(X_set)
        y_pred_prob = model.predict(Dset)

        if num_classes > 2:
            y_pred = y_pred_prob.argmax(axis=1)
        else:
            y_pred = (y_pred_prob > 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_set, y_pred)
        report = classification_report(y_set, y_pred, output_dict=True)
        target_counts = pd.Series(y_set).value_counts().to_dict()
        class_labels = list(range(num_classes))
        cm = confusion_matrix(y_set, y_pred, labels=class_labels)

        # Create the plot
        plt.figure(figsize=(num_classes*2, num_classes*2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {region} ({set_name})")

        # Save the plot
        save_path = os.path.join(region_dir, f"confusion_matrix_{set_name}.png")
        plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' avoids cutting off labels
        plt.close()  # Close the figure to free memory

        # Store results
        region_summary = {"Region": region, "Dataset": set_name, "Accuracy": accuracy}
        for i in range(num_classes):
            region_summary[f"Class_{i}_count"] = target_counts.get(i, 0)
            if str(i) in report:
                region_summary[f"Precision_{i}"] = report[str(i)]['precision']
                region_summary[f"Recall_{i}"] = report[str(i)]['recall']
                region_summary[f"F1_{i}"] = report[str(i)]['f1-score']
        region_summary["Macro_Precision"] = report['macro avg']['precision']
        region_summary["Macro_Recall"] = report['macro avg']['recall']
        region_summary["Macro_F1"] = report['macro avg']['f1-score']
        region_summary["Weighted_Precision"] = report['weighted avg']['precision']
        region_summary["Weighted_Recall"] = report['weighted avg']['recall']
        region_summary["Weighted_F1"] = report['weighted avg']['f1-score']
        all_results.append(region_summary)
        print(f"{set_name} Accuracy: {accuracy:.4f} | Target counts: {target_counts}")

        # =========================================================================
        # --- SHAP Analysis (full dataset) ---
        # =========================================================================
        try:
            shap_features = list(X_set.columns)
            X_shap = X_set

            explainer = shap.explainers.GPUTree(model, feature_perturbation="tree_path_dependent")
            
            # Full SHAP computation (no slicing)
            shap_values = explainer.shap_values(X_shap, check_additivity=False)

            # Compute top 5 features per class (excluding specific features)
            top_5_features_by_class = []
            for class_idx in range(num_classes):
                if num_classes > 2:
                    mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)

                # Rank all features by absolute SHAP magnitude
                sorted_idx = mean_abs_shap.argsort()[::-1]
                sorted_features = [shap_features[i] for i in sorted_idx]

                # Exclude unwanted features, then take top 5 remaining
                top_features = [f for f in sorted_features if f not in excluded_feature][:5]
                top_5_features_by_class.append(top_features)
                print(f"Top 5 plotted features for Class {class_idx} ({set_name}): {top_features}")

        except Exception as e:
            print(f"SHAP failed for {region} ({set_name}): {e}")


        # =========================================================================
        # --- SHAP plotting (full dataset) ---
        # =========================================================================
        try:
            shap_features = list(X_set.columns)
            X_shap_plot = X_set  # full feature set for plotting

            # --- Beeswarm plots ---
            for class_idx in range(num_classes):
                plt.figure(figsize=(12, 8))
                if num_classes > 2:
                    shap_vals = shap_values[class_idx]
                else:
                    shap_vals = shap_values

                # Exclude specific features only for visualization
                included_cols = [c for c in X_shap_plot.columns if c not in excluded_feature]
                shap_vals_plot = shap_vals[:, [i for i, c in enumerate(X_shap_plot.columns) if c in included_cols]]
                X_slice_plot = X_shap_plot[included_cols]

                shap.summary_plot(
                    shap_vals_plot,
                    X_slice_plot,
                    show=False
                )
                plt.title(f"SHAP Beeswarm - Class {class_idx} ({set_name}, excl. {excluded_feature})")
                plt.savefig(
                    os.path.join(beeswarm_dir, f"class_{class_idx}_beeswarm({set_name}).png"),
                    bbox_inches="tight", dpi=300
                )
                plt.close()
                print(f"[OK] Saved SHAP beeswarm for Class {class_idx} ({set_name})")

            # --- Dependence plots (top 5 features per class) ---
            for class_idx in range(num_classes):
                top_features = [f for f in top_5_features_by_class[class_idx] if f not in excluded_feature][:5]
                for feature in top_features:
                    plt.figure(figsize=(10, 6))
                    if num_classes > 2:
                        shap_vals = shap_values[class_idx]
                    else:
                        shap_vals = shap_values

                    shap.dependence_plot(
                        feature,
                        shap_vals,
                        X_shap_plot,
                        interaction_index='auto',
                        show=False
                    )
                    plt.title(f"SHAP Dependence - {feature}, Class {class_idx} ({set_name})")
                    plt.savefig(
                        os.path.join(dependence_dir, f"class_{class_idx}_{feature}_dependence({set_name}).png"),
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()
                    print(f"[OK] Saved SHAP dependence for {feature}, Class {class_idx} ({set_name})")

        except Exception as e:
            print(f"SHAP plotting failed for {region} ({set_name}): {e}")
    # Free GPU memory after each region
    del model, X_train_val, X_test
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


# --- Save summary table ---
summary_df = pd.DataFrame(all_results)
csv_filename = TABLES_DIR / f"{all_study_name}_regional_results.csv"
summary_df.to_csv(csv_filename, index=False, float_format="%.4f")
print(f"\nRegional summary table saved to '{csv_filename}'")

