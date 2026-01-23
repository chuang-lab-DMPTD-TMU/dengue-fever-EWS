import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # New classification metrics
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import cudf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cupy as cp
import optuna

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KT0822"

# --- Run sweep for the entire nation ---
n_trials_for_national_study = 50 # Number of trials for the national study

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
study_name = f"lstm-national-regression-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
print(f"Name of study: {study_name}")

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
# Load the data once outside the train function for efficiency
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" / "monthly_dengue_env_id_updated.csv")

# --- REGION REASSIGNMENT (Keep this for consistency) ---
df['Region_Group'] = df['Region'].replace({
    'Maluku Islands': 'Maluku-Papua', 
    'Papua': 'Maluku-Papua'})
# df['Risk_Category'] = df['Risk_Category'].replace({
#     'Zero-risk': 0,
#     'Low-risk': 0,
#     'Medium-risk': 1,
#     'High-risk': 2}).infer_objects(copy=False)
# df['Risk_Category'] = df['Risk_Category'].astype('int32')
print("--- DataFrame after Region_Group creation ---")
print(df['Region_Group'].value_counts())
print("-" * 50)
# Create a list of regions to iterate over
regions_to_model = ['Maluku-Papua']


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



# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
variable_columns = []
for var in env_vars + climate_vars:
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
target = 'Incidence_Rate'
metadata_columns = ['YearMonth', 'ID_2', 'Region_Group']
# Final feature list excluding metadata and target
variable_columns = [
    col for col in variable_columns
    if col not in [metadata_columns, target]
]
print("Starting training with the following columns:")

print("--- Target Column ---")
print([target])
print("-" * 50)
print("--- Metadata Columns ---")
print(metadata_columns)
print("-" * 50)
print("--- Variable Columns ---")
print(variable_columns)



# Data for hyperparameter tuning
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=variable_columns + [target])
# Data for final, unseen test
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=variable_columns + [target])

print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

X_train_val_national = df_train_val_national[variable_columns]
y_train_val_national = df_train_val_national[target]
X_test_national = df_test_national[variable_columns]
y_test_national = df_test_national[target]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (batch, seq_len, hidden)
        last_step_out = lstm_out[:, -1, :]         # take last timestep
        last_step_out = self.dropout(last_step_out)
        output = self.fc(last_step_out)            # regression output
        return output


def get_splits(df, validation_strategy):
    """
    Generate train/validation splits dynamically based on the chosen strategy.
    """
    splits = []

    if validation_strategy == 'kfold':
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)
        for train_index, test_index in kf.split(df):
            splits.append((np.array(train_index), np.array(test_index)))

    elif validation_strategy == 'walk':
        initial_train_months = 60
        test_window = 6
        unique_time_periods = df['YearMonth'].unique()
        n_time_periods = len(unique_time_periods)
        n_walks = (n_time_periods - initial_train_months) // test_window

        for i in range(n_walks):
            train_end_period_idx = initial_train_months + i * test_window
            test_start_period_idx = train_end_period_idx
            test_end_period_idx = test_start_period_idx + test_window

            if test_end_period_idx > n_time_periods:
                break

            train_end_time = unique_time_periods[train_end_period_idx - 1]
            test_start_time = unique_time_periods[test_start_period_idx]
            test_end_time = unique_time_periods[test_end_period_idx - 1]

            train_indices_pd = df.loc[df['YearMonth'] <= train_end_time].index
            test_indices_pd = df.loc[
                (df['YearMonth'] >= test_start_time) &
                (df['YearMonth'] <= test_end_time)
            ].index

            splits.append((train_indices_pd.to_numpy(), test_indices_pd.to_numpy()))

    else:
        raise ValueError(f"Unsupported validation strategy: {validation_strategy}")

    print(f"Pre-calculated {len(splits)} splits for {validation_strategy} validation.")
    return splits


def objective(trial, X, y, splits, validation_strategy):
    """
    Objective function for Optuna to minimize.
    Search space + epochs adapt to the validation strategy.
    """

    # --- Hyperparameter Search Space ---
    if validation_strategy == "kfold":
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 128),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        }
        num_epochs = trial.suggest_int('n_epochs', 15, 40)

    elif validation_strategy == "walk":
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 16, 64),
            'num_layers': trial.suggest_int('num_layers', 1, 2),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        }
        num_epochs = trial.suggest_int('n_epochs', 5, 25)

    else:
        raise ValueError(f"Unsupported validation strategy: {validation_strategy}")

    all_preds, all_true = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    for train_index, test_index in splits:
        train_dataset = TensorDataset(X_tensor[train_index], y_tensor[train_index])
        test_dataset = TensorDataset(X_tensor[test_index], y_tensor[test_index])

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

        model = NeuralNetwork(
            input_size=X.shape[1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            fold_preds, fold_true = [], []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                fold_preds.append(outputs.cpu().numpy())
                fold_true.append(targets.cpu().numpy())

            all_preds.extend(np.concatenate(fold_preds).flatten())
            all_true.extend(np.concatenate(fold_true).flatten())

    if all_true:
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))
        print(f"Trial finished with RMSE: {rmse:.4f}")
        return rmse
    else:
        return float('inf')

for region in regions_to_model:
    print(f"\n--- Starting Study and Hyperparameter Retrieval for Region: {region}")

    # Filter data for this region
    df_region = df_train_val_national[df_train_val_national['Region_Group'] == region]

    # Generate CV splits for this region
    splits = get_splits(df_region, validation_strategy)

    # Define study name (region-specific)
    study_name = f"lstm-{region}-regressor-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True
    )

    print(f"Optuna study created/loaded: {study_name}.db")
    print(f"Starting {n_trials_for_national_study} trials...")

    # Prepare features/labels for this region
    X_region = df_region[variable_columns]  # replace target_column
    y_region = df_region[target]

    # Optimize for this region
    study.optimize(
        lambda trial: objective(
            trial,
            X=X_region,
            y=y_region,
            splits=splits,
            validation_strategy=validation_strategy
        ),
        n_trials=n_trials_for_national_study,
        n_jobs=-1
    )

    print(f"\nStudy completed for {region}.")

    # --- Best params ---
    best_params = study.best_params
    best_value = study.best_value
    print(f"\nBest RMSE for {region}: {best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save best hyperparameters for this region
    best_hypers_df = pd.DataFrame([{
        'Region': region,
        'best_rmse': best_value,
        **best_params
    }])
    best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
    best_hypers_df.to_csv(best_hypers_csv_path, index=False)
    print(f"\nSaved best hyperparameters for {region} to {best_hypers_csv_path}")

print("\n--- Regional Study and Hyperparameter Retrieval Completed ---")
