import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path
import sys
import shap
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cudf
import cupy as cp

# Define project root

def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"  
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data and region-specific hyperparameters
df = pd.read_csv(PROCESSED_DIR / "monthly_dengue_env_id.csv")
# --- REGION REASSIGNMENT ---
df['Region'] = df['Region'].replace({'Maluku Islands': 'Maluku_Papua', 'Papua': 'Maluku_Papua'})
print("--- DataFrame after Region modification ---")
print(df['Region'].value_counts())
print("-" * 50)
# Read hyperparameters for each region
hyperparams_df = pd.read_csv(FIGURES_DIR / "xgb_region_kfold_hyperparameters.csv")
hyperparams_df.set_index("Region", inplace=True)

# Define variables
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum',
    'evaporative_stress_index', 'aridity_index',
    'temperature_2m_ANOM', 'temperature_2m_min_ANOM', 'temperature_2m_max_ANOM',
    'potential_evaporation_sum_ANOM', 'total_evaporation_sum_ANOM', 'precipitation_ANOM'
]

climate_vars = ['ANOM1+2','ANOM3','ANOM4','ANOM3.4', 'DMI', 'DMI_East']
target = 'Incidence_Rate'

# Sort and create lag features
df = df.sort_values(['YearMonth', 'ID_2'])
for var in env_vars:
    for lag in [1, 2, 3]:
        df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

for var in climate_vars:
    for lag in [1, 3, 6]:
        df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

features = [col for col in env_vars + climate_vars if col in df.columns]
for var in env_vars:
    for lag in [1, 2, 3]:
        if f'{var}_lag{lag}' in df.columns:
            features.append(f'{var}_lag{lag}')
for var in climate_vars:
    for lag in [1, 3, 6]:
        if f'{var}_lag{lag}' in df.columns:
            features.append(f'{var}_lag{lag}')

actual_feature_columns = [col for col in features if col not in ['YearMonth', 'ID_2', 'Year', target]]

# Add Year column
df['Year'] = df['YearMonth'].astype(str).str[:4].astype(int)

# Store all SHAP info for plotting
all_shap_plots = []
region_summary_data = {} # Dictionary to store metrics and IR stats for each region

# Iterate through each region to train models, calculate metrics, and generate SHAP plots
for region_name, params in hyperparams_df.iterrows():
    print(f"\n=== Processing Region: {region_name} ===")

    # Filter data for the current region
    df_region = df[df['Region'] == region_name].copy()

    # Split data into training/validation (before 2023) and test (2023) sets
    df_train_val = df_region[df_region['Year'] < 2023].dropna(subset=actual_feature_columns + [target])
    df_test = df_region[df_region['Year'] == 2023].dropna(subset=actual_feature_columns + [target])

    # Concatenate for full dataset analysis
    df_full = pd.concat([df_train_val, df_test], ignore_index=True)

    # Skip region if insufficient data for training or testing
    if df_train_val.empty or df_test.empty:
        print(f"Skipping {region_name}: insufficient data for training or testing.")
        continue

    # Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
    X_train = cudf.DataFrame(df_train_val[actual_feature_columns])
    y_train = cudf.DataFrame(df_train_val[[target]])
    X_test = cudf.DataFrame(df_test[actual_feature_columns])
    y_test = cudf.DataFrame(df_test[[target]])
    X_full = cudf.DataFrame(df_full[actual_feature_columns])
    y_full = cudf.DataFrame(df_full[[target]])

    # Extract hyperparameters for the current region
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

    # Initialize and train the XGBoost Regressor model
    model = xgboost.XGBRegressor(
        objective='reg:squarederror', # Objective for regression tasks
        tree_method='hist',           # Use histogram-based tree method for faster training
        device='cuda',                # Specify GPU device for training
        random_state=42,              # For reproducibility
        **hyperparams                 # Unpack region-specific hyperparameters
    )
    model.fit(X_train, y_train)

    # Initialize dictionary to store metrics for the current region
    current_region_metrics = {}

    # Evaluate model performance and generate SHAP values for Train/Val, Test, and Full datasets
    for label, X_gpu, y_gpu in [
        ("Train/Val", X_train, y_train),
        ("Test", X_test, y_test),
        ("Full", X_full, y_full)
    ]:
        try:
            # Make predictions and ensure they are non-negative
            preds = model.predict(X_gpu)
            preds = np.maximum(0, cp.asnumpy(preds)) # Convert CuPy array to NumPy and ensure non-negative
            y_true = y_gpu.to_numpy().flatten() # Convert cuDF Series to NumPy array

            # Calculate RMSE and MAE
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)

            # Store metrics for the current dataset split
            current_region_metrics[f'{label} RMSE'] = rmse
            current_region_metrics[f'{label} MAE'] = mae

            # Convert cuDF DataFrame to pandas DataFrame for SHAP
            X_pd = X_gpu.to_pandas()
            # Initialize SHAP explainer and calculate SHAP values
            # Using feature_perturbation="interventional" for TreeExplainer
            explainer = shap.TreeExplainer(model, X_pd, feature_perturbation="interventional")
            shap_vals = explainer(X_pd)

            # Store SHAP values and related info for later plotting
            all_shap_plots.append((shap_vals.values, X_pd, f"{region_name} - {label}", rmse, mae))
        except Exception as e:
            print(f"Failed SHAP or metric calculation for {region_name} - {label}: {e}")

    # Store all calculated metrics and Incidence Rate statistics for the current region
    region_summary_data[region_name] = current_region_metrics
    region_summary_data[region_name]['IR Min'] = df_region[target].min()
    region_summary_data[region_name]['IR Max'] = df_region[target].max()
    region_summary_data[region_name]['IR 25th Quantile'] = df_region[target].quantile(0.25)
    region_summary_data[region_name]['IR 50th Quantile'] = df_region[target].quantile(0.50)
    region_summary_data[region_name]['IR 75th Quantile'] = df_region[target].quantile(0.75)


# Generate SHAP plots and save them to a PDF file
pdf_path = FIGURES_DIR / "xgb_region_kfold_shap.pdf"
with PdfPages(pdf_path) as pdf:
    for shap_vals, X_df, title, rmse, mae in all_shap_plots:
        # Create a new figure for each SHAP plot
        # Adjust figure height based on the number of features for better readability
        fig, ax = plt.subplots(figsize=(10, 8 + len(actual_feature_columns) * 0.25))
        plt.sca(ax) # Set the current axes to the newly created one
        shap.summary_plot(shap_vals, X_df, show=False) # Generate the SHAP summary plot
        ax.set_title(f"{title} | RMSE: {rmse:.2f}, MAE: {mae:.2f}") # Add title with metrics
        pdf.savefig(fig, bbox_inches='tight') # Save the current figure to PDF
        plt.close(fig) # Close the figure to free up memory

print(f"\n✅ All SHAP plots saved to: {pdf_path}")

# --- Generate Region-wise Summary Table ---
print("\n--- Region-wise Summary Table ---")

# Convert the collected summary data into a pandas DataFrame
summary_df = pd.DataFrame.from_dict(region_summary_data, orient='index').T

# Save the DataFrame to a CSV file
# Change filename depending on validation 
csv_filename = TABLES_DIR / "xgb_region_kfold_test.csv"
summary_df.to_csv(csv_filename, index=True, float_format="%.2f")

print(f"Region-wise summary table saved to '{csv_filename}'")
print("-" * 50)