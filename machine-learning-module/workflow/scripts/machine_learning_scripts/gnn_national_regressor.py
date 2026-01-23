import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignalBatch, StaticGraphTemporalSignal
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from pathlib import Path

# IMPORTANT: You need to import the DataLoader
from torch_geometric.loader import DataLoader

# Set whether to use land use features
USE_LANDUSE_FEATURES = True
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

# --- 1. Data Loading and Preprocessing ---
# IMPORTANT: Use the incidence rate column directly for regression.
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" / "monthly_dengue_env_id_class_log.csv")
# Remove the classification-specific columns
if 'Risk_Category' in df.columns:
    df = df.drop(columns=['Risk_Category'])

print("-" * 50)
# Create a list of regions to iterate over
regions_to_model = df['Region_Group'].unique()

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
    if col not in [metadata_columns, target]]


# Normalize numerical features. We'll use a simple min-max scaler for demonstration.
for col in variable_columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Create a mapping for regions to node indices
unique_regions = df['ID_2'].unique()
node_mapping = {region: i for i, region in enumerate(unique_regions)}
num_nodes = len(unique_regions)
df['node_idx'] = df['ID_2'].map(node_mapping)

# --- 2. Create the Graph Structure (Temporal Snapshots) ---

# We need to construct a sequence of graph snapshots.
# A simple approach for a *fully connected* graph with learned weights (GAT-like)
# is to define an edge between all nodes, and let the model learn the importance.

# Create a fully connected graph edge list.
# 1. Create fully connected graph
source_nodes, target_nodes = [], []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            source_nodes.append(i)
            target_nodes.append(j)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

# 2. Add self-loops (important!)
from torch_geometric.utils import add_self_loops
edge_index, edge_weight = add_self_loops(
    edge_index,
    edge_attr=edge_weight,
    fill_value=1.0,
    num_nodes=num_nodes
)
edge_index = edge_index.long()
edge_weight = edge_weight.float()


# Group data by time to create snapshots
grouped_data = df.groupby('YearMonth')
all_features, all_targets = [], []
timestamps = []

for date, group in grouped_data:
    group = group.sort_values('node_idx')

    # Fill missing nodes
    if len(group) != num_nodes:
        full_df = pd.DataFrame({'node_idx': range(num_nodes)})
        group = pd.merge(full_df, group, on='node_idx', how='left')

    # Ensure all feature columns and target are numeric
    group[variable_columns] = group[variable_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    group[target] = pd.to_numeric(group[target], errors='coerce').fillna(0)

    # Convert to NumPy arrays
    features = group[variable_columns].values.astype(np.float32)
    target_values = group[target].values.astype(np.float32).reshape(-1, 1)

    all_features.append(features)
    all_targets.append(target_values)
    timestamps.append(date)


# Create the temporal dataset object
dataset = StaticGraphTemporalSignal(
    edge_index=edge_index,
    edge_weight=edge_weight,
    features=all_features,
    targets=all_targets
)
snapshot = dataset[0]         # returns PyG Data object
print("Type of snapshot:", type(snapshot))
print("Length of snapshot:", len(snapshot))

for i, elem in enumerate(snapshot):
    print(f"Element {i}: type={type(elem)}")

print("Type of snapshot:", type(snapshot))
print("Length of snapshot:", len(snapshot))


x_seq = snapshot.x             # [N, F]
y_seq = snapshot.y             # [N, 1]
snapshot.edge_weight = dataset.edge_weight

# Inspect first snapshot
print("x[0] shape:", x_seq[0].shape)
print("y[0] shape:", y_seq[0].shape)
print("edge_index shape:", snapshot.edge_index)
print("edge_weight shape:", snapshot.edge_weight.shape)


# --- 3. Split the Data into Training and Testing Sets ---

# You would then split your dataset into train and test
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)


# ----- Safety checks before training -----
# Ensure edge_index indices are within [0, num_nodes-1]
if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
    raise ValueError(f"edge_index contains indices outside [0, {num_nodes-1}] "
                     f"(min={edge_index.min().item()}, max={edge_index.max().item()})")

# --- 4. Define the GNN Model for Regression ---

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features):
        super(TemporalGNN, self).__init__()
        self.recurrent_layer = A3TGCN(
            in_channels=node_features,
            out_channels=32, # Hidden layer size
            periods=1,
        )
        # Final linear layer for regression with output size 1
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        h = self.recurrent_layer(x, edge_index)
        h = h.view(-1, h.shape[2]) # Flatten to [batch_size * num_nodes, out_channels]
        out = self.linear(h)
        return out.squeeze(1) # Squeeze to get a 1D tensor of predictions

# Check for NaNs or Infs in normalized data
if np.isnan(all_features[0]).any() or np.isinf(all_features[0]).any():
    raise ValueError("Features contain NaN or Inf values after normalization.")

# Check for isolated nodes
unique_nodes_in_edge_index = torch.unique(edge_index)
if len(unique_nodes_in_edge_index) != num_nodes:
    print(f"WARNING: Found isolated nodes! "
          f"Nodes in edge_index: {len(unique_nodes_in_edge_index)}, "
          f"Total nodes: {num_nodes}")
    print("Consider adding self-loops to the graph.")
print("--- END DEBUGGING: Data Sanity Checks ---")
print("-" * 50)

# --- 5. Training and Evaluation Loop ---

print("edge_index min/max:", edge_index.min().item(), edge_index.max().item())
print("num_nodes:", num_nodes)
print("edge_index shape:", edge_index.shape)
print("edge_weight shape:", edge_weight.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_features = len(variable_columns)

# Initialize model, optimizer, loss
model = TemporalGNN(node_features=num_features).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# --- Training loop ---
def train():
    model.train()
    total_loss = 0.0
    num_snapshots = len(train_dataset.features)

    for t in range(num_snapshots):
        # Get snapshot
        x_snapshot, y_snapshot = train_dataset[t].x, train_dataset[t].y
        
        # Convert to tensors
        # Add a temporal dimension (batch of 1 snapshot)
        x = x_snapshot.clone().detach().unsqueeze(0).float()
        y = torch.tensor(y_snapshot, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 1]

        edge_index_t = edge_index.to(device)
        edge_weight_t = edge_weight.to(device)

        optimizer.zero_grad()
        out = model(x, edge_index_t)  # [1, N, 1]

        # Compute loss
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() 

    avg_loss = total_loss / num_snapshots
    return avg_loss



# --- Testing / evaluation loop ---
def test():
    model.eval()
    all_preds = []
    all_targets = []

    num_snapshots = len(test_dataset.features)

    with torch.no_grad():
        for t in range(num_snapshots):
            x, y = test_dataset[t].x, test_dataset[t].y
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            edge_index_t = edge_index.to(device)

            out = model(x, edge_index_t)

            all_preds.append(out.cpu().view(-1))
            all_targets.append(y.cpu().view(-1))

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return mse, mae, r2


# --- Run training ---
epochs = 100
for epoch in range(1, epochs + 1):
    avg_loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Avg Loss: {avg_loss:.4f}")

# --- Evaluate ---
mse, mae, r2 = test()
print("\n" + "="*50)
print("Final Model Evaluation on Test Data")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print("="*50)

# You can also add code to save the model:
# torch.save(model.state_dict(), "gnn_regression_model.pt")