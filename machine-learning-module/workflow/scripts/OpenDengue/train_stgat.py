import torch
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DengueGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels=1):
        super(DengueGNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Spatial Layer: GAT
        # We use GAT to let cities "talk" to neighbors
        self.gat = GATConv(in_channels, hidden_dim, heads=1)
        
        # 2. Temporal Layer: GRU
        # This maintains the latent memory for each city
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 3. Output Head: Linear
        # Projects the 64-dim latent vector to a single Risk Score
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, x_seq, edge_index):
        # x_seq shape: [Time, Nodes, Features] -> [24, 400, 8]
        seq_len, num_nodes, _ = x_seq.size()
        
        # Initialize hidden state (The Latent Representation)
        h = torch.zeros(num_nodes, self.hidden_dim).to(x_seq.device)
        
        # Process the sequence month-by-month
        for t in range(seq_len):
            # Spatial mixing: Each city looks at its neighbors
            # x_t shape: [400, 8] -> [400, 64]
            xt = x_seq[t]
            spatial_context = F.relu(self.gat(xt, edge_index))
            
            # Temporal update: Update the city's individual memory
            # h shape: [400, 64]
            h = self.gru(spatial_context, h)
            
        # Final prediction based on the last hidden state
        # out shape: [400, 1]
        out = self.fc(h)
        return out

# --- The Masked Loss Function ---
def masked_mse_loss(preds, targets, target_mask):
    """
    Only calculates loss for cities that have a valid ground truth report.
    """
    # MSE for all nodes
    loss = F.mse_loss(preds, targets, reduction='none')
    
    # Zero out the loss for missing reports
    masked_loss = loss * target_mask
    
    # Return average loss over existing reports only
    return masked_loss.sum() / (target_mask.sum() + 1e-8)


# 1. Initialize WandB
# Snakemake 'params' are accessible via snakemake.params
wandb.init(
    project="dengue-forecasting-400-cities",
    config={
        "learning_rate": snakemake.params.lr,
        "epochs": snakemake.params.epochs,
        "hidden_dim": snakemake.params.hidden_dim,
        "architecture": "GAT-GRU"
    }
)

def train():
    # Load data from Snakemake inputs
    x_seq = torch.load(snakemake.input.features) # [Time, Cities, Features]
    edge_index = torch.load(snakemake.input.adjacency)
    
    model = DengueGNN(in_channels=x_seq.shape[2], hidden_dim=wandb.config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Optional: Watch the model to track gradients and topology
    wandb.watch(model, log="all")

    for epoch in range(wandb.config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        # x_seq: [24, 400, F], y_true: [400, 1], mask: [400, 1]
        preds = model(x_seq, edge_index)
        
        loss = masked_mse_loss(preds, y_true, target_mask)
        
        loss.backward()
        optimizer.step()
        
        # 2. Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": loss.item(),
            "learning_rate": wandb.config.learning_rate
        })

    # Save final model to Snakemake output path
    torch.save(model.state_dict(), snakemake.output.model_checkpoints)
    wandb.finish()

if __name__ == "__main__":
    train()