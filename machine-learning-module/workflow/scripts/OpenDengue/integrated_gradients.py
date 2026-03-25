import torch
import wandb
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from model_def import DengueGNN

# Load the WandB Run ID to upload plots to the same dashboard
with open(snakemake.input.run_id, 'r') as f:
    run_id = f.read()

wandb.init(project="dengue-forecasting", id=run_id, resume="must")

# Load model and data
model = DengueGNN(...)
model.load_state_dict(torch.load(snakemake.input.model))
x_seq = torch.load(snakemake.input.data)

# IG Analysis
ig = IntegratedGradients(model)
# target=0 is 'Infection Risk'
attr = ig.attribute(x_seq, target=0, additional_forward_args=(edge_index))

# Visualize: Average attribution across all 400 cities
# To see which features/months mattered most overall
avg_attr = attr.mean(dim=1).detach().numpy() # [Time, Features]

plt.figure(figsize=(12, 8))
plt.imshow(avg_attr.T, aspect='auto')
plt.xlabel("Months in Past")
plt.ylabel("Features")
plt.colorbar(label="Attribution Score")
plt.savefig(snakemake.output[0])

# Log to WandB
wandb.log({"interpretability/feature_importance_over_time": wandb.Image(plt)})
wandb.finish()