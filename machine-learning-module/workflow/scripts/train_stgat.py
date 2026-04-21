"""
scripts/train_stgat.py
----------------------
Train a Spatiotemporal Graph Attention Network (STGAT) for dengue risk
prediction.

Architecture:  stacked GATConv (spatial) → stacked GRUCell (temporal) → FC (output)
Task:          node-level regression (log incidence rate) or classification (risk class)

Standalone (synthetic data — no preprocessing required):
    python scripts/train_stgat.py --config config/config_stgat.yaml

Via Snakemake (stgat.smk):
    The rule passes --config, --train-data, --val-data, --graph,
    --output-checkpoint and --output-metrics as CLI args.
    When --train-data is omitted the script falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GATConv


# ============================================================
# Model
# ============================================================

class DengueGNN(nn.Module):
    """
    GAT → GRU → Linear.

    Spatial context at each time step is computed by ``num_gat_layers``
    stacked GATConv layers.  That context is fed into ``num_gru_layers``
    stacked GRUCells which maintain a per-node hidden state across the
    look-back sequence.  The final hidden state is projected to an output
    by a two-layer MLP head.

    Parameters
    ----------
    in_channels      : number of input node features
    gat_hidden_dim   : hidden dim of each GAT layer *per head*
    gat_heads        : list of head counts, one entry per GAT layer;
                       the last entry should be 1 so the GRU input size
                       equals gat_hidden_dim without concat ambiguity
    gru_hidden_dim   : GRU hidden state size
    num_gru_layers   : number of stacked GRUCells
    out_channels     : output size (1 for regression, num_classes for classification)
    gat_dropout      : dropout on GAT attention weights
    out_hidden_dim   : intermediate FC dim before the final output projection;
                       pass None to skip the intermediate layer
    out_dropout      : dropout before the final projection
    use_residual     : add skip connections between GAT layers where dims match
    use_projection   : project GAT output to gru_hidden_dim before the GRU
                       (required when gat_hidden_dim != gru_hidden_dim)
    """

    def __init__(
        self,
        in_channels: int,
        gat_hidden_dim: int,
        gat_heads: list[int],
        gru_hidden_dim: int,
        num_gru_layers: int,
        out_channels: int,
        gat_dropout: float = 0.2,
        out_hidden_dim: int | None = 32,
        out_dropout: float = 0.1,
        use_residual: bool = True,
        use_projection: bool = False,
    ):
        super().__init__()

        # --- GAT layers ---
        gat_layers = []
        residual_projections = []

        prev_dim = in_channels
        for layer_idx, n_heads in enumerate(gat_heads):
            is_last = layer_idx == len(gat_heads) - 1
            # Last layer: average heads (concat=False) → output dim = gat_hidden_dim
            # Other layers: concatenate heads (concat=True) → output dim = gat_hidden_dim * n_heads
            concat = not is_last
            gat_layers.append(
                GATConv(
                    in_channels=prev_dim,
                    out_channels=gat_hidden_dim,
                    heads=n_heads,
                    dropout=gat_dropout,
                    concat=concat,
                )
            )
            out_dim = gat_hidden_dim * n_heads if concat else gat_hidden_dim
            # Residual: project prev_dim → out_dim if they differ
            if use_residual and prev_dim != out_dim:
                residual_projections.append(nn.Linear(prev_dim, out_dim, bias=False))
            else:
                residual_projections.append(None)
            prev_dim = out_dim

        self.gat_layers = nn.ModuleList(gat_layers)
        # Store None-able projections in a plain list; wrap non-None ones so
        # parameters are registered
        self._res_projs_raw = residual_projections
        self.res_projs = nn.ModuleList(
            [p for p in residual_projections if p is not None]
        )

        self.use_residual = use_residual
        self.gat_out_dim = prev_dim  # gat_hidden_dim after last layer

        # --- Optional GAT → GRU projection ---
        self.use_projection = use_projection
        if use_projection and self.gat_out_dim != gru_hidden_dim:
            self.gat_to_gru = nn.Linear(self.gat_out_dim, gru_hidden_dim)
            gru_input_dim = gru_hidden_dim
        else:
            self.gat_to_gru = None
            gru_input_dim = self.gat_out_dim

        # --- GRU layers (stacked GRUCells for explicit temporal loop) ---
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(
                input_size=gru_input_dim if i == 0 else gru_hidden_dim,
                hidden_size=gru_hidden_dim,
            )
            for i in range(num_gru_layers)
        ])
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers

        # --- Output head ---
        self.out_dropout = nn.Dropout(out_dropout)
        if out_hidden_dim:
            self.fc_hidden = nn.Linear(gru_hidden_dim, out_hidden_dim)
            self.fc_out = nn.Linear(out_hidden_dim, out_channels)
        else:
            self.fc_hidden = None
            self.fc_out = nn.Linear(gru_hidden_dim, out_channels)

    def forward(
        self,
        x_seq: torch.Tensor,          # [T, N, F]
        edge_index: torch.Tensor,     # [2, E]
        edge_weight: torch.Tensor | None = None,  # [E]
    ) -> torch.Tensor:
        """
        Returns output of shape [N, out_channels].
        Hidden state is reset to zero at the start of each forward call
        (stateless across batches / windows).
        """
        T, N, _ = x_seq.shape
        device = x_seq.device

        # Initialise per-layer hidden states
        h_list = [
            torch.zeros(N, self.gru_hidden_dim, device=device)
            for _ in range(self.num_gru_layers)
        ]

        for t in range(T):
            x_t = x_seq[t]  # [N, F]

            # --- Spatial mixing via stacked GAT ---
            res_proj_iter = iter(self._res_projs_raw)
            for gat_layer in self.gat_layers:
                proj = next(res_proj_iter)
                x_new = F.elu(gat_layer(x_t, edge_index))
                if self.use_residual:
                    residual = proj(x_t) if proj is not None else x_t
                    x_new = x_new + residual
                x_t = x_new  # [N, gat_out_dim]

            # --- Optional projection ---
            if self.gat_to_gru is not None:
                x_t = self.gat_to_gru(x_t)  # [N, gru_hidden_dim]

            # --- Temporal update via stacked GRU ---
            for layer_idx, cell in enumerate(self.gru_cells):
                inp = x_t if layer_idx == 0 else h_list[layer_idx - 1]
                h_list[layer_idx] = cell(inp, h_list[layer_idx])

        # Final hidden state of the top GRU layer
        h_final = h_list[-1]  # [N, gru_hidden_dim]

        # --- Output head ---
        h_final = self.out_dropout(h_final)
        if self.fc_hidden is not None:
            h_final = F.relu(self.fc_hidden(h_final))
            h_final = self.out_dropout(h_final)
        return self.fc_out(h_final)  # [N, out_channels]


def build_model(cfg: dict, in_channels: int) -> DengueGNN:
    """Construct a DengueGNN from a config dict."""
    arch = cfg["model"]
    gat  = arch["gat"]
    gru  = arch["gru"]
    out  = arch["output"]
    proj = arch.get("projection", {})

    task = cfg["target"]["task"]
    if task == "classification":
        out_channels = len(cfg["target"]["class_map"])
    else:
        out_channels = 1

    return DengueGNN(
        in_channels=in_channels,
        gat_hidden_dim=gat["hidden_dim"],
        gat_heads=gat["heads"],
        gru_hidden_dim=gru["hidden_dim"],
        num_gru_layers=gru["num_layers"],
        out_channels=out_channels,
        gat_dropout=gat.get("dropout", 0.2),
        out_hidden_dim=out.get("hidden_dim"),
        out_dropout=out.get("dropout", 0.1),
        use_residual=gat.get("residual", True),
        use_projection=proj.get("enabled", False),
    )


# ============================================================
# Loss functions
# ============================================================

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(pred.squeeze(-1), target, reduction="none")
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred.squeeze(-1) - target).abs()
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(pred.squeeze(-1), target, reduction="none", delta=delta)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def focal_cross_entropy(
    logits: torch.Tensor,   # [N, C]
    targets: torch.Tensor,  # [N] int
    mask: torch.Tensor,     # [N] bool/float
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")  # [N]
    p_t = torch.exp(-ce)
    loss = (1 - p_t) ** gamma * ce
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def get_loss_fn(cfg: dict):
    task = cfg["target"]["task"]
    if task == "regression":
        loss_type = cfg["loss"]["regression"]["type"]
        if loss_type == "masked_mse":
            return masked_mse
        elif loss_type == "masked_mae":
            return masked_mae
        elif loss_type == "masked_huber":
            delta = cfg["loss"]["regression"].get("huber_delta", 1.0)
            return lambda p, t, m: masked_huber(p, t, m, delta=delta)
        else:
            raise ValueError(f"Unknown regression loss: {loss_type}")
    else:
        loss_type = cfg["loss"]["classification"]["type"]
        gamma = cfg["loss"]["classification"].get("focal_gamma", 2.0)
        if loss_type == "focal":
            return lambda p, t, m: focal_cross_entropy(p, t, m, gamma=gamma)
        else:
            return lambda p, t, m: (
                F.cross_entropy(p, t, reduction="none") * m
            ).sum() / (m.sum() + 1e-8)


# ============================================================
# Optimiser and scheduler
# ============================================================

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg["training"]
    lr = tr["learning_rate"]
    wd = tr.get("weight_decay", 0.0)
    name = tr.get("optimizer", "adam").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    sched_cfg = cfg["training"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()
    if sched_type == "plateau":
        p = sched_cfg.get("plateau", {})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=p.get("patience", 10),
            factor=p.get("factor", 0.5),
            min_lr=p.get("min_lr", 1e-6),
        )
    elif sched_type == "cosine":
        p = sched_cfg.get("cosine", {})
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=p.get("t_max", 50),
            eta_min=p.get("eta_min", 1e-6),
        )
    elif sched_type == "step":
        p = sched_cfg.get("step", {})
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=p.get("step_size", 30),
            gamma=p.get("gamma", 0.5),
        )
    else:
        return None


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model state in memory so it can be restored after stopping.
    """

    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy keeps the best weights even if training continues
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# Synthetic data (no preprocessing required)
# ============================================================

class Batch(NamedTuple):
    x_seq:       torch.Tensor   # [T, N, F]
    edge_index:  torch.Tensor   # [2, E]
    edge_weight: torch.Tensor   # [E]
    y:           torch.Tensor   # [N] float (regression) or long (classification)
    mask:        torch.Tensor   # [N] float  — 1 = valid report, 0 = missing


def make_synthetic_data(cfg: dict, device: torch.device) -> tuple[Batch, Batch]:
    """
    Generate random tensors with the correct shapes and dtypes.
    Useful for smoke-testing the architecture and training loop before
    real data preprocessing is implemented.

    Returns (train_batch, val_batch).
    """
    torch.manual_seed(cfg["training"].get("random_state", 42))

    T = cfg["sequence"]["lookback_steps"]
    N = 50        # small node count for fast iteration
    F = 20        # approximate feature count
    missing_rate = 0.2

    task = cfg["target"]["task"]

    def _make_batch() -> Batch:
        x_seq = torch.randn(T, N, F, device=device)
        # Random sparse graph: ~5 neighbours per node
        src, dst = [], []
        for i in range(N):
            neighbours = torch.randperm(N)[:5].tolist()
            for j in neighbours:
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_weight = torch.rand(edge_index.shape[1], device=device)
        mask = (torch.rand(N, device=device) > missing_rate).float()
        if task == "classification":
            n_classes = len(cfg["target"]["class_map"])
            y = torch.randint(0, n_classes, (N,), device=device)
        else:
            y = torch.rand(N, device=device)
        return Batch(x_seq, edge_index, edge_weight, y, mask)

    return _make_batch(), _make_batch()


# ============================================================
# One-epoch helpers
# ============================================================

def _compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    task: str,
    log_transform: bool = False,
) -> dict:
    """Return a small dict of interpretable metrics computed on masked nodes only."""
    valid = mask.bool()
    if valid.sum() == 0:
        return {}
    p = pred[valid]
    t = target[valid]
    metrics: dict = {}
    if task == "regression":
        p = p.squeeze(-1)
        metrics["mae"]  = (p - t).abs().mean().item()
        metrics["rmse"] = ((p - t) ** 2).mean().sqrt().item()
        if log_transform:
            p_ir = torch.expm1(p.clamp(min=0))
            t_ir = torch.expm1(t.clamp(min=0))
            metrics["mae_ir"]  = (p_ir - t_ir).abs().mean().item()
            metrics["rmse_ir"] = ((p_ir - t_ir) ** 2).mean().sqrt().item()
    else:
        preds_cls = p.argmax(dim=-1)
        metrics["accuracy"] = (preds_cls == t).float().mean().item()
    return metrics


def train_epoch(
    model: nn.Module,
    batches: "Batch | list[Batch]",
    optimizer: torch.optim.Optimizer,
    loss_fn,
    grad_clip: float | None,
    task: str,
    log_transform: bool = False,
) -> tuple[float, dict]:
    """
    Train for one epoch.  `batches` may be a single Batch (synthetic / legacy)
    or a list of Batch objects (real sliding-window data).  When a list is
    passed, windows are shuffled and iterated; gradients are accumulated and
    applied once per epoch.
    """
    model.train()
    if isinstance(batches, Batch):
        batches = [batches]
    order = list(range(len(batches)))
    random.shuffle(order)

    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    optimizer.zero_grad()
    for idx in order:
        b = batches[idx]
        pred = model(b.x_seq, b.edge_index, b.edge_weight)
        loss = loss_fn(pred, b.y, b.mask)
        (loss / len(order)).backward()
        total_loss += loss.item()
        with torch.no_grad():
            all_preds.append(pred.detach())
            all_targets.append(b.y)
            all_masks.append(b.mask)

    if grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    combined_pred   = torch.cat(all_preds)
    combined_target = torch.cat(all_targets)
    combined_mask   = torch.cat(all_masks)
    metrics = _compute_metrics(combined_pred, combined_target, combined_mask, task, log_transform)
    return total_loss / len(order), metrics


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    batches: "Batch | list[Batch]",
    loss_fn,
    task: str,
    log_transform: bool = False,
) -> tuple[float, dict]:
    """Average loss and metrics over all validation batches."""
    model.eval()
    if isinstance(batches, Batch):
        batches = [batches]

    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    for b in batches:
        pred = model(b.x_seq, b.edge_index, b.edge_weight)
        total_loss += loss_fn(pred, b.y, b.mask).item()
        all_preds.append(pred)
        all_targets.append(b.y)
        all_masks.append(b.mask)

    combined_pred   = torch.cat(all_preds)
    combined_target = torch.cat(all_targets)
    combined_mask   = torch.cat(all_masks)
    metrics = _compute_metrics(combined_pred, combined_target, combined_mask, task, log_transform)
    return total_loss / len(batches), metrics


# ============================================================
# Main training loop
# ============================================================

def fit(
    model: nn.Module,
    train_batches: "Batch | list[Batch]",
    val_batches: "Batch | list[Batch]",
    cfg: dict,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict:
    """
    Run the full training loop.  Returns a dict of final metrics.
    Accepts either a single Batch (synthetic) or list[Batch] (real data windows).
    """
    wandb_cfg     = cfg.get("wandb", {})
    use_wandb     = wandb_cfg.get("enabled", False)
    task          = cfg["target"]["task"]
    log_transform = cfg["target"].get("log_transform", False)

    if use_wandb:
        import os
        import wandb
        api_key = wandb_cfg.get("api_key")
        api_key_file = wandb_cfg.get("api_key_file")
        if api_key_file:
            api_key = Path(api_key_file).read_text().strip()
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            group=cfg.get("experiment", cfg.get("arch", "stgat")),
            job_type="train",
            config=cfg,
            tags=wandb_cfg.get("tags", []),
        )
        if wandb_cfg.get("log_gradients"):
            wandb.watch(model, log="all")

    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)
    loss_fn    = get_loss_fn(cfg)
    es_cfg     = cfg["training"].get("early_stopping", {})
    stopper    = EarlyStopping(
        patience=es_cfg.get("patience", 25),
        min_delta=es_cfg.get("min_delta", 1e-4),
    )
    grad_clip  = cfg["training"].get("gradient_clip")
    n_epochs   = cfg["training"]["epochs"]

    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_batches, optimizer, loss_fn, grad_clip, task, log_transform
        )
        val_loss, val_metrics = val_epoch(model, val_batches, loss_fn, task, log_transform)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        log = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(log)

        if epoch % 10 == 0 or epoch == 1:
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "epoch")
            print(f"[epoch {epoch:04d}]  {metric_str}")

        if use_wandb:
            import wandb
            wandb.log(log)

        if stopper.step(val_loss, model):
            print(f"[early stop] No val improvement for {stopper.patience} epochs. "
                  f"Best val_loss={stopper.best_loss:.4f}")
            break

    stopper.restore_best(model)
    print(f"[fit] Restored best weights (val_loss={stopper.best_loss:.4f})")

    # Save checkpoint
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[fit] Checkpoint → {checkpoint_path}")

    # Save metrics
    final_metrics = history[-1] if history else {}
    final_metrics["best_val_loss"] = stopper.best_loss
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fh:
            json.dump(final_metrics, fh, indent=2)
        print(f"[fit] Metrics → {metrics_path}")

    if use_wandb:
        import wandb
        wandb.log({"best_val_loss": stopper.best_loss})
        wandb.finish()

    return final_metrics


# ============================================================
# Real data loading
# ============================================================

def load_batches(path: str, device: torch.device) -> list[Batch]:
    """
    Load a .pt file produced by preprocess_stgat.py and return a list of
    Batch objects — one per sliding window.

    Expected keys in the .pt dict:
        windows      [W, L, N, F]  float32
        y            [W, N]        float32
        mask         [W, N]        float32
        edge_index   [2, E]        int64
        edge_weight  [E]           float32
    """
    data        = torch.load(path, map_location=device, weights_only=False)
    windows     = data["windows"].to(device)      # [W, L, N, F]
    y_all       = data["y"].to(device)            # [W, N]
    mask_all    = data["mask"].to(device)         # [W, N]
    edge_index  = data["edge_index"].to(device)
    edge_weight = data.get(
        "edge_weight",
        torch.ones(data["edge_index"].shape[1], device=device),
    )

    batches = []
    for i in range(windows.shape[0]):
        batches.append(Batch(
            x_seq       = windows[i],       # [L, N, F]
            edge_index  = edge_index,
            edge_weight = edge_weight,
            y           = y_all[i],         # [N]
            mask        = mask_all[i],      # [N]
        ))
    return batches


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STGAT for dengue risk prediction")
    p.add_argument("--config",            required=True,  help="Path to config_stgat.yaml")
    p.add_argument("--train-data",        default=None,   help="Path to train tensor file (.pt)")
    p.add_argument("--val-data",          default=None,   help="Path to val tensor file (.pt)")
    p.add_argument("--graph",             default=None,   help="Path to graph.pkl")
    p.add_argument("--best-params",       default=None,   help="Path to best_params.json from tune_stgat")
    p.add_argument("--output-checkpoint", default=None,   help="Where to save model weights (.pt)")
    p.add_argument("--output-metrics",    default=None,   help="Where to save metrics (.json)")
    p.add_argument("--no-wandb",          action="store_true", help="Disable W&B logging regardless of config")
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def apply_best_params(cfg: dict, params: dict) -> None:
    """Merge tune_stgat best_params.json into a loaded config dict (in-place)."""
    tr  = cfg["training"]
    gat = cfg["model"]["gat"]
    gru = cfg["model"]["gru"]
    out = cfg["model"]["output"]

    if "learning_rate"       in params: tr["learning_rate"]  = params["learning_rate"]
    if "weight_decay"        in params: tr["weight_decay"]   = params["weight_decay"]
    if "gat_hidden_dim"      in params: gat["hidden_dim"]    = params["gat_hidden_dim"]
    # tune_stgat writes "gat_heads_first_layer"; accept both spellings
    heads_key = next((k for k in ("gat_heads_first_layer", "gat_heads_first") if k in params), None)
    if heads_key is not None:
        n_layers     = len(gat["heads"])
        gat["heads"] = [params[heads_key]] * (n_layers - 1) + [1]
    if "gru_hidden_dim"      in params: gru["hidden_dim"]    = params["gru_hidden_dim"]
    # tune_stgat writes a single "dropout" key; map it to both gat and output dropout
    if "dropout"             in params:
        gat["dropout"] = params["dropout"]
        out["dropout"] = params["dropout"]
    if "gat_dropout"         in params: gat["dropout"]       = params["gat_dropout"]
    if "out_dropout"         in params: out["dropout"]       = params["out_dropout"]


def main():
    # --- Handle both standalone (argparse) and Snakemake invocation ---
    if "snakemake" in dir():
        # Called by Snakemake via shell: — reconstruct a compatible namespace
        sm = snakemake  # noqa: F821
        args = argparse.Namespace(
            config=sm.input.config_file,
            train_data=getattr(sm.input, "train_data", None),
            val_data=getattr(sm.input, "val_data", None),
            graph=getattr(sm.input, "graph", None),
            best_params=getattr(sm.input, "best_params", None),
            output_checkpoint=getattr(sm.output, "checkpoint", None),
            output_metrics=getattr(sm.output, "metrics", None),
            device=f"cuda:{sm.params.cuda_visible_devices}" if hasattr(sm.params, "cuda_visible_devices") else "cpu",
        )
    else:
        args = parse_args()

    # --- Load config ---
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # --- Override W&B if requested ---
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    # --- Apply best params from tune step (if provided) ---
    if args.best_params is not None:
        import json as _json
        with open(args.best_params) as fh:
            bp = _json.load(fh)
        bp.pop("best_val_loss", None)   # metadata key, not a hyperparam
        apply_best_params(cfg, bp)
        print(f"[main] Applied best params from {args.best_params}: {bp}")

    torch.manual_seed(cfg["training"].get("random_state", 42))
    device = torch.device(args.device)
    print(f"[main] Device: {device}")

    # --- Data loading ---
    if args.train_data is not None:
        print(f"[main] Loading real data from {args.train_data} ...")
        train_batches = load_batches(args.train_data, device)
        val_batches   = load_batches(args.val_data,   device)
        in_channels   = train_batches[0].x_seq.shape[2]
        n_nodes       = train_batches[0].x_seq.shape[1]
        lookback      = train_batches[0].x_seq.shape[0]
        print(f"[main] Windows — train: {len(train_batches)}  val: {len(val_batches)}")
    else:
        print("[main] No data files provided — using synthetic data.")
        train_batch, val_batch = make_synthetic_data(cfg, device)
        train_batches = train_batch
        val_batches   = val_batch
        in_channels   = train_batch.x_seq.shape[2]
        n_nodes       = train_batch.x_seq.shape[1]
        lookback      = train_batch.x_seq.shape[0]

    print(f"[main] Nodes: {n_nodes}  Features: {in_channels}  Lookback: {lookback} steps")

    # --- Build model ---
    model = build_model(cfg, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Model parameters: {n_params:,}")

    # --- Train ---
    fit(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        cfg=cfg,
        checkpoint_path=Path(args.output_checkpoint) if args.output_checkpoint else None,
        metrics_path=Path(args.output_metrics) if args.output_metrics else None,
    )


if __name__ == "__main__":
    main()
