"""
scripts/train_bayes_stgat.py
----------------------------
Train a Bayesian Spatiotemporal Graph Attention Network (BayesianSTGAT).

Architecture:  BayesianGATConv (spatial) → BayesianGRUCell (temporal)
               → BayesianLinear (output MLP)
Inference:     Mean-field variational inference with reparameterization trick
Loss:          ELBO = masked_mse (likelihood) + β × KL divergence
KL annealing:  β ramped linearly from 0 → kl_weight over anneal_fraction of total epochs
Early stopping: monitors val_mse only (not total ELBO — ELBO rises during annealing
               even when the model learns, so it would fire too early)

All data loading utilities (Batch, load_batches, make_synthetic_data) and
shared helpers (build_optimizer, build_scheduler, masked_mse, _compute_metrics)
are imported from train_stgat to avoid duplication.
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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_stgat import (
    Batch,
    load_batches,
    make_synthetic_data,
    build_optimizer,
    build_scheduler,
    masked_mse,
    masked_mae,
    masked_huber,
    _compute_metrics,
)

scaler = GradScaler()
# ============================================================
# Bayesian layers
# ============================================================

class BayesianLinear(nn.Module):
    """
    Linear layer with mean-field variational weights.
      Posterior: q(w) = N(mu, softplus(rho)^2)
      Prior:     p(w) = N(0, prior_sigma^2)
    Always samples weights on every forward pass (reparameterization trick).
    MC averaging is handled externally by the training/eval loops.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        rho_init: float = -3.0,
        mu_init: str = "normal",
    ):
        super().__init__()
        self.prior_sigma = prior_sigma

        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features))
        self.bias_rho   = nn.Parameter(torch.full((out_features,), rho_init))

        if mu_init == "xavier":
            nn.init.xavier_uniform_(self.weight_mu)
        elif mu_init == "kaiming":
            nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        else:
            nn.init.normal_(self.weight_mu, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_std = F.softplus(self.weight_rho)
        b_std = F.softplus(self.bias_rho)
        weight = self.weight_mu + w_std * torch.randn_like(self.weight_mu)
        bias   = self.bias_mu   + b_std * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(w) || p(w)) summed over all weight and bias elements."""
        w_std = F.softplus(self.weight_rho)
        b_std = F.softplus(self.bias_rho)

        def _kl(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            # KL(N(mu, std^2) || N(0, prior_sigma^2))
            var_ratio = (std / self.prior_sigma) ** 2
            return 0.5 * (var_ratio + (mu / self.prior_sigma) ** 2 - 1.0
                          - torch.log(var_ratio + 1e-8))

        return _kl(self.weight_mu, w_std).sum() + _kl(self.bias_mu, b_std).sum()


class BayesianGATConv(MessagePassing):
    """
    GAT layer with a variational (BayesianLinear) feature projection.
    Attention scoring (att_src / att_dst) uses deterministic parameters —
    variational attention is numerically unstable during early KL annealing.
    Samples new weights on every forward call via BayesianLinear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        prior_sigma: float = 1.0,
        rho_init: float = -3.0,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.concat = concat

        self.lin = BayesianLinear(
            in_channels, heads * out_channels, prior_sigma, rho_init
        )
        # Deterministic attention parameters
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att_src.view(heads, out_channels).unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.view(heads, out_channels).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,           # [N, in_channels]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:
        N = x.size(0)
        h = self.lin(x).view(N, self.heads, self.out_channels)  # [N, H, D]

        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [N, H]

        # propagate: alpha=(alpha_src, alpha_dst) → alpha_j=src coeff, alpha_i=dst coeff
        out = self.propagate(edge_index, h=h, alpha=(alpha_src, alpha_dst))  # [N, H, D]

        if self.concat:
            return out.view(N, self.heads * self.out_channels)
        return out.mean(dim=1)  # [N, D]

    def message(
        self,
        h_j: torch.Tensor,      # [E, H, D]
        alpha_i: torch.Tensor,  # [E, H] — dst attention
        alpha_j: torch.Tensor,  # [E, H] — src attention
        index: torch.Tensor,    # [E]    — target node indices
        size_i: int,
    ) -> torch.Tensor:
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=0.2)
        alpha = pyg_softmax(alpha, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return h_j * alpha.unsqueeze(-1)  # [E, H, D]

    def kl_divergence(self) -> torch.Tensor:
        return self.lin.kl_divergence()


class BayesianGRUCell(nn.Module):
    """
    GRU cell with variational weights on all three gate paths.
    Reset (r) and update (z) gates share a single BayesianLinear for efficiency;
    the new-gate (n) input and hidden paths are separate to match standard GRU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        prior_sigma: float = 1.0,
        rho_init: float = -3.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # r and z gates computed jointly from [x, h]
        self.lin_rz  = BayesianLinear(input_size + hidden_size, 2 * hidden_size, prior_sigma, rho_init)
        self.lin_n_x = BayesianLinear(input_size,  hidden_size, prior_sigma, rho_init)
        self.lin_n_h = BayesianLinear(hidden_size, hidden_size, prior_sigma, rho_init)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.lin_rz(torch.cat([x, h], dim=-1)))
        r, z = gates.chunk(2, dim=-1)
        n = torch.tanh(self.lin_n_x(x) + r * self.lin_n_h(h))
        return (1.0 - z) * n + z * h

    def kl_divergence(self) -> torch.Tensor:
        return (self.lin_rz.kl_divergence()
                + self.lin_n_x.kl_divergence()
                + self.lin_n_h.kl_divergence())


# ============================================================
# Model
# ============================================================

class BayesGNN(nn.Module):
    """
    Bayesian GAT → GRU → MLP for dengue incidence rate regression.

    Each forward call draws a fresh set of weights from the variational
    posteriors — run multiple times externally for MC uncertainty estimates.

    Parameters
    ----------
    bayesian_gat / bayesian_gru / bayesian_output :
        Replace each block with its Bayesian counterpart when True.
        All-True is the full BNN; partial-True gives a cheaper hybrid.
    prior_sigma, rho_init :
        Shared prior width and posterior initialisation across all Bayesian layers.
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
        prior_sigma: float = 1.0,
        rho_init: float = -3.0,
        bayesian_gat: bool = True,
        bayesian_gru: bool = True,
        bayesian_output: bool = True,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv  # deterministic fallback

        # --- GAT layers ---
        gat_layers, residual_projections = [], []
        prev_dim = in_channels

        for layer_idx, n_heads in enumerate(gat_heads):
            is_last = layer_idx == len(gat_heads) - 1
            concat = not is_last
            out_dim = gat_hidden_dim * n_heads if concat else gat_hidden_dim

            if bayesian_gat:
                gat_layers.append(BayesianGATConv(
                    prev_dim, gat_hidden_dim, n_heads, gat_dropout, concat,
                    prior_sigma, rho_init,
                ))
            else:
                gat_layers.append(GATConv(
                    prev_dim, gat_hidden_dim, n_heads, concat=concat, dropout=gat_dropout
                ))

            if use_residual and prev_dim != out_dim:
                residual_projections.append(nn.Linear(prev_dim, out_dim, bias=False))
            else:
                residual_projections.append(None)
            prev_dim = out_dim

        self.gat_layers = nn.ModuleList(gat_layers)
        self._res_projs_raw = residual_projections
        self.res_projs = nn.ModuleList([p for p in residual_projections if p is not None])
        self.use_residual = use_residual
        self.gat_out_dim = prev_dim

        # --- Optional GAT → GRU projection ---
        self.use_projection = use_projection
        if use_projection and self.gat_out_dim != gru_hidden_dim:
            self.gat_to_gru = nn.Linear(self.gat_out_dim, gru_hidden_dim)
            gru_input_dim = gru_hidden_dim
        else:
            self.gat_to_gru = None
            gru_input_dim = self.gat_out_dim

        # --- GRU cells ---
        gru_cells = []
        for i in range(num_gru_layers):
            inp = gru_input_dim if i == 0 else gru_hidden_dim
            if bayesian_gru:
                gru_cells.append(BayesianGRUCell(inp, gru_hidden_dim, prior_sigma, rho_init))
            else:
                gru_cells.append(nn.GRUCell(inp, gru_hidden_dim))
        self.gru_cells = nn.ModuleList(gru_cells)
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers

        # --- Output head ---
        self.out_dropout = nn.Dropout(out_dropout)
        if out_hidden_dim:
            if bayesian_output:
                self.fc_hidden = BayesianLinear(gru_hidden_dim, out_hidden_dim, prior_sigma, rho_init)
                self.fc_out    = BayesianLinear(out_hidden_dim, out_channels,  prior_sigma, rho_init)
            else:
                self.fc_hidden = nn.Linear(gru_hidden_dim, out_hidden_dim)
                self.fc_out    = nn.Linear(out_hidden_dim, out_channels)
        else:
            self.fc_hidden = None
            if bayesian_output:
                self.fc_out = BayesianLinear(gru_hidden_dim, out_channels, prior_sigma, rho_init)
            else:
                self.fc_out = nn.Linear(gru_hidden_dim, out_channels)

    def forward(
        self,
        x_seq: torch.Tensor,           # [T, N, F]
        edge_index: torch.Tensor,      # [2, E]
        edge_weight: torch.Tensor | None = None,  # unused — GAT learns its own weights
    ) -> torch.Tensor:
        """Single stochastic forward pass. Returns [N, out_channels]."""
        T, N, _ = x_seq.shape
        device = x_seq.device

        h_list = [
            torch.zeros(N, self.gru_hidden_dim, device=device)
            for _ in range(self.num_gru_layers)
        ]

        for t in range(T):
            x_t = x_seq[t]  # [N, F]

            res_iter = iter(self._res_projs_raw)
            for gat_layer in self.gat_layers:
                proj = next(res_iter)
                x_new = F.elu(gat_layer(x_t, edge_index))
                if self.use_residual:
                    residual = proj(x_t) if proj is not None else x_t
                    x_new = x_new + residual
                x_t = x_new

            if self.gat_to_gru is not None:
                x_t = self.gat_to_gru(x_t)

            for layer_idx, cell in enumerate(self.gru_cells):
                inp = x_t if layer_idx == 0 else h_list[layer_idx - 1]
                h_list[layer_idx] = cell(inp, h_list[layer_idx])

        h_final = self.out_dropout(h_list[-1])
        if self.fc_hidden is not None:
            h_final = F.relu(self.fc_hidden(h_final))
            h_final = self.out_dropout(h_final)
        return self.fc_out(h_final)  # [N, out_channels]

    def total_kl(self) -> torch.Tensor:
        """Sum KL over every BayesianLinear in the model (recursive module scan)."""
        device = next(self.parameters()).device
        kl = torch.zeros(1, device=device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence()
        return kl.squeeze()


# ============================================================
# Build model from config
# ============================================================

def build_bayes_model(cfg: dict, in_channels: int) -> BayesGNN:
    arch  = cfg["model"]
    gat   = arch["gat"]
    gru   = arch["gru"]
    out   = arch["output"]
    proj  = arch.get("projection", {})

    bayes    = cfg.get("bayesian", {})
    prior    = bayes.get("prior", {})
    posterior = bayes.get("posterior", {})
    layers   = bayes.get("bayesian_layers", {})

    task = cfg["target"]["task"]
    out_channels = len(cfg["target"].get("class_map", [])) if task == "classification" else 1

    return BayesGNN(
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
        prior_sigma=prior.get("sigma", 1.0),
        rho_init=posterior.get("rho_init", -3.0),
        bayesian_gat=layers.get("gat", True),
        bayesian_gru=layers.get("gru", True),
        bayesian_output=layers.get("output", True),
    )


# ============================================================
# ELBO helpers
# ============================================================

def compute_beta(
    epoch: int,
    total_epochs: int,
    kl_weight: float,
    anneal_fraction: float,
    annealing: bool,
) -> float:
    """Return the KL coefficient β for this epoch."""
    if not annealing:
        return kl_weight
    anneal_epochs = max(1.0, anneal_fraction * total_epochs)
    return float(min(kl_weight, kl_weight * (epoch / anneal_epochs)))


def elbo_loss(
    likelihood: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    free_bits: float,
    kl_scale: int,
) -> torch.Tensor:
    """ELBO = likelihood + β × KL/scale (with optional free-bits floor)."""
    if free_bits > 0.0:
        kl = (kl - free_bits).clamp(min=0.0)
    return likelihood + beta * (kl / max(kl_scale, 1))


def get_likelihood_fn(cfg: dict):
    """Return the masked likelihood function for the configured loss type."""
    task = cfg["target"]["task"]
    loss_type = cfg["loss"][task]["type"]
    if task == "regression":
        if loss_type == "masked_mse":
            return masked_mse
        if loss_type == "masked_mae":
            return masked_mae
        delta = cfg["loss"]["regression"].get("huber_delta", 1.0)
        return lambda p, t, m: masked_huber(p, t, m, delta=delta)
    raise ValueError(f"Bayesian pipeline only supports regression; got task={task!r}")


# ============================================================
# Early stopping (monitors val_mse, not total ELBO)
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_mse = math.inf
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_mse: float, model: nn.Module) -> bool:
        if val_mse < self.best_val_mse - self.min_delta:
            self.best_val_mse = val_mse
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# One-epoch helpers
# ============================================================

def train_epoch_bayes(
    model: BayesGNN,
    batches: "Batch | list[Batch]",
    optimizer: torch.optim.Optimizer,
    likelihood_fn,
    cfg: dict,
    epoch: int,
    total_epochs: int,
    grad_clip: float | None,
    task: str,
    log_transform: bool,
    mc_samples_train: int,
) -> tuple[float, float, dict]:
    """
    One training epoch.
    Returns (avg_elbo_loss, avg_likelihood_loss, metrics).
    mc_samples_train forward passes are averaged per window before backprop.
    """
    model.train()
    if isinstance(batches, Batch):
        batches = [batches]
    order = list(range(len(batches)))
    random.shuffle(order)

    bayes_cfg = cfg.get("bayesian", {})
    kl_cfg    = bayes_cfg.get("kl", {})
    beta = compute_beta(
        epoch, total_epochs,
        kl_weight=kl_cfg.get("weight", 0.1),
        anneal_fraction=kl_cfg.get("anneal_fraction", 0.25),
        annealing=kl_cfg.get("annealing", True),
    )
    free_bits = cfg.get("loss", {}).get("elbo", {}).get("free_bits", 0.0)

    total_elbo, total_likeli = 0.0, 0.0
    all_preds, all_targets, all_masks = [], [], []

    optimizer.zero_grad()
    for idx in order:
        b = batches[idx]
        N = b.x_seq.shape[1]

        elbo_sum, likeli_sum = torch.zeros(1, device=b.x_seq.device), torch.zeros(1, device=b.x_seq.device)
        pred_sum: torch.Tensor | None = None

        for _ in range(mc_samples_train):
            pred = model(b.x_seq, b.edge_index, b.edge_weight)
            likeli = likelihood_fn(pred, b.y, b.mask)
            kl = model.total_kl()
            loss = elbo_loss(likeli, kl, beta, free_bits, kl_scale=N)
            elbo_sum  = elbo_sum  + loss
            likeli_sum = likeli_sum + likeli
            pred_sum = pred.detach() if pred_sum is None else pred_sum + pred.detach()

        (elbo_sum / (mc_samples_train * len(order))).backward()
        total_elbo  += (elbo_sum  / mc_samples_train).item()
        total_likeli += (likeli_sum / mc_samples_train).item()
        all_preds.append(pred_sum / mc_samples_train)
        all_targets.append(b.y)
        all_masks.append(b.mask)

    if grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    metrics = _compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_masks),
        task, log_transform,
    )
    return total_elbo / len(order), total_likeli / len(order), metrics


@torch.no_grad()
def val_epoch_bayes(
    model: BayesGNN,
    batches: "Batch | list[Batch]",
    likelihood_fn,
    n_mc: int,
    task: str,
    log_transform: bool,
) -> tuple[float, dict]:
    """
    Validation with MC sampling.
    Returns (avg_val_mse, metrics).
    model.train() is intentionally set so BayesianLinear continues sampling;
    @torch.no_grad() prevents gradient computation.
    """
    model.train()  # keep Bayesian sampling active
    if isinstance(batches, Batch):
        batches = [batches]

    total_mse = 0.0
    all_preds, all_targets, all_masks = [], [], []

    for b in batches:
        preds_mc = torch.stack(
            [model(b.x_seq, b.edge_index, b.edge_weight) for _ in range(n_mc)], dim=0
        )  # [n_mc, N, 1]
        pred_mean = preds_mc.mean(dim=0)  # [N, 1]
        total_mse += likelihood_fn(pred_mean, b.y, b.mask).item()
        all_preds.append(pred_mean)
        all_targets.append(b.y)
        all_masks.append(b.mask)

    metrics = _compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_masks),
        task, log_transform,
    )
    return total_mse / len(batches), metrics


# ============================================================
# Full training loop
# ============================================================

def fit_bayes(
    model: BayesGNN,
    train_batches: "Batch | list[Batch]",
    val_batches: "Batch | list[Batch]",
    cfg: dict,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict:
    task          = cfg["target"]["task"]
    log_transform = cfg["target"].get("log_transform", False)

    bayes_cfg    = cfg.get("bayesian", {})
    mc_train     = bayes_cfg.get("mc_samples", {}).get("train", 1)
    mc_val       = bayes_cfg.get("mc_samples", {}).get("val", 20)
    kl_cfg       = bayes_cfg.get("kl", {})

    optimizer     = build_optimizer(model, cfg)
    scheduler     = build_scheduler(optimizer, cfg)
    likelihood_fn = get_likelihood_fn(cfg)
    es_cfg        = cfg["training"].get("early_stopping", {})
    stopper       = EarlyStopping(
        patience=es_cfg.get("patience", 25),
        min_delta=es_cfg.get("min_delta", 1e-4),
    )
    grad_clip = cfg["training"].get("gradient_clip")
    n_epochs  = cfg["training"]["epochs"]
    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        train_elbo, train_mse, train_metrics = train_epoch_bayes(
            model, train_batches, optimizer, likelihood_fn,
            cfg, epoch, n_epochs, grad_clip, task, log_transform, mc_train,
        )
        val_mse, val_metrics = val_epoch_bayes(
            model, val_batches, likelihood_fn, mc_val, task, log_transform,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mse)  # step on MSE, not ELBO
            else:
                scheduler.step()

        beta = compute_beta(
            epoch, n_epochs,
            kl_weight=kl_cfg.get("weight", 0.1),
            anneal_fraction=kl_cfg.get("anneal_fraction", 0.25),
            annealing=kl_cfg.get("annealing", True),
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch":      epoch,
            "train_elbo": train_elbo,
            "train_mse":  train_mse,
            "val_mse":    val_mse,
            "beta":       beta,
            "lr":         current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(log)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[epoch {epoch:04d}]  elbo={train_elbo:.4f}  mse={train_mse:.4f}"
                  f"  val_mse={val_mse:.4f}  β={beta:.4f}  lr={current_lr:.2e}")

        if stopper.step(val_mse, model):
            print(f"[early stop] val_mse no improvement for {stopper.patience} epochs. "
                  f"Best val_mse={stopper.best_val_mse:.4f}")
            break

    stopper.restore_best(model)
    print(f"[fit_bayes] Restored best weights (val_mse={stopper.best_val_mse:.4f})")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[fit_bayes] Checkpoint → {checkpoint_path}")

    final_metrics = (history[-1] if history else {}) | {"best_val_mse": stopper.best_val_mse}
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fh:
            json.dump(final_metrics, fh, indent=2)
        print(f"[fit_bayes] Metrics → {metrics_path}")

    return final_metrics


# ============================================================
# apply_best_params — merge tune output into config (in-place)
# ============================================================

def apply_best_params(cfg: dict, params: dict) -> None:
    """Merge bayes_best_params.json into a loaded config dict in-place."""
    tr   = cfg["training"]
    gat  = cfg["model"]["gat"]
    gru  = cfg["model"]["gru"]
    out  = cfg["model"]["output"]
    seq  = cfg["sequence"]

    # --- Architecture (same keys as deterministic model) ---
    if "lookback_steps"  in params: seq["lookback_steps"] = params["lookback_steps"]
    if "gat_hidden_dim"  in params: gat["hidden_dim"]     = params["gat_hidden_dim"]
    if "gat_dropout"     in params: gat["dropout"]        = params["gat_dropout"]
    if "gat_residual"    in params: gat["residual"]       = params["gat_residual"]

    if any(k in params for k in ("gat_num_layers", "gat_heads_layer1", "gat_heads_layer2")):
        n_layers = params.get("gat_num_layers", len(gat["heads"]))
        h1 = params.get("gat_heads_layer1", gat["heads"][0])
        h2 = params.get("gat_heads_layer2", 2)
        if n_layers == 1:
            gat["heads"] = [1]
        elif n_layers == 2:
            gat["heads"] = [h1, 1]
        else:
            gat["heads"] = [h1] + [h2] * (n_layers - 2) + [1]
        gat["num_layers"] = n_layers

    if "gru_hidden_dim"    in params: gru["hidden_dim"]  = params["gru_hidden_dim"]
    if "gru_num_layers"    in params: gru["num_layers"]  = params["gru_num_layers"]
    if "gru_dropout"       in params: gru["dropout"]     = params["gru_dropout"]
    if "output_hidden_dim" in params: out["hidden_dim"]  = params["output_hidden_dim"]
    if "output_dropout"    in params: out["dropout"]     = params["output_dropout"]

    if "learning_rate"    in params: tr["learning_rate"] = params["learning_rate"]
    if "weight_decay"     in params: tr["weight_decay"]  = params["weight_decay"]
    if "gradient_clip"    in params: tr["gradient_clip"] = params["gradient_clip"]

    if "scheduler_type"   in params: tr["scheduler"]["type"]               = params["scheduler_type"]
    if "plateau_patience" in params: tr["scheduler"]["plateau"]["patience"] = params["plateau_patience"]
    if "cosine_t_max"     in params: tr["scheduler"]["cosine"]["t_max"]     = params["cosine_t_max"]

    # --- Bayesian-specific params ---
    bayes   = cfg.setdefault("bayesian", {})
    kl      = bayes.setdefault("kl", {})
    prior   = bayes.setdefault("prior", {})
    post    = bayes.setdefault("posterior", {})
    mc_samp = bayes.setdefault("mc_samples", {})
    elbo    = cfg.setdefault("loss", {}).setdefault("elbo", {})

    if "kl_weight"          in params:
        kl["weight"]   = params["kl_weight"]
        elbo["kl_weight"] = params["kl_weight"]
    if "kl_anneal_fraction" in params: kl["anneal_fraction"]  = params["kl_anneal_fraction"]
    if "prior_sigma"        in params: prior["sigma"]          = params["prior_sigma"]
    if "rho_init"           in params: post["rho_init"]        = params["rho_init"]
    if "mc_samples_train"   in params: mc_samp["train"]        = params["mc_samples_train"]
    if "free_bits"          in params: elbo["free_bits"]       = params["free_bits"]


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BayesianSTGAT")
    p.add_argument("--config",            required=True)
    p.add_argument("--train-data",        default=None)
    p.add_argument("--val-data",          default=None)
    p.add_argument("--best-params",       default=None)
    p.add_argument("--output-checkpoint", default=None)
    p.add_argument("--output-metrics",    default=None)
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    if "snakemake" in dir():
        sm = snakemake  # noqa: F821
        args = argparse.Namespace(
            config=sm.input.config_file,
            train_data=getattr(sm.input, "train_data", None),
            val_data=getattr(sm.input, "val_data", None),
            best_params=getattr(sm.input, "best_params", None),
            output_checkpoint=getattr(sm.output, "checkpoint", None),
            output_metrics=getattr(sm.output, "metrics", None),
            device=f"cuda:{sm.params.cuda_device}" if hasattr(sm.params, "cuda_device") else "cpu",
        )
    else:
        args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.best_params is not None:
        with open(args.best_params) as fh:
            bp = json.load(fh)
        bp.pop("best_val_mse", None)
        apply_best_params(cfg, bp)
        print(f"[main] Applied best params: {bp}")

    torch.manual_seed(cfg["training"].get("random_state", 42))
    device = torch.device(args.device)
    print(f"[main] Device: {device}")

    if args.train_data is not None and Path(args.train_data).exists():
        train_batches = load_batches(args.train_data, device)
        val_batches   = load_batches(args.val_data,   device)
        in_channels   = train_batches[0].x_seq.shape[2]
        print(f"[main] Windows — train: {len(train_batches)}  val: {len(val_batches)}")
    else:
        print("[main] No data files — using synthetic data.")
        train_batches, val_batches = make_synthetic_data(cfg, device)
        in_channels = train_batches.x_seq.shape[2]

    model = build_bayes_model(cfg, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Model parameters: {n_params:,}")

    fit_bayes(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        cfg=cfg,
        checkpoint_path=Path(args.output_checkpoint) if args.output_checkpoint else None,
        metrics_path=Path(args.output_metrics) if args.output_metrics else None,
    )


if __name__ == "__main__":
    main()
