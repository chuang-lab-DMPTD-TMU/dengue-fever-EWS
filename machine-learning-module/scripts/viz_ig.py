"""
scripts/viz_ig.py
-----------------
Compute Integrated Gradients (IG) attributions for a trained STGAT and
produce visualisation figures.

IG attributes each input feature's contribution to the model's prediction
by integrating gradients along a straight path from a baseline (zeros in
scaled feature space ≈ column means) to the actual input.

    IG(x) ≈ (x - x') * (1/m) Σ_{k=1}^{m} ∂F(x' + k/m·(x-x')) / ∂x

Attribution target: sum of predicted IR over all nodes with observed targets
in a given test window.  Attributions are averaged across all test windows
and across nodes to give interpretable global summaries.

Outputs (written to reports/stgat/{run_name}/):
    ig_attributions.pt          raw IG tensor [W, T, N, F] averaged to [T, N, F]
    ig_feature_importance.png   global |IG| per feature (bar chart, sorted)
    ig_temporal_heatmap.png     mean |IG| over nodes → [T, top_k features] heatmap
    ig_node_importance.png      mean |IG| over time & features → [N] (top nodes)

Standalone:
    python workflow/scripts/viz_ig.py \\
        --config     config/stgat/sea_baseline.yaml \\
        --checkpoint models/stgat/sea_baseline/best.pt \\
        --test-data  data/processed/stgat/sea_baseline_test.pt \\
        --out-dir    reports/stgat/sea_baseline \\
        --steps      50 \\
        --device     cuda

Via Snakemake (rules/IG.smk):
    Inputs/outputs are injected by the rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from train_stgat import apply_best_params, DengueGNN, build_model, load_batches


# ============================================================
# Integrated Gradients
# ============================================================

def integrated_gradients(
    model: DengueGNN,
    x_seq: torch.Tensor,       # [T, N, F]
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    mask: torch.Tensor,        # [N] — which nodes contribute to the target scalar
    baseline: torch.Tensor,    # [T, N, F]  (zeros after scaling ≈ column means)
    steps: int = 50,
) -> torch.Tensor:
    """
    Returns IG attributions with the same shape as x_seq: [T, N, F].
    The attribution target is the sum of predicted log-IR over observed nodes.
    """
    model.eval()
    delta = x_seq - baseline
    grads_acc = torch.zeros_like(x_seq)

    for k in range(1, steps + 1):
        alpha = k / steps
        interp = (baseline + alpha * delta).requires_grad_(True)
        pred   = model(interp, edge_index, edge_weight)   # [N, out]
        # Scalar target: sum of predictions for observed nodes only
        target = (pred.squeeze(-1) * mask).sum()
        target.backward()
        grads_acc += interp.grad.detach()

    # Riemann approximation of the integral
    attributions = delta.detach() * grads_acc / steps
    return attributions   # [T, N, F]


# ============================================================
# Helpers
# ============================================================

def _load_feature_names(data: dict) -> list[str]:
    names = data.get("feature_names", [])
    if names:
        return list(names)
    F = data["windows"].shape[-1]
    return [f"feat_{i}" for i in range(F)]


def _load_node_names(data: dict) -> list[str]:
    names = data.get("node_names", [])
    if names:
        return list(names)
    N = data["windows"].shape[2]
    return [f"node_{i}" for i in range(N)]


# ============================================================
# Figures
# ============================================================

def plot_feature_importance(
    ig_mean_fn: np.ndarray,   # [F]  mean |IG| over time × nodes
    feature_names: list[str],
    out_path: Path,
    top_k: int = 30,
) -> None:
    idx   = np.argsort(ig_mean_fn)[::-1][:top_k]
    vals  = ig_mean_fn[idx]
    names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.28)))
    ax.barh(np.arange(top_k), vals[::-1], color="steelblue")
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Mean |IG| attribution")
    ax.set_title(f"Top {top_k} features by Integrated Gradients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_temporal_heatmap(
    ig_mean_tf: np.ndarray,   # [T, F]  mean |IG| over nodes
    feature_names: list[str],
    out_path: Path,
    top_k: int = 20,
) -> None:
    # Select top_k features by overall attribution magnitude
    feat_importance = ig_mean_tf.mean(axis=0)
    top_idx         = np.argsort(feat_importance)[::-1][:top_k]
    data_plot       = ig_mean_tf[:, top_idx]          # [T, top_k]
    top_names       = [feature_names[i] for i in top_idx]

    T = data_plot.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.5), max(4, T * 0.35)))
    im = ax.imshow(data_plot, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Lookback step (0 = oldest)")
    ax.set_title("Mean |IG| over nodes — temporal × feature")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_node_importance(
    ig_mean_n: np.ndarray,    # [N]  mean |IG| over time × features
    node_names: list[str],
    out_path: Path,
    top_k: int = 30,
) -> None:
    idx   = np.argsort(ig_mean_n)[::-1][:top_k]
    vals  = ig_mean_n[idx]
    names = [node_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.28)))
    ax.barh(np.arange(top_k), vals[::-1], color="coral")
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Mean |IG| attribution")
    ax.set_title(f"Top {top_k} nodes by Integrated Gradients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute IG attributions for a trained STGAT")
    p.add_argument("--config",      required=True,  help="Path to config_stgat.yaml")
    p.add_argument("--checkpoint",  required=True,  help="Path to best.pt checkpoint")
    p.add_argument("--test-data",   required=True,  help="Path to test .pt tensor file")
    p.add_argument("--out-dir",     required=True,  help="Directory for output figures and tensors")
    p.add_argument("--best-params", default=None,   help="Path to best_params.json from tune_stgat")
    p.add_argument("--steps",       type=int, default=50,
                   help="Number of IG integration steps (default: 50)")
    p.add_argument("--top-k-feat",  type=int, default=20,
                   help="Top-k features shown in heatmap (default: 20)")
    p.add_argument("--top-k-nodes", type=int, default=30,
                   help="Top-k nodes shown in node importance plot (default: 30)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    if "snakemake" in dir():
        sm = snakemake  # noqa: F821
        args = argparse.Namespace(
            config      = sm.input.config_file,
            checkpoint  = sm.input.checkpoint,
            test_data   = sm.input.test_data,
            best_params = getattr(sm.input, "best_params", None),
            out_dir     = sm.params.out_dir,
            steps       = sm.params.get("ig_steps", 50),
            top_k_feat  = sm.params.get("top_k_feat", 20),
            top_k_nodes = sm.params.get("top_k_nodes", 30),
            device      = sm.params.get("device", "cpu"),
        )
    else:
        args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.best_params is not None:
        import json
        with open(args.best_params) as fh:
            bp = json.load(fh)
        bp.pop("best_val_loss", None)
        apply_best_params(cfg, bp)
        print(f"[viz_ig] Applied best params from {args.best_params}: {bp}")

    device = torch.device(args.device)
    print(f"[viz_ig] Device: {device}  |  IG steps: {args.steps}")

    # ---- Load test data ----
    raw_data     = torch.load(args.test_data, map_location=device, weights_only=False)
    test_batches = load_batches(args.test_data, device)
    feature_names = _load_feature_names(raw_data)
    node_names    = _load_node_names(raw_data)
    print(f"[viz_ig] Test windows: {len(test_batches)}  "
          f"Nodes: {len(node_names)}  Features: {len(feature_names)}")

    # ---- Load model ----
    in_channels = test_batches[0].x_seq.shape[2]
    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"[viz_ig] Loaded checkpoint: {args.checkpoint}")

    # ---- Compute IG per test window, accumulate ----
    T, N, F = test_batches[0].x_seq.shape
    ig_sum = torch.zeros(T, N, F)

    for w_idx, batch in enumerate(test_batches):
        baseline = torch.zeros_like(batch.x_seq)   # zeros in scaled space ≈ feature means
        ig_w = integrated_gradients(
            model       = model,
            x_seq       = batch.x_seq,
            edge_index  = batch.edge_index,
            edge_weight = batch.edge_weight,
            mask        = batch.mask,
            baseline    = baseline,
            steps       = args.steps,
        )
        ig_sum += ig_w.cpu()
        print(f"  Window {w_idx+1}/{len(test_batches)} done")

    ig_mean = (ig_sum / len(test_batches)).numpy()   # [T, N, F]

    # ---- Save raw attributions ----
    raw_out = out_dir / "ig_attributions.pt"
    torch.save(
        {"ig_mean_tnf": ig_mean, "feature_names": feature_names, "node_names": node_names},
        raw_out,
    )
    print(f"[viz_ig] Raw attributions → {raw_out}")

    # ---- Aggregate for figures ----
    ig_abs        = np.abs(ig_mean)            # [T, N, F]
    ig_mean_fn    = ig_abs.mean(axis=(0, 1))   # [F]  — global feature importance
    ig_mean_tf    = ig_abs.mean(axis=1)        # [T, F] — temporal × feature
    ig_mean_n     = ig_abs.mean(axis=(0, 2))   # [N]  — node importance

    # ---- Plot ----
    print("[viz_ig] Generating figures ...")
    plot_feature_importance(
        ig_mean_fn, feature_names,
        out_dir / "ig_feature_importance.png",
        top_k=min(args.top_k_feat * 2, len(feature_names)),
    )
    plot_temporal_heatmap(
        ig_mean_tf, feature_names,
        out_dir / "ig_temporal_heatmap.png",
        top_k=args.top_k_feat,
    )
    plot_node_importance(
        ig_mean_n, node_names,
        out_dir / "ig_node_importance.png",
        top_k=min(args.top_k_nodes, len(node_names)),
    )

    print("[viz_ig] Done.")


if __name__ == "__main__":
    main()
