"""
scripts/eval_bayes_stgat.py
----------------------------
Evaluate a trained BayesianSTGAT checkpoint on the held-out test set.

Runs mc_samples.test (default 50) stochastic forward passes per test window and
computes uncertainty estimates as configured in uncertainty_output:
  predictive_mean  — average prediction across MC samples
  predictive_std   — std across MC samples (= epistemic std when aleatoric=False)
  epistemic        — variance of per-sample predictions (same as predictive_std^2 here)
  predict_interval — percentile-based credible interval (default 0.95 → [2.5%, 97.5%])

Results written to --output-metrics as JSON, including standard regression metrics
in both log-space and original IR space (after expm1 if log_transform=True).

Via Snakemake (bayes_stgat.smk → eval_bayes_stgat):
    python scripts/eval_bayes_stgat.py \\
        --config         config/stgat/{run_name}.yaml \\
        --checkpoint     models/stgat/{run_name}/bayes_best.pt \\
        --test-data      data/processed/stgat/{run_name}_test.pt \\
        --output-metrics reports/stgat/{run_name}/bayes_test_metrics.json \\
        --device         cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from train_bayes_stgat import (
    Batch,
    apply_best_params,
    build_bayes_model,
    get_likelihood_fn,
    load_batches,
    make_synthetic_data,
)
from train_stgat import _compute_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BayesianSTGAT checkpoint")
    p.add_argument("--config",         required=True)
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--test-data",      default=None)
    p.add_argument("--best-params",    default=None,
                   help="Optional bayes_best_params.json to apply before loading")
    p.add_argument("--output-metrics", default=None)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MC evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_predict(
    model,
    batches: "Batch | list[Batch]",
    n_mc: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run n_mc stochastic forward passes over all test windows.

    Returns (pred_mean, pred_std, y_all, mask_all, preds_raw) where:
      pred_mean  [W, N]        — predictive mean across MC samples
      pred_std   [W, N]        — predictive std (epistemic uncertainty)
      y_all      [W, N]        — ground-truth targets
      mask_all   [W, N]        — validity mask (1 = observed)
      preds_raw  [n_mc, W, N]  — all raw MC predictions (for interval computation)
    """
    model.train()  # keep BayesianLinear sampling active
    if isinstance(batches, Batch):
        batches = [batches]

    all_means, all_stds, all_y, all_masks, all_raw = [], [], [], [], []

    for b in batches:
        # [n_mc, N, 1] → squeeze to [n_mc, N]
        preds = torch.stack(
            [model(b.x_seq, b.edge_index, b.edge_weight).squeeze(-1) for _ in range(n_mc)],
            dim=0,
        )  # [n_mc, N]
        all_means.append(preds.mean(dim=0))   # [N]
        all_stds.append(preds.std(dim=0))     # [N]
        all_y.append(b.y)
        all_masks.append(b.mask)
        all_raw.append(preds)

    pred_mean = torch.stack(all_means, dim=0)  # [W, N]
    pred_std  = torch.stack(all_stds,  dim=0)  # [W, N]
    y_all     = torch.stack(all_y,     dim=0)  # [W, N]
    mask_all  = torch.stack(all_masks, dim=0)  # [W, N]
    preds_raw = torch.stack(all_raw,   dim=0)  # [W, n_mc, N]  (permuted from list)

    # raw list is [n_mc, N] per window → stack gives [W, n_mc, N]
    # but torch.stack(all_raw) stacks [n_mc, N] tensors → [W, n_mc, N]
    return pred_mean, pred_std, y_all, mask_all, preds_raw


def compute_regression_metrics(
    pred_mean: torch.Tensor,  # [W, N]
    y_all: torch.Tensor,      # [W, N]
    mask_all: torch.Tensor,   # [W, N]
    log_transform: bool,
) -> dict:
    """Compute MAE / RMSE in log-space and original IR space on masked nodes."""
    valid = mask_all.bool().flatten()
    p = pred_mean.flatten()[valid]
    t = y_all.flatten()[valid]

    mae  = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()
    mse  = ((p - t) ** 2).mean().item()
    metrics = {"test_mse": mse, "test_mae": mae, "test_rmse": rmse}

    if log_transform:
        p_ir = torch.expm1(p.clamp(min=0.0))
        t_ir = torch.expm1(t.clamp(min=0.0))
        metrics["test_mae_ir"]  = (p_ir - t_ir).abs().mean().item()
        metrics["test_rmse_ir"] = ((p_ir - t_ir) ** 2).mean().sqrt().item()
        metrics["test_mse_ir"]  = ((p_ir - t_ir) ** 2).mean().item()

    return metrics


def compute_uncertainty_metrics(
    pred_std: torch.Tensor,   # [W, N]
    preds_raw: torch.Tensor,  # [W, n_mc, N]
    mask_all: torch.Tensor,   # [W, N]
    y_all: torch.Tensor,      # [W, N]
    interval_level: float,
    log_transform: bool,
) -> dict:
    """
    Summarise uncertainty over masked nodes.
    epistemic_std ≈ predictive_std when aleatoric output is disabled.
    """
    valid = mask_all.bool()

    mean_epistemic_std = pred_std[valid].mean().item()

    # Credible interval coverage
    alpha = (1.0 - interval_level) / 2.0
    lower = torch.quantile(preds_raw, alpha,         dim=1)  # [W, N]
    upper = torch.quantile(preds_raw, 1.0 - alpha,   dim=1)  # [W, N]
    covered = ((y_all >= lower) & (y_all <= upper) & valid).float().sum()
    n_valid  = valid.float().sum()
    coverage = (covered / n_valid).item() if n_valid > 0 else float("nan")

    metrics: dict = {
        "mean_epistemic_std": mean_epistemic_std,
        f"interval_{int(interval_level*100)}_coverage": coverage,
    }

    if log_transform:
        std_ir = torch.expm1(pred_std[valid].clamp(min=0.0))
        metrics["mean_epistemic_std_ir"] = std_ir.mean().item()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if "snakemake" in dir():
        sm = snakemake  # noqa: F821
        args = argparse.Namespace(
            config=sm.input.config_file,
            checkpoint=sm.input.checkpoint,
            test_data=getattr(sm.input, "test_data", None),
            best_params=None,
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
        print(f"[eval_bayes] Applied best params from {args.best_params}: {bp}")

    device = torch.device(args.device)
    print(f"[eval_bayes] Device: {device}")

    # --- Test data ---
    if args.test_data is not None and Path(args.test_data).exists():
        test_batches = load_batches(args.test_data, device)
        print(f"[eval_bayes] Loaded {len(test_batches)} test windows from {args.test_data}")
    else:
        print("[eval_bayes] No test data — using synthetic data.")
        _, test_batches = make_synthetic_data(cfg, device)

    ref = test_batches[0] if isinstance(test_batches, list) else test_batches
    in_channels = ref.x_seq.shape[2]

    # --- Load model ---
    model = build_bayes_model(cfg, in_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    print(f"[eval_bayes] Loaded checkpoint from {args.checkpoint}")

    # --- MC inference settings ---
    bayes_cfg      = cfg.get("bayesian", {})
    n_mc           = bayes_cfg.get("mc_samples", {}).get("test", 50)
    unc_cfg        = cfg.get("uncertainty_output", {})
    interval_level = unc_cfg.get("predict_interval", 0.95)
    log_transform  = cfg["target"].get("log_transform", False)

    print(f"[eval_bayes] MC samples: {n_mc}  |  interval: {interval_level:.0%}  |  "
          f"log_transform: {log_transform}")

    # --- Run MC prediction ---
    pred_mean, pred_std, y_all, mask_all, preds_raw = mc_predict(
        model, test_batches, n_mc, device
    )

    # --- Metrics ---
    regression_metrics  = compute_regression_metrics(pred_mean, y_all, mask_all, log_transform)
    uncertainty_metrics = compute_uncertainty_metrics(
        pred_std, preds_raw, mask_all, y_all, interval_level, log_transform
    )

    results = {
        **regression_metrics,
        **uncertainty_metrics,
        "n_mc_samples": n_mc,
        "n_test_windows": len(test_batches) if isinstance(test_batches, list) else 1,
    }
    print(f"[eval_bayes] {results}")

    if args.output_metrics:
        out = Path(args.output_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[eval_bayes] Metrics → {out}")


if __name__ == "__main__":
    main()
