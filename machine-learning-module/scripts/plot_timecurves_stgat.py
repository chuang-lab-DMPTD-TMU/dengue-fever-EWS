"""
scripts/plot_timecurves_stgat.py
--------------------------------
Plot real vs. predicted IR time curves for 20 randomly sampled regions
across the full 2011–2018 period.

All three splits (train / val / test) are run through the model so the
predicted curve spans the maximum observable range (first prediction at
month lookback+1 ≈ Jan 2012 through Dec 2018).

Usage (standalone):
    python workflow/scripts/plot_timecurves_stgat.py \
        --config      config/stgat/sea_baseline.yaml \
        --checkpoint  models/stgat/sea_baseline/best.pt \
        --train-data  data/processed/stgat/sea_baseline_train.pt \
        --val-data    data/processed/stgat/sea_baseline_val.pt \
        --test-data   data/processed/stgat/sea_baseline_test.pt \
        --output-dir  reports/stgat/sea_baseline \
        --seed        42

Via Snakemake (rule plot_timecurves_stgat in stgat.smk):
    The rule passes all of the above flags automatically.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from train_stgat import apply_best_params, build_model, load_batches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot real vs predicted IR time curves")
    p.add_argument("--config",       required=True, help="Path to config_stgat.yaml")
    p.add_argument("--checkpoint",   required=True, help="Path to best.pt checkpoint")
    p.add_argument("--train-data",   required=True, help="Path to train .pt tensor file")
    p.add_argument("--val-data",     required=True, help="Path to val .pt tensor file")
    p.add_argument("--test-data",    required=True, help="Path to test .pt tensor file")
    p.add_argument("--best-params",  default=None,  help="Optional best_params.json")
    p.add_argument("--output-dir",   required=True, help="Directory to write output PNG")
    p.add_argument("--n-regions",    type=int, default=20, help="Number of regions to plot")
    p.add_argument("--seed",         type=int, default=42,  help="Random seed for region sampling")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _load_raw(path: str, device: torch.device) -> dict:
    """Return the raw dict from a .pt file (preserves node_names, target_dates)."""
    return torch.load(path, map_location=device, weights_only=False)


def run_inference(
    model: torch.nn.Module,
    batches,
    raw: dict,
    log_transform: bool,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run model on every batch, collect per-node predictions.

    Returns a DataFrame with columns: node_name, date (period[M]), pred_ir.
    """
    model.eval()
    node_names   = raw["node_names"]          # list[str], length N
    target_dates = raw["target_dates"]        # list[str] "YYYY-MM", length W

    all_preds = []
    with torch.no_grad():
        for w_idx, batch in enumerate(batches):
            pred = model(batch.x_seq, batch.edge_index, batch.edge_weight)  # [N, 1]
            pred = pred.squeeze(-1).cpu().numpy()                            # [N]
            if log_transform:
                pred = np.expm1(pred)
            pred = np.maximum(pred, 0.0)   # IR can't be negative

            date_str = target_dates[w_idx]  # "YYYY-MM"
            date     = pd.Period(date_str, freq="M")
            for n_idx, node in enumerate(node_names):
                all_preds.append({"node_name": node, "date": date, "pred_ir": pred[n_idx]})

    return pd.DataFrame(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    import json
    best_params_path = args.best_params
    if best_params_path is None:
        # Auto-detect alongside the checkpoint
        candidate = Path(args.checkpoint).parent / "best_params.json"
        if candidate.exists():
            best_params_path = str(candidate)
            print(f"[plot_timecurves] Auto-detected best_params: {best_params_path}")

    if best_params_path is not None:
        with open(best_params_path) as fh:
            bp = json.load(fh)
        bp.pop("best_val_loss", None)
        apply_best_params(cfg, bp)
        print(f"[plot_timecurves] Applied best params: {bp}")

    log_transform = cfg["target"].get("log_transform", False)
    device        = torch.device(args.device)
    print(f"[plot_timecurves] Device: {device}")

    # --- Load raw dicts first (need node_names + target_dates) ---
    raw_train = _load_raw(args.train_data, device)
    raw_val   = _load_raw(args.val_data,   device)
    raw_test  = _load_raw(args.test_data,  device)

    # --- Load batches (Batch NamedTuple list) ---
    batches_train = load_batches(args.train_data, device)
    batches_val   = load_batches(args.val_data,   device)
    batches_test  = load_batches(args.test_data,  device)

    node_names   = raw_train["node_names"]
    in_channels  = batches_train[0].x_seq.shape[2]
    print(f"[plot_timecurves] {len(node_names)} nodes, {in_channels} features")

    # --- Load model ---
    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"[plot_timecurves] Loaded checkpoint from {args.checkpoint}")

    # --- Run inference on all three splits ---
    dfs = []
    for label, batches, raw in [
        ("train", batches_train, raw_train),
        ("val",   batches_val,   raw_val),
        ("test",  batches_test,  raw_test),
    ]:
        df = run_inference(model, batches, raw, log_transform, device)
        print(f"[plot_timecurves] {label}: {len(df)} predictions")
        dfs.append(df)

    pred_df = pd.concat(dfs, ignore_index=True).sort_values(["node_name", "date"])

    # --- Load original CSV for real IR ---
    data_path = cfg["paths"]["data_processed"]
    # Resolve relative to the project root (workflow/ parent)
    proj_root = Path(args.config).parents[2]  # config/stgat/X.yaml → root
    csv_path  = proj_root / data_path
    if not csv_path.exists():
        # Try relative to cwd
        csv_path = Path(data_path)
    print(f"[plot_timecurves] Loading real IR from {csv_path}")
    raw_csv = pd.read_csv(csv_path, parse_dates=["Date"])
    raw_csv["date"] = raw_csv["Date"].dt.to_period("M")
    real_df = raw_csv[["name", "admin", "date", "IR"]].rename(columns={"name": "node_name"})

    # --- Merge real + predicted ---
    merged = real_df.merge(pred_df, on=["node_name", "date"], how="left")

    # --- Sample 20 random regions ---
    rng = random.Random(args.seed)
    sampled = rng.sample(node_names, min(args.n_regions, len(node_names)))
    print(f"[plot_timecurves] Sampled regions: {sampled}")

    # --- Plot ---
    ncols = 5
    nrows = int(np.ceil(args.n_regions / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2), sharey=False)
    axes_flat = axes.flatten()

    for ax_idx, region in enumerate(sampled):
        ax  = axes_flat[ax_idx]
        sub = merged[merged["node_name"] == region].sort_values("date")

        # Convert Period → datetime for matplotlib
        dates = sub["date"].dt.to_timestamp()

        ax.plot(dates, sub["IR"],      color="#2166ac", linewidth=1.4,
                label="Real IR",       zorder=3)
        ax.plot(dates, sub["pred_ir"], color="#d6604d", linewidth=1.4,
                linestyle="--",        label="Predicted IR", zorder=4)

        # Shade where real IR is missing
        missing = sub["IR"].isna()
        if missing.any():
            for start, end in _missing_spans(dates, missing):
                ax.axvspan(start, end, color="grey", alpha=0.15, linewidth=0)

        country = sub["admin"].iloc[0] if "admin" in sub.columns and not sub["admin"].isna().all() else ""
        title   = f"{region}" + (f"\n({country})" if country else "")
        ax.set_title(title, fontsize=8, pad=3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel("IR / 100k", fontsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.4)

    # Hide unused axes
    for ax in axes_flat[args.n_regions:]:
        ax.set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02), framealpha=0.9)

    fig.suptitle("STGAT — Real vs Predicted IR (2011–2018)", fontsize=13, y=1.01)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"timecurves_ir_seed{args.seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_timecurves] Saved → {out_path}")


def _missing_spans(dates: pd.Series, missing: pd.Series):
    """Yield (start, end) datetime pairs for contiguous missing-IR spans."""
    in_span = False
    span_start = None
    for dt, is_missing in zip(dates, missing):
        if is_missing and not in_span:
            span_start = dt
            in_span = True
        elif not is_missing and in_span:
            yield span_start, dt
            in_span = False
    if in_span:
        yield span_start, dates.iloc[-1]


if __name__ == "__main__":
    main()
