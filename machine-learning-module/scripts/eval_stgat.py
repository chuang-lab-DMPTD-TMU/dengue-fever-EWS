"""
scripts/eval_stgat.py
---------------------
Evaluate a trained STGAT checkpoint on the held-out test set.

Via Snakemake (stgat.smk):
    python workflow/scripts/eval_stgat.py \
        --config         config/stgat/{run_name}.yaml \
        --checkpoint     models/stgat/{run_name}/best.pt \
        --test-data      data/processed/stgat/{run_name}_test.pt \
        --graph          data/processed/stgat/{run_name}_graph.pkl \
        --output-metrics reports/stgat/{run_name}/test_metrics.json \
        --device cuda

Expected .pt format (saved with torch.save as a dict):
    windows     : [W, L, N, F]  float32  (W sliding windows, L lookback steps)
    y           : [W, N]        float32 (regression) or int64 (classification)
    mask        : [W, N]        float32
    edge_index  : [2, E]        int64
    edge_weight : [E]           float32  (optional, defaults to ones)

Falls back to synthetic data when --test-data is omitted (smoke-test mode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from train_stgat import apply_best_params, build_model, get_loss_fn, load_batches, make_synthetic_data, val_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate STGAT checkpoint on test data")
    p.add_argument("--config",          required=True,  help="Path to config_stgat.yaml")
    p.add_argument("--checkpoint",      required=True,  help="Path to best.pt checkpoint")
    p.add_argument("--test-data",       default=None,   help="Path to test tensor file (.pt)")
    p.add_argument("--best-params",     default=None,   help="Path to best_params.json from tune_stgat")
    p.add_argument("--output-metrics",  default=None,   help="Where to write test_metrics.json")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.best_params is not None:
        with open(args.best_params) as fh:
            bp = json.load(fh)
        bp.pop("best_val_loss", None)
        apply_best_params(cfg, bp)
        print(f"[eval_stgat] Applied best params from {args.best_params}: {bp}")

    device = torch.device(args.device)
    print(f"[eval_stgat] Device: {device}")

    # --- Test data ---
    if args.test_data is not None and Path(args.test_data).exists():
        test_batches = load_batches(args.test_data, device)
        print(f"[eval_stgat] Loaded {len(test_batches)} test windows from {args.test_data}")
    else:
        print("[eval_stgat] No test data found — using synthetic data.")
        _, synth = make_synthetic_data(cfg, device)
        test_batches = synth

    ref         = test_batches[0] if isinstance(test_batches, list) else test_batches
    in_channels = ref.x_seq.shape[2]

    # --- Load model ---
    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"[eval_stgat] Loaded checkpoint from {args.checkpoint}")

    # --- Evaluate ---
    loss_fn       = get_loss_fn(cfg)
    task          = cfg["target"]["task"]
    log_transform = cfg["target"].get("log_transform", False)
    test_loss, test_metrics = val_epoch(model, test_batches, loss_fn, task, log_transform)

    results = {
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    print(f"[eval_stgat] {results}")

    if args.output_metrics:
        out = Path(args.output_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[eval_stgat] Metrics → {out}")


if __name__ == "__main__":
    main()
