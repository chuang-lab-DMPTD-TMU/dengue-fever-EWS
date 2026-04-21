"""
scripts/preprocess_stgat.py
---------------------------
Build train / val / test .pt tensor files for the STGAT pipeline.

Graph: fully connected (N*(N-1) directed edges, unit weights).
GAT attention heads learn which connections matter — no coordinates needed.

Sliding-window layout (lookback L, stride 1):
    window i  →  x_seq = features[i : i+L],  y = IR[i+L],  mask = ~isnan(IR[i+L])

Split (based on target date, not window start):
    test   — target month in test_year (config split.test_year, default 2018) → 12 windows
    val    — last val_fraction of remaining windows
    train  — the rest

Output per split  (data/processed/stgat/{run_name}_{split}.pt):
    windows      [W, L, N, F]  float32   stacked input sequences
    y            [W, N]        float32   log1p(IR) targets  (or raw IR if log_transform=False)
    mask         [W, N]        float32   1 = observed, 0 = missing
    edge_index   [2, E]        int64     same for all splits
    edge_weight  [E]           float32   uniform = 1.0
    node_names   list[str]
    feature_names list[str]
    target_dates list[str]

Usage (standalone):
    python workflow/scripts/preprocess_stgat.py \\
        --config   config/config_stgat.yaml \\
        --input    machine-learning-module/main/data/processed/SEA_dengue_env_monthly_2011-2018.csv \\
        --out-dir  data/processed/stgat \\
        --run-name sea_baseline

Via Snakemake (rule preprocess_stgat in stgat.smk):
    The rule passes --config, --input, --out-dir, --run-name.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CSV → STGAT tensor files")
    p.add_argument("--config",    required=True,  help="Path to config_stgat.yaml")
    p.add_argument("--input",     required=True,  help="Path to SEA monthly CSV")
    p.add_argument("--out-dir",   required=True,  help="Directory to write .pt files")
    p.add_argument("--run-name",  required=True,  help="Run name prefix for output files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_lags(node_df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged columns for each variable×lag pair within one node's time series."""
    for col in columns:
        for lag in lags:
            node_df[f"{col}_lag{lag}"] = node_df[col].shift(lag)
    return node_df


def build_feature_matrix(
    df: pd.DataFrame,
    env_vars: list[str],
    population_vars: list[str],
    lag_steps: list[int],
    use_lulc: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return a wide DataFrame indexed by (name, Date) with all input features,
    after imputing missing values and adding lag columns.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Column-wise median imputation for isolated NaNs in LST / NDVI / EVI,
    # computed per node via linear interpolation then forward/back fill.
    interp_cols = [c for c in ["LST_Day_1km_mean", "LST_Night_1km_mean", "NDVI_mean", "EVI_mean"] if c in env_vars]
    for col in interp_cols:
        df[col] = df.groupby("name")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    # W.P. Labuan has all-NaN ERA5 columns — fill with global column mean.
    era5_cols = [
        c for c in env_vars
        if c not in interp_cols and c != "precipitation_sum"
    ]
    for col in era5_cols:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean)

    # Remaining scattered NaNs: forward then backward fill per node.
    for col in env_vars:
        df[col] = df.groupby("name")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # LULC columns are annual (MODIS LC_Type1) — same cadence as WorldPop.
    # Collected here so they join ANNUAL_VARS: no lags, optional via use_lulc.
    lulc_cols: list[str] = []
    if use_lulc:
        lulc_cols = [c for c in df.columns if c.startswith("LC_Type1_pct_class")]
        for col in lulc_cols:
            df[col] = df[col].fillna(0.0)

    annual_cols = population_vars + lulc_cols  # all annually-varying, no lags

    # Add lag features per node (env vars only).
    groups = []
    for name, grp in df.groupby("name", sort=False):
        grp = grp.sort_values("Date").copy()
        grp = _add_lags(grp, env_vars, lag_steps)
        groups.append(grp)
    df = pd.concat(groups, ignore_index=True)

    # Drop rows where any lag is NaN (first max_lag rows per node).
    max_lag = max(lag_steps)
    df = df.sort_values(["name", "Date"])
    df = df[df.groupby("name").cumcount() >= max_lag].reset_index(drop=True)

    lag_cols = [f"{v}_lag{l}" for v in env_vars for l in lag_steps]
    feature_cols = env_vars + lag_cols + annual_cols

    # Final NaN safety net — fill any remaining with 0 after scaling.
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df, feature_cols


# ---------------------------------------------------------------------------
# Build 3-D tensor [T_months, N_nodes, F_features]
# ---------------------------------------------------------------------------

def to_node_time_matrix(
    df: pd.DataFrame,
    node_order: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Pivot to shape [T, N, F] and also return IR matrix [T, N].
    Returns (X, IR, sorted_dates).
    """
    dates = sorted(df["Date"].unique())
    N, F, T = len(node_order), len(feature_cols), len(dates)

    X  = np.zeros((T, N, F), dtype=np.float32)
    IR = np.full((T, N), np.nan, dtype=np.float32)

    node_idx = {n: i for i, n in enumerate(node_order)}
    date_idx = {d: i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        t = date_idx[row["Date"]]
        n = node_idx[row["name"]]
        X[t, n, :] = [row[c] for c in feature_cols]
        IR[t, n]   = row["IR"] if pd.notna(row["IR"]) else np.nan

    return X, IR, [str(d.date()) for d in dates]


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

def make_windows(
    X: np.ndarray,         # [T, N, F]
    IR: np.ndarray,        # [T, N]
    dates: list[str],
    lookback: int,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Slide a window of length `lookback` across T.
    Window i:  x = X[i : i+lookback],  y = IR[i+lookback],  date = dates[i+lookback]
    Returns windows [W, lookback, N, F], y [W, N], mask [W, N], target_dates [W].
    """
    T = X.shape[0]
    n_windows = T - lookback
    assert n_windows > 0, f"Not enough time steps: T={T}, lookback={lookback}"

    N, F = X.shape[1], X.shape[2]
    windows = np.zeros((n_windows, lookback, N, F), dtype=np.float32)
    y_arr   = np.zeros((n_windows, N), dtype=np.float32)
    mask    = np.zeros((n_windows, N), dtype=np.float32)
    tdates  = []

    for i in range(n_windows):
        windows[i] = X[i : i + lookback]
        ir_t = IR[i + lookback].copy()
        valid = ~np.isnan(ir_t)
        if log_transform:
            ir_t_clean = np.where(valid, np.log1p(ir_t), 0.0)
        else:
            ir_t_clean = np.where(valid, ir_t, 0.0)
        y_arr[i]  = ir_t_clean
        mask[i]   = valid.astype(np.float32)
        tdates.append(dates[i + lookback])

    return windows, y_arr, mask, tdates


# ---------------------------------------------------------------------------
# Fully connected graph
# ---------------------------------------------------------------------------

def build_fully_connected_graph(N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    N*(N-1) directed edges (every node → every other node), unit weights.
    GAT attention will learn which connections matter.
    """
    src, dst = zip(*[(i, j) for i in range(N) for j in range(N) if i != j])
    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.ones(len(src), dtype=torch.float32)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

def _save_split(
    path: Path,
    windows: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    target_dates: list[str],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_names: list[str],
    feature_names: list[str],
) -> None:
    torch.save(
        {
            "windows":       torch.from_numpy(windows),
            "y":             torch.from_numpy(y),
            "mask":          torch.from_numpy(mask),
            "edge_index":    edge_index,
            "edge_weight":   edge_weight,
            "node_names":    node_names,
            "feature_names": feature_names,
            "target_dates":  target_dates,
        },
        path,
    )
    print(f"  Saved {path}  ({windows.shape[0]} windows, shape {windows.shape})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Handle both standalone (argparse) and Snakemake invocation ----
    if "snakemake" in dir():
        sm = snakemake  # noqa: F821
        args = argparse.Namespace(
            config   = sm.input.config_file,
            input    = sm.input.data_csv,
            out_dir  = sm.params.out_dir,
            run_name = sm.params.run_name,
        )
    else:
        args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    seq_cfg    = cfg["sequence"]
    tgt_cfg    = cfg["target"]
    split_cfg  = cfg["split"]
    feat_cfg   = cfg["features"]

    lookback      = seq_cfg["lookback_steps"]
    test_year     = split_cfg["test_year"]
    val_fraction  = split_cfg.get("val_fraction", 0.15)
    log_transform = tgt_cfg.get("log_transform", True)
    env_vars        = feat_cfg["env_vars"]
    population_vars = feat_cfg.get("population_vars", [])
    lag_steps       = feat_cfg.get("lag_steps", [1, 2, 3])
    use_lulc      = feat_cfg.get("use_landuse", False)

    # ---- Load and inspect ----
    print(f"[preprocess] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df["Date"] = pd.to_datetime(df["Date"])

    node_order = sorted(df["name"].unique())
    N = len(node_order)
    print(f"  Nodes: {N}  |  Months: {df['Date'].nunique()}  |  IR missing: "
          f"{df['IR'].isna().mean()*100:.1f}%")

    # ---- Validate columns ----
    required_cols = env_vars + population_vars + ["IR", "name", "Date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[preprocess] Columns declared in config but absent from CSV:\n  {missing}\n"
            f"CSV columns: {sorted(df.columns.tolist())}"
        )

    # ---- Feature engineering ----
    print("[preprocess] Building features and lag columns ...")
    df_feat, feature_cols = build_feature_matrix(df, env_vars, population_vars, lag_steps, use_lulc)
    print(f"  Features: {len(feature_cols)}  |  Time steps after lag trim: "
          f"{df_feat['Date'].nunique()}")

    # ---- Pivot to [T, N, F] ----
    X_raw, IR_raw, dates = to_node_time_matrix(df_feat, node_order, feature_cols)
    T = X_raw.shape[0]
    print(f"  Matrix shape: {X_raw.shape}  |  Date range: {dates[0]} → {dates[-1]}")

    # ---- Sliding windows ----
    windows, y_all, mask_all, target_dates = make_windows(
        X_raw, IR_raw, dates, lookback, log_transform
    )
    W = windows.shape[0]
    print(f"  Total windows: {W}")

    # ---- Split by target year ----
    td_years = [int(d[:4]) for d in target_dates]
    test_idx  = [i for i, yr in enumerate(td_years) if yr == test_year]
    train_val = [i for i, yr in enumerate(td_years) if yr < test_year]

    n_val   = max(1, int(len(train_val) * val_fraction))
    val_idx   = train_val[-n_val:]
    train_idx = train_val[:-n_val]

    print(f"  Split — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # ---- Fit scaler on training data only ----
    print("[preprocess] Fitting StandardScaler on train windows ...")
    train_windows_flat = windows[train_idx].reshape(-1, windows.shape[-1])
    scaler = StandardScaler()
    scaler.fit(train_windows_flat)

    def _scale(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        scaled = scaler.transform(arr.reshape(-1, orig_shape[-1]))
        return scaled.reshape(orig_shape).astype(np.float32)

    windows_train = _scale(windows[train_idx])
    windows_val   = _scale(windows[val_idx])
    windows_test  = _scale(windows[test_idx])

    # ---- Save scaler for inference ----
    scaler_path = out_dir / f"{args.run_name}_scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  Scaler saved → {scaler_path}")

    # ---- Build fully connected graph ----
    print(f"[preprocess] Building fully-connected graph ({N}×{N-1} = {N*(N-1)} edges) ...")
    edge_index, edge_weight = build_fully_connected_graph(N)

    # ---- Save splits ----
    print("[preprocess] Saving tensor files ...")
    _save_split(
        out_dir / f"{args.run_name}_train.pt",
        windows_train, y_all[train_idx], mask_all[train_idx],
        [target_dates[i] for i in train_idx],
        edge_index, edge_weight, node_order, feature_cols,
    )
    _save_split(
        out_dir / f"{args.run_name}_val.pt",
        windows_val, y_all[val_idx], mask_all[val_idx],
        [target_dates[i] for i in val_idx],
        edge_index, edge_weight, node_order, feature_cols,
    )
    _save_split(
        out_dir / f"{args.run_name}_test.pt",
        windows_test, y_all[test_idx], mask_all[test_idx],
        [target_dates[i] for i in test_idx],
        edge_index, edge_weight, node_order, feature_cols,
    )

    # ---- Summary JSON ----
    summary = {
        "run_name":      args.run_name,
        "n_nodes":       N,
        "n_features":    len(feature_cols),
        "lookback":      lookback,
        "n_train":       len(train_idx),
        "n_val":         len(val_idx),
        "n_test":        len(test_idx),
        "test_year":     test_year,
        "log_transform": log_transform,
        "graph_type":    "fully_connected",
        "n_edges":       N * (N - 1),
        "feature_names": feature_cols,
        "node_names":    node_order,
    }
    summary_path = out_dir / f"{args.run_name}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[preprocess] Done. Summary → {summary_path}")


if __name__ == "__main__":
    main()
