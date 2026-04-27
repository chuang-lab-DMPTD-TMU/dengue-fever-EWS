"""
scripts/utils.py
----------------
Shared helpers for the dengue XGBoost pipeline.
All scripts import from here so path logic and feature
engineering stay consistent across steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import cudf
import numpy as np
import pandas as pd
import pynvml
import psutil


# ---------------------------------------------------------------------------
# Memory logging
# ---------------------------------------------------------------------------

pynvml.nvmlInit()


def log_memory(tag: str = "") -> None:
    """Print current system RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 ** 2
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_used_mb = gpu_mem.used / 1024 ** 2
    print(f"[{tag}] RAM: {ram_mb:.2f} MB | GPU: {gpu_used_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def find_project_root(start: Path, marker: str = ".git") -> Path:
    for parent in start.resolve().parents:
        if (parent / marker).exists():
            return parent
    return start.resolve()


def resolve_paths(cfg: dict) -> dict[str, Path]:
    """Return absolute Path objects for every path key in config['paths']."""
    root = find_project_root(Path(__file__))
    return {k: root / v for k, v in cfg["paths"].items()}


# ---------------------------------------------------------------------------
# Run metadata helpers
# ---------------------------------------------------------------------------

def run_id(run_cfg: dict) -> str:
    return f"{run_cfg.get('validation', 'w')}{run_cfg.get('landuse', 'f')}{run_cfg['version']}"


def landuse_suffix(run_cfg: dict) -> str:
    return "lulc" if run_cfg.get("landuse") == "t" else ""


def validation_strategy(run_cfg: dict) -> str:
    return "kfold" if run_cfg.get("validation") == "k" else "walk"


def study_name(run_cfg: dict) -> str:
    """
    Study identifier — matches the {study} wildcard in the Snakefile.
    Must stay in sync with _study_name() in Snakefile.
    """
    arch = run_cfg.get("arch", "xgbC")
    parts = [arch, validation_strategy(run_cfg), landuse_suffix(run_cfg), run_cfg["version"]]
    return "-".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """
    Apply lag features and return the final ordered feature column list,
    respecting the USE_LANDUSE_FEATURES flag from the run config.
    """
    feat_cfg = cfg["features"]
    env_vars = feat_cfg["env_vars"]
    climate_vars = feat_cfg["climate_vars"]
    epidemic_vars = feat_cfg["epidemic_vars"]
    land_use_vars = feat_cfg["land_use_vars"]
    lag_steps = feat_cfg["lag_steps"]
    use_landuse = cfg.get("use_landuse", True)  # injected by scripts

    # Add lag 1 epidemic feature
    df["Incidence_Rate_lag1"] = df.groupby("ID_2")["Incidence_Rate"].shift(1)

    # Add lags for env + climate groups
    for var in env_vars + climate_vars:
        for lag in lag_steps:
            col = f"{var}_lag{lag}"
            if col not in df.columns:
                df[col] = df.groupby("ID_2")[var].shift(lag)

    # Build ordered feature list
    variable_columns: list[str] = []

    for var in env_vars + climate_vars + epidemic_vars:
        if var in df.columns:
            variable_columns.append(var)

    if use_landuse:
        for var in land_use_vars:
            if var in df.columns:
                variable_columns.append(var)

    for var in env_vars + climate_vars:
        for lag in lag_steps:
            col = f"{var}_lag{lag}"
            if col in df.columns:
                variable_columns.append(col)

    target = cfg["target"]["column"]
    meta = ["YearMonth", "ID_2", "Region_Group", "Incidence_Rate"]
    variable_columns = [c for c in variable_columns if c not in meta + [target]]

    return variable_columns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict, run_cfg: dict | None = None) -> pd.DataFrame:
    """
    Load the processed CSV, encode the target, and return a sorted DataFrame.

    If ``run_cfg`` contains a ``data`` key it is used as the CSV path
    (relative to the project root), overriding ``cfg["paths"]["data_processed"]``.
    """
    paths = resolve_paths(cfg)
    if run_cfg and run_cfg.get("data"):
        root = find_project_root(Path(__file__))
        data_path = root / run_cfg["data"]
    else:
        data_path = paths["data_processed"]
    df = pd.read_csv(data_path)

    target_col = cfg["target"]["column"]
    class_map = cfg["target"]["class_map"]
    df[target_col] = df[target_col].replace(class_map).infer_objects(copy=False)
    df[target_col] = df[target_col].astype("int32")

    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    df = df.sort_values(["YearMonth", "ID_2"])
    return df


def split_train_test(
    df: pd.DataFrame,
    variable_columns: list[str],
    cfg: dict,
    region: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val and held-out test by year.

    Parameters
    ----------
    region : str | None
        If provided, filter the DataFrame to rows where Region_Group == region
        before splitting.  Pass None to use all regions (e.g. for a nationwide
        model).
    """
    test_year = cfg["split"]["test_year"]
    target = cfg["target"]["column"]
    keep_cols = variable_columns + ["YearMonth", "ID_2", "Region_Group", "Incidence_Rate", target]

    df_region = df[df["Region_Group"] == region].copy() if region is not None else df.copy()

    df_train_val = (
        df_region[df_region["YearMonth"].dt.year < test_year]
        .dropna(subset=variable_columns + [target])
    )
    df_test = (
        df_region[df_region["YearMonth"].dt.year == test_year]
        .dropna(subset=variable_columns + [target])
    )
    return df_train_val[keep_cols], df_test[keep_cols]


def to_cudf(df: pd.DataFrame, columns: list[str]) -> cudf.DataFrame:
    return cudf.DataFrame(df[columns])


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for class imbalance."""
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(unique_classes)
    weights = {cls: total / (n_cls * cnt) for cls, cnt in zip(unique_classes, counts)}
    return np.array([weights[c] for c in y])


# ---------------------------------------------------------------------------
# Walk-forward splits (GPU-based)
# ---------------------------------------------------------------------------

def get_walk_forward_splits(
    df_cudf: cudf.DataFrame,
    test_window: int,
    train_window: int | None = None,
    initial_train_window: int = 36,
) -> list[tuple[int, cudf.RangeIndex, cudf.RangeIndex]]:
    """
    Generate expanding-window (or rolling-window) train/test index pairs.

    Parameters
    ----------
    df_cudf : cudf.DataFrame  (must contain 'YearMonth' column)
    test_window : int          number of time steps per test fold
    train_window : int | None  if None → expanding window; else rolling
    initial_train_window : int minimum initial training steps

    Returns
    -------
    list of (fold_num, train_idx, test_idx)
    """
    df_sorted = df_cudf.sort_values("YearMonth").reset_index(drop=True)
    unique_time_steps = df_sorted["YearMonth"].unique().sort_values()
    n_time_steps = len(unique_time_steps)

    eff_train_window = initial_train_window if train_window is None else train_window
    num_folds = (n_time_steps - eff_train_window) // test_window

    splits = []
    for fold_num in range(num_folds):
        test_start_idx = eff_train_window + fold_num * test_window
        test_end_idx = min(test_start_idx + test_window, n_time_steps)

        test_start_time = unique_time_steps.iloc[test_start_idx]
        test_end_time = unique_time_steps.iloc[test_end_idx - 1]

        if train_window is None:
            train_start_time = unique_time_steps.iloc[0]
        else:
            train_start_pos = test_start_idx - train_window
            if train_start_pos < 0:
                print(f"[walk_forward] Fold {fold_num}: skipped (not enough train data)")
                continue
            train_start_time = unique_time_steps.iloc[train_start_pos]

        train_mask = (df_sorted["YearMonth"] >= train_start_time) & (
            df_sorted["YearMonth"] < test_start_time
        )
        test_mask = (df_sorted["YearMonth"] >= test_start_time) & (
            df_sorted["YearMonth"] <= test_end_time
        )

        train_idx = df_sorted.index[train_mask]
        test_idx = df_sorted.index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((fold_num, train_idx, test_idx))

    return splits
