"""
scripts/tune_xgboost.py
-----------------------
Snakemake Step 1 (per region): Optuna hyperparameter search for XGBoost.

Wildcards : {study}, {region}
Reads     : reports/tables/{study}_regions.txt
Writes    : reports/tables/{study}/{region}_params.csv

Validation strategy (from run_cfg):
  'w' → walk-forward expanding window (default)
  'k' → k-fold cross-validation

If run_cfg contains a 'hyperparams' key, Optuna is skipped and those
fixed values are written directly to the params CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import optuna
import pandas as pd
import xgboost
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    calculate_sample_weights,
    load_data,
    split_train_test,
    validation_strategy,
)

# ---------------------------------------------------------------------------
cfg: dict       = snakemake.config             # noqa: F821
run_cfg: dict   = snakemake.params.run_cfg     # noqa: F821
region: str     = snakemake.wildcards.region   # noqa: F821
study: str      = snakemake.wildcards.study    # noqa: F821

cfg["use_landuse"] = run_cfg.get("landuse") == "t"

params_csv = Path(snakemake.output.params_csv)  # noqa: F821
params_csv.parent.mkdir(parents=True, exist_ok=True)

n_trials   = cfg.get("optuna", {}).get("n_trials", 40)
val_strat  = validation_strategy(run_cfg)          # 'walk' | 'kfold'
rand_state = cfg["training"].get("random_state", 64)

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg, run_cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg["target"]["column"]

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)
num_classes = int(df[target].nunique())

print(f"[tune_xgb:{region}] {len(df_train_val)} samples | strategy={val_strat} | trials={n_trials}")

# ---------------------------------------------------------------------------
# If hyperparams are fixed, skip Optuna and write directly
# ---------------------------------------------------------------------------
fixed = run_cfg.get("hyperparams")
if fixed:
    hp = dict(fixed)
    print(f"[tune_xgb:{region}] Fixed hyperparams from run_cfg — skipping Optuna")
    pd.DataFrame([{"Region": region, **hp}]).to_csv(params_csv, index=False)
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Build CV splits
# ---------------------------------------------------------------------------
def _walk_forward_splits(df_sorted: pd.DataFrame, test_window: int = 12, init_window: int = 36):
    unique_months = sorted(df_sorted["YearMonth"].unique())
    splits = []
    for i in range(init_window, len(unique_months) - test_window + 1, test_window):
        train_months = set(unique_months[:i])
        val_months   = set(unique_months[i : i + test_window])
        tr_idx = df_sorted.index[df_sorted["YearMonth"].isin(train_months)].tolist()
        val_idx = df_sorted.index[df_sorted["YearMonth"].isin(val_months)].tolist()
        if tr_idx and val_idx:
            splits.append((tr_idx, val_idx))
    return splits


df_sorted = df_train_val.sort_values("YearMonth").reset_index(drop=True)
X_all = df_sorted[variable_columns]
y_all = df_sorted[target]
test_window = cfg["training"].get("walk_forward_test_window", 12)

if val_strat == "walk":
    cv_splits = _walk_forward_splits(df_sorted, test_window=test_window)
else:
    kf = KFold(n_splits=5, shuffle=False)
    cv_splits = list(kf.split(X_all))

print(f"[tune_xgb:{region}] {len(cv_splits)} CV folds")

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective":        "multi:softprob" if num_classes > 2 else "binary:logistic",
        "num_class":        num_classes if num_classes > 2 else None,
        "tree_method":      "hist",
        "device":           "cuda",
        "seed":             rand_state,
        "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "gamma":            trial.suggest_float("gamma", 1e-2, 10.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
    }
    # Remove None values (binary case has no num_class)
    params = {k: v for k, v in params.items() if v is not None}
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    all_preds, all_true = [], []
    for tr_idx, val_idx in cv_splits:
        X_tr  = X_all.iloc[tr_idx]
        y_tr  = y_all.iloc[tr_idx]
        X_val = X_all.iloc[val_idx]
        y_val = y_all.iloc[val_idx]

        sw     = calculate_sample_weights(y_tr.values)
        dtrain = xgboost.DMatrix(X_tr, label=y_tr, weight=sw)
        dval   = xgboost.DMatrix(X_val, label=y_val)

        booster = xgboost.train(
            params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            verbose_eval=False,
        )
        preds_prob = booster.predict(dval)
        preds = preds_prob.argmax(axis=1) if num_classes > 2 else (preds_prob > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_true.extend(y_val.values.tolist())

    return f1_score(all_true, all_preds, average="macro", zero_division=0)


study_obj = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rand_state))
study_obj.optimize(objective, n_trials=n_trials, show_progress_bar=False)

best = study_obj.best_params
best_score = study_obj.best_value
print(f"[tune_xgb:{region}] Best f1_macro={best_score:.4f} | params={best}")

# ---------------------------------------------------------------------------
# Write params CSV
# ---------------------------------------------------------------------------
pd.DataFrame([{"Region": region, **best}]).to_csv(params_csv, index=False)
print(f"[tune_xgb:{region}] Params → {params_csv}")
