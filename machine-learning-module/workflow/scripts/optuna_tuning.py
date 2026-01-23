#!/usr/bin/env python3
"""
optuna_tuning.py
----------------
Loads training data produced by prepare_data.py,
runs Optuna-based hyperparameter tuning using walk-forward
(or k-fold) validation, and writes best hyperparameters to disk.
"""

import argparse
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import cudf
import gc
import xgboost
from sklearn.metrics import accuracy_score


def calculate_weights(y):
    """Compute inverse-frequency class weights."""
    unique, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
    return np.array([w[v] for v in y])


def get_walk_forward_splits(df, test_window=12, train_window=None):
    """Walk-forward splits for gpu/cudf DataFrame."""
    df = df.sort_values("YearMonth").reset_index(drop=True)
    times = df["YearMonth"].unique()
    n = len(times)

    init_train = 36 if train_window is None else train_window
    folds = (n - init_train) // test_window
    all_splits = []

    for f in range(folds):
        test_start_i = init_train + f * test_window
        test_end_i = test_start_i + test_window

        train_start_time = times[0] if train_window is None else times[max(0, test_start_i - train_window)]
        train_end_time = times[test_start_i]
        test_start_time = times[test_start_i]
        test_end_time = times[test_end_i]

        train_start = df["YearMonth"].searchsorted(train_start_time)
        train_end = df["YearMonth"].searchsorted(train_end_time)
        test_start = df["YearMonth"].searchsorted(test_start_time)
        test_end = df["YearMonth"].searchsorted(test_end_time)

        train_idx = cudf.RangeIndex(train_start, train_end)
        test_idx = cudf.RangeIndex(test_start, test_end)

        if len(train_idx) > 0 and len(test_idx) > 0:
            all_splits.append((f, train_idx, test_idx))
    return all_splits


def objective(trial, X, y, splits, num_classes):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
        "gamma": trial.suggest_float("gamma", 0.1, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True),
        "eval_metric": "mlogloss"
    }
    n_round = trial.suggest_int("n_estimators", 50, 1000)

    all_preds = []
    all_true = []

    for fold, train_idx, test_idx in splits:
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        w = calculate_weights(y_tr.to_numpy())
        dtrain = xgboost.DMatrix(X_tr, label=y_tr.to_numpy(), weight=w)
        dval = xgboost.DMatrix(X_te, label=y_te.to_numpy())

        model = xgboost.train(params, dtrain, num_boost_round=n_round, evals=[(dval, "val")], verbose_eval=False)

        pred_prob = model.predict(dval)
        pred = pred_prob.argmax(axis=1)

        all_preds.append(pred)
        all_true.append(y_te.to_numpy())

        del model, dtrain, dval
        gc.collect()

    if not all_true:
        return 0.0

    acc = accuracy_score(np.concatenate(all_true), np.concatenate(all_preds))
    return acc


def main(args):
    prepared = Path(args.prepared_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(prepared / "train.csv")
    var_cols = (prepared / "variable_columns.txt").read_text().splitlines()

    # Convert to cudf
    cdf = cudf.DataFrame(df_train)
    X = cdf[var_cols]
    y = cdf["Risk_Category"].astype("int32")

    splits = get_walk_forward_splits(cdf, test_window=12)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y, splits, num_classes=df_train["Risk_Category"].nunique()),
                   n_trials=args.n_trials)

    best = study.best_params
    best["best_accuracy"] = study.best_value

    pd.DataFrame([best]).to_csv(outdir / "best_params.csv", index=False)
    print("✅ optuna_tuning.py completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args())
