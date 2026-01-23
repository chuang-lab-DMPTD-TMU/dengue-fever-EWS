#!/usr/bin/env python3
"""
final_train_eval.py
-------------------
Loads data, loads best hyperparameters, trains final XGBoost model,
generates evaluation metrics, confusion matrices, and SHAP plots.
"""

import argparse
import pandas as pd
import numpy as np
import xgboost
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main(args):
    prepared = Path(args.prepared_dir)
    optuna_dir = Path(args.optuna_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_dir = outdir / "figures"
    test_dir.mkdir(exist_ok=True, parents=True)

    df_train = pd.read_csv(prepared / "train.csv")
    df_test = pd.read_csv(prepared / "test.csv")
    variable_columns = (prepared / "variable_columns.txt").read_text().splitlines()

    num_classes = df_train["Risk_Category"].nunique()

    # Load tuned hyperparameters
    best = pd.read_csv(optuna_dir / "best_params.csv").iloc[0].to_dict()

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": best["learning_rate"],
        "max_depth": int(best["max_depth"]),
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": int(best["min_child_weight"]),
        "gamma": best["gamma"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "eval_metric": "mlogloss"
    }
    n_round = int(best["n_estimators"])

    X_tr = df_train[variable_columns]
    y_tr = df_train["Risk_Category"]
    X_te = df_test[variable_columns]
    y_te = df_test["Risk_Category"]

    dtrain = xgboost.DMatrix(X_tr, label=y_tr)
    dtest = xgboost.DMatrix(X_te)

    model = xgboost.train(params, dtrain, num_boost_round=n_round)
    model.save_model(str(outdir / "final_model.json"))

    # Predictions
    pred_prob = model.predict(dtest)
    pred = pred_prob.argmax(axis=1)

    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, output_dict=True)

    # Save evaluation
    json.dump(report, open(outdir / "classification_report.json", "w"), indent=2)
    (outdir / "accuracy.txt").write_text(str(acc))

    # Confusion matrix
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix (Test)")
    plt.savefig(outdir / "confusion_matrix_test.png", dpi=200)
    plt.close()

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    shap.summary_plot(shap_values, X_te, show=False)
    plt.savefig(outdir / "shap_beeswarm.png", dpi=200)
    plt.close()

    print("✅ final_train_eval.py completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--optuna_dir", required=True)
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args())
