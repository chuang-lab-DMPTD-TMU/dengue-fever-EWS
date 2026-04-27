"""
scripts/eval_RF.py
------------------
Snakemake Step 3 (per region): Evaluate a trained Random Forest classifier.

Wildcards : {study}, {region}
Reads     : models/rf/{study}/{region}.joblib
Writes    : reports/tables/{study}/{region}_rf_results.csv
            reports/figures/{study}/{region}/confusion_matrix_rf_{split}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    load_data,
    resolve_paths,
    split_train_test,
)

# ---------------------------------------------------------------------------
cfg: dict       = snakemake.config             # noqa: F821
run_cfg: dict   = snakemake.params.run_cfg     # noqa: F821
region: str     = snakemake.wildcards.region   # noqa: F821
study: str      = snakemake.wildcards.study    # noqa: F821

cfg["use_landuse"] = run_cfg.get("landuse") == "t"

model_path  = Path(snakemake.input.model)         # noqa: F821
results_csv = Path(snakemake.output.results_csv)   # noqa: F821
results_csv.parent.mkdir(parents=True, exist_ok=True)

paths = resolve_paths(cfg)
region_dir = paths["figures_dir"] / study / region
region_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg, run_cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg["target"]["column"]

df_train_val, df_test = split_train_test(df, variable_columns, cfg, region=region)
num_classes  = int(df[target].nunique())
class_labels = list(range(num_classes))

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = joblib.load(str(model_path))
print(f"[eval_rf:{region}] Loaded model from {model_path}")

# ---------------------------------------------------------------------------
# Evaluate each split
# ---------------------------------------------------------------------------
datasets = {
    "Train_Val": (df_train_val[variable_columns], df_train_val[target].values.flatten()),
    "Test":      (df_test[variable_columns],      df_test[target].values.flatten()),
}

all_results = []

for set_name, (X_set, y_set) in datasets.items():
    print(f"\n[eval_rf:{region}] --- {set_name} ---")

    y_pred = model.predict(X_set)

    accuracy      = accuracy_score(y_set, y_pred)
    report        = classification_report(y_set, y_pred, output_dict=True)
    target_counts = pd.Series(y_set).value_counts().to_dict()
    cm            = confusion_matrix(y_set, y_pred, labels=class_labels)

    print(f"  Accuracy: {accuracy:.4f} | Counts: {target_counts}")

    # Confusion matrix plot
    fig_size = max(4, num_classes * 2)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {region} ({set_name})")
    cm_path = region_dir / f"confusion_matrix_rf_{set_name}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {cm_path}")

    row: dict = {"Region": region, "Dataset": set_name, "Accuracy": accuracy}
    for i in class_labels:
        row[f"Class_{i}_count"] = target_counts.get(i, 0)
        if str(i) in report:
            row[f"Precision_{i}"] = report[str(i)]["precision"]
            row[f"Recall_{i}"]    = report[str(i)]["recall"]
            row[f"F1_{i}"]        = report[str(i)]["f1-score"]
    row["Macro_Precision"]    = report["macro avg"]["precision"]
    row["Macro_Recall"]       = report["macro avg"]["recall"]
    row["Macro_F1"]           = report["macro avg"]["f1-score"]
    row["Weighted_Precision"] = report["weighted avg"]["precision"]
    row["Weighted_Recall"]    = report["weighted avg"]["recall"]
    row["Weighted_F1"]        = report["weighted avg"]["f1-score"]
    all_results.append(row)

# ---------------------------------------------------------------------------
# Save per-region CSV
# ---------------------------------------------------------------------------
pd.DataFrame(all_results).to_csv(results_csv, index=False, float_format="%.4f")
print(f"\n[eval_rf:{region}] Saved results → {results_csv}")
