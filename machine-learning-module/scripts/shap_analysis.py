"""
scripts/shap_analysis.py
------------------------
Snakemake Step 4 (per region): GPU SHAP values + beeswarm/dependence plots.

Wildcards : {study}, {region}
Reads     : models/xgboost/{study}/{region}.json
Writes    : reports/figures/{study}/{region}/beeswarm_plots/...
            reports/figures/{study}/{region}/dependence_plots/...
            .snakemake/flags/{study}/{region}_shap.done
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    load_data,
    log_memory,
    resolve_paths,
    split_train_test,
)

# ---------------------------------------------------------------------------
cfg: dict     = snakemake.config            # noqa: F821
run_cfg: dict = snakemake.params.run_cfg    # noqa: F821
region: str   = snakemake.wildcards.region  # noqa: F821
study: str    = snakemake.wildcards.study   # noqa: F821

cfg["use_landuse"] = run_cfg["landuse"] == "t"

model_path = Path(snakemake.input.model)  # noqa: F821
flag_out   = Path(snakemake.output.flag)  # noqa: F821
flag_out.parent.mkdir(parents=True, exist_ok=True)

paths = resolve_paths(cfg)
region_dir     = paths["figures_dir"] / study / region
beeswarm_dir   = region_dir / "beeswarm_plots"
dependence_dir = region_dir / "dependence_plots"
beeswarm_dir.mkdir(parents=True, exist_ok=True)
dependence_dir.mkdir(parents=True, exist_ok=True)

excluded_features: list[str] = cfg["features"]["epidemic_vars"]
num_top: int = cfg["training"]["num_top_shap_features"]

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg["target"]["column"]

df_train_val, df_test = split_train_test(df, variable_columns, cfg, region=region)
num_classes = int(df[target].nunique())

model = xgboost.Booster()
model.load_model(str(model_path))
print(f"[shap:{region}] Loaded model from {model_path}")

# ---------------------------------------------------------------------------
# SHAP per dataset split
# ---------------------------------------------------------------------------
datasets = {
    "Train_Val": df_train_val[variable_columns],
    "Test":      df_test[variable_columns],
}

for set_name, X_set in datasets.items():
    print(f"\n[shap:{region}] === {set_name} ===")
    log_memory(f"shap:{region} {set_name} start")

    shap_features = list(X_set.columns)
    explainer = shap.explainers.GPUTree(model, feature_perturbation="tree_path_dependent")

    try:
        shap_values = explainer.shap_values(X_set, check_additivity=False)
    except Exception as exc:
        print(f"[shap:{region}] SHAP computation failed for {set_name}: {exc}")
        continue

    # ------------------------------------------------------------------
    # Top N features per class (excluding epidemic vars)
    # ------------------------------------------------------------------
    top_features_by_class: list[list[str]] = []
    for class_idx in range(num_classes):
        sv = shap_values[class_idx] if num_classes > 2 else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        sorted_feats = [shap_features[i] for i in mean_abs.argsort()[::-1]]
        top = [f for f in sorted_feats if f not in excluded_features][:num_top]
        top_features_by_class.append(top)
        print(f"  Class {class_idx} top features: {top}")

    # ------------------------------------------------------------------
    # Beeswarm plots
    # ------------------------------------------------------------------
    for class_idx in range(num_classes):
        sv = shap_values[class_idx] if num_classes > 2 else shap_values
        included_cols = [c for c in X_set.columns if c not in excluded_features]
        included_idx  = [i for i, c in enumerate(X_set.columns) if c in included_cols]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv[:, included_idx], X_set[included_cols], show=False)
        plt.title(f"SHAP Beeswarm — Class {class_idx} ({region}, {set_name})")
        out = beeswarm_dir / f"class_{class_idx}_beeswarm({set_name}).png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  [OK] Beeswarm class {class_idx} → {out.name}")

    # ------------------------------------------------------------------
    # Dependence plots
    # ------------------------------------------------------------------
    for class_idx in range(num_classes):
        sv = shap_values[class_idx] if num_classes > 2 else shap_values
        for feature in top_features_by_class[class_idx]:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, sv, X_set, interaction_index="auto", show=False)
            plt.title(f"SHAP Dependence — {feature}, Class {class_idx} ({region}, {set_name})")
            out = dependence_dir / f"class_{class_idx}_{feature}_dependence({set_name}).png"
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"  [OK] Dependence {feature} class {class_idx} → {out.name}")

    log_memory(f"shap:{region} {set_name} end")

# ---------------------------------------------------------------------------
# Write completion flag
# ---------------------------------------------------------------------------
flag_out.write_text("done")
print(f"\n[shap:{region}] Wrote flag → {flag_out}")
