"""
scripts/train_RF.py
-------------------
Snakemake Step 1 (per region): Train a Random Forest classifier.

Hyperparameters are read from run_cfg['hyperparams'] in config.yaml.
If absent, sklearn defaults are used.

Wildcards : {study}, {region}
Reads     : reports/tables/{study}_regions.txt
Writes    : models/rf/{study}/{region}.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_feature_columns,
    calculate_sample_weights,
    load_data,
    split_train_test,
)

# ---------------------------------------------------------------------------
cfg: dict       = snakemake.config             # noqa: F821
run_cfg: dict   = snakemake.params.run_cfg     # noqa: F821
region: str     = snakemake.wildcards.region   # noqa: F821
study: str      = snakemake.wildcards.study    # noqa: F821

cfg["use_landuse"] = run_cfg.get("landuse") == "t"

model_out = Path(snakemake.output.model)        # noqa: F821
model_out.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — from run_cfg or sklearn defaults
# ---------------------------------------------------------------------------
hp = dict(run_cfg.get("hyperparams", {}))
print(f"[train_rf:{region}] Hyperparams: {hp if hp else 'sklearn defaults'}")

# ---------------------------------------------------------------------------
# Load data, filter to region
# ---------------------------------------------------------------------------
df = load_data(cfg, run_cfg)
variable_columns = build_feature_columns(df, cfg)
target = cfg["target"]["column"]

df_train_val, _ = split_train_test(df, variable_columns, cfg, region=region)

X_train = df_train_val[variable_columns].values
y_train = df_train_val[target].values.flatten()
sample_weights = calculate_sample_weights(y_train)

print(f"[train_rf:{region}] {len(X_train)} samples, {len(variable_columns)} features")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    **hp,
    n_jobs=snakemake.threads,   # noqa: F821
    random_state=cfg["training"].get("random_state", 64),
)
model.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(model, model_out)
print(f"[train_rf:{region}] Saved model → {model_out}")
