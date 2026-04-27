# ============================================================
# rules/RF.smk — Random Forest pipeline (tune → train → evaluate → aggregate)
# ============================================================
# Mirrors the XGBoost pipeline structure.  Uses the same {study}/{region}
# wildcards and the discover_regions checkpoint defined in xgboost.smk.
#
# Pipeline per (study, region):
#   [checkpoint] discover_regions   ← defined in xgboost.smk
#       → tune_rf  →  train_rf  →  eval_rf  →  aggregate_rf_results
# ============================================================


# ---------------------------------------------------------------------------
# Step 1 — Hyperparameter search (RandomizedSearchCV), per region
# ---------------------------------------------------------------------------

rule tune_rf:
    """
    RandomizedSearchCV hyperparameter search for one (study, region).
    Writes the best params to a per-region CSV.
    If run_cfg contains a 'hyperparams' key, search is skipped
    and those fixed params are written directly.
    """
    input:
        region_list = "reports/tables/{study}_regions.txt"
    output:
        params_csv = "reports/tables/{study}/{region}_rf_params.csv"
    log:
        "logs/{study}/{region}_rf_tune.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    threads: 8
    script:
        "../scripts/tune_RF.py"


# ---------------------------------------------------------------------------
# Step 2 — Train final model, per region
# ---------------------------------------------------------------------------

rule train_rf:
    """
    Train the final Random Forest model for one (study, region).
    Saves a joblib file.
    """
    input:
        params_csv = "reports/tables/{study}/{region}_rf_params.csv"
    output:
        model = "models/rf/{study}/{region}.joblib"
    log:
        "logs/{study}/{region}_rf_train.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    threads: 8
    script:
        "../scripts/train_RF.py"


# ---------------------------------------------------------------------------
# Step 3 — Evaluate, per region
# ---------------------------------------------------------------------------

rule eval_rf:
    """
    Classification report + confusion matrix for one (study, region).
    """
    input:
        model = "models/rf/{study}/{region}.joblib"
    output:
        results_csv = "reports/tables/{study}/{region}_rf_results.csv"
    log:
        "logs/{study}/{region}_rf_evaluate.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    script:
        "../scripts/eval_RF.py"


# ---------------------------------------------------------------------------
# Step 4 — Aggregate all per-region CSVs → one national summary
# ---------------------------------------------------------------------------

rule aggregate_rf_results:
    """
    Concatenate every region's RF results CSV into a single national summary.
    Input function triggers the discover_regions checkpoint so all regions
    are resolved before aggregation is planned.
    """
    input:
        csvs = lambda wc: expand(
            "reports/tables/{study}/{region}_rf_results.csv",
            study  = wc.study,
            region = _regions_for_study(wc)
        )
    output:
        summary = "reports/tables/{study}_rf_national_results.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(f) for f in input.csvs]
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(output.summary, index=False, float_format="%.4f")
        print(f"[aggregate_rf] {len(dfs)} regions → {output.summary}")
