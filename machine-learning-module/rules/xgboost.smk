# ============================================================
# rules/xgboost.smk — XGBoost pipeline (tune → train → evaluate → aggregate)
# ============================================================
# Pipeline per (study, region):
#   [checkpoint] discover_regions
#       → tune  →  train  →  evaluate  →  aggregate_results
#                                       → compare_experiments
# ============================================================


# ---------------------------------------------------------------------------
# Checkpoint: discover regions from data
# ---------------------------------------------------------------------------

checkpoint discover_regions:
    """
    Read the processed CSV and write one Region_Group per line.
    Downstream rules use this checkpoint so Snakemake re-evaluates
    the DAG after the region list is known.
    """
    output:
        region_list = "reports/tables/{study}_regions.txt"
    params:
        run_cfg = lambda wc: _run_for_study(wc.study)
    script:
        "../scripts/discover_regions.py"


# ---------------------------------------------------------------------------
# Step 1 — Hyperparameter search (Optuna), per region
# ---------------------------------------------------------------------------

rule tune_xgboost:
    """
    Optuna hyperparameter search for one (study, region).
    Writes the best params to a per-region CSV.
    If run_cfg contains a 'hyperparams' key, Optuna is skipped
    and those fixed params are written directly.
    """
    input:
        region_list = "reports/tables/{study}_regions.txt"
    output:
        params_csv = "reports/tables/{study}/{region}_params.csv"
    log:
        "logs/{study}/{region}_tune.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    resources:
        gpu = 1
    script:
        "../scripts/tune_xgboost.py"


# ---------------------------------------------------------------------------
# Step 2 — Train final model, per region
# ---------------------------------------------------------------------------

rule train_xgboost:
    """
    Train the final XGBoost model for one (study, region).
    """
    input:
        params_csv = "reports/tables/{study}/{region}_params.csv"
    output:
        model = "models/xgboost/{study}/{region}.json"
    log:
        "logs/{study}/{region}_train.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    resources:
        gpu = 1
    script:
        "../scripts/train_xgboost.py"


# ---------------------------------------------------------------------------
# Step 3 — Evaluate, per region
# ---------------------------------------------------------------------------

rule eval_xgboost:
    """
    Classification report + confusion matrix for one (study, region).
    """
    input:
        model = "models/xgboost/{study}/{region}.json"
    output:
        results_csv = "reports/tables/{study}/{region}_results.csv"
    log:
        "logs/{study}/{region}_evaluate.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    resources:
        gpu = 1
    script:
        "../scripts/eval_xgboost.py"

