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

rule tune:
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
        run_cfg   = lambda wc: _run_for_study(wc.study),
        wandb_cfg = config.get("wandb", {})
    resources:
        gpu = 1
    script:
        "../scripts/tune_xgboost.py"


# ---------------------------------------------------------------------------
# Step 2 — Train final model, per region
# ---------------------------------------------------------------------------

rule train:
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
        run_cfg   = lambda wc: _run_for_study(wc.study),
        wandb_cfg = config.get("wandb", {})
    resources:
        gpu = 1
    script:
        "../scripts/train_xgboost.py"


# ---------------------------------------------------------------------------
# Step 3 — Evaluate, per region
# ---------------------------------------------------------------------------

rule evaluate:
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
        run_cfg   = lambda wc: _run_for_study(wc.study),
        wandb_cfg = config.get("wandb", {})
    resources:
        gpu = 1
    script:
        "../scripts/eval_xgboost.py"


# ---------------------------------------------------------------------------
# Step 4 — Aggregate all per-region CSVs → one national summary
# ---------------------------------------------------------------------------

rule aggregate_results:
    """
    Concatenate every region's results CSV into a single national summary.
    This rule's input function triggers the checkpoint so all regions are
    resolved before aggregation is planned.
    """
    input:
        csvs = lambda wc: expand(
            "reports/tables/{study}/{region}_results.csv",
            study  = wc.study,
            region = _regions_for_study(wc)
        )
    output:
        summary = "reports/tables/{study}_national_results.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(f) for f in input.csvs]
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(output.summary, index=False, float_format="%.4f")
        print(f"[aggregate] {len(dfs)} regions → {output.summary}")


# ---------------------------------------------------------------------------
# Step 5 — Cross-arch comparison within an experiment
# ---------------------------------------------------------------------------

rule compare_experiments:
    """
    Concatenate national results from every study that shares the same
    experiment tag, adding a 'study' column so rows are identifiable.
    Produced automatically for any experiment that has ≥2 runs, or
    can be targeted explicitly:
      snakemake reports/tables/indo-baseline_comparison.csv
    """
    input:
        csvs = lambda wc: expand(
            "reports/tables/{study}_national_results.csv",
            study = _studies_for_experiment(wc.experiment)
        )
    output:
        comparison = "reports/tables/{experiment}_comparison.csv"
    run:
        import pandas as pd
        from pathlib import Path
        dfs = []
        for f in input.csvs:
            study_id = Path(f).stem.replace("_national_results", "")
            df = pd.read_csv(f)
            df.insert(0, "study", study_id)
            dfs.append(df)
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(output.comparison, index=False, float_format="%.4f")
        print(f"[compare] {len(dfs)} studies → {output.comparison}")
