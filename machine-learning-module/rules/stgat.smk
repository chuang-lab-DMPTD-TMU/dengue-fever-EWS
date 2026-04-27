# ============================================================
# rules/stgat.smk — Spatiotemporal Graph Attention Network pipeline
# ============================================================
# Graph: fully connected — GAT attention learns spatial relations.
# No coordinate data or shapefile required.
#
# Pipeline per {run_name} (config lives in config/stgat/{run_name}.yaml):
#
#   preprocess_stgat  →  tune_stgat  →  train_stgat  →  eval_stgat
#
# Run standalone example:
#   snakemake data/processed/stgat/sea_baseline_train.pt --cores 4
#   snakemake models/stgat/sea_baseline/best.pt --cores 4 --resources gpu=1
# ============================================================


# ---------------------------------------------------------------------------
# Step 0 — Preprocessing: CSV → sliding-window .pt tensor files
# ---------------------------------------------------------------------------

rule preprocess_stgat:
    """
    Reads the SEA monthly CSV, builds lag features, creates sliding-window
    [lookback, N, F] tensors, fits a StandardScaler on train data, builds a
    fully-connected graph, and writes train/val/test .pt files.
    """
    input:
        data_csv    = config.get("stgat_data",
                        "data/interim/"
                        "SEA_dengue_env_monthly_2011-2018.csv"),
        config_file = "config/stgat/{run_name}.yaml"
    output:
        train_pt = "data/processed/stgat/{run_name}_train.pt",
        val_pt   = "data/processed/stgat/{run_name}_val.pt",
        test_pt  = "data/processed/stgat/{run_name}_test.pt",
        scaler   = "data/processed/stgat/{run_name}_scaler.pkl",
        summary  = "data/processed/stgat/{run_name}_summary.json"
    log:
        "logs/stgat/{run_name}/preprocess.log"
    params:
        out_dir  = "data/processed/stgat",
        run_name = "{run_name}"
    threads: 2
    resources:
        mem_mb = 8000
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.out_dir} $(dirname {log})

        python scripts/preprocess_stgat.py \
            --config   {input.config_file} \
            --input    {input.data_csv}    \
            --out-dir  {params.out_dir}    \
            --run-name {params.run_name}   \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 1 — Hyperparameter search (Optuna)
# ---------------------------------------------------------------------------

rule tune_stgat:
    """
    Optuna hyperparameter search for one STGAT run.
    Each trial trains for a reduced number of epochs.
    Writes best_params.json consumed by train_stgat.
    """
    input:
        train_data  = "data/processed/stgat/{run_name}_train.pt",
        val_data    = "data/processed/stgat/{run_name}_val.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        best_params = "models/stgat/{run_name}/best_params.json"
    log:
        "logs/stgat/{run_name}/tune.log"
    params:
        cuda_device  = config.get("cuda_visible_devices", "0"),
        n_trials     = config.get("optuna", {}).get("n_trials", 20),
        trial_epochs = 40,
        device       = "cuda" if config.get("use_gpu", True) else "cpu"
    threads: 4
    resources:
        gpu    = 1,
        mem_mb = 16000
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.best_params}) $(dirname {log})

        export CUDA_VISIBLE_DEVICES="{params.cuda_device}"

        python scripts/tune_stgat.py \
            --config       {input.config_file}   \
            --train-data   {input.train_data}     \
            --val-data     {input.val_data}       \
            --output       {output.best_params}   \
            --n-trials     {params.n_trials}      \
            --trial-epochs {params.trial_epochs}  \
            --device       {params.device}        \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 2 — Train
# ---------------------------------------------------------------------------

rule train_stgat:
    """
    Train one STGAT run end-to-end (GAT → GRU → FC).
    Best hyperparams from tune_stgat are applied via --best-params.
    """
    input:
        train_data  = "data/processed/stgat/{run_name}_train.pt",
        val_data    = "data/processed/stgat/{run_name}_val.pt",
        config_file = "config/stgat/{run_name}.yaml",
        best_params = "models/stgat/{run_name}/best_params.json"
    output:
        checkpoint = "models/stgat/{run_name}/best.pt",
        metrics    = "reports/stgat/{run_name}/train_metrics.json"
    log:
        "logs/stgat/{run_name}/train.log"
    params:
        cuda_device = config.get("cuda_visible_devices", "0"),
        device      = "cuda" if config.get("use_gpu", True) else "cpu"
    threads: 4
    resources:
        gpu    = 1,
        mem_mb = 32000
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.checkpoint}) $(dirname {output.metrics}) $(dirname {log})

        export CUDA_VISIBLE_DEVICES="{params.cuda_device}"

        python scripts/train_stgat.py \
            --config            {input.config_file}   \
            --train-data        {input.train_data}     \
            --val-data          {input.val_data}       \
            --best-params       {input.best_params}    \
            --output-checkpoint {output.checkpoint}    \
            --output-metrics    {output.metrics}       \
            --device            {params.device}        \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 3 — Evaluate
# ---------------------------------------------------------------------------

rule eval_stgat:
    """
    Evaluate a trained STGAT checkpoint on the held-out test set.
    Writes a JSON of test metrics to reports/stgat/{run_name}/.
    """
    input:
        checkpoint  = "models/stgat/{run_name}/best.pt",
        test_data   = "data/processed/stgat/{run_name}_test.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        metrics = "reports/stgat/{run_name}/test_metrics.json"
    log:
        "logs/stgat/{run_name}/eval.log"
    params:
        cuda_device = config.get("cuda_visible_devices", "0"),
        device      = "cuda" if config.get("use_gpu", True) else "cpu"
    threads: 2
    resources:
        gpu    = 1,
        mem_mb = 16000
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.metrics}) $(dirname {log})

        export CUDA_VISIBLE_DEVICES="{params.cuda_device}"

        python scripts/eval_stgat.py \
            --config         {input.config_file}  \
            --checkpoint     {input.checkpoint}   \
            --test-data      {input.test_data}    \
            --output-metrics {output.metrics}     \
            --device         {params.device}      \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 4 — Time-curve visualisation: real vs predicted IR for 20 regions
# ---------------------------------------------------------------------------

rule plot_timecurves_stgat:
    """
    Run the trained STGAT checkpoint over all three splits (train / val / test)
    and produce a 4×5 grid of real vs. predicted IR time curves for 20 randomly
    sampled regions covering the full 2011–2018 period.
    """
    input:
        checkpoint  = "models/stgat/{run_name}/best.pt",
        train_data  = "data/processed/stgat/{run_name}_train.pt",
        val_data    = "data/processed/stgat/{run_name}_val.pt",
        test_data   = "data/processed/stgat/{run_name}_test.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        plot = "reports/stgat/{run_name}/timecurves_ir_seed{seed}.png"
    log:
        "logs/stgat/{run_name}/timecurves_seed{seed}.log"
    params:
        cuda_device = config.get("cuda_visible_devices", "0"),
        device      = "cuda" if config.get("use_gpu", True) else "cpu",
        out_dir     = "reports/stgat/{run_name}"
    threads: 2
    resources:
        gpu    = 1,
        mem_mb = 16000
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.out_dir} $(dirname {log})

        export CUDA_VISIBLE_DEVICES="{params.cuda_device}"

        python scripts/plot_timecurves_stgat.py \
            --config      {input.config_file}  \
            --checkpoint  {input.checkpoint}   \
            --train-data  {input.train_data}   \
            --val-data    {input.val_data}      \
            --test-data   {input.test_data}     \
            --output-dir  {params.out_dir}      \
            --seed        {wildcards.seed}      \
            --device      {params.device}       \
            > {log} 2>&1
        """
