# ============================================================
# rules/bayes_stgat.smk — Bayesian STGAT pipeline
# ============================================================
# Architecture: BayesianGATConv (spatial) → BayesianGRUCell (temporal)
#               → BayesianLinear (output)
# Inference:    Mean-field variational inference + KL annealing
# Loss:         ELBO = masked_mse (likelihood) + β × KL divergence
# Uncertainty:  MC sampling at val (20 passes) and test (50 passes)
#
# Pipeline per {run_name} (config in config/stgat/{run_name}.yaml):
#
#   [preprocess_stgat]  →  tune_bayes_stgat  →  train_bayes_stgat  →  eval_bayes_stgat
#
# Preprocessing is shared with the deterministic pipeline (rules/stgat.smk).
# The preprocess_stgat rule there produces the same .pt tensor files consumed here.
#
# Standalone usage (run_name = sea_bayesian):
#   snakemake data/processed/stgat/sea_bayesian_train.pt --cores 4
#   snakemake models/stgat/sea_bayesian/best.pt --cores 4 --resources gpu=1
# ============================================================


# ---------------------------------------------------------------------------
# Step 1 — Hyperparameter search (Optuna, Bayesian-aware)
# ---------------------------------------------------------------------------

rule tune_bayes_stgat:
    """
    Optuna hyperparameter search for BayesianSTGAT.
    Extends the deterministic search space with Bayesian-specific params:
      kl_weight, kl_anneal_fraction, prior_sigma, rho_init,
      mc_samples_train, free_bits.
    Monitors val_mse (not total ELBO) for trial scoring — ELBO rises during
    KL annealing even when the model is learning, so val_mse is the stable signal.
    Best params written to bayes_best_params.json (separate from deterministic
    best_params.json to allow both pipelines to run under the same run_name).
    """
    input:
        train_data  = "data/processed/stgat/{run_name}_train.pt",
        val_data    = "data/processed/stgat/{run_name}_val.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        best_params = "models/stgat/{run_name}/bayes_best_params.json"
    log:
        "logs/stgat/{run_name}/bayes_tune.log"
    params:
        cuda_device  = config.get("cuda_visible_devices", "0"),
        n_trials     = config.get("optuna", {}).get("n_trials", 20),
        trial_epochs = 40,
        n_mc_val     = 5,      # reduced MC passes per trial for speed
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
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

        python scripts/tune_bayes_stgat.py \
            --config       {input.config_file}    \
            --train-data   {input.train_data}      \
            --val-data     {input.val_data}        \
            --output       {output.best_params}    \
            --n-trials     {params.n_trials}       \
            --trial-epochs {params.trial_epochs}   \
            --n-mc-val     {params.n_mc_val}       \
            --device       {params.device}         \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 2 — Train BayesianSTGAT
# ---------------------------------------------------------------------------

rule train_bayes_stgat:
    """
    Full training run of BayesianSTGAT.
    Reads best hyperparams (including Bayesian-specific) from bayes_best_params.json.
    Early stopping monitors val_mse — not total ELBO — consistent with tuning.
    """
    input:
        train_data  = "data/processed/stgat/{run_name}_train.pt",
        val_data    = "data/processed/stgat/{run_name}_val.pt",
        config_file = "config/stgat/{run_name}.yaml",
        best_params = "models/stgat/{run_name}/bayes_best_params.json"
    output:
        checkpoint = "models/stgat/{run_name}/bayes_best.pt",
        metrics    = "reports/stgat/{run_name}/bayes_train_metrics.json"
    log:
        "logs/stgat/{run_name}/bayes_train.log"
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

        python scripts/train_bayes_stgat.py \
            --config            {input.config_file}    \
            --train-data        {input.train_data}      \
            --val-data          {input.val_data}        \
            --best-params       {input.best_params}     \
            --output-checkpoint {output.checkpoint}     \
            --output-metrics    {output.metrics}        \
            --device            {params.device}         \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 3 — Evaluate BayesianSTGAT
# ---------------------------------------------------------------------------

rule eval_bayes_stgat:
    """
    Evaluate a trained BayesianSTGAT on the held-out test set.
    Runs mc_samples.test (default 50) stochastic forward passes per window.
    Reports: predictive_mean, predictive_std (epistemic), 95% credible interval,
    and standard regression metrics in both log-space and original IR space.
    """
    input:
        checkpoint  = "models/stgat/{run_name}/bayes_best.pt",
        test_data   = "data/processed/stgat/{run_name}_test.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        metrics = "reports/stgat/{run_name}/bayes_test_metrics.json"
    log:
        "logs/stgat/{run_name}/bayes_eval.log"
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

        python scripts/eval_bayes_stgat.py \
            --config         {input.config_file}   \
            --checkpoint     {input.checkpoint}    \
            --test-data      {input.test_data}     \
            --output-metrics {output.metrics}      \
            --device         {params.device}       \
            > {log} 2>&1
        """
