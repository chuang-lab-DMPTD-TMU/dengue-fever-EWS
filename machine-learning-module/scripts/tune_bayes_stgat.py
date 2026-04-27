"""
scripts/tune_bayes_stgat.py
-----------------------------------
Optuna hyperparameter search for BayesianSTGAT.

Extends the deterministic search space (tune_stgat.py) with Bayesian-specific
parameters read directly from cfg["sweep_config"]["parameters"]:
  kl_weight, kl_anneal_fraction, prior_sigma, rho_init,
  mc_samples_train, free_bits

Objective scores trials on val_mse (not total ELBO). During KL annealing the
ELBO rises even when the model is learning — stopping on it would fire too
early. val_mse isolates the prediction quality signal.

Via Snakemake (bayes_stgat.smk → tune_bayes_stgat):
    python scripts/tune_bayes_stgat.py \\
        --config       config/stgat/{run_name}.yaml \\
        --train-data   data/processed/stgat/{run_name}_train.pt \\
        --val-data     data/processed/stgat/{run_name}_val.pt \\
        --output       models/stgat/{run_name}/bayes_best_params.json \\
        --n-trials     20 \\
        --trial-epochs 40 \\
        --n-mc-val     5 \\
        --device       cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
import gc
import optuna
import torch
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from train_bayes_stgat import (
    Batch,
    BayesGNN,
    EarlyStopping,
    apply_best_params,
    build_bayes_model,
    build_optimizer,
    build_scheduler,
    compute_beta,
    get_likelihood_fn,
    load_batches,
    make_synthetic_data,
    train_epoch_bayes,
    val_epoch_bayes,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for BayesianSTGAT")
    p.add_argument("--config",             required=True)
    p.add_argument("--train-data",         default=None)
    p.add_argument("--val-data",           default=None)
    p.add_argument("--output",             required=True, help="Where to write bayes_best_params.json")
    p.add_argument("--n-trials",           type=int,   default=20)
    p.add_argument("--trial-epochs",       type=int,   default=40)
    p.add_argument("--tune-data-fraction", type=float, default=0.3)
    p.add_argument("--n-mc-val",           type=int,   default=5,
                   help="MC passes per val batch during tuning (reduced for speed)")
    p.add_argument("--storage",            default=None)
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Read search space from config and suggest trial values
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    dist = spec.get("distribution", "")
    if dist == "log_uniform_values":
        return trial.suggest_float(name, spec["min"], spec["max"], log=True)
    if dist in ("uniform", "uniform_values"):
        return trial.suggest_float(name, spec["min"], spec["max"])
    if dist == "int_uniform":
        return trial.suggest_int(name, int(spec["min"]), int(spec["max"]))
    raise ValueError(f"Unsupported distribution '{dist}' for param '{name}'")


def apply_trial_params(base_cfg: dict, trial: optuna.Trial) -> dict:
    """
    Suggest all sweep parameters for this trial and write them into a copy of
    base_cfg.  Handles both deterministic and Bayesian-specific keys.
    """
    cfg = copy.deepcopy(base_cfg)
    params_spec = base_cfg["sweep_config"]["parameters"]
    s = {name: _suggest(trial, name, spec) for name, spec in params_spec.items()}

    # --- Sequence ---
    if "lookback_steps" in s:
        cfg["sequence"]["lookback_steps"] = s["lookback_steps"]

    # --- GAT ---
    if "gat_hidden_dim" in s: cfg["model"]["gat"]["hidden_dim"] = s["gat_hidden_dim"]
    if "gat_dropout"    in s: cfg["model"]["gat"]["dropout"]    = s["gat_dropout"]
    if "gat_residual"   in s: cfg["model"]["gat"]["residual"]   = s["gat_residual"]

    if any(k in s for k in ("gat_num_layers", "gat_heads_layer1", "gat_heads_layer2")):
        n_layers = s.get("gat_num_layers", len(cfg["model"]["gat"]["heads"]))
        h1 = s.get("gat_heads_layer1", cfg["model"]["gat"]["heads"][0])
        h2 = s.get("gat_heads_layer2", 2)
        if n_layers == 1:
            cfg["model"]["gat"]["heads"] = [1]
        elif n_layers == 2:
            cfg["model"]["gat"]["heads"] = [h1, 1]
        else:
            cfg["model"]["gat"]["heads"] = [h1] + [h2] * (n_layers - 2) + [1]
        cfg["model"]["gat"]["num_layers"] = n_layers

    # --- GRU ---
    if "gru_hidden_dim" in s: cfg["model"]["gru"]["hidden_dim"]  = s["gru_hidden_dim"]
    if "gru_num_layers" in s: cfg["model"]["gru"]["num_layers"]  = s["gru_num_layers"]
    if "gru_dropout"    in s: cfg["model"]["gru"]["dropout"]     = s["gru_dropout"]

    # --- Output head ---
    if "output_hidden_dim" in s: cfg["model"]["output"]["hidden_dim"] = s["output_hidden_dim"]
    if "output_dropout"    in s: cfg["model"]["output"]["dropout"]    = s["output_dropout"]

    # --- Optimiser ---
    if "learning_rate" in s: cfg["training"]["learning_rate"] = s["learning_rate"]
    if "weight_decay"  in s: cfg["training"]["weight_decay"]  = s["weight_decay"]
    if "gradient_clip" in s: cfg["training"]["gradient_clip"] = s["gradient_clip"]

    # --- Scheduler ---
    if "scheduler_type"   in s: cfg["training"]["scheduler"]["type"]               = s["scheduler_type"]
    if "plateau_patience" in s: cfg["training"]["scheduler"]["plateau"]["patience"] = s["plateau_patience"]
    if "cosine_t_max"     in s: cfg["training"]["scheduler"]["cosine"]["t_max"]     = s["cosine_t_max"]

    # --- Bayesian-specific ---
    bayes = cfg.setdefault("bayesian", {})
    kl    = bayes.setdefault("kl", {})
    prior = bayes.setdefault("prior", {})
    post  = bayes.setdefault("posterior", {})
    mc_s  = bayes.setdefault("mc_samples", {})
    elbo  = cfg.setdefault("loss", {}).setdefault("elbo", {})

    if "kl_weight"          in s:
        kl["weight"]      = s["kl_weight"]
        elbo["kl_weight"] = s["kl_weight"]
    if "kl_anneal_fraction" in s: kl["anneal_fraction"] = s["kl_anneal_fraction"]
    if "prior_sigma"        in s: prior["sigma"]         = s["prior_sigma"]
    if "rho_init"           in s: post["rho_init"]       = s["rho_init"]
    if "mc_samples_train"   in s: mc_s["train"]          = s["mc_samples_train"]
    if "free_bits"          in s: elbo["free_bits"]      = s["free_bits"]

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as fh:
        base_cfg = yaml.safe_load(fh)

    device = torch.device(args.device)
    task   = base_cfg["target"]["task"]
    print(f"[tune_bayes] Device: {device}  |  trials: {args.n_trials}  |  "
          f"epochs/trial: {args.trial_epochs}  |  n_mc_val: {args.n_mc_val}")

    use_synthetic = args.train_data is None or not Path(args.train_data).exists()
    if use_synthetic:
        print("[tune_bayes] No data files — using synthetic data.")
        train_batch, val_batch = make_synthetic_data(base_cfg, device)
        all_train_batches: "Batch | list[Batch]" = train_batch
        val_batches:       "Batch | list[Batch]" = val_batch
    else:
        all_train_batches = load_batches(args.train_data, device)
        val_batches       = load_batches(args.val_data,   device)
        print(f"[tune_bayes] Loaded train ({len(all_train_batches)} windows) / "
              f"val ({len(val_batches)} windows)")

    ref = all_train_batches[0] if isinstance(all_train_batches, list) else all_train_batches
    in_channels = ref.x_seq.shape[2]

    rand_state = base_cfg["training"].get("random_state", 42)
    rng = random.Random(rand_state)

    def _subsample(batches):
        if not isinstance(batches, list):
            return batches
        k = max(1, int(len(batches) * args.tune_data_fraction))
        return rng.sample(batches, k)

    def _trim_lookback(batches, lookback: int):
        if not isinstance(batches, list):
            b = batches
            return b._replace(x_seq=b.x_seq[-lookback:]) if b.x_seq.shape[0] > lookback else b
        return [
            b._replace(x_seq=b.x_seq[-lookback:]) if b.x_seq.shape[0] > lookback else b
            for b in batches
        ]

    def objective(trial: optuna.Trial) -> float:
        torch.cuda.empty_cache()
        gc.collect()
        cfg = apply_trial_params(base_cfg, trial)
        cfg["training"]["epochs"] = args.trial_epochs
        cfg["training"]["early_stopping"]["patience"] = max(5, args.trial_epochs // 8)

        log_transform = cfg["target"].get("log_transform", False)
        mc_train = cfg.get("bayesian", {}).get("mc_samples", {}).get("train", 1)
        lookback = cfg["sequence"]["lookback_steps"]

        print(f"\n[trial {trial.number}] params: {trial.params}")

        train_batches = _trim_lookback(_subsample(all_train_batches), lookback)
        trial_val     = _trim_lookback(val_batches, lookback)
        n_train = len(train_batches) if isinstance(train_batches, list) else 1
        print(f"[trial {trial.number}] {n_train} windows  lookback={lookback}")

        model         = build_bayes_model(cfg, in_channels).to(device)
        likelihood_fn = get_likelihood_fn(cfg)
        optim         = build_optimizer(model, cfg)
        sched         = build_scheduler(optim, cfg)
        es_cfg        = cfg["training"].get("early_stopping", {})
        stopper       = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 1e-4),
        )
        grad_clip = cfg["training"].get("gradient_clip")
        n_epochs  = cfg["training"]["epochs"]

        try:
            for epoch in range(1, n_epochs + 1):
                train_epoch_bayes(
                    model, train_batches, optim, likelihood_fn,
                    cfg, epoch, n_epochs, grad_clip, task, log_transform, mc_train,
                )
                val_mse, _ = val_epoch_bayes(
                    model, trial_val, likelihood_fn, args.n_mc_val, task, log_transform,
                )

                if sched is not None:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched.step(val_mse)
                    else:
                        sched.step()

                trial.report(val_mse, epoch)
                if trial.should_prune():
                    trial.set_user_attr("pruned_reason", "median_pruner")
                    trial.set_user_attr("pruned_epoch", epoch)
                    raise optuna.exceptions.TrialPruned()

                if stopper.step(val_mse, model):
                    break

        except torch.cuda.OutOfMemoryError:
            print(f"[trial {trial.number}] OOM — pruning trial and clearing cache")
            trial.set_user_attr("pruned_reason", "oom")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()

        print(f"[trial {trial.number}] best_val_mse={stopper.best_val_mse:.4f}")
        return stopper.best_val_mse

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    storage    = args.storage or f"sqlite:///{out.parent / 'bayes_optuna.db'}"
    study_name = base_cfg.get("run_name", "bayes_stgat_tune")
    print(f"[tune_bayes] Storage: {storage}  |  study: {study_name}")
    print(f"[tune_bayes] Live dashboard:  optuna-dashboard {storage}")

    sweep_method = base_cfg.get("sweep_config", {}).get("method", "bayes").lower()
    if sweep_method == "random":
        sampler = optuna.samplers.RandomSampler(seed=rand_state)
    elif sweep_method == "grid":
        sweep_params = base_cfg.get("sweep_config", {}).get("parameters", {})
        grid_space = {n: s["values"] for n, s in sweep_params.items() if "values" in s}
        skipped = [n for n in sweep_params if n not in grid_space]
        if skipped:
            print(f"[tune_bayes] Grid: skipping continuous params: {skipped}")
        sampler = optuna.samplers.GridSampler(grid_space)
    else:  # bayes / tpe (default — matches sweep_config.method: bayes in sea_bayes.yaml)
        sampler = optuna.samplers.TPESampler(seed=rand_state)

    print(f"[tune_bayes] Sampler: {type(sampler).__name__}  (method={sweep_method!r})")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.set_user_attr("task",   task)
    study.set_user_attr("config", args.config)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best       = study.best_trial
    best_loss  = best.value
    best_params = best.params
    print(f"[tune_bayes] Best val_mse={best_loss:.4f} | params={best_params}")

    with open(out, "w") as fh:
        json.dump({"best_val_mse": best_loss, **best_params}, fh, indent=2)
    print(f"[tune_bayes] Best params → {out}")


if __name__ == "__main__":
    main()
