"""
scripts/tune_stgat.py
---------------------
Optuna hyperparameter search for STGAT.

Each trial trains the model for a reduced number of epochs on a random
fraction of the training windows, then reports the best validation loss.
The best params are written to a JSON file that train_stgat.py reads via
--best-params.

Search space is read entirely from cfg["wandb"]["sweep_config"]["parameters"]
in the config YAML — no hardcoded bounds here.

Via Snakemake (stgat.smk):
    python workflow/scripts/tune_stgat.py \
        --config              config/stgat/{run_name}.yaml \
        --train-data          data/processed/stgat/{run_name}_train.pt \
        --val-data            data/processed/stgat/{run_name}_val.pt \
        --output              models/stgat/{run_name}/best_params.json \
        --n-trials            20 \
        --tune-data-fraction  0.3 \
        --device              cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import optuna
import torch
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from train_stgat import (
    Batch,
    build_model,
    get_loss_fn,
    load_batches,
    make_synthetic_data,
    train_epoch,
    val_epoch,
    build_optimizer,
    build_scheduler,
    EarlyStopping,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for STGAT")
    p.add_argument("--config",             required=True,  help="Path to config_stgat.yaml")
    p.add_argument("--train-data",         default=None,   help="Path to train tensor (.pt)")
    p.add_argument("--val-data",           default=None,   help="Path to val tensor (.pt)")
    p.add_argument("--graph",              default=None,   help="Path to graph.pkl (unused if bundled in .pt)")
    p.add_argument("--output",             required=True,  help="Where to write best_params.json")
    p.add_argument("--n-trials",           type=int,   default=20)
    p.add_argument("--trial-epochs",       type=int,   default=40,
                   help="Max epochs per Optuna trial (shorter than full training)")
    p.add_argument("--tune-data-fraction", type=float, default=0.3,
                   help="Fraction of training windows used per trial (speeds up search)")
    p.add_argument("--storage",            default=None,
                   help="Optuna storage URL, e.g. sqlite:///models/stgat/sea_baseline/optuna.db "
                        "(defaults to sqlite next to --output)")
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# load_batches is imported from train_stgat above.


# ---------------------------------------------------------------------------
# Read search space from config and suggest values for one trial
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    """Convert one sweep_config parameter spec into an Optuna suggestion."""
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    dist = spec.get("distribution", "")
    if dist == "log_uniform_values":
        return trial.suggest_float(name, spec["min"], spec["max"], log=True)
    if dist in ("uniform", "uniform_values"):
        return trial.suggest_float(name, spec["min"], spec["max"])
    if dist in ("int_uniform",):
        return trial.suggest_int(name, int(spec["min"]), int(spec["max"]))
    raise ValueError(f"Unsupported distribution '{dist}' for param '{name}'")


def apply_trial_params(base_cfg: dict, trial: optuna.Trial) -> dict:
    cfg = copy.deepcopy(base_cfg)
    params = base_cfg["wandb"]["sweep_config"]["parameters"]

    suggested = {name: _suggest(trial, name, spec) for name, spec in params.items()}

    # Map suggested values into the model config
    if "learning_rate" in suggested:
        cfg["training"]["learning_rate"] = suggested["learning_rate"]
    if "weight_decay" in suggested:
        cfg["training"]["weight_decay"] = suggested["weight_decay"]
    if "gat_hidden_dim" in suggested:
        cfg["model"]["gat"]["hidden_dim"] = suggested["gat_hidden_dim"]
    if "gat_heads_first_layer" in suggested:
        n_layers = len(cfg["model"]["gat"]["heads"])
        cfg["model"]["gat"]["heads"] = [suggested["gat_heads_first_layer"]] * (n_layers - 1) + [1]
    if "dropout" in suggested:
        cfg["model"]["gat"]["dropout"] = suggested["dropout"]
    if "gru_hidden_dim" in suggested:
        cfg["model"]["gru"]["hidden_dim"] = suggested["gru_hidden_dim"]

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as fh:
        base_cfg = yaml.safe_load(fh)

    device = torch.device(args.device)
    task      = base_cfg["target"]["task"]
    loss_type = base_cfg["loss"][task]["type"]
    log_xform = base_cfg["target"].get("log_transform", False)
    print(f"[tune_stgat] Device: {device}  |  trials: {args.n_trials}  |  "
          f"epochs/trial: {args.trial_epochs}  |  data fraction: {args.tune_data_fraction}")
    print(f"[tune_stgat] Task: {task}  |  loss: {loss_type}  |  log_transform: {log_xform}")

    # Load data once — shared across all trials
    use_synthetic = args.train_data is None or not Path(args.train_data).exists()
    if use_synthetic:
        print("[tune_stgat] No data files — using synthetic data.")
        train_batch, val_batch = make_synthetic_data(base_cfg, device)
        all_train_batches: "Batch | list[Batch]" = train_batch
        val_batches:       "Batch | list[Batch]" = val_batch
    else:
        all_train_batches = load_batches(args.train_data, device)
        val_batches       = load_batches(args.val_data,   device)
        print(f"[tune_stgat] Loaded train ({len(all_train_batches)} windows) / "
              f"val ({len(val_batches)} windows) from {args.train_data}")

    ref = all_train_batches[0] if isinstance(all_train_batches, list) else all_train_batches
    in_channels = ref.x_seq.shape[2]

    rand_state = base_cfg["training"].get("random_state", 42)
    rng = random.Random(rand_state)

    def _subsample(batches):
        if not isinstance(batches, list):
            return batches
        k = max(1, int(len(batches) * args.tune_data_fraction))
        return rng.sample(batches, k)

    def objective(trial: optuna.Trial) -> float:
        cfg = apply_trial_params(base_cfg, trial)
        cfg["training"]["epochs"] = args.trial_epochs
        cfg["training"]["early_stopping"]["patience"] = max(5, args.trial_epochs // 8)
        task = cfg["target"]["task"]

        print(f"\n[trial {trial.number}] params: {trial.params}")

        train_batches = _subsample(all_train_batches)
        print(f"[trial {trial.number}] training on {len(train_batches)} windows"
              f" ({args.tune_data_fraction*100:.0f}% of train)")

        model   = build_model(cfg, in_channels).to(device)
        loss_fn = get_loss_fn(cfg)
        optim   = build_optimizer(model, cfg)
        sched   = build_scheduler(optim, cfg)
        es_cfg  = cfg["training"].get("early_stopping", {})
        stopper = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 1e-4),
        )
        grad_clip = cfg["training"].get("gradient_clip")

        for _ in range(cfg["training"]["epochs"]):
            train_epoch(model, train_batches, optim, loss_fn, grad_clip, task)
            val_loss, _ = val_epoch(model, val_batches, loss_fn, task)

            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_loss)
                else:
                    sched.step()

            if stopper.step(val_loss, model):
                break

        print(f"[trial {trial.number}] best_val_loss={stopper.best_loss:.4f}")
        return stopper.best_loss

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{out.parent / 'optuna.db'}"
    study_name = base_cfg.get("run_name", "stgat_tune")
    print(f"[tune_stgat] Storage: {storage}  |  study: {study_name}")
    print(f"[tune_stgat] Live dashboard:  optuna-dashboard {storage}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=rand_state),
    )
    study.set_user_attr("task",         task)
    study.set_user_attr("loss",         loss_type)
    study.set_user_attr("log_transform", str(log_xform))
    study.set_user_attr("config",       args.config)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_trial  = study.best_trial
    best_params = best_trial.params
    best_loss   = best_trial.value
    print(f"[tune_stgat] Best val_loss={best_loss:.4f} | params={best_params}")

    with open(out, "w") as fh:
        json.dump({"best_val_loss": best_loss, **best_params}, fh, indent=2)
    print(f"[tune_stgat] Best params → {out}")


if __name__ == "__main__":
    main()
