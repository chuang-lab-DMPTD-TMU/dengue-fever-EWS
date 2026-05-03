"""
scripts/tune_stgat.py
---------------------
Optuna hyperparameter search for STGAT.

Each trial trains the model for a reduced number of epochs on a random
fraction of the training windows, then reports the best validation loss.
The best params are written to a JSON file that train_stgat.py reads via
--best-params.

Search space is read entirely from cfg["sweep_config"]["parameters"]
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
    apply_best_params,
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
    p.add_argument("--update-config",      action="store_true",
                   help="Write best params back into the config YAML after the sweep")
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# load_batches is imported from train_stgat above.


# ---------------------------------------------------------------------------
# Write best params back into the config YAML (preserving comments)
# ---------------------------------------------------------------------------

def write_best_params_to_config(config_path: str, best_params: dict) -> None:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq

    ryaml = YAML()
    ryaml.preserve_quotes = True

    with open(config_path) as fh:
        cfg = ryaml.load(fh)

    apply_best_params(cfg, best_params)

    # apply_best_params replaces gat.heads with a plain Python list.
    # Restore ruamel flow style so it round-trips as [4, 1] not a block sequence.
    heads = cfg["model"]["gat"].get("heads")
    if heads is not None and not isinstance(heads, CommentedSeq):
        flow = CommentedSeq(heads)
        flow.fa.set_flow_style()
        cfg["model"]["gat"]["heads"] = flow

    with open(config_path, "w") as fh:
        ryaml.dump(cfg, fh)

    print(f"[tune_stgat] Config updated with best params → {config_path}")


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
    params = base_cfg["sweep_config"]["parameters"]

    s = {name: _suggest(trial, name, spec) for name, spec in params.items()}

    # --- Sequence ---
    if "lookback_steps" in s:
        cfg["sequence"]["lookback_steps"] = s["lookback_steps"]

    # --- GAT ---
    if "gat_hidden_dim" in s:
        cfg["model"]["gat"]["hidden_dim"] = s["gat_hidden_dim"]
    if "gat_dropout" in s:
        cfg["model"]["gat"]["dropout"] = s["gat_dropout"]
    if "gat_residual" in s:
        cfg["model"]["gat"]["residual"] = s["gat_residual"]
    # Rebuild heads list from num_layers + per-layer head counts
    n_layers = s.get("gat_num_layers", len(cfg["model"]["gat"]["heads"]))
    if any(k in s for k in ("gat_num_layers", "gat_heads_layer1", "gat_heads_layer2")):
        h1 = s.get("gat_heads_layer1", cfg["model"]["gat"]["heads"][0])
        h2 = s.get("gat_heads_layer2", 2)
        if n_layers == 1:
            cfg["model"]["gat"]["heads"] = [1]
        elif n_layers == 2:
            cfg["model"]["gat"]["heads"] = [h1, 1]
        else:  # 3+
            cfg["model"]["gat"]["heads"] = [h1] + [h2] * (n_layers - 2) + [1]
        cfg["model"]["gat"]["num_layers"] = n_layers

    # --- GRU ---
    if "gru_hidden_dim" in s:
        cfg["model"]["gru"]["hidden_dim"] = s["gru_hidden_dim"]
    if "gru_num_layers" in s:
        cfg["model"]["gru"]["num_layers"] = s["gru_num_layers"]
    if "gru_dropout" in s:
        cfg["model"]["gru"]["dropout"] = s["gru_dropout"]

    # --- Output head ---
    if "output_hidden_dim" in s:
        cfg["model"]["output"]["hidden_dim"] = s["output_hidden_dim"]
    if "output_dropout" in s:
        cfg["model"]["output"]["dropout"] = s["output_dropout"]

    # --- Optimiser ---
    if "learning_rate" in s:
        cfg["training"]["learning_rate"] = s["learning_rate"]
    if "weight_decay" in s:
        cfg["training"]["weight_decay"] = s["weight_decay"]
    if "gradient_clip" in s:
        cfg["training"]["gradient_clip"] = s["gradient_clip"]

    # --- Projection ---
    if "projection_enabled" in s:
        cfg["model"]["projection"]["enabled"] = s["projection_enabled"]

    # --- Optimiser ---
    if "optimizer" in s:
        cfg["training"]["optimizer"] = s["optimizer"]

    # --- Scheduler ---
    if "scheduler_type" in s:
        cfg["training"]["scheduler"]["type"] = s["scheduler_type"]
    if "plateau_patience" in s:
        cfg["training"]["scheduler"]["plateau"]["patience"] = s["plateau_patience"]
    if "cosine_t_max" in s:
        cfg["training"]["scheduler"]["cosine"]["t_max"] = s["cosine_t_max"]

    # --- Early stopping: derive patience from scheduler so LR can change twice ---
    sched_type = cfg["training"]["scheduler"]["type"]
    if sched_type == "plateau":
        cfg["training"]["early_stopping"]["patience"] = (
            2 * cfg["training"]["scheduler"]["plateau"]["patience"]
        )
    elif sched_type == "cosine":
        cfg["training"]["early_stopping"]["patience"] = (
            cfg["training"]["scheduler"]["cosine"]["t_max"]
        )

    # --- Loss ---
    if "loss_type" in s:
        cfg["loss"]["regression"]["type"] = s["loss_type"]
    if "huber_delta" in s:
        cfg["loss"]["regression"]["huber_delta"] = s["huber_delta"]

    # --- Target ---
    if "log_transform_target" in s:
        cfg["target"]["log_transform"] = s["log_transform_target"]

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

    def _trim_lookback(batches, lookback: int):
        """Truncate x_seq to the last `lookback` timesteps (no-op if already shorter)."""
        if not isinstance(batches, list):
            b = batches
            return b._replace(x_seq=b.x_seq[-lookback:]) if b.x_seq.shape[0] > lookback else b
        return [
            b._replace(x_seq=b.x_seq[-lookback:]) if b.x_seq.shape[0] > lookback else b
            for b in batches
        ]

    def objective(trial: optuna.Trial) -> float:
        cfg = apply_trial_params(base_cfg, trial)
        cfg["training"]["epochs"] = args.trial_epochs
        cfg["training"]["early_stopping"]["patience"] = max(5, args.trial_epochs // 8)
        task         = cfg["target"]["task"]
        log_transform = cfg["target"].get("log_transform", False)

        print(f"\n[trial {trial.number}] params: {trial.params}")

        train_batches = _subsample(all_train_batches)
        lookback = cfg["sequence"]["lookback_steps"]
        train_batches = _trim_lookback(train_batches, lookback)
        trial_val     = _trim_lookback(val_batches, lookback)
        print(f"[trial {trial.number}] training on {len(train_batches) if isinstance(train_batches, list) else 1} windows"
              f" ({args.tune_data_fraction*100:.0f}% of train)  lookback={lookback}")

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
            train_epoch(model, train_batches, optim, loss_fn, grad_clip, task, log_transform)
            val_loss, _ = val_epoch(model, trial_val, loss_fn, task, log_transform)

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

    sweep_method = base_cfg.get("sweep_config", {}).get("method", "bayesian").lower()
    if sweep_method == "random":
        sampler = optuna.samplers.RandomSampler(seed=rand_state)
    elif sweep_method == "grid":
        sweep_params = base_cfg.get("sweep_config", {}).get("parameters", {})
        grid_space = {
            name: spec["values"]
            for name, spec in sweep_params.items()
            if "values" in spec
        }
        skipped = [n for n in sweep_params if n not in grid_space]
        if skipped:
            print(f"[tune_stgat] Grid search: skipping continuous params (no 'values'): {skipped}")
        sampler = optuna.samplers.GridSampler(grid_space)
    else:  # bayesian / tpe (default)
        sampler = optuna.samplers.TPESampler(seed=rand_state)
    print(f"[tune_stgat] Sampler: {type(sampler).__name__}  (sweep_config.method={sweep_method!r})")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
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

    if args.update_config:
        write_best_params_to_config(args.config, best_params)


if __name__ == "__main__":
    main()
