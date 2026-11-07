"""Optuna hyperparameter search for CSBrain fine-tuning, mirroring
cbramod_optuna_search.py's approach with one deliberate difference: the
objective metric.

  - IMPORTANT DIFFERENCE FROM cbramod_optuna_search.py: CSBrain's own
    finetune_trainer.py selects its "best" checkpoint by plain ACCURACY for
    BOTH binary and multiclass tasks (train_for_binaryclass: `if acc >
    acc_best`, train_for_multiclass: `if acc > acc_best`) -- NOT roc_auc/
    kappa like CBraMod's trainer does. Per this project's "reuse whatever
    the existing code already tracks" rule, this search's objective is
    val_acc for every dataset, not a per-task-type metric. Verify this
    still matches CSBrain/finetune_trainer.py on your cluster before
    trusting results -- if that file has been edited since, re-check the
    actual `if ... > ..._best:` lines in both train_for_* methods.
  - Same non-invasive approach as CBraMod's search: builds each trial's
    argparse.Namespace in-process, reuses finetune_trainer.Trainer
    UNMODIFIED, pulls the built pieces off it, and runs our own epoch loop
    with Optuna pruning -- no checkpoint saving, no test-set eval.
  - CSBrain-specific params CBraMod's Namespace didn't need: `model`
    (model_for_<dataset>.py's Model.__init__ checks `if param.model ==
    'CSBrain':` before constructing the backbone) and `n_layer` (passed
    straight into CSBrain(... n_layer=param.n_layer ...) by every
    model_for_<dataset>.py). Fixed at "CSBrain" / 12 respectively -- not
    searched, matching production's implicit defaults.
  - foundation_dir is CSBrain's own convention ("pth/CSBrain.pth"), not
    CBraMod's "pretrained_weights/pretrained_weights.pth".
  - dropout NOT searched, same reasoning as CBraMod's search: only Dream
    and Yang2025's CSBrain classifier heads reference param.dropout at all
    (nn.Dropout(param.dropout)) -- confirmed by reading every
    model_for_<dataset>.py built for this project. Fixed at 0.1 everywhere.
  - weight_decay fixed at 0.05, clip_value categorical [0, 0.5, 1, 2, 5]
    (0 == disabled, same `clip_value <= 0` skip-clip convention as
    CBraMod's identical finetune_trainer.py clip logic), lr ranges
    INHERITED from CBraMod's own search as a starting assumption (no
    CSBrain-specific prior exists yet beyond the single original CHB-MIT
    script's --lr 0.0001 and ISRUC's --lr 0.001) -- revisit if CSBrain's
    search results look off relative to these ranges.
  - batch_size per dataset carried over from the single-run sbatch scripts
    (csbrain_<dataset>_<arm>.sbatch) already built for this project.
  - ISRUC excluded from DATASET_CONFIG, same caution as CBraMod's search:
    unconfirmed whether CSBrain's shipped isruc_dataset.py/model_for_isruc.py
    were adapted for this project's isruc_processed_80h data or are still
    the original CSBrain-repo ISRUC-III files. Add back once confirmed.
  - japaneeg's root is /home/nourmaj/links/scratch/japaneeg_processed_80h -- the
    already-confirmed-corrected path (see the CBraMod japaneeg path bug
    this project already found and fixed).

Usage:
  python csbrain_optuna_search.py --dataset clinical --arm pretrained --n_trials 20
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import optuna
import torch

CSBRAIN_SRC = "/home/nourmaj/links/scratch/csbrain/CSBrain"
sys.path.insert(0, CSBRAIN_SRC)
os.chdir(CSBRAIN_SRC)  # datasets/models packages, and the relative foundation_dir, assume this cwd

from datasets import (  # noqa: E402
    chb_dataset, clinical_dataset, dream_dataset, japaneeg_dataset,
    karaone_dataset, thoughtviz_dataset, yang2025_dataset,
)
from models import (  # noqa: E402
    model_for_chb, model_for_clinical, model_for_dream, model_for_japaneeg,
    model_for_karaone, model_for_thoughtviz, model_for_yang2025,
)
from finetune_trainer import Trainer  # noqa: E402

FOUNDATION_DIR = "pth/CSBrain.pth"  # CSBrain's own convention, not CBraMod's
FIXED_WEIGHT_DECAY = 0.05
FIXED_DROPOUT = 0.1        # only Dream/Yang2025 heads use this -- not worth searching
FIXED_N_LAYER = 12         # CSBrain-specific -- matches finetune_main.py's own --n_layer default
CLIP_CHOICES = [0, 0.5, 1.0, 2.0, 5.0]
SEED = 42

# INHERITED from CBraMod's own search ranges -- no independent CSBrain prior
# exists yet. Revisit if results look off.
LR_RANGE = {
    "pretrained": (1e-5, 2.5e-4),
    "random": (1e-4, 1e-3),
}

# task_type only matters for the loss function / ISRUC's transpose special
# case now -- the OBJECTIVE METRIC is val_acc for every dataset regardless
# of task_type, since that's what CSBrain's finetune_trainer.py actually
# selects "best" on for both train_for_binaryclass and train_for_multiclass.
DATASET_CONFIG = {
    "chbmit": dict(
        dataset_module=chb_dataset, model_module=model_for_chb,
        ds_name="CHB-MIT", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/chb-mit-processed-80h",
        batch_size=64, epochs=50,
    ),
    "clinical": dict(
        dataset_module=clinical_dataset, model_module=model_for_clinical,
        ds_name="Clinical", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/ds004504_processed",
        batch_size=32, epochs=50,
    ),
    "dream": dict(
        dataset_module=dream_dataset, model_module=model_for_dream,
        ds_name="Dream", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/dream_processed",
        batch_size=16, epochs=50,
    ),
    "japaneeg": dict(
        dataset_module=japaneeg_dataset, model_module=model_for_japaneeg,
        ds_name="JapanEEG", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/japaneeg_processed_80h",  # already-confirmed-corrected path
        batch_size=32, epochs=50,
    ),
    "karaone": dict(
        dataset_module=karaone_dataset, model_module=model_for_karaone,
        ds_name="KaraOne", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/karaone_processed",
        batch_size=32, epochs=50,
    ),
    "thoughtviz": dict(
        dataset_module=thoughtviz_dataset, model_module=model_for_thoughtviz,
        ds_name="ThoughtViz", num_classes=10, task_type="multiclass",
        root="/home/nourmaj/links/scratch/thoughtviz_processed",
        batch_size=32, epochs=50,
    ),
    "yang2025": dict(
        dataset_module=yang2025_dataset, model_module=model_for_yang2025,
        ds_name="Yang2025", num_classes=2, task_type="binary",
        root="/home/nourmaj/links/scratch/yang2025_processed",
        batch_size=32, epochs=50,
    ),
    # isruc intentionally excluded -- see module docstring.
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_params(dataset_key, arm, lr, clip_value, fraction="100"):
    cfg = DATASET_CONFIG[dataset_key]
    return argparse.Namespace(
        seed=SEED,
        cuda=0,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=lr,
        weight_decay=FIXED_WEIGHT_DECAY,
        optimizer="AdamW",
        clip_value=clip_value,
        dropout=FIXED_DROPOUT,
        downstream_dataset=cfg["ds_name"],
        datasets_dir=cfg["root"] if fraction == "100" else f'{cfg["root"]}_{fraction}pct',
        num_of_classes=cfg["num_classes"],
        model_dir="/home/nourmaj/links/scratch/csbrain_optuna_unused",  # never written to -- no train_for_* call, no save
        num_workers=16,
        label_smoothing=0.1,
        multi_lr=False,
        frozen=False,
        use_pretrained_weights=(arm == "pretrained"),
        foundation_dir=FOUNDATION_DIR,
        model="CSBrain",       # CSBrain-specific: model_for_<dataset>.py checks this
        n_layer=FIXED_N_LAYER,  # CSBrain-specific: passed into CSBrain(... n_layer=...)
    )


def run_binaryclass_optuna(trial, params, t):
    """Same per-epoch body as Trainer.train_for_binaryclass(), minus the
    final test-set eval + checkpoint save, plus Optuna report/prune.
    Objective is ACCURACY (first return value), matching CSBrain's own
    `if acc > acc_best` selection -- not roc_auc."""
    acc_best = 0.0
    for epoch in range(params.epochs):
        t.model.train()
        for x, y in t.data_loader["train"]:
            t.optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            pred = t.model(x)
            loss = t.criterion(pred, y)
            loss.backward()
            if params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(t.model.parameters(), params.clip_value)
            t.optimizer.step()
            t.optimizer_scheduler.step()

        with torch.no_grad():
            acc, _, _, _ = t.val_eval.get_metrics_for_binaryclass(t.model)

        if acc > acc_best:
            acc_best = acc

        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return acc_best


def run_multiclass_optuna(trial, params, t, dataset_key):
    """Same per-epoch body as Trainer.train_for_multiclass(), minus the
    final test-set eval + checkpoint save, plus Optuna report/prune.
    Objective is ACCURACY (first return value), matching CSBrain's own
    `if acc > acc_best` selection -- not kappa."""
    acc_best = 0.0
    for epoch in range(params.epochs):
        t.model.train()
        for x, y in t.data_loader["train"]:
            t.optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            pred = t.model(x)
            if dataset_key == "isruc":  # kept for forward-compat once ISRUC is migrated
                loss = t.criterion(pred.transpose(1, 2), y)
            else:
                loss = t.criterion(pred, y)
            loss.backward()
            if params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(t.model.parameters(), params.clip_value)
            t.optimizer.step()
            t.optimizer_scheduler.step()

        with torch.no_grad():
            acc, _, _, _ = t.val_eval.get_metrics_for_multiclass(t.model)

        if acc > acc_best:
            acc_best = acc

        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return acc_best


def objective(trial, dataset_key, arm, fraction="100"):
    cfg = DATASET_CONFIG[dataset_key]
    lo, hi = LR_RANGE[arm]
    lr = trial.suggest_float("lr", lo, hi, log=True)
    clip_value = trial.suggest_categorical("clip_value", CLIP_CHOICES)

    params = build_params(dataset_key, arm, lr, clip_value, fraction)
    setup_seed(params.seed)

    load_dataset = cfg["dataset_module"].LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = cfg["model_module"].Model(params)
    t = Trainer(params, data_loader, model)

    if cfg["task_type"] == "binary":
        best_val = run_binaryclass_optuna(trial, params, t)
    else:
        best_val = run_multiclass_optuna(trial, params, t, dataset_key)

    del t, model
    torch.cuda.empty_cache()
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--arm", required=True, choices=["pretrained", "random"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--fraction", default="100", choices=["100", "50", "25"])
    cli_args = parser.parse_args()

    out_dir = "/home/nourmaj/links/scratch/csbrain_optuna_results"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct"
    db_path = f"{out_dir}/{tag}.db"
    study_name = f"csbrain_{tag}"

    metric_name = "val_acc"  # same for every dataset -- see module docstring

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = len([
        tr for tr in study.trials
        if tr.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ])
    n_remaining = max(0, cli_args.n_trials - n_done)
    print(f"=== CSBrain Optuna search: dataset={cli_args.dataset} arm={cli_args.arm} ===")
    print(f"objective metric = {metric_name}")
    print(f"{n_done} trials already recorded in {db_path}, running {n_remaining} more")

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cli_args.dataset, cli_args.arm, cli_args.fraction),
            n_trials=n_remaining,
        )

    print("=== BEST TRIAL ===")
    print(f"dataset={cli_args.dataset} arm={cli_args.arm}")
    print(f"best {metric_name} = {study.best_value}")
    print(f"best params = {study.best_params}")

    with open(f"{out_dir}/{tag}_best.json", "w") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "arm": cli_args.arm,
                "fraction": cli_args.fraction,
                "metric": metric_name,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "weight_decay": FIXED_WEIGHT_DECAY,
                "dropout": FIXED_DROPOUT,
                "n_layer": FIXED_N_LAYER,
                "batch_size": DATASET_CONFIG[cli_args.dataset]["batch_size"],
                "epochs": DATASET_CONFIG[cli_args.dataset]["epochs"],
                "optimizer": "AdamW (not searched)",
                "multi_lr": False,
                "n_trials_run": len(study.trials),
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_dir}/{tag}_best.json")


if __name__ == "__main__":
    main()