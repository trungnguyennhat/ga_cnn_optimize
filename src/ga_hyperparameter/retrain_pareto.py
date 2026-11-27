"""Retrain hardware-aware Pareto fronts (ga, random, surrogate) with transferred GA hyperparameters."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import recall_score
import torch
from torch import nn

from src.model import architecture_cost, build_model, count_parameters
from src.search_runtime import HARDWARE_RESULTS, MAC_CONVENTION, save_json
from src.test_eval import evaluate, load_test_data, select_pareto_architectures
from src.train import (
    DATA_PATH, DEFAULT_EPOCHS, PROJECT_ROOT, load_data, resolve_device,
    set_seed, train_model,
)


BATCH_SIZE = 64
SCHEDULER_STEP = 20
DEFAULT_HP_SUMMARY = (
    PROJECT_ROOT / "results" / "ga_hyperparameter" / "seed_42"
    / "run_20261126_203124_030102" / "summary.json"
)
DEFAULT_SURROGATE_SUMMARY = (
    PROJECT_ROOT / "results" / HARDWARE_RESULTS / "surrogate_search"
    / "run_20261125_223530_003810" / "seed_42" / "summary.json"
)


def load_best_hyperparameters(summary_path: Path) -> tuple[dict[str, float], int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_trial = summary["best_trial"]
    for fin in summary["finalists"]:
        if fin["trial"] == best_trial:
            return fin["hyperparameters"], best_trial
    raise ValueError(f"Best trial {best_trial} not found in {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--hp-summary", type=Path, default=DEFAULT_HP_SUMMARY)
    parser.add_argument("--ga-summary", type=Path)
    parser.add_argument("--random-summary", type=Path)
    parser.add_argument("--surrogate-summary", type=Path, default=DEFAULT_SURROGATE_SUMMARY)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    hw_root = PROJECT_ROOT / "results" / HARDWARE_RESULTS
    seed_dir = f"seed_{args.seed}"
    args.ga_summary = args.ga_summary or hw_root / "ga_search" / seed_dir / "summary.json"
    args.random_summary = args.random_summary or hw_root / "random_search" / seed_dir / "summary.json"
    args.output_root = args.output_root or hw_root / "test_eval_ga_hyperparam"

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root / seed_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    hyperparameters, best_trial = load_best_hyperparameters(args.hp_summary)
    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path)
    test_data = load_test_data(args.data_path)

    train_labels = loaded_data[0].tensors[1]
    class_counts = train_labels.bincount(minlength=7)
    inverse_frequency = len(train_labels) / (len(class_counts) * class_counts)
    alpha = hyperparameters["class_weight_alpha"]
    class_weights = inverse_frequency.pow(alpha) if alpha else None
    criterion = nn.CrossEntropyLoss()

    methods = [
        ("ga", args.ga_summary),
        ("random", args.random_summary),
        ("surrogate", args.surrogate_summary),
    ]

    started = time.perf_counter()
    print(f"output={run_dir} device={device}", flush=True)
    print(f"transferred_hp (trial {best_trial}): {hyperparameters}", flush=True)

    overall_results = {}
    for method, summary_path in methods:
        candidates = select_pareto_architectures(summary_path, args.seed, f"{method}_search")
        method_results = []
        print(f"\n=== Method: {method} ({len(candidates)} Pareto architectures) ===", flush=True)
        for idx, (role, architecture, selection) in enumerate(candidates, 1):
            model_dir = run_dir / method / role / seed_dir
            model_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_path = model_dir / "checkpoint.pt"

            set_seed(args.seed)
            model = build_model(architecture)
            metrics, history, runtime = train_model(
                model, loaded_data,
                learning_rate=hyperparameters["learning_rate"],
                batch_size=BATCH_SIZE,
                optimizer_name="AdamW",
                epochs=args.epochs,
                seed=args.seed,
                device=device,
                scheduler_step_size=SCHEDULER_STEP,
                weight_decay=hyperparameters["weight_decay"],
                class_weights=class_weights,
            )

            val_metrics, val_labels, val_preds = evaluate(
                model, loaded_data[1], criterion, batch_size=BATCH_SIZE, device=device, prefix="val",
            )
            val_metrics["val_recall_by_class"] = recall_score(
                val_labels, val_preds, labels=np.arange(7), average=None, zero_division=0,
            ).tolist()

            test_metrics, test_labels, test_preds = evaluate(
                model, test_data, criterion, batch_size=BATCH_SIZE, device=device, prefix="test",
            )
            test_metrics["test_recall_by_class"] = recall_score(
                test_labels, test_preds, labels=np.arange(7), average=None, zero_division=0,
            ).tolist()

            params = count_parameters(model)
            _, macs = architecture_cost(architecture)

            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": architecture,
                "best_epoch": metrics["best_epoch"],
                "val_macro_auc_ovr": val_metrics["val_macro_auc_ovr"],
                "hyperparameters": hyperparameters,
                "optimizer": "AdamW",
            }, checkpoint_path)

            result = {
                "model": f"{method}_{role}",
                "method": method,
                "architecture": architecture,
                "selection": selection,
                "config": {
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": BATCH_SIZE,
                    "optimizer": "AdamW",
                    "hyperparameters": hyperparameters,
                    "hyperparameter_source": {
                        "summary": str(args.hp_summary),
                        "trial": best_trial,
                    },
                    "class_weights": class_weights.tolist() if class_weights is not None else None,
                    "scheduler": {"name": "StepLR", "step_size": SCHEDULER_STEP, "gamma": 0.1},
                    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                    "device": str(device),
                    "data_path": str(args.data_path),
                },
                "parameter_count": params,
                "macs": macs,
                "mac_convention": MAC_CONVENTION,
                "best_epoch": metrics["best_epoch"],
                "history": history,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "test_labels": test_labels,
                "test_predictions": test_preds,
                "checkpoint": str(checkpoint_path),
                "runtime_seconds": runtime,
            }
            save_json(model_dir / "result.json", result)

            compact = {
                "model": result["model"],
                "parameter_count": params,
                "macs": macs,
                "best_epoch": metrics["best_epoch"],
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "result": str(model_dir / "result.json"),
            }
            method_results.append(compact)
            print(
                f"[{method} {idx}/{len(candidates)}] params={params:,} macs={macs:,} "
                f"val_auc={val_metrics['val_macro_auc_ovr']:.4f} "
                f"test_auc={test_metrics['test_macro_auc_ovr']:.4f} "
                f"test_f1={test_metrics['test_macro_f1']:.4f} "
                f"({runtime:.1f}s)",
                flush=True,
            )

        method_summary = {
            "experiment": "pareto_retrain_ga_hyperparam",
            "method": method,
            "seed": args.seed,
            "hyperparameters": hyperparameters,
            "hyperparameter_source": str(args.hp_summary),
            "results": method_results,
        }
        save_json(run_dir / method / "summary.json", method_summary)
        overall_results[method] = method_results

    save_json(run_dir / "summary.json", {
        "experiment": "pareto_retrain_ga_hyperparam",
        "seed": args.seed,
        "hyperparameters": hyperparameters,
        "hyperparameter_source": {"summary": str(args.hp_summary), "trial": best_trial},
        "runtime_seconds": time.perf_counter() - started,
        "results": overall_results,
    })
    print(f"\nDONE! summary={run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
