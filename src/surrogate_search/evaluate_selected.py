"""Retrain the selected surrogate-search ResNet and evaluate it on test."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from src.model import build_model, count_parameters
from src.search_runtime import save_json
from src.search_space import canonical_architecture
from src.test_eval import evaluate, load_test_data, select_three_architectures, train_selected
from src.train import DATA_PATH, PROJECT_ROOT, load_data, resolve_device, set_seed, train_model


def select_best_surrogate(summary_path: Path, *, expected_seed: int) -> tuple[dict, dict]:
    """Select the highest-validation-AUC architecture from a surrogate Pareto front."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Surrogate summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != "surrogate_search":
        raise ValueError(f"Expected a surrogate_search summary: {summary_path}")
    if summary.get("config", {}).get("seed") != expected_seed:
        raise ValueError(f"Surrogate summary seed does not match {expected_seed}")
    evaluations = summary.get("evaluations")
    if type(evaluations) is not int or evaluations <= 0:
        raise ValueError("Surrogate summary must contain completed evaluations")
    front = summary.get("pareto_front", [])
    if not front:
        raise ValueError("Surrogate Pareto front is empty")
    selected = max(
        front,
        key=lambda item: (item["val_macro_auc_ovr"], -item["parameter_count"]),
    )
    return deepcopy(selected["architecture"]), {
        "rule": "maximum_validation_macro_auc_ovr",
        "source": str(summary_path),
        "search_evaluations": evaluations,
        "search_val_macro_auc_ovr": selected["val_macro_auc_ovr"],
        "search_parameter_count": selected["parameter_count"],
    }


def evaluate_selected_surrogate(args: argparse.Namespace) -> dict:
    if getattr(args, "hardware_aware", False):
        candidates = select_three_architectures(args.summary, args.seed, "surrogate_search")
        device = resolve_device(args.device)
        loaded_data = load_data(args.data_path)
        test_data = load_test_data(args.data_path)
        results = [
            train_selected(
                f"surrogate_{role}", architecture, selection,
                loaded_data, test_data, args, device,
            )
            for role, architecture, selection in candidates
        ]
        summary = {
            "seed": args.seed,
            "selection_uses": "validation_only",
            "test_uses": "test_evaluation_only",
            "results": [{
                "model": result["model"],
                "selection": result["selection"],
                "test_metrics": result["test_metrics"],
                "result": str(Path(result["checkpoint"]).with_name("result.json")),
            } for result in results],
        }
        summary_dir = args.output_dir / f"seed_{args.seed}"
        summary_dir.mkdir(parents=True, exist_ok=True)
        save_json(summary_dir / "summary.json", summary)
        return summary

    architecture, selection = select_best_surrogate(
        args.summary, expected_seed=args.seed,
    )
    output = args.output_dir / f"seed_{args.seed}"
    result_path = output / "result.json"
    checkpoint_path = output / "checkpoint.pt"
    if result_path.is_file() and checkpoint_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            canonical_architecture(result.get("architecture"))
            == canonical_architecture(architecture)
            and result.get("config", {}).get("seed") == args.seed
            and result.get("config", {}).get("epochs") == args.epochs
        ):
            print(f"reusing={result_path}", flush=True)
            return result

    set_seed(args.seed)
    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path)
    test_data = load_test_data(args.data_path)
    model = build_model(architecture)
    metrics, history, runtime = train_model(
        model,
        loaded_data,
        learning_rate=0.001,
        batch_size=64,
        optimizer_name="Adam",
        epochs=args.epochs,
        seed=args.seed,
        device=device,
        scheduler_step_size=20,
    )

    criterion = nn.CrossEntropyLoss()
    validation_metrics, _, _ = evaluate(
        model, loaded_data[1], criterion,
        batch_size=64, device=device, prefix="val",
    )
    test_metrics, test_labels, test_predictions = evaluate(
        model, test_data, criterion,
        batch_size=64, device=device, prefix="test",
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "architecture": architecture,
        "best_epoch": metrics["best_epoch"],
        "val_macro_auc_ovr": metrics["val_macro_auc_ovr"],
    }, checkpoint_path)
    result = {
        "model": "surrogate",
        "architecture": architecture,
        "selection": selection,
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "Adam",
            "scheduler": {"name": "StepLR", "step_size": 20, "gamma": 0.1},
            "loss": "unweighted_cross_entropy",
            "normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "device": str(device),
            "data_path": str(args.data_path),
        },
        "parameter_count": count_parameters(model),
        "best_epoch": metrics["best_epoch"],
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
        "checkpoint": str(checkpoint_path),
        "runtime_seconds": runtime,
    }
    save_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hardware-aware", action="store_true", help="Retrain three Pareto architectures")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(60,), default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "surrogate_search" / "test_eval",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_selected_surrogate(parse_args())
