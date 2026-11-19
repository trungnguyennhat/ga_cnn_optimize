"""Retrain the selected surrogate-search ResNet and evaluate it on test."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from src.datasets import DATASETS, add_dataset_args, dataset_path, results_root
from src.objectives import AUC, OBJECTIVES
from src.search_runtime import add_protocol_args
from src.test_eval import load_test_data, proxy_training_config, train_selected, verify_proxy_search
from src.train import load_data, resolve_device, resolve_protocol_settings, split_digest


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
    objectives = summary["config"].get("objectives", OBJECTIVES["params"])
    selected = max(front, key=lambda item: (item[AUC], *(-item[key] for key in objectives[1:])))
    return deepcopy(selected["architecture"]), {
        "rule": "maximum_validation_macro_auc_ovr",
        "source": str(summary_path),
        "search_evaluations": evaluations,
        "search_val_macro_auc_ovr": selected["val_macro_auc_ovr"],
        "search_parameter_count": selected["parameter_count"],
    }


def evaluate_selected_surrogate(args: argparse.Namespace) -> dict:
    architecture, selection = select_best_surrogate(
        args.summary, expected_seed=args.seed,
    )
    config = json.loads(args.summary.read_text(encoding="utf-8"))["config"]
    settings = resolve_protocol_settings(
        args.protocol, optimizer_name=args.optimizer, weight_decay=args.weight_decay,
        scheduler_name=args.scheduler, augmentation=args.augmentation, loss_name=args.loss,
        preload_gpu=args.preload_gpu, use_amp=args.amp,
    )
    args.data_path = dataset_path(args.dataset, args.data_path)
    args.num_classes = DATASETS[args.dataset].num_classes
    args.output_dir = args.output_dir or (
        results_root(args.protocol, args.dataset) / "surrogate_search" / "test_eval"
    )
    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path, args.dataset)
    args.train_val_sha256 = split_digest(loaded_data)
    verify_proxy_search(config, proxy_training_config(args, device, settings, config["epochs"]))
    test_data = load_test_data(args.data_path, args.dataset)
    args.test_sha256 = split_digest((test_data,))
    result = train_selected(
        "surrogate", architecture, selection, loaded_data, test_data, args, device, settings,
        model_dir=args.output_dir / f"seed_{args.seed}",
    )
    print(json.dumps(result["test_metrics"], indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--latency", action="store_true")
    parser.set_defaults(reuse_test_dir=None)
    add_protocol_args(parser)
    add_dataset_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_selected_surrogate(parse_args())
