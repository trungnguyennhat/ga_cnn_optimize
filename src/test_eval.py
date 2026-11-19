"""Retrain the selected GA and random ResNets, then test them with the saved baseline."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import DATASETS, dataset_path, load_splits, results_root
from src.model import BASELINE_ARCHITECTURE, build_model, count_macs, count_parameters, measure_latency_ms
from src.objectives import AUC, MACS, OBJECTIVES, PARAMS, cost_bounds, normalize
from src.search_runtime import SEARCH_SCHEDULER_STEP, add_search_args, save_json, search_root
from src.search_space import canonical_architecture
from src.train import (
    DEFAULT_EPOCHS,
    TRAINING_VERSION,
    build_criterion,
    is_legacy_protocol,
    load_data,
    loss_config_name,
    loss_weight,
    resolve_device,
    resolve_protocol_settings,
    scheduler_config_dict,
    set_seed,
    split_digest,
    train_model,
)


def load_test_data(path: Path, dataset: str = "dermamnist") -> TensorDataset:
    return load_splits(path, dataset, "test")[0]


def evaluate(
    model,
    dataset,
    criterion,
    *,
    batch_size,
    device,
    prefix,
    return_probabilities: bool = False,
):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    labels_all, probabilities_all = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            weight = float(loss_weight(criterion, labels))
            loss_sum += criterion(logits, labels).item() * weight
            weight_sum += weight
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        f"{prefix}_loss": loss_sum / weight_sum,
        f"{prefix}_accuracy": accuracy_score(labels, predictions),
        f"{prefix}_macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        f"{prefix}_macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        f"{prefix}_macro_f1": f1_score(labels, predictions, average="macro"),
        f"{prefix}_macro_auc_ovr": roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"),
    }
    if return_probabilities:
        return metrics, labels.tolist(), predictions.tolist(), probabilities.tolist()
    return metrics, labels.tolist(), predictions.tolist()


def _load_validated_summary(summary_path: Path, *, expected_seed: int, expected_budget: int) -> dict:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Search summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("evaluations") != expected_budget
        or len(summary.get("evaluated", [])) != expected_budget
        or summary.get("config", {}).get("seed") != expected_seed):
        raise ValueError(
            f"Expected a completed {expected_budget}-evaluation seed-{expected_seed} search: {summary_path}"
        )
    if not summary.get("pareto_front"):
        raise ValueError(f"Pareto front is empty: {summary_path}")
    return summary


def select_best_auc(summary_path: Path, *, expected_seed: int = 42, expected_budget: int = 60):
    summary = _load_validated_summary(
        summary_path, expected_seed=expected_seed, expected_budget=expected_budget
    )
    front = summary["pareto_front"]
    objectives = summary["config"].get("objectives", OBJECTIVES["params"])
    selected = max(front, key=lambda item: (item[AUC], *(-item[key] for key in objectives[1:])))
    return deepcopy(selected["architecture"]), {
        "rule": "maximum_validation_macro_auc_ovr",
        "source": str(summary_path),
        "search_val_macro_auc_ovr": selected["val_macro_auc_ovr"],
        "search_parameter_count": selected["parameter_count"],
    }


def select_pareto_representatives(
    method: str,
    summary_path: Path,
    *,
    baseline_val_auc: float,
    baseline_proxy_val_auc: float | None = None,
    expected_seed: int = 42,
    expected_budget: int = 60,
    knee_scale: str = "linear",
) -> list[tuple[str, dict, dict]]:
    summary = _load_validated_summary(
        summary_path, expected_seed=expected_seed, expected_budget=expected_budget
    )
    front = summary["pareto_front"]
    evaluated = [item for item in summary["evaluated"] if item.get("full_fidelity", True)]
    config = summary["config"]
    objectives = config.get("objectives", OBJECTIVES["params"])
    bounds = cost_bounds(config.get("architecture_space", "v1").rsplit("_", 1)[-1],
                         config.get("num_classes", 7))
    cost = objectives[1]
    max_auc_item = max(front, key=lambda item: (item[AUC], *(-item[key] for key in objectives[1:])))
    knee_item = min(
        front,
        key=lambda item: sum(
            (1 - value) ** 2 for value in normalize(
                tuple(item[key] for key in objectives), objectives, bounds, log_scale=knee_scale == "log",
            )
        ),
    )

    representatives = [
        (
            method,
            deepcopy(max_auc_item["architecture"]),
            {
                "rule": "maximum_validation_macro_auc_ovr",
                "source": str(summary_path),
                "search_val_macro_auc_ovr": max_auc_item["val_macro_auc_ovr"],
                "search_parameter_count": max_auc_item["parameter_count"],
            },
        ),
        (
            f"{method}_knee",
            deepcopy(knee_item["architecture"]),
            {
                "rule": "pareto_knee_point",
                "knee_scale": knee_scale,
                "objectives": list(objectives),
                "source": str(summary_path),
                "search_val_macro_auc_ovr": knee_item["val_macro_auc_ovr"],
                "search_parameter_count": knee_item["parameter_count"],
            },
        ),
    ]

    threshold_name = "baseline_full_val_auc" if baseline_proxy_val_auc is None else "baseline_matched_proxy_val_auc"
    threshold_auc = baseline_val_auc if baseline_proxy_val_auc is None else baseline_proxy_val_auc
    eligible = [item for item in evaluated if item[AUC] >= threshold_auc]
    if eligible:
        smallest_item = min(
            eligible,
            key=lambda item: (item[cost], -item[AUC], *(item[key] for key in objectives[2:])),
        )
        representatives.append((
            f"{method}_smallest_baseline",
            deepcopy(smallest_item["architecture"]),
            {
                "rule": f"smallest_model_reaching_{threshold_name}",
                "threshold_val_macro_auc_ovr": threshold_auc,
                "minimized_cost": cost,
                "source": str(summary_path),
                "search_val_macro_auc_ovr": smallest_item["val_macro_auc_ovr"],
                "search_parameter_count": smallest_item["parameter_count"],
            },
        ))
    return representatives


def _build_eval_config(args, device, settings: dict) -> dict:
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": settings["optimizer_name"],
        "scheduler": scheduler_config_dict(
            settings["scheduler_name"],
            epochs=args.epochs,
            step_size=args.scheduler_step_size,
        ),
        "loss": loss_config_name(settings["loss_name"]),
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "device": str(device),
        "data_path": str(args.data_path),
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "train_val_sha256": args.train_val_sha256,
        "preload_gpu": settings["preload_gpu"],
    }
    if args.dataset != "dermamnist" or not is_legacy_protocol(settings):
        config.update({
            "protocol": settings["protocol"],
            "weight_decay": settings["weight_decay"],
            "augmentation": settings["augmentation"],
            "amp": settings["use_amp"],
            "training_version": TRAINING_VERSION,
        })
    return config


def _config_matches(saved_config: dict, expected_config: dict) -> bool:
    keys = ("seed", "epochs", "optimizer", "scheduler", "loss", "learning_rate", "batch_size", "normalization")
    if any(saved_config.get(key) != expected_config.get(key) for key in keys):
        return False
    if saved_config.get("protocol", "v1") != expected_config.get("protocol", "v1"):
        return False
    if saved_config.get("weight_decay", 0.0) != expected_config.get("weight_decay", 0.0):
        return False
    if saved_config.get("augmentation", "none") != expected_config.get("augmentation", "none"):
        return False
    legacy_defaults = {"amp": False, "dataset": "dermamnist", "num_classes": 7, "preload_gpu": False}
    if any(saved_config.get(key, default) != expected_config.get(key, default)
           for key, default in legacy_defaults.items()):
        return False
    if saved_config.get("training_version") != expected_config.get("training_version"):
        return False
    legacy_digest = expected_config["train_val_sha256"] if expected_config.get("training_version") is None else None
    return saved_config.get("train_val_sha256", legacy_digest) == expected_config["train_val_sha256"]


def proxy_training_config(args, device, settings, epochs):
    config = _build_eval_config(args, device, settings)
    config.update(epochs=epochs, scheduler=scheduler_config_dict(
        settings["scheduler_name"], epochs=epochs, step_size=SEARCH_SCHEDULER_STEP,
    ))
    return config


def verify_proxy_search(config, expected):
    if config.get("training_version") not in (3, TRAINING_VERSION):
        raise ValueError("Search uses an unsupported training version")
    search_config = {**config, "training_version": expected.get("training_version")}
    if not _config_matches(search_config, expected):
        raise ValueError("Search settings do not match the requested protocol or dataset")


def finish_result(
    name, model, architecture, selection, history, best_epoch, runtime,
    loaded_data, test_data, args, device, checkpoint_path, settings: dict,
):
    criterion = build_criterion(settings["loss_name"], loaded_data[0].tensors[1], device, args.num_classes)
    val_metrics, _, _ = evaluate(
        model, loaded_data[1], criterion, batch_size=64, device=device, prefix="val"
    )
    test_metrics, test_labels, test_predictions, test_probabilities = evaluate(
        model,
        test_data,
        criterion,
        batch_size=64,
        device=device,
        prefix="test",
        return_probabilities=True,
    )
    result = {
        "model": name,
        "architecture": architecture,
        "selection": selection,
        "config": {**_build_eval_config(args, device, settings), "test_sha256": args.test_sha256},
        PARAMS: count_parameters(model),
        MACS: count_macs(model),
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "checkpoint": str(checkpoint_path),
        "runtime_seconds": runtime,
    }
    if args.latency:
        result.update(latency_ms=measure_latency_ms(model, device), latency_device=str(device))
    save_json(checkpoint_path.parent / "result.json", result)
    return result


def train_selected(
    name, architecture, selection, loaded_data, test_data, args, device,
    settings: dict, *, trained_cache: dict[str, dict] | None = None, model_dir: Path | None = None,
):
    model_dir = model_dir or args.output_dir / name / f"seed_{args.seed}"
    checkpoint_path = model_dir / "checkpoint.pt"
    result_path = model_dir / "result.json"
    expected_config = _build_eval_config(args, device, settings)
    arch_key = canonical_architecture(architecture)

    source_dirs = [model_dir]
    if args.reuse_test_dir is not None:
        source_dirs.append(args.reuse_test_dir / name / f"seed_{args.seed}")
    for source_dir in source_dirs:
        source_result = source_dir / "result.json"
        source_checkpoint = source_dir / "checkpoint.pt"
        if not source_result.is_file() or not source_checkpoint.is_file():
            continue
        result = json.loads(source_result.read_text(encoding="utf-8"))
        saved_arch = result.get("architecture")
        if (
            saved_arch is not None
            and canonical_architecture(saved_arch) == arch_key
            and _config_matches(result.get("config", {}), expected_config)
        ):
            model_dir.mkdir(parents=True, exist_ok=True)
            if source_dir.resolve() != model_dir.resolve():
                shutil.copy2(source_checkpoint, checkpoint_path)
            if ("test_probabilities" not in result
                or result["config"].get("test_sha256") != args.test_sha256 or args.latency):
                checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
                model = build_model(architecture, args.num_classes).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                result = finish_result(
                    name, model, architecture, selection, result["history"], result["best_epoch"],
                    result["runtime_seconds"], loaded_data, test_data, args, device,
                    checkpoint_path, settings,
                )
            result.update(model=name, selection=selection, checkpoint=str(checkpoint_path))
            save_json(result_path, result)
            print(f"model={name} reusing_checkpoint={source_checkpoint}", flush=True)
            if trained_cache is not None:
                trained_cache[arch_key] = result
            return result

    if trained_cache is not None and arch_key in trained_cache:
        cached = deepcopy(trained_cache[arch_key])
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(cached["checkpoint"]), checkpoint_path)
        cached["model"] = name
        cached["selection"] = selection
        cached["checkpoint"] = str(checkpoint_path)
        save_json(result_path, cached)
        print(f"model={name} reused_duplicate_arch={arch_key}", flush=True)
        return cached

    set_seed(args.seed)
    model = build_model(architecture, args.num_classes)
    metrics, history, runtime = train_model(
        model,
        loaded_data,
        learning_rate=0.001,
        batch_size=64,
        optimizer_name=settings["optimizer_name"],
        epochs=args.epochs,
        seed=args.seed,
        device=device,
        scheduler_step_size=args.scheduler_step_size,
        weight_decay=settings["weight_decay"],
        scheduler_name=settings["scheduler_name"],
        augmentation=settings["augmentation"],
        loss_name=settings["loss_name"],
        preload_gpu=settings["preload_gpu"],
        use_amp=settings["use_amp"],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": architecture,
        "best_epoch": metrics["best_epoch"],
        "val_macro_auc_ovr": metrics["val_macro_auc_ovr"],
    }, checkpoint_path)
    result = finish_result(
        name, model, architecture, selection, history, metrics["best_epoch"], runtime,
        loaded_data, test_data, args, device, checkpoint_path, settings,
    )
    if trained_cache is not None:
        trained_cache[arch_key] = result
    return result


def evaluate_saved_baseline(loaded_data, test_data, args, device, settings: dict):
    result_path = args.baseline_dir / "result.json"
    source_checkpoint = args.baseline_dir / "checkpoint.pt"
    if not result_path.is_file() or not source_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline result/checkpoint not found in: {args.baseline_dir}")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_config = source_result.get("config", {})
    expected_config = _build_eval_config(args, device, settings)
    if (
        not _config_matches(saved_config, expected_config)
        or canonical_architecture(source_result.get("architecture"))
        != canonical_architecture(BASELINE_ARCHITECTURE)
    ):
        raise ValueError(
            f"Saved baseline in {args.baseline_dir} does not match the requested protocol/config"
        )
    # This checkpoint was created locally by src.train; legacy NumPy metadata is
    # incompatible with PyTorch's restricted weights-only unpickler.
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = build_model(BASELINE_ARCHITECTURE, args.num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model_dir = args.output_dir / "baseline" / f"seed_{args.seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.pt"
    if source_checkpoint.resolve() != checkpoint_path.resolve():
        shutil.copy2(source_checkpoint, checkpoint_path)
    return finish_result(
        "baseline", model, deepcopy(BASELINE_ARCHITECTURE),
        {"rule": "reuse_saved_baseline", "source": str(result_path)},
        source_result["history"], source_result["metrics"]["best_epoch"],
        source_result["runtime_seconds"], loaded_data, test_data, args, device,
        checkpoint_path, settings,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--expected-budget", type=int, default=60)
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ga-summary", type=Path, default=None)
    parser.add_argument("--random-summary", type=Path, default=None)
    parser.add_argument("--extra-summary", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--baseline-proxy-result", type=Path)
    parser.add_argument("--reuse-test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pareto-representatives", action="store_true")
    parser.add_argument("--knee-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--latency", action="store_true")
    add_search_args(parser)
    args = parser.parse_args()

    base_results = search_root(args)
    protocol_results = results_root(args.protocol, args.dataset)
    args.data_path = dataset_path(args.dataset, args.data_path)
    args.num_classes = DATASETS[args.dataset].num_classes
    if args.ga_summary is None:
        args.ga_summary = base_results / "ga_search" / f"seed_{args.seed}" / "summary.json"
    if args.random_summary is None:
        args.random_summary = base_results / "random_search" / f"seed_{args.seed}" / "summary.json"
    if args.baseline_dir is None:
        args.baseline_dir = protocol_results / "baseline" / f"seed_{args.seed}"
    if args.output_dir is None:
        args.output_dir = base_results / "test_eval"

    settings = resolve_protocol_settings(
        args.protocol,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        augmentation=args.augmentation,
        loss_name=args.loss,
        preload_gpu=args.preload_gpu,
        use_amp=args.amp,
    )

    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path, args.dataset)
    args.train_val_sha256 = split_digest(loaded_data)
    baseline = json.loads((args.baseline_dir / "result.json").read_text(encoding="utf-8"))
    baseline_val_auc = baseline["metrics"][AUC]
    proxy = None
    if args.baseline_proxy_result is not None:
        proxy = json.loads(args.baseline_proxy_result.read_text(encoding="utf-8"))
        if canonical_architecture(proxy["architecture"]) != canonical_architecture(BASELINE_ARCHITECTURE):
            parser.error("--baseline-proxy-result must reference the ResNet-18 baseline")

    paths = {"ga": args.ga_summary, "random": args.random_summary}
    for spec in args.extra_summary:
        method, separator, path = spec.partition("=")
        if (not separator or not path or not method.isascii() or not method.islower()
            or not method.isidentifier() or method in {*paths, "baseline"}
            or method.endswith(("_knee", "_smallest_baseline"))):
            parser.error("--extra-summary requires a unique method NAME=PATH")
        paths[method] = Path(path)
    candidates = []
    for method, path in paths.items():
        budget = args.expected_budget if method in ("ga", "random") else json.loads(
            path.read_text(encoding="utf-8")
        )["evaluations"]
        search = _load_validated_summary(path, expected_seed=args.seed, expected_budget=budget)
        config = search["config"]
        comparison = proxy_training_config(args, device, settings, config["epochs"])
        verify_proxy_search(config, comparison)
        if method in ("ga", "random") and (
            config.get("architecture_space", "v1").rsplit("_", 1)[-1] != args.space
            or tuple(config.get("objectives", OBJECTIVES["params"])) != OBJECTIVES[args.objectives]
        ):
            raise ValueError(f"Search space or objectives do not match the requested configuration: {path}")
        if proxy is not None and not _config_matches(proxy["config"], comparison):
            raise ValueError("Baseline proxy configuration must match each search configuration")
        if args.pareto_representatives:
            candidates.extend(select_pareto_representatives(
                method, path, baseline_val_auc=baseline_val_auc,
                baseline_proxy_val_auc=None if proxy is None else proxy["metrics"][AUC],
                expected_seed=args.seed, expected_budget=budget, knee_scale=args.knee_scale,
            ))
        else:
            architecture, selection = select_best_auc(path, expected_seed=args.seed, expected_budget=budget)
            candidates.append((method, architecture, selection))

    test_data = load_test_data(args.data_path, args.dataset)
    args.test_sha256 = split_digest((test_data,))
    baseline_result = evaluate_saved_baseline(loaded_data, test_data, args, device, settings)
    trained_cache: dict[str, dict] = {}
    results = [
        train_selected(
            name, arch, sel, loaded_data, test_data, args, device, settings,
            trained_cache=trained_cache,
        )
        for name, arch, sel in candidates
    ]
    results.append(baseline_result)

    summary = {
        "experiment": "test_evaluation",
        "protocol": settings["protocol"],
        "dataset": args.dataset,
        "objectives": list(OBJECTIVES[args.objectives]),
        "seed": args.seed,
        "selection_uses": "validation_only",
        "test_uses": "test_evaluation_only",
        "baseline_training": "reused_saved_checkpoint",
        "results": [{
            "model": result["model"],
            "parameter_count": result["parameter_count"],
            MACS: result[MACS],
            "selection": result["selection"],
            "best_epoch": result["best_epoch"],
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
            "result": str(args.output_dir / result["model"] / f"seed_{args.seed}" / "result.json"),
        } for result in results],
    }
    seed_dir = args.output_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    save_json(seed_dir / "summary.json", summary)
    if args.seed == 42:
        save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
