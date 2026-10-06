"""Retrain the selected GA and random ResNets, then test them with the saved baseline."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import BASELINE_ARCHITECTURE, build_model, count_parameters
from src.search_runtime import save_json
from src.search_space import canonical_architecture
from src.train import DATA_PATH, DEFAULT_EPOCHS, PROJECT_ROOT, load_data, resolve_device, set_seed, train_model


TEST_SHAPES = {"test_images": (2005, 64, 64, 3), "test_labels": (2005, 1)}


def load_test_data(path: Path) -> TensorDataset:
    with np.load(path) as data:
        missing = TEST_SHAPES.keys() - data.files
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
        images, labels = data["test_images"], data["test_labels"]
    if images.shape != TEST_SHAPES["test_images"] or labels.shape != TEST_SHAPES["test_labels"]:
        raise ValueError(f"Unexpected test shapes: images={images.shape}, labels={labels.shape}")
    labels = labels.reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 6)):
        raise ValueError("test_labels must contain class IDs from 0 to 6")
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.5)
    return TensorDataset(images, torch.from_numpy(labels).long())


def evaluate(model, dataset, criterion, *, batch_size, device, prefix):
    model.eval()
    loss_sum = 0.0
    labels_all, probabilities_all = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        f"{prefix}_loss": loss_sum / len(dataset),
        f"{prefix}_accuracy": accuracy_score(labels, predictions),
        f"{prefix}_macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        f"{prefix}_macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        f"{prefix}_macro_f1": f1_score(labels, predictions, average="macro"),
        f"{prefix}_macro_auc_ovr": roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"),
    }
    return metrics, labels.tolist(), predictions.tolist()


def select_best_auc(summary_path: Path):
    if not summary_path.is_file():
        raise FileNotFoundError(f"Search summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evaluations") != 60 or summary.get("config", {}).get("seed") != 42:
        raise ValueError(f"Expected a completed 60-evaluation seed-42 search: {summary_path}")
    front = summary.get("pareto_front", [])
    if not front:
        raise ValueError(f"Pareto front is empty: {summary_path}")
    selected = max(front, key=lambda item: (item["val_macro_auc_ovr"], -item["parameter_count"]))
    return deepcopy(selected["architecture"]), {
        "rule": "maximum_validation_macro_auc_ovr",
        "source": str(summary_path),
        "search_val_macro_auc_ovr": selected["val_macro_auc_ovr"],
        "search_parameter_count": selected["parameter_count"],
    }


def finish_result(name, model, architecture, selection, history, best_epoch, runtime,
                  loaded_data, test_data, args, device, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    val_metrics, _, _ = evaluate(model, loaded_data[1], criterion, batch_size=64, device=device, prefix="val")
    test_metrics, test_labels, test_predictions = evaluate(
        model, test_data, criterion, batch_size=64, device=device, prefix="test",
    )
    result = {
        "model": name,
        "architecture": architecture,
        "selection": selection,
        "config": {
            "seed": args.seed, "epochs": args.epochs,
            "learning_rate": 0.001, "batch_size": 64, "optimizer": "Adam",
            "scheduler": {"name": "StepLR", "step_size": 20, "gamma": 0.1},
            "loss": "unweighted_cross_entropy",
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "device": str(device), "data_path": str(args.data_path),
        },
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
        "checkpoint": str(checkpoint_path),
        "runtime_seconds": runtime,
    }
    save_json(checkpoint_path.parent / "result.json", result)
    return result


def train_selected(name, architecture, selection, loaded_data, test_data, args, device):
    model_dir = args.output_dir / name / f"seed_{args.seed}"
    checkpoint_path = model_dir / "checkpoint.pt"
    result_path = model_dir / "result.json"
    if result_path.is_file() and checkpoint_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            canonical_architecture(result.get("architecture")) == canonical_architecture(architecture)
            and result.get("config", {}).get("seed") == args.seed
            and result.get("config", {}).get("epochs") == args.epochs
        ):
            print(f"model={name} reusing={result_path}", flush=True)
            return result
    set_seed(args.seed)
    model = build_model(architecture)
    metrics, history, runtime = train_model(
        model, loaded_data, learning_rate=0.001, batch_size=64,
        optimizer_name="Adam", epochs=args.epochs, seed=args.seed,
        device=device, scheduler_step_size=20,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": architecture,
        "best_epoch": metrics["best_epoch"],
        "val_macro_auc_ovr": metrics["val_macro_auc_ovr"],
    }, checkpoint_path)
    return finish_result(
        name, model, architecture, selection, history, metrics["best_epoch"], runtime,
        loaded_data, test_data, args, device, checkpoint_path,
    )


def evaluate_saved_baseline(loaded_data, test_data, args, device):
    result_path = args.baseline_dir / "result.json"
    source_checkpoint = args.baseline_dir / "checkpoint.pt"
    if not result_path.is_file() or not source_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline result/checkpoint not found in: {args.baseline_dir}")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    config = source_result.get("config", {})
    if (
        config.get("seed") != args.seed
        or config.get("epochs") != args.epochs
        or config.get("loss") != "unweighted_cross_entropy"
        or config.get("scheduler", {}).get("step_size") != 20
        or canonical_architecture(source_result.get("architecture")) != canonical_architecture(BASELINE_ARCHITECTURE)
    ):
        raise ValueError("Saved baseline does not match the Stage 5 protocol")
    # This checkpoint was created locally by src.train; legacy NumPy metadata is
    # incompatible with PyTorch's restricted weights-only unpickler.
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = build_model(BASELINE_ARCHITECTURE).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model_dir = args.output_dir / "baseline" / f"seed_{args.seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.pt"
    shutil.copy2(source_checkpoint, checkpoint_path)
    return finish_result(
        "baseline", model, deepcopy(BASELINE_ARCHITECTURE),
        {"rule": "reuse_saved_baseline", "source": str(result_path)},
        source_result["history"], source_result["metrics"]["best_epoch"],
        source_result["runtime_seconds"], loaded_data, test_data, args, device, checkpoint_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(60,), default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--ga-summary", type=Path, default=PROJECT_ROOT / "results" / "ga_search" / "seed_42" / "summary.json")
    parser.add_argument("--random-summary", type=Path, default=PROJECT_ROOT / "results" / "random_search" / "seed_42" / "summary.json")
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT_ROOT / "results" / "baseline" / "seed_42")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "test_eval")
    args = parser.parse_args()

    ga_architecture, ga_selection = select_best_auc(args.ga_summary)
    random_architecture, random_selection = select_best_auc(args.random_summary)
    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path)
    test_data = load_test_data(args.data_path)
    results = [
        train_selected("ga", ga_architecture, ga_selection, loaded_data, test_data, args, device),
        train_selected("random", random_architecture, random_selection, loaded_data, test_data, args, device),
        evaluate_saved_baseline(loaded_data, test_data, args, device),
    ]
    summary = {
        "experiment": "test_evaluation", "seed": args.seed,
        "selection_uses": "validation_only", "test_uses": "test_evaluation_only",
        "baseline_training": "reused_saved_checkpoint",
        "results": [{
            "model": result["model"], "parameter_count": result["parameter_count"],
            "best_epoch": result["best_epoch"], "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
            "result": str(args.output_dir / result["model"] / f"seed_{args.seed}" / "result.json"),
        } for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
