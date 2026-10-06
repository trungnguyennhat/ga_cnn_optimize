"""Retrain the selected GA, random-search, and baseline models, then test them."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import BASELINE_ARCHITECTURE, build_model, count_parameters
from src.search_runtime import save_json
from src.train import DATA_PATH, DEFAULT_EPOCHS, PROJECT_ROOT, load_data, resolve_device, set_seed


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
    return TensorDataset(
        torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255),
        torch.from_numpy(labels).long(),
    )


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
    if summary.get("evaluations") != 80 or summary.get("config", {}).get("seed") != 42:
        raise ValueError(f"Expected a completed 80-evaluation seed-42 search: {summary_path}")
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


def train_final(name, architecture, selection, loaded_data, test_data, args, device):
    train_data, val_data, class_weights = loaded_data
    set_seed(args.seed)
    model = build_model(architecture).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
        train_loss = loss_sum / len(train_data)
        val_metrics, _, _ = evaluate(
            model, val_data, criterion, batch_size=64, device=device, prefix="val",
        )
        val_auc = val_metrics["val_macro_auc_ovr"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"model={name} epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_auc={val_auc:.6f}", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"model={name} early_stopping best_epoch={best_epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    val_metrics, _, _ = evaluate(model, val_data, criterion, batch_size=64, device=device, prefix="val")
    test_metrics, test_labels, test_predictions = evaluate(
        model, test_data, criterion, batch_size=64, device=device, prefix="test",
    )
    model_dir = args.output_dir / name / f"seed_{args.seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "architecture": architecture}, checkpoint_path)
    result = {
        "model": name,
        "architecture": architecture,
        "selection": selection,
        "config": {
            "seed": args.seed, "max_epochs": args.epochs, "patience": args.patience,
            "learning_rate": 0.001, "batch_size": 64, "optimizer": "Adam",
            "early_stopping_metric": "val_macro_auc_ovr", "device": str(device),
            "data_path": str(args.data_path),
        },
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
        "checkpoint": str(checkpoint_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    save_json(model_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--ga-summary", type=Path, default=PROJECT_ROOT / "results" / "ga_search" / "seed_42" / "summary.json")
    parser.add_argument("--random-summary", type=Path, default=PROJECT_ROOT / "results" / "random_search" / "seed_42" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "final_eval")
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("epochs and patience must be positive")

    ga_architecture, ga_selection = select_best_auc(args.ga_summary)
    random_architecture, random_selection = select_best_auc(args.random_summary)
    selected = {
        "ga": (ga_architecture, ga_selection),
        "random": (random_architecture, random_selection),
        "baseline": (deepcopy(BASELINE_ARCHITECTURE), {"rule": "fixed_baseline_architecture"}),
    }
    device = resolve_device(args.device)
    loaded_data = load_data(args.data_path)
    test_data = load_test_data(args.data_path)
    results = [
        train_final(name, architecture, selection, loaded_data, test_data, args, device)
        for name, (architecture, selection) in selected.items()
    ]
    summary = {
        "experiment": "final_evaluation", "seed": args.seed,
        "selection_uses": "validation_only", "test_uses": "final_evaluation_only",
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
