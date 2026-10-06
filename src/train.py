"""Train and validate the DermaMNIST baseline."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import BASELINE_ARCHITECTURE, BaselineResNet, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dermamnist_64.npz"
DEFAULT_EPOCHS = 60
EXPECTED = {
    "train_images": (7007, 64, 64, 3),
    "train_labels": (7007, 1),
    "val_images": (1003, 64, 64, 3),
    "val_labels": (1003, 1),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: Path = DATA_PATH) -> tuple[TensorDataset, TensorDataset]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path) as data:
        missing = EXPECTED.keys() - data.files
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")

        arrays = {key: data[key] for key in EXPECTED}

    for key, expected_shape in EXPECTED.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(
                f"{key} has shape {arrays[key].shape}; expected {expected_shape}"
            )

    train_labels = arrays["train_labels"].reshape(-1)
    val_labels = arrays["val_labels"].reshape(-1)
    for name, labels in (("train_labels", train_labels), ("val_labels", val_labels)):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} must contain integer class IDs")
        if np.any((labels < 0) | (labels > 6)):
            raise ValueError(f"{name} must contain only class IDs 0 through 6")

    def dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
        image_tensor = (
            torch.from_numpy(images).permute(0, 3, 1, 2).float()
            .div_(255).sub_(0.5).div_(0.5)
        )
        return TensorDataset(image_tensor, torch.from_numpy(labels).long())

    return (
        dataset(arrays["train_images"], train_labels),
        dataset(arrays["val_images"], val_labels),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def evaluate_model(model, dataset, criterion, *, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    loss_sum = 0.0
    labels_all, probabilities_all = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())

    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    predictions_np = probabilities_np.argmax(axis=1)
    return {
        "val_loss": loss_sum / len(dataset),
        "val_accuracy": accuracy_score(labels_np, predictions_np),
        "val_macro_f1": f1_score(labels_np, predictions_np, average="macro"),
        "val_macro_auc_ovr": roc_auc_score(
            labels_np, probabilities_np, average="macro", multi_class="ovr"
        ),
    }


def train_model(
    model, loaded_data, *, learning_rate, batch_size, optimizer_name, epochs,
    seed, device, scheduler_step_size=20,
):
    train_data, val_data = loaded_data
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError(f"optimizer must be one of {sorted(optimizers)}")
    optimizer = optimizers[optimizer_name](model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.1,
    )

    started = time.perf_counter()
    history = []
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        current_learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
        train_loss = loss_sum / len(train_data)
        val_metrics = evaluate_model(
            model, val_data, criterion, batch_size=batch_size, device=device,
        )
        history.append({
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_loss,
            **val_metrics,
        })
        if val_metrics["val_macro_auc_ovr"] > best_auc:
            best_auc = val_metrics["val_macro_auc_ovr"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch}/{epochs} lr={current_learning_rate:.6g} "
            f"train_loss={train_loss:.6f} "
            f"val_auc={val_metrics['val_macro_auc_ovr']:.6f}",
            flush=True,
        )
        scheduler.step()

    model.load_state_dict(best_state)
    best_record = history[best_epoch - 1]
    metrics = {
        "train_loss": best_record["train_loss"],
        "val_loss": best_record["val_loss"],
        "val_accuracy": best_record["val_accuracy"],
        "val_macro_f1": best_record["val_macro_f1"],
        "val_macro_auc_ovr": best_record["val_macro_auc_ovr"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "seed": seed,
        "device": str(device),
    }
    return metrics, history, time.perf_counter() - started


def train_baseline(
    *,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    optimizer_name: str = "Adam",
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 42,
    device_name: str = "auto",
    data_path: Path = DATA_PATH,
    output_dir: Path | None = None,
) -> dict[str, float | int | str]:
    if learning_rate <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("learning_rate, batch_size, and epochs must be positive")

    set_seed(seed)
    device = resolve_device(device_name)
    loaded_data = load_data(data_path)
    model = BaselineResNet()
    metrics, history, runtime = train_model(
        model, loaded_data, learning_rate=learning_rate,
        batch_size=batch_size, optimizer_name=optimizer_name, epochs=epochs,
        seed=seed, device=device,
    )
    if output_dir is not None:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "result.json"
        checkpoint_path = seed_dir / "checkpoint.pt"
        result = {
            "experiment": "baseline",
            "config": {
                "learning_rate": learning_rate,
                "scheduler": {"name": "StepLR", "step_size": 20, "gamma": 0.1},
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "loss": "unweighted_cross_entropy",
                "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "epochs": epochs,
                "seed": seed,
                "device": str(device),
                "data_path": str(data_path),
            },
            "architecture": BASELINE_ARCHITECTURE,
            "parameter_count": count_parameters(model),
            "history": history,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "runtime_seconds": runtime,
        }
        checkpoint = {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": BASELINE_ARCHITECTURE,
            "best_epoch": metrics["best_epoch"],
            "val_macro_auc_ovr": metrics["val_macro_auc_ovr"],
        }
        torch.save(checkpoint, checkpoint_path)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result={result_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="Adam")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "baseline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_baseline(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))
