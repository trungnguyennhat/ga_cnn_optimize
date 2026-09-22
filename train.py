"""Train and validate the Stage 1 DermaMNIST baseline."""

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

from model import BaselineCNN


DATA_PATH = Path("data/dermamnist_64.npz")
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


def load_data(path: Path = DATA_PATH) -> tuple[TensorDataset, TensorDataset, torch.Tensor]:
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

    counts = np.bincount(train_labels, minlength=7)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training split")
    class_weights = torch.tensor(len(train_labels) / (7 * counts), dtype=torch.float32)

    def dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
        image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
        return TensorDataset(image_tensor, torch.from_numpy(labels).long())

    return (
        dataset(arrays["train_images"], train_labels),
        dataset(arrays["val_images"], val_labels),
        class_weights,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_baseline(
    *,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    dropout: float = 0.3,
    optimizer_name: str = "Adam",
    filters: int = 32,
    epochs: int = 15,
    seed: int = 42,
    device_name: str = "auto",
    data_path: Path = DATA_PATH,
    output_dir: Path | None = None,
) -> dict[str, float | int | str]:
    if learning_rate <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("learning_rate, batch_size, and epochs must be positive")

    set_seed(seed)
    device = resolve_device(device_name)
    train_data, val_data, class_weights = load_data(data_path)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = BaselineCNN(filters=filters, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError(f"optimizer must be one of {sorted(optimizers)}")
    optimizer = optimizers[optimizer_name](model.parameters(), lr=learning_rate)

    started = time.perf_counter()
    train_loss_history = []
    train_loss = 0.0
    for epoch in range(1, epochs + 1):
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
        train_loss_history.append(train_loss)
        print(f"epoch={epoch}/{epochs} train_loss={train_loss:.6f}")

    model.eval()
    val_loss_sum = 0.0
    labels_all, probabilities_all = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_loss_sum += criterion(logits, labels).item() * labels.size(0)
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())

    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    predictions_np = probabilities_np.argmax(axis=1)
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss_sum / len(val_data),
        "val_accuracy": accuracy_score(labels_np, predictions_np),
        "val_macro_f1": f1_score(labels_np, predictions_np, average="macro"),
        "val_macro_auc_ovr": roc_auc_score(
            labels_np, probabilities_np, average="macro", multi_class="ovr"
        ),
        "epochs": epochs,
        "seed": seed,
        "device": str(device),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"seed_{seed}.json"
        result = {
            "experiment": "baseline",
            "config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dropout": dropout,
                "optimizer": optimizer_name,
                "filters": filters,
                "epochs": epochs,
                "seed": seed,
                "device": str(device),
                "data_path": str(data_path),
            },
            "history": {"train_loss": train_loss_history},
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result={result_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="Adam")
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_baseline(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout=args.dropout,
        optimizer_name=args.optimizer,
        filters=args.filters,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))
