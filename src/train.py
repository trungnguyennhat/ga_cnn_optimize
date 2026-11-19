"""Train and validate the DermaMNIST baseline."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import DATA_PATH, DATASETS, PROJECT_ROOT, add_dataset_args, dataset_path, load_splits, results_root
from src.model import BASELINE_ARCHITECTURE, BaselineResNet, build_model, count_parameters
from src.search_space import validate_architecture


DEFAULT_EPOCHS = 60
NUM_CLASSES = 7
TRAINING_VERSION = 5

PROTOCOL_PRESETS = {
    "v1": {
        "optimizer_name": "Adam",
        "weight_decay": 0.0,
        "scheduler_name": "StepLR",
        "augmentation": "none",
        "loss_name": "unweighted_ce",
        "preload_gpu": False,
        "use_amp": False,
    },
    "v2": {
        "optimizer_name": "AdamW",
        "weight_decay": 1e-4,
        "scheduler_name": "CosineAnnealingLR",
        "augmentation": "derma_v1",
        "loss_name": "weighted_ce_sqrt",
        "preload_gpu": True,
        "use_amp": False,
    },
}

LOSS_CHOICES = ("unweighted_ce", "weighted_ce_sqrt", "weighted_ce_inv", "focal")
SCHEDULER_CHOICES = ("StepLR", "CosineAnnealingLR")
AUGMENTATION_CHOICES = ("none", "derma_v1")


def resolve_protocol_settings(
    protocol: str = "v1",
    *,
    optimizer_name: str | None = None,
    weight_decay: float | None = None,
    scheduler_name: str | None = None,
    augmentation: str | None = None,
    loss_name: str | None = None,
    preload_gpu: bool | None = None,
    use_amp: bool | None = None,
) -> dict:
    if protocol not in PROTOCOL_PRESETS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOL_PRESETS)}")
    settings = dict(PROTOCOL_PRESETS[protocol])
    settings["protocol"] = protocol
    overrides = {
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "augmentation": augmentation,
        "loss_name": loss_name,
        "preload_gpu": preload_gpu,
        "use_amp": use_amp,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if settings["scheduler_name"] not in SCHEDULER_CHOICES:
        raise ValueError(f"scheduler must be one of {SCHEDULER_CHOICES}")
    if settings["augmentation"] not in AUGMENTATION_CHOICES:
        raise ValueError(f"augmentation must be one of {AUGMENTATION_CHOICES}")
    if settings["loss_name"] not in LOSS_CHOICES:
        raise ValueError(f"loss must be one of {LOSS_CHOICES}")
    if settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    return settings


def is_legacy_protocol(settings: dict) -> bool:
    return settings["protocol"] == "v1" and all(
        settings[key] == value for key, value in PROTOCOL_PRESETS["v1"].items()
    )


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with mean reduction for imbalanced classification."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_index = targets.unsqueeze(1)
        target_log_probs = log_probs.gather(1, target_index).squeeze(1)
        target_probs = probs.gather(1, target_index).squeeze(1)
        return (-((1.0 - target_probs) ** self.gamma) * target_log_probs).mean()


def compute_class_weights(
    labels: torch.Tensor, *, num_classes: int = NUM_CLASSES, power: float = 0.5
) -> torch.Tensor:
    counts = torch.bincount(labels.detach().cpu().long(), minlength=num_classes).float()
    total = float(counts.sum().item())
    weights = (total / counts.clamp(min=1.0)) ** power
    return weights / weights.mean()


def build_criterion(
    loss_name: str, train_labels: torch.Tensor, device: torch.device, num_classes: int = NUM_CLASSES
) -> nn.Module:
    if loss_name == "unweighted_ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted_ce_sqrt":
        weights = compute_class_weights(train_labels, num_classes=num_classes, power=0.5).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "weighted_ce_inv":
        weights = compute_class_weights(train_labels, num_classes=num_classes, power=1.0).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"loss must be one of {LOSS_CHOICES}")


def augment_batch(
    images: torch.Tensor, augmentation: str, generator: torch.Generator
) -> torch.Tensor:
    """Apply deterministic, label-preserving dermoscopy augmentations on [-1, 1] tensors."""
    if augmentation == "none":
        return images
    if augmentation != "derma_v1":
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    batch_size = images.size(0)
    device = images.device
    out = images

    h_mask = (torch.rand(batch_size, generator=generator) < 0.5).to(device)
    if h_mask.any():
        out = torch.where(h_mask[:, None, None, None], out.flip(-1), out)

    v_mask = (torch.rand(batch_size, generator=generator) < 0.5).to(device)
    if v_mask.any():
        out = torch.where(v_mask[:, None, None, None], out.flip(-2), out)

    rot_k = torch.randint(0, 4, (batch_size,), generator=generator).to(device)
    if (rot_k != 0).any():
        out = out.clone()
        for k in (1, 2, 3):
            mask = rot_k == k
            if mask.any():
                out[mask] = torch.rot90(out[mask], k, dims=(-2, -1))

    # Mild brightness/contrast jitter (+/-10%) in [0, 1] without distorting lesion hue.
    out01 = out * 0.5 + 0.5
    brightness = (0.9 + 0.2 * torch.rand((batch_size, 1, 1, 1), generator=generator)).to(device)
    out01 = (out01 * brightness).clamp(0.0, 1.0)
    contrast = (0.9 + 0.2 * torch.rand((batch_size, 1, 1, 1), generator=generator)).to(device)
    mean = out01.mean(dim=(-2, -1), keepdim=True)
    out01 = ((out01 - mean) * contrast + mean).clamp(0.0, 1.0)
    return out01 * 2.0 - 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: Path = DATA_PATH, name: str = "dermamnist") -> tuple[TensorDataset, TensorDataset]:
    train, val = load_splits(path, name, "train", "val")
    return train, val


def split_digest(datasets) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for tensor in dataset.tensors:
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def preload_data(loaded_data, device):
    return tuple(
        TensorDataset(*(tensor.to(device) for tensor in dataset.tensors))
        for dataset in loaded_data
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def scheduler_config_dict(
    scheduler_name: str, *, epochs: int, step_size: int
) -> dict:
    if scheduler_name == "StepLR":
        return {"name": "StepLR", "step_size": step_size, "gamma": 0.1}
    if scheduler_name == "CosineAnnealingLR":
        return {"name": "CosineAnnealingLR", "t_max": epochs, "eta_min": 0.0}
    raise ValueError(f"scheduler must be one of {SCHEDULER_CHOICES}")


def loss_config_name(loss_name: str) -> str:
    if loss_name == "unweighted_ce":
        return "unweighted_cross_entropy"
    return loss_name


def _same_device(tensor_device: torch.device, target_device: torch.device) -> bool:
    if tensor_device.type != target_device.type:
        return False
    if target_device.type != "cuda" or target_device.index is None:
        return True
    return tensor_device.index == target_device.index


def loss_weight(criterion, labels):
    weights = getattr(criterion, "weight", None)
    return labels.size(0) if weights is None else weights[labels].sum()


def evaluate_model(model, dataset, criterion, *, batch_size, device):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    weight_sum = torch.zeros_like(loss_sum)
    labels_all, probabilities_all = [], []
    preloaded = hasattr(dataset, "tensors") and _same_device(dataset.tensors[0].device, device)
    loader = (
        (dataset[start : start + batch_size] for start in range(0, len(dataset), batch_size))
        if preloaded else DataLoader(dataset, batch_size=batch_size)
    )
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            weight = loss_weight(criterion, labels)
            loss_sum += criterion(logits, labels).double() * weight
            weight_sum += weight
            labels_all.append(labels)
            probabilities_all.append(torch.softmax(logits, dim=1))

    labels_np = torch.cat(labels_all).cpu().numpy()
    probabilities_np = torch.cat(probabilities_all).cpu().numpy()
    predictions_np = probabilities_np.argmax(axis=1)
    return {
        "val_loss": (loss_sum / weight_sum).item(),
        "val_accuracy": accuracy_score(labels_np, predictions_np),
        "val_macro_recall": recall_score(
            labels_np, predictions_np, average="macro", zero_division=0
        ),
        "val_macro_f1": f1_score(labels_np, predictions_np, average="macro"),
        "val_macro_auc_ovr": roc_auc_score(
            labels_np, probabilities_np, average="macro", multi_class="ovr"
        ),
    }


def _iter_train_batches(
    train_data, *, batch_size: int, device: torch.device,
    shuffle_generator: torch.Generator, preload_gpu: bool,
):
    if preload_gpu and device.type == "cuda" and hasattr(train_data, "tensors"):
        images_all, labels_all = train_data.tensors
        permutation = torch.randperm(len(labels_all), generator=shuffle_generator).to(device)
        for start in range(0, len(labels_all), batch_size):
            indices = permutation[start : start + batch_size]
            yield images_all[indices], labels_all[indices]
        return
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )
    for images, labels in loader:
        yield images.to(device), labels.to(device)


def train_model(
    model, loaded_data, *, learning_rate, batch_size, optimizer_name, epochs,
    seed, device, scheduler_step_size=20,
    weight_decay: float = 0.0,
    scheduler_name: str = "StepLR",
    augmentation: str = "none",
    loss_name: str = "unweighted_ce",
    preload_gpu: bool = False,
    use_amp: bool = False,
    epoch_callback=None,
):
    train_data, val_data = loaded_data
    if preload_gpu and device.type == "cuda" and hasattr(train_data, "tensors") and hasattr(val_data, "tensors"):
        train_data, val_data = preload_data(loaded_data, device)

    shuffle_generator = torch.Generator().manual_seed(seed)
    aug_generator = torch.Generator().manual_seed(seed + 100003)

    model = model.to(device)
    criterion = build_criterion(loss_name, train_data.tensors[1], device, model.classifier.out_features)
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError(f"optimizer must be one of {sorted(optimizers)}")
    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=0.1,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
        )
    else:
        raise ValueError(f"scheduler must be one of {SCHEDULER_CHOICES}")

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    started = time.perf_counter()
    history = []
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        weight_sum = torch.zeros_like(loss_sum)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in _iter_train_batches(
            train_data,
            batch_size=batch_size,
            device=device,
            shuffle_generator=shuffle_generator,
            preload_gpu=preload_gpu,
        ):
            images = augment_batch(images, augmentation, aug_generator)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = criterion(model(images), labels)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            weight = loss_weight(criterion, labels)
            loss_sum += loss.detach().double() * weight
            weight_sum += weight
        train_loss = (loss_sum / weight_sum).item()
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
        if epoch_callback is not None and epoch < epochs and not epoch_callback(history):
            break

    model.load_state_dict(best_state)
    best_record = history[best_epoch - 1]
    final_record = history[-1]
    metrics = {
        "train_loss": best_record["train_loss"],
        "val_loss": best_record["val_loss"],
        "val_accuracy": best_record["val_accuracy"],
        "val_macro_recall": best_record["val_macro_recall"],
        "val_macro_f1": best_record["val_macro_f1"],
        "val_macro_auc_ovr": best_record["val_macro_auc_ovr"],
        "final_train_loss": final_record["train_loss"],
        "final_val_loss": final_record["val_loss"],
        "overfit_gap_best": best_record["val_loss"] - best_record["train_loss"],
        "overfit_gap_final": final_record["val_loss"] - final_record["train_loss"],
        "epochs": epochs,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "seed": seed,
        "device": str(device),
    }
    return metrics, history, time.perf_counter() - started


def train_baseline(
    *,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    optimizer_name: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 42,
    device_name: str = "auto",
    data_path: Path | None = None,
    output_dir: Path | None = None,
    protocol: str = "v1",
    weight_decay: float | None = None,
    scheduler_name: str | None = None,
    scheduler_step_size: int = 20,
    augmentation: str | None = None,
    loss_name: str | None = None,
    preload_gpu: bool | None = None,
    use_amp: bool | None = None,
    architecture: dict | None = None,
    experiment_name: str = "baseline",
    dataset: str = "dermamnist",
) -> dict[str, float | int | str]:
    if learning_rate <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("learning_rate, batch_size, and epochs must be positive")

    settings = resolve_protocol_settings(
        protocol,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        scheduler_name=scheduler_name,
        augmentation=augmentation,
        loss_name=loss_name,
        preload_gpu=preload_gpu,
        use_amp=use_amp,
    )
    set_seed(seed)
    device = resolve_device(device_name)
    data_path = dataset_path(dataset, data_path)
    loaded_data = load_data(data_path, dataset)
    num_classes = DATASETS[dataset].num_classes
    if architecture is None:
        selected_arch = dict(BASELINE_ARCHITECTURE)
        model = BaselineResNet(num_classes)
    else:
        validate_architecture(architecture)
        selected_arch = architecture
        model = build_model(selected_arch, num_classes)

    metrics, history, runtime = train_model(
        model,
        loaded_data,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer_name=settings["optimizer_name"],
        epochs=epochs,
        seed=seed,
        device=device,
        scheduler_step_size=scheduler_step_size,
        weight_decay=settings["weight_decay"],
        scheduler_name=settings["scheduler_name"],
        augmentation=settings["augmentation"],
        loss_name=settings["loss_name"],
        preload_gpu=settings["preload_gpu"],
        use_amp=settings["use_amp"],
    )
    if output_dir is not None:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "result.json"
        checkpoint_path = seed_dir / "checkpoint.pt"
        config_dict = {
            "learning_rate": learning_rate,
            "scheduler": scheduler_config_dict(
                settings["scheduler_name"], epochs=epochs, step_size=scheduler_step_size
            ),
            "batch_size": batch_size,
            "optimizer": settings["optimizer_name"],
            "loss": loss_config_name(settings["loss_name"]),
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "epochs": epochs,
            "seed": seed,
            "device": str(device),
            "data_path": str(data_path),
            "dataset": dataset,
            "num_classes": num_classes,
            "train_val_sha256": split_digest(loaded_data),
            "preload_gpu": settings["preload_gpu"],
        }
        if dataset != "dermamnist" or not is_legacy_protocol(settings):
            config_dict.update({
                "protocol": settings["protocol"],
                "weight_decay": settings["weight_decay"],
                "augmentation": settings["augmentation"],
                "preload_gpu": settings["preload_gpu"],
                "amp": settings["use_amp"],
                "training_version": TRAINING_VERSION,
            })
        result = {
            "experiment": experiment_name,
            "config": config_dict,
            "architecture": selected_arch,
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
            "architecture": selected_arch,
            "best_epoch": metrics["best_epoch"],
            "val_macro_auc_ovr": metrics["val_macro_auc_ovr"],
        }
        torch.save(checkpoint, checkpoint_path)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result={result_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default=None)
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--augmentation", choices=AUGMENTATION_CHOICES, default=None)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default=None)
    parser.add_argument("--preload-gpu", action="store_true", default=None)
    parser.add_argument("--no-preload-gpu", dest="preload_gpu", action="store_false")
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--architecture-json", type=Path, default=None)
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = results_root(args.protocol, args.dataset) / "baseline"
    architecture = None
    if args.architecture_json is not None:
        payload = json.loads(args.architecture_json.read_text(encoding="utf-8"))
        architecture = payload["architecture"] if "architecture" in payload else payload
    metrics = train_baseline(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        data_path=args.data_path,
        dataset=args.dataset,
        output_dir=output_dir,
        protocol=args.protocol,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        augmentation=args.augmentation,
        loss_name=args.loss,
        preload_gpu=args.preload_gpu,
        use_amp=args.amp,
        architecture=architecture,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(metrics, indent=2))
