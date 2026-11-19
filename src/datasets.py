"""Local MedMNIST RGB64 datasets with their official splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class DatasetSpec:
    classes: tuple[str, ...]
    split_sizes: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


DATASETS = {
    "dermamnist": DatasetSpec(
        ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
        {"train": 7007, "val": 1003, "test": 2005},
    ),
    "bloodmnist": DatasetSpec(
        ("basophil", "eosinophil", "erythroblast", "immature granulocytes",
         "lymphocyte", "monocyte", "neutrophil", "platelet"),
        {"train": 11959, "val": 1712, "test": 3421},
    ),
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dermamnist_64.npz"


def dataset_path(name: str, path: Path | None) -> Path:
    return path if path is not None else PROJECT_ROOT / "data" / f"{name}_64.npz"


def results_root(protocol: str, name: str = "dermamnist", variant: str = "") -> Path:
    root = PROJECT_ROOT / "results"
    if name != "dermamnist":
        root /= name
    if protocol != "v1":
        root /= f"{protocol}_protocol"
    return root / variant if variant else root


def add_dataset_args(parser) -> None:
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="dermamnist")
    parser.add_argument("--data-path", type=Path)


def load_splits(path: Path, name: str, *splits: str) -> tuple[TensorDataset, ...]:
    spec = DATASETS[name]
    datasets = []
    with np.load(path, allow_pickle=False) as data:
        for split in splits:
            count = spec.split_sizes[split]
            images = data[f"{split}_images"]
            labels = data[f"{split}_labels"]
            if images.shape != (count, 64, 64, 3) or labels.shape not in ((count,), (count, 1)):
                raise ValueError(f"Unexpected {name}/{split} shapes in {path}: {images.shape}, {labels.shape}")
            if not np.issubdtype(labels.dtype, np.integer) or np.any(
                (labels < 0) | (labels >= spec.num_classes)
            ):
                raise ValueError(f"Invalid class IDs in {name}/{split}")
            image_tensor = (
                torch.from_numpy(images).permute(0, 3, 1, 2).float()
                .div_(255).sub_(0.5).div_(0.5)
            )
            datasets.append(TensorDataset(image_tensor, torch.from_numpy(labels.reshape(-1)).long()))
    return tuple(datasets)
