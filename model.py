"""CNN baseline and architecture builder for DermaMNIST GA-NAS."""

from copy import deepcopy

from torch import nn


FILTER_CHOICES = (16, 32, 64, 128)
KERNEL_CHOICES = (3, 5)
POOLING_CHOICES = ("max", "avg")
DROPOUT_CHOICES = (0.1, 0.2, 0.3, 0.4, 0.5)
MIN_BLOCKS = 2
MAX_BLOCKS = 4

BASELINE_ARCHITECTURE = {
    "blocks": [
        {"filters": 32, "kernel_size": 3, "pooling": "max", "batch_norm": False},
        {"filters": 64, "kernel_size": 3, "pooling": "max", "batch_norm": False},
        {"filters": 128, "kernel_size": 3, "pooling": "max", "batch_norm": False},
    ],
    "dropout": 0.3,
}


def validate_architecture(architecture: dict) -> None:
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be a dictionary")
    if set(architecture) != {"blocks", "dropout"}:
        raise ValueError("architecture must contain only blocks and dropout")
    blocks = architecture["blocks"]
    if not isinstance(blocks, list) or not MIN_BLOCKS <= len(blocks) <= MAX_BLOCKS:
        raise ValueError(f"architecture must contain {MIN_BLOCKS} to {MAX_BLOCKS} blocks")
    if architecture["dropout"] not in DROPOUT_CHOICES:
        raise ValueError(f"dropout must be one of {DROPOUT_CHOICES}")

    required = {"filters", "kernel_size", "pooling", "batch_norm"}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or set(block) != required:
            raise ValueError(f"block {index} must contain exactly {sorted(required)}")
        if block["filters"] not in FILTER_CHOICES:
            raise ValueError(f"block {index} filters must be one of {FILTER_CHOICES}")
        if block["kernel_size"] not in KERNEL_CHOICES:
            raise ValueError(f"block {index} kernel_size must be one of {KERNEL_CHOICES}")
        if block["pooling"] not in POOLING_CHOICES:
            raise ValueError(f"block {index} pooling must be one of {POOLING_CHOICES}")
        if type(block["batch_norm"]) is not bool:
            raise ValueError(f"block {index} batch_norm must be a boolean")


class ArchitectureCNN(nn.Module):
    def __init__(self, architecture: dict, *, validate: bool = True) -> None:
        super().__init__()
        if validate:
            validate_architecture(architecture)
        self.architecture = deepcopy(architecture)

        layers = []
        in_channels = 3
        for block in architecture["blocks"]:
            out_channels = block["filters"]
            kernel_size = block["kernel_size"]
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if block["batch_norm"]:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            pool = nn.MaxPool2d if block["pooling"] == "max" else nn.AvgPool2d
            layers.append(pool(2))
            in_channels = out_channels

        spatial_size = 64 // (2 ** len(architecture["blocks"]))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(architecture["dropout"]),
            nn.Linear(in_channels * spatial_size * spatial_size, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class BaselineCNN(ArchitectureCNN):
    def __init__(self, filters: int = 32, dropout: float = 0.3) -> None:
        if filters <= 0:
            raise ValueError("filters must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        architecture = deepcopy(BASELINE_ARCHITECTURE)
        architecture["dropout"] = dropout
        for index, block in enumerate(architecture["blocks"]):
            block["filters"] = filters * (2**index)
        super().__init__(architecture, validate=False)


def build_model(architecture: dict) -> ArchitectureCNN:
    return ArchitectureCNN(architecture)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
