"""CNN baseline and architecture builder for DermaMNIST GA-NAS."""

from copy import deepcopy

from torch import nn
from src.search_space import validate_architecture

BASELINE_ARCHITECTURE = {
    "blocks": [
        {"filters": 32, "kernel_size": 3, "pooling": "max", "batch_norm": False},
        {"filters": 64, "kernel_size": 3, "pooling": "max", "batch_norm": False},
        {"filters": 128, "kernel_size": 3, "pooling": "max", "batch_norm": False},
    ],
    "dropout": 0.3,
}


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
