"""ResNet-18 baseline and ResNet architecture builder for DermaMNIST."""

from copy import deepcopy

from torch import nn

from src.search_space import validate_architecture


BASELINE_ARCHITECTURE = {
    "stage_blocks": [2, 2, 2, 2],
    "stage_channels": [64, 128, 256, 512],
    "kernel_sizes": [3, 3, 3, 3],
    "dropout": 0.0,
}


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.relu(self.residual(inputs) + self.shortcut(inputs))


class ArchitectureResNet(nn.Module):
    def __init__(self, architecture: dict) -> None:
        super().__init__()
        validate_architecture(architecture)
        self.architecture = deepcopy(architecture)
        first_channels = architecture["stage_channels"][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_channels = first_channels
        for index, (block_count, out_channels, kernel_size) in enumerate(
            zip(
                architecture["stage_blocks"],
                architecture["stage_channels"],
                architecture["kernel_sizes"],
            )
        ):
            blocks = [BasicBlock(in_channels, out_channels, kernel_size, 1 if index == 0 else 2)]
            blocks.extend(BasicBlock(out_channels, out_channels, kernel_size, 1) for _ in range(block_count - 1))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(architecture["dropout"])
        self.classifier = nn.Linear(in_channels, 7)

    def forward(self, images):
        features = self.stages(self.stem(images))
        features = self.pool(features).flatten(1)
        return self.classifier(self.dropout(features))


class BaselineResNet(ArchitectureResNet):
    def __init__(self) -> None:
        super().__init__(deepcopy(BASELINE_ARCHITECTURE))


def build_model(architecture: dict) -> ArchitectureResNet:
    return ArchitectureResNet(architecture)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
