"""ResNet-18 baseline and ResNet architecture builder for DermaMNIST."""

from copy import deepcopy
from math import prod
import statistics
import time

import torch
from torch import nn

from src.search_space import validate_architecture


BASELINE_ARCHITECTURE = {
    "stage_blocks": [2, 2, 2, 2],
    "stage_channels": [64, 128, 256, 512],
    "kernel_sizes": [3, 3, 3, 3],
    "dropout": 0.0,
}


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
        *, activation=nn.ReLU, use_se: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
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
        self.relu = activation(inplace=True)
        self.se = SqueezeExcitation(out_channels, activation) if use_se else nn.Identity()

    def forward(self, inputs):
        return self.relu(self.se(self.residual(inputs)) + self.shortcut(inputs))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        hidden = max(4, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            activation(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.gate(inputs)


class MBConv(nn.Module):
    """Expansion ratio 2 with a depthwise spatial convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, *, activation, use_se):
        super().__init__()
        expanded = 2 * in_channels
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded), activation(inplace=True),
            nn.Conv2d(expanded, expanded, kernel_size, stride, kernel_size // 2,
                      groups=expanded, bias=False),
            nn.BatchNorm2d(expanded), activation(inplace=True),
            SqueezeExcitation(expanded, activation) if use_se else nn.Identity(),
            nn.Conv2d(expanded, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.use_skip = stride == 1 and in_channels == out_channels
        self.shortcut = nn.Identity()

    def forward(self, inputs):
        output = self.residual(inputs)
        return output + inputs if self.use_skip else output


class ArchitectureResNet(nn.Module):
    def __init__(self, architecture: dict, num_classes: int = 7) -> None:
        super().__init__()
        validate_architecture(architecture)
        self.architecture = deepcopy(architecture)
        first_channels = architecture["stage_channels"][0]
        activation = {"relu": nn.ReLU, "silu": nn.SiLU}[architecture.get("activation", "relu")]
        block_types = architecture.get("block_types", ["basic"] * 4)
        use_se = architecture.get("use_se", [False] * 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            activation(inplace=True),
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
            block_type = {"basic": BasicBlock, "mbconv": MBConv}[block_types[index]]
            settings = {"activation": activation, "use_se": use_se[index]}
            blocks = [block_type(in_channels, out_channels, kernel_size, 1 if index == 0 else 2, **settings)]
            blocks.extend(
                block_type(out_channels, out_channels, kernel_size, 1, **settings)
                for _ in range(block_count - 1)
            )
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(architecture["dropout"])
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, images):
        features = self.stages(self.stem(images))
        features = self.pool(features).flatten(1)
        return self.classifier(self.dropout(features))


class BaselineResNet(ArchitectureResNet):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__(deepcopy(BASELINE_ARCHITECTURE), num_classes)


def build_model(architecture: dict, num_classes: int = 7) -> ArchitectureResNet:
    return ArchitectureResNet(architecture, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _conv_macs(
    layer: nn.Conv2d, input_size: tuple[int, int]
) -> tuple[int, tuple[int, int]]:
    height, width = (
        (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        for size, padding, dilation, kernel, stride in zip(
            input_size, layer.padding, layer.dilation, layer.kernel_size, layer.stride
        )
    )
    macs = (
        height * width * layer.out_channels * (layer.in_channels // layer.groups)
        * prod(layer.kernel_size)
    )
    return macs, (height, width)


def count_macs(model: ArchitectureResNet, input_size: tuple[int, int] = (64, 64)) -> int:
    """Count convolution and linear multiply-accumulates per image without a forward pass."""
    total, size = _conv_macs(model.stem[0], input_size)
    for stage in model.stages:
        for block in stage:
            block_input_size = size
            for layer in block.residual:
                if isinstance(layer, nn.Conv2d):
                    macs, size = _conv_macs(layer, size)
                    total += macs
                elif isinstance(layer, SqueezeExcitation):
                    total += _se_macs(layer)
            if isinstance(getattr(block, "se", None), SqueezeExcitation):
                total += _se_macs(block.se)
            for layer in block.shortcut.children():
                if isinstance(layer, nn.Conv2d):
                    macs, _ = _conv_macs(layer, block_input_size)
                    total += macs
    return total + model.classifier.in_features * model.classifier.out_features


def _se_macs(module: SqueezeExcitation) -> int:
    return sum(_conv_macs(layer, (1, 1))[0] for layer in module.gate if isinstance(layer, nn.Conv2d))


@torch.inference_mode()
def measure_latency_ms(model, device, *, warmup: int = 20, runs: int = 100) -> float:
    if warmup < 0 or runs <= 0:
        raise ValueError("Latency benchmark requires nonnegative warmup and positive runs")
    images = torch.zeros((1, 3, 64, 64), device=device)
    training = model.training
    model.eval()
    try:
        for _ in range(warmup):
            model(images)
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(1000 * (time.perf_counter() - start))
        return statistics.median(times)
    finally:
        model.train(training)
