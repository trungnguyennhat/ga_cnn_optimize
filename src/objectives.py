"""Exact architecture costs, normalization and dominated hypervolume."""

import math
from functools import lru_cache
from itertools import product

import torch

from src.model import build_model, count_macs, count_parameters
from src.nsga2 import dominates
from src.search_space import BLOCK_TYPES, KERNEL_CHOICES, STAGE_BLOCK_CHOICES, STAGE_CHANNEL_CHOICES


AUC = "val_macro_auc_ovr"
PARAMS = "parameter_count"
MACS = "macs_per_image"
OBJECTIVES = {
    "params": (AUC, PARAMS),
    "macs": (AUC, MACS),
    "params-macs": (AUC, PARAMS, MACS),
}


def architecture_costs(architecture: dict, num_classes: int) -> dict[str, int]:
    with torch.device("meta"):
        model = build_model(architecture, num_classes)
    return {PARAMS: count_parameters(model), MACS: count_macs(model)}


@lru_cache(maxsize=None)
def cost_bounds(space: str, num_classes: int) -> dict[str, tuple[int, int]]:
    extrema = []
    for index, reducer in ((0, min), (-1, max)):
        arch = {
            "stage_blocks": [STAGE_BLOCK_CHOICES[index]] * 4,
            "stage_channels": [choices[index] for choices in STAGE_CHANNEL_CHOICES],
            "kernel_sizes": [KERNEL_CHOICES[index]] * 4,
            "dropout": 0.0,
        }
        if space == "v2":
            candidates = [
                {**arch, "block_types": list(blocks), "use_se": [index == -1] * 4,
                 "activation": "relu"}
                for blocks in product(BLOCK_TYPES, repeat=4)
            ]
        else:
            candidates = [arch]
        costs = [architecture_costs(candidate, num_classes) for candidate in candidates]
        extrema.append({key: reducer(row[key] for row in costs) for key in (PARAMS, MACS)})
    return {key: (extrema[0][key], extrema[1][key]) for key in (PARAMS, MACS)}


def normalize(fitness, objectives, bounds, *, log_scale=True):
    point = [float(fitness[0])]
    for value, key in zip(fitness[1:], objectives[1:]):
        low, high = bounds[key]
        if log_scale:
            low, high, value = map(math.log10, (low, high, value))
        point.append(min(1.0, max(0.0, (high - value) / (high - low))))
    return tuple(point)


def front_of(rows, objectives):
    fitnesses = [tuple(row[key] for key in objectives) for row in rows]
    return [
        row for row, fitness in zip(rows, fitnesses)
        if not any(dominates(other, fitness) for other in fitnesses)
    ]


def coverage(first, second, objectives):
    if not second:
        return None
    left = [tuple(row[key] for key in objectives) for row in first]
    return sum(
        any(dominates(a, tuple(row[key] for key in objectives)) for a in left)
        for row in second
    ) / len(second)


def hypervolume(points):
    """Union of axis-aligned boxes from the origin to normalized maximization points."""
    points = sorted(set(tuple(point) for point in points))
    if not points:
        return 0.0
    if len(points[0]) == 1:
        return points[-1][0]
    volume = previous = 0.0
    for index, point in enumerate(points):
        if point[0] > previous:
            volume += (point[0] - previous) * hypervolume([row[1:] for row in points[index:]])
            previous = point[0]
    return volume
