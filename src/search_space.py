"""ResNet architecture genes and genetic operators; no training or data access."""

import json
from copy import deepcopy


STAGE_BLOCK_CHOICES = (1, 2, 3)
STAGE_CHANNEL_CHOICES = (
    (16, 32, 64),
    (32, 64, 128),
    (64, 128, 256),
    (128, 256, 512),
)
KERNEL_CHOICES = (3, 5)
DROPOUT_CHOICES = (0.0, 0.1, 0.2, 0.3, 0.5)
GENES = {"stage_blocks", "stage_channels", "kernel_sizes", "dropout"}


def _valid_choice(value, choices) -> bool:
    return type(value) is type(choices[0]) and value in choices


def validate_architecture(architecture: dict) -> None:
    if not isinstance(architecture, dict) or set(architecture) != GENES:
        raise ValueError(f"architecture must contain exactly {sorted(GENES)}")

    stage_blocks = architecture["stage_blocks"]
    if not isinstance(stage_blocks, list) or len(stage_blocks) != 4 or any(
        not _valid_choice(value, STAGE_BLOCK_CHOICES) for value in stage_blocks
    ):
        raise ValueError(f"stage_blocks must contain four values from {STAGE_BLOCK_CHOICES}")

    stage_channels = architecture["stage_channels"]
    if not isinstance(stage_channels, list) or len(stage_channels) != 4 or any(
        not _valid_choice(value, STAGE_CHANNEL_CHOICES[index])
        for index, value in enumerate(stage_channels)
    ):
        raise ValueError("stage_channels contains a value outside its stage choices")

    kernel_sizes = architecture["kernel_sizes"]
    if not isinstance(kernel_sizes, list) or len(kernel_sizes) != 4 or any(
        not _valid_choice(value, KERNEL_CHOICES) for value in kernel_sizes
    ):
        raise ValueError(f"kernel_sizes must contain four values from {KERNEL_CHOICES}")

    if not _valid_choice(architecture["dropout"], DROPOUT_CHOICES):
        raise ValueError(f"dropout must be one of {DROPOUT_CHOICES}")


def _repair_list(value, choices_by_position) -> list:
    values = value if isinstance(value, list) else []
    return [
        values[index]
        if index < len(values) and _valid_choice(values[index], choices)
        else choices[0]
        for index, choices in enumerate(choices_by_position)
    ]


def repair_architecture(architecture: dict) -> dict:
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be a dictionary")
    dropout = architecture.get("dropout")
    repaired = {
        "stage_blocks": _repair_list(
            architecture.get("stage_blocks"), (STAGE_BLOCK_CHOICES,) * 4
        ),
        "stage_channels": _repair_list(
            architecture.get("stage_channels"), STAGE_CHANNEL_CHOICES
        ),
        "kernel_sizes": _repair_list(
            architecture.get("kernel_sizes"), (KERNEL_CHOICES,) * 4
        ),
        "dropout": dropout if _valid_choice(dropout, DROPOUT_CHOICES) else DROPOUT_CHOICES[0],
    }
    validate_architecture(repaired)
    return repaired


def canonical_architecture(architecture: dict) -> str:
    validate_architecture(architecture)
    return json.dumps(architecture, sort_keys=True, separators=(",", ":"))


def random_architecture(rng) -> dict:
    return {
        "stage_blocks": [rng.choice(STAGE_BLOCK_CHOICES) for _ in range(4)],
        "stage_channels": [rng.choice(choices) for choices in STAGE_CHANNEL_CHOICES],
        "kernel_sizes": [rng.choice(KERNEL_CHOICES) for _ in range(4)],
        "dropout": rng.choice(DROPOUT_CHOICES),
    }


def uniform_crossover(first: dict, second: dict, rng) -> dict:
    validate_architecture(first)
    validate_architecture(second)
    child = {
        name: [rng.choice((first[name][index], second[name][index])) for index in range(4)]
        for name in ("stage_blocks", "stage_channels", "kernel_sizes")
    }
    child["dropout"] = rng.choice((first["dropout"], second["dropout"]))
    return child


def mutate_architecture(architecture: dict, rng, probability: float = 0.15) -> dict:
    validate_architecture(architecture)
    if not 0 <= probability <= 1:
        raise ValueError("mutation probability must be in [0, 1]")
    result = deepcopy(architecture)
    choices_by_gene = {
        "stage_blocks": (STAGE_BLOCK_CHOICES,) * 4,
        "stage_channels": STAGE_CHANNEL_CHOICES,
        "kernel_sizes": (KERNEL_CHOICES,) * 4,
    }
    for name, choices_by_position in choices_by_gene.items():
        for index, choices in enumerate(choices_by_position):
            if rng.random() < probability:
                current = result[name][index]
                result[name][index] = rng.choice([value for value in choices if value != current])
    if rng.random() < probability:
        result["dropout"] = rng.choice(
            [value for value in DROPOUT_CHOICES if value != result["dropout"]]
        )
    validate_architecture(result)
    return result
