"""Architecture genes and operators; standard library only, no CNN or dataset."""

import json
from copy import deepcopy


FILTER_CHOICES = (16, 32, 64, 128)
KERNEL_CHOICES = (3, 5)
POOLING_CHOICES = ("max", "avg")
DROPOUT_CHOICES = (0.1, 0.2, 0.3, 0.4, 0.5)
MIN_BLOCKS = 2
MAX_BLOCKS = 4
GENES = {
    "filters": FILTER_CHOICES,
    "kernel_size": KERNEL_CHOICES,
    "pooling": POOLING_CHOICES,
    "batch_norm": (False, True),
}


def valid_gene(value, choices) -> bool:
    return type(value) is type(choices[0]) and value in choices


def validate_architecture(architecture: dict) -> None:
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be a dictionary")
    if set(architecture) != {"blocks", "dropout"}:
        raise ValueError("architecture must contain only blocks and dropout")
    blocks = architecture["blocks"]
    if not isinstance(blocks, list) or not MIN_BLOCKS <= len(blocks) <= MAX_BLOCKS:
        raise ValueError(f"architecture must contain {MIN_BLOCKS} to {MAX_BLOCKS} blocks")
    if not valid_gene(architecture["dropout"], DROPOUT_CHOICES):
        raise ValueError(f"dropout must be one of {DROPOUT_CHOICES}")
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or set(block) != set(GENES):
            raise ValueError(f"block {index} must contain exactly {sorted(GENES)}")
        for name, choices in GENES.items():
            if not valid_gene(block[name], choices):
                raise ValueError(f"block {index} {name} must be one of {choices}")


def repair_architecture(architecture: dict) -> dict:
    """Drop extra keys, truncate/pad blocks, replace invalid genes with first choices."""
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be a dictionary")
    blocks = architecture.get("blocks", [])
    blocks = blocks[:MAX_BLOCKS] if isinstance(blocks, list) else []
    blocks = blocks + [{} for _ in range(max(0, MIN_BLOCKS - len(blocks)))]
    repaired = []
    for block in blocks:
        block = block if isinstance(block, dict) else {}
        repaired.append({
            name: block.get(name) if valid_gene(block.get(name), choices) else choices[0]
            for name, choices in GENES.items()
        })
    dropout = architecture.get("dropout")
    result = {
        "blocks": repaired,
        "dropout": dropout if valid_gene(dropout, DROPOUT_CHOICES) else DROPOUT_CHOICES[0],
    }
    validate_architecture(result)
    return result


def canonical_architecture(architecture: dict) -> str:
    """Stable key; preserve block order because it changes the network."""
    validate_architecture(architecture)
    return json.dumps(architecture, sort_keys=True, separators=(",", ":"))


def random_block(rng) -> dict:
    return {name: rng.choice(choices) for name, choices in GENES.items()}


def random_architecture(rng) -> dict:
    return {
        "blocks": [random_block(rng) for _ in range(rng.randint(MIN_BLOCKS, MAX_BLOCKS))],
        "dropout": rng.choice(DROPOUT_CHOICES),
    }


def uniform_crossover(first: dict, second: dict, rng) -> dict:
    """Choose length from either parent and each shared block as a whole."""
    validate_architecture(first)
    validate_architecture(second)
    length = rng.choice((len(first["blocks"]), len(second["blocks"])))
    blocks = []
    for index in range(length):
        available = [parent["blocks"][index] for parent in (first, second)
                     if index < len(parent["blocks"])]
        blocks.append(deepcopy(rng.choice(available)))
    return {"blocks": blocks, "dropout": rng.choice((first["dropout"], second["dropout"]))}


def mutate_architecture(architecture: dict, rng, probability: float = 0.15) -> dict:
    """Mutate each gene and, independently, try one block insertion/deletion."""
    validate_architecture(architecture)
    if not 0 <= probability <= 1:
        raise ValueError("mutation probability must be in [0, 1]")
    result = deepcopy(architecture)
    for block in result["blocks"]:
        for name, choices in GENES.items():
            if rng.random() < probability:
                block[name] = rng.choice([value for value in choices if value != block[name]])
    if rng.random() < probability:
        result["dropout"] = rng.choice([
            value for value in DROPOUT_CHOICES if value != result["dropout"]
        ])
    if rng.random() < probability:
        length = len(result["blocks"])
        actions = (["add"] if length < MAX_BLOCKS else []) + (
            ["remove"] if length > MIN_BLOCKS else []
        )
        if rng.choice(actions) == "add":
            result["blocks"].insert(rng.randrange(length + 1), random_block(rng))
        else:
            del result["blocks"][rng.randrange(length)]
    validate_architecture(result)
    return result
