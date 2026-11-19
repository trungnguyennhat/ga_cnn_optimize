"""Shared ResNet evaluation, cache, and logging for architecture searches."""

import hashlib
import json
import math
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.datasets import DATASETS, add_dataset_args, dataset_path, results_root
from src.model import build_model, count_parameters
from src.nsga2 import dominates
from src.objectives import OBJECTIVES, PARAMS, architecture_costs
from src.search_space import canonical_architecture
from src.train import (
    AUGMENTATION_CHOICES,
    LOSS_CHOICES,
    SCHEDULER_CHOICES,
    TRAINING_VERSION,
    is_legacy_protocol,
    load_data,
    loss_config_name,
    preload_data,
    resolve_device,
    resolve_protocol_settings,
    scheduler_config_dict,
    set_seed,
    split_digest,
    train_model,
)


SEARCH_EPOCHS = 20
SEARCH_SCHEDULER_STEP = 15


def add_protocol_args(parser) -> None:
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default=None)
    parser.add_argument("--augmentation", choices=AUGMENTATION_CHOICES, default=None)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default=None)
    parser.add_argument("--preload-gpu", action="store_true", default=None)
    parser.add_argument("--no-preload-gpu", dest="preload_gpu", action="store_false")
    parser.add_argument("--amp", action="store_true", default=None)


def add_search_args(parser) -> None:
    add_protocol_args(parser)
    add_dataset_args(parser)
    parser.add_argument("--space", choices=("v1", "v2"), default="v1")
    parser.add_argument("--objectives", choices=tuple(OBJECTIVES), default="params")


def search_root(args):
    variant = f"space_{args.space}_{args.objectives}" if (args.space, args.objectives) != ("v1", "params") else ""
    return results_root(args.protocol, args.dataset, variant)


def save_json(path, value):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SearchEvaluator:
    def __init__(self, args, *, experiment: str, label: str, search_config: dict) -> None:
        self.args = args
        self.experiment = experiment
        self.label = label
        self.started = time.perf_counter()
        self.device = resolve_device(args.device)
        self.dataset = getattr(args, "dataset", "dermamnist")
        self.space = getattr(args, "space", "v1")
        self.objectives = OBJECTIVES[getattr(args, "objectives", "params")]
        self.num_classes = DATASETS[self.dataset].num_classes
        self.protocol_settings = resolve_protocol_settings(
            getattr(args, "protocol", "v1"),
            optimizer_name=getattr(args, "optimizer", None),
            weight_decay=getattr(args, "weight_decay", None),
            scheduler_name=getattr(args, "scheduler", None),
            augmentation=getattr(args, "augmentation", None),
            loss_name=getattr(args, "loss", None),
            preload_gpu=getattr(args, "preload_gpu", None),
            use_amp=getattr(args, "amp", None),
        )
        self.loaded_data = load_data(dataset_path(self.dataset, getattr(args, "data_path", None)), self.dataset)
        digest = split_digest(self.loaded_data)

        settings = self.protocol_settings
        is_legacy_v1 = (
            is_legacy_protocol(settings)
            and self.dataset == "dermamnist"
            and self.space == "v1"
            and self.objectives == OBJECTIVES["params"]
            and "fidelity" not in search_config
        )
        self.config = {
            "epochs": args.epochs,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": settings["optimizer_name"],
            "seed": args.seed,
            "device": str(self.device),
            "scheduler": scheduler_config_dict(
                settings["scheduler_name"],
                epochs=args.epochs,
                step_size=SEARCH_SCHEDULER_STEP,
            ),
            "loss": loss_config_name(settings["loss_name"]),
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "architecture_space": f"resnet_stage_channels_{self.space}",
            "train_val_sha256": digest,
            "training_version": 3 if is_legacy_v1 else TRAINING_VERSION,
        }
        if not is_legacy_v1:
            self.config.update({
                "protocol": settings["protocol"],
                "weight_decay": settings["weight_decay"],
                "augmentation": settings["augmentation"],
                "amp": settings["use_amp"],
                "preload_gpu": settings["preload_gpu"],
                "dataset": self.dataset,
                "num_classes": self.num_classes,
                "objectives": list(self.objectives),
            })
        self.run_config = {
            **self.config, **search_config, "objectives": list(self.objectives),
            "objective_directions": ["max", *(["min"] * (len(self.objectives) - 1))],
        }
        if settings["preload_gpu"] and self.device.type == "cuda":
            self.loaded_data = preload_data(self.loaded_data, self.device)
        self.output = args.output_dir / f"seed_{args.seed}"
        self.cache = self.output / "evaluations"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.trained = 0
        self.hits = 0
        self.records = []
        self.fitnesses = []
        self.costs_by_key = {}
        self.training_seconds = 0.0
        print(
            f"{label} protocol={settings['protocol']} seed={args.seed} budget={args.budget} "
            f"epochs={args.epochs} device={self.device}", flush=True,
        )

    def architecture_costs(self, architecture):
        key = canonical_architecture(architecture)
        if key not in self.costs_by_key:
            self.costs_by_key[key] = architecture_costs(architecture, self.num_classes)
        return self.costs_by_key[key]

    def __call__(self, architecture, seed, *, epoch_callback=None, fidelity_context=None):
        if seed != self.args.seed:
            raise ValueError("Evaluation seed must match the search configuration")
        key = canonical_architecture(architecture)
        config = self.config if fidelity_context is None else {**self.config, "fidelity": fidelity_context}
        cache_key = hashlib.sha256(
            (key + json.dumps(config, sort_keys=True)).encode()
        ).hexdigest()
        path = self.cache / f"{cache_key}.json"
        costs = self.architecture_costs(architecture)
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["config"] != config or canonical_architecture(record["architecture"]) != key:
                raise ValueError(f"Cache does not match evaluation: {path}")
            self.hits += 1
            status = "cache"
        else:
            print(
                f"evaluation={len(self.records)+1}/{self.args.budget} "
                f"training architecture={key}", flush=True,
            )
            set_seed(seed)
            model = build_model(architecture, self.num_classes)
            if count_parameters(model) != costs[PARAMS]:
                raise ValueError("Measured architecture parameter count differs from its metadata")
            settings = self.protocol_settings
            metrics, history, runtime = train_model(
                model,
                self.loaded_data,
                learning_rate=0.001,
                batch_size=64,
                optimizer_name=settings["optimizer_name"],
                epochs=self.args.epochs,
                seed=seed,
                device=self.device,
                scheduler_step_size=SEARCH_SCHEDULER_STEP,
                weight_decay=settings["weight_decay"],
                scheduler_name=settings["scheduler_name"],
                augmentation=settings["augmentation"],
                loss_name=settings["loss_name"],
                preload_gpu=settings["preload_gpu"],
                use_amp=settings["use_amp"],
                epoch_callback=epoch_callback,
            )
            record = {
                "architecture": architecture, "config": config, "metrics": metrics,
                **costs, "history": history,
                "runtime_seconds": runtime,
            }
            save_json(path, record)
            self.trained += 1
            status = "trained"

        auc = record["metrics"]["val_macro_auc_ovr"]
        if (not math.isfinite(auc) or not 0 <= auc <= 1
            or any(record.get(key, actual) != actual for key, actual in costs.items())):
            raise ValueError(f"Invalid measured fitness: {path}")
        fitness = (float(auc), *(costs[key] for key in self.objectives[1:]))
        epochs_trained = len(record["history"])
        full = epochs_trained == self.args.epochs
        if full:
            self.fitnesses.append(fitness)
        self.training_seconds += record["runtime_seconds"]
        pareto_size = sum(
            not any(dominates(other, candidate) for other in self.fitnesses)
            for candidate in self.fitnesses
        )
        self.records.append({
            "evaluation": len(self.records) + 1,
            "status": status,
            "result": str(path),
            **costs,
            "best_auc_so_far": max((item[0] for item in self.fitnesses), default=None),
            "smallest_parameters_so_far": min(
                [row[PARAMS] for row in self.records if row["full_fidelity"]]
                + ([costs[PARAMS]] if full else []), default=None,
            ),
            "pareto_size_so_far": pareto_size,
            "epochs_trained": epochs_trained,
            "full_fidelity": full,
            "runtime_seconds": record["runtime_seconds"],
            "training_seconds_so_far": self.training_seconds,
            "elapsed_seconds": self.runtime_seconds,
        })
        self.last_history = record["history"]
        save_json(self.output / "progress.json", {
            "experiment": self.experiment,
            "config": self.run_config,
            "new_trainings": self.trained,
            "cache_hits": self.hits,
            "evaluations": self.records,
        })
        print(
            f"evaluation={len(self.records)}/{self.args.budget} {status} "
            f"val_auc={auc:.6f} parameters={costs[PARAMS]} epochs={epochs_trained}", flush=True,
        )
        return fitness

    def serialize(self, individual):
        row = serialize_individual(individual, self.objectives)
        return {**row, **self.architecture_costs(individual.architecture)}

    def serialize_evaluated(self, individuals):
        return [{**record, **self.serialize(item)} for record, item in zip(self.records, individuals, strict=True)]

    @property
    def runtime_seconds(self):
        return time.perf_counter() - self.started


def serialize_individual(individual, objectives=OBJECTIVES["params"]):
    return {
        "architecture": individual.architecture,
        **dict(zip(objectives, individual.fitness, strict=True)),
        "pareto_rank": individual.rank,
        "crowding_distance": None if math.isinf(individual.crowding) else individual.crowding,
    }


def serialize_evaluated(individuals, records):
    return [
        {**record, **serialize_individual(individual)}
        for record, individual in zip(records, individuals)
    ]
