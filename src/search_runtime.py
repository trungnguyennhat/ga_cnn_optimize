"""Shared ResNet evaluation, cache, and logging for architecture searches."""

import hashlib
import json
import math
import time

from src.model import build_model, count_parameters
from src.nsga2 import dominates
from src.search_space import canonical_architecture
from src.train import DATA_PATH, load_data, resolve_device, set_seed, train_model


SEARCH_EPOCHS = 20
SEARCH_SCHEDULER_STEP = 15


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class SearchEvaluator:
    def __init__(self, args, *, experiment: str, label: str, search_config: dict) -> None:
        self.args = args
        self.experiment = experiment
        self.label = label
        self.started = time.perf_counter()
        self.device = resolve_device(args.device)
        self.loaded_data = load_data(DATA_PATH)
        digest = hashlib.sha256()
        for dataset in self.loaded_data[:2]:
            for tensor in dataset.tensors:
                digest.update(tensor.numpy().tobytes())
        self.config = {
            "epochs": args.epochs, "learning_rate": 0.001, "batch_size": 64,
            "optimizer": "Adam", "seed": args.seed, "device": str(self.device),
            "scheduler": {
                "name": "StepLR", "step_size": SEARCH_SCHEDULER_STEP, "gamma": 0.1,
            },
            "loss": "unweighted_cross_entropy",
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "architecture_space": "resnet_stage_channels_v1",
            "train_val_sha256": digest.hexdigest(), "training_version": 3,
        }
        self.run_config = {**self.config, **search_config}
        self.output = args.output_dir / f"seed_{args.seed}"
        self.cache = self.output / "evaluations"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.trained = 0
        self.hits = 0
        self.records = []
        self.fitnesses = []
        print(
            f"{label} seed={args.seed} budget={args.budget} "
            f"epochs={args.epochs} device={self.device}", flush=True,
        )

    def __call__(self, architecture, seed):
        key = canonical_architecture(architecture)
        cache_key = hashlib.sha256(
            (key + json.dumps(self.config, sort_keys=True)).encode()
        ).hexdigest()
        path = self.cache / f"{cache_key}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["config"] != self.config or canonical_architecture(record["architecture"]) != key:
                raise ValueError(f"Cache does not match evaluation: {path}")
            self.hits += 1
            status = "cache"
        else:
            print(
                f"evaluation={len(self.records)+1}/{self.args.budget} "
                f"training architecture={key}", flush=True,
            )
            set_seed(seed)
            model = build_model(architecture)
            parameters = count_parameters(model)
            metrics, history, runtime = train_model(
                model, self.loaded_data, learning_rate=0.001, batch_size=64,
                optimizer_name="Adam", epochs=self.args.epochs, seed=seed,
                device=self.device, scheduler_step_size=SEARCH_SCHEDULER_STEP,
            )
            record = {
                "architecture": architecture, "config": self.config, "metrics": metrics,
                "parameter_count": parameters, "history": history,
                "runtime_seconds": runtime,
            }
            save_json(path, record)
            self.trained += 1
            status = "trained"

        auc = record["metrics"]["val_macro_auc_ovr"]
        parameters = record["parameter_count"]
        if not math.isfinite(auc) or not 0 <= auc <= 1 or type(parameters) is not int or parameters <= 0:
            raise ValueError(f"Invalid measured fitness: {path}")
        self.fitnesses.append((float(auc), parameters))
        pareto_size = sum(
            not any(dominates(other, candidate) for other in self.fitnesses)
            for candidate in self.fitnesses
        )
        self.records.append({
            "evaluation": len(self.records) + 1,
            "status": status,
            "result": str(path),
            "best_auc_so_far": max(item[0] for item in self.fitnesses),
            "smallest_parameters_so_far": min(item[1] for item in self.fitnesses),
            "pareto_size_so_far": pareto_size,
        })
        save_json(self.output / "progress.json", {
            "experiment": self.experiment,
            "config": self.run_config,
            "new_trainings": self.trained,
            "cache_hits": self.hits,
            "evaluations": self.records,
        })
        print(
            f"evaluation={len(self.records)}/{self.args.budget} {status} "
            f"val_auc={auc:.6f} parameters={parameters}", flush=True,
        )
        return auc, parameters

    @property
    def runtime_seconds(self):
        return time.perf_counter() - self.started


def serialize_individual(individual):
    return {
        "architecture": individual.architecture,
        "val_macro_auc_ovr": individual.fitness[0],
        "parameter_count": individual.fitness[1],
        "pareto_rank": individual.rank,
        "crowding_distance": None if math.isinf(individual.crowding) else individual.crowding,
    }


def serialize_evaluated(individuals, records):
    return [
        {**record, **serialize_individual(individual)}
        for record, individual in zip(records, individuals)
    ]
