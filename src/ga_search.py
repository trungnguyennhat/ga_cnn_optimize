import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from src.model import build_model, count_parameters
from src.nsga2 import run_nsga2
from src.search_space import canonical_architecture
from src.train import DATA_PATH, DEFAULT_EPOCHS, PROJECT_ROOT, load_data, resolve_device, set_seed, train_model


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "ga_search")
    args = parser.parse_args()
    if args.budget <= 0 or args.population_size <= 0 or args.epochs <= 0:
        parser.error("budget, population-size and epochs must be positive")
    device = resolve_device(args.device)
    started = time.perf_counter()
    loaded_data = load_data(DATA_PATH)
    digest = hashlib.sha256()
    for dataset in loaded_data[:2]:
        for tensor in dataset.tensors:
            digest.update(tensor.numpy().tobytes())
    config = {
        "epochs": args.epochs, "learning_rate": 0.001, "batch_size": 64,
        "optimizer": "Adam", "seed": args.seed, "device": str(device),
        "train_val_sha256": digest.hexdigest(), "training_version": 1,
    }
    output = args.output_dir / f"seed_{args.seed}"
    cache = output / "evaluations"
    cache.mkdir(parents=True, exist_ok=True)
    trained = 0
    hits = 0
    records = []
    print(f"GA-NAS seed={args.seed} budget={args.budget} epochs={args.epochs} device={device}", flush=True)

    def fitness(architecture, seed):
        nonlocal trained, hits
        key = canonical_architecture(architecture)
        cache_key = hashlib.sha256((key + json.dumps(config, sort_keys=True)).encode()).hexdigest()
        path = cache / f"{cache_key}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["config"] != config or canonical_architecture(record["architecture"]) != key:
                raise ValueError(f"Cache does not match evaluation: {path}")
            hits += 1
            status = "cache"
        else:
            print(f"evaluation={len(records)+1}/{args.budget} training architecture={key}", flush=True)
            set_seed(seed)
            model = build_model(architecture)
            parameters = count_parameters(model)
            metrics, history, runtime = train_model(
                model, loaded_data, learning_rate=0.001, batch_size=64,
                optimizer_name="Adam", epochs=args.epochs, seed=seed, device=device,
            )
            record = {
                "architecture": architecture, "config": config, "metrics": metrics,
                "parameter_count": parameters, "history": {"train_loss": history},
                "runtime_seconds": runtime,
            }
            save_json(path, record)
            trained += 1
            status = "trained"
        auc = record["metrics"]["val_macro_auc_ovr"]
        parameters = record["parameter_count"]
        if not math.isfinite(auc) or not 0 <= auc <= 1 or type(parameters) is not int or parameters <= 0:
            raise ValueError(f"Invalid measured fitness: {path}")
        records.append({"evaluation": len(records)+1, "status": status, "result": str(path)})
        save_json(output / "progress.json", {
            "config": config, "new_trainings": trained, "cache_hits": hits, "evaluations": records,
        })
        print(f"evaluation={len(records)}/{args.budget} {status} val_auc={auc:.6f} parameters={parameters}", flush=True)
        return auc, parameters

    result = run_nsga2(fitness, seed=args.seed, evaluation_budget=args.budget,
                       population_size=args.population_size)
    assert result["evaluations"] == len(records) == trained + hits == args.budget

    def serialize(individual):
        return {"architecture": individual.architecture,
                "val_macro_auc_ovr": individual.fitness[0],
                "parameter_count": individual.fitness[1]}

    summary = {
        "experiment": "ga_search", "config": {**config, "budget": args.budget,
        "population_size": args.population_size, "tournament_size": 3,
        "crossover_rate": 0.8, "mutation_probability": 0.15},
        "evaluations": result["evaluations"], "generations": result["generations"],
        "new_trainings": trained, "cache_hits": hits,
        "runtime_seconds": time.perf_counter() - started,
        "pareto_front": [serialize(item) for item in result["pareto_front"]],
        "evaluated": [serialize(item) for item in result["evaluated"]],
    }
    save_json(output / "summary.json", summary)
    print(json.dumps({"evaluations": result["evaluations"], "new_trainings": trained,
                      "cache_hits": hits, "pareto_front": summary["pareto_front"],
                      "result": str(output / "summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
