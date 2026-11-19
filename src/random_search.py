"""Uniform random architecture search with the same ResNet fitness as GA-NAS."""

import argparse
import json
import random
from pathlib import Path

from src.nsga2 import Individual, crowding_distance, nondominated_sort, validate_fitness
from src.search_runtime import (
    SEARCH_EPOCHS,
    SearchEvaluator,
    add_search_args,
    save_json,
    search_root,
)
from src.search_space import canonical_architecture, random_architecture, space_size


def run_random_search(fitness_callback, *, evaluation_budget: int, seed: int, space: str = "v1"):
    if type(evaluation_budget) is not int or evaluation_budget <= 0:
        raise ValueError("evaluation_budget must be a positive integer")
    if evaluation_budget > space_size(space):
        raise ValueError("evaluation budget exceeds the number of unique architectures")
    rng = random.Random(seed)
    evaluated = {}
    while len(evaluated) < evaluation_budget:
        architecture = random_architecture(rng, space)
        key = canonical_architecture(architecture)
        if key in evaluated:
            continue
        fitness = validate_fitness(fitness_callback(architecture, seed))
        evaluated[key] = Individual(architecture, fitness)
    archive = list(evaluated.values())
    fronts = nondominated_sort(archive)
    for front in fronts:
        crowding_distance(front)
    return {"evaluated": archive, "pareto_front": fronts[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_search_args(parser)
    args = parser.parse_args()
    if args.budget <= 0 or args.epochs <= 0:
        parser.error("budget and epochs must be positive")
    if args.output_dir is None:
        args.output_dir = search_root(args) / "random_search"

    evaluator = SearchEvaluator(
        args, experiment="random_search", label="Random Search",
        search_config={"budget": args.budget, "sampling": "uniform_without_replacement"},
    )
    result = run_random_search(
        evaluator, evaluation_budget=args.budget, seed=args.seed, space=args.space,
    )
    assert len(result["evaluated"]) == len(evaluator.records) == evaluator.trained + evaluator.hits == args.budget
    summary = {
        "experiment": "random_search", "config": evaluator.run_config,
        "evaluations": len(result["evaluated"]),
        "new_trainings": evaluator.trained, "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "training_seconds": evaluator.training_seconds,
        "pareto_front": [evaluator.serialize(item) for item in result["pareto_front"]],
        "evaluated": evaluator.serialize_evaluated(result["evaluated"]),
    }
    save_json(evaluator.output / "summary.json", summary)
    print(json.dumps({
        "evaluations": len(result["evaluated"]),
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "pareto_front": summary["pareto_front"],
        "result": str(evaluator.output / "summary.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
