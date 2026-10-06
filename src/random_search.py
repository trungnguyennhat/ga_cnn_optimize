"""Uniform random architecture search with the same ResNet fitness as GA-NAS."""

import argparse
import json
import random
from pathlib import Path

from src.nsga2 import Individual, crowding_distance, nondominated_sort
from src.search_runtime import (
    SEARCH_EPOCHS,
    SearchEvaluator,
    save_json,
    serialize_evaluated,
    serialize_individual,
)
from src.search_space import canonical_architecture, random_architecture
from src.train import PROJECT_ROOT


def run_random_search(fitness_callback, *, evaluation_budget: int, seed: int):
    if type(evaluation_budget) is not int or evaluation_budget <= 0:
        raise ValueError("evaluation_budget must be a positive integer")
    if evaluation_budget > (3**4) * (3**4) * (2**4) * 5:
        raise ValueError("evaluation budget exceeds the number of unique architectures")
    rng = random.Random(seed)
    evaluated = {}
    while len(evaluated) < evaluation_budget:
        architecture = random_architecture(rng)
        key = canonical_architecture(architecture)
        if key in evaluated:
            continue
        auc, parameters = fitness_callback(architecture, seed)
        evaluated[key] = Individual(architecture, (float(auc), parameters))
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
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "random_search",
    )
    args = parser.parse_args()
    if args.budget <= 0 or args.epochs <= 0:
        parser.error("budget and epochs must be positive")

    evaluator = SearchEvaluator(
        args, experiment="random_search", label="Random Search",
        search_config={"budget": args.budget, "sampling": "uniform_without_replacement"},
    )
    result = run_random_search(
        evaluator, evaluation_budget=args.budget, seed=args.seed,
    )
    assert len(result["evaluated"]) == len(evaluator.records) == evaluator.trained + evaluator.hits == args.budget
    summary = {
        "experiment": "random_search", "config": evaluator.run_config,
        "evaluations": len(result["evaluated"]),
        "new_trainings": evaluator.trained, "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "pareto_front": [serialize_individual(item) for item in result["pareto_front"]],
        "evaluated": serialize_evaluated(result["evaluated"], evaluator.records),
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
