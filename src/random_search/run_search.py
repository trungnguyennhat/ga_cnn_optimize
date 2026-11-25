"""Uniform random architecture search with the same ResNet fitness as GA-NAS."""

import argparse
import json
import random
from pathlib import Path

from src.ga_search.nsga2 import DEFAULT_DIRECTIONS, Individual, crowding_distance, nondominated_sort
from src.search_runtime import (
    HARDWARE_RESULTS,
    SEARCH_EPOCHS,
    SearchEvaluator,
    save_json,
    serialize_evaluated,
    serialize_individual,
)
from src.search_space import canonical_architecture, random_architecture
from src.train import PROJECT_ROOT


def run_random_search(fitness_callback, *, evaluation_budget: int, seed: int,
                      directions=DEFAULT_DIRECTIONS):
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
        auc, *costs = fitness_callback(architecture, seed)
        if len(costs) + 1 != len(directions):
            raise ValueError("fitness lengths must match the objective directions")
        evaluated[key] = Individual(architecture, (float(auc), *costs))
    archive = list(evaluated.values())
    fronts = nondominated_sort(archive, directions)
    for front in fronts:
        crowding_distance(front)
    return {"evaluated": archive, "pareto_front": fronts[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hardware-aware", action="store_true", help="Add MACs as a third objective")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.budget <= 0 or args.epochs <= 0:
        parser.error("budget and epochs must be positive")
    if args.output_dir is None:
        root = PROJECT_ROOT / "results"
        args.output_dir = (root / HARDWARE_RESULTS if args.hardware_aware else root) / "random_search"

    evaluator = SearchEvaluator(
        args, experiment="random_search", label="Random Search",
        search_config={"budget": args.budget, "sampling": "uniform_without_replacement"},
        hardware_aware=args.hardware_aware,
    )
    result = run_random_search(
        evaluator, evaluation_budget=args.budget, seed=args.seed,
        directions=evaluator.directions,
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
