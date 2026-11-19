"""Run NSGA-II architecture search with measured ResNet validation fitness."""

import argparse
import json
from pathlib import Path

from src.nsga2 import run_nsga2
from src.search_runtime import (
    SEARCH_EPOCHS,
    SearchEvaluator,
    add_search_args,
    save_json,
    search_root,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_search_args(parser)
    args = parser.parse_args()
    if min(args.budget, args.population_size, args.tournament_size, args.epochs) <= 0:
        parser.error("budget, population-size, tournament-size and epochs must be positive")
    if args.output_dir is None:
        args.output_dir = search_root(args) / "ga_search"
    search_config = {
        "budget": args.budget, "population_size": args.population_size,
        "tournament_size": args.tournament_size, "crossover_rate": 0.8,
        "mutation_probability": 0.15,
    }
    evaluator = SearchEvaluator(
        args, experiment="ga_search", label="GA-NAS", search_config=search_config,
    )
    result = run_nsga2(
        evaluator,
        seed=args.seed,
        evaluation_budget=args.budget,
        population_size=args.population_size,
        tournament_size=search_config["tournament_size"],
        crossover_rate=search_config["crossover_rate"],
        mutation_probability=search_config["mutation_probability"],
        space=args.space,
    )
    assert result["evaluations"] == len(evaluator.records) == evaluator.trained + evaluator.hits == args.budget

    summary = {
        "experiment": "ga_search", "config": evaluator.run_config,
        "evaluations": result["evaluations"], "generations": result["generations"],
        "new_trainings": evaluator.trained, "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "training_seconds": evaluator.training_seconds,
        "pareto_front": [evaluator.serialize(item) for item in result["pareto_front"]],
        "evaluated": evaluator.serialize_evaluated(result["evaluated"]),
    }
    save_json(evaluator.output / "summary.json", summary)
    print(json.dumps({"evaluations": result["evaluations"],
                      "new_trainings": evaluator.trained,
                      "cache_hits": evaluator.hits,
                      "pareto_front": summary["pareto_front"],
                      "result": str(evaluator.output / "summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
