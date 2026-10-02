import argparse
import json
from pathlib import Path

from src.nsga2 import run_nsga2
from src.search_runtime import SearchEvaluator, save_json, serialize_evaluated, serialize_individual
from src.train import DEFAULT_EPOCHS, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "ga_search")
    args = parser.parse_args()
    if args.budget <= 0 or args.population_size <= 0 or args.epochs <= 0:
        parser.error("budget, population-size and epochs must be positive")
    search_config = {
        "budget": args.budget, "population_size": args.population_size,
        "tournament_size": 3, "crossover_rate": 0.8,
        "mutation_probability": 0.15,
    }
    evaluator = SearchEvaluator(
        args, experiment="ga_search", label="GA-NAS", search_config=search_config,
    )
    result = run_nsga2(evaluator, seed=args.seed, evaluation_budget=args.budget,
                       population_size=args.population_size)
    assert result["evaluations"] == len(evaluator.records) == evaluator.trained + evaluator.hits == args.budget

    summary = {
        "experiment": "ga_search", "config": evaluator.run_config,
        "evaluations": result["evaluations"], "generations": result["generations"],
        "new_trainings": evaluator.trained, "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "pareto_front": [serialize_individual(item) for item in result["pareto_front"]],
        "evaluated": serialize_evaluated(result["evaluated"], evaluator.records),
    }
    save_json(evaluator.output / "summary.json", summary)
    print(json.dumps({"evaluations": result["evaluations"],
                      "new_trainings": evaluator.trained,
                      "cache_hits": evaluator.hits,
                      "pareto_front": summary["pareto_front"],
                      "result": str(evaluator.output / "summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
