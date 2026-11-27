"""Run MOEA/D on ResNet to maximize AUC and minimize parameters and MACs."""

import argparse
import json
from pathlib import Path

from src.moead_search.moead import MAX_EVALUATIONS, run_moead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--neighborhood-size", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.budget <= MAX_EVALUATIONS:
        parser.error(f"budget must be between 1 and {MAX_EVALUATIONS}")
    if args.population_size <= 0 or args.neighborhood_size <= 0:
        parser.error("population-size and neighborhood-size must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")

    from src.search_runtime import (
        HARDWARE_RESULTS,
        SEARCH_EPOCHS,
        SearchEvaluator,
        save_json,
        serialize_evaluated,
        serialize_individual,
    )
    from src.train import PROJECT_ROOT

    if args.epochs is None:
        args.epochs = SEARCH_EPOCHS
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / HARDWARE_RESULTS / "moead_search"

    search_config = {
        "budget": args.budget,
        "population_size": args.population_size,
        "neighborhood_size": args.neighborhood_size,
        "crossover_rate": 0.8,
        "mutation_probability": 0.15,
        "decomposition": "normalized_weighted_tchebycheff",
    }
    evaluator = SearchEvaluator(
        args,
        experiment="moead_search",
        label="MOEA/D-NAS",
        search_config=search_config,
        hardware_aware=True,
    )
    result = run_moead(
        evaluator,
        seed=args.seed,
        evaluation_budget=args.budget,
        population_size=args.population_size,
        neighborhood_size=args.neighborhood_size,
        crossover_rate=search_config["crossover_rate"],
        mutation_probability=search_config["mutation_probability"],
    )
    output = evaluator.output / "summary.json"
    summary = {
        "experiment": "moead_search",
        "config": evaluator.run_config,
        "evaluations": result["evaluations"],
        "generations": result["generations"],
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "population": [
            {
                "weights": dict(zip(("auc_error", "parameter_count", "macs"), weight)),
                **serialize_individual(item),
            }
            for weight, item in zip(result["weights"], result["population"])
        ],
        "pareto_front": [serialize_individual(item) for item in result["pareto_front"]],
        "evaluated": serialize_evaluated(result["evaluated"], evaluator.records),
    }
    save_json(output, summary)
    print(json.dumps({
        "evaluations": result["evaluations"],
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "pareto_front": summary["pareto_front"],
        "result": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
