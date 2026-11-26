"""Run fresh online XGBoost-assisted NSGA-II on ResNet candidates."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.search_runtime import (
    HARDWARE_RESULTS,
    SEARCH_EPOCHS,
    SearchEvaluator,
    save_json,
    serialize_evaluated,
    serialize_individual,
)
from src.surrogate_search import SurrogateAssistedSearch, XGBoostSurrogate
from src.train import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--warmup-size", type=int, default=20)
    parser.add_argument(
        "--acquisition-batch-size", "--batch-size",
        dest="acquisition_batch_size", type=int, default=5,
        help=(
            "Architectures evaluated with real training per surrogate iteration; "
            "--batch-size is retained as a compatibility alias"
        ),
    )
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--hardware-aware", action="store_true", help="Add MACs as a third objective"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.budget < args.warmup_size:
        parser.error("budget must be at least warmup-size")
    if args.output_dir is None:
        root = PROJECT_ROOT / "results"
        args.output_dir = (root / HARDWARE_RESULTS if args.hardware_aware else root) / "surrogate_search"

    # A new timestamped directory prevents SearchEvaluator from seeing cache
    # files produced by any earlier GA, random, or surrogate-search run.
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    args.output_dir = args.output_dir / run_id
    search_config = {
        "budget": args.budget,
        "warmup_size": args.warmup_size,
        "acquisition_batch_size": args.acquisition_batch_size,
        "candidate_pool_size": args.candidate_pool_size,
        "population_size": args.population_size,
        "exploration_fraction": args.exploration_fraction,
        "historical_evaluations": False,
        "run_id": run_id,
    }
    evaluator = SearchEvaluator(
        args,
        experiment="surrogate_search",
        label="XGBoost Surrogate NAS",
        search_config=search_config,
        hardware_aware=args.hardware_aware,
    )
    search = SurrogateAssistedSearch(
        evaluator,
        hardware_aware=args.hardware_aware,
        surrogate=XGBoostSurrogate(random_state=args.seed),
        warmup_size=args.warmup_size,
        acquisition_batch_size=args.acquisition_batch_size,
        candidate_pool_size=args.candidate_pool_size,
        population_size=args.population_size,
        exploration_fraction=args.exploration_fraction,
        seed=args.seed,
    )
    result = search.run(evaluation_budget=args.budget)
    summary = {
        "experiment": "surrogate_search",
        "config": evaluator.run_config,
        "evaluations": result["evaluations"],
        "warmup_evaluations": result["warmup_evaluations"],
        "surrogate_iterations": result["surrogate_iterations"],
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "pareto_front": [
            serialize_individual(item) for item in result["pareto_front"]
        ],
        "evaluated": serialize_evaluated(result["evaluated"], evaluator.records),
    }
    save_json(evaluator.output / "summary.json", summary)
    print(json.dumps({
        "evaluations": result["evaluations"],
        "warmup_evaluations": result["warmup_evaluations"],
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "pareto_front": summary["pareto_front"],
        "result": str(evaluator.output / "summary.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
