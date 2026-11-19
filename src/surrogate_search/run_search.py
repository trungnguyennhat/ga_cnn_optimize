"""Run online surrogate-assisted NSGA-II on ResNet candidates."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.search_runtime import (
    SEARCH_EPOCHS,
    SearchEvaluator,
    add_search_args,
    save_json,
    search_root,
)
from src.objectives import cost_bounds
from src.surrogate_search import SurrogateAssistedSearch, XGBoostSurrogate
from src.surrogate_search.uncertainty import UncertaintySurrogate


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
    parser.add_argument("--backend", choices=("xgboost", "rf", "gp"), default="xgboost")
    parser.add_argument("--acquisition", choices=("point", "ucb", "ehvi"))
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ehvi-samples", type=int, default=64)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    add_search_args(parser)
    args = parser.parse_args()
    if args.budget < args.warmup_size or args.epochs <= 0:
        parser.error("budget must be at least warmup-size and epochs must be positive")
    acquisition = args.acquisition or ("point" if args.backend == "xgboost" else "ucb")
    if args.backend == "xgboost" and acquisition != "point":
        parser.error("UCB and EHVI require --backend rf or gp")

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if not run_id.replace("-", "_").isidentifier():
        parser.error("--run-id must be a single identifier, optionally containing hyphens")
    args.output_dir = (args.output_dir or search_root(args) / "surrogate_search") / run_id
    search_config = {
        "budget": args.budget,
        "warmup_size": args.warmup_size,
        "acquisition_batch_size": args.acquisition_batch_size,
        "candidate_pool_size": args.candidate_pool_size,
        "population_size": args.population_size,
        "exploration_fraction": args.exploration_fraction,
        "historical_evaluations": False,
        "run_id": run_id,
        "backend": args.backend,
        "acquisition": acquisition,
        "beta": args.beta,
        "ehvi_samples": args.ehvi_samples,
    }
    evaluator = SearchEvaluator(
        args,
        experiment="surrogate_search",
        label=f"{args.backend} surrogate NAS",
        search_config=search_config,
    )
    surrogate = (
        XGBoostSurrogate(random_state=args.seed) if args.backend == "xgboost"
        else UncertaintySurrogate(args.backend, args.seed)
    )
    search = SurrogateAssistedSearch(
        evaluator,
        surrogate=surrogate,
        cost_counter=lambda arch: tuple(
            evaluator.architecture_costs(arch)[key] for key in evaluator.objectives[1:]
        ),
        objectives=evaluator.objectives,
        bounds=cost_bounds(args.space, evaluator.num_classes),
        space=args.space,
        acquisition=acquisition,
        beta=args.beta,
        ehvi_samples=args.ehvi_samples,
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
        "surrogate_diagnostics": result["surrogate_diagnostics"],
        "new_trainings": evaluator.trained,
        "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds,
        "training_seconds": evaluator.training_seconds,
        "pareto_front": [evaluator.serialize(item) for item in result["pareto_front"]],
        "evaluated": evaluator.serialize_evaluated(result["evaluated"]),
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
