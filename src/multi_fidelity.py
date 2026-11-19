"""Run a serial, multi-objective ASHA search with random or NSGA-II proposals."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from src.nsga2 import Individual, crowding_distance, nondominated_sort, select_survivors, tournament_selection
from src.search_runtime import SEARCH_EPOCHS, SearchEvaluator, add_search_args, save_json, search_root
from src.search_space import canonical_architecture, mutate_architecture, random_architecture, space_size, uniform_crossover


class RungScheduler:
    def __init__(self, epochs: tuple[int, ...], reduction_factor: int, grace_trials: int):
        self.rungs = {epoch: [] for epoch in epochs}
        self.reduction_factor = reduction_factor
        self.grace_trials = grace_trials

    def context(self) -> dict:
        scores = {epoch: [item.fitness for item in items] for epoch, items in self.rungs.items()}
        return {
            "algorithm": "serial_mo_asha_v1",
            "rungs": list(self.rungs),
            "reduction_factor": self.reduction_factor,
            "grace_trials": self.grace_trials,
            "rung_state_sha256": hashlib.sha256(json.dumps(scores).encode("utf-8")).hexdigest(),
        }

    def continue_trial(self, history, architecture, costs) -> bool:
        past = self.rungs.get(len(history))
        if past is None or len(past) < self.grace_trials:
            return True
        candidate = Individual(architecture, self.fitness(history, costs))
        cutoff = math.ceil((len(past) + 1) / self.reduction_factor)
        contenders = [Individual(item.architecture, item.fitness) for item in past]
        return any(item is candidate for item in select_survivors([*contenders, candidate], cutoff))

    @staticmethod
    def fitness(history, costs):
        return max(epoch["val_macro_auc_ovr"] for epoch in history), *costs

    def observe(self, history, architecture, costs):
        for epoch, observations in self.rungs.items():
            if len(history) >= epoch:
                observations.append(Individual(architecture, self.fitness(history[:epoch], costs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_search_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--rungs", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--grace-trials", type=int, default=6)
    parser.add_argument("--sampler", choices=("ga", "random"), default="ga")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (min(args.population_size, args.tournament_size, args.grace_trials, args.budget) <= 0
        or args.budget > space_size(args.space) or args.reduction_factor < 2
        or args.rungs != sorted(set(args.rungs)) or not 0 < args.rungs[0] < args.epochs
        or args.rungs[-1] >= args.epochs):
        parser.error("Require positive settings and unique increasing rungs below --epochs")
    args.output_dir = args.output_dir or search_root(args) / f"mo_asha_{args.sampler}"
    scheduler = RungScheduler(tuple(args.rungs), args.reduction_factor, args.grace_trials)
    evaluator = SearchEvaluator(
        args, experiment="mo_asha", label="MO-ASHA", search_config={
            "budget": args.budget, "sampler": args.sampler, "population_size": args.population_size,
            "tournament_size": args.tournament_size, "fidelity": scheduler.context(),
        },
    )
    archive, complete, population = [], [], []
    seen = set()
    rng = random.Random(args.seed)
    for _ in range(args.budget):
        for attempt in range(1000):
            if args.sampler == "ga" and len(population) >= args.population_size and attempt < 100:
                first = tournament_selection(population, rng, args.tournament_size)
                second = tournament_selection(population, rng, args.tournament_size)
                architecture = mutate_architecture(uniform_crossover(first.architecture, second.architecture, rng), rng)
            else:
                architecture = random_architecture(rng, args.space)
            key = canonical_architecture(architecture)
            if key not in seen:
                break
        else:
            raise RuntimeError("Could not find an unseen architecture")
        seen.add(key)
        metadata = evaluator.architecture_costs(architecture)
        costs = tuple(metadata[key] for key in evaluator.objectives[1:])
        fitness = evaluator(
            architecture, args.seed,
            epoch_callback=lambda history: scheduler.continue_trial(history, architecture, costs),
            fidelity_context=scheduler.context(),
        )
        scheduler.observe(evaluator.last_history, architecture, costs)
        item = Individual(architecture, fitness, rank=None)
        archive.append(item)
        if len(evaluator.last_history) == args.epochs:
            complete.append(item)
            population = select_survivors(complete, min(args.population_size, len(complete)))

    fronts = nondominated_sort(complete)
    for front in fronts:
        crowding_distance(front)
    summary = {
        "experiment": "mo_asha", "config": evaluator.run_config,
        "evaluations": len(archive), "full_evaluations": len(complete),
        "new_trainings": evaluator.trained, "cache_hits": evaluator.hits,
        "runtime_seconds": evaluator.runtime_seconds, "training_seconds": evaluator.training_seconds,
        "total_epochs": sum(row["epochs_trained"] for row in evaluator.records),
        "trials_per_rung": {str(epoch): len(items) for epoch, items in scheduler.rungs.items()},
        "pareto_front": [evaluator.serialize(item) for item in fronts[0]],
        "evaluated": evaluator.serialize_evaluated(archive),
    }
    path = evaluator.output / "summary.json"
    save_json(path, summary)
    print(path, flush=True)


if __name__ == "__main__":
    main()
