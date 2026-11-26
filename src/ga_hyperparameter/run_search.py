"""Tune ResNet18 training with AdamW and a real-valued genetic algorithm."""

import argparse
from datetime import datetime
import math
from pathlib import Path
import time

import torch
from torch import nn

from src.ga_hyperparameter.ga import (
    BOUNDS, CROSSOVER_ETA, CROSSOVER_RATE, MUTATION_ETA, MUTATION_RATE,
    TOURNAMENT_SIZE, decode, run_ga,
)
from src.model import BASELINE_ARCHITECTURE, BaselineResNet
from src.search_runtime import SEARCH_EPOCHS, SEARCH_SCHEDULER_STEP, save_json
from src.train import (
    DATA_PATH, DEFAULT_EPOCHS, PROJECT_ROOT, evaluate_model, load_data,
    resolve_device, set_seed, train_model,
)


EXPERIMENT = "ga_hyperparameter"
FITNESS = "val_macro_auc_ovr"
BATCH_SIZE = 64
FINAL_SCHEDULER_STEP = 20


def score(record):
    return record["metrics"][FITNESS]


def run_search(args):
    started = time.perf_counter()
    device = resolve_device(args.device)
    data = load_data(args.data_path)
    train_labels = data[0].tensors[1]
    class_counts = train_labels.bincount(minlength=7)
    if (class_counts == 0).any():
        raise ValueError("Class weighting requires training examples for all seven classes")
    inverse_frequency = len(train_labels) / (len(class_counts) * class_counts)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output = args.output_dir / f"seed_{args.seed}" / run_id
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    finalists = []
    config = {
        "architecture": BASELINE_ARCHITECTURE,
        "data_path": str(args.data_path.resolve()),
        "seed": args.seed,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "class_counts": class_counts.tolist(),
        "objective": FITNESS,
        "bounds": BOUNDS,
        "budget": args.budget,
        "population_size": args.population_size,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover": {"name": "bounded_sbx", "rate": CROSSOVER_RATE, "eta": CROSSOVER_ETA},
        "mutation": {"name": "polynomial", "rate_per_gene": MUTATION_RATE, "eta": MUTATION_ETA},
        "proxy": {"epochs": args.epochs, "scheduler_step_size": SEARCH_SCHEDULER_STEP},
        "retrain": {
            "top_k": args.top_k, "epochs": args.retrain_epochs,
            "scheduler_step_size": FINAL_SCHEDULER_STEP,
        },
    }
    progress = {"experiment": EXPERIMENT, "config": config, "trials": trials, "finalists": finalists}

    def evaluate(chromosome, hyperparameters, *, optimizer, epochs, scheduler_step, checkpoint=None):
        alpha = hyperparameters["class_weight_alpha"]
        weights = inverse_frequency.pow(alpha)
        set_seed(args.seed)
        model = BaselineResNet()
        metrics, history, runtime = train_model(
            model, data, learning_rate=hyperparameters["learning_rate"],
            batch_size=BATCH_SIZE, optimizer_name=optimizer, epochs=epochs,
            seed=args.seed, device=device, scheduler_step_size=scheduler_step,
            weight_decay=hyperparameters["weight_decay"],
            class_weights=weights if alpha else None,
        )
        if not math.isfinite(metrics[FITNESS]) or not 0 <= metrics[FITNESS] <= 1:
            raise ValueError(f"Invalid validation fitness: {metrics[FITNESS]}")
        record = {
            "chromosome": dict(zip(BOUNDS, chromosome)) if chromosome is not None else None,
            "hyperparameters": hyperparameters,
            "optimizer": optimizer,
            "class_weights": weights.tolist(),
            "metrics": metrics,
            "history": history,
            "runtime_seconds": runtime,
        }
        if checkpoint is not None:
            validation = evaluate_model(
                model, data[1], nn.CrossEntropyLoss(), batch_size=BATCH_SIZE,
                device=device, per_class_recall=True,
            )
            metrics["val_recall_by_class"] = validation["val_recall_by_class"]
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "architecture": BASELINE_ARCHITECTURE,
                "best_epoch": metrics["best_epoch"],
                FITNESS: metrics[FITNESS],
                "hyperparameters": hyperparameters,
                "optimizer": optimizer,
            }, checkpoint)
            record["checkpoint"] = str(checkpoint)
        return record

    def evaluate_proxy(chromosome, hyperparameters, *, optimizer, generation):
        trial = len(trials) + 1
        print(f"trial={trial}/{args.budget} optimizer={optimizer} {hyperparameters}", flush=True)
        record = evaluate(
            chromosome, hyperparameters, optimizer=optimizer,
            epochs=args.epochs, scheduler_step=SEARCH_SCHEDULER_STEP,
        )
        record.update(trial=trial, generation=generation)
        trials.append(record)
        save_json(output / "progress.json", progress)
        print(f"trial={trial}/{args.budget} {FITNESS}={score(record):.6f}", flush=True)
        return record

    print(f"output={output} device={device}", flush=True)
    baseline = evaluate_proxy(
        None, {"learning_rate": 1e-3, "weight_decay": 0.0, "class_weight_alpha": 0.0},
        optimizer="Adam", generation=None,
    )
    generations = run_ga(
        lambda chromosome, generation: score(evaluate_proxy(
            chromosome, decode(chromosome), optimizer="AdamW", generation=generation,
        )),
        budget=args.budget - 1, population_size=args.population_size, seed=args.seed,
    )
    ranked = sorted(trials[1:], key=score, reverse=True)
    for candidate in ranked[:args.top_k]:
        trial = candidate["trial"]
        print(f"retrain trial={trial} epochs={args.retrain_epochs}", flush=True)
        result = evaluate(
            tuple(candidate["chromosome"].values()), candidate["hyperparameters"],
            optimizer="AdamW", epochs=args.retrain_epochs,
            scheduler_step=FINAL_SCHEDULER_STEP, checkpoint=output / f"trial_{trial}.pt",
        )
        finalists.append({"trial": trial, **result})
        save_json(output / "progress.json", progress)

    finalists.sort(key=score, reverse=True)
    save_json(output / "summary.json", {
        **progress,
        "generations": generations,
        "baseline_trial": baseline["trial"],
        "proxy_ranking": [record["trial"] for record in ranked],
        "best_trial": (finalists or ranked)[0]["trial"],
        "selection_stage": "retrain" if finalists else "proxy",
        "runtime_seconds": time.perf_counter() - started,
    })
    print(f"summary={output / 'summary.json'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=30, help="Proxy trials, including one Adam reference")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--top-k", type=int, default=3, help="AdamW finalists to retrain; 0 skips retraining")
    parser.add_argument("--retrain-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / EXPERIMENT)
    args = parser.parse_args()
    if args.budget < 2 or args.population_size < 2:
        parser.error("budget and population-size must be at least 2")
    if args.epochs < 1 or args.retrain_epochs < 1 or args.top_k < 0:
        parser.error("epochs and retrain-epochs must be positive and top-k must be nonnegative")
    run_search(args)


if __name__ == "__main__":
    main()
