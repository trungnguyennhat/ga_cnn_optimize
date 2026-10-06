"""Create Stage 6 tables and figures from completed search and test results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import build_model, count_parameters
from src.nsga2 import dominates
from src.search_runtime import save_json
from src.search_space import DROPOUT_CHOICES, KERNEL_CHOICES, STAGE_BLOCK_CHOICES, STAGE_CHANNEL_CHOICES
from src.train import PROJECT_ROOT


METHODS = ("ga", "random")
MODELS = ("ga", "random", "baseline")


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pareto_front(points):
    return [
        point for point in points
        if not any(
            dominates(
                (other["val_macro_auc_ovr"], other["parameter_count"]),
                (point["val_macro_auc_ovr"], point["parameter_count"]),
            )
            for other in points if other is not point
        )
    ]


def parameter_bounds():
    minimum = {
        "stage_blocks": [STAGE_BLOCK_CHOICES[0]] * 4,
        "stage_channels": [choices[0] for choices in STAGE_CHANNEL_CHOICES],
        "kernel_sizes": [KERNEL_CHOICES[0]] * 4,
        "dropout": DROPOUT_CHOICES[0],
    }
    maximum = {
        "stage_blocks": [STAGE_BLOCK_CHOICES[-1]] * 4,
        "stage_channels": [choices[-1] for choices in STAGE_CHANNEL_CHOICES],
        "kernel_sizes": [KERNEL_CHOICES[-1]] * 4,
        "dropout": DROPOUT_CHOICES[-1],
    }
    return count_parameters(build_model(minimum)), count_parameters(build_model(maximum))


def normalized_point(point, min_parameters, max_parameters):
    size_score = (max_parameters - point["parameter_count"]) / (max_parameters - min_parameters)
    return float(point["val_macro_auc_ovr"]), float(size_score)


def hypervolume(points, min_parameters, max_parameters):
    front = pareto_front(points)
    normalized = sorted(
        (normalized_point(point, min_parameters, max_parameters) for point in front),
        key=lambda item: item[0],
    )
    area, previous_auc = 0.0, 0.0
    for index, (auc, _) in enumerate(normalized):
        height = max(size for _, size in normalized[index:])
        area += (auc - previous_auc) * height
        previous_auc = auc
    return area


def coverage(first, second):
    return sum(
        any(
            dominates(
                (left["val_macro_auc_ovr"], left["parameter_count"]),
                (right["val_macro_auc_ovr"], right["parameter_count"]),
            )
            for left in first
        )
        for right in second
    ) / len(second)


def knee_point(front, min_parameters, max_parameters):
    return min(
        front,
        key=lambda point: sum(
            (1.0 - value) ** 2
            for value in normalized_point(point, min_parameters, max_parameters)
        ),
    )


def convergence(points, baseline_auc, min_parameters, max_parameters):
    rows = []
    for end in range(1, len(points) + 1):
        prefix = points[:end]
        eligible = [point["parameter_count"] for point in prefix if point["val_macro_auc_ovr"] >= baseline_auc]
        rows.append({
            "evaluation": end,
            "best_auc": max(point["val_macro_auc_ovr"] for point in prefix),
            "hypervolume": hypervolume(prefix, min_parameters, max_parameters),
            "smallest_parameters_reaching_baseline_auc": min(eligible) if eligible else None,
        })
    return rows


def save_metrics_csv(path, results):
    fields = [
        "model", "parameter_count", "best_epoch", "runtime_seconds", "test_loss",
        "test_accuracy", "test_macro_precision", "test_macro_recall",
        "test_macro_f1", "test_macro_auc_ovr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "model": result["model"],
                "parameter_count": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                "runtime_seconds": result["runtime_seconds"],
                **result["test_metrics"],
            })


def plot_pareto(path, searches, baseline):
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {"ga": ("tab:blue", "o", "GA"), "random": ("tab:orange", "s", "Random")}
    for method in METHODS:
        color, marker, label = styles[method]
        points = searches[method]["evaluated"]
        front = sorted(pareto_front(points), key=lambda point: point["parameter_count"])
        ax.scatter([p["parameter_count"] / 1e6 for p in points], [p["val_macro_auc_ovr"] for p in points],
                   color=color, marker=marker, alpha=0.25, label=f"{label} evaluations")
        ax.plot([p["parameter_count"] / 1e6 for p in front], [p["val_macro_auc_ovr"] for p in front],
                color=color, marker=marker, linewidth=2, label=f"{label} Pareto front")
    ax.scatter(baseline["parameter_count"] / 1e6, baseline["metrics"]["val_macro_auc_ovr"],
               color="black", marker="*", s=180, label="ResNet-18 baseline")
    ax.set(title="Validation Pareto front", xlabel="Parameters (millions)", ylabel="Macro ROC AUC (OvR)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(path, curves):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for method, color, label in (("ga", "tab:blue", "GA"), ("random", "tab:orange", "Random")):
        rows = curves[method]
        x = [row["evaluation"] for row in rows]
        axes[0].plot(x, [row["best_auc"] for row in rows], color=color, label=label)
        axes[1].plot(x, [row["hypervolume"] for row in rows], color=color, label=label)
        axes[2].plot(x, [np.nan if row["smallest_parameters_reaching_baseline_auc"] is None
                         else row["smallest_parameters_reaching_baseline_auc"] / 1e6 for row in rows],
                     color=color, label=label)
    axes[0].set_ylabel("Best validation AUC")
    axes[1].set_ylabel("Hypervolume")
    axes[2].set_ylabel("Smallest model ≥ baseline AUC (M)")
    axes[2].set_xlabel("Evaluated architectures")
    axes[0].set_title("Search convergence")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test_metrics(path, results):
    metric_names = ("test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_macro_auc_ovr")
    labels = ("Accuracy", "Precision", "Recall", "F1", "AUC")
    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, result in enumerate(results):
        values = [result["test_metrics"][metric] for metric in metric_names]
        ax.bar(x + (index - 1) * width, values, width, label=result["model"].upper())
    ax.set(title="Test metrics", ylabel="Score", xticks=x, xticklabels=labels, ylim=(0, 1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusions(path, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, result in zip(axes, results):
        matrix = confusion_matrix(result["test_labels"], result["test_predictions"], labels=range(7))
        ConfusionMatrixDisplay(matrix, display_labels=range(7)).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(result["model"].upper())
    fig.suptitle("Test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-summary", type=Path, default=PROJECT_ROOT / "results" / "ga_search" / "seed_42" / "summary.json")
    parser.add_argument("--random-summary", type=Path, default=PROJECT_ROOT / "results" / "random_search" / "seed_42" / "summary.json")
    parser.add_argument("--baseline-result", type=Path, default=PROJECT_ROOT / "results" / "baseline" / "seed_42" / "result.json")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "results" / "test_eval")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "visualizations")
    args = parser.parse_args()

    searches = {"ga": load_json(args.ga_summary), "random": load_json(args.random_summary)}
    if any(searches[name].get("evaluations") != 60 for name in METHODS):
        raise ValueError("Stage 6 requires completed 60-evaluation GA and Random searches")
    baseline = load_json(args.baseline_result)
    results = [load_json(args.test_dir / name / "seed_42" / "result.json") for name in MODELS]
    min_parameters, max_parameters = parameter_bounds()
    baseline_auc = baseline["metrics"]["val_macro_auc_ovr"]
    fronts = {name: pareto_front(searches[name]["evaluated"]) for name in METHODS}
    curves = {
        name: convergence(searches[name]["evaluated"], baseline_auc, min_parameters, max_parameters)
        for name in METHODS
    }
    analysis = {
        "normalization": {
            "auc": {"minimum": 0.0, "maximum": 1.0},
            "parameters": {"minimum": min_parameters, "maximum": max_parameters},
            "hypervolume_reference_point": [0.0, 0.0],
        },
        "search": {
            name: {
                "hypervolume": curves[name][-1]["hypervolume"],
                "best_auc": max(point["val_macro_auc_ovr"] for point in searches[name]["evaluated"]),
                "pareto_size": len(fronts[name]),
                "knee_point": knee_point(fronts[name], min_parameters, max_parameters),
                "smallest_model_reaching_baseline_auc": min(
                    (point for point in searches[name]["evaluated"] if point["val_macro_auc_ovr"] >= baseline_auc),
                    key=lambda point: point["parameter_count"], default=None,
                ),
                "runtime_seconds": searches[name]["runtime_seconds"],
                "convergence": curves[name],
            }
            for name in METHODS
        },
        "coverage": {
            "ga_over_random": coverage(fronts["ga"], fronts["random"]),
            "random_over_ga": coverage(fronts["random"], fronts["ga"]),
        },
        "test_results": [{
            "model": result["model"], "parameter_count": result["parameter_count"],
            "best_epoch": result["best_epoch"], "runtime_seconds": result["runtime_seconds"],
            **result["test_metrics"],
        } for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "analysis.json", analysis)
    save_metrics_csv(args.output_dir / "test_metrics.csv", results)
    plot_pareto(args.output_dir / "pareto_front.png", searches, baseline)
    plot_convergence(args.output_dir / "convergence.png", curves)
    plot_test_metrics(args.output_dir / "test_metrics.png", results)
    plot_confusions(args.output_dir / "confusion_matrices.png", results)
    print(json.dumps({"output": str(args.output_dir), "files": sorted(path.name for path in args.output_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
