"""Create Stage 6 tables and figures from completed search and test results."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.compare_searches import attach_costs, compare_searches, write_comparison
from src.datasets import DATASETS, results_root
from src.nsga2 import dominates
from src.objectives import MACS, OBJECTIVES, PARAMS, cost_bounds
from src.search_runtime import save_json, search_root


METHODS = ("ga", "random")
REQUIRED_MODELS = ("ga", "random", "baseline")
MODEL_ORDER = (
    "ga",
    "ga_knee",
    "ga_smallest_baseline",
    "random",
    "random_knee",
    "random_smallest_baseline",
    "baseline",
)
DERMAMNIST_CLASSES = DATASETS["dermamnist"].classes
PARAMETER_COST = PARAMS
MAC_COST = MACS
MAC_INPUT_SIZE = (64, 64)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pareto_front(points, *, cost_key=PARAMETER_COST):
    return [
        point for point in points
        if not any(
            dominates(
                (other["val_macro_auc_ovr"], other[cost_key]),
                (point["val_macro_auc_ovr"], point[cost_key]),
            )
            for other in points if other is not point
        )
    ]


def normalized_point(point, min_cost, max_cost, *, cost_key=PARAMETER_COST, log_scale=False):
    cost = point[cost_key]
    if log_scale:
        min_cost, max_cost, cost = map(math.log10, (min_cost, max_cost, cost))
    size_score = (max_cost - cost) / (max_cost - min_cost)
    return float(point["val_macro_auc_ovr"]), float(size_score)


def _hypervolume_2d(normalized_points: list[tuple[float, float]]) -> float:
    ordered = sorted(normalized_points, key=lambda item: item[0])
    area, previous_x = 0.0, 0.0
    for index, (x_val, _) in enumerate(ordered):
        height = max(y_val for _, y_val in ordered[index:])
        area += max(0.0, x_val - previous_x) * max(0.0, height)
        previous_x = max(previous_x, x_val)
    return area


def hypervolume(front, min_cost, max_cost, *, cost_key=PARAMETER_COST, log_scale=False):
    return _hypervolume_2d(
        [
            normalized_point(point, min_cost, max_cost, cost_key=cost_key, log_scale=log_scale)
            for point in front
        ]
    )


def coverage(first, second, *, cost_key=PARAMETER_COST):
    return sum(
        any(
            dominates(
                (left["val_macro_auc_ovr"], left[cost_key]),
                (right["val_macro_auc_ovr"], right[cost_key]),
            )
            for left in first
        )
        for right in second
    ) / len(second)


def knee_point(front, min_cost, max_cost, *, cost_key=PARAMETER_COST, log_scale=False):
    return min(
        front,
        key=lambda point: (
            sum(
                (1.0 - value) ** 2
                for value in normalized_point(
                    point, min_cost, max_cost, cost_key=cost_key, log_scale=log_scale
                )
            ),
            point[cost_key],
        ),
    )


def convergence(points, baseline_auc, parameter_limits, mac_limits):
    rows = []
    for end in range(1, len(points) + 1):
        prefix = points[:end]
        eligible = [point["parameter_count"] for point in prefix if point["val_macro_auc_ovr"] >= baseline_auc]
        parameter_front = pareto_front(prefix)
        mac_front = pareto_front(prefix, cost_key=MAC_COST)
        rows.append({
            "evaluation": end,
            "best_auc": max(point["val_macro_auc_ovr"] for point in prefix),
            "hypervolume": hypervolume(parameter_front, *parameter_limits),
            "log_hypervolume": hypervolume(parameter_front, *parameter_limits, log_scale=True),
            "macs_hypervolume": hypervolume(mac_front, *mac_limits, cost_key=MAC_COST),
            "log_macs_hypervolume": hypervolume(
                mac_front, *mac_limits, cost_key=MAC_COST, log_scale=True
            ),
            "smallest_parameters_reaching_baseline_auc": min(eligible) if eligible else None,
        })
    return rows


def compute_per_class_metrics(result: dict, classes=DERMAMNIST_CLASSES) -> list[dict]:
    labels = np.asarray(result["test_labels"], dtype=np.int64)
    predictions = np.asarray(result["test_predictions"], dtype=np.int64)
    probabilities = (
        np.asarray(result["test_probabilities"], dtype=np.float64)
        if "test_probabilities" in result
        else None
    )
    class_ids = list(range(len(classes)))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0
    )
    rows = []
    for idx, class_name in enumerate(classes):
        auc_ovr = None
        if probabilities is not None:
            binary_true = (labels == idx).astype(np.int64)
            if 0 < binary_true.sum() < len(binary_true):
                auc_ovr = float(roc_auc_score(binary_true, probabilities[:, idx]))
        rows.append({
            "model": result["model"],
            "class_id": idx,
            "class_name": class_name,
            "support": int(supports[idx]),
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "auc_ovr": auc_ovr,
        })
    return rows


def _bootstrap_distributions(
    results: list[dict], *, n_bootstrap: int, seed: int, num_classes: int = len(DERMAMNIST_CLASSES),
) -> tuple[dict[str, dict], list[dict]]:
    if n_bootstrap <= 0:
        return {}, []
    reference_labels = np.asarray(results[0]["test_labels"], dtype=np.int64)
    class_ids = np.unique(reference_labels)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(reference_labels == class_id) for class_id in class_ids]
    boot_indices = np.concatenate([
        rng.choice(indices, size=(n_bootstrap, len(indices)), replace=True)
        for indices in class_indices
    ], axis=1)

    metrics_by_model: dict[str, dict[str, np.ndarray]] = {}
    observed_by_model: dict[str, dict[str, float]] = {}
    ci_summary: dict[str, dict] = {}

    for result in results:
        name = result["model"]
        y_true = np.asarray(result["test_labels"], dtype=np.int64)
        y_pred = np.asarray(result["test_predictions"], dtype=np.int64)
        probs = (
            np.asarray(result["test_probabilities"], dtype=np.float64)
            if "test_probabilities" in result
            else None
        )
        metric_arrays = {
            "test_accuracy": np.empty(n_bootstrap),
            "test_macro_recall": np.empty(n_bootstrap),
            "test_macro_f1": np.empty(n_bootstrap),
        }
        if probs is not None and len(class_ids) == num_classes:
            metric_arrays["test_macro_auc_ovr"] = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            idx = boot_indices[b]
            yt = y_true[idx]
            yp = y_pred[idx]
            metric_arrays["test_accuracy"][b] = accuracy_score(yt, yp)
            metric_arrays["test_macro_recall"][b] = recall_score(
                yt, yp, labels=class_ids, average="macro", zero_division=0
            )
            metric_arrays["test_macro_f1"][b] = f1_score(
                yt, yp, labels=class_ids, average="macro", zero_division=0
            )
            if "test_macro_auc_ovr" in metric_arrays:
                metric_arrays["test_macro_auc_ovr"][b] = roc_auc_score(
                    yt, probs[idx], average="macro", multi_class="ovr"
                )

        observed = {metric: float(result["test_metrics"][metric]) for metric in metric_arrays}
        if "test_macro_auc_ovr" in metric_arrays:
            observed["test_macro_auc_ovr"] = float(
                roc_auc_score(y_true, probs, average="macro", multi_class="ovr")
            )
        metrics_by_model[name] = metric_arrays
        observed_by_model[name] = observed
        ci_summary[name] = {}
        for metric_name, values in metric_arrays.items():
            ci_summary[name][metric_name] = {
                "observed": observed[metric_name],
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if n_bootstrap > 1 else 0.0,
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
            }

    model_names = [result["model"] for result in results]
    comparison_pairs = []
    if "baseline" in metrics_by_model:
        for name in model_names:
            if name != "baseline":
                comparison_pairs.append((name, "baseline"))
    for suffix in ("", "_knee", "_smallest_baseline"):
        if f"ga{suffix}" in metrics_by_model and f"random{suffix}" in metrics_by_model:
            comparison_pairs.append((f"ga{suffix}", f"random{suffix}"))

    paired_summary = []
    for left, right in comparison_pairs:
        pair_entry = {"left": left, "right": right, "metrics": {}}
        for metric_name, left_vals in metrics_by_model[left].items():
            if metric_name not in metrics_by_model[right]:
                continue
            deltas = left_vals - metrics_by_model[right][metric_name]
            pair_entry["metrics"][metric_name] = {
                "delta_observed": observed_by_model[left][metric_name] - observed_by_model[right][metric_name],
                "delta_mean": float(np.mean(deltas)),
                "ci95_low": float(np.percentile(deltas, 2.5)),
                "ci95_high": float(np.percentile(deltas, 97.5)),
            }
        paired_summary.append(pair_entry)

    return ci_summary, paired_summary


def save_metrics_csv(path, results):
    fields = [
        "model", "parameter_count", MAC_COST, "gflops_per_image", "latency_ms", "latency_device",
        "best_epoch", "runtime_seconds", "test_loss",
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
                MAC_COST: result[MAC_COST],
                "gflops_per_image": 2 * result[MAC_COST] / 1e9,
                "latency_ms": result.get("latency_ms"),
                "latency_device": result.get("latency_device"),
                "best_epoch": result["best_epoch"],
                "runtime_seconds": result["runtime_seconds"],
                **result["test_metrics"],
            })


def save_per_class_csv(path, per_class_rows: list[dict]) -> None:
    fields = ["model", "class_id", "class_name", "support", "precision", "recall", "f1", "auc_ovr"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)


def save_architecture_costs_csv(path, searches):
    fields = ["method", "evaluation", "val_macro_auc_ovr", PARAMETER_COST, MAC_COST]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for method in METHODS:
            writer.writerows({"method": method, **point} for point in searches[method]["evaluated"])


def plot_pareto(path, searches, baseline, *, cost_key=PARAMETER_COST):
    fig, ax = plt.subplots(figsize=(9, 6))
    unit, cost_label = (1e9, "GMAC per image") if cost_key == MAC_COST else (1e6, "Parameters (millions)")
    styles = {"ga": ("tab:blue", "o", "GA"), "random": ("tab:orange", "s", "Random")}
    for method in METHODS:
        color, marker, label = styles[method]
        points = searches[method]["evaluated"]
        front = sorted(pareto_front(points, cost_key=cost_key), key=lambda point: point[cost_key])
        ax.scatter([p[cost_key] / unit for p in points], [p["val_macro_auc_ovr"] for p in points],
                   color=color, marker=marker, alpha=0.25, label=f"{label} evaluations")
        ax.plot([p[cost_key] / unit for p in front], [p["val_macro_auc_ovr"] for p in front],
                color=color, marker=marker, linewidth=2, label=f"{label} Pareto front")
    ax.scatter(baseline[cost_key] / unit, baseline["metrics"]["val_macro_auc_ovr"],
               color="black", marker="*", s=180, label="ResNet-18 baseline (full training)")
    ax.set(title="Validation Pareto front", xlabel=cost_label, ylabel="Macro ROC AUC (OvR)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(path, curves):
    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
    for method, color, label in (("ga", "tab:blue", "GA"), ("random", "tab:orange", "Random")):
        rows = curves[method]
        x = [row["evaluation"] for row in rows]
        axes[0].plot(x, [row["best_auc"] for row in rows], color=color, label=label)
        axes[1].plot(x, [row["hypervolume"] for row in rows], color=color, label=f"{label} linear")
        axes[1].plot(x, [row["log_hypervolume"] for row in rows], "--", color=color, label=f"{label} log")
        axes[2].plot(x, [row["macs_hypervolume"] for row in rows], color=color, label=f"{label} linear")
        axes[2].plot(x, [row["log_macs_hypervolume"] for row in rows], "--", color=color, label=f"{label} log")
        axes[3].plot(x, [np.nan if row["smallest_parameters_reaching_baseline_auc"] is None
                         else row["smallest_parameters_reaching_baseline_auc"] / 1e6 for row in rows],
                     color=color, label=label)
    axes[0].set_ylabel("Best validation AUC")
    axes[1].set_ylabel("AUC–parameters HV")
    axes[2].set_ylabel("AUC–MACs HV")
    axes[3].set_ylabel("Smallest reaching baseline AUC (M params)")
    axes[3].set_xlabel("Evaluated architectures")
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
    count = len(results)
    width = min(0.25, 0.82 / max(1, count))
    fig, ax = plt.subplots(figsize=(max(10, 2 + 1.4 * count), 6))
    for index, result in enumerate(results):
        values = [result["test_metrics"][metric] for metric in metric_names]
        offset = (index - (count - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=result["model"].upper())
    ax.set(title="Test metrics", ylabel="Score", xticks=x, xticklabels=labels, ylim=(0, 1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2 if count > 4 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusions(path, results, num_classes=len(DERMAMNIST_CLASSES)):
    count = len(results)
    ncols = min(3, count) if count <= 3 else min(4, math.ceil(count / 2))
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, result in zip(axes_flat, results):
        matrix = confusion_matrix(result["test_labels"], result["test_predictions"], labels=range(num_classes))
        ConfusionMatrixDisplay(matrix, display_labels=range(num_classes)).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(result["model"].upper())
    for axis in axes_flat[count:]:
        axis.axis("off")
    fig.suptitle("Test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _discover_test_results(test_dir: Path, seed: int, dataset="dermamnist") -> list[dict]:
    results = []
    manifest = test_dir / f"seed_{seed}" / "summary.json"
    if manifest.is_file():
        names = [row["model"] for row in load_json(manifest)["results"]]
        if not set(REQUIRED_MODELS).issubset(names):
            raise ValueError("Test results must contain GA, Random and the baseline")
    else:
        names = MODEL_ORDER
    for model_name in names:
        result_path = test_dir / model_name / f"seed_{seed}" / "result.json"
        if manifest.is_file() or model_name in REQUIRED_MODELS:
            results.append(load_json(result_path))
        elif result_path.is_file():
            results.append(load_json(result_path))
    reference_labels = np.asarray(results[0]["test_labels"])
    class_count = DATASETS[dataset].num_classes
    if (
        reference_labels.ndim != 1
        or reference_labels.size == 0
        or not np.issubdtype(reference_labels.dtype, np.integer)
        or np.any((reference_labels < 0) | (reference_labels >= class_count))
    ):
        raise ValueError("Saved test labels must be a nonempty array of dataset class IDs")
    for result in results:
        name = result["model"]
        config = result.get("config", {})
        if config.get("dataset", "dermamnist") != dataset or config.get("seed") != seed:
            raise ValueError(f"{name}: saved dataset or seed does not match the analysis")
        ref_digest = results[0].get("config", {}).get("test_sha256")
        if ref_digest is not None and config.get("test_sha256") != ref_digest:
            raise ValueError(f"{name}: saved test split differs from the reference")
        if not np.array_equal(reference_labels, result["test_labels"]):
            raise ValueError(f"{name}: saved test labels do not match the reference order")
        predictions = np.asarray(result["test_predictions"])
        if (
            predictions.shape != reference_labels.shape
            or not np.issubdtype(predictions.dtype, np.integer)
            or np.any((predictions < 0) | (predictions >= class_count))
        ):
            raise ValueError(f"{name}: expected one valid prediction per test label")
        if "test_probabilities" in result:
            probabilities = np.asarray(result["test_probabilities"], dtype=np.float64)
            if (
                probabilities.shape != (reference_labels.size, class_count)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0)
            ):
                raise ValueError(f"{name}: expected one normalized probability row per test label")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="dermamnist")
    parser.add_argument("--space", choices=("v1", "v2"), default="v1")
    parser.add_argument("--objectives", choices=tuple(OBJECTIVES), default="params")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-budget", type=int, default=60)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ga-summary", type=Path, default=None)
    parser.add_argument("--random-summary", type=Path, default=None)
    parser.add_argument("--baseline-result", type=Path, default=None)
    parser.add_argument("--baseline-proxy-result", type=Path)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.expected_budget <= 0 or args.n_bootstrap < 0:
        parser.error("--expected-budget must be positive and --n-bootstrap must be nonnegative")

    base_results = search_root(args)
    protocol_results = results_root(args.protocol, args.dataset)
    if args.ga_summary is None:
        args.ga_summary = base_results / "ga_search" / f"seed_{args.seed}" / "summary.json"
    if args.random_summary is None:
        args.random_summary = base_results / "random_search" / f"seed_{args.seed}" / "summary.json"
    if args.baseline_result is None:
        args.baseline_result = protocol_results / "baseline" / f"seed_{args.seed}" / "result.json"
    if args.test_dir is None:
        args.test_dir = base_results / "test_eval"
    if args.output_dir is None:
        args.output_dir = base_results / "visualizations"
        if args.seed != 42:
            args.output_dir /= f"seed_{args.seed}"

    searches = {"ga": load_json(args.ga_summary), "random": load_json(args.random_summary)}
    if any(
        search.get("evaluations") != args.expected_budget
        or len(search.get("evaluated", [])) != args.expected_budget
        or search["config"]["seed"] != args.seed
        or search["config"].get("protocol", "v1") != args.protocol
        or search["config"].get("architecture_space", "v1").rsplit("_", 1)[-1] != args.space
        or tuple(search["config"].get("objectives", OBJECTIVES["params"])) != OBJECTIVES[args.objectives]
        or any(not row.get("full_fidelity", True) for row in search["evaluated"])
        for search in searches.values()
    ):
        raise ValueError(f"Analyze requires completed {args.expected_budget}-evaluation GA and Random searches")
    baseline = load_json(args.baseline_result)
    baseline_config = baseline["config"]
    if (baseline_config.get("dataset", "dermamnist") != args.dataset
        or baseline_config["seed"] != args.seed or baseline_config.get("protocol", "v1") != args.protocol):
        raise ValueError("Baseline dataset, protocol or seed differs from the analysis")
    proxy_settings = searches["ga"]["config"]
    if ("train_val_sha256" in baseline_config
        and baseline_config["train_val_sha256"] != proxy_settings.get("train_val_sha256")):
        raise ValueError("Baseline and search train/validation split digests differ")
    classes = DATASETS[args.dataset].classes
    results = _discover_test_results(args.test_dir, args.seed, args.dataset)
    comparison = compare_searches(searches, OBJECTIVES[args.objectives], args.dataset)
    attach_costs([baseline, *results], len(classes))
    limits = cost_bounds(args.space, len(classes))
    parameter_limits, mac_limits = limits[PARAMS], limits[MACS]
    min_parameters, max_parameters = parameter_limits
    min_macs, max_macs = mac_limits
    baseline_auc = baseline["metrics"]["val_macro_auc_ovr"]
    proxy_auc = None
    if args.baseline_proxy_result is not None:
        proxy = load_json(args.baseline_proxy_result)
        defaults = {"augmentation": "none", "weight_decay": 0.0, "amp": False, "preload_gpu": False}
        keys = ("seed", "epochs", "optimizer", "scheduler", "loss", "learning_rate",
                "batch_size", "train_val_sha256", *defaults)
        if any(proxy["config"].get(key, defaults.get(key)) != proxy_settings.get(key, defaults.get(key))
               for key in keys):
            raise ValueError("Baseline proxy settings differ from the search")
        proxy_auc = proxy["metrics"]["val_macro_auc_ovr"]
    search_epochs = int(searches["ga"].get("config", {}).get("epochs", 20))
    baseline_history = baseline.get("history", [])
    baseline_full_prefix_auc = (
        max(item["val_macro_auc_ovr"] for item in baseline_history[:search_epochs])
        if len(baseline_history) >= search_epochs
        else None
    )

    fronts = {name: pareto_front(searches[name]["evaluated"]) for name in METHODS}
    mac_fronts = {
        name: pareto_front(searches[name]["evaluated"], cost_key=MAC_COST) for name in METHODS
    }
    curves = {
        name: convergence(searches[name]["evaluated"], baseline_auc if proxy_auc is None else proxy_auc,
                          parameter_limits, mac_limits)
        for name in METHODS
    }
    for name, curve in curves.items():
        for row, objective_row in zip(curve, comparison["methods"][name]["convergence"], strict=True):
            row["objective_log_hypervolume"] = objective_row["log_hypervolume"]
    per_class_rows = []
    per_class_by_model = {}
    for result in results:
        rows = compute_per_class_metrics(result, classes)
        per_class_rows.extend(rows)
        per_class_by_model[result["model"]] = rows

    bootstrap_ci, paired_comparisons = _bootstrap_distributions(
        results, n_bootstrap=args.n_bootstrap, seed=args.seed, num_classes=len(classes),
    )

    analysis = {
        "protocol": args.protocol,
        "dataset": args.dataset,
        "space": args.space,
        "objectives": comparison["objectives"],
        "search_settings": {
            name: {key: value for key, value in searches[name]["config"].items()
                   if key not in ("seed", "device")}
            for name in METHODS
        },
        "seed": args.seed,
        "normalization": {
            "auc": {"minimum": 0.0, "maximum": 1.0},
            "parameters": {"minimum": min_parameters, "maximum": max_parameters},
            "hypervolume_reference_point": [0.0, 0.0],
        },
        "log_scaled_normalization": {
            "auc": {"minimum": 0.0, "maximum": 1.0},
            "parameters": {
                "minimum": min_parameters,
                "maximum": max_parameters,
                "log10_minimum": math.log10(min_parameters),
                "log10_maximum": math.log10(max_parameters),
            },
            "hypervolume_reference_point": [0.0, 0.0],
        },
        "macs_normalization": {
            "auc": {"minimum": 0.0, "maximum": 1.0},
            MAC_COST: {
                "minimum": min_macs,
                "maximum": max_macs,
                "log10_minimum": math.log10(min_macs),
                "log10_maximum": math.log10(max_macs),
            },
            "hypervolume_reference_point": [0.0, 0.0],
        },
        "macs_definition": {
            "input_shape_chw": [3, *MAC_INPUT_SIZE],
            "included_layers": ["Conv2d", "Linear"],
            "flops_per_mac_if_counted_separately": 2,
        },
        "baseline_reference": {
            "val_macro_auc_ovr_full": baseline_auc,
            "val_macro_auc_ovr_matched_proxy": proxy_auc,
            "val_macro_auc_ovr_full_training_prefix": baseline_full_prefix_auc,
            "full_training_prefix_epochs": search_epochs,
            MAC_COST: baseline[MAC_COST],
        },
        "search": {
            name: {
                "hypervolume": curves[name][-1]["hypervolume"],
                "log_hypervolume": curves[name][-1]["log_hypervolume"],
                "macs_hypervolume": curves[name][-1]["macs_hypervolume"],
                "log_macs_hypervolume": curves[name][-1]["log_macs_hypervolume"],
                "objective_log_hypervolume": comparison["methods"][name]["log_hypervolume"],
                "objective_linear_hypervolume": comparison["methods"][name]["linear_hypervolume"],
                "objective_pareto_size": len(comparison["methods"][name]["pareto_front"]),
                "best_auc": max(point["val_macro_auc_ovr"] for point in searches[name]["evaluated"]),
                "pareto_size": len(fronts[name]),
                "knee_point": knee_point(fronts[name], min_parameters, max_parameters),
                "log_knee_point": knee_point(fronts[name], min_parameters, max_parameters, log_scale=True),
                "pareto_front_macs": sorted(mac_fronts[name], key=lambda point: point[MAC_COST]),
                "macs_knee_point": knee_point(mac_fronts[name], *mac_limits, cost_key=MAC_COST),
                "smallest_model_reaching_baseline_auc": min(
                    (point for point in searches[name]["evaluated"] if point["val_macro_auc_ovr"] >= baseline_auc),
                    key=lambda point: point["parameter_count"], default=None,
                ),
                "smallest_model_reaching_matched_proxy_auc": min(
                    (point for point in searches[name]["evaluated"]
                     if proxy_auc is not None and point["val_macro_auc_ovr"] >= proxy_auc),
                    key=lambda point: point[OBJECTIVES[args.objectives][1]], default=None,
                ),
                "smallest_model_reaching_baseline_full_prefix_auc": min(
                    (
                        point for point in searches[name]["evaluated"]
                        if baseline_full_prefix_auc is not None
                        and point["val_macro_auc_ovr"] >= baseline_full_prefix_auc
                    ),
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
        "coverage_macs": {
            "ga_over_random": coverage(mac_fronts["ga"], mac_fronts["random"], cost_key=MAC_COST),
            "random_over_ga": coverage(mac_fronts["random"], mac_fronts["ga"], cost_key=MAC_COST),
        },
        "coverage_objectives": comparison["coverage"],
        "test_results": [{
            "model": result["model"],
            "parameter_count": result["parameter_count"],
            MAC_COST: result[MAC_COST],
            "gflops_per_image": 2 * result[MAC_COST] / 1e9,
            "latency_ms": result.get("latency_ms"),
            "best_epoch": result["best_epoch"],
            "runtime_seconds": result["runtime_seconds"],
            **result["test_metrics"],
        } for result in results],
        "per_class_metrics": per_class_by_model,
        "bootstrap": {
            "method": "paired_class_stratified_percentile",
            "replicates": args.n_bootstrap,
            "seed": args.seed,
            "models_missing_test_probabilities": [
                result["model"] for result in results if "test_probabilities" not in result
            ],
        },
        "bootstrap_ci": bootstrap_ci,
        "paired_comparisons": paired_comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison(comparison, args.output_dir)
    save_json(args.output_dir / "analysis.json", analysis)
    save_metrics_csv(args.output_dir / "test_metrics.csv", results)
    save_per_class_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    save_architecture_costs_csv(args.output_dir / "architecture_costs.csv", searches)
    plot_pareto(args.output_dir / "pareto_front.png", searches, baseline)
    plot_pareto(args.output_dir / "pareto_macs.png", searches, baseline, cost_key=MAC_COST)
    plot_convergence(args.output_dir / "convergence.png", curves)
    plot_test_metrics(args.output_dir / "test_metrics.png", results)
    plot_confusions(args.output_dir / "confusion_matrices.png", results, len(classes))
    print(json.dumps({"output": str(args.output_dir), "files": sorted(path.name for path in args.output_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
