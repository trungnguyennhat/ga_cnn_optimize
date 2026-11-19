"""Build an ablation table from named checkpoint results."""

import argparse
import csv
import json
from pathlib import Path

import torch

from src.analyze import plot_confusions
from src.datasets import DATASETS, add_dataset_args, dataset_path
from src.model import build_model
from src.search_runtime import save_json
from src.test_eval import evaluate, load_test_data
from src.train import build_criterion, load_data, resolve_device, split_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=RESULT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", default="auto")
    add_dataset_args(parser)
    args = parser.parse_args()
    rows, predictions = [], []
    classes = DATASETS[args.dataset].num_classes
    test_data = None
    if args.evaluate_test:
        device = resolve_device(args.device)
        path = dataset_path(args.dataset, args.data_path)
        loaded_data = load_data(path, args.dataset)
        digest = split_digest(loaded_data)
        test_data = load_test_data(path, args.dataset)
    names = set()
    for spec in args.run:
        name, separator, path = spec.partition("=")
        if not separator or not path or not name or name in names:
            parser.error("--run requires a unique NAME=RESULT")
        names.add(name)
        path = Path(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        if config.get("dataset", "dermamnist") != args.dataset:
            raise ValueError(f"{path}: dataset does not match the report")
        history = result["history"]
        best_epoch = result.get("best_epoch") or result["metrics"]["best_epoch"]
        best, final = history[best_epoch - 1], history[-1]
        row = {
            "run": name, "seed": config["seed"], "protocol": config.get("protocol", "v1"),
            "optimizer": config["optimizer"], "loss": config["loss"],
            "scheduler": config["scheduler"]["name"], "augmentation": config.get("augmentation", "none"),
            "preload_gpu": config.get("preload_gpu", False), "amp": config.get("amp", False),
            "runtime_seconds": result["runtime_seconds"], "best_epoch": best_epoch,
            "val_macro_auc_ovr": best["val_macro_auc_ovr"], "val_macro_f1": best["val_macro_f1"],
            "val_macro_recall": best.get("val_macro_recall", result.get("validation_metrics", {}).get("val_macro_recall")),
            "final_train_loss": final["train_loss"], "final_val_loss": final["val_loss"],
            "final_loss_gap": final["val_loss"] - final["train_loss"],
        }
        if test_data is not None:
            if config.get("train_val_sha256", digest) != digest:
                raise ValueError(f"{path}: train/validation data differ from the checkpoint")
            model = build_model(result["architecture"], classes).to(device)
            checkpoint = torch.load(path.parent / "checkpoint.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            loss = "unweighted_ce" if config["loss"] == "unweighted_cross_entropy" else config["loss"]
            criterion = build_criterion(loss, loaded_data[0].tensors[1], device, classes)
            metrics, labels, predicted = evaluate(
                model, test_data, criterion, batch_size=config["batch_size"], device=device, prefix="test",
            )
            row.update(metrics)
            if row["val_macro_recall"] is None:
                validation_metrics, _, _ = evaluate(
                    model, loaded_data[1], criterion, batch_size=config["batch_size"],
                    device=device, prefix="val",
                )
                row["val_macro_recall"] = validation_metrics["val_macro_recall"]
            predictions.append({"model": name, "test_labels": labels, "test_predictions": predicted})
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "protocol_ablation.json", {"dataset": args.dataset, "runs": rows})
    with (args.output_dir / "protocol_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if predictions:
        plot_confusions(args.output_dir / "protocol_confusions.png", predictions, classes)
    print(args.output_dir / "protocol_ablation.csv")


if __name__ == "__main__":
    main()
