"""Evaluate the saved GA hyperparameter finalists on the test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import scalar
from torch import nn

from src.model import build_model
from src.search_runtime import save_json
from src.test_eval import evaluate, load_test_data
from src.train import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, help="Override the dataset path saved in the run summary.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    finalists = summary["finalists"]
    if not finalists:
        parser.error("The run has no retrained finalists to evaluate.")
    config = summary["config"]
    device = resolve_device(args.device)
    test_data = load_test_data(args.data_path or Path(config["data_path"]))
    criterion = nn.CrossEntropyLoss()
    results = []

    for finalist in finalists:
        trial = finalist["trial"]
        checkpoint_path = args.run_dir / f"trial_{trial}.pt"
        # Legacy finalists store the validation AUC as a NumPy float64.
        with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype("float64"))]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        metrics, _, _ = evaluate(
            model, test_data, criterion,
            batch_size=config["batch_size"], device=device, prefix="test",
        )
        results.append({
            "trial": trial,
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint["best_epoch"],
            "hyperparameters": checkpoint["hyperparameters"],
            "val_macro_auc_ovr": float(checkpoint["val_macro_auc_ovr"]),
            "test_metrics": metrics,
        })
        print(f"trial={trial} test_macro_auc_ovr={metrics['test_macro_auc_ovr']:.6f}", flush=True)
        del model

    output = args.run_dir / "test_summary.json"
    save_json(output, {
        "selected_trial_by_validation": summary["best_trial"],
        "results": results,
    })
    print(f"summary={output}", flush=True)


if __name__ == "__main__":
    main()
