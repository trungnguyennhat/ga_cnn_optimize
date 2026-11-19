"""Measure rank correlation (Spearman, Kendall) between proxy and full ResNet training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.datasets import DATASETS, add_dataset_args, dataset_path, results_root
from src.model import build_model, count_parameters
from src.search_runtime import add_protocol_args, save_json
from src.search_space import canonical_architecture
from src.train import (
    TRAINING_VERSION,
    is_legacy_protocol,
    load_data,
    loss_config_name,
    preload_data,
    resolve_device,
    resolve_protocol_settings,
    scheduler_config_dict,
    set_seed,
    split_digest,
    train_model,
)


def sample_stratified_architectures(
    summary_paths: list[Path], *, num_models: int
) -> list[dict]:
    pool: dict[str, dict] = {}
    for path in summary_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Search summary not found: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        for item in summary.get("evaluated", []):
            if not item.get("full_fidelity", True):
                continue
            key = canonical_architecture(item["architecture"])
            if key not in pool:
                pool[key] = {
                    "architecture": item["architecture"],
                    "search_val_macro_auc_ovr": float(item["val_macro_auc_ovr"]),
                    "parameter_count": int(item["parameter_count"]),
                    "legacy_record": item.get("result"),
                }
    if len(pool) < num_models:
        raise ValueError(f"Requested {num_models} architectures, found only {len(pool)}")

    ordered = sorted(
        pool.values(),
        key=lambda item: (item["search_val_macro_auc_ovr"], item["parameter_count"]),
    )
    indices = np.linspace(0, len(ordered) - 1, num=num_models, dtype=int)
    seen = set()
    selected = []
    for idx in indices:
        while idx in seen and idx + 1 < len(ordered):
            idx += 1
        seen.add(int(idx))
        selected.append(ordered[int(idx)])
    return selected


def _run_or_load(
    architecture: dict,
    *,
    epochs: int,
    step_size: int,
    seed: int,
    device,
    loaded_data,
    settings: dict,
    cache_dir: Path,
    data_digest: str,
    dataset: str,
    legacy_record: str | None = None,
) -> dict:
    arch_key = canonical_architecture(architecture)
    run_config = {
        "epochs": epochs,
        "learning_rate": 0.001,
        "batch_size": 64,
        "seed": seed,
        "protocol": settings["protocol"],
        "optimizer": settings["optimizer_name"],
        "weight_decay": settings["weight_decay"],
        "scheduler": scheduler_config_dict(
            settings["scheduler_name"], epochs=epochs, step_size=step_size
        ),
        "augmentation": settings["augmentation"],
        "loss": loss_config_name(settings["loss_name"]),
        "amp": settings["use_amp"],
        "preload_gpu": settings["preload_gpu"],
        "device": str(device),
        "dataset": dataset,
        "train_val_sha256": data_digest,
        "training_version": TRAINING_VERSION,
    }
    cache_key = hashlib.sha256(
        (arch_key + json.dumps(run_config, sort_keys=True)).encode("utf-8")
    ).hexdigest()
    path = cache_dir / f"{cache_key}.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("config") == run_config
            and canonical_architecture(record["architecture"]) == arch_key
        ):
            print(f"cache_hit epochs={epochs} arch={arch_key}", flush=True)
            return record

    if legacy_record is not None:
        legacy_path = Path(legacy_record)
        if not legacy_path.is_file():
            legacy_path = Path(str(legacy_path).replace("\\", "/"))
        if legacy_path.is_file():
            record = json.loads(legacy_path.read_text(encoding="utf-8"))
            old = record["config"]
            keys = ("epochs", "learning_rate", "batch_size", "seed", "optimizer", "scheduler",
                    "loss", "device", "train_val_sha256")
            if (dataset == "dermamnist" and old.get("training_version") == 3
                and is_legacy_protocol(settings)
                and all(old.get(key) == run_config[key] for key in keys)
                and canonical_architecture(record["architecture"]) == arch_key
                and len(record["history"]) == epochs):
                print(f"reusing_proxy={legacy_path}", flush=True)
                return record

    set_seed(seed)
    model = build_model(architecture, DATASETS[dataset].num_classes)
    parameters = count_parameters(model)
    metrics, history, runtime = train_model(
        model,
        loaded_data,
        learning_rate=0.001,
        batch_size=64,
        optimizer_name=settings["optimizer_name"],
        epochs=epochs,
        seed=seed,
        device=device,
        scheduler_step_size=step_size,
        weight_decay=settings["weight_decay"],
        scheduler_name=settings["scheduler_name"],
        augmentation=settings["augmentation"],
        loss_name=settings["loss_name"],
        preload_gpu=settings["preload_gpu"],
        use_amp=settings["use_amp"],
    )
    record = {
        "architecture": architecture,
        "parameter_count": parameters,
        "config": run_config,
        "metrics": metrics,
        "history": history,
        "runtime_seconds": runtime,
    }
    save_json(path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-models", type=int, default=12)
    parser.add_argument("--proxy-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=60)
    parser.add_argument("--proxy-step-size", type=int, default=15)
    parser.add_argument("--full-step-size", type=int, default=20)
    parser.add_argument("--device", default="auto")
    add_dataset_args(parser)
    parser.add_argument(
        "--ga-summary",
        type=Path,
    )
    parser.add_argument(
        "--random-summary",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--early-epochs", type=int, nargs="+", default=[5, 10])
    add_protocol_args(parser)
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.num_models or args.num_models < 2 or not 0 < args.proxy_epochs < args.full_epochs:
        parser.error("Require at least two models, 1 <= top-k <= num-models and 0 < proxy-epochs < full-epochs")
    early_epochs = sorted({epoch for epoch in args.early_epochs if 0 < epoch < args.proxy_epochs})
    samples_root = results_root("v1", args.dataset)
    args.ga_summary = args.ga_summary or samples_root / "ga_search" / "seed_42" / "summary.json"
    args.random_summary = args.random_summary or samples_root / "random_search" / "seed_42" / "summary.json"

    if args.output_dir is None:
        args.output_dir = (
            results_root(args.protocol, args.dataset) / "proxy_correlation" / f"seed_{args.seed}"
        )

    settings = resolve_protocol_settings(
        args.protocol,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        augmentation=args.augmentation,
        loss_name=args.loss,
        preload_gpu=args.preload_gpu,
        use_amp=args.amp,
    )
    device = resolve_device(args.device)
    loaded_data = load_data(dataset_path(args.dataset, args.data_path), args.dataset)
    data_digest = split_digest(loaded_data)
    if settings["preload_gpu"] and device.type == "cuda":
        loaded_data = preload_data(loaded_data, device)
    sampled = sample_stratified_architectures(
        [args.ga_summary, args.random_summary], num_models=args.num_models
    )

    cache_dir = args.output_dir / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, item in enumerate(sampled, start=1):
        arch = item["architecture"]
        print(
            f"[{index}/{len(sampled)}] params={item['parameter_count']} "
            f"protocol={settings['protocol']}",
            flush=True,
        )
        proxy_rec = _run_or_load(
            arch,
            epochs=args.proxy_epochs,
            step_size=args.proxy_step_size,
            seed=args.seed,
            device=device,
            loaded_data=loaded_data,
            settings=settings,
            cache_dir=cache_dir,
            data_digest=data_digest,
            dataset=args.dataset,
            legacy_record=item["legacy_record"],
        )
        full_rec = _run_or_load(
            arch,
            epochs=args.full_epochs,
            step_size=args.full_step_size,
            seed=args.seed,
            device=device,
            loaded_data=loaded_data,
            settings=settings,
            cache_dir=cache_dir,
            data_digest=data_digest,
            dataset=args.dataset,
        )
        rows.append({
            "index": index,
            "architecture": arch,
            "parameter_count": int(full_rec["parameter_count"]),
            "proxy_val_macro_auc_ovr": float(proxy_rec["metrics"]["val_macro_auc_ovr"]),
            "full_val_macro_auc_ovr": float(full_rec["metrics"]["val_macro_auc_ovr"]),
            "full_val_macro_f1": float(full_rec["metrics"]["val_macro_f1"]),
            "full_val_macro_recall": float(full_rec["metrics"]["val_macro_recall"]),
            "full_best_epoch": int(full_rec["metrics"]["best_epoch"]),
            "full_overfit_gap_final": float(full_rec["metrics"]["overfit_gap_final"]),
            "proxy_runtime_seconds": proxy_rec["runtime_seconds"],
            "full_runtime_seconds": full_rec["runtime_seconds"],
            "early_proxy_auc": {
                str(epoch): max(item["val_macro_auc_ovr"] for item in proxy_rec["history"][:epoch])
                for epoch in early_epochs
            },
        })

    proxy_aucs = [row["proxy_val_macro_auc_ovr"] for row in rows]
    full_aucs = [row["full_val_macro_auc_ovr"] for row in rows]
    rho, rho_p = spearmanr(proxy_aucs, full_aucs)
    tau, tau_p = kendalltau(proxy_aucs, full_aucs)
    proxy_top = set(np.argsort(-np.asarray(proxy_aucs), kind="stable")[:args.top_k])
    full_top = set(np.argsort(-np.asarray(full_aucs), kind="stable")[:args.top_k])
    early_rankings = {}
    for epoch in early_epochs:
        early = np.array([row["early_proxy_auc"][str(epoch)] for row in rows])
        early_top = set(np.argsort(-early, kind="stable")[:args.top_k])
        early_rho = spearmanr(early, proxy_aucs).statistic
        early_rankings[str(epoch)] = {
            "spearman_vs_proxy": float(early_rho) if np.isfinite(early_rho) else None,
            "top_k_recall_vs_proxy": len(early_top & proxy_top) / args.top_k,
        }

    summary = {
        "protocol": settings["protocol"],
        "settings": {
            "optimizer": settings["optimizer_name"],
            "weight_decay": settings["weight_decay"],
            "scheduler": settings["scheduler_name"],
            "augmentation": settings["augmentation"],
            "loss": loss_config_name(settings["loss_name"]),
            "preload_gpu": settings["preload_gpu"],
            "amp": settings["use_amp"],
        },
        "seed": args.seed,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "top_k_recall": len(proxy_top & full_top) / args.top_k,
        "early_proxy_rankings": early_rankings,
        "num_models": len(rows),
        "proxy_epochs": args.proxy_epochs,
        "full_epochs": args.full_epochs,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_pvalue": float(rho_p) if np.isfinite(rho_p) else None,
        "kendall_tau": float(tau) if np.isfinite(tau) else None,
        "kendall_pvalue": float(tau_p) if np.isfinite(tau_p) else None,
        "mean_full_val_auc": float(np.mean(full_aucs)),
        "mean_full_val_macro_f1": float(np.mean([row["full_val_macro_f1"] for row in rows])),
        "mean_full_val_macro_recall": float(np.mean([row["full_val_macro_recall"] for row in rows])),
        "mean_full_best_epoch": float(np.mean([row["full_best_epoch"] for row in rows])),
        "mean_full_overfit_gap_final": float(np.mean([row["full_overfit_gap_final"] for row in rows])),
        "models": rows,
    }
    save_json(args.output_dir / "correlation_summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "models"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
