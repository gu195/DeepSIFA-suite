"""Train MonoXtract with the deposited five-fold MLKL development split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from monoxtract.data import TraceDataset
from monoxtract.metrics import classification_metrics
from monoxtract.model import build_model
from scripts.common import choose_device, resolve_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(REPOSITORY_ROOT / "data/mlkl/train/traces"),
    )
    parser.add_argument(
        "--split-dir",
        default=str(REPOSITORY_ROOT / "data/mlkl/train/splits"),
    )
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--tc-depth", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, Dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels = []
    predictions = []
    probabilities = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for traces, batch_labels, _ in tqdm(loader, leave=False):
            traces = traces.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.long)
            logits = model(traces)
            loss = criterion(logits, batch_labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * traces.shape[0]
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(batch_probabilities.detach().cpu().tolist())

    metrics = classification_metrics(labels, predictions, probabilities)
    return total_loss / len(loader.dataset), metrics


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    minimum_learning_rate: float,
    initial_learning_rate: float,
):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=minimum_learning_rate,
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=minimum_learning_rate / initial_learning_rate,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=minimum_learning_rate,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train_fold(args: argparse.Namespace, fold: int, device: torch.device) -> None:
    data_dir = resolve_path(args.data_dir)
    split_dir = resolve_path(args.split_dir)
    fold_dir = resolve_path(args.output_dir) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TraceDataset(
        data_dir, split_dir / f"train_fold{fold}.csv"
    )
    val_dataset = TraceDataset(data_dir, split_dir / f"val_fold{fold}.csv")
    generator = torch.Generator().manual_seed(args.seed + fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(
        tc_depth=args.tc_depth, drop_ratio=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        minimum_learning_rate=args.minimum_learning_rate,
        initial_learning_rate=args.learning_rate,
    )

    history = []
    best_accuracy = -1.0
    checkpoint_path = fold_dir / "best_acc.pth"

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "tc_depth": args.tc_depth,
                    "validation_metrics": val_metrics,
                    "configuration": vars(args),
                },
                checkpoint_path,
            )
        print(
            f"Fold {fold} epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

    with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_history(history, fold_dir / "training_curves.png")


def plot_history(history, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation accuracy",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    for fold in args.folds:
        if fold not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Fold must be in 1..5, received {fold}")
        train_fold(args, fold, device)


if __name__ == "__main__":
    main()
