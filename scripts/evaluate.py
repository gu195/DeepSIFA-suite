"""Evaluate a MonoXtract checkpoint on a labeled trace manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from monoxtract.data import TraceDataset
from monoxtract.metrics import classification_metrics
from monoxtract.model import load_model_from_checkpoint
from scripts.common import choose_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPOSITORY_ROOT / "checkpoints/best_acc.pth"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPOSITORY_ROOT / "data/mlkl/validation/traces"),
    )
    parser.add_argument(
        "--labels",
        default=str(REPOSITORY_ROOT / "data/mlkl/validation/labels.csv"),
    )
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dataset = TraceDataset(resolve_path(args.data_dir), resolve_path(args.labels))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = load_model_from_checkpoint(
        resolve_path(args.checkpoint), device=device, strict=True
    )
    model.eval()

    rows = []
    labels = []
    predictions = []
    valid_probabilities = []
    with torch.no_grad():
        for traces, batch_labels, file_names in tqdm(loader):
            traces = traces.to(device=device, dtype=torch.float32)
            logits = model(traces)
            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = probabilities.argmax(dim=1)
            for name, label, prediction, probability in zip(
                file_names,
                batch_labels.tolist(),
                batch_predictions.cpu().tolist(),
                probabilities[:, 1].cpu().tolist(),
            ):
                rows.append(
                    {
                        "file_name": name,
                        "label": label,
                        "prediction": prediction,
                        "valid_probability": probability,
                    }
                )
                labels.append(label)
                predictions.append(prediction)
                valid_probabilities.append(probability)

    summary = classification_metrics(labels, predictions, valid_probabilities)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_confusion_matrix(summary, output_dir / "confusion_matrix.png")
    plot_roc_pr(labels, valid_probabilities, output_dir)
    print(json.dumps(summary, indent=2))


def plot_confusion_matrix(summary: dict, output_path: Path) -> None:
    matrix = np.asarray(summary["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Invalid", "Valid"])
    axis.set_yticks([0, 1], labels=["Invalid", "Valid"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Manual label")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_roc_pr(labels, probabilities, output_dir: Path) -> None:
    if np.unique(labels).size < 2:
        return
    fpr, tpr, _ = sk_metrics.roc_curve(labels, probabilities)
    precision, recall, _ = sk_metrics.precision_recall_curve(
        labels, probabilities
    )
    roc_auc = sk_metrics.roc_auc_score(labels, probabilities)
    average_precision = sk_metrics.average_precision_score(labels, probabilities)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(recall, precision, label=f"AP = {average_precision:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
