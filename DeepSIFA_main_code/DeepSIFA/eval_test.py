"""Evaluate the released MonoXtract checkpoint on labeled fluorescence traces.

This script remains compatible with the GUI arguments ``--data_dir`` and
``--weight_path`` while providing deterministic evaluation, strict checkpoint
loading, automatic processed-data discovery, and reproducibility metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vit import vit_base_patch16_224


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "logs_test"
DEFAULT_VALIDATION_DIR = REPOSITORY_ROOT / "DeepSIFA_automation" / "data" / "mlkl" / "test" / "v1"
PREFERRED_PROCESSED_DIRS = (
    "processing_data",
    "归一化插值后npz_1024_高斯平滑1",
)


class TraceDataset(Dataset):
    """Load one-dimensional traces listed in a two-column CSV manifest."""

    def __init__(self, processed_dir: Path, manifest: Path) -> None:
        table = pd.read_csv(manifest)
        required_columns = {"file_name", "label"}
        missing_columns = required_columns.difference(table.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest} is missing columns: {sorted(missing_columns)}"
            )
        self.processed_dir = processed_dir
        self.file_names = table["file_name"].astype(str).tolist()
        self.labels = table["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int):
        file_name = self.file_names[index]
        npz_path = self.processed_dir / file_name
        with np.load(npz_path) as archive:
            if "data" not in archive:
                raise KeyError(f"NPZ file has no 'data' array: {npz_path}")
            trace = np.asarray(archive["data"], dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(trace), self.labels[index], file_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight_path",
        "--checkpoint",
        dest="weight_path",
        default=str(SCRIPT_DIR / "checkpoints" / "best_acc.pth"),
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        default=str(DEFAULT_VALIDATION_DIR),
        help="Dataset root containing a CSV manifest and processed NPZ files.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional manifest CSV. By default, a CSV in --data_dir is used.",
    )
    parser.add_argument(
        "--processed_dir",
        "--processed-dir",
        dest="processed_dir",
        default=None,
        help="Optional directory containing the processed NPZ files.",
    )
    parser.add_argument(
        "--results_dir",
        "--output-dir",
        dest="results_dir",
        default=str(DEFAULT_RESULTS_DIR),
    )
    parser.add_argument("--bt_size", "--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--tc-depth", type=int, default=7)
    return parser.parse_args()


def resolve_manifest(data_dir: Path, labels: Optional[str]) -> Path:
    if labels:
        manifest = Path(labels).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        return manifest

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV manifest found in {data_dir}")
    if len(csv_files) == 1:
        return csv_files[0]

    preferred = [path for path in csv_files if path.stem.isdigit()]
    if len(preferred) == 1:
        return preferred[0]
    names = ", ".join(path.name for path in csv_files)
    raise ValueError(f"Multiple CSV manifests found ({names}); specify --labels.")


def manifest_file_names(manifest: Path) -> List[str]:
    table = pd.read_csv(manifest)
    if "file_name" not in table:
        raise ValueError(f"Manifest has no 'file_name' column: {manifest}")
    return table["file_name"].astype(str).tolist()


def missing_files(directory: Path, file_names: Sequence[str]) -> List[str]:
    return [name for name in file_names if not (directory / name).is_file()]


def resolve_processed_dir(
    data_dir: Path, processed_dir: Optional[str], file_names: Sequence[str]
) -> Path:
    if processed_dir:
        candidate = Path(processed_dir).expanduser().resolve()
        missing = missing_files(candidate, file_names)
        if missing:
            raise FileNotFoundError(
                f"Processed directory {candidate} is missing {len(missing)} manifest files; "
                f"first missing file: {missing[0]}"
            )
        return candidate

    candidates: List[Path] = []
    for name in PREFERRED_PROCESSED_DIRS:
        candidate = data_dir / name
        if candidate.is_dir():
            candidates.append(candidate)
    candidates.extend(
        path
        for path in sorted(data_dir.iterdir())
        if path.is_dir() and path not in candidates
    )
    if all((data_dir / name).is_file() for name in file_names):
        candidates.append(data_dir)

    diagnostics = []
    for candidate in candidates:
        missing = missing_files(candidate, file_names)
        diagnostics.append(f"{candidate.name}: {len(missing)} missing")
        if not missing:
            return candidate
    details = "; ".join(diagnostics) if diagnostics else "no candidate directories"
    raise FileNotFoundError(
        f"Could not locate all {len(file_names)} processed NPZ files under {data_dir} "
        f"({details}). Use --processed_dir to specify their location."
    )


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def read_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    return state


def checkpoint_block_count(state: Dict[str, torch.Tensor]) -> int:
    indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    return max(indices) + 1 if indices else 0


def load_model(checkpoint_path: Path, tc_depth: int, device: torch.device):
    state = read_checkpoint(checkpoint_path)
    model = vit_base_patch16_224(num_classes=2, tc_depth=tc_depth)
    expected_blocks = 3 + tc_depth
    trained_blocks = checkpoint_block_count(state)
    if trained_blocks != expected_blocks:
        raise RuntimeError(
            f"Checkpoint contains {trained_blocks} total blocks, but the requested model "
            f"contains {expected_blocks} (3 SAC + {tc_depth} TC)."
        )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader: DataLoader, device: torch.device):
    rows = []
    labels: List[int] = []
    predictions: List[int] = []
    valid_probabilities: List[float] = []
    print("------------- evaluating -------------")
    with torch.no_grad():
        for traces, batch_labels, file_names in tqdm(loader):
            traces = traces.to(device=device, dtype=torch.float32)
            logits = model(traces)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
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
                        "name": name,
                        "label": int(label),
                        "score": float(probability),
                        "pre": int(prediction),
                    }
                )
                labels.append(int(label))
                predictions.append(int(prediction))
                valid_probabilities.append(float(probability))
    return rows, labels, predictions, valid_probabilities


def calculate_metrics(
    labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]
) -> Dict[str, object]:
    matrix = sk_metrics.confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    summary: Dict[str, object] = {
        "n": len(labels),
        "accuracy": sk_metrics.accuracy_score(labels, predictions),
        "precision": sk_metrics.precision_score(labels, predictions, zero_division=0),
        "recall": sk_metrics.recall_score(labels, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": sk_metrics.f1_score(labels, predictions, zero_division=0),
        "cohen_kappa": sk_metrics.cohen_kappa_score(labels, predictions),
        "roc_auc": sk_metrics.roc_auc_score(labels, probabilities),
        "average_precision": sk_metrics.average_precision_score(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return summary


def write_predictions(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "label", "score", "pre"))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(summary: Dict[str, object], output_path: Path) -> None:
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


def plot_roc_pr(
    labels: Sequence[int], probabilities: Sequence[float], output_dir: Path
) -> None:
    fpr, tpr, _ = sk_metrics.roc_curve(labels, probabilities)
    precision, recall, _ = sk_metrics.precision_recall_curve(labels, probabilities)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(fpr, tpr, label=f"AUC = {sk_metrics.roc_auc_score(labels, probabilities):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(
        recall,
        precision,
        label=f"AP = {sk_metrics.average_precision_score(labels, probabilities):.3f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close(figure)


def plot_score_distributions(probabilities: Sequence[float], output_path: Path) -> None:
    bin_sizes = (0.025, 0.0125, 0.00625)
    figure, axes = plt.subplots(3, 1, figsize=(10, 16))
    for axis, bin_size in zip(axes, bin_sizes):
        bins = np.arange(0, 1 + bin_size, bin_size)
        axis.hist(probabilities, bins=bins, edgecolor="black", alpha=0.7)
        axis.set_title(
            f"Score Distribution (Total: {len(probabilities)}, Bin Size: {bin_size:.4f})"
        )
        axis.set_xlabel("Predicted valid probability")
        axis.set_ylabel("Count")
        axis.set_xticks(np.arange(0, 1.1, 0.1))
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint_path = Path(args.weight_path).expanduser().resolve()
    output_dir = Path(args.results_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest = resolve_manifest(data_dir, args.labels)
    file_names = manifest_file_names(manifest)
    processed_dir = resolve_processed_dir(data_dir, args.processed_dir, file_names)
    device = choose_device(args.device)

    print(f"Dataset root: {data_dir}")
    print(f"Manifest: {manifest}")
    print(f"Processed traces: {processed_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: 3 SAC + {args.tc_depth} TC")
    print(f"Device: {device}")

    dataset = TraceDataset(processed_dir, manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(checkpoint_path, args.tc_depth, device)
    rows, labels, predictions, probabilities = evaluate(model, loader, device)
    summary = calculate_metrics(labels, predictions, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(rows, output_dir / "score.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_confusion_matrix(summary, output_dir / "confusion_matrix.png")
    plot_roc_pr(labels, probabilities, output_dir)
    plot_score_distributions(probabilities, output_dir / "score_distribution.png")

    print(json.dumps(summary, indent=2))
    print(f"Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
