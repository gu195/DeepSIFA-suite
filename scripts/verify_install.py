"""Verify dependencies, model construction, and deposited dataset manifests."""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from monoxtract.data import validate_manifest
from monoxtract.model import build_model


PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
]


def main() -> None:
    print("Installed package versions:")
    for package in PACKAGES:
        print(f"  {package}=={metadata.version(package)}")

    model = build_model(tc_depth=7).cpu().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 1024))
    if tuple(output.shape) != (1, 2):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    print("Model forward pass: OK (1 x 2 logits)")

    train_dir = REPOSITORY_ROOT / "data/mlkl/train/traces"
    split_dir = REPOSITORY_ROOT / "data/mlkl/train/splits"
    for fold in range(1, 6):
        train_counts = validate_manifest(
            train_dir, split_dir / f"train_fold{fold}.csv"
        )
        val_counts = validate_manifest(
            train_dir, split_dir / f"val_fold{fold}.csv"
        )
        print(f"Fold {fold}: train={train_counts}, internal_val={val_counts}")

    validation_counts = validate_manifest(
        REPOSITORY_ROOT / "data/mlkl/validation/traces",
        REPOSITORY_ROOT / "data/mlkl/validation/labels.csv",
    )
    print(f"Independent validation: {validation_counts}")
    print("MonoXtract installation and deposited data: OK")


if __name__ == "__main__":
    main()
