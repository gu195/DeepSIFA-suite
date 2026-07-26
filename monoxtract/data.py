"""Dataset and trace-preprocessing utilities for MonoXtract."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset


TRACE_LENGTH = 1024


def load_npz_trace(path: Path) -> np.ndarray:
    """Load and validate a processed trace stored under the ``data`` key."""
    with np.load(path) as archive:
        if "data" not in archive:
            raise KeyError(f"{path} does not contain a 'data' array")
        trace = np.asarray(archive["data"], dtype=np.float32).reshape(-1)
    if trace.size != TRACE_LENGTH:
        raise ValueError(
            f"{path} contains {trace.size} points; expected {TRACE_LENGTH}"
        )
    if not np.isfinite(trace).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return trace


def preprocess_trace(
    values: Sequence[float],
    output_length: int = TRACE_LENGTH,
    sigma: float = 1.0,
) -> np.ndarray:
    """Normalize, linearly resample, and Gaussian-smooth one trace."""
    trace = np.asarray(values, dtype=np.float64).reshape(-1)
    if trace.size < 2:
        raise ValueError("A trace must contain at least two points")
    if not np.isfinite(trace).all():
        raise ValueError("The input trace contains NaN or infinite values")

    minimum = float(trace.min())
    maximum = float(trace.max())
    if maximum > minimum:
        trace = (trace - minimum) / (maximum - minimum)
    else:
        trace = np.zeros_like(trace)

    original_positions = np.linspace(0.0, 1.0, trace.size)
    output_positions = np.linspace(0.0, 1.0, output_length)
    trace = np.interp(output_positions, original_positions, trace)
    if sigma > 0:
        trace = gaussian_filter1d(trace, sigma=sigma)
    return trace.astype(np.float32)


def read_numeric_trace(path: Path, column: Optional[int] = None) -> np.ndarray:
    """Read a numeric text, CSV, NPY, or NPZ trace."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as archive:
            key = "data" if "data" in archive else archive.files[0]
            values = np.asarray(archive[key])
    elif suffix == ".npy":
        values = np.asarray(np.load(path))
    else:
        delimiter = "," if suffix == ".csv" else None
        values = np.genfromtxt(path, delimiter=delimiter)

    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError(f"No usable numeric sequence was found in {path}")
    if values.ndim > 1:
        selected_column = column if column is not None else values.shape[1] - 1
        values = values[:, selected_column]
    return np.asarray(values, dtype=np.float64).reshape(-1)


class TraceDataset(Dataset):
    """Load labeled MonoXtract traces from a two-column CSV manifest."""

    def __init__(self, data_dir: Path, labels_csv: Path):
        self.data_dir = Path(data_dir)
        self.labels_csv = Path(labels_csv)
        table = pd.read_csv(self.labels_csv)
        required = {"file_name", "label"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(
                f"{self.labels_csv} is missing columns: {sorted(missing)}"
            )
        self.file_names = table["file_name"].astype(str).tolist()
        self.labels = table["label"].astype(int).tolist()
        invalid_labels = sorted(set(self.labels).difference({0, 1}))
        if invalid_labels:
            raise ValueError(f"Only labels 0 and 1 are supported: {invalid_labels}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        file_name = self.file_names[index]
        trace = load_npz_trace(self.data_dir / file_name)
        tensor = torch.from_numpy(trace).unsqueeze(0)
        return tensor, self.labels[index], file_name


class UnlabeledTraceDataset(Dataset):
    """Load every processed NPZ trace in a directory."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.paths = sorted(self.data_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files were found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[index]
        trace = load_npz_trace(path)
        return torch.from_numpy(trace).unsqueeze(0), path.name


def validate_manifest(data_dir: Path, labels_csv: Path) -> dict:
    """Validate all rows in a manifest and return class counts."""
    dataset = TraceDataset(data_dir, labels_csv)
    for index in range(len(dataset)):
        dataset[index]
    labels = np.asarray(dataset.labels)
    return {
        "total": int(labels.size),
        "invalid": int((labels == 0).sum()),
        "valid": int((labels == 1).sum()),
    }
