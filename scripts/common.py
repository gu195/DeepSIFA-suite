"""Shared command-line helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    """Resolve a CLI path without relying on a developer-specific directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def choose_device(requested: str) -> torch.device:
    """Select CUDA when available unless the caller requests a fixed device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible data handling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
