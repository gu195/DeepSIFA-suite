"""Convert labeled variable-length traces into MonoXtract NPZ inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from monoxtract.data import preprocess_trace, read_numeric_trace
from scripts.common import resolve_path


SUPPORTED_SUFFIXES = {".txt", ".csv", ".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good-dir", required=True, help="Directory of valid traces")
    parser.add_argument("--bad-dir", required=True, help="Directory of invalid traces")
    parser.add_argument("--output-dir", required=True, help="Output NPZ directory")
    parser.add_argument("--manifest", required=True, help="Output labels CSV")
    parser.add_argument(
        "--column",
        type=int,
        default=None,
        help="Zero-based value column for multi-column text/CSV files; default: last",
    )
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=1.0)
    return parser.parse_args()


def iter_inputs(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def convert_group(
    input_dir: Path,
    output_dir: Path,
    label: int,
    column: int,
    length: int,
    sigma: float,
) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for source in iter_inputs(input_dir):
        values = read_numeric_trace(source, column=column)
        processed = preprocess_trace(values, output_length=length, sigma=sigma)
        output_name = f"{source.stem}.npz"
        np.savez(output_dir / output_name, data=processed)
        rows.append((output_name, label))
    return rows


def main() -> None:
    args = parse_args()
    good_dir = resolve_path(args.good_dir)
    bad_dir = resolve_path(args.bad_dir)
    output_dir = resolve_path(args.output_dir)
    manifest = resolve_path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = convert_group(
        good_dir,
        output_dir,
        label=1,
        column=args.column,
        length=args.length,
        sigma=args.sigma,
    )
    rows.extend(
        convert_group(
            bad_dir,
            output_dir,
            label=0,
            column=args.column,
            length=args.length,
            sigma=args.sigma,
        )
    )
    if not rows:
        raise RuntimeError("No supported trace files were found")
    table = pd.DataFrame(rows, columns=["file_name", "label"])
    table.to_csv(manifest, index=False)
    print(
        f"Wrote {len(table)} traces to {output_dir} "
        f"(valid={(table.label == 1).sum()}, invalid={(table.label == 0).sum()})"
    )


if __name__ == "__main__":
    main()
