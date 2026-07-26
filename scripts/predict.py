"""Predict valid-trace probabilities for unlabeled processed NPZ files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from monoxtract.data import UnlabeledTraceDataset
from monoxtract.model import load_model_from_checkpoint
from scripts.common import choose_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPOSITORY_ROOT / "checkpoints/best_acc.pth"),
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-csv", default="outputs/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dataset = UnlabeledTraceDataset(resolve_path(args.data_dir))
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
    with torch.no_grad():
        for traces, file_names in tqdm(loader):
            logits = model(traces.to(device=device, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            for name, prediction, valid_probability in zip(
                file_names,
                predictions.cpu().tolist(),
                probabilities[:, 1].cpu().tolist(),
            ):
                rows.append(
                    {
                        "file_name": name,
                        "prediction": prediction,
                        "valid_probability": valid_probability,
                    }
                )

    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
