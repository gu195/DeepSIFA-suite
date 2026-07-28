"""Generate temporal attention and attribution maps for MonoXtract.

The trained seven-TC checkpoint contains three SAC blocks followed by seven
Transformer blocks (``blocks.0`` through ``blocks.9``).  The current model
factory constructs additional blocks, so this script infers the trained depth
from the checkpoint, truncates the model, and then loads weights strictly.

Two complementary explanations are produced:

1. First-token attention rollout: the actual self-attention matrices from all
   trained Transformer blocks are averaged across heads and rolled out across
   layers.  MonoXtract's current forward pass does not insert ``cls_token``;
   the classifier reads token 0, so row 0 of the rollout is visualized.
2. Integrated Gradients: absolute, input-level attribution for the predicted
   class, projected directly onto all 1024 input samples.

Attention weights are not treated as proof of causal importance.  A small
occlusion check compares the effect of replacing the most- and least-attributed
windows by linear interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.vit import Attention, vit_base_patch16_224  # noqa: E402


@dataclass
class Sample:
    name: str
    label: int
    trace: np.ndarray
    probability_valid: float
    prediction: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MonoXtract attention rollout and input attributions."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "best_acc.pth",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV with file_name,label columns. Defaults to the labeled 337.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing 1024-point NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "attention_results",
    )
    parser.add_argument("--examples-per-class", type=int, default=3)
    parser.add_argument("--ig-steps", type=int, default=48)
    parser.add_argument("--occlusion-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=None,
        help=(
            "Optional independent transition annotations with columns "
            "file_name,index. Multiple rows per file are allowed."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def discover_dataset(
    csv_path: Optional[Path], data_dir: Optional[Path]
) -> Tuple[Path, Path]:
    if csv_path is None:
        matches = sorted(WORKSPACE_ROOT.rglob("337.csv"))
        if not matches:
            raise FileNotFoundError(
                "Could not find 337.csv; pass --csv and --data-dir explicitly."
            )
        csv_path = matches[0]
    csv_path = csv_path.resolve()

    if data_dir is None:
        candidates = [
            item
            for item in csv_path.parent.iterdir()
            if item.is_dir() and "1024" in item.name
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a 1024-point data directory beside {csv_path}."
            )
        data_dir = sorted(candidates)[0]
    return csv_path, data_dir.resolve()


def safe_load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state dictionary in {path}, got {type(obj)!r}.")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in obj.items()
        if torch.is_tensor(value)
    }
    return state


def trained_block_count(state: Dict[str, torch.Tensor]) -> int:
    indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if not indices:
        raise ValueError("Checkpoint has no blocks.* parameters.")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(f"Checkpoint block indices are not contiguous: {sorted(indices)}")
    return max(indices) + 1


def build_model(checkpoint: Path, device: torch.device) -> nn.Module:
    state = safe_load_state_dict(checkpoint)
    n_blocks = trained_block_count(state)
    model = vit_base_patch16_224(num_classes=2)
    blocks = list(model.blocks.children())
    if n_blocks > len(blocks):
        raise ValueError(
            f"Checkpoint needs {n_blocks} blocks, but model factory has {len(blocks)}."
        )
    model.blocks = nn.Sequential(*blocks[:n_blocks])
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def read_rows(csv_path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["file_name"], int(row["label"])))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    return rows


def load_trace(data_dir: Path, name: str) -> np.ndarray:
    with np.load(data_dir / name) as archive:
        trace = np.asarray(archive["data"], dtype=np.float32).reshape(-1)
    if trace.size != 1024:
        raise ValueError(f"{name} has {trace.size} samples; expected 1024.")
    return trace


def predict_dataset(
    model: nn.Module,
    rows: Sequence[Tuple[str, int]],
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> List[Sample]:
    results: List[Sample] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            traces = np.stack([load_trace(data_dir, name) for name, _ in batch_rows])
            inputs = torch.from_numpy(traces[:, None, :]).to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            for (name, label), trace, probability, prediction in zip(
                batch_rows, traces, probabilities, predictions
            ):
                results.append(
                    Sample(
                        name=name,
                        label=label,
                        trace=trace,
                        probability_valid=float(probability),
                        prediction=int(prediction),
                    )
                )
    return results


def morphology_scores(trace: np.ndarray) -> Tuple[float, float]:
    """Return large-scale step strength and high-frequency roughness."""
    kernel = np.ones(33, dtype=np.float32) / 33.0
    smooth = np.convolve(trace, kernel, mode="same")
    large_scale_diff = np.abs(np.diff(smooth))
    step_strength = float(np.percentile(large_scale_diff, 99.0))
    residual = trace - smooth
    roughness = float(np.std(np.diff(residual)))
    return step_strength, roughness


def quantile_examples(
    samples: Sequence[Sample], label: int, count: int
) -> List[Sample]:
    if label == 1:
        pool = [
            sample
            for sample in samples
            if sample.label == 1
            and sample.prediction == 1
            and sample.probability_valid >= 0.9
        ]
        score_index = 0
    else:
        pool = [
            sample
            for sample in samples
            if sample.label == 0
            and sample.prediction == 0
            and sample.probability_valid <= 0.3
        ]
        score_index = 1
    if len(pool) < count:
        pool = [
            sample
            for sample in samples
            if sample.label == label and sample.prediction == label
        ]
    ranked = sorted(pool, key=lambda item: morphology_scores(item.trace)[score_index])
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} correct examples are available for class {label}.")
    positions = np.linspace(0, len(ranked) - 1, count).round().astype(int)
    return [ranked[index] for index in positions]


class AttentionCollector:
    """Collect self-attention tensors from the standard Transformer blocks."""

    def __init__(self, model: nn.Module):
        self.values: List[torch.Tensor] = []
        self.handles = []
        for block in model.blocks:
            if hasattr(block, "attn") and isinstance(block.attn, Attention):
                self.handles.append(
                    block.attn.attn_drop.register_forward_hook(self._capture)
                )

    def _capture(self, _module, _inputs, output):
        self.values.append(output.detach().cpu())

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attention_rollout(
    model: nn.Module,
    collector: AttentionCollector,
    trace: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    collector.clear()
    inputs = torch.from_numpy(trace[None, None, :]).to(device)
    with torch.no_grad():
        model(inputs)
    if not collector.values:
        raise RuntimeError("No Transformer attention matrices were captured.")

    joint: Optional[torch.Tensor] = None
    for attention in collector.values:
        matrix = attention[0].mean(dim=0)
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
        matrix = matrix + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        joint = matrix if joint is None else matrix @ joint
    assert joint is not None
    token_scores = joint[0].numpy()
    token_scores = normalize_importance(token_scores)
    input_positions = np.linspace(0, token_scores.size - 1, trace.size)
    input_scores = np.interp(
        input_positions, np.arange(token_scores.size), token_scores
    )
    return normalize_importance(input_scores), len(collector.values)


def normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.nanmin(values)
    maximum = np.nanmax(values)
    if maximum > 0:
        values = values / maximum
    return values.astype(np.float32)


def integrated_gradients(
    model: nn.Module,
    trace: np.ndarray,
    target: int,
    steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    input_tensor = torch.from_numpy(trace[None, None, :]).to(device)
    baseline = torch.zeros_like(input_tensor)
    total_gradient = torch.zeros_like(input_tensor)

    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=device):
        interpolated = (baseline + alpha * (input_tensor - baseline)).detach()
        interpolated.requires_grad_(True)
        logit = model(interpolated)[0, target]
        gradient = torch.autograd.grad(logit, interpolated)[0]
        weight = 0.5 if alpha.item() in (0.0, 1.0) else 1.0
        total_gradient += weight * gradient

    average_gradient = total_gradient / steps
    attribution = (input_tensor - baseline) * average_gradient
    signed = attribution.detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        output_delta = (
            model(input_tensor)[0, target] - model(baseline)[0, target]
        ).item()
    completeness_error = float(abs(signed.sum() - output_delta))
    return normalize_importance(np.abs(signed)), completeness_error


def window_sums(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width, dtype=np.float32), mode="valid")


def select_occlusion_windows(
    importance: np.ndarray, width: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    sums = window_sums(importance, width)
    high_start = int(np.argmax(sums))
    exclusion_start = max(0, high_start - width)
    exclusion_end = min(sums.size, high_start + width)
    allowed = np.ones(sums.size, dtype=bool)
    allowed[exclusion_start:exclusion_end] = False
    if allowed.any():
        allowed_indices = np.flatnonzero(allowed)
        low_start = int(allowed_indices[np.argmin(sums[allowed])])
    else:
        low_start = int(np.argmin(sums))
    return (high_start, high_start + width), (low_start, low_start + width)


def interpolate_window(trace: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    start, end = window
    result = trace.copy()
    left_index = max(0, start - 1)
    right_index = min(trace.size - 1, end)
    result[start:end] = np.linspace(
        trace[left_index], trace[right_index], end - start + 2, dtype=np.float32
    )[1:-1]
    return result


def target_probability(
    model: nn.Module, trace: np.ndarray, target: int, device: torch.device
) -> float:
    inputs = torch.from_numpy(trace[None, None, :]).to(device)
    with torch.no_grad():
        return float(torch.softmax(model(inputs), dim=1)[0, target].item())


def read_transitions(path: Optional[Path]) -> Dict[str, List[int]]:
    transitions: Dict[str, List[int]] = {}
    if path is None:
        return transitions
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            transitions.setdefault(row["file_name"], []).append(int(row["index"]))
    return transitions


def colored_trace(
    axis: plt.Axes,
    trace: np.ndarray,
    importance: np.ndarray,
    cmap: str = "inferno",
) -> None:
    x = np.arange(trace.size)
    points = np.column_stack((x, trace)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=Normalize(0, 1),
        linewidth=1.4,
    )
    collection.set_array((importance[:-1] + importance[1:]) / 2)
    axis.add_collection(collection)
    axis.set_xlim(0, trace.size - 1)
    margin = max(0.03, 0.05 * float(np.ptp(trace)))
    axis.set_ylim(float(trace.min() - margin), float(trace.max() + margin))


def plot_individual(
    output_path: Path,
    sample: Sample,
    attention: np.ndarray,
    attribution: np.ndarray,
    high_window: Tuple[int, int],
    low_window: Tuple[int, int],
    transitions: Sequence[int],
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": (3.3, 0.8, 0.8), "hspace": 0.14},
    )
    colored_trace(axes[0], sample.trace, attribution)
    axes[0].axvspan(*high_window, color="#ef8a62", alpha=0.16, label="top IG window")
    axes[0].axvspan(*low_window, color="#67a9cf", alpha=0.13, label="low IG window")
    for index, transition in enumerate(transitions):
        axes[0].axvline(
            transition,
            color="#2ca25f",
            linestyle="--",
            linewidth=1.1,
            label="independent transition" if index == 0 else None,
        )
    axes[0].set_ylabel("Normalized intensity")
    axes[0].set_title(
        f"{sample.name} | manual label={sample.label} | "
        f"P(valid)={sample.probability_valid:.3f}"
    )
    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)

    axes[1].imshow(
        attention[None, :],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=(0, sample.trace.size - 1, 0, 1),
    )
    axes[1].set_yticks([])
    axes[1].set_ylabel("Attention\nrollout", rotation=0, labelpad=42, va="center")

    axes[2].imshow(
        attribution[None, :],
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
        extent=(0, sample.trace.size - 1, 0, 1),
    )
    axes[2].set_yticks([])
    axes[2].set_ylabel("|IG|\nimportance", rotation=0, labelpad=42, va="center")
    axes[2].set_xlabel("Resampled trace index")
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def plot_montage(
    output_path: Path,
    records: Sequence[Tuple[Sample, np.ndarray, np.ndarray]],
) -> None:
    columns = 3
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows), squeeze=False)
    for axis, (sample, attention, attribution) in zip(axes.flat, records):
        colored_trace(axis, sample.trace, attribution)
        attention_scaled = sample.trace.min() + attention * 0.12 * max(
            float(np.ptp(sample.trace)), 0.1
        )
        axis.fill_between(
            np.arange(sample.trace.size),
            sample.trace.min(),
            attention_scaled,
            color="#2b8cbe",
            alpha=0.30,
            linewidth=0,
            label="attention rollout",
        )
        label_name = "valid" if sample.label == 1 else "invalid"
        axis.set_title(
            f"{sample.name.replace('.npz', '')}\n"
            f"manual={label_name}, P(valid)={sample.probability_valid:.3f}",
            fontsize=10,
        )
        axis.set_xlabel("Trace index")
        axis.set_ylabel("Normalized intensity")
    for axis in axes.flat[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        "MonoXtract temporal explanations\n"
        "trace color = |Integrated Gradients|; blue band = first-token attention rollout",
        y=1.01,
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def classification_metrics(samples: Sequence[Sample]) -> Dict[str, float]:
    labels = np.asarray([sample.label for sample in samples])
    predictions = np.asarray([sample.prediction for sample in samples])
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "n": int(labels.size),
        "accuracy": float((tp + tn) / labels.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    csv_path, data_dir = discover_dataset(args.csv, args.data_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(args.checkpoint.resolve(), device)
    rows = read_rows(csv_path)
    samples = predict_dataset(model, rows, data_dir, args.batch_size, device)
    selected = quantile_examples(samples, 1, args.examples_per_class)
    selected += quantile_examples(samples, 0, args.examples_per_class)
    transitions = read_transitions(args.transition_csv)

    prediction_rows = [
        {
            "file_name": sample.name,
            "label": sample.label,
            "prediction": sample.prediction,
            "probability_valid": f"{sample.probability_valid:.8f}",
        }
        for sample in samples
    ]
    write_csv(output_dir / "predictions.csv", prediction_rows)

    collector = AttentionCollector(model)
    montage_records = []
    result_rows = []
    attention_layer_count = 0
    for sample in selected:
        attention, attention_layer_count = attention_rollout(
            model, collector, sample.trace, device
        )
        attribution, completeness_error = integrated_gradients(
            model,
            sample.trace,
            sample.prediction,
            args.ig_steps,
            device,
        )
        high_window, low_window = select_occlusion_windows(
            attribution, args.occlusion_width
        )
        target = sample.prediction
        original_probability = target_probability(model, sample.trace, target, device)
        high_probability = target_probability(
            model, interpolate_window(sample.trace, high_window), target, device
        )
        low_probability = target_probability(
            model, interpolate_window(sample.trace, low_window), target, device
        )
        correlation = float(np.corrcoef(attention, attribution)[0, 1])
        step_strength, roughness = morphology_scores(sample.trace)

        stem = sample.name.replace(".npz", "")
        np.savez_compressed(
            output_dir / f"{stem}_explanations.npz",
            trace=sample.trace,
            attention_rollout=attention,
            integrated_gradients_abs=attribution,
            high_window=np.asarray(high_window),
            low_window=np.asarray(low_window),
        )
        plot_individual(
            output_dir / f"{stem}_attention_ig.png",
            sample,
            attention,
            attribution,
            high_window,
            low_window,
            transitions.get(sample.name, []),
        )
        montage_records.append((sample, attention, attribution))
        result_rows.append(
            {
                "file_name": sample.name,
                "manual_label": sample.label,
                "prediction": sample.prediction,
                "probability_valid": f"{sample.probability_valid:.8f}",
                "attention_ig_correlation": f"{correlation:.8f}",
                "ig_completeness_abs_error": f"{completeness_error:.8f}",
                "top_window_start": high_window[0],
                "top_window_end": high_window[1],
                "low_window_start": low_window[0],
                "low_window_end": low_window[1],
                "target_probability_original": f"{original_probability:.8f}",
                "target_probability_top_window_occluded": f"{high_probability:.8f}",
                "target_probability_low_window_occluded": f"{low_probability:.8f}",
                "target_probability_drop_top": f"{original_probability - high_probability:.8f}",
                "target_probability_drop_low": f"{original_probability - low_probability:.8f}",
                "step_strength": f"{step_strength:.8f}",
                "roughness": f"{roughness:.8f}",
                "independent_transition_count": len(transitions.get(sample.name, [])),
            }
        )
    collector.close()

    write_csv(output_dir / "selected_examples_and_occlusion.csv", result_rows)
    plot_montage(
        output_dir / "figure_attention_attribution_examples.png", montage_records
    )

    checkpoint_state = safe_load_state_dict(args.checkpoint.resolve())
    n_blocks = trained_block_count(checkpoint_state)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_csv": str(csv_path),
        "data_dir": str(data_dir),
        "device": str(device),
        "trained_total_blocks": n_blocks,
        "trained_sac_blocks": n_blocks - attention_layer_count,
        "trained_transformer_blocks": attention_layer_count,
        "classification_metrics": classification_metrics(samples),
        "selected_examples": [sample.name for sample in selected],
        "integrated_gradients_steps": args.ig_steps,
        "occlusion_width": args.occlusion_width,
        "transition_annotations_supplied": bool(args.transition_csv),
        "interpretation_note": (
            "The current forward pass does not insert cls_token or pos_embed. "
            "Classification uses token 0; attention rollout therefore starts from "
            "the first ordinary token. Maps explain valid/invalid classification "
            "and are not direct state-transition predictions."
        ),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
