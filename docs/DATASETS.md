# Deposited datasets

## Label definition

MonoXtract performs binary trace classification:

- `label=1` means a manually annotated valid fluorescence trace.
- `label=0` means a manually annotated invalid fluorescence trace.

The current deposited data contain MLKL traces only. Fine-grained manual
categories such as Static, Dynamic, Noisy, Impurity, and Multiple molecules are
not encoded in the available CSV files; the deposited manifests therefore
report only the binary valid/invalid labels.

## Development data

Location: `data/mlkl/train/`

- `traces/`: 335 processed `.npz` traces.
- `splits/`: five stratified fold definitions.
- Class composition: 218 valid and 117 invalid traces.

For each fold, `train_foldN.csv` contains 268 traces and `val_foldN.csv`
contains 67 traces. Every development trace appears exactly once in the
internal validation portion across the five folds.

## Independent validation data

Location: `data/mlkl/validation/`

- `traces/`: 337 processed `.npz` traces.
- `labels.csv`: one row per trace.
- Class composition: 190 valid and 147 invalid traces.

This dataset does not overlap with the 335-trace development set according to
the deposited file names.

## File format

Every processed trace is stored as:

```python
np.savez(output_path, data=processed_trace)
```

The `data` array has shape `(1024,)`. Each manifest contains:

```text
file_name,label
```

## Preprocessing

For an original trace with `N` points:

1. The fluorescence values are min-max normalized.
2. The original coordinate is mapped to `[0, 1]`.
3. Linear interpolation resamples the sequence to 1,024 equally spaced
   positions.
4. A one-dimensional Gaussian filter with `sigma=1` is applied.

Traces shorter than 1,024 points are upsampled and longer traces are
downsampled. No zero padding, direct truncation, or sliding-window segmentation
is used.

## Availability limitation

Only the MLKL development and independent validation datasets described above
are presently deposited. Other experimental datasets used for additional
manuscript evaluations are not included in this release. The README and
reviewer response should therefore describe the deposited scope precisely
rather than state that every manuscript dataset is already available.
