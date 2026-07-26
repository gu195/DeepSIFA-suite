# MonoXtract

MonoXtract is a one-dimensional convolution-Transformer model for classifying
single-molecule fluorescence-intensity traces as valid or invalid. This
repository contains a cleaned, English-language implementation of the model,
data preprocessing, training, validation, prediction, and model-interpretation
workflows used in the study.

## Repository layout

```text
MonoXtract/
|-- checkpoints/
|   `-- best_acc.pth
|-- data/
|   `-- mlkl/
|       |-- train/
|       |   |-- traces/
|       |   `-- splits/
|       `-- validation/
|           |-- traces/
|           `-- labels.csv
|-- docs/
|   |-- DATASETS.md
|   `-- REPRODUCIBILITY.md
|-- monoxtract/
|   |-- data.py
|   |-- metrics.py
|   `-- model.py
|-- scripts/
|   |-- attention_analysis.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- prepare_dataset.py
|   |-- train.py
|   `-- verify_install.py
`-- requirements.txt
```

All public-facing comments, docstrings, file names, and instructions in this
release are in English. The command-line programs use paths supplied by the
user or paths resolved relative to the repository root; no developer-specific
absolute paths are required.

## Installation

Python 3.8.20 was used for the pinned environment.

```bash
conda create -n monoxtract python=3.8.20
conda activate monoxtract
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.verify_install
```

PyTorch installation can depend on the local CUDA driver. The pinned
`torch==2.4.1` entry is sufficient for CPU execution. Users who require a
specific CUDA wheel may install the corresponding PyTorch 2.4.1 build first
and then install the remaining packages with:

```bash
python -m pip install -r requirements.txt --no-deps
```

## Included MLKL data

The repository currently includes the processed MLKL traces that are available
for model development:

| Dataset role | Valid (`label=1`) | Invalid (`label=0`) | Total |
|---|---:|---:|---:|
| Training/development set | 218 | 117 | 335 |
| Independent validation set | 190 | 147 | 337 |

The 335 development traces are accompanied by five reproducible stratified
fold definitions. Each fold contains 268 training traces and 67 internal
validation traces. The separate 337-trace MLKL set is used only for independent
validation/evaluation.

The deposited files are normalized, linearly resampled to 1,024 points, and
smoothed with a one-dimensional Gaussian filter (`sigma=1`). See
[`docs/DATASETS.md`](docs/DATASETS.md) for the directory structure, labels, and
data limitations.

## Train the five folds

From the repository root:

```bash
python -m scripts.train
```

The default command uses:

- three Spatial Aware Cells (fixed by the model definition);
- seven Transformer Cells;
- batch size 32;
- AdamW with learning rate `1e-4` and weight decay `0.05`;
- 200 epochs and a five-epoch linear warm-up followed by cosine decay.

SAC and TC counts were compared in a targeted architectural sensitivity
analysis. Learning rate, batch size, and dropout were selected empirically and
were not subjected to exhaustive automated optimization.

To run one fold or change an output location:

```bash
python -m scripts.train --folds 1 --output-dir outputs/fold1
```

All data and output paths can be overridden through command-line options:

```bash
python -m scripts.train --help
```

## Evaluate the released checkpoint

```bash
python -m scripts.evaluate \
  --checkpoint checkpoints/best_acc.pth \
  --data-dir data/mlkl/validation/traces \
  --labels data/mlkl/validation/labels.csv \
  --output-dir outputs/validation
```

The evaluation command writes `metrics.json`, `predictions.csv`, a confusion
matrix, an ROC curve, and a precision-recall curve.

## Predict unlabeled traces

```bash
python -m scripts.predict \
  --checkpoint checkpoints/best_acc.pth \
  --data-dir path/to/processed_npz \
  --output-csv outputs/predictions.csv
```

Each `.npz` file must contain a one-dimensional array named `data` with 1,024
values.

## Prepare variable-length traces

Raw text/CSV traces can be converted to the model input representation with:

```bash
python -m scripts.prepare_dataset \
  --good-dir path/to/valid_traces \
  --bad-dir path/to/invalid_traces \
  --output-dir data/custom/traces \
  --manifest data/custom/labels.csv
```

The script applies min-max normalization, linear resampling to 1,024 points,
and Gaussian smoothing with `sigma=1`. It does not use zero padding,
truncation, or sliding windows.

## Attention and feature-attribution analysis

The reproducible attention-rollout and Integrated Gradients analysis is
available through:

```bash
python -m scripts.attention_analysis --help
```

These visualizations explain which temporal regions contribute to the
valid/invalid classification. They are not direct molecular-state or
state-transition annotations.

## Reproducibility scope

This release contains the MLKL development and independent validation data
listed above, the released checkpoint, and executable scripts for
preprocessing, training, validation, prediction, and interpretation. Additional
experimental evaluation datasets reported in the manuscript are not yet
included in this repository and should not be inferred to be present. This
scope is stated explicitly in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
