# MonoXtract

MonoXtract extracts and classifies single-molecule fluorescence-intensity
traces with a hybrid convolution-Transformer network. This repository retains
the complete research directories used for inference/testing and for
training/validation; the source has not been collapsed into a smaller
reimplementation.

## Repository layout

```text
MonoXtract/
|-- DeepSIFA_main_code/   Complete main algorithm, test workflow, MATLAB code,
|                        example data, checkpoints, DLRunner, and analysis tools
|-- DeepSIFA_automation/  Complete five-fold training and validation workflow,
|                        MLKL datasets, checkpoints, logs, and auxiliary tools
|-- docs/                 Installation and reproducibility notes
|-- requirements.txt      Version-pinned Python dependencies
`-- .gitattributes        Git LFS rules for binary research artifacts
```

The source directory originally named `DeepSIFA自动化` is published as
`DeepSIFA_automation` for an English-language repository. Its internal files
and data are retained. Some legacy data folders and scripts keep their original
Chinese names because other scripts refer to those names.

## Clone the complete repository

Large binary research artifacts are tracked with Git LFS. Install Git LFS
before cloning:

```bash
git lfs install
git clone https://github.com/gu195/MonoXtract.git
cd MonoXtract
git lfs pull
```

A normal GitHub source-code ZIP contains LFS pointer files rather than every
large binary payload. Use `git clone` plus `git lfs pull`, or download the
corresponding assets from the GitHub Release.

## Installation

The released Python environment is based on Python 3.8.20:

```bash
conda create -n monoxtract python=3.8.20
conda activate monoxtract
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The preprocessing workflow calls MATLAB through MATLAB Engine for Python.
The development environment used MATLAB Engine R2021b. Install it from the
MATLAB installation directory:

```bash
cd "<matlabroot>/extern/engines/python"
python -m pip install .
```

MATLAB is not required for loading already processed `.npz` traces or for
running the neural-network evaluation scripts.

See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) for CUDA and optional
dependency notes.

## Included MLKL datasets

The complete automation directory contains the currently available MLKL
training and validation data:

| Role | Valid (`good`) | Invalid (`bad`) | Total |
|---|---:|---:|---:|
| Model training/development | 218 | 117 | 335 |
| Independent validation/evaluation | 190 | 147 | 337 |

The training files and five-fold CSV definitions are under:

```text
DeepSIFA_automation/data/mlkl/train/v1/
```

The 337-trace validation dataset is under:

```text
DeepSIFA_automation/data/mlkl/test/v1/
```

These are MLKL datasets. Additional independent experimental datasets
mentioned in the manuscript are not claimed to be included unless they are
explicitly present in the repository.

## Training

The complete five-fold training workflow is:

```bash
cd DeepSIFA_automation/DeepSIFA
python train.py
```

The default data and split paths are resolved relative to
`DeepSIFA_automation`; they no longer depend on a developer-specific drive.
Use `python train.py --help` to override training parameters or paths.

The reported architecture was selected by comparing the numbers of Spatial
Aware Cells (SACs) and Transformer Cells (TCs). Learning rate, batch size, and
dropout were selected empirically and were not subjected to an exhaustive
automated search.

## Training-set and validation-set evaluation

From `DeepSIFA_automation/DeepSIFA`:

```bash
python eval_train.py
python eval_val.py
python eval_test.py
```

The scripts use repository-relative data, checkpoint, and result directories.
All path arguments can still be overridden from the command line.

## Main trace-extraction and inference workflow

The complete main workflow is retained under `DeepSIFA_main_code`.

```bash
cd DeepSIFA_main_code
python sample.py --help
cd DeepSIFA
python eval_test.py --help
```

`sample.py` performs the MATLAB-assisted trace-extraction and preprocessing
pipeline. `DeepSIFA/eval_test.py` runs neural-network classification using the
released checkpoint. `DLRunner.exe` provides the Windows graphical runner.

## Preprocessing of variable-length traces

The preprocessing scripts apply min-max normalization, linearly resample each
complete trace to 1,024 points, and apply one-dimensional Gaussian smoothing
with `sigma=1`. They do not use zero padding, direct truncation, or sliding
windows.

## Source-language and legacy utilities

Python comments and docstrings in the deposited source have been translated to
English. Runtime data labels and legacy file or directory names are preserved
when changing them would break existing path references. Scripts under folders
named `其他功能` are historical auxiliary analyses; several require the user to
supply paths for their own external datasets. The supported main entry points
listed above use repository-relative defaults.

## Release assets

The GitHub Release provides the Windows runner and large checkpoint bundle for
users who do not retrieve them through Git LFS. Release assets and repository
files correspond to the same public source snapshot.

