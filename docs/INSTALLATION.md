# Installation and dependencies

## Python

The deposited environment uses Python 3.8.20. Install all version-pinned
packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

The PyTorch and torchvision versions in `requirements.txt` are compatible.
The default PyPI PyTorch wheel can run on CPU. For GPU execution, install the
PyTorch 2.4.1 wheel appropriate for the local CUDA driver, then install the
remaining requirements.

## MATLAB Engine

Trace extraction calls MATLAB through MATLAB Engine for Python. The development
environment used MATLAB R2021b:

```bash
cd "<matlabroot>/extern/engines/python"
python -m pip install .
```

MATLAB Engine is distributed with MATLAB and therefore cannot be represented
as a normal PyPI requirement in `requirements.txt`.

## Git LFS

The repository contains model weights, microscopy images, processed arrays,
figures, archives, and the Windows executable. These binary files are tracked
with Git LFS:

```bash
git lfs install
git lfs pull
```

If a large file contains a short text pointer beginning with
`version https://git-lfs.github.com/spec/v1`, run `git lfs pull`.

