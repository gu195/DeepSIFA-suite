# Installation and dependencies

## Python

The released environment uses Python 3.9 and was validated with Python 3.9.25,
PyTorch 2.7.1, and torchvision 0.22.1. Create the environment and install the
version-pinned packages from the repository root:

```bash
conda create -n monoxtract python=3.9
conda activate monoxtract
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the validated NVIDIA GPU configuration (PyTorch 2.7.1 with CUDA 12.8),
install PyTorch from the official CUDA 12.8 wheel index before installing the
remaining requirements:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The scripts also support CPU evaluation. In that case, install the standard
PyTorch packages from `requirements.txt` and run `eval_test.py --device cpu`.

`segmentation-models-pytorch` is not required by the supported MonoXtract
training, preprocessing, or evaluation workflows. It was removed from the
main requirements because an unused import in historical auxiliary scripts
caused an installation conflict.

## MATLAB Engine

Trace extraction calls MATLAB through MATLAB Engine for Python. The development
and validation environment uses MATLAB R2023b (MATLAB Engine 23.2):

```bash
cd "<matlabroot>/extern/engines/python"
python -m pip install .
```

MATLAB Engine is distributed with MATLAB and therefore cannot be represented
as a normal PyPI requirement in `requirements.txt`.

The engine must be installed into the same Python 3.9 environment used to run
MonoXtract. The validated command on Windows was:

```bat
cd /d D:\matlab2023b\extern\engines\python
python -m pip install .
```

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

