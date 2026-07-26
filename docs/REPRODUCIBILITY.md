# Reproducibility guide

## Environment

Create a Python 3.8.20 environment and install the exact package versions in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m scripts.verify_install
```

The verification command checks imports, constructs the default three-SAC plus
seven-TC model, runs a CPU forward pass, validates all deposited manifests, and
reports the class counts.

## Path handling

All scripts accept command-line paths and resolve relative paths against the
current working directory. Defaults are relative to the repository root.
No script requires a path from the original developer workstation.

## Model configuration

The released configuration contains three Spatial Aware Cells followed by
seven Transformer Cells. The model input is a single-channel trace of length
1,024 and the output contains two logits (`invalid`, `valid`).

The number of SACs and TCs was evaluated through a targeted architectural
sensitivity analysis. Other settings, including learning rate, batch size, and
dropout, were chosen empirically and were kept fixed; they were not optimized
through grid search, random search, Bayesian optimization, or another
exhaustive automated procedure.

## Reproduction commands

Train all five folds:

```bash
python -m scripts.train
```

Evaluate the released checkpoint:

```bash
python -m scripts.evaluate \
  --checkpoint checkpoints/best_acc.pth \
  --data-dir data/mlkl/validation/traces \
  --labels data/mlkl/validation/labels.csv \
  --output-dir outputs/validation
```

Run prediction on unlabeled processed traces:

```bash
python -m scripts.predict \
  --checkpoint checkpoints/best_acc.pth \
  --data-dir path/to/traces \
  --output-csv outputs/predictions.csv
```

## Expected deposited counts

| Manifest | Rows | Invalid | Valid |
|---|---:|---:|---:|
| Development data (unique traces) | 335 | 117 | 218 |
| Each fold training manifest | 268 | 93 or 94 | 174 or 175 |
| Each fold internal validation manifest | 67 | 23 or 24 | 43 or 44 |
| Independent validation manifest | 337 | 147 | 190 |

## Current scope

The code and data in this release reproduce the deposited MLKL training and
validation workflow. Additional evaluation datasets used elsewhere in the
manuscript are not presently deposited. Once those data are available, they
can be added as separate manifests without changing the path-independent
evaluation command.
