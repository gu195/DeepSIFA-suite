# Reproducibility scope

This repository publishes the full local research directories supplied for the
MonoXtract release:

- `DeepSIFA_main_code`: 5,376 source, data, model, result, and auxiliary files
  before English-comment and portability edits;
- `DeepSIFA_automation`: 7,018 training, validation, data, model, result, and
  auxiliary files before English-comment and portability edits.

The automation data contain 335 MLKL training/development traces (218 valid and
117 invalid) and 337 MLKL validation/evaluation traces (190 valid and 147
invalid), including the original, normalized, resampled, and visualization
artifacts present in the supplied directories.

The public workflow was validated using Python 3.9.25, MATLAB R2023b Update 11
(MATLAB Engine 23.2), PyTorch 2.7.1, torchvision 0.22.1, and CUDA 12.8. CPU
evaluation is also supported by the revised evaluator.

The supported training and evaluation entry points now resolve default paths
relative to their source directory. Historical one-off scripts under
`其他功能` are retained as part of the full research snapshot; scripts that
refer to external datasets require user-supplied paths.

Only the numbers of SACs and TCs were subjected to targeted architectural
comparison. Learning rate, batch size, and dropout were chosen empirically.
This should not be described as exhaustive hyperparameter optimization.

## Reproduce the 337-trace MLKL evaluation

The released checkpoint and 337-trace validation dataset are the script
defaults. Therefore, from `DeepSIFA_main_code/DeepSIFA`, run:

```bash
python eval_test.py
```

The equivalent explicit command from the repository root is:

```bash
python DeepSIFA_main_code/DeepSIFA/eval_test.py \
  --data_dir DeepSIFA_automation/data/mlkl/test/v1 \
  --weight_path DeepSIFA_main_code/DeepSIFA/checkpoints/best_acc.pth \
  --device auto \
  --bt_size 32 \
  --results_dir reproduction_results/mlkl_337
```

The evaluator reads `337.csv`, automatically locates the complete directory
of 1,024-point smoothed NPZ traces, constructs the released 3-SAC + 7-TC
network, and loads the checkpoint strictly. The expected label-based results
are:

- 337 traces;
- accuracy: 88.43%;
- TN = 123, FP = 24, FN = 15, TP = 175;
- ROC-AUC = 0.9495;
- average precision = 0.9602.

The output directory contains `score.csv`, `metrics.json`, a confusion matrix,
ROC and precision-recall curves, and the score-distribution figure. Small
floating-point differences in probabilities can occur between CPU/GPU and
PyTorch versions, but the released environment should reproduce the reported
labels and rounded metrics.

