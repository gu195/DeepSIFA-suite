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

The supported training and evaluation entry points now resolve default paths
relative to their source directory. Historical one-off scripts under
`其他功能` are retained as part of the full research snapshot; scripts that
refer to external datasets require user-supplied paths.

Only the numbers of SACs and TCs were subjected to targeted architectural
comparison. Learning rate, batch size, and dropout were chosen empirically.
This should not be described as exhaustive hyperparameter optimization.

