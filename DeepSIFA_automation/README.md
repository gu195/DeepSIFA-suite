# DeepSIFA training and validation automation

This directory is the complete training and validation research directory
supplied for public release. It contains the five-fold training workflow,
training-set evaluation, validation-set evaluation, MLKL datasets,
checkpoints, logs, figures, and auxiliary analyses.

Main entry points:

```bash
cd DeepSIFA
python train.py
python eval_train.py
python eval_val.py
python eval_test.py
```

Default paths are resolved relative to this directory and can be overridden
through the command-line options shown by `--help`.

