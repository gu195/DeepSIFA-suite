# MonoXtract attention and attribution results

These files were generated with:

```powershell
D:\anaconda\envs\spy38\python.exe `
  D:\Code\MonoXtract\DeepSIFA_main_code\DeepSIFA\attention_analysis.py `
  --device cpu
```

## Model and data

- Checkpoint: `DeepSIFA_main_code/DeepSIFA/checkpoints/best_acc.pth`
- Trained depth inferred from the checkpoint: 3 SAC blocks + 7 Transformer
  blocks.
- Evaluation data: 337 manually labeled MLKL traces from `337.csv`.
- Accuracy in this run: 298/337 = 88.43% (TP 175, TN 123, FP 24, FN 15).
- Six correctly classified, high-confidence examples were selected across
  quantiles of valid-trace step strength and invalid-trace roughness. This
  avoids selecting only the most visually favorable example.

## What the maps mean

`Attention rollout` is calculated from the actual 64 x 64 self-attention
matrices in all seven trained Transformer blocks. The present model forward
pass does not concatenate the defined `cls_token`; the classification head
uses the first ordinary token. Consequently, the rollout starts from token 0
and must be described as **first-token attention rollout**, not class-token
attention.

`|IG| importance` is the normalized absolute Integrated Gradients attribution
of the predicted-class logit to each of the 1024 input samples. It answers
which input positions locally affect the output, whereas attention rollout
shows how token information is mixed. The two quantities need not coincide.

The orange and blue regions in individual figures are the highest- and
lowest-IG 64-point windows. Each window was replaced by a straight line
between its boundary values as a small occlusion check. Exact probability
changes are in `selected_examples_and_occlusion.csv`.

## Files

- `figure_attention_attribution_examples.png`: six-example overview.
- `*_attention_ig.png`: detailed trace, attention, and IG panels.
- `*_explanations.npz`: numerical trace and map arrays.
- `predictions.csv`: predictions for all 337 traces.
- `selected_examples_and_occlusion.csv`: selected-example statistics.
- `run_summary.json`: checkpoint, depth, data, and run metadata.

## Interpretation limit

These maps explain the valid/invalid trace classification. They do not show
that MonoXtract directly predicts molecular state-transition positions,
because transition positions were not training targets. The workspace does
not currently contain independent HMM transition annotations for these test
traces. To compare map hotspots with HMM transitions, populate
`transition_annotations_template.csv` with one row per transition and rerun:

```powershell
D:\anaconda\envs\spy38\python.exe `
  D:\Code\MonoXtract\DeepSIFA_main_code\DeepSIFA\attention_analysis.py `
  --transition-csv `
  D:\Code\MonoXtract\DeepSIFA_main_code\DeepSIFA\attention_results\transition_annotations.csv
```

Any manuscript statement should therefore say that the current maps identify
temporal regions contributing to trace classification. A claim of alignment
with true biophysical transitions requires the independent HMM overlay and a
quantitative comparison against matched non-transition regions.
