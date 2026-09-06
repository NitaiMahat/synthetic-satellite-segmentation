# Rural-to-Urban Cross-Domain Evaluation

All models were trained only on `Train/Rural`. `Val/Urban` was never used for training or checkpoint selection.
The Rural result is in-domain performance; the Urban result measures transfer to an unseen environment.

## Average Metrics Across Three Seeds

| Experiment | Rural mIoU | Urban mIoU | Rural-to-Urban change | Urban pixel accuracy |
| --- | ---: | ---: | ---: | ---: |
| Real-only baseline | 24.64% +/- 0.18% | 21.44% +/- 1.02% | -3.21% | 45.28% |
| +10% brightness | 23.71% +/- 0.65% | 23.16% +/- 3.43% | -0.54% | 46.54% |
| Barren-focused sampling | 24.60% +/- 0.40% | 22.20% +/- 1.13% | -2.40% | 45.04% |

## Urban Comparison With Real-only Baseline

| Experiment | Urban mIoU change | Urban pixel accuracy change |
| --- | ---: | ---: |
| Real-only baseline | +0.00% | +0.00% |
| +10% brightness | +1.73% | +1.25% |
| Barren-focused sampling | +0.77% | -0.24% |

## Mean Urban Per-class IoU

| Class | Baseline | +10% Brightness | Barren-focused |
| --- | ---: | ---: | ---: |
| Background | 30.52% | 30.72% | 30.64% |
| Building | 12.00% | 18.28% | 14.96% |
| Road | 3.50% | 8.30% | 11.86% |
| Water | 45.83% | 48.40% | 45.73% |
| Barren | 0.00% | 0.00% | 5.44% |
| Forest | 22.16% | 19.84% | 11.81% |
| Agriculture | 36.05% | 36.61% | 34.97% |
