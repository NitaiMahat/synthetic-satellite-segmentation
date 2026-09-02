# Brightness Robustness Evaluation

Both frozen models were evaluated on the same unchanged Rural validation masks. Only image brightness was modified for the non-clean conditions.

| Condition | Real-only mIoU | +10% Brightness mIoU | Change |
| --- | ---: | ---: | ---: |
| Clean (1.0) | 24.85% | 24.31% | -0.54% |
| Darker (0.8) | 23.71% | 21.64% | -2.07% |
| Much darker (0.6) | 20.76% | 15.96% | -4.80% |
| Brighter (1.2) | 23.21% | 22.44% | -0.77% |
| Much brighter (1.4) | 20.88% | 18.39% | -2.49% |

## Per-class IoU

### Clean (1.0)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 48.74% | 39.86% | -8.89% |
| Building | 29.74% | 27.29% | -2.45% |
| Road | 6.34% | 12.90% | +6.56% |
| Water | 26.22% | 29.27% | +3.05% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 20.48% | 19.08% | -1.40% |
| Agriculture | 42.42% | 41.75% | -0.67% |

### Darker (0.8)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 47.61% | 27.04% | -20.57% |
| Building | 21.65% | 31.43% | +9.78% |
| Road | 2.52% | 9.05% | +6.53% |
| Water | 27.68% | 31.24% | +3.57% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 25.26% | 13.25% | -12.02% |
| Agriculture | 41.29% | 39.48% | -1.81% |

### Much darker (0.6)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 44.55% | 13.42% | -31.13% |
| Building | 9.60% | 19.57% | +9.97% |
| Road | 0.30% | 1.68% | +1.38% |
| Water | 31.39% | 32.93% | +1.54% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 20.31% | 9.43% | -10.88% |
| Agriculture | 39.17% | 34.70% | -4.48% |

### Brighter (1.2)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 47.06% | 41.45% | -5.61% |
| Building | 27.73% | 17.45% | -10.27% |
| Road | 9.02% | 11.19% | +2.17% |
| Water | 27.62% | 31.19% | +3.57% |
| Barren | 0.00% | 0.09% | +0.09% |
| Forest | 7.84% | 15.03% | +7.19% |
| Agriculture | 43.21% | 40.67% | -2.54% |

### Much brighter (1.4)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 42.68% | 36.08% | -6.60% |
| Building | 18.55% | 11.15% | -7.40% |
| Road | 9.37% | 6.90% | -2.47% |
| Water | 29.89% | 33.93% | +4.04% |
| Barren | 0.00% | 0.40% | +0.40% |
| Forest | 2.50% | 6.89% | +4.39% |
| Agriculture | 43.16% | 33.38% | -9.79% |
