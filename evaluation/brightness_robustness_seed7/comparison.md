# Brightness Robustness Evaluation

Both frozen models were evaluated on the same unchanged Rural validation masks. Only image brightness was modified for the non-clean conditions.

| Condition | Real-only mIoU | +10% Brightness mIoU | Change |
| --- | ---: | ---: | ---: |
| Clean (1.0) | 24.67% | 24.02% | -0.64% |
| Darker (0.8) | 21.50% | 24.55% | +3.05% |
| Much darker (0.6) | 16.16% | 19.42% | +3.26% |
| Brighter (1.2) | 22.83% | 22.90% | +0.07% |
| Much brighter (1.4) | 20.98% | 21.91% | +0.93% |

## Per-class IoU

### Clean (1.0)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 40.17% | 45.96% | +5.79% |
| Building | 29.15% | 27.08% | -2.08% |
| Road | 9.44% | 15.31% | +5.86% |
| Water | 26.87% | 29.65% | +2.78% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 24.20% | 7.61% | -16.59% |
| Agriculture | 42.84% | 42.56% | -0.28% |

### Darker (0.8)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 32.14% | 44.79% | +12.64% |
| Building | 24.28% | 23.17% | -1.11% |
| Road | 5.81% | 10.77% | +4.96% |
| Water | 27.27% | 27.71% | +0.44% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 20.24% | 23.73% | +3.49% |
| Agriculture | 40.74% | 41.65% | +0.92% |

### Much darker (0.6)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 22.88% | 36.55% | +13.67% |
| Building | 9.96% | 13.07% | +3.11% |
| Road | 0.93% | 1.92% | +0.99% |
| Water | 30.89% | 26.36% | -4.53% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 13.62% | 21.79% | +8.17% |
| Agriculture | 34.86% | 36.26% | +1.40% |

### Brighter (1.2)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 44.50% | 46.05% | +1.55% |
| Building | 20.92% | 21.26% | +0.34% |
| Road | 9.33% | 14.78% | +5.45% |
| Water | 30.13% | 32.61% | +2.47% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 11.85% | 2.98% | -8.87% |
| Agriculture | 43.11% | 42.64% | -0.47% |

### Much brighter (1.4)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 41.73% | 43.46% | +1.73% |
| Building | 11.62% | 11.34% | -0.28% |
| Road | 8.83% | 13.76% | +4.93% |
| Water | 37.45% | 39.61% | +2.16% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 4.93% | 2.20% | -2.74% |
| Agriculture | 42.32% | 43.01% | +0.69% |
