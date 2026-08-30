# Brightness Robustness Evaluation

Both frozen models were evaluated on the same unchanged Rural validation masks. Only image brightness was modified for the non-clean conditions.

| Condition | Real-only mIoU | +10% Brightness mIoU | Change |
| --- | ---: | ---: | ---: |
| Clean (1.0) | 24.42% | 22.80% | -1.62% |
| Darker (0.8) | 22.26% | 21.59% | -0.67% |
| Much darker (0.6) | 12.32% | 15.64% | +3.32% |
| Brighter (1.2) | 20.51% | 21.22% | +0.71% |
| Much brighter (1.4) | 17.76% | 19.62% | +1.86% |

## Per-class IoU

### Clean (1.0)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 46.22% | 42.04% | -4.19% |
| Building | 21.01% | 22.75% | +1.74% |
| Road | 8.74% | 7.17% | -1.57% |
| Water | 30.33% | 28.59% | -1.73% |
| Barren | 0.00% | 0.01% | +0.01% |
| Forest | 22.50% | 19.15% | -3.36% |
| Agriculture | 42.12% | 39.90% | -2.22% |

### Darker (0.8)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 39.50% | 35.36% | -4.14% |
| Building | 22.11% | 28.28% | +6.17% |
| Road | 5.37% | 2.91% | -2.46% |
| Water | 30.25% | 27.07% | -3.17% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 21.40% | 19.73% | -1.66% |
| Agriculture | 37.17% | 37.75% | +0.58% |

### Much darker (0.6)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 24.52% | 18.68% | -5.84% |
| Building | 7.17% | 20.99% | +13.82% |
| Road | 0.58% | 0.22% | -0.36% |
| Water | 22.59% | 24.97% | +2.38% |
| Barren | 0.00% | 0.00% | +0.00% |
| Forest | 15.55% | 14.30% | -1.25% |
| Agriculture | 15.82% | 30.28% | +14.46% |

### Brighter (1.2)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 43.85% | 41.40% | -2.45% |
| Building | 12.06% | 16.45% | +4.38% |
| Road | 8.90% | 10.55% | +1.66% |
| Water | 28.52% | 32.98% | +4.46% |
| Barren | 0.00% | 0.16% | +0.16% |
| Forest | 8.98% | 7.39% | -1.59% |
| Agriculture | 41.27% | 39.64% | -1.63% |

### Much brighter (1.4)

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 38.12% | 36.52% | -1.61% |
| Building | 6.00% | 10.94% | +4.94% |
| Road | 7.46% | 9.16% | +1.69% |
| Water | 30.24% | 39.75% | +9.50% |
| Barren | 0.00% | 0.02% | +0.02% |
| Forest | 3.11% | 2.17% | -0.94% |
| Agriculture | 39.40% | 38.82% | -0.57% |
