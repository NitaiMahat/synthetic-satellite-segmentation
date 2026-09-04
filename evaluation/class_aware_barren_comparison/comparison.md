# Class-Aware Barren Oversampling Comparison

Each seed used the same Rural split, U-Net, 15 epochs, image size, and 1,366 training samples per epoch. The class-aware method sampled Barren-containing images with a weight of 4.0.

| Seed | Baseline mIoU | Class-aware mIoU | Change | Baseline Barren IoU | Class-aware Barren IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 24.42% | 24.08% | -0.34% | 0.00% | 4.59% |
| 7 | 24.67% | 24.67% | -0.00% | 0.00% | 3.62% |
| 123 | 24.85% | 25.06% | +0.21% | 0.00% | 1.14% |

## Three-Seed Average

- Baseline mIoU: 24.64%
- Class-aware mIoU: 24.60%
- mIoU change: -0.04%
- Baseline Barren IoU: 0.00%
- Class-aware Barren IoU: 3.12%
- Barren IoU change: +3.12%
- Pixel accuracy change: -3.97%

## Interpretation

Class-aware Barren oversampling produced nonzero Barren IoU in all three seeds while keeping average mIoU effectively unchanged. Pixel accuracy decreased, so this is a rare-class recall tradeoff rather than an across-the-board improvement.