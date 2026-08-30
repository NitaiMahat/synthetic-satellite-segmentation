# Baseline v1 vs. 10% Brightness Synthetic Data

## Controlled setup

Both runs used the Rural LoveDA train and validation splits, 512 x 512 images, the same compact U-Net, Adam optimizer at a learning rate of 0.0001, seed 42, 15 epochs, and 1,366 training samples per epoch.

The synthetic run replaced no validation images and added 152 deterministic brightness-modified training copies to a 1,518-entry mixed pool. Brightness factors were 0.8 and 1.2. Synthetic samples made up 10.01% of that pool.

## Best validation result

| Metric | Real-only Baseline v1 | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Best epoch | 10 | 11 | - |
| mIoU | 24.42% | 22.80% | -1.62 percentage points |
| Pixel accuracy | 54.52% | 51.43% | -3.09 percentage points |

## Per-class IoU at the best epoch

| Class | Real-only | +10% Brightness | Change |
| --- | ---: | ---: | ---: |
| Background | 46.22% | 42.04% | -4.19 percentage points |
| Building | 21.01% | 22.75% | +1.74 percentage points |
| Road | 8.74% | 7.17% | -1.57 percentage points |
| Water | 30.33% | 28.59% | -1.73 percentage points |
| Barren | 0.00% | 0.01% | +0.01 percentage points |
| Forest | 22.50% | 19.15% | -3.36 percentage points |
| Agriculture | 42.12% | 39.90% | -2.22 percentage points |

## Interpretation

For this single controlled run, adding 10% mild brightness-modified images did not improve overall clean-validation performance. It improved Building IoU slightly, but mIoU and pixel accuracy were lower than the real-only baseline, and most other class scores declined.

This is one measured result, not proof that brightness augmentation is generally harmful. A stronger conclusion would require repeated seeds and tests on brightness-shifted imagery, where this augmentation may still improve robustness even if it lowers clean-domain performance.
