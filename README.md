# Synthetic Satellite Segmentation

This project studies whether lightweight synthetic data methods can help a satellite-image segmentation model work in a new environment.

## Project Status

**Completed research phase:** Rural-to-Urban cross-domain evaluation.

The model was trained only on Rural LoveDA satellite images and then tested on Urban LoveDA images it never saw during training. No Urban image or mask was used to train a model or select a checkpoint.

The complete final report is [Synthetic Satellite Segmentation: Rural-to-Urban Generalization](output/pdf/synthetic_satellite_segmentation_cross_domain_final_report.pdf).

## Research Question

> Can lightweight synthetic image augmentation and rare-class sampling improve a satellite segmentation model's ability to generalize from Rural to Urban environments?

In simple terms: the model learned from countryside satellite images. We then tested whether simple changes to its training data helped it understand city images.

## Main Finding

There is a real Rural-to-Urban performance gap. The real-only model fell from **24.64% mIoU** on Rural validation images to **21.44% mIoU** on unseen Urban validation images.

| Experiment | Rural mIoU | Urban mIoU | Change vs. Urban baseline | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Real-only baseline | 24.64% +/- 0.18% | 21.44% +/- 1.02% | Reference | The starting point. |
| +10% mild brightness copies | 23.71% +/- 0.65% | 23.16% +/- 3.43% | +1.73 points | Promising average Urban improvement, but results vary too much across seeds for a strong claim. |
| Barren-focused sampling | 24.60% +/- 0.40% | 22.20% +/- 1.13% | +0.77 points | Small overall Urban improvement and meaningful rare-class recovery. |

The Barren-focused method increased Urban **Barren IoU from 0.00% to 5.44%**. This is the clearest result: targeted sampling helps the model recognize a rare class that the normal baseline misses.

## What Was Tested

All experiments used the same compact U-Net, 512 x 512 inputs, 15 epochs, Adam optimizer, learning rate `1e-4`, batch size `1`, and three random seeds: `42`, `7`, and `123`.

| Method | What changed | Why it was tested | Outcome |
| --- | --- | --- | --- |
| Real-only baseline | Trained only on original Rural images. | Establish a fair reference. | Baseline for every comparison. |
| Brightness augmentation | 10% of the final training mix used mild darker or brighter copies, with factors `0.8` and `1.2`. | Reduce sensitivity to lighting. | Mixed but encouraging Urban result; not yet conclusive. |
| Rotation augmentation | 10% of the final training mix used 90, 180, or 270 degree rotations. | Reduce orientation sensitivity. | Rejected after its seed-42 pilot reduced mIoU from 24.42% to 21.54%. |
| Barren-focused sampling | Images containing Barren land were sampled more often, with weight `4.0`. | Help a rare class. | Recovered nonzero Barren recognition in Rural and Urban evaluation. |

## Dataset

The project uses the [LoveDA](https://arxiv.org/abs/2110.08733) land-cover segmentation dataset.

| Split | Images | Use |
| --- | ---: | --- |
| `Train/Rural` | 1,366 | The only training data used. |
| `Val/Rural` | 992 | In-domain validation. |
| `Val/Urban` | 677 | Unseen-environment evaluation only. |

Each original image is 1024 x 1024 pixels. Experiments resize images and masks to 512 x 512 during loading.

LoveDA raw labels are remapped before training:

| Raw label | Meaning | Model label |
| ---: | --- | ---: |
| 0 | No Data | 255, ignored by the loss and metrics |
| 1 | Background | 0 |
| 2 | Building | 1 |
| 3 | Road | 2 |
| 4 | Water | 3 |
| 5 | Barren | 4 |
| 6 | Forest | 5 |
| 7 | Agriculture | 6 |

Dataset files are intentionally not committed. Put the LoveDA PNG folders here:

```text
data/raw/
  Train/
    Rural/images_png/
    Rural/masks_png/
    Urban/images_png/
    Urban/masks_png/
  Val/
    Rural/images_png/
    Rural/masks_png/
    Urban/images_png/
    Urban/masks_png/
```

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/dataset.py` | Loads image-mask pairs and remaps LoveDA labels. |
| `src/model.py` | Compact U-Net architecture. |
| `src/metrics.py` | mIoU, per-class IoU, pixel accuracy, and confusion matrix. |
| `src/synthetic.py` | Brightness and rotation augmentation datasets. |
| `src/sampling.py` | Barren-focused oversampling utilities. |
| `experiments/train_baseline.py` | Reproducible training entry point for all completed experiment types. |
| `evaluation/evaluate_rural_to_urban.py` | Evaluates the nine frozen checkpoints on Rural and Urban validation data. |
| `evaluation/rural_to_urban/` | Final JSON results, readable comparison, chart, and Urban prediction examples. |
| `configs/` | Saved experiment settings. |
| `models/` | Saved checkpoints, histories, and metrics. |
| `output/pdf/` | Final report and archived project reports. |
| `tests/` | Unit tests for metrics, model shape, augmentation, sampling, and cross-domain aggregation. |

## Setup

This project was developed on Windows PowerShell with Python and a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

CPU execution is supported. Full training is slow on CPU and can take several hours per 15-epoch run.

## Reproduce the Final Evaluation

Run all tests first:

```powershell
python -m unittest discover -s tests -v
```

The final Rural-to-Urban evaluation uses the nine saved checkpoints. It does not train any model:

```powershell
python -u evaluation\evaluate_rural_to_urban.py --batch-size 4 --image-size 512 --base-channels 16 --output-dir evaluation\rural_to_urban
```

The evaluation prints batch progress and can take several hours on CPU. If it is interrupted, resume safely without repeating completed checkpoints:

```powershell
python -u evaluation\evaluate_rural_to_urban.py --batch-size 4 --image-size 512 --base-channels 16 --output-dir evaluation\rural_to_urban --resume
```

Expected outputs:

```text
evaluation/rural_to_urban/
  results.json
  comparison.md
  rural_urban_miou_chart.png
  urban_validation_examples_seed42.png
```

## Reproduce a Training Run

Example: train the real-only Rural baseline.

```powershell
python -u experiments\train_baseline.py --experiment-name baseline_v1_seed42 --epochs 15 --batch-size 1 --image-size 512 --learning-rate 1e-4 --base-channels 16 --seed 42 --output-dir models\baseline_v1_complete
```

Example: train the 10% brightness experiment.

```powershell
python -u experiments\train_baseline.py --experiment-name synthetic_ratio_10_brightness_seed42 --epochs 15 --batch-size 1 --image-size 512 --learning-rate 1e-4 --base-channels 16 --seed 42 --synthetic-ratio 0.1 --brightness-factors 0.8 1.2 --train-samples-per-epoch 1366 --output-dir models\synthetic_ratio_10_brightness
```

## Reports and Results

The final cross-domain report is the recommended document to read first:

- [Final Rural-to-Urban report](output/pdf/synthetic_satellite_segmentation_cross_domain_final_report.pdf)
- [Cross-domain comparison table](evaluation/rural_to_urban/comparison.md)
- [Cross-domain raw results](evaluation/rural_to_urban/results.json)
- [Cross-domain mIoU chart](evaluation/rural_to_urban/rural_urban_miou_chart.png)
- [Urban prediction examples](evaluation/rural_to_urban/urban_validation_examples_seed42.png)

Earlier documents are retained as an experiment history:

- `output/pdf/synthetic_satellite_segmentation_progress_review.pdf`
- `output/pdf/synthetic_satellite_segmentation_experiment_update.pdf`
- `output/pdf/two_seed_brightness_experiment_record.pdf`
- `output/pdf/three_seed_brightness_final_report.pdf`
- `output/pdf/final_project_experiment_report.pdf`
- `output/pdf/synthetic_satellite_segmentation_final_research_paper.pdf`
- `output/pdf/current_project_question_and_findings.pdf`

## Limits and Honest Interpretation

- The project measures transfer from LoveDA Rural to LoveDA Urban, not global geographic generalization across countries or satellite sensors.
- Brightness augmentation improved average Urban mIoU but varied substantially by random seed. More matched seeds are needed before calling it reliable.
- Barren-focused sampling is a tradeoff: it improves rare-class recognition but is not a universal improvement for every class or metric.
- Rotation was intentionally not repeated because the first pilot was clearly worse than the baseline.

## Reference

Wang, J., et al. *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation.* 2021. https://arxiv.org/abs/2110.08733
