# Kvasir-SEG Polyp Segmentation and Feature-Characterization Pipeline

This project implements and evaluates an image-level, leakage-aware research
pipeline for colorectal polyp segmentation on Kvasir-SEG. It includes image
preprocessing, handcrafted-feature baselines, CNN segmentation models, contour-
based bounding-box evaluation, five-fold cross-validation, subgroup analysis,
and resumable experiment outputs.

> **Research use only.** This software is not a medical device and has not been
> validated for diagnosis, screening, clinical decision-making, or use on
> patients. Kvasir-SEG consists of polyp-positive images; it does not provide a
> normal-patient control cohort for estimating clinical specificity.

## Project scope

The implementation focuses on the core AI engine: polyp segmentation, mask-
derived localization, and feature characterization. It does not implement a
telemedicine workflow, patient management, or clinical deployment.

The complete executed matrix contains **13 experiments**:

- One classical experiment containing Random Forest and RBF-SVM patch
  classifiers.
- Twelve deep-learning configurations: eight SMP architectures, three
  additional raw/CLAHE and BCE/BCE+Dice ablation cells, and one weighted
  BCE+IoU loss condition.

The results below are the final saved five-fold results from the run completed
on 6 November 2026. The reproducible machine and run configuration are recorded
in the [benchmark plan](runs/full_benchmark/benchmark_plan.json) and each
experiment's `run_config.json`.

## Dataset

The pipeline expects paired images and binary masks, plus the Kvasir bounding-
box annotation file when available:

```text
kvasir-seg/
└── Kvasir-SEG/
    ├── images/
    ├── masks/
    └── kavsir_bboxes.json
```

The loader checks exact image/mask basename pairing, readability, matching
dimensions, and bounding-box metadata. The benchmark used **1,000 paired
images**. It fails on unmatched files or incomplete bounding-box metadata
rather than silently dropping cases.

The sessile-polyp subset contains 196 filenames that also occur in Kvasir-SEG.
Those are tagged as a subgroup in out-of-fold predictions; they are **not**
added as extra cases and are not an independent external validation cohort.

The dataset and archive files are not included in this repository. Arrange the
dataset locally in the layout above before running commands.

## End-to-end workflow

The commands in this section use PowerShell and assume the working directory is
the project root.

### 1. Create an environment and install dependencies

Python 3.10 or later is required. The completed benchmark was run with Python
3.11.9, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121,
segmentation-models-pytorch 0.6.0, and CUDA on an NVIDIA GeForce RTX 3050 Ti
Laptop GPU (4 GB VRAM).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If pip installed a CPU-only PyTorch wheel, install a CUDA wheel compatible with
the machine's NVIDIA driver. The following exact pair was used for the saved
checkpoints; use the official PyTorch installation selector for other systems:

```powershell
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The benchmark was run with batch size 4, 352 x 352 images, 30 maximum epochs,
early-stopping patience 6, seed 42, and randomly initialized encoders
(`pretrained_encoder=False`). CUDA is the default for deep training. Use
`--device cpu` for small smoke tests or if CUDA is unavailable; a full CPU
benchmark is likely to be substantially slower.

### 2. Check the dataset and run classical baselines

```powershell
python -m polyp_pipeline classical `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir runs `
  --folds 5 --seed 42 --size 352 `
  --preprocessing clahe
```

This computes/caches the image descriptors, fits both classifiers using
training-fold images only, and evaluates each held-out fold.

### 3. Run or resume the complete benchmark

```powershell
python -m polyp_pipeline benchmark `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir runs `
  --folds 5 --seed 42 --size 352 `
  --batch-size 4 --epochs 30 --patience 6 `
  --device cuda
```

The full run writes to `runs/full_benchmark/`. To check progress:

```powershell
python -m polyp_pipeline status --input-dir "runs\full_benchmark"
```

Training checkpoints are written atomically after each epoch. If execution is
interrupted, rerun the **same command with the same configuration** to resume.
The runner validates saved settings, fold definitions, and case IDs; it rejects
incompatible resumes rather than silently mixing experiments. Do not delete
checkpoints for experiments that need resuming.

### 4. Generate the final comparison and plots

```powershell
python -m polyp_pipeline report `
  --input-dir "runs\full_benchmark" `
  --output-dir "reports\full_benchmark"
```

This generates:

- `reports/full_benchmark/benchmark_comparison.csv` - fold-level mean and
  sample standard deviation for every model/configuration.
- `reports/full_benchmark/subgroup_comparison.csv` - out-of-fold sessile and
  small-box subgroup metrics.
- `reports/full_benchmark/ablation_and_model_comparison.png` - Dice and
  sensitivity comparison plot.

The tracked report artifacts are compact. The dataset, archives, virtual
environment, checkpoints, cached features, masks, and per-image prediction
outputs are local runtime data and should not be uploaded with source.

### Optional: train one deep-learning condition

```powershell
python -m polyp_pipeline train `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir runs `
  --model unet_resnet34 `
  --loss bce_dice `
  --preprocessing clahe `
  --folds 5 --seed 42 --size 352 `
  --batch-size 4 --epochs 30 --patience 6 --device cuda
```

Available model names and losses are registered in
[models.py](polyp_pipeline/models.py). Add `--pretrained-encoder` to initialize
the SMP encoder from ImageNet weights; this can require a network download and
does not reproduce the benchmark's random initialization.

## Advanced pretrained and transformer benchmark

The advanced runner builds on the completed scratch-initialized benchmark
without overwriting its results. It trains six new configurations, all on the
same image-level folds and CLAHE preprocessing:

1. ImageNet-pretrained FPN/ResNet-34 with BCE+Dice.
2. ImageNet-pretrained DeepLabV3+/ResNet-34 with BCE+Dice.
3. ImageNet-pretrained SegFormer/MiT-B0 with BCE+Dice.
4. ImageNet-pretrained UPerNet/MiT-B0 with BCE+Dice.
5. ImageNet-pretrained FPN/ResNet-34 with focal Tversky loss.
6. ImageNet-pretrained FPN/ResNet-34 with Dice plus signed-distance boundary
   loss.

Each selected checkpoint is evaluated both with ordinary inference and
four-view test-time augmentation (original, horizontal flip, vertical flip,
and both flips). Flipped predictions are unflipped back to the original
orientation and averaged as probabilities before the fixed 0.5 threshold.
Training runs once per configuration; the ordinary-vs-TTA comparison uses the
same checkpoint and held-out outer-fold images. This isolates TTA's
post-processing effect from training variability.

Run the advanced matrix with the same dataset and device settings:

```powershell
python -m polyp_pipeline advanced-benchmark `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "runs\advanced_benchmark" `
  --folds 5 --seed 42 --size 352 `
  --batch-size 4 --epochs 30 --patience 6 `
  --device cuda
```

Monitor or resume with the exact same command:

```powershell
python -m polyp_pipeline status --input-dir "runs\advanced_benchmark"
```

Pretrained weights are downloaded through the registered SMP encoder weights
and cached by the model library. The advanced runner saves its plan, status,
progress events, per-fold best and latest checkpoints, and model outputs under
`runs/advanced_benchmark/`. Per-fold `fold_metrics.csv` includes the ordinary
metrics and `tta_*` metrics. TTA per-image rows are stored separately in
`per_image_metrics_tta.csv`; probability outputs use `predictions_tta/`.

To produce an advanced-only report:

```powershell
python -m polyp_pipeline report `
  --input-dir "runs\advanced_benchmark" `
  --output-dir "reports\advanced_benchmark"
```

The report emits model names with a `_tta` suffix for the TTA version, making
the paired comparison visible alongside regular inference. This project uses
TTA as the selected inference-refinement experiment; it does not claim to
implement or evaluate a Dense CRF.

## Pipeline details

### Preprocessing and augmentation

Every image is resized to 352 x 352 using area interpolation. Masks are
binarized and resized with nearest-neighbor interpolation to preserve class
labels.

Two input conditions are evaluated:

- **Raw:** resize without specular-reflection inpainting or CLAHE.
- **CLAHE:** identify bright, low-saturation pixels in HSV (`V >= 242`,
  `S <= 55`), apply a 3 x 3 median blur to the mask, and inpaint detected
  regions with OpenCV Telea inpainting (radius 3). Convert to CIELAB and apply
  CLAHE to the L channel only (clip limit 2.0, 8 x 8 tile grid), then convert
  back to RGB.

Augmentation is enabled only for training data. Each image/mask pair receives
the same horizontal and vertical flips (each probability 0.5) and a random
multiple-of-90-degree rotation. With probability 0.7, image brightness and
contrast are perturbed; with probability 0.3, hue and saturation are perturbed.
Validation and test inputs are deterministic and are not augmented.

### Handcrafted features and classical models

Each resized image is divided into a 4 x 4 grid, giving 16 patch examples per
image. Features are computed from the image only; ground-truth masks are used
only to create patch labels. Each patch descriptor includes:

- 8-bin histograms and mean/standard deviation for each HSV and CIELAB channel.
- Four gray-level co-occurrence matrix (GLCM) properties (contrast,
  homogeneity, energy, correlation) at two distances and four angles, using 16
  quantized gray levels.
- A uniform local binary pattern (LBP) histogram (`P=8`, `R=1`).
- Canny edge fraction, Sobel gradient magnitude statistics, and largest-edge-
  contour proxies (circularity, aspect ratio, and area fraction).

A patch is labeled positive when at least 2% of its pixels are foreground.
Features and labels are grouped by source image before fold assignment and
training, preventing patches from the same image from crossing train/test
boundaries.

- **Random Forest:** 300 trees, balanced-subsample class weights, square-root
  feature selection, minimum leaf size 2.
- **SVM:** standardized features and an RBF kernel (`C=2`), with balanced class
  weights. Decision scores are sigmoid-mapped for patch-map evaluation; they
  are not calibrated probabilities.

The resulting 4 x 4 scores are bilinearly upsampled to form a coarse mask.
These classical methods are patch classifiers, not full-resolution pixel
segmenters; their segmentation and box scores should not be interpreted as
directly equivalent to deep models.

### Deep-learning models and losses

All deep models use `segmentation-models-pytorch` with three input channels,
one foreground logit, and no output activation inside the model:

| Model family | SMP architecture | Encoder |
|---|---|---|
| U-Net | `Unet` | ResNet-34, ResNet-50, EfficientNet-B0 |
| U-Net++ | `UnetPlusPlus` | ResNet-34, ResNet-50, EfficientNet-B0 |
| FPN | `FPN` | ResNet-34 |
| DeepLabV3+ | `DeepLabV3Plus` | ResNet-34 |

The sigmoid converts logits to foreground scores for validation/evaluation.
Training uses AdamW with learning rate 1e-4 and weight decay 1e-4,
`ReduceLROnPlateau`, a maximum of 30 epochs, and early stopping after 6 epochs
without validation-Dice improvement. Best checkpoints are selected using only
the inner validation split. CUDA automatic mixed precision and gradient scaling
are used on CUDA.

For binary target `y`, logit `z`, and probability `p = sigmoid(z)`, the losses
are:

- **Binary cross entropy (BCE):**
  `L_BCE = mean(-y log(p) - (1-y) log(1-p))`.
- **BCE + soft Dice loss:**
  `L = L_BCE + 1 - (2 sum(p y) + 1) / (sum(p) + sum(y) + 1)`.
  The Dice term is computed per image and averaged over the batch.
- **Weighted BCE + weighted IoU loss:** define
  `w = 1 + 5 |AvgPool31x31(y) - y|`. The loss is the sum of weighted BCE
  normalized by `sum(w)` and
  `1 - (sum(w p y) + 1) / (sum(w (p + y)) - sum(w p y) + 1)`.
  This emphasizes mask-boundary pixels.
- **Focal Tversky loss:** for each image, let `TP=sum(p y)`,
  `FP=sum(p (1-y))`, and `FN=sum((1-p)y)`. Then
  `T = (TP + 1) / (TP + 0.3 FP + 0.7 FN + 1)` and
  `L = (1-T)^0.75`, averaged over images. The larger false-negative weight
  emphasizes recall; it does not guarantee a clinical sensitivity gain.
- **Dice + signed-distance boundary loss:** let `d` be the signed Euclidean
  distance transform of the binary mask (negative inside the foreground,
  positive outside), normalized by the larger image dimension. The loss is
  `1 - Dice_soft + 0.1 mean(p d)`. The distance map is computed from training
  labels only; it is not learned from or estimated using held-out test images.
  The coefficient is fixed rather than selected using test performance.

### Cross-validation and leakage controls

The benchmark uses deterministic, shuffled, image-level five-fold
cross-validation (`seed=42`). Each image appears in exactly one outer test fold.
For every outer fold, 10% of the outer training images are split off for
validation. Training, augmentation, early stopping, and checkpoint selection
use only outer-training/inner-validation images. Outer test images are held out
until final fold evaluation. Image IDs are checked for train/validation/test
overlap.

The sessile subset is an annotation for subgroup reporting only. Since it
overlaps the Kvasir-SEG cases, it is not an external validation set.

### Metrics and their definitions

For binary ground-truth mask `G` and predicted mask `P`, with true positives
`TP`, false positives `FP`, false negatives `FN`, and true negatives `TN`:

```text
Dice = 2 TP / (2 TP + FP + FN)
IoU  = TP / (TP + FP + FN)
Sensitivity (recall) = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

Deep-model Dice and IoU are computed per test image and averaged within each
fold. Sensitivity and specificity use pixel counts pooled over that fold's
test images. The comparison report then gives the arithmetic mean and **sample
standard deviation across the five outer folds** (`n=5`); it does not report a
confidence interval or a significance test. Classical patch-map Dice/IoU are
computed from pooled fold pixel counts.

For box evaluation, provided Kvasir ground-truth boxes are rescaled to the
352 x 352 frame. If annotations are unavailable, mask contours are the
fallback. Predicted boxes come from connected components in the binary
prediction at a fixed threshold of 0.5; components with fewer than four pixels
are excluded. A predicted box matches at most one same-image ground-truth box
at IoU >= 0.5. Box precision/recall use this fixed-threshold set of boxes.
AP50 ranks the component boxes by their mean mask score and applies greedy
matching at IoU 0.5. It is an AP50 estimate for this implementation, not COCO
multi-threshold mAP.

Specificity is **pixel-level within-image background specificity**. Because
the dataset contains polyp-positive frames rather than normal patients, it is
not patient-level specificity or a clinical false-alarm rate. Inference FPS is
model-only batch throughput and excludes image reading, preprocessing, and
writing outputs. Reported speed is hardware-dependent.

## Completed benchmark results

All values below are **mean +/- sample SD across five outer folds**. Metrics
use a fixed foreground threshold of 0.5. The results are empirical estimates
for this dataset and protocol, not clinical performance estimates.

| Model | Input / loss | Dice | mIoU | Sensitivity | Pixel specificity | AP50 | Box P / R | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | CLAHE / patch classifier | 0.505 +/- 0.020 | 0.338 +/- 0.018 | 0.721 +/- 0.036 | 0.788 +/- 0.011 | 0.093 +/- 0.019 | 0.208 / 0.228 | n/a |
| RBF-SVM | CLAHE / patch classifier | 0.468 +/- 0.018 | 0.306 +/- 0.015 | 0.878 +/- 0.023 | 0.650 +/- 0.007 | 0.070 +/- 0.014 | 0.173 / 0.183 | n/a |
| U-Net, EfficientNet-B0 | CLAHE / BCE+Dice | 0.778 +/- 0.033 | 0.683 +/- 0.038 | 0.759 +/- 0.048 | 0.966 +/- 0.009 | 0.699 +/- 0.061 | 0.381 / 0.745 | 49.4 |
| U-Net, ResNet-34 | CLAHE / BCE | 0.797 +/- 0.017 | 0.712 +/- 0.021 | 0.780 +/- 0.058 | 0.974 +/- 0.009 | 0.751 +/- 0.025 | 0.350 / 0.784 | 49.6 |
| U-Net, ResNet-34 | CLAHE / BCE+Dice | 0.829 +/- 0.018 | 0.749 +/- 0.019 | 0.811 +/- 0.045 | 0.976 +/- 0.007 | 0.792 +/- 0.038 | 0.399 / 0.818 | 49.1 |
| U-Net, ResNet-34 | CLAHE / weighted BCE+IoU | 0.839 +/- 0.023 | 0.760 +/- 0.029 | 0.838 +/- 0.027 | 0.973 +/- 0.006 | 0.804 +/- 0.040 | 0.345 / 0.831 | 51.0 |
| U-Net, ResNet-34 | Raw / BCE | 0.799 +/- 0.014 | 0.713 +/- 0.016 | 0.786 +/- 0.039 | 0.974 +/- 0.007 | 0.761 +/- 0.026 | 0.364 / 0.796 | 67.3 |
| U-Net, ResNet-34 | Raw / BCE+Dice | 0.815 +/- 0.040 | 0.733 +/- 0.044 | 0.794 +/- 0.032 | 0.974 +/- 0.010 | 0.759 +/- 0.065 | 0.492 / 0.792 | 73.0 |
| U-Net, ResNet-50 | CLAHE / BCE+Dice | 0.773 +/- 0.025 | 0.679 +/- 0.027 | 0.754 +/- 0.022 | 0.972 +/- 0.004 | 0.719 +/- 0.042 | 0.370 / 0.762 | 49.1 |
| U-Net++, EfficientNet-B0 | CLAHE / BCE+Dice | 0.778 +/- 0.027 | 0.684 +/- 0.030 | 0.751 +/- 0.043 | 0.967 +/- 0.004 | 0.703 +/- 0.057 | 0.374 / 0.748 | 43.1 |
| U-Net++, ResNet-34 | CLAHE / BCE+Dice | 0.835 +/- 0.020 | 0.755 +/- 0.025 | 0.814 +/- 0.046 | 0.975 +/- 0.004 | 0.794 +/- 0.035 | 0.462 / 0.826 | 44.9 |
| U-Net++, ResNet-50 | CLAHE / BCE+Dice | 0.785 +/- 0.013 | 0.692 +/- 0.017 | 0.768 +/- 0.039 | 0.971 +/- 0.004 | 0.735 +/- 0.014 | 0.375 / 0.769 | 19.9 |
| FPN, ResNet-34 | CLAHE / BCE+Dice | **0.840 +/- 0.014** | 0.759 +/- 0.015 | 0.806 +/- 0.036 | **0.980 +/- 0.007** | 0.804 +/- 0.014 | 0.670 / 0.829 | 55.3 |
| DeepLabV3+, ResNet-34 | CLAHE / BCE+Dice | 0.837 +/- 0.020 | 0.756 +/- 0.023 | 0.815 +/- 0.027 | 0.975 +/- 0.006 | **0.810 +/- 0.032** | **0.604 / 0.832** | 50.0 |

The full-precision values and additional metrics are in
[benchmark_comparison.csv](reports/full_benchmark/benchmark_comparison.csv).
The table gives mean box precision/recall and mean FPS; those columns are
available in the CSV with fold standard deviations.

### Interpretation

- **Best mean Dice:** FPN with ResNet-34 (0.840 +/- 0.014).
- **Best mean AP50:** DeepLabV3+ with ResNet-34 (0.810 +/- 0.032).
- **Highest sensitivity among deep configurations:** U-Net/ResNet-34 with
  weighted BCE+IoU (0.838 +/- 0.027), with Dice 0.839 +/- 0.023. The tradeoff
  was slightly lower pixel specificity than the BCE+Dice reference condition.
- **Reference architecture comparison:** U-Net++/ResNet-34 achieved Dice
  0.835 +/- 0.020, close to FPN, but lower measured throughput (44.9 vs
  55.3 FPS). U-Net++/ResNet-50 had the lowest measured speed (19.9 FPS).
- **Classical baseline tradeoff:** SVM had higher sensitivity than Random
  Forest but lower Dice and pixel specificity. Both classical baselines are
  coarse 4 x 4 patch maps, so they are not full-resolution segmenters.

### Preprocessing and loss ablation

The controlled ablation used U-Net/ResNet-34 with the same folds, seed,
training settings, and threshold:

| Input | Loss | Dice | Sensitivity | Pixel specificity | AP50 |
|---|---|---:|---:|---:|---:|
| Raw | BCE | 0.799 +/- 0.014 | 0.786 +/- 0.039 | 0.974 +/- 0.007 | 0.761 +/- 0.026 |
| Raw | BCE+Dice | 0.815 +/- 0.040 | 0.794 +/- 0.032 | 0.974 +/- 0.010 | 0.759 +/- 0.065 |
| CLAHE | BCE | 0.797 +/- 0.017 | 0.780 +/- 0.058 | 0.974 +/- 0.009 | 0.751 +/- 0.025 |
| CLAHE | BCE+Dice | 0.829 +/- 0.018 | 0.811 +/- 0.045 | 0.976 +/- 0.007 | 0.792 +/- 0.038 |

For this run, CLAHE with BCE+Dice exceeded raw BCE+Dice by 0.014 Dice and
0.017 sensitivity. BCE+Dice exceeded BCE by 0.016 Dice on raw inputs and
0.032 Dice on CLAHE inputs. CLAHE did not improve every metric in every
condition: the CLAHE+BCE cell was slightly below raw+BCE on Dice and
sensitivity. These descriptive differences are not significance-tested.
Weighted BCE+IoU is an additional loss experiment, not part of the four-cell
factorial table.

### Subgroup observations

For the U-Net++/ResNet-34 CLAHE+BCE+Dice model, out-of-fold metrics were:

| Subgroup | Images | Dice mean +/- SD | Sensitivity mean |
|---|---:|---:|---:|
| Sessile subset | 196 | 0.732 +/- 0.288 | 0.788 |
| Other cases | 804 | 0.860 +/- 0.167 | 0.871 |
| Ground-truth box area <= 2% of frame | 13 | 0.391 +/- 0.419 | 0.651 |
| Ground-truth box area > 2% of frame | 987 | 0.841 +/- 0.192 | 0.857 |

Small-box results are based on only 13 cases and have high variance. Sessile
cases are a filename-overlapping subset, not an external test set. See
[subgroup_comparison.csv](reports/full_benchmark/subgroup_comparison.csv) for
additional per-model metrics.

### Model complexity

THOP single-image estimates use 352 x 352 RGB input. FPS is measured on the
RTX 3050 Ti during model-only batched inference and is not an end-to-end system
latency measurement.

| Architecture | Parameters (M) | Approx. GFLOPs/image | Mean FPS |
|---|---:|---:|---:|
| U-Net / EfficientNet-B0 | 2.283 | 4.789 | 49.4 |
| U-Net / ResNet-34 | 24.436 | 14.841 | 49.1 |
| U-Net / ResNet-50 | 32.521 | 20.243 | 49.1 |
| U-Net++ / EfficientNet-B0 | 2.601 | 9.656 | 43.1 |
| U-Net++ / ResNet-34 | 26.079 | 34.882 | 44.9 |
| U-Net++ / ResNet-50 | 48.986 | 108.827 | 19.9 |
| FPN / ResNet-34 | 23.154 | 12.999 | 55.3 |
| DeepLabV3+ / ResNet-34 | 22.437 | 14.983 | 50.0 |

These are approximate operation counts from THOP and hardware-specific
throughput measurements, not guaranteed runtime on other machines.

## Output structure

```text
runs/full_benchmark/
├── benchmark_plan.json
├── benchmark_state.json
├── progress.jsonl
├── classical__clahe/
└── deep/
    └── <model>__<loss>__<preprocessing>/
        ├── folds.json
        ├── run_config.json
        ├── fold_metrics.csv
        ├── per_image_metrics.csv
        ├── summary.json
        └── fold_1/ ... fold_5/
            ├── split_manifest.json
            ├── best_model.pt
            ├── latest_training.pt
            ├── training_history.json
            ├── predictions/
            └── overlays/
```

Each deep experiment records outer-fold metrics, per-image out-of-fold metrics,
best validation-selected weights, and atomic resumable training checkpoints.
Overlays show the target mask/box and predicted mask/box. The classical output
stores its fold metrics, feature cache, and progress log. Runtime artifacts are
excluded from source control by `.gitignore`; regenerated summary tables and
plots are kept under `reports/full_benchmark/`.

## Scope and limitations

- HarDNet-MSEG, Polyp-PVT, and MedSAM 2 were not included. Their official
  implementations, weights, and training procedures are distinct; a generic
  SMP model is not a valid substitute.
- Encoders in the completed benchmark were randomly initialized. Results do
  not represent ImageNet-pretrained encoder performance.
- There is no independent external dataset, normal-patient control set, or
  clinical validation.
- The sessile and small-box subgroup comparisons are descriptive; subgroup
  counts and overlap are stated above.
- The probability threshold is fixed at 0.5. Scores are not probability-
  calibrated, and no threshold was tuned on the outer test folds.
- Mean +/- SD across five folds is descriptive. No confidence intervals,
  hypothesis tests, or claims of statistical significance are reported.
- Bounding-box metrics derive predictions from segmentation components and
  should not be interpreted as a separately trained object detector.
- Results must not be used to guide clinical care.

## Tests

Run the unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q polyp_pipeline tests
```

The test suite covers strict dataset pairing, fold isolation, bounding-box
scaling, feature-label separation, metrics/reporting, batch sampling, and
checkpoint interruption/resumption.
