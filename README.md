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

The previously completed baseline matrix contains **13 experiments**:

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

### Auxiliary and external datasets

The additional CVC/ETIS data are sourced from the open Zenodo record
[5579392](https://zenodo.org/records/5579392), which publishes the archive
`PolypDataset.zip` and MD5 `0c45c453bc16424df7a2528966ccd939`. The archive's
`NewTRimage`/`NewTRmask` split contains 900 Kvasir cases and 550 CVC-ClinicDB
cases. Its test split contains 62 CVC-ClinicDB, 380 CVC-ColonDB, 196
ETIS-LaribPolypDB, 100 Kvasir, and 60 CVC-300 cases.

The preparation command verifies the archive checksum and pair counts,
identifies and excludes the 900 Kvasir training-set overlaps by case ID, and
rejects exact decoded-image duplicates across the selected training and test
splits. It records duplicates within an individual split without discarding
cases; the external report exposes the duplicate-pair count because those
cases are not independent observations. It extracts only the 550 CVC-ClinicDB
auxiliary-training cases plus the three held-out test sets (ClinicDB, ColonDB,
and ETIS). The archive's Kvasir test subset is deliberately not used:
Kvasir-SEG is evaluated with the existing five-fold outer cross-validation.
CVC-300 is not part of this study. The archive contains 34 exact-image
duplicate pairs in ColonDB and 3 in ETIS; 4 ColonDB pairs and all 3 ETIS pairs
have different masks. These published test-case counts are retained, but
observations sharing an image are not independent. The prepared dataset
manifest records each duplicate pair, split counts, and local paths. The
combined external report includes an `exact_image_duplicate_pairs` column.

The Zenodo record lists CC-BY 4.0; the original CVC-ClinicDB terms restrict its
use to research and education. Review the source terms before reuse or
redistribution. The archive, extracted data, and model checkpoints are local
runtime data and must not be added to the source repository.

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
implement or evaluate a Dense CRF. The completed run and its results are
summarized below; the complete comparison and subgroup tables are also saved
in [reports/advanced_benchmark](reports/advanced_benchmark/).

### Completed pretrained and transformer results

The advanced benchmark completed all six configurations with five outer folds
each on the 1,000-image dataset. Values are mean +/- sample standard deviation
across the five held-out folds. Ordinary and TTA evaluations use the same
fold-specific best checkpoint; TTA averages four transformed views before
thresholding. FPS is model-only inference throughput on the RTX 3050 Ti, not
end-to-end application latency.

| Model | Loss | Dice | TTA Dice | mIoU | Sensitivity | Pixel specificity | AP50 | Box precision | Box recall | FPS | TTA FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FPN / ResNet-34 (ImageNet) | BCE+Dice | 0.885 +/- 0.013 | **0.895 +/- 0.016** | 0.818 +/- 0.017 | 0.881 +/- 0.028 | 0.983 +/- 0.003 | 0.870 +/- 0.027 | 0.700 +/- 0.042 | 0.887 +/- 0.023 | 51.8 +/- 2.3 | 35.0 +/- 0.7 |
| DeepLabV3+ / ResNet-34 (ImageNet) | BCE+Dice | **0.886 +/- 0.008** | 0.894 +/- 0.007 | **0.821 +/- 0.009** | 0.888 +/- 0.018 | 0.982 +/- 0.001 | **0.882 +/- 0.023** | 0.732 +/- 0.057 | **0.894 +/- 0.025** | 50.2 +/- 2.0 | 32.0 +/- 0.2 |
| SegFormer / MiT-B0 (ImageNet) | BCE+Dice | 0.881 +/- 0.011 | 0.888 +/- 0.008 | 0.811 +/- 0.013 | 0.881 +/- 0.025 | 0.981 +/- 0.004 | 0.878 +/- 0.015 | 0.673 +/- 0.084 | 0.891 +/- 0.010 | **59.6 +/- 4.7** | 34.0 +/- 0.4 |
| UPerNet / MiT-B0 (ImageNet) | BCE+Dice | 0.881 +/- 0.014 | 0.887 +/- 0.013 | 0.812 +/- 0.017 | 0.878 +/- 0.019 | 0.982 +/- 0.004 | 0.866 +/- 0.027 | 0.739 +/- 0.022 | 0.879 +/- 0.027 | 45.8 +/- 0.4 | 18.9 +/- 0.1 |
| FPN / ResNet-34 (ImageNet) | Focal Tversky | 0.872 +/- 0.008 | 0.881 +/- 0.004 | 0.799 +/- 0.013 | **0.886 +/- 0.033** | 0.973 +/- 0.006 | 0.866 +/- 0.014 | 0.747 +/- 0.067 | 0.882 +/- 0.012 | 53.6 +/- 3.4 | 34.9 +/- 0.2 |
| FPN / ResNet-34 (ImageNet) | Dice + boundary | 0.879 +/- 0.008 | 0.889 +/- 0.008 | 0.809 +/- 0.010 | 0.883 +/- 0.029 | 0.976 +/- 0.010 | 0.869 +/- 0.026 | **0.762 +/- 0.068** | 0.885 +/- 0.031 | 54.9 +/- 2.8 | 35.1 +/- 0.2 |

TTA improved mean Dice in all six configurations. Across configurations, the
paired mean Dice improvement was approximately 0.008, while mean throughput
dropped by approximately 21 FPS. For the BCE+Dice FPN reference, TTA increased
Dice from 0.885 to 0.895 and sensitivity from 0.881 to 0.889, while throughput
decreased from 51.8 to 35.0 FPS. Thus TTA is an accuracy/latency tradeoff, not
a free improvement.

The pretrained FPN BCE+Dice mean Dice (0.885) is 0.045 higher than the
scratch-initialized FPN BCE+Dice result in the original benchmark (0.840).
This is a descriptive comparison using the same dataset and fold protocol,
not a significance test or an isolated causal estimate of pretraining.
DeepLabV3+ reached the highest ordinary-inference Dice (0.886) and AP50
(0.882). SegFormer delivered the highest measured ordinary FPS (59.6) with
Dice 0.881, making it a favorable speed/accuracy point in this particular
hardware and evaluation setup. Transformer performance was competitive with
the pretrained CNN models, but did not exceed the best mean Dice.

For the FPN loss ablation, focal Tversky produced the highest mean sensitivity
(0.886 versus 0.881 for BCE+Dice) but lower mean Dice (0.872 versus 0.885)
and lower pixel specificity. Dice plus boundary loss also did not improve
mean Dice (0.879). The loss changes therefore did not establish a general
improvement; their tradeoffs should be judged across metrics rather than
sensitivity alone.

Subgroup analysis remains descriptive. With TTA, FPN BCE+Dice had mean Dice
0.839 +/- 0.191 on the 196 overlapping sessile-subset images and 0.730 +/-
0.234 on the 13 images whose target boxes cover at most 2% of the frame.
The small-box sample is very limited and has high variability; these results
do not establish reliable performance on small or flat polyps. Pixel
specificity is measured on positive Kvasir-SEG frames and is not clinical
specificity on normal patients. No confidence intervals, significance tests,
external validation, or clinical claims are made.

The full-precision fold summaries and subgroup results are in
[benchmark_comparison.csv](reports/advanced_benchmark/benchmark_comparison.csv)
and
[subgroup_comparison.csv](reports/advanced_benchmark/subgroup_comparison.csv).
The chart is
[ablation_and_model_comparison.png](reports/advanced_benchmark/ablation_and_model_comparison.png).

## Specialized architecture, small-lesion, and refinement studies

The `specialized-benchmark` command adds two research implementations:

- **PraNet-style reverse attention:** ImageNet-pretrained ResNet-34 features are
  decoded from coarse to fine. Each refinement stage gates features with
  `1 - sigmoid(previous_logit)` and predicts a residual mask correction.
- **TransFuse-style hybrid fusion:** ImageNet-pretrained ResNet-34 and
  Swin-Tiny feature pyramids are projected to a shared width, fused with
  reciprocal learned gates, and decoded with auxiliary output heads.

These are documented, locally implemented PraNet-/TransFuse-style models; they
are not claimed to be bit-for-bit reproductions of the papers' official code or
checkpoints. Both emit full-resolution logits at multiple scales. The
deep-supervision objective is
`L_final + 0.4 L_aux1 + 0.2 L_aux2`, with each term using the same BCE+soft-Dice
loss and ground-truth mask. Ablation cells compare deep supervision on/off and
small-lesion crops on/off.

For training only, the crop condition selects a case with probability 0.5 when
its resized foreground mask occupies at most 2% of the frame. A square crop
centered near the mask is expanded to retain context and resized to the model
input size. Validation and outer-test images are never cropped. This is a
training sampling intervention, not an independent small-polyp test set.

### Prepare and run the joint-data study

Download and verify the Zenodo archive, then prepare the 550-case auxiliary
training set and independent external test sets:

```powershell
New-Item -ItemType Directory -Force "D:\PolypDatasets\Zenodo5579392" | Out-Null
curl.exe --fail --location --retry 5 --retry-all-errors --continue-at - `
  "https://zenodo.org/api/records/5579392/files/PolypDataset.zip/content" `
  --output "D:\PolypDatasets\Zenodo5579392\PolypDataset.zip"
$md5 = (Get-FileHash "D:\PolypDatasets\Zenodo5579392\PolypDataset.zip" -Algorithm MD5).Hash.ToLowerInvariant()
if ($md5 -ne "0c45c453bc16424df7a2528966ccd939") { throw "Zenodo archive checksum mismatch: $md5" }

python -m polyp_pipeline prepare-zenodo-datasets `
  --archive "D:\PolypDatasets\Zenodo5579392\PolypDataset.zip" `
  --kvasir-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "D:\PolypDatasets\prepared"
```

Preparation produces `clinic_train/` (550 cases), `clinic_test/` (62),
`colon_test/` (380), and `etis_test/` (196), each with paired `images/` and
`masks/` directories. It excludes the archive's 900 Kvasir cases from auxiliary
training by matching source case IDs. The five Kvasir outer folds remain
unchanged; the auxiliary training set is included only in each fold's training
portion. The fold manifests record auxiliary IDs and exact-image exclusions
against that fold's Kvasir test cases.

Run only the two matched joint-training cells (PraNet-style and
TransFuse-style) with the same five Kvasir folds. Checkpoints are saved after
each epoch; rerun the identical command to resume:

```powershell
python -m polyp_pipeline specialized-benchmark `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --aux-dataset-root "D:\PolypDatasets\prepared\clinic_train" `
  --output-dir "D:\PolypRuns\specialized_joint" `
  --folds 5 --seed 42 --size 352 `
  --batch-size 2 --epochs 30 --patience 6 --device cuda `
  --no-overlays --joint-only
```

This is joint training, not sequential pretraining and fine-tuning. The 550
ClinicDB cases are combined with each fold's Kvasir training cases; the
ClinicDB, ColonDB, and ETIS test cases are never included in training or model
selection. Run external evaluation only after checkpoints are complete:

```powershell
python -m polyp_pipeline external-evaluate `
  --input-dir "D:\PolypRuns\specialized_joint" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\clinic_test" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\colon_test" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\etis_test" `
  --batch-size 2 --device cuda
```

For comparison against the Kvasir-only architecture runs, evaluate those
checkpoints separately by setting `--input-dir "runs\specialized_benchmark"`.
External fold metrics are saved as `external_metrics.csv` and
`external_metrics_summary.json`; combine the Kvasir-only and joint results
with:

```powershell
python -m polyp_pipeline report `
  --input-dir "runs\specialized_benchmark" `
  --input-dir "D:\PolypRuns\specialized_joint" `
  --output-dir "reports\external_joint"
```

Joint training is not the same protocol as pretraining on external datasets
followed by separate Kvasir fine-tuning. The pipeline also excludes any
auxiliary image that exactly matches the current outer-test image; that
pixel-identical check does not guarantee discovery of resized or otherwise
altered duplicates.

### DenseCRF and execution

DenseCRF is optional and is applied to ordinary-inference probability maps
using Gaussian and bilateral pairwise terms. It reports a separate `*_crf`
row beside the unrefined result; it is not trained and its parameters are not
selected on the outer test folds. On Windows, `pydensecrf2` may require
Microsoft C++ Build Tools to compile:

```powershell
python -m pip install pydensecrf2==1.1
```

The Windows environment used for this run lacked those build tools, so DenseCRF
could not be installed or empirically evaluated here. Without that optional
package, omit `--dense-crf`; requesting it fails before training with an
installation message.

Run the Kvasir-only architecture and ablation matrix:

```powershell
python -m polyp_pipeline specialized-benchmark `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "runs\specialized_benchmark" `
  --folds 5 --seed 42 --size 352 `
  --batch-size 2 --epochs 30 --patience 6 --device cuda --no-overlays
```

Add `--dense-crf` only after installing the optional dependency. Monitor or
resume by rerunning the identical command and check progress with:

```powershell
python -m polyp_pipeline status --input-dir "runs\specialized_benchmark"
```

The runner checkpoints after every epoch and records outer-fold manifests,
including auxiliary training IDs and exact duplicate exclusions. The five
Kvasir-only comparison cells cover PraNet-style baseline, PraNet-style deep
supervision, PraNet-style deep supervision plus small-lesion cropping,
TransFuse-style baseline, and TransFuse-style deep supervision plus cropping.
When auxiliary datasets are supplied, two matched joint-training cells are
added. CRF, when enabled, is a paired post-processing ablation for each cell.
The comparison report uses the same standard command:

```powershell
python -m polyp_pipeline report `
  --input-dir "runs\specialized_benchmark" `
  --output-dir "reports\specialized_benchmark"
```

### Completed Kvasir-only specialized results

The five Kvasir-only configurations completed five-fold cross-validation on
the 1,000 Kvasir-SEG cases. Values are mean +/- sample standard deviation
across outer folds. The standard and TTA scores are paired evaluations of the
same fold-specific checkpoint. These rows are the Kvasir-only baseline for the
separate joint-data experiment. DenseCRF was not run because its optional
Windows extension could not be built in this environment.

| Model / ablation | Dice | TTA Dice | mIoU | Sensitivity | Pixel specificity | AP50 | FPS | TTA FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PraNet-style, no deep supervision, no crop | 0.869 +/- 0.004 | 0.883 +/- 0.011 | 0.801 +/- 0.007 | **0.885 +/- 0.024** | 0.976 +/- 0.003 | 0.850 +/- 0.015 | 49.6 +/- 0.8 | 20.5 +/- 0.1 |
| PraNet-style, deep supervision, no crop | 0.865 +/- 0.013 | 0.879 +/- 0.010 | 0.794 +/- 0.016 | 0.863 +/- 0.018 | 0.978 +/- 0.003 | 0.834 +/- 0.023 | 48.7 +/- 0.4 | 20.5 +/- 0.2 |
| PraNet-style, deep supervision, 50% small-lesion crop | 0.867 +/- 0.017 | 0.877 +/- 0.018 | 0.796 +/- 0.022 | 0.856 +/- 0.037 | 0.982 +/- 0.006 | 0.851 +/- 0.025 | 48.7 +/- 1.4 | 20.2 +/- 0.3 |
| TransFuse-style, no deep supervision, no crop | 0.879 +/- 0.018 | 0.888 +/- 0.018 | 0.812 +/- 0.021 | 0.862 +/- 0.031 | **0.986 +/- 0.004** | 0.864 +/- 0.032 | 28.1 +/- 0.6 | 7.4 +/- 0.1 |
| TransFuse-style, deep supervision, 50% small-lesion crop | **0.884 +/- 0.017** | **0.892 +/- 0.016** | **0.819 +/- 0.020** | 0.871 +/- 0.010 | 0.985 +/- 0.003 | **0.873 +/- 0.035** | 28.3 +/- 0.3 | 7.4 +/- 0.1 |

The specialized models did not reach the 0.93-0.95 target. The best
specialized result was TransFuse-style with deep supervision and cropping:
Dice 0.884 ordinary and 0.892 with TTA. This remains below the previously
measured pretrained FPN BCE+Dice with TTA (0.895); these experiments did not
establish a new best score.

The PraNet-style ablations do not show a gain from deep supervision in this
setup: adding it without cropping changed Dice from 0.869 to 0.865 and
sensitivity from 0.885 to 0.863. Adding crops to that deep-supervision run
raised Dice slightly from 0.865 to 0.867, but remained below the no-auxiliary
baseline. For TransFuse-style, deep supervision and cropping together raised
Dice from 0.879 to 0.884 and sensitivity from 0.862 to 0.871. Since both
interventions changed together in that comparison, their individual effects
cannot be separated.

Four-view TTA increased Dice in all five configurations; the mean paired
improvement was about 0.011. It reduced measured throughput substantially:
from about 49 FPS to 20 FPS for PraNet-style and 28 FPS to 7 FPS for
TransFuse-style. The TransFuse-style TTA score was still below the earlier FPN
TTA result, and the latency cost is significant.

Small-box results remain uncertain because only 13 cases meet the <=2% target
bounding-box area subgroup criterion. For example, the best specialized TTA
small-box Dice was 0.735 for TransFuse-style without the two training
interventions; with deep supervision and cropping it was 0.639. Sessile
subset results are likewise descriptive on 196 overlapping Kvasir-SEG images,
not an external cohort. These data do not show reliable small/flat-lesion
performance gains. Specificity remains pixel-level on polyp-positive frames,
not clinical specificity on normal patients. Fold SDs are descriptive; no
significance tests or confidence intervals were computed.

The final tables and plot are committed under
[reports/specialized_benchmark](reports/specialized_benchmark/):
[benchmark_comparison.csv](reports/specialized_benchmark/benchmark_comparison.csv),
[subgroup_comparison.csv](reports/specialized_benchmark/subgroup_comparison.csv),
and [ablation_and_model_comparison.png](reports/specialized_benchmark/ablation_and_model_comparison.png).

### Joint training and held-out external results

Both joint-data configurations completed five-fold training: PraNet-style
ResNet-34 and TransFuse-style ResNet-34/Swin-Tiny, each with deep supervision
and 50% small-lesion crop sampling. Every fold trained with the 550 auxiliary
ClinicDB cases plus its Kvasir training portion. The 62-case ClinicDB,
380-case ColonDB, and 196-case ETIS test splits were held out for evaluation.
The Kvasir-only comparison uses the same architecture and ablation settings
without auxiliary training data. Values are mean +/- sample standard
deviation across the five fold-specific checkpoints; the same external cases
are scored by each checkpoint, so fold SD describes model variation, not
uncertainty from five independent external cohorts.

| External set (cases; exact-image duplicate pairs) | Model | Kvasir-only Dice | Joint Dice | Joint TTA Dice |
|---|---|---:|---:|---:|
| ClinicDB (62; 0) | PraNet-style | 0.728 +/- 0.034 | 0.840 +/- 0.033 | **0.854 +/- 0.026** |
| ClinicDB (62; 0) | TransFuse-style | 0.738 +/- 0.017 | **0.854 +/- 0.029** | 0.852 +/- 0.033 |
| ColonDB (380; 34) | PraNet-style | 0.596 +/- 0.035 | 0.688 +/- 0.033 | 0.703 +/- 0.014 |
| ColonDB (380; 34) | TransFuse-style | 0.664 +/- 0.028 | 0.713 +/- 0.039 | **0.719 +/- 0.029** |
| ETIS (196; 3) | PraNet-style | 0.473 +/- 0.054 | 0.623 +/- 0.011 | **0.645 +/- 0.018** |
| ETIS (196; 3) | TransFuse-style | 0.578 +/- 0.051 | 0.623 +/- 0.052 | 0.638 +/- 0.065 |

Joint training raised ordinary Dice against the matched Kvasir-only model in
all six architecture/dataset comparisons, by 0.044 to 0.151. This is a
descriptive result, not a significance test. TTA raised joint Dice in five of
six comparisons; TransFuse-style ClinicDB Dice decreased slightly from 0.854
to 0.852. Best joint TTA Dice was 0.854 on ClinicDB, 0.719 on ColonDB, and
0.645 on ETIS. All remain below the 0.93-0.95 research target, and the
ColonDB/ETIS results show a substantial cross-dataset performance drop.

On Kvasir outer-fold evaluation, joint PraNet-style Dice was
0.881 +/- 0.012 (TTA 0.889 +/- 0.007); joint TransFuse-style Dice was
0.884 +/- 0.015 (TTA 0.890 +/- 0.012). Joint training did not surpass the
previous pretrained FPN BCE+Dice Kvasir TTA result of 0.895.

The external report includes mIoU, sensitivity, pixel specificity, AP50,
bounding-box precision/recall, ordinary and TTA throughput, and fold standard
deviations. For example, joint TTA sensitivity/AP50 were 0.879/0.882 for
PraNet-style on ClinicDB, 0.576/0.697 for TransFuse-style on ColonDB, and
0.840/0.604 for PraNet-style on ETIS. Pixel specificity is computed on
polyp-positive frames and must not be interpreted as clinical specificity on
normal patients. TTA is a four-view accuracy/latency tradeoff; measured TTA
throughput was about 22 FPS for PraNet-style and 8 FPS for TransFuse-style.

The archive's test data contain 34 exact-image duplicate pairs in ColonDB and
3 in ETIS; these cases are retained and counted rather than removed. Four
ColonDB pairs and all three ETIS pairs have different masks, so repeated
images are not independent observations and can reflect annotation
differences. The report exposes duplicate-pair counts for interpretation.
There are no confidence intervals or significance tests, and these results
are not clinical validation.

The combined report is in
[reports/external_joint](reports/external_joint/):
[external_test_comparison.csv](reports/external_joint/external_test_comparison.csv)
contains the full external metric comparison,
[benchmark_comparison.csv](reports/external_joint/benchmark_comparison.csv)
contains Kvasir fold results, and
[subgroup_comparison.csv](reports/external_joint/subgroup_comparison.csv)
contains sessile/small-box subgroup results. The plots are in
[ablation_and_model_comparison.png](reports/external_joint/ablation_and_model_comparison.png).
Datasets and checkpoints remain outside the repository.

## Frequency, boundary, and EMA-consistency study

The current Swin branch is a ResNet-34/Swin-Tiny hybrid (TransFuse-style).
This experiment keeps that backbone and adds a lightweight fixed-Haar
frequency-fusion block after each fused feature scale; it does not claim to
implement Vision Mamba or a paper's exact VMDU-Net/Swin-MSEG-Pro architecture.
For each channel, a 2 x 2 Haar transform produces one low-frequency component
and three detail components (horizontal, vertical, and diagonal). Learned
1 x 1 projections combine the low-frequency and detail branches, and a
learned gate applies the result residually to the spatial feature. Odd-sized
feature maps are replicate-padded before the transform and restored to their
original size afterward. This is a feature-fusion inductive bias, not a
guaranteed glare-removal operation; the existing specular preprocessing
remains separate.

For one 2 x 2 patch with values `(a, b; c, d)`, the normalized Haar
coefficients are
`LL=(a+b+c+d)/2`, `LH=(a-b+c-d)/2`,
`HL=(a+b-c-d)/2`, and `HH=(a-b-c+d)/2`.
At each feature scale, the three detail maps are concatenated and projected
separately from `LL`; both branches are resized back to the feature resolution.
The fusion is
`X' = ReLU(BN(X + sigmoid(Conv1x1([X, Xlow, Xhigh])) * (Xlow + Xhigh)))`.

The boundary arm uses the full-resolution objective
`L = L_BCE + L_Dice + 0.1 L_signed-distance`. The signed Euclidean distance
map is negative inside the foreground, positive outside, and normalized by
the larger image dimension; `L_signed-distance = mean(sigmoid(z) * d)`.
Distance maps are computed only from the training batch's ground-truth masks.
The existing Dice-plus-distance loss benchmark remains unchanged, so its
previous results are not silently redefined.

The final arm adds a Mean Teacher-style consistency regularizer. An EMA teacher
with decay 0.99 predicts a weak photometric view; the student receives a
stronger photometric view with the same geometry. Mean-squared probability
consistency is applied only where the teacher probability is at least 0.7 or
at most 0.3, and its weight 0.1 is linearly ramped up over five epochs. The
teacher is updated after student optimizer steps; validation selects the EMA
weights and the outer-fold test data are used only once for evaluation.
Kvasir-SEG and the auxiliary datasets available here are labeled, and there is
no separate unlabeled training corpus. Accordingly, this is consistency
regularization on labeled training-fold images, not evidence of a true
semi-supervised gain or of pseudo-labeling on held-out data.

The comparison consists of a matched pretrained TransFuse/Swin BCE+Dice
reference, the same model with BCE+Dice+boundary loss, frequency fusion with
that boundary loss, and frequency fusion plus the EMA consistency term. The
reference reuses the already completed Kvasir-only TransFuse run; the three
new conditions share its five image-level folds, seed 42, CLAHE preprocessing,
ImageNet encoders, no small-polyp crop, no deep supervision, and four-view TTA.
Each fold trains only on its training partition, selects checkpoints using
its inner validation subset, and evaluates the outer fold once. The run is
resumable from atomic epoch checkpoints and records the fold manifests,
training history, configuration, and metrics outside Git.
If an epoch checkpoint is unreadable, the runner records a recovery event and
warm-starts from the valid best-validation weights, resetting optimizer and
random-number state; that fallback is not an exact continuation.

```powershell
python -m polyp_pipeline frequency-boundary-benchmark `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "D:\PolypRuns\frequency_boundary" `
  --folds 5 --seed 42 --size 352 `
  --batch-size 2 --epochs 30 --patience 6 `
  --device cuda --no-overlays
```

Rerun the same command after interruption to continue from the latest epoch or,
if needed, recover from the best-validation checkpoint as described above.
To combine the original matched baseline with the new ablations:

```powershell
python -m polyp_pipeline report `
  --input-dir "runs\specialized_benchmark\deep\transfuse_resnet34_swin_tiny__ds-0__small-crop-0.0__kvasir_only" `
  --input-dir "D:\PolypRuns\frequency_boundary" `
  --output-dir "reports\frequency_boundary"
```

The resulting report contains fold mean and sample SD for segmentation,
localization, speed, and TTA, plus sessile/small-box subgroup summaries.
These results must be interpreted alongside the earlier boundary-loss
ablation, which did not improve mean Dice, and should not be assumed to reach
0.93-0.95. Literature scores from different splits, preprocessing,
post-processing, or training data are not directly comparable to this
five-fold protocol.

The target Dice range of 0.93-0.95 is an experimental goal, not a guaranteed
outcome. Report only completed held-out results; do not tune architecture,
thresholds, or CRF parameters against outer-test metrics.

### Measured frequency, boundary, and EMA results

All four Kvasir conditions below use the same five outer folds and inner
validation splits. Values are mean +/- sample standard deviation across folds.

| TransFuse-style condition | Ordinary Dice | TTA Dice | mIoU | Sensitivity | Pixel specificity | AP50 | FPS | TTA FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched BCE+Dice baseline | 0.8787 +/- 0.0180 | 0.8879 +/- 0.0177 | 0.8124 +/- 0.0215 | 0.8622 +/- 0.0306 | 0.9856 +/- 0.0041 | 0.8638 +/- 0.0319 | 28.1 +/- 0.6 | 7.4 +/- 0.1 |
| Spatial BCE+Dice+boundary | 0.8738 +/- 0.0218 | 0.8827 +/- 0.0190 | 0.8041 +/- 0.0266 | 0.8688 +/- 0.0246 | 0.9814 +/- 0.0039 | 0.8630 +/- 0.0343 | 24.1 +/- 1.5 | 6.3 +/- 0.5 |
| Haar frequency + BCE+Dice+boundary | **0.8891 +/- 0.0067** | **0.8948 +/- 0.0095** | **0.8243 +/- 0.0083** | 0.8782 +/- 0.0166 | 0.9850 +/- 0.0011 | **0.8767 +/- 0.0202** | 23.6 +/- 0.3 | 6.2 +/- 0.0 |
| Haar frequency + boundary + EMA consistency | 0.8782 +/- 0.0208 | 0.8851 +/- 0.0177 | 0.8130 +/- 0.0273 | 0.87197 +/- 0.0306 | 0.9838 +/- 0.0032 | 0.8596 +/- 0.0323 | 23.4 +/- 0.8 | 6.2 +/- 0.2 |

The fixed-Haar condition improved ordinary Dice by 0.0104 over the matched
baseline and 0.0153 over the spatial boundary arm. TTA raised the Haar
condition to 0.8948, close to the project-wide pretrained FPN BCE+Dice TTA
result of 0.895 but not above it. The boundary-only spatial result did not
improve over the matched BCE+Dice baseline in this experiment, consistent with
the earlier boundary-loss ablation. EMA consistency did not improve the Haar
arm: it was lower than Haar boundary by 0.0109 Dice and 0.0097 TTA Dice.

TTA and Haar fusion both add a throughput cost. The matched baseline measured
28.1 ordinary FPS, while the Haar result measured 23.6 ordinary FPS and 6.2
FPS for four-view TTA. These measurements are hardware-dependent and exclude
dataset loading and writing.

### Measured external-domain results

The completed frequency ablations were evaluated on 62 ClinicDB, 380 ColonDB,
and 196 ETIS held-out images using the five fold-specific Kvasir checkpoints.
The tables below report mean +/- sample SD across checkpoints.

| Dataset | Spatial boundary ordinary / TTA Dice | Haar boundary ordinary / TTA Dice | Haar boundary + EMA ordinary / TTA Dice |
|---|---:|---:|---:|
| ClinicDB | 0.6889 +/- 0.0252 / 0.7142 +/- 0.0289 | **0.7535 +/- 0.0134 / 0.7620 +/- 0.0156** | 0.7158 +/- 0.0403 / 0.7232 +/- 0.0360 |
| ColonDB | 0.6223 +/- 0.0429 / 0.6322 +/- 0.0373 | **0.6624 +/- 0.0218 / 0.6627 +/- 0.0250** | 0.6412 +/- 0.0258 / 0.6446 +/- 0.0310 |
| ETIS-LaribPolypDB | 0.4908 +/- 0.0746 / 0.4884 +/- 0.0534 | **0.5734 +/- 0.0419 / 0.5542 +/- 0.0369** | 0.5121 +/- 0.0748 / 0.4899 +/- 0.0706 |

The fixed-Haar boundary arm was strongest on all three external datasets.
Joint-dataset or domain-adaptation checkpoints are separate previous results,
not mixed into this table. The 0.93-0.95 target was not reached in this
Kvasir-only study. External results remain substantially lower than Kvasir
performance and must not be presented as clinical validation.

### Explicit morphological edge supervision

The train command also provides `bce_dice_edge_boundary` for a controlled
edge-aware extension to the existing signed-distance objective. Given a binary
mask `y`, its 3 x 3 morphological gradient is
`E(y) = max_pool(y) - min_pool(y)`, implemented as
`max_pool(y) + max_pool(-y)`. The model's soft edge map is the same operation
applied to its sigmoid probabilities `p`. The added loss is
`L_edge = BCE(E(p), E(y)) + (1 - Dice(E(p), E(y)))`, and the full objective is
`L = BCE(z,y) + (1 - Dice(p,y)) + 0.1 mean(p d) + 0.1 L_edge`, where `d` is the
normalized signed distance map described above. This is a differentiable
morphological edge penalty, not a Canny edge detector or a Hausdorff-distance
implementation. Ground-truth edge maps are constructed from each supervised
training mask only.

Run the new loss on the pretrained Swin/Haar hybrid with five-fold CV, EMA
consistency, and TTA:

```powershell
python -m polyp_pipeline train `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "D:\PolypRuns\edge_boundary_ema" `
  --model transfuse_frequency_swin_tiny `
  --loss bce_dice_edge_boundary `
  --folds 5 --seed 42 --size 352 `
  --batch-size 2 --epochs 30 --patience 6 `
  --pretrained-encoder --consistency-weight 0.1 `
  --ema-decay 0.99 --consistency-rampup-epochs 5 `
  --test-time-augmentation --device cuda --no-overlays
```

The edge-loss arm is a separate condition from the earlier Haar study. Its
performance must be recorded from its own outer-fold metrics, not inferred from
the prior study.

## Genuine unlabeled-data training

The pipeline can download HyperKvasir's image-only corpus from its public OSF
folder. The downloader samples from seeded, randomly selected pages spread
across the published listing, verifies each downloaded file against its
published SHA-256 and size, and writes a `download_manifest.json`. It defaults
to a reproducible 1,000-image sample; use `--all-images` for the full release
(approximately 99,000 published entries / 29.4 GB). Zero-byte OSF placeholder
records are excluded; the manifest distinguishes the published listing count
from the usable sample count.

```powershell
python -m polyp_pipeline download-hyperkvasir `
  --output-dir "D:\PolypDatasets\HyperKvasir\unlabeled_clean" `
  --max-images 1000 --seed 42 --workers 8
```

HyperKvasir is licensed CC BY 4.0. Cite Borgli et al., *HyperKvasir, a
comprehensive multi-class image and video dataset for gastrointestinal
endoscopy*, Scientific Data 7, 283 (2020), DOI
[10.1038/s41597-020-00622-y](https://doi.org/10.1038/s41597-020-00622-y).
The images are a research dataset, not clinical evidence.
The unlabeled collection spans gastrointestinal endoscopy and is not guaranteed
to contain colon-only frames; this domain mismatch is a limitation of the
pseudo-label experiment.

For each Kvasir-SEG run, the pipeline hashes the decoded pixels of every
candidate unlabeled image and excludes exact duplicates of **all Kvasir-SEG
cases**, including the current outer test fold, as well as duplicates within
the unlabeled pool. Provide each held-out external cohort at training time too;
external evaluation repeats the overlap check and refuses to score an external
cohort if it overlaps the unlabeled training corpus:

```powershell
python -m polyp_pipeline train `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "D:\PolypRuns\edge_boundary_hyperkvasir" `
  --model transfuse_frequency_swin_tiny `
  --loss bce_dice_edge_boundary `
  --folds 5 --seed 42 --size 352 `
  --batch-size 2 --epochs 30 --patience 6 `
  --pretrained-encoder --consistency-weight 0.1 `
  --ema-decay 0.99 --consistency-rampup-epochs 5 `
  --unlabeled-image-dir "D:\PolypDatasets\HyperKvasir\unlabeled_clean" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\clinic_test" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\colon_test" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\etis_test" `
  --test-time-augmentation --device cuda --no-overlays
```

The outer and inner splits remain image-disjoint. The fixed unlabeled pool is
filtered once against every Kvasir case (not only the current fold), which
prevents transductive exposure to any Kvasir validation/test image. Before each
supervised batch, the EMA teacher labels a weak photometric view of an unlabeled
image; the student receives a stronger view with the same geometry. For teacher
probability `p`, normalized binary entropy is
`H(p) = -[p log(p) + (1-p) log(1-p)] / log(2)`. Pixels with `p <= 0.3` or
`p >= 0.7` receive weight `1-H(p)`; uncertain pixels are ignored. The
unsupervised objective adds confidence-weighted pseudo-label BCE and
probability-consistency MSE. Its weight is ramped over the first configured
epochs and added to supervised loss. The run writes the exact deduplication
decisions and content hashes to `unlabeled_data_manifest.json`; checkpoints
and run configuration include the unlabeled corpus fingerprint, so an
incompatible resume is rejected.

The duplicate guard detects exact decoded-pixel matches, not resized or
recompressed near-duplicates. It cannot certify absence of video-adjacent or
near-duplicate frames; review dataset provenance and split definitions before
making external generalization claims. Pass every known external test cohort
with `--exclude-unlabeled-dataset-root`, and keep those cohorts out of all
training and model-selection decisions.

No semi-supervised performance gain is assumed. The edge-loss arm and the
HyperKvasir pseudo-label arm must be evaluated using their saved outer-fold
results; a target Dice score is not a guaranteed outcome.

### HyperKvasir experiment status

A seeded 1,000-image HyperKvasir sample was downloaded to
`D:\PolypDatasets\HyperKvasir\unlabeled_clean`. All 1,000 image files passed
published size/SHA-256 verification and OpenCV decoding. After exact decoded
pixel deduplication against Kvasir-SEG and the three configured external test
cohorts, all 1,000 remained eligible for training.

The initial five-fold CUDA pilot uses the frequency-fusion Swin hybrid,
ImageNet-pretrained encoders, CLAHE, BCE+Dice+boundary loss, EMA consistency
with unlabeled images, four-view TTA, batch size 4, and at most 10 epochs per
fold. It is an exploratory, shorter run than the documented 30-epoch command.
The first two outer folds are complete:

All five outer folds completed and were scored once on their held-out images.
Mean +/- sample standard deviation across folds:

| Metric | Ordinary | Four-view TTA |
|---|---:|---:|
| Dice | 0.8786 +/- 0.0105 | 0.8869 +/- 0.0114 |
| mIoU | 0.8085 +/- 0.0138 | 0.8199 +/- 0.0153 |
| Sensitivity | 0.9133 +/- 0.0123 | 0.9212 +/- 0.0145 |
| Specificity | 0.9740 +/- 0.0040 | 0.9757 +/- 0.0034 |
| AP50 | 0.8600 +/- 0.0138 | 0.8678 +/- 0.0252 |
| Bounding-box precision | 0.7585 +/- 0.0126 | 0.8046 +/- 0.0380 |
| Bounding-box recall | 0.8760 +/- 0.0148 | 0.8818 +/- 0.0237 |
| FPS | 36.9 +/- 13.4 | 10.1 +/- 3.6 |

The frequency-fusion Swin hybrid has 49.78M parameters and approximately
23.86 GFLOPs per 352 x 352 image by THOP.

| Fold | Dice | TTA Dice |
|---:|---:|---:|
| 1 | 0.8861 | 0.8920 |
| 2 | 0.8726 | 0.8877 |
| 3 | 0.8928 | 0.9020 |
| 4 | 0.8740 | 0.8811 |
| 5 | 0.8673 | 0.8718 |

The small-box subgroup (13 images with annotated box area <=2% of the frame)
scored Dice 0.5327 +/- 0.2913, rising to 0.5630 +/- 0.2734 with TTA. This
small subgroup has high variability and demonstrates the persistent small-lesion
failure mode. TTA improves the fold-mean Dice by 0.0084 at roughly 3.7x lower
inference throughput. DataLoader worker processes failed because the system
paging file was insufficient, so the reproducible resume command uses
`--num-workers 0`.

### Matched supervised ablation

A second, matched five-fold control was run with the same model, loss, folds,
seed, preprocessing, image size, pretrained encoders, batch size, and 10-epoch
cap, but without the EMA consistency term or unlabeled images. Mean +/- sample
standard deviation:

| Training condition | Dice | TTA Dice | Sensitivity | Specificity | AP50 |
|---|---:|---:|---:|---:|---:|
| Supervised only | 0.8747 +/- 0.0082 | 0.8839 +/- 0.0055 | 0.8877 +/- 0.0384 | 0.9764 +/- 0.0119 | 0.8555 +/- 0.0116 |
| EMA + unlabeled consistency | 0.8786 +/- 0.0105 | 0.8869 +/- 0.0114 | 0.9133 +/- 0.0123 | 0.9740 +/- 0.0040 | 0.8600 +/- 0.0138 |
| Paired fold delta (EMA - supervised) | +0.0039 +/- 0.0128 | +0.0030 +/- 0.0136 | - | - | - |

The EMA-plus-unlabeled condition improved ordinary Dice in 4/5 folds and TTA
Dice in 3/5. It had better fold-mean sensitivity, but slightly lower mean
specificity. This is a modest, noisy result and does not establish a
statistically significant gain. The EMA arm includes consistency on labeled
training views as well as pseudo-label consistency on unlabeled images; this
two-arm comparison does not isolate the unlabeled-image contribution from the
EMA regularizer. The TTA small-box Dice was lower for the EMA arm
(0.5630 +/- 0.2734) than the supervised control (0.6788 +/- 0.2817), so this
pilot does not support a small-lesion benefit. The 0.93-0.95 target was not
reached.

The comparison report and plotted summary are
[benchmark_comparison.csv](reports/hyperkvasir_comparison/benchmark_comparison.csv),
[subgroup_comparison.csv](reports/hyperkvasir_comparison/subgroup_comparison.csv),
and [ablation_and_model_comparison.png](reports/hyperkvasir_comparison/ablation_and_model_comparison.png).
The pilot-only metrics are also preserved under
[reports/hyperkvasir_pilot10/](reports/hyperkvasir_pilot10).

Run or resume this exact pilot configuration:

```powershell
python -m polyp_pipeline train `
  --dataset-root "kvasir-seg\Kvasir-SEG" `
  --output-dir "D:\PolypRuns\edge_boundary_hyperkvasir_pilot10" `
  --model transfuse_frequency_swin_tiny `
  --loss bce_dice_edge_boundary `
  --folds 5 --seed 42 --size 352 --preprocessing clahe `
  --batch-size 4 --epochs 10 --patience 6 --pretrained-encoder `
  --consistency-weight 0.1 --ema-decay 0.99 --consistency-rampup-epochs 5 `
  --unlabeled-image-dir "D:\PolypDatasets\HyperKvasir\unlabeled_clean" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\clinic_test" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\colon_test" `
  --exclude-unlabeled-dataset-root "D:\PolypDatasets\prepared\etis_test" `
  --test-time-augmentation --device cuda --num-workers 0 `
  --no-overlays --no-predictions
```

The command is resumable: rerun it with the same arguments after interruption.
Epoch checkpoints, fold results, and the unlabeled-image exclusion manifest
remain in the output directory outside Git.

External evaluation was run only after all five Kvasir checkpoints completed.
The same fixed external images are scored by each of the five Kvasir fold
models, so the standard deviation reflects model/fold variation, not sampling
variation among independent external cohorts. Dice is mean +/- sample SD across
the five fold-specific models:

| Held-out dataset | Images | Dice | TTA Dice | mIoU | Sensitivity | AP50 |
|---|---:|---:|---:|---:|---:|---:|
| ClinicDB | 62 | 0.7455 +/- 0.0230 | 0.7637 +/- 0.0180 | 0.6557 | 0.7647 | 0.7228 |
| ColonDB | 380 | 0.6886 +/- 0.0195 | 0.6857 +/- 0.0180 | 0.5934 | 0.6676 | 0.6698 |
| ETIS-LaribPolypDB | 196 | 0.6218 +/- 0.0417 | 0.6307 +/- 0.0417 | 0.5300 | 0.8729 | 0.5913 |

TTA did not improve ColonDB Dice. The external data roots contained 34
within-dataset exact duplicate pairs in ColonDB and 3 in ETIS; those cases were
retained and the duplicate counts are recorded in `external_eval_config.json`,
so these scores are not fully independent across images. None of the external
images had an exact decoded-pixel match in the downloaded unlabeled pool.
External results are research estimates, not clinical validation.
Fold-by-fold results and duplicate metadata are saved in
[external_test_comparison.csv](reports/hyperkvasir_comparison/external_test_comparison.csv)
and the run-local `external_eval_config.json`.

Reproduce the external evaluation after training:

```powershell
python -m polyp_pipeline external-evaluate `
  --input-dir "D:\PolypRuns\edge_boundary_hyperkvasir_pilot10" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\clinic_test" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\colon_test" `
  --external-test-dataset-root "D:\PolypDatasets\prepared\etis_test" `
  --batch-size 2 --device cuda --num-workers 0
```

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
| SegFormer | `Segformer` | MiT-B0 |
| UPerNet | `UPerNet` | MiT-B0 |

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

- The completed 13-experiment benchmark used randomly initialized encoders.
  The separate completed advanced benchmark uses ImageNet-pretrained encoders
  and adds SegFormer/MiT-B0 and UPerNet/MiT-B0 transformer models with four-view
  TTA. All six configurations have five outer-fold evaluations.
- HarDNet-MSEG, Polyp-PVT, and MedSAM 2 were not included. SegFormer and
  UPerNet are general transformer segmentation architectures, not substitutes
  for those polyp-specific or foundation-model implementations.
- ClinicDB, ColonDB, and ETIS were evaluated as held-out external sets for the
  completed EMA/unlabeled pilot, but are not independent clinical cohorts;
  ColonDB and ETIS include within-dataset exact duplicate pairs. There is no
  normal-patient control set or clinical validation.
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
