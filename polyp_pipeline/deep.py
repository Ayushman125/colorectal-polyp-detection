from __future__ import annotations

import csv
import copy
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import random
import re
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .data import (
    Case,
    KvasirDataset,
    NonSingletonBatchSampler,
    UnlabeledImageDataset,
    create_outer_folds,
    inner_train_validation_split,
    write_fold_manifest,
)
from .metrics import (
    average_precision_50,
    boxes_from_mask,
    detection_precision_recall,
    predicted_boxes_with_scores,
    scale_boxes,
    segmentation_counts,
    segmentation_metrics,
)
from .models import (
    DEEP_SUPERVISION_MODELS,
    MODEL_SPECS,
    build_model,
    segmentation_objective,
)


def _final_logits(
    outputs: torch.Tensor | tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs[0]


def _photometric_view(images: torch.Tensor, strength: float) -> torch.Tensor:
    batch_size, channels = images.shape[:2]
    contrast = 1.0 + torch.randn(
        (batch_size, 1, 1, 1), device=images.device, dtype=images.dtype
    ) * strength
    brightness = torch.randn(
        (batch_size, channels, 1, 1), device=images.device, dtype=images.dtype
    ) * strength
    return (images * contrast + brightness).clamp(-5.0, 5.0)


def _confidence_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.7,
) -> torch.Tensor:
    teacher_probabilities = torch.sigmoid(teacher_logits.float()).detach()
    student_probabilities = torch.sigmoid(student_logits.float())
    confidence = (
        (teacher_probabilities >= confidence_threshold)
        | (teacher_probabilities <= 1.0 - confidence_threshold)
    ).to(dtype=student_probabilities.dtype)
    squared_error = (student_probabilities - teacher_probabilities).square()
    return (squared_error * confidence).sum() / confidence.sum().clamp_min(1.0)


def _uncertainty_pseudo_label_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.7,
) -> torch.Tensor:
    teacher_probabilities = torch.sigmoid(teacher_logits.float()).detach()
    student_probabilities = torch.sigmoid(student_logits.float())
    epsilon = torch.finfo(teacher_probabilities.dtype).eps
    clipped_probabilities = teacher_probabilities.clamp(epsilon, 1.0 - epsilon)
    entropy = -(
        clipped_probabilities * clipped_probabilities.log()
        + (1.0 - clipped_probabilities)
        * (1.0 - clipped_probabilities).log()
    ) / np.log(2.0)
    confident = (
        (teacher_probabilities >= confidence_threshold)
        | (teacher_probabilities <= 1.0 - confidence_threshold)
    )
    confidence_weights = (1.0 - entropy).clamp_min(0.0) * confident
    pseudo_labels = (teacher_probabilities >= 0.5).to(student_probabilities.dtype)
    pseudo_label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits.float(), pseudo_labels, reduction="none"
    )
    consistency_loss = (student_probabilities - teacher_probabilities).square()
    weighted_loss = (pseudo_label_loss + consistency_loss) * confidence_weights
    return weighted_loss.sum() / confidence_weights.sum().clamp_min(1.0)


@torch.no_grad()
def _update_ema_model(
    teacher: torch.nn.Module, student: torch.nn.Module, decay: float
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.lerp_(student_parameter, 1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.lerp_(student_buffer, 1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def _image_content_hash(case: Case | Path) -> str:
    image_path = case.image_path if isinstance(case, Case) else case
    case_id = case.case_id if isinstance(case, Case) else str(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image for case {case_id}")
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _filter_unlabeled_images(
    image_paths: list[Path],
    image_root: Path | None,
    protected_cases: list[Case],
    evaluation_cases: list[Case],
) -> tuple[list[Path], list[dict[str, str]], dict[str, object]]:
    if not image_paths:
        raise ValueError("The unlabeled image directory contains no supported images.")
    resolved_paths = sorted(path.resolve() for path in image_paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Unlabeled image paths must be unique")
    root = (
        image_root.resolve()
        if image_root is not None
        else Path(os.path.commonpath([str(path.parent) for path in resolved_paths]))
    )
    protected_hashes: dict[str, str] = {}
    for case in protected_cases:
        protected_hashes.setdefault(
            _image_content_hash(case), f"case:{case.case_id}"
        )

    eligible_paths: list[Path] = []
    manifest_entries: list[dict[str, str]] = []
    seen_unlabeled_hashes: dict[str, str] = {}
    eligible_hashes: list[tuple[str, str]] = []
    excluded_counts: dict[str, int] = {}
    for image_path in resolved_paths:
        image_hash = _image_content_hash(image_path)
        try:
            image_id = image_path.relative_to(root).as_posix()
        except ValueError:
            image_id = str(image_path)
        reason = ""
        if image_hash in protected_hashes:
            reason = f"duplicate:{protected_hashes[image_hash]}"
        elif image_hash in seen_unlabeled_hashes:
            reason = f"duplicate:unlabeled:{seen_unlabeled_hashes[image_hash]}"
        else:
            seen_unlabeled_hashes[image_hash] = image_id
            eligible_paths.append(image_path)
            eligible_hashes.append((image_id, image_hash))
        if reason:
            exclusion_type = reason.split(":", 1)[0]
            excluded_counts[exclusion_type] = excluded_counts.get(exclusion_type, 0) + 1
        manifest_entries.append(
            {
                "image_id": image_id,
                "decoded_image_sha256": image_hash,
                "status": "excluded" if reason else "included",
                "reason": reason,
            }
        )

    if len(eligible_paths) < 2:
        raise ValueError(
            "At least two unique unlabeled images must remain after duplicate "
            "exclusion against Kvasir-SEG and configured evaluation cohorts."
        )
    eligible_manifest_hash = hashlib.sha256(
        "\n".join(f"{image_id}\t{digest}" for image_id, digest in eligible_hashes)
        .encode("utf-8")
    ).hexdigest()
    dataset_config: dict[str, object] = {
        "directory": str(root),
        "requested_image_count": len(resolved_paths),
        "eligible_image_count": len(eligible_paths),
        "excluded_exact_duplicate_count": len(resolved_paths) - len(eligible_paths),
        "excluded_by_reason": excluded_counts,
        "eligible_manifest_sha256": eligible_manifest_hash,
        "duplicate_check": "SHA-256 of decoded BGR pixel dimensions and bytes",
        "excluded_evaluation_roots": sorted(
            {
                str(case.image_path.parent.parent.resolve())
                for case in evaluation_cases
            }
        ),
    }
    return eligible_paths, manifest_entries, dataset_config


def _atomic_json_write(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _log_progress(path: Path, event: str, **details: object) -> None:
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gaussian"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _scaled_annotation_boxes(
    case: Case, size: int, target_mask: np.ndarray
) -> list[tuple[int, int, int, int]]:
    if not case.bbox_annotations:
        return boxes_from_mask(target_mask)
    if not case.image_width or not case.image_height:
        raise ValueError(f"Bounding-box dimensions are missing for {case.case_id}")
    return scale_boxes(
        case.bbox_annotations, case.image_width, case.image_height, size
    )


def _model_complexity(model: torch.nn.Module, image_size: int) -> tuple[float, float]:
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError(
            "Install the requirements.txt dependencies to profile model complexity."
        ) from error

    profile_model = copy.deepcopy(model).cpu()
    profile_model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        operations, parameters = profile(profile_model, inputs=(dummy,), verbose=False)
    return float(parameters / 1_000_000), float(operations / 1_000_000_000)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _case_map(cases: list[Case]) -> dict[str, Case]:
    return {case.case_id: case for case in cases}


def _make_loader(
    cases: list[Case],
    size: int,
    preprocessing: str,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    workers: int = 0,
    imagenet_normalize: bool = False,
    small_polyp_crop_probability: float = 0.0,
) -> DataLoader:
    dataset = KvasirDataset(
        cases,
        size=size,
        preprocessing=preprocessing,
        augment=augment,
        remove_specular=True,
        imagenet_normalize=imagenet_normalize,
        small_polyp_crop_probability=small_polyp_crop_probability,
    )
    loader_options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=NonSingletonBatchSampler(
                len(dataset), batch_size=batch_size, shuffle=True
            ),
            **loader_options,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )


def _make_unlabeled_loader(
    image_paths: list[Path],
    size: int,
    preprocessing: str,
    batch_size: int,
    workers: int,
    imagenet_normalize: bool,
) -> DataLoader:
    dataset = UnlabeledImageDataset(
        image_paths,
        size=size,
        preprocessing=preprocessing,
        augment=True,
        imagenet_normalize=imagenet_normalize,
    )
    return DataLoader(
        dataset,
        batch_sampler=NonSingletonBatchSampler(
            len(dataset), batch_size=batch_size, shuffle=True
        ),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _validate_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float
) -> float:
    model.eval()
    total_dice = 0.0
    count = 0
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = _final_logits(model(images))
            predictions = torch.sigmoid(logits).cpu().numpy()[:, 0] >= threshold
            targets = masks.numpy()[:, 0] > 0
            for target, prediction in zip(targets, predictions):
                total_dice += segmentation_metrics(target, prediction)["dice"]
                count += 1
    return total_dice / max(1, count)


def _predict_outer_test(
    model: torch.nn.Module,
    cases: list[Case],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    size: int,
    write_overlays: bool,
    sessile_case_ids: set[str],
    test_time_augmentation: bool = False,
    dense_crf: bool = False,
    fold_number: int | None = None,
    write_predictions: bool = True,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    sum_counts = np.zeros(4, dtype=np.int64)
    image_metrics: list[dict[str, float]] = []
    per_image_rows: list[dict[str, object]] = []
    detection_predictions: list[
        tuple[str, tuple[int, int, int, int], float]
    ] = []
    detection_targets: list[tuple[str, tuple[int, int, int, int]]] = []
    inference_seconds = 0.0
    crf_seconds = 0.0
    crf_image_metrics: list[dict[str, float]] = []
    crf_sum_counts = np.zeros(4, dtype=np.int64)
    crf_predictions: list[
        tuple[str, tuple[int, int, int, int], float]
    ] = []
    case_by_id = _case_map(cases)
    prediction_dir = output_dir / (
        "predictions_tta" if test_time_augmentation else "predictions"
    )
    if write_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    crf_prediction_dir: Path | None = None
    if dense_crf:
        from .refinement import dense_crf_refine

        if write_predictions:
            crf_prediction_dir = output_dir / "predictions_crf"
            crf_prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for images, masks, case_ids in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if test_time_augmentation:
                probability_views = [
                    torch.sigmoid(_final_logits(model(images)))
                ]
                for dimensions in ((3,), (2,), (2, 3)):
                    transformed = torch.flip(images, dims=dimensions)
                    transformed_probabilities = torch.sigmoid(
                        _final_logits(model(transformed))
                    )
                    probability_views.append(
                        torch.flip(transformed_probabilities, dims=dimensions)
                    )
                probabilities = torch.stack(probability_views).mean(dim=0)
            else:
                probabilities = torch.sigmoid(_final_logits(model(images)))
            probabilities = probabilities.cpu().numpy()[:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start

            for probability, target_tensor, case_id in zip(
                probabilities, masks.numpy()[:, 0], case_ids
            ):
                target = target_tensor.astype(np.uint8)
                prediction = (probability >= threshold).astype(np.uint8)
                metrics = segmentation_metrics(target, prediction)
                image_metrics.append(metrics)
                sum_counts += np.asarray(segmentation_counts(target, prediction))
                case = case_by_id[case_id]
                target_boxes = _scaled_annotation_boxes(case, size, target)
                crf_metrics: dict[str, float] = {}
                if dense_crf and not test_time_augmentation:
                    crf_started = time.perf_counter()
                    original = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
                    if original is None:
                        raise OSError(f"Could not read image for case {case_id}")
                    original = cv2.cvtColor(
                        cv2.resize(original, (size, size)), cv2.COLOR_BGR2RGB
                    )
                    crf_probability = dense_crf_refine(original, probability)
                    crf_seconds += time.perf_counter() - crf_started
                    crf_prediction = (crf_probability >= threshold).astype(np.uint8)
                    crf_metrics = segmentation_metrics(target, crf_prediction)
                    crf_image_metrics.append(crf_metrics)
                    crf_sum_counts += np.asarray(
                        segmentation_counts(target, crf_prediction)
                    )
                    crf_predictions.extend(
                        (case_id, box, score)
                        for box, score in predicted_boxes_with_scores(
                            crf_probability, threshold
                        )
                    )
                    if crf_prediction_dir is not None:
                        cv2.imwrite(
                            str(crf_prediction_dir / f"{case_id}.png"),
                            (crf_probability * 255).clip(0, 255).astype(np.uint8),
                        )
                target_area_fraction = (
                    sum(
                        (box[2] - box[0]) * (box[3] - box[1])
                        for box in target_boxes
                    )
                    / (size * size)
                )
                per_image_rows.append(
                    {
                        "case_id": case_id,
                        "fold": (
                            fold_number
                            if fold_number is not None
                            else int(output_dir.name.removeprefix("fold_"))
                        ),
                        **metrics,
                        "target_box_count": len(target_boxes),
                        "target_box_area_fraction": target_area_fraction,
                        "is_sessile_subset": case_id in sessile_case_ids,
                        **{
                            f"crf_{key}": value
                            for key, value in crf_metrics.items()
                        },
                    }
                )
                detection_targets.extend(
                    (case_id, box) for box in target_boxes
                )
                detection_predictions.extend(
                    (case_id, box, score)
                    for box, score in predicted_boxes_with_scores(
                        probability, threshold
                    )
                )
                if write_predictions:
                    cv2.imwrite(
                        str(prediction_dir / f"{case_id}.png"),
                        (probability * 255).clip(0, 255).astype(np.uint8),
                    )
                if write_predictions and write_overlays and not test_time_augmentation:
                    original = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
                    original = cv2.resize(original, (size, size))
                    target_overlay = original.copy()
                    target_region = target > 0
                    target_overlay[target_region] = (
                        0.55 * target_overlay[target_region]
                        + 0.45 * np.array([0, 255, 0])
                    ).astype(np.uint8)
                    predicted_overlay = original.copy()
                    binary = prediction.astype(bool)
                    predicted_overlay[binary] = (
                        0.55 * predicted_overlay[binary]
                        + 0.45 * np.array([0, 0, 255])
                    ).astype(np.uint8)
                    for x0, y0, x1, y1 in target_boxes:
                        cv2.rectangle(
                            target_overlay, (x0, y0), (x1, y1), (0, 255, 0), 2
                        )
                    for box, _ in predicted_boxes_with_scores(
                        probability, threshold
                    ):
                        cv2.rectangle(
                            predicted_overlay, box[:2], box[2:], (255, 0, 0), 2
                        )
                    panel = np.concatenate(
                        (original, target_overlay, predicted_overlay), axis=1
                    )
                    cv2.imwrite(
                        str(prediction_dir / f"{case_id}_overlay.jpg"), panel
                    )

    tp, fp, fn, tn = sum_counts.tolist()
    box_precision, box_recall = detection_precision_recall(
        detection_predictions, detection_targets
    )
    result = {
        "dice": float(np.mean([metric["dice"] for metric in image_metrics])),
        "miou": float(np.mean([metric["miou"] for metric in image_metrics])),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "ap50": average_precision_50(detection_predictions, detection_targets),
        "box_precision": box_precision,
        "box_recall": box_recall,
        "inference_fps": float(len(cases) / max(inference_seconds, 1e-8)),
        "inference_seconds": float(inference_seconds),
    }
    if dense_crf and not test_time_augmentation:
        crf_tp, crf_fp, crf_fn, crf_tn = crf_sum_counts.tolist()
        crf_box_precision, crf_box_recall = detection_precision_recall(
            crf_predictions, detection_targets
        )
        result.update(
            {
                "crf_dice": float(
                    np.mean([metric["dice"] for metric in crf_image_metrics])
                ),
                "crf_miou": float(
                    np.mean([metric["miou"] for metric in crf_image_metrics])
                ),
                "crf_sensitivity": float(crf_tp / (crf_tp + crf_fn + 1e-8)),
                "crf_specificity": float(crf_tn / (crf_tn + crf_fp + 1e-8)),
                "crf_ap50": average_precision_50(
                    crf_predictions, detection_targets
                ),
                "crf_box_precision": crf_box_precision,
                "crf_box_recall": crf_box_recall,
                "crf_inference_fps": float(
                    len(cases) / max(inference_seconds + crf_seconds, 1e-8)
                ),
                "crf_seconds": float(crf_seconds),
            }
        )
    return result, per_image_rows


def evaluate_external_sets(
    experiment_dir: Path,
    external_test_sets: dict[str, list[Case]],
    batch_size: int = 2,
    device_name: str = "cuda",
    num_workers: int = 0,
    dense_crf: bool = False,
) -> list[dict[str, object]]:
    if not external_test_sets:
        raise ValueError("At least one external test dataset is required")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
        for name in external_test_sets
    ):
        raise ValueError("External dataset names may contain only letters, digits, _ and -")
    all_case_ids: set[str] = set()
    image_hashes: dict[str, tuple[str, str]] = {}
    duplicate_pairs_by_dataset: dict[str, list[dict[str, str]]] = {
        dataset_name: [] for dataset_name in external_test_sets
    }
    for dataset_name, cases in external_test_sets.items():
        if not cases:
            raise ValueError(f"External test dataset {dataset_name} contains no cases")
        case_ids = {case.case_id for case in cases}
        if len(case_ids) != len(cases) or all_case_ids & case_ids:
            raise ValueError("External test case IDs must be unique across datasets")
        all_case_ids.update(case_ids)
        for case in cases:
            digest = _image_content_hash(case)
            if digest in image_hashes:
                previous_dataset, previous_case = image_hashes[digest]
                if previous_dataset == dataset_name:
                    duplicate_pairs_by_dataset[dataset_name].append(
                        {
                            "first_case_id": previous_case,
                            "duplicate_case_id": case.case_id,
                        }
                    )
                    continue
                raise ValueError(
                    "External test datasets share an identical image: "
                    f"{previous_case} and {case.case_id}"
                )
            image_hashes[digest] = (dataset_name, case.case_id)
    if dense_crf:
        from .refinement import dense_crf_refine

        dense_crf_refine(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2), 0.5, dtype=np.float32),
            iterations=1,
        )

    config_path = experiment_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing training configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unlabeled_config = config.get("unlabeled_dataset")
    unlabeled_manifest_hash: str | None = None
    if unlabeled_config is not None:
        manifest_path = experiment_dir / "unlabeled_data_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing unlabeled-data manifest required for leakage checks: "
                f"{manifest_path}"
            )
        unlabeled_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unlabeled_manifest_hash = str(
            unlabeled_config["eligible_manifest_sha256"]
        )
        training_image_hashes = {
            entry["decoded_image_sha256"]
            for entry in unlabeled_manifest["images"]
            if entry["status"] == "included"
        }
        overlaps = [
            {"dataset": dataset_name, "case_id": case_id}
            for digest, (dataset_name, case_id) in image_hashes.items()
            if digest in training_image_hashes
        ]
        if overlaps:
            raise ValueError(
                "External evaluation images overlap the unlabeled training corpus "
                f"by exact decoded-image hash: {overlaps[:5]}"
            )
    model_name = str(config["model"])
    preprocessing = str(config["preprocessing"])
    image_size = int(config["image_size"])
    threshold = float(config.get("threshold", 0.5))
    folds = int(config["folds"])
    pretrained_encoder = bool(config.get("pretrained_encoder", False))
    use_tta = bool(config.get("test_time_augmentation", False))
    run_config = {
        "experiment": str(experiment_dir.resolve()),
        "model": model_name,
        "preprocessing": preprocessing,
        "image_size": image_size,
        "threshold": threshold,
        "folds": folds,
        "batch_size": batch_size,
        "device": device_name,
        "num_workers": num_workers,
        "dense_crf": dense_crf,
        "external_test_case_ids": {
            name: sorted(case.case_id for case in cases)
            for name, cases in sorted(external_test_sets.items())
        },
        "within_dataset_exact_image_duplicate_pairs": duplicate_pairs_by_dataset,
    }
    if unlabeled_manifest_hash is not None:
        run_config["unlabeled_manifest_sha256"] = unlabeled_manifest_hash
    external_config_path = experiment_dir / "external_eval_config.json"
    if external_config_path.is_file():
        previous_config = json.loads(
            external_config_path.read_text(encoding="utf-8")
        )
        if previous_config != run_config:
            raise ValueError(
                f"Cannot resume external evaluation in {experiment_dir}; "
                "configuration or external test cases differ."
            )
    else:
        _atomic_json_write(external_config_path, run_config)

    metrics_path = experiment_dir / "external_metrics.csv"
    results: list[dict[str, object]] = []
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as file:
            results = list(csv.DictReader(file))
    completed = {
        (str(row["dataset"]), int(row["fold"]))
        for row in results
        if row.get("dataset") and row.get("fold")
    }
    device = torch.device(device_name)
    for fold_number in range(1, folds + 1):
        fold_dir = experiment_dir / f"fold_{fold_number}"
        checkpoint_path = fold_dir / "best_model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot evaluate external data before fold training completes: "
                f"{checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("model_name") != model_name:
            raise ValueError(f"Model name mismatch in {checkpoint_path}")
        model = build_model(model_name, pretrained_encoder=False, image_size=image_size)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        for dataset_name, cases in external_test_sets.items():
            if (dataset_name, fold_number) in completed:
                continue
            loader = _make_loader(
                cases,
                image_size,
                preprocessing,
                batch_size,
                shuffle=False,
                workers=num_workers,
                imagenet_normalize=pretrained_encoder,
            )
            evaluation_dir = (
                fold_dir / "external_evaluation" / dataset_name
            )
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            metrics, per_image_rows = _predict_outer_test(
                model,
                cases,
                loader,
                device,
                evaluation_dir,
                threshold,
                image_size,
                False,
                set(),
                dense_crf=dense_crf,
                fold_number=fold_number,
                write_predictions=False,
            )
            tta_metrics: dict[str, float] = {}
            tta_rows: list[dict[str, object]] = []
            if use_tta:
                tta_metrics, tta_rows = _predict_outer_test(
                    model,
                    cases,
                    loader,
                    device,
                    evaluation_dir,
                    threshold,
                    image_size,
                    False,
                    set(),
                    test_time_augmentation=True,
                    fold_number=fold_number,
                    write_predictions=False,
                )
            row: dict[str, object] = {
                "experiment": experiment_dir.name,
                "model": model_name,
                "data_condition": (
                    "joint_auxiliary"
                    if config.get("auxiliary_case_ids")
                    else "kvasir_only"
                ),
                "dataset": dataset_name,
                "fold": fold_number,
                "exact_image_duplicate_pairs": len(
                    duplicate_pairs_by_dataset[dataset_name]
                ),
                "deep_supervision": int(bool(config.get("deep_supervision", False))),
                "small_polyp_crop_probability": float(
                    config.get("small_polyp_crop_probability", 0.0)
                ),
                "auxiliary_training_cases": len(
                    config.get("auxiliary_case_ids", [])
                ),
                **metrics,
                **{f"tta_{key}": value for key, value in tta_metrics.items()},
            }
            results.append(row)
            results.sort(key=lambda result: (str(result["dataset"]), int(result["fold"])))
            _write_csv_atomic(metrics_path, results)
            for image_row in per_image_rows:
                image_row.update(dataset=dataset_name, model=model_name)
            _write_csv_atomic(
                evaluation_dir / "per_image_metrics.csv", per_image_rows
            )
            if tta_rows:
                for image_row in tta_rows:
                    image_row.update(dataset=dataset_name, model=model_name)
                _write_csv_atomic(
                    evaluation_dir / "per_image_metrics_tta.csv", tta_rows
                )
            completed.add((dataset_name, fold_number))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = (
        "dice",
        "miou",
        "sensitivity",
        "specificity",
        "ap50",
        "box_precision",
        "box_recall",
        "inference_fps",
        "tta_dice",
        "tta_miou",
        "tta_sensitivity",
        "tta_specificity",
        "tta_ap50",
        "tta_box_precision",
        "tta_box_recall",
        "tta_inference_fps",
        "crf_dice",
        "crf_miou",
        "crf_sensitivity",
        "crf_specificity",
        "crf_ap50",
        "crf_box_precision",
        "crf_box_recall",
        "crf_inference_fps",
    )
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_name in external_test_sets:
        dataset_rows = [
            row for row in results if row.get("dataset") == dataset_name
        ]
        if len(dataset_rows) != folds:
            continue
        summaries[dataset_name] = {}
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in dataset_rows
                if row.get(metric) not in (None, "")
            ]
            if values:
                summaries[dataset_name][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
    _atomic_json_write(
        experiment_dir / "external_metrics_summary.json", summaries
    )
    return results


def train_cross_validation(
    cases: list[Case],
    output_dir: Path,
    model_name: str,
    loss_name: str = "bce_dice",
    preprocessing: str = "clahe",
    n_splits: int = 5,
    seed: int = 42,
    image_size: int = 352,
    batch_size: int = 4,
    epochs: int = 30,
    patience: int = 6,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.1,
    threshold: float = 0.5,
    device_name: str = "cuda",
    pretrained_encoder: bool = False,
    num_workers: int = 0,
    overlays: bool = True,
    progress_log_path: Path | None = None,
    sessile_case_ids: set[str] | None = None,
    test_time_augmentation: bool = False,
    deep_supervision: bool = False,
    small_polyp_crop_probability: float = 0.0,
    auxiliary_cases: list[Case] | None = None,
    dense_crf: bool = False,
    consistency_weight: float = 0.0,
    ema_decay: float = 0.99,
    consistency_rampup_epochs: int = 5,
    write_predictions: bool = True,
    unlabeled_image_paths: list[Path] | None = None,
    unlabeled_image_dir: Path | None = None,
    unlabeled_exclusion_cases: list[Case] | None = None,
) -> list[dict[str, float | int | str]]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")
    if deep_supervision and model_name not in DEEP_SUPERVISION_MODELS:
        raise ValueError(
            f"{model_name} does not provide auxiliary outputs for deep supervision"
        )
    if not 0.0 <= small_polyp_crop_probability <= 1.0:
        raise ValueError("small_polyp_crop_probability must be between 0 and 1")
    if not np.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and non-negative")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if consistency_rampup_epochs < 1:
        raise ValueError("consistency_rampup_epochs must be positive")
    unlabeled_exclusion_cases = unlabeled_exclusion_cases or []
    eligible_unlabeled_paths: list[Path] = []
    unlabeled_manifest_entries: list[dict[str, str]] = []
    unlabeled_dataset_config: dict[str, object] | None = None
    if unlabeled_image_paths is not None:
        if consistency_weight <= 0.0:
            raise ValueError(
                "Unlabeled images require --consistency-weight greater than zero."
            )
        (
            eligible_unlabeled_paths,
            unlabeled_manifest_entries,
            unlabeled_dataset_config,
        ) = _filter_unlabeled_images(
            unlabeled_image_paths,
            unlabeled_image_dir,
            [*cases, *unlabeled_exclusion_cases],
            unlabeled_exclusion_cases,
        )
    auxiliary_cases = auxiliary_cases or []
    auxiliary_ids = [case.case_id for case in auxiliary_cases]
    if len(auxiliary_ids) != len(set(auxiliary_ids)):
        raise ValueError("Auxiliary dataset case IDs must be unique")
    if set(auxiliary_ids) & {case.case_id for case in cases}:
        raise ValueError("Auxiliary and Kvasir case IDs must be disjoint")
    if dense_crf:
        from .refinement import dense_crf_refine

        dense_crf_refine(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2), 0.5, dtype=np.float32),
            iterations=1,
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch "
            "build and verify the NVIDIA driver, or use --device cpu explicitly."
        )
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = create_outer_folds(cases, n_splits, seed)
    write_fold_manifest(output_dir / "folds.json", cases, folds, seed)
    progress_log = progress_log_path or output_dir / "progress.jsonl"
    sessile_case_ids = sessile_case_ids or set()
    auxiliary_hashes = {
        case.case_id: _image_content_hash(case) for case in auxiliary_cases
    }
    run_config = {
        "model": model_name,
        "loss": loss_name,
        "preprocessing": preprocessing,
        "folds": n_splits,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "patience": patience,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "threshold": threshold,
        "device": device_name,
        "pretrained_encoder": pretrained_encoder,
        "deep_supervision": deep_supervision,
        "small_polyp_crop_probability": small_polyp_crop_probability,
        "auxiliary_case_ids": auxiliary_ids,
        "dense_crf": dense_crf,
        "case_ids": [case.case_id for case in cases],
        "sessile_case_ids": sorted(sessile_case_ids),
        "max_epochs": epochs,
        "resume_checkpoint": "latest_training.pt",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "opencv_version": cv2.__version__,
    }
    if unlabeled_dataset_config is not None:
        run_config["unlabeled_dataset"] = unlabeled_dataset_config
        run_config["unlabeled_consistency"] = {
            "loss": "entropy-weighted pseudo-label BCE plus probability MSE",
            "confidence_threshold": 0.7,
            "ema_decay": ema_decay,
            "consistency_rampup_epochs": consistency_rampup_epochs,
        }
    if test_time_augmentation:
        run_config["test_time_augmentation"] = True
    if loss_name in {"bce_dice_boundary", "bce_dice_edge_boundary"}:
        run_config["boundary_weight"] = 0.1
    if loss_name == "bce_dice_edge_boundary":
        run_config["edge_weight"] = 0.1
    if consistency_weight > 0.0:
        run_config.update(
            consistency_weight=consistency_weight,
            ema_decay=ema_decay,
            consistency_rampup_epochs=consistency_rampup_epochs,
            consistency_confidence_threshold=0.7,
            consistency_view_strengths=[0.025, 0.1],
        )
    if not write_predictions:
        run_config["write_predictions"] = False
    try:
        import segmentation_models_pytorch as smp

        run_config["segmentation_models_pytorch_version"] = smp.__version__
    except ImportError as error:
        raise RuntimeError("segmentation-models-pytorch is required for deep training") from error
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        immutable_keys = set(run_config) - {"max_epochs", "resume_checkpoint"}
        differences = [
            key
            for key in sorted(immutable_keys)
            if existing_config.get(key) != run_config.get(key)
        ]
        if differences:
            raise ValueError(
                f"Cannot resume {output_dir}; run configuration differs in "
                f"{differences}. Choose a new output directory."
            )
        if epochs < int(existing_config.get("max_epochs", epochs)):
            raise ValueError(
                "Resuming with fewer epochs is not supported; keep or increase --epochs."
            )
    _atomic_json_write(config_path, run_config)
    if unlabeled_dataset_config is not None:
        _atomic_json_write(
            output_dir / "unlabeled_data_manifest.json",
            {
                **unlabeled_dataset_config,
                "images": unlabeled_manifest_entries,
            },
        )
    by_id = _case_map(cases)
    metrics_path = output_dir / "fold_metrics.csv"
    results: list[dict[str, float | int | str]] = []
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as file:
            results = list(csv.DictReader(file))  # type: ignore[assignment]
    stale_completed_folds = {
        int(row["fold"])
        for row in results
        if int(row.get("max_epochs", epochs)) < epochs
    }
    results = [
        row for row in results if int(row["fold"]) not in stale_completed_folds
    ]
    completed_folds = {int(row["fold"]) for row in results}
    per_image_path = output_dir / "per_image_metrics.csv"
    per_image_rows: list[dict[str, object]] = []
    if per_image_path.is_file():
        with per_image_path.open(newline="", encoding="utf-8") as file:
            per_image_rows = list(csv.DictReader(file))
    per_image_rows = [
        row
        for row in per_image_rows
        if int(row.get("fold", -1)) not in stale_completed_folds
    ]
    _log_progress(
        progress_log,
        "experiment_started",
        output_dir=str(output_dir),
        model=model_name,
        loss=loss_name,
        preprocessing=preprocessing,
        completed_folds=len(completed_folds),
        total_folds=n_splits,
        max_epochs=epochs,
    )

    for fold_index, fold in enumerate(folds):
        fold_number = fold_index + 1
        if fold_number in completed_folds:
            continue
        fold_dir = output_dir / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids, validation_ids = inner_train_validation_split(
            fold["train_ids"], validation_fraction, seed + fold_index
        )
        split_manifest = {
            "fold": fold_index + 1,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_ids": fold["test_ids"],
        }
        training_cases = [by_id[case_id] for case_id in train_ids]
        validation_cases = [by_id[case_id] for case_id in validation_ids]
        test_cases = [by_id[case_id] for case_id in fold["test_ids"]]
        test_image_hashes = {_image_content_hash(case) for case in test_cases}
        fold_auxiliary_cases = [
            case
            for case in auxiliary_cases
            if auxiliary_hashes[case.case_id] not in test_image_hashes
        ]
        split_manifest["auxiliary_train_ids"] = [
            case.case_id for case in fold_auxiliary_cases
        ]
        split_manifest["auxiliary_excluded_test_duplicates"] = [
            case.case_id
            for case in auxiliary_cases
            if auxiliary_hashes[case.case_id] in test_image_hashes
        ]
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )
        effective_training_cases = training_cases + fold_auxiliary_cases

        # Assert image-level isolation before creating loaders.
        if (
            set(train_ids) & set(validation_ids)
            or set(train_ids) & set(fold["test_ids"])
            or set(validation_ids) & set(fold["test_ids"])
        ):
            raise RuntimeError(f"Image leakage detected in fold {fold_index + 1}")

        _seed_everything(seed + fold_index)
        train_loader = _make_loader(
            effective_training_cases,
            image_size,
            preprocessing,
            batch_size,
            shuffle=True,
            augment=True,
            workers=num_workers,
            imagenet_normalize=pretrained_encoder,
            small_polyp_crop_probability=small_polyp_crop_probability,
        )
        unlabeled_loader = (
            _make_unlabeled_loader(
                eligible_unlabeled_paths,
                image_size,
                preprocessing,
                batch_size,
                num_workers,
                pretrained_encoder,
            )
            if eligible_unlabeled_paths
            else None
        )
        validation_loader = _make_loader(
            validation_cases,
            image_size,
            preprocessing,
            batch_size,
            shuffle=False,
            workers=num_workers,
            imagenet_normalize=pretrained_encoder,
        )

        model = build_model(model_name, pretrained_encoder, image_size=image_size)
        parameters_m, gflops_per_image = _model_complexity(model, image_size)
        model = model.to(device)
        ema_model: torch.nn.Module | None = None
        if consistency_weight > 0.0:
            ema_model = copy.deepcopy(model).eval()
            ema_model.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_dice = -1.0
        best_model_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
        checkpoint_path = fold_dir / "best_model.pt"
        latest_checkpoint_path = fold_dir / "latest_training.pt"
        start_epoch = 0
        if latest_checkpoint_path.is_file():
            try:
                checkpoint = torch.load(
                    latest_checkpoint_path, map_location="cpu", weights_only=False
                )
            except (
                RuntimeError,
                EOFError,
                pickle.UnpicklingError,
                zipfile.BadZipFile,
            ) as error:
                if not checkpoint_path.is_file():
                    raise RuntimeError(
                        f"Latest training checkpoint is unreadable and no valid "
                        f"best-model checkpoint exists for {model_name} fold "
                        f"{fold_number} at {fold_dir}"
                    ) from error
                fallback_checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                if (
                    fallback_checkpoint.get("model_name") != model_name
                    or fallback_checkpoint.get("loss_name") != loss_name
                ):
                    raise ValueError(
                        f"Best-model checkpoint configuration mismatch at "
                        f"{checkpoint_path}"
                    ) from error
                history_path = fold_dir / "training_history.json"
                if not history_path.is_file():
                    raise RuntimeError(
                        f"Cannot recover training progress without {history_path}"
                    ) from error
                recovered_history = json.loads(
                    history_path.read_text(encoding="utf-8")
                )
                if not recovered_history:
                    raise RuntimeError(
                        f"Cannot recover training progress from empty {history_path}"
                    ) from error
                history = recovered_history
                start_epoch = int(history[-1]["epoch"])
                if start_epoch < 1 or start_epoch > epochs:
                    raise RuntimeError(
                        f"Invalid recovered epoch {start_epoch} in {history_path}"
                    ) from error
                best_dice = float(fallback_checkpoint["best_validation_dice"])
                best_model_state = fallback_checkpoint["model_state"]
                model.load_state_dict(best_model_state)
                if ema_model is not None:
                    ema_model.load_state_dict(best_model_state)
                best_epoch = next(
                    (
                        int(record["epoch"])
                        for record in history
                        if float(record["validation_dice"]) >= best_dice
                    ),
                    start_epoch,
                )
                epochs_without_improvement = start_epoch - best_epoch
                print(
                    f"Recovering {model_name} fold {fold_number} from its "
                    f"best-validation weights at epoch {start_epoch}; optimizer "
                    "and random-number state will be reinitialized."
                )
                _log_progress(
                    progress_log,
                    "checkpoint_recovered_from_best_model",
                    model=model_name,
                    fold=fold_number,
                    completed_epoch=start_epoch,
                    best_validation_dice=best_dice,
                    optimizer_state_restored=False,
                    checkpoint_error=str(error),
                )
                _log_progress(
                    progress_log,
                    "fold_resumed",
                    model=model_name,
                    fold=fold_number,
                    completed_epoch=start_epoch,
                    max_epochs=epochs,
                    recovered_from_best_model=True,
                )
            else:
                if checkpoint["model_name"] != model_name or checkpoint["loss_name"] != loss_name:
                    raise ValueError(f"Checkpoint configuration mismatch at {latest_checkpoint_path}")
                model.load_state_dict(checkpoint["model_state"])
                if ema_model is not None:
                    teacher_state = checkpoint.get("ema_model_state")
                    if teacher_state is None:
                        raise ValueError(
                            f"EMA teacher state is missing from {latest_checkpoint_path}"
                        )
                    ema_model.load_state_dict(teacher_state)
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                scaler.load_state_dict(checkpoint["scaler_state"])
                start_epoch = int(checkpoint["completed_epoch"])
                best_dice = float(checkpoint["best_validation_dice"])
                epochs_without_improvement = int(
                    checkpoint["epochs_without_improvement"]
                )
                previous_max_epochs = int(checkpoint.get("max_epochs", epochs))
                if epochs > previous_max_epochs:
                    epochs_without_improvement = 0
                    _log_progress(
                        progress_log,
                        "fold_extended",
                        model=model_name,
                        fold=fold_number,
                        previous_max_epochs=previous_max_epochs,
                        new_max_epochs=epochs,
                    )
                history = checkpoint["history"]
                best_model_state = checkpoint["best_model_state"]
                _restore_rng_state(checkpoint["rng_state"])
                print(
                    f"Resuming {model_name} fold {fold_number} at "
                    f"epoch {start_epoch + 1}/{epochs}"
                )
                _log_progress(
                    progress_log,
                    "fold_resumed",
                    model=model_name,
                    fold=fold_number,
                    completed_epoch=start_epoch,
                    max_epochs=epochs,
                )
        else:
            _log_progress(
                progress_log,
                "fold_started",
                model=model_name,
                fold=fold_number,
                total_folds=n_splits,
                max_epochs=epochs,
            )

        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_supervised_loss = 0.0
            total_consistency_loss = 0.0
            unlabeled_iterator = (
                iter(unlabeled_loader) if unlabeled_loader is not None else None
            )
            for images, masks, _ in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if ema_model is None:
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        outputs = model(images)
                        supervised_loss = segmentation_objective(
                            loss_name, outputs, masks, deep_supervision
                        )
                    consistency_loss = supervised_loss.new_zeros(())
                    loss = supervised_loss
                else:
                    teacher_images = _photometric_view(images, 0.025)
                    student_images = _photometric_view(images, 0.1)
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type, enabled=amp_enabled
                    ):
                        teacher_logits = _final_logits(ema_model(teacher_images))
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        student_outputs = model(student_images)
                        student_logits = _final_logits(student_outputs)
                        supervised_loss = segmentation_objective(
                            loss_name, student_outputs, masks, deep_supervision
                        )
                        consistency_loss = _confidence_consistency_loss(
                            student_logits, teacher_logits
                        )
                        if unlabeled_iterator is not None and unlabeled_loader is not None:
                            try:
                                unlabeled_images = next(unlabeled_iterator)
                            except StopIteration:
                                unlabeled_iterator = iter(unlabeled_loader)
                                unlabeled_images = next(unlabeled_iterator)
                            unlabeled_images = unlabeled_images.to(
                                device, non_blocking=True
                            )
                            teacher_unlabeled_images = _photometric_view(
                                unlabeled_images, 0.025
                            )
                            student_unlabeled_images = _photometric_view(
                                unlabeled_images, 0.1
                            )
                            with torch.no_grad(), torch.autocast(
                                device_type=device.type, enabled=amp_enabled
                            ):
                                teacher_unlabeled_logits = _final_logits(
                                    ema_model(teacher_unlabeled_images)
                                )
                            with torch.autocast(
                                device_type=device.type, enabled=amp_enabled
                            ):
                                student_unlabeled_logits = _final_logits(
                                    model(student_unlabeled_images)
                                )
                                unlabeled_loss = _uncertainty_pseudo_label_loss(
                                    student_unlabeled_logits,
                                    teacher_unlabeled_logits,
                                )
                            consistency_loss = torch.stack(
                                (consistency_loss, unlabeled_loss)
                            ).mean()
                        consistency_scale = min(
                            1.0, (epoch + 1) / consistency_rampup_epochs
                        )
                        loss = (
                            supervised_loss
                            + consistency_weight
                            * consistency_scale
                            * consistency_loss
                        )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    _update_ema_model(ema_model, model, ema_decay)
                total_loss += float(loss.detach()) * len(images)
                total_supervised_loss += float(supervised_loss.detach()) * len(images)
                total_consistency_loss += float(consistency_loss.detach()) * len(images)

            validation_dice = _validate_epoch(
                ema_model if ema_model is not None else model,
                validation_loader,
                device,
                threshold,
            )
            scheduler.step(validation_dice)
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, len(effective_training_cases)),
                "supervised_loss": total_supervised_loss
                / max(1, len(effective_training_cases)),
                "consistency_loss": total_consistency_loss
                / max(1, len(effective_training_cases)),
                "validation_dice": validation_dice,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            print(
                f"fold={fold_number}/{n_splits} epoch={epoch + 1}/{epochs} "
                f"train_loss={epoch_record['train_loss']:.4f} "
                f"val_dice={validation_dice:.4f} "
                f"seconds={epoch_record['epoch_seconds']:.1f}"
            )
            if validation_dice > best_dice:
                best_dice = validation_dice
                epochs_without_improvement = 0
                best_model_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in (
                        ema_model if ema_model is not None else model
                    ).state_dict().items()
                }
                _atomic_torch_save(
                    checkpoint_path,
                    {
                        "model_state": best_model_state,
                        "selected_weights": (
                            "ema_teacher" if ema_model is not None else "student"
                        ),
                        "model_name": model_name,
                        "loss_name": loss_name,
                        "consistency_weight": consistency_weight,
                        "ema_decay": ema_decay if ema_model is not None else None,
                        "unlabeled_image_count": len(eligible_unlabeled_paths),
                        "deep_supervision": deep_supervision,
                        "small_polyp_crop_probability": small_polyp_crop_probability,
                        "preprocessing": preprocessing,
                        "image_size": image_size,
                        "threshold": threshold,
                        "best_validation_dice": best_dice,
                        "seed": seed + fold_index,
                    },
                )
            else:
                epochs_without_improvement += 1
            _atomic_json_write(fold_dir / "training_history.json", history)
            latest_checkpoint: dict[str, object] = {
                "model_name": model_name,
                "loss_name": loss_name,
                "completed_epoch": epoch + 1,
                "max_epochs": epochs,
                "model_state": model.state_dict(),
                "best_model_state": best_model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_validation_dice": best_dice,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "rng_state": _capture_rng_state(),
            }
            if ema_model is not None:
                latest_checkpoint["ema_model_state"] = ema_model.state_dict()
            _atomic_torch_save(latest_checkpoint_path, latest_checkpoint)
            _log_progress(
                progress_log,
                "epoch_completed",
                model=model_name,
                fold=fold_number,
                epoch=epoch + 1,
                max_epochs=epochs,
                train_loss=epoch_record["train_loss"],
                supervised_loss=epoch_record["supervised_loss"],
                consistency_loss=epoch_record["consistency_loss"],
                validation_dice=validation_dice,
                epoch_seconds=epoch_record["epoch_seconds"],
                best_validation_dice=best_dice,
                epochs_without_improvement=epochs_without_improvement,
            )
            if epochs_without_improvement >= patience:
                break

        if best_model_state is None:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            best_model_state = checkpoint["model_state"]
        evaluation_model = ema_model if ema_model is not None else model
        evaluation_model.load_state_dict(best_model_state)
        test_loader = _make_loader(
            test_cases,
            image_size,
            preprocessing,
            batch_size,
            shuffle=False,
            workers=num_workers,
            imagenet_normalize=pretrained_encoder,
        )
        test_metrics, fold_per_image_rows = _predict_outer_test(
            evaluation_model,
            test_cases,
            test_loader,
            device,
            fold_dir,
            threshold,
            image_size,
            overlays,
            sessile_case_ids,
            dense_crf=dense_crf,
            write_predictions=write_predictions,
        )
        tta_per_image_rows: list[dict[str, object]] = []
        if test_time_augmentation:
            tta_metrics, tta_per_image_rows = _predict_outer_test(
                evaluation_model,
                test_cases,
                test_loader,
                device,
                fold_dir,
                threshold,
                image_size,
                False,
                sessile_case_ids,
                test_time_augmentation=True,
                write_predictions=write_predictions,
            )
        else:
            tta_metrics = {}
        _write_csv_atomic(
            fold_dir / "per_image_metrics.csv", fold_per_image_rows
        )
        per_image_rows = [
            row
            for row in per_image_rows
            if int(row.get("fold", -1)) != fold_number
        ] + fold_per_image_rows
        if test_time_augmentation:
            tta_per_image_path = output_dir / "per_image_metrics_tta.csv"
            all_tta_rows: list[dict[str, object]] = []
            if tta_per_image_path.is_file():
                with tta_per_image_path.open(newline="", encoding="utf-8") as file:
                    all_tta_rows = list(csv.DictReader(file))
            all_tta_rows = [
                row
                for row in all_tta_rows
                if int(row.get("fold", -1)) != fold_number
            ] + tta_per_image_rows
            _write_csv_atomic(tta_per_image_path, all_tta_rows)
        result: dict[str, float | int | str] = {
            "model": model_name,
            "loss": loss_name,
            "preprocessing": preprocessing,
            "fold": fold_index + 1,
            "best_validation_dice": best_dice,
            "best_epoch": int(np.argmax([item["validation_dice"] for item in history]) + 1),
            "max_epochs": epochs,
            "parameters_m": parameters_m,
            "gflops_per_image": gflops_per_image,
            "profiled_image_size": image_size,
            "deep_supervision": int(deep_supervision),
            "small_polyp_crop_probability": small_polyp_crop_probability,
            "auxiliary_training_cases": len(fold_auxiliary_cases),
            **test_metrics,
        }
        result.update(
            {f"tta_{metric}": value for metric, value in tta_metrics.items()}
        )
        results.append(result)
        results.sort(key=lambda row: int(row["fold"]))
        _write_csv_atomic(metrics_path, results)
        _write_csv_atomic(per_image_path, per_image_rows)
        completed_folds.add(fold_number)
        _log_progress(
            progress_log,
            "fold_completed",
            model=model_name,
            fold=fold_number,
            completed_folds=len(completed_folds),
            total_folds=n_splits,
            metrics=test_metrics,
        )
        del model, optimizer, scaler, ema_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = (
        "dice",
        "miou",
        "sensitivity",
        "specificity",
        "ap50",
        "box_precision",
        "box_recall",
        "inference_fps",
        "parameters_m",
        "gflops_per_image",
        "profiled_image_size",
    )
    if test_time_augmentation:
        metric_names += tuple(
            f"tta_{metric}"
            for metric in (
                "dice",
                "miou",
                "sensitivity",
                "specificity",
                "ap50",
                "box_precision",
                "box_recall",
                "inference_fps",
            )
        )
    if dense_crf:
        metric_names += tuple(
            f"crf_{metric}"
            for metric in (
                "dice",
                "miou",
                "sensitivity",
                "specificity",
                "ap50",
                "box_precision",
                "box_recall",
                "inference_fps",
                "seconds",
            )
        )
    if len(completed_folds) == n_splits:
        _atomic_json_write(
            output_dir / "summary.json",
            {
                "fold_results": results,
                "mean": {
                    metric: float(np.mean([float(row[metric]) for row in results]))
                    for metric in metric_names
                },
                "std": {
                    metric: float(
                        np.std([float(row[metric]) for row in results], ddof=1)
                    )
                    for metric in metric_names
                },
            },
        )
        _log_progress(
            progress_log,
            "experiment_completed",
            output_dir=str(output_dir),
            model=model_name,
            completed_folds=n_splits,
            total_folds=n_splits,
        )
    return results
