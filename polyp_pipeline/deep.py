from __future__ import annotations

import csv
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import random
import time
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


def _image_content_hash(case: Case) -> str:
    image = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image for case {case.case_id}")
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


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
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if dense_crf:
        from .refinement import dense_crf_refine

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
                        "fold": int(output_dir.name.removeprefix("fold_")),
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
                cv2.imwrite(
                    str(prediction_dir / f"{case_id}.png"),
                    (probability * 255).clip(0, 255).astype(np.uint8),
                )
                if write_overlays and not test_time_augmentation:
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
) -> list[dict[str, float | int | str]]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")
    if deep_supervision and model_name not in DEEP_SUPERVISION_MODELS:
        raise ValueError(
            f"{model_name} does not provide auxiliary outputs for deep supervision"
        )
    if not 0.0 <= small_polyp_crop_probability <= 1.0:
        raise ValueError("small_polyp_crop_probability must be between 0 and 1")
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
    if test_time_augmentation:
        run_config["test_time_augmentation"] = True
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
            checkpoint = torch.load(
                latest_checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint["model_name"] != model_name or checkpoint["loss_name"] != loss_name:
                raise ValueError(f"Checkpoint configuration mismatch at {latest_checkpoint_path}")
            model.load_state_dict(checkpoint["model_state"])
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
            for images, masks, _ in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images)
                    loss = segmentation_objective(
                        loss_name, outputs, masks, deep_supervision
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(images)

            validation_dice = _validate_epoch(
                model, validation_loader, device, threshold
            )
            scheduler.step(validation_dice)
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, len(effective_training_cases)),
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
                    for name, tensor in model.state_dict().items()
                }
                _atomic_torch_save(
                    checkpoint_path,
                    {
                        "model_state": best_model_state,
                        "model_name": model_name,
                        "loss_name": loss_name,
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
            _atomic_torch_save(
                latest_checkpoint_path,
                {
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
                },
            )
            _log_progress(
                progress_log,
                "epoch_completed",
                model=model_name,
                fold=fold_number,
                epoch=epoch + 1,
                max_epochs=epochs,
                train_loss=epoch_record["train_loss"],
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
        model.load_state_dict(best_model_state)
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
            model,
            test_cases,
            test_loader,
            device,
            fold_dir,
            threshold,
            image_size,
            overlays,
            sessile_case_ids,
            dense_crf=dense_crf,
        )
        tta_per_image_rows: list[dict[str, object]] = []
        if test_time_augmentation:
            tta_metrics, tta_per_image_rows = _predict_outer_test(
                model,
                test_cases,
                test_loader,
                device,
                fold_dir,
                threshold,
                image_size,
                False,
                sessile_case_ids,
                test_time_augmentation=True,
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
        del model, optimizer, scaler
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
