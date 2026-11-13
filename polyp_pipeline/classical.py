from __future__ import annotations

import csv
import json
import os
import time
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import (
    Case,
    _preprocess_image,
    create_outer_folds,
    write_fold_manifest,
)
from .features import extract_patch_features, label_patches
from .metrics import (
    average_precision_50,
    boxes_from_mask,
    detection_precision_recall,
    predicted_boxes_with_scores,
    scale_boxes,
    segmentation_counts,
    segmentation_metrics,
)


def _read_case(case: Case, size: int, preprocessing: str, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(case.mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        raise OSError(f"Could not read image/mask for {case.case_id}")
    if bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask dimensions do not match for {case.case_id}")
    if (
        case.image_width is not None
        and case.image_height is not None
        and (bgr.shape[1] != case.image_width or bgr.shape[0] != case.image_height)
    ):
        raise ValueError(f"Image dimensions differ from bbox metadata for {case.case_id}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = _preprocess_image(
        image, preprocessing, remove_specular=preprocessing == "clahe"
    )
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    binary_mask = cv2.resize(
        (mask > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
    )
    descriptors = extract_patch_features(image, grid_size)
    patch_labels = label_patches(binary_mask, grid_size)
    return image, binary_mask, descriptors, patch_labels


def run_classical_cv(
    cases: list[Case],
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    size: int = 352,
    grid_size: int = 4,
    preprocessing: str = "clahe",
    sessile_case_ids: set[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Group all patches by source image before fitting/evaluating classifiers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = create_outer_folds(cases, n_splits, seed)
    write_fold_manifest(output_dir / "folds.json", cases, folds, seed)
    config = {
        "case_ids": [case.case_id for case in cases],
        "folds": n_splits,
        "seed": seed,
        "size": size,
        "grid_size": grid_size,
        "preprocessing": preprocessing,
        "sessile_case_ids": sorted(sessile_case_ids or set()),
    }
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Cannot resume {output_dir}; classical run configuration changed."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    progress_path = output_dir / "progress.jsonl"

    def log_progress(event: str, **details: object) -> None:
        with progress_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "event": event,
                        **details,
                    }
                )
                + "\n"
            )
            file.flush()
            os.fsync(file.fileno())

    cache_path = output_dir / "feature_cache.npz"
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            cached_ids = cache["case_ids"].tolist()
            if cached_ids != config["case_ids"]:
                raise ValueError(f"Feature cache case order differs at {cache_path}")
            features = cache["features"]
            patch_labels = cache["patch_labels"]
            masks = cache["masks"]
            feature_seconds = float(cache["feature_seconds"][0])
        log_progress("features_loaded", cases=len(cases), seconds=feature_seconds)
    else:
        feature_started = time.perf_counter()
        examples = [_read_case(case, size, preprocessing, grid_size) for case in cases]
        feature_seconds = time.perf_counter() - feature_started
        features = np.stack([example[2] for example in examples])
        patch_labels = np.stack([example[3] for example in examples])
        masks = np.stack([example[1] for example in examples])
        temporary_cache = cache_path.with_suffix(".npz.tmp")
        with temporary_cache.open("wb") as file:
            np.savez_compressed(
                file,
                case_ids=np.asarray(config["case_ids"]),
                features=features,
                patch_labels=patch_labels,
                masks=masks,
                feature_seconds=np.asarray([feature_seconds]),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_cache, cache_path)
        log_progress("features_completed", cases=len(cases), seconds=feature_seconds)

    by_id = {case.case_id: index for index, case in enumerate(cases)}
    sessile_case_ids = sessile_case_ids or set()
    rows: list[dict[str, float | int | str]] = []
    per_image_path = output_dir / "per_image_metrics.csv"
    per_image_rows: list[dict[str, object]] = []
    if per_image_path.is_file():
        with per_image_path.open(newline="", encoding="utf-8") as file:
            per_image_rows = list(csv.DictReader(file))
    metrics_path = output_dir / "classical_metrics.json"
    if metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    completed = {(str(row["model"]), int(row["fold"])) for row in rows}
    log_progress(
        "experiment_started",
        completed_models=len(completed),
        total_model_folds=2 * n_splits,
    )

    for classifier_name in ("random_forest", "svm"):
        for fold_index, fold in enumerate(folds):
            completed_key = (classifier_name, fold_index + 1)
            if completed_key in completed:
                continue
            log_progress(
                "fold_started",
                model=classifier_name,
                fold=fold_index + 1,
                total_folds=n_splits,
            )
            train_indices = np.asarray([by_id[case_id] for case_id in fold["train_ids"]])
            test_indices = np.asarray([by_id[case_id] for case_id in fold["test_ids"]])
            x_train = features[train_indices].reshape(-1, features.shape[-1])
            y_train = patch_labels[train_indices].reshape(-1)
            x_test = features[test_indices].reshape(-1, features.shape[-1])
            y_test = patch_labels[test_indices].reshape(-1)
            if np.unique(y_train).size < 2:
                raise ValueError(
                    "Training patches contain only one class. Lower the patch "
                    "positive_fraction threshold or change the grid size."
                )

            if classifier_name == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed + fold_index,
                )
            else:
                model = make_pipeline(
                    StandardScaler(),
                    SVC(C=2.0, kernel="rbf", class_weight="balanced"),
                )
            model.fit(x_train, y_train)
            if classifier_name == "random_forest":
                patch_scores = model.predict_proba(x_test)[:, 1]
            else:
                decision_scores = model.decision_function(x_test)
                patch_scores = 1.0 / (
                    1.0 + np.exp(-np.clip(decision_scores, -50.0, 50.0))
                )
            patch_probabilities = patch_scores.reshape(
                len(test_indices), grid_size * grid_size
            )

            counts = np.zeros(4, dtype=np.int64)
            ap_predictions: list[tuple[str, tuple[int, int, int, int], float]] = []
            ap_targets: list[tuple[str, tuple[int, int, int, int]]] = []
            fold_image_rows: list[dict[str, object]] = []
            for local_index, case_index in enumerate(test_indices):
                case_id = cases[case_index].case_id
                target_mask = masks[case_index]
                probability_map = cv2.resize(
                    patch_probabilities[local_index].reshape(grid_size, grid_size),
                    (size, size),
                    interpolation=cv2.INTER_LINEAR,
                )
                prediction = (probability_map >= 0.5).astype(np.uint8)
                case = cases[case_index]
                target_boxes = (
                    scale_boxes(
                        case.bbox_annotations,
                        case.image_width,
                        case.image_height,
                        size,
                    )
                    if case.bbox_annotations
                    and case.image_width
                    and case.image_height
                    else boxes_from_mask(target_mask, min_area=4)
                )
                ap_targets.extend((case_id, box) for box in target_boxes)
                ap_predictions.extend(
                    (case_id, box, score)
                    for box, score in predicted_boxes_with_scores(probability_map)
                )
                tp, fp, fn, tn = segmentation_counts(target_mask, prediction)
                counts += np.asarray([tp, fp, fn, tn], dtype=np.int64)
                case_area = sum(
                    (box[2] - box[0]) * (box[3] - box[1])
                    for box in target_boxes
                ) / (size * size)
                fold_image_rows.append(
                    {
                        "case_id": case_id,
                        "model": classifier_name,
                        "loss": "not_applicable",
                        "preprocessing": preprocessing,
                        "fold": fold_index + 1,
                        **segmentation_metrics(target_mask, prediction),
                        "target_box_count": len(target_boxes),
                        "target_box_area_fraction": case_area,
                        "is_sessile_subset": case_id in sessile_case_ids,
                    }
                )

            tp, fp, fn, tn = counts.tolist()
            box_precision, box_recall = detection_precision_recall(
                ap_predictions, ap_targets
            )
            fold_result: dict[str, float | int | str] = {
                "model": classifier_name,
                "fold": fold_index + 1,
                "preprocessing": preprocessing,
                "patch_accuracy": float(
                    accuracy_score(y_test, (patch_probabilities.ravel() >= 0.5).astype(int))
                ),
                "dice": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
                "miou": float(tp / (tp + fp + fn + 1e-8)),
                "sensitivity": float(tp / (tp + fn + 1e-8)),
                "specificity": float(tn / (tn + fp + 1e-8)),
                "ap50": average_precision_50(ap_predictions, ap_targets),
                "box_precision": box_precision,
                "box_recall": box_recall,
                "feature_extraction_seconds": float(feature_seconds),
            }
            rows.append(fold_result)
            rows.sort(key=lambda row: (str(row["model"]), int(row["fold"])))
            temporary_metrics = metrics_path.with_suffix(".json.tmp")
            temporary_metrics.write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            os.replace(temporary_metrics, metrics_path)
            per_image_rows = [
                row
                for row in per_image_rows
                if not (
                    row.get("model") == classifier_name
                    and int(row.get("fold", -1)) == fold_index + 1
                )
            ] + fold_image_rows
            temporary_per_image = per_image_path.with_suffix(".csv.tmp")
            with temporary_per_image.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0]))
                writer.writeheader()
                writer.writerows(per_image_rows)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_per_image, per_image_path)
            completed.add(completed_key)
            log_progress(
                "fold_completed",
                model=classifier_name,
                fold=fold_index + 1,
                completed_model_folds=len(completed),
                total_model_folds=2 * n_splits,
                dice=fold_result["dice"],
            )

    log_progress(
        "experiment_completed",
        completed_model_folds=len(completed),
        total_model_folds=2 * n_splits,
    )
    return rows
