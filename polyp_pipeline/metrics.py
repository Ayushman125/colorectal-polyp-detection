from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def boxes_from_mask(mask: np.ndarray, min_area: int = 4) -> list[tuple[int, int, int, int]]:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        boxes.append((x, y, x + width, y + height))
    return boxes


def scale_boxes(
    boxes: Iterable[tuple[int, int, int, int]],
    original_width: int,
    original_height: int,
    size: int,
) -> list[tuple[int, int, int, int]]:
    if original_width <= 0 or original_height <= 0 or size <= 0:
        raise ValueError("Image dimensions and output size must be positive")
    scaled = []
    for xmin, ymin, xmax, ymax in boxes:
        x0 = max(0, min(size, round(xmin * size / original_width)))
        y0 = max(0, min(size, round(ymin * size / original_height)))
        x1 = max(0, min(size, round(xmax * size / original_width)))
        y1 = max(0, min(size, round(ymax * size / original_height)))
        if x1 > x0 and y1 > y0:
            scaled.append((x0, y0, x1, y1))
    return scaled


def box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def segmentation_counts(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    target_bool, prediction_bool = target > 0, prediction > 0
    tp = int(np.logical_and(target_bool, prediction_bool).sum())
    fp = int(np.logical_and(~target_bool, prediction_bool).sum())
    fn = int(np.logical_and(target_bool, ~prediction_bool).sum())
    tn = int(np.logical_and(~target_bool, ~prediction_bool).sum())
    return tp, fp, fn, tn


def segmentation_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = segmentation_counts(target, prediction)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        "dice": float(dice),
        "miou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def average_precision_50(
    predictions: Iterable[
        tuple[str, tuple[int, int, int, int], float]
    ],
    targets: list[tuple[str, tuple[int, int, int, int]]],
) -> float:
    ranked = sorted(predictions, key=lambda item: item[2], reverse=True)
    if not targets:
        return 0.0
    matched: set[int] = set()
    true_positive: list[int] = []
    false_positive: list[int] = []
    for image_id, prediction_box, _ in ranked:
        candidates = [
            (box_iou(prediction_box, target_box), index)
            for index, (target_image_id, target_box) in enumerate(targets)
            if target_image_id == image_id and index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched.add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    if not ranked:
        return 0.0

    tp_cumulative = np.cumsum(true_positive)
    fp_cumulative = np.cumsum(false_positive)
    recall = tp_cumulative / max(1, len(targets))
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    change_points = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum(
            (recall[change_points + 1] - recall[change_points])
            * precision[change_points + 1]
        )
    )


def detection_precision_recall(
    predictions: Iterable[
        tuple[str, tuple[int, int, int, int], float]
    ],
    targets: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[float, float]:
    ranked = sorted(predictions, key=lambda item: item[2], reverse=True)
    matched: set[int] = set()
    true_positives = 0
    for image_id, prediction_box, _ in ranked:
        candidates = [
            (box_iou(prediction_box, target_box), index)
            for index, (target_image_id, target_box) in enumerate(targets)
            if target_image_id == image_id and index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched.add(best_index)
            true_positives += 1
    precision = true_positives / max(1, len(ranked))
    recall = true_positives / max(1, len(targets))
    return float(precision), float(recall)


def predicted_boxes_with_scores(
    probability: np.ndarray, threshold: float = 0.5
) -> list[tuple[tuple[int, int, int, int], float]]:
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if area < 4:
            continue
        score = float(probability[labels == component].mean())
        boxes.append(((int(x), int(y), int(x + width), int(y + height)), score))
    return boxes
