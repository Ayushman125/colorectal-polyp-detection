from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
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
    "patch_accuracy",
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


def _read_result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("fold_metrics.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            for source_row in csv.DictReader(file):
                row = dict(source_row)
                rows.append(row)
                if row.get("tta_dice") not in (None, ""):
                    tta_row = {
                        key: value
                        for key, value in row.items()
                        if not key.startswith(("tta_", "crf_"))
                    }
                    tta_row["model"] = f"{row['model']}_tta"
                    for metric in (
                        "dice",
                        "miou",
                        "sensitivity",
                        "specificity",
                        "ap50",
                        "box_precision",
                        "box_recall",
                        "inference_fps",
                    ):
                        tta_row[metric] = row.get(f"tta_{metric}", "")
                    rows.append(tta_row)
                if row.get("crf_dice") not in (None, ""):
                    crf_row = {
                        key: value
                        for key, value in row.items()
                        if not key.startswith(("crf_", "tta_"))
                    }
                    crf_row["model"] = f"{row['model']}_crf"
                    for metric in (
                        "dice",
                        "miou",
                        "sensitivity",
                        "specificity",
                        "ap50",
                        "box_precision",
                        "box_recall",
                        "inference_fps",
                    ):
                        crf_row[metric] = row.get(f"crf_{metric}", "")
                    rows.append(crf_row)
    for path in root.rglob("classical_metrics.json"):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(
            f"No fold_metrics.csv or classical_metrics.json found under {root}"
        )
    return rows


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_result_rows(input_dir)
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model", "unknown")),
            str(row.get("preprocessing", "unknown")),
            str(row.get("loss", "not_applicable")),
            str(row.get("deep_supervision", "0")),
            str(row.get("small_polyp_crop_probability", "0")),
            str(row.get("auxiliary_training_cases", "0")),
        )
        grouped[key].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "benchmark_comparison.csv"
    report_rows: list[dict[str, Any]] = []
    for (model, preprocessing, loss, deep_supervision, crop_probability, auxiliary_count), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "model": model,
            "preprocessing": preprocessing,
            "loss": loss,
            "folds": len(group),
            "deep_supervision": deep_supervision,
            "small_polyp_crop_probability": crop_probability,
            "auxiliary_training_cases": auxiliary_count,
        }
        for metric in METRICS:
            values = []
            for row in group:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if value == value:
                    values.append(value)
            record[f"{metric}_mean"] = mean(values) if values else ""
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else (
                0.0 if values else ""
            )
        report_rows.append(record)

    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    subgroup_rows: list[dict[str, Any]] = []
    per_image_paths = list(input_dir.rglob("per_image_metrics.csv"))
    per_image_paths.extend(input_dir.rglob("per_image_metrics_tta.csv"))
    for path in per_image_paths:
        is_tta = path.name == "per_image_metrics_tta.csv"
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with path.open(newline="", encoding="utf-8") as file:
            image_rows = list(csv.DictReader(file))
        grouped_images: dict[
            tuple[str, str, str, str, str, str, str], list[dict[str, str]]
        ] = defaultdict(list)
        for row in image_rows:
            model_name = row.get("model", config.get("model", "unknown"))
            variants = (
                [("_tta", "")]
                if is_tta
                else [("", "")]
                + ([("_crf", "crf_")] if row.get("crf_dice", "") else [])
            )
            for suffix, metric_prefix in variants:
                grouped_images[
                    (
                        f"{model_name}{suffix}",
                        row.get("loss", config.get("loss", "not_applicable")),
                        row.get("preprocessing", config.get("preprocessing", "unknown")),
                        str(int(bool(config.get("deep_supervision", False)))),
                        str(config.get("small_polyp_crop_probability", 0.0)),
                        str(len(config.get("auxiliary_case_ids", []))),
                        metric_prefix,
                    )
                ].append(row)
        for (
            model,
            loss,
            preprocessing,
            deep_supervision,
            crop_probability,
            auxiliary_count,
            metric_prefix,
        ), model_rows in grouped_images.items():
            groups = {
                "sessile_subset": [
                    row
                    for row in model_rows
                    if row.get("is_sessile_subset", "").lower() == "true"
                ],
                "other_cases": [
                    row
                    for row in model_rows
                    if row.get("is_sessile_subset", "").lower() != "true"
                ],
                "small_bbox_area_le_2pct": [
                    row
                    for row in model_rows
                    if float(row.get("target_box_area_fraction", 0.0)) <= 0.02
                ],
                "bbox_area_over_2pct": [
                    row
                    for row in model_rows
                    if float(row.get("target_box_area_fraction", 0.0)) > 0.02
                ],
            }
            for subgroup, selected_rows in groups.items():
                if not selected_rows:
                    continue
                subgroup_record: dict[str, Any] = {
                    "model": model,
                    "loss": loss,
                    "preprocessing": preprocessing,
                    "deep_supervision": deep_supervision,
                    "small_polyp_crop_probability": crop_probability,
                    "auxiliary_training_cases": auxiliary_count,
                    "subgroup": subgroup,
                    "image_count": len(selected_rows),
                }
                for metric in ("dice", "miou", "sensitivity", "specificity"):
                    values = [
                        float(row[f"{metric_prefix}{metric}"])
                        for row in selected_rows
                    ]
                    subgroup_record[f"{metric}_mean"] = mean(values)
                    subgroup_record[f"{metric}_std"] = (
                        stdev(values) if len(values) > 1 else 0.0
                    )
                subgroup_rows.append(subgroup_record)
    if subgroup_rows:
        subgroup_path = output_dir / "subgroup_comparison.csv"
        with subgroup_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(subgroup_rows[0]))
            writer.writeheader()
            writer.writerows(subgroup_rows)

    deep_groups = [
        row
        for row in report_rows
        if row["loss"] != "not_applicable" and row["sensitivity_mean"] != ""
    ]
    if deep_groups:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [
            f"{row['model']}\n{row['preprocessing']} / {row['loss']}\n"
            f"DS={row['deep_supervision']} crop={row['small_polyp_crop_probability']} "
            f"aux={row['auxiliary_training_cases']}"
            for row in deep_groups
        ]
        x_positions = list(range(len(labels)))
        figure, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.25), 5))
        for axis, metric, title in (
            (axes[0], "dice", "Out-of-fold Dice"),
            (axes[1], "sensitivity", "Out-of-fold sensitivity"),
        ):
            means = [float(row[f"{metric}_mean"]) for row in deep_groups]
            deviations = [float(row[f"{metric}_std"]) for row in deep_groups]
            axis.bar(x_positions, means, yerr=deviations, capsize=3)
            axis.set_xticks(x_positions, labels, rotation=45, ha="right")
            axis.set_ylim(0, 1)
            axis.set_ylabel("Mean across outer folds")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "ablation_and_model_comparison.png", dpi=180)
        plt.close(figure)
    return table_path
