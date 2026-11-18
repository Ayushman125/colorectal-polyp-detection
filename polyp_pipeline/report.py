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

EXTERNAL_METRICS = (
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


def _input_roots(input_dirs: Path | list[Path]) -> list[Path]:
    return [input_dirs] if isinstance(input_dirs, Path) else input_dirs


def _read_result_rows(input_dirs: Path | list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _input_roots(input_dirs):
        for path in root.rglob("fold_metrics.csv"):
            config_path = path.parent / "run_config.json"
            config = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.is_file()
                else {}
            )
            with path.open(newline="", encoding="utf-8") as file:
                for source_row in csv.DictReader(file):
                    row = dict(source_row)
                    row.setdefault(
                        "consistency_weight",
                        config.get("consistency_weight", 0.0),
                    )
                    row.setdefault(
                        "ema_decay",
                        config.get("ema_decay", "")
                        if float(config.get("consistency_weight", 0.0)) > 0.0
                        else "",
                    )
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


def _write_external_report(
    input_dirs: Path | list[Path], output_dir: Path
) -> None:
    rows: list[dict[str, str]] = []
    for root in _input_roots(input_dirs):
        for path in root.rglob("external_metrics.csv"):
            config_path = path.parent / "run_config.json"
            config = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.is_file()
                else {}
            )
            with path.open(newline="", encoding="utf-8") as file:
                for source_row in csv.DictReader(file):
                    row = dict(source_row)
                    row.setdefault(
                        "consistency_weight",
                        config.get("consistency_weight", 0.0),
                    )
                    row.setdefault(
                        "ema_decay",
                        config.get("ema_decay", "")
                        if float(config.get("consistency_weight", 0.0)) > 0.0
                        else "",
                    )
                    rows.append(row)
    if not rows:
        return

    grouped: dict[
        tuple[str, str, str, str, str, str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model", "unknown"),
            row.get("data_condition", "unknown"),
            row.get("dataset", "unknown"),
            row.get("deep_supervision", "0"),
            row.get("small_polyp_crop_probability", "0"),
            row.get("auxiliary_training_cases", "0"),
            str(row.get("consistency_weight", "0")),
            str(row.get("ema_decay", "")),
        )
        grouped[key].append(row)

    records: list[dict[str, Any]] = []
    for (
        model,
        data_condition,
        dataset,
        deep_supervision,
        crop_probability,
        auxiliary_count,
        consistency_weight,
        ema_decay,
    ), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "model": model,
            "data_condition": data_condition,
            "dataset": dataset,
            "folds": len(group),
            "exact_image_duplicate_pairs": max(
                (int(row.get("exact_image_duplicate_pairs", 0)) for row in group),
                default=0,
            ),
            "deep_supervision": deep_supervision,
            "small_polyp_crop_probability": crop_probability,
            "auxiliary_training_cases": auxiliary_count,
            "consistency_weight": consistency_weight,
            "ema_decay": ema_decay,
        }
        for metric in EXTERNAL_METRICS:
            values = []
            for row in group:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if value == value:
                    values.append(value)
            record[f"{metric}_mean"] = mean(values) if values else ""
            record[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else (0.0 if values else "")
            )
        records.append(record)

    report_path = output_dir / "external_test_comparison.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_report(input_dir: Path | list[Path], output_dir: Path) -> Path:
    rows = _read_result_rows(input_dir)
    grouped: dict[
        tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model", "unknown")),
            str(row.get("preprocessing", "unknown")),
            str(row.get("loss", "not_applicable")),
            str(row.get("deep_supervision", "0")),
            str(row.get("small_polyp_crop_probability", "0")),
            str(row.get("auxiliary_training_cases", "0")),
            str(row.get("consistency_weight", "0")),
            str(row.get("ema_decay", "")),
        )
        grouped[key].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "benchmark_comparison.csv"
    report_rows: list[dict[str, Any]] = []
    for (
        model,
        preprocessing,
        loss,
        deep_supervision,
        crop_probability,
        auxiliary_count,
        consistency_weight,
        ema_decay,
    ), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "model": model,
            "preprocessing": preprocessing,
            "loss": loss,
            "folds": len(group),
            "deep_supervision": deep_supervision,
            "small_polyp_crop_probability": crop_probability,
            "auxiliary_training_cases": auxiliary_count,
            "consistency_weight": consistency_weight,
            "ema_decay": ema_decay,
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
    per_image_paths = []
    for root in _input_roots(input_dir):
        per_image_paths.extend(root.rglob("per_image_metrics.csv"))
        per_image_paths.extend(root.rglob("per_image_metrics_tta.csv"))
    for path in per_image_paths:
        is_tta = path.name == "per_image_metrics_tta.csv"
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with path.open(newline="", encoding="utf-8") as file:
            image_rows = list(csv.DictReader(file))
        grouped_images: dict[
            tuple[str, str, str, str, str, str, str, str, str],
            list[dict[str, str]],
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
                        str(config.get("consistency_weight", 0.0)),
                        str(
                            config.get("ema_decay", "")
                            if float(config.get("consistency_weight", 0.0)) > 0.0
                            else ""
                        ),
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
            consistency_weight,
            ema_decay,
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
                    "consistency_weight": consistency_weight,
                    "ema_decay": ema_decay,
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

    _write_external_report(input_dir, output_dir)

    deep_groups = [
        row
        for row in report_rows
        if row["loss"] != "not_applicable" and row["sensitivity_mean"] != ""
    ]
    if deep_groups:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        short_names = {
            "pranet_resnet34": "PraNet / R34",
            "transfuse_resnet34_swin_tiny": "TransFuse / Swin-T",
            "transfuse_frequency_swin_tiny": "TransFuse + Haar / Swin-T",
            "deeplabv3plus_resnet34": "DeepLabV3+ / R34",
            "segformer_mit_b0": "SegFormer / MiT-B0",
            "upernet_mit_b0": "UPerNet / MiT-B0",
        }
        labels = []
        for row in deep_groups:
            model_name = str(row["model"])
            suffix = ""
            for ending, label in (("_tta", " + TTA"), ("_crf", " + CRF")):
                if model_name.endswith(ending):
                    model_name = model_name[: -len(ending)]
                    suffix = label
                    break
            labels.append(
                f"{short_names.get(model_name, model_name)}{suffix}\n"
                f"{row['preprocessing']} / {row['loss']}\n"
                f"DS={row['deep_supervision']} crop="
                f"{row['small_polyp_crop_probability']} "
                f"consistency={row['consistency_weight']}"
            )
        x_positions = list(range(len(labels)))
        figure, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.8), 6))
        for axis, metric, title in (
            (axes[0], "dice", "Out-of-fold Dice"),
            (axes[1], "sensitivity", "Out-of-fold sensitivity"),
        ):
            means = [float(row[f"{metric}_mean"]) for row in deep_groups]
            deviations = [float(row[f"{metric}_std"]) for row in deep_groups]
            axis.bar(x_positions, means, yerr=deviations, capsize=3)
            axis.set_xticks(x_positions, labels, rotation=35, ha="right", fontsize=8)
            axis.set_ylim(0, 1)
            axis.set_ylabel("Mean across outer folds")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "ablation_and_model_comparison.png", dpi=180)
        plt.close(figure)
    return table_path
