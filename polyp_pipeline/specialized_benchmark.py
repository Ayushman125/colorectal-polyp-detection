from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .data import Case
from .deep import _log_progress, train_cross_validation


def specialized_benchmark_plan(
    use_auxiliary_data: bool,
) -> list[dict[str, str | bool | float]]:
    plan: list[dict[str, str | bool | float]] = [
        {
            "model": "pranet_resnet34",
            "deep_supervision": False,
            "small_polyp_crop_probability": 0.0,
            "data_condition": "kvasir_only",
        },
        {
            "model": "pranet_resnet34",
            "deep_supervision": True,
            "small_polyp_crop_probability": 0.0,
            "data_condition": "kvasir_only",
        },
        {
            "model": "pranet_resnet34",
            "deep_supervision": True,
            "small_polyp_crop_probability": 0.5,
            "data_condition": "kvasir_only",
        },
        {
            "model": "transfuse_resnet34_swin_tiny",
            "deep_supervision": False,
            "small_polyp_crop_probability": 0.0,
            "data_condition": "kvasir_only",
        },
        {
            "model": "transfuse_resnet34_swin_tiny",
            "deep_supervision": True,
            "small_polyp_crop_probability": 0.5,
            "data_condition": "kvasir_only",
        },
    ]
    if use_auxiliary_data:
        plan.extend(
            [
                {
                    "model": "pranet_resnet34",
                    "deep_supervision": True,
                    "small_polyp_crop_probability": 0.5,
                    "data_condition": "joint_auxiliary",
                },
                {
                    "model": "transfuse_resnet34_swin_tiny",
                    "deep_supervision": True,
                    "small_polyp_crop_probability": 0.5,
                    "data_condition": "joint_auxiliary",
                },
            ]
        )
    return plan


def _atomic_json(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _experiment_id(item: dict[str, str | bool | float]) -> str:
    return (
        f"{item['model']}__ds-{int(bool(item['deep_supervision']))}"
        f"__small-crop-{item['small_polyp_crop_probability']}"
        f"__{item['data_condition']}"
    )


def run_specialized_benchmark(
    cases: list[Case],
    output_dir: Path,
    auxiliary_cases: list[Case] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    image_size: int = 352,
    batch_size: int = 2,
    epochs: int = 30,
    patience: int = 6,
    device_name: str = "cuda",
    num_workers: int = 0,
    sessile_case_ids: set[str] | None = None,
    dense_crf: bool = False,
    overlays: bool = True,
) -> dict[str, Any]:
    if dense_crf:
        from .refinement import dense_crf_refine

        dense_crf_refine(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2), 0.5, dtype=np.float32),
            iterations=1,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "benchmark_state.json"
    plan_path = output_dir / "benchmark_plan.json"
    progress_path = output_dir / "progress.jsonl"
    auxiliary_cases = auxiliary_cases or []
    plan = specialized_benchmark_plan(bool(auxiliary_cases))
    run_config = {
        "dataset_cases": len(cases),
        "auxiliary_cases": len(auxiliary_cases),
        "auxiliary_case_ids": [case.case_id for case in auxiliary_cases],
        "folds": n_splits,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "device": device_name,
        "num_workers": num_workers,
        "sessile_case_ids": sorted(sessile_case_ids or set()),
        "dense_crf": dense_crf,
        "experiments": plan,
    }
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != run_config:
            raise ValueError(
                f"Cannot resume {output_dir}; specialized benchmark configuration "
                "differs. Resume with the original command or choose a new output directory."
            )
    _atomic_json(plan_path, run_config)

    existing_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    statuses = existing_state.get("experiments", {})
    for item in plan:
        statuses.setdefault(_experiment_id(item), "pending")
    _atomic_json(
        state_path,
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "experiments": statuses,
            "completed_experiments": sum(
                status == "completed" for status in statuses.values()
            ),
            "total_experiments": len(statuses),
        },
    )
    _log_progress(
        progress_path,
        "specialized_benchmark_started",
        cases=len(cases),
        auxiliary_cases=len(auxiliary_cases),
        experiments=len(plan),
        folds=n_splits,
    )

    for index, item in enumerate(plan, start=1):
        key = _experiment_id(item)
        if statuses[key] == "completed":
            continue
        statuses[key] = "running"
        _atomic_json(
            state_path,
            {
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiments": statuses,
                "active_experiment": key,
                "experiment_index": index,
                "experiment_count": len(plan),
                "completed_experiments": sum(
                    status == "completed" for status in statuses.values()
                ),
                "total_experiments": len(statuses),
            },
        )
        _log_progress(progress_path, "specialized_experiment_started", experiment=key)
        selected_auxiliary = (
            auxiliary_cases if item["data_condition"] == "joint_auxiliary" else []
        )
        train_cross_validation(
            cases,
            output_dir / "deep" / key,
            model_name=str(item["model"]),
            loss_name="bce_dice",
            preprocessing="clahe",
            n_splits=n_splits,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device_name=device_name,
            pretrained_encoder=True,
            num_workers=num_workers,
            overlays=overlays,
            progress_log_path=progress_path,
            sessile_case_ids=sessile_case_ids,
            test_time_augmentation=True,
            deep_supervision=bool(item["deep_supervision"]),
            small_polyp_crop_probability=float(
                item["small_polyp_crop_probability"]
            ),
            auxiliary_cases=selected_auxiliary,
            dense_crf=dense_crf,
        )
        statuses[key] = "completed"
        _atomic_json(
            state_path,
            {
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiments": statuses,
                "completed_experiments": sum(
                    status == "completed" for status in statuses.values()
                ),
                "total_experiments": len(statuses),
            },
        )
        _log_progress(progress_path, "specialized_experiment_completed", experiment=key)

    state = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiments": statuses,
        "completed_experiments": sum(
            status == "completed" for status in statuses.values()
        ),
        "total_experiments": len(statuses),
    }
    _atomic_json(state_path, state)
    _log_progress(
        progress_path,
        "specialized_benchmark_completed",
        completed_experiments=state["completed_experiments"],
        total_experiments=state["total_experiments"],
    )
    return state
