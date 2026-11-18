from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from .data import Case
from .deep import train_cross_validation


def frequency_boundary_plan() -> list[dict[str, str | float]]:
    return [
        {
            "model": "transfuse_resnet34_swin_tiny",
            "loss": "bce_dice_boundary",
            "consistency_weight": 0.0,
        },
        {
            "model": "transfuse_frequency_swin_tiny",
            "loss": "bce_dice_boundary",
            "consistency_weight": 0.0,
        },
        {
            "model": "transfuse_frequency_swin_tiny",
            "loss": "bce_dice_boundary",
            "consistency_weight": 0.1,
        },
    ]


def _atomic_json(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _log(path: Path, event: str, **details: object) -> None:
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _experiment_id(item: dict[str, str | float]) -> str:
    frequency = "haar" if item["model"] == "transfuse_frequency_swin_tiny" else "spatial"
    consistency = f"ema-{item['consistency_weight']}"
    return f"{frequency}__{item['loss']}__{consistency}"


def run_frequency_boundary_benchmark(
    cases: list[Case],
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    image_size: int = 352,
    batch_size: int = 2,
    epochs: int = 30,
    patience: int = 6,
    device_name: str = "cuda",
    num_workers: int = 0,
    sessile_case_ids: set[str] | None = None,
    overlays: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "benchmark_state.json"
    plan_path = output_dir / "benchmark_plan.json"
    progress_path = output_dir / "progress.jsonl"
    plan = frequency_boundary_plan()
    run_config = {
        "dataset_cases": len(cases),
        "folds": n_splits,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "device": device_name,
        "num_workers": num_workers,
        "preprocessing": "clahe",
        "pretrained_encoder": True,
        "test_time_augmentation": True,
        "deep_supervision": False,
        "small_polyp_crop_probability": 0.0,
        "boundary_weight": 0.1,
        "ema_decay": 0.99,
        "consistency_rampup_epochs": 5,
        "consistency_confidence_threshold": 0.7,
        "consistency_view_strengths": [0.025, 0.1],
        "write_predictions": False,
        "sessile_case_ids": sorted(sessile_case_ids or set()),
        "experiments": plan,
    }
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != run_config:
            raise ValueError(
                f"Cannot resume {output_dir}; frequency/boundary benchmark "
                "configuration differs. Resume with the original command or "
                "choose a new output directory."
            )
    _atomic_json(plan_path, run_config)

    existing_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
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
    _log(
        progress_path,
        "frequency_boundary_benchmark_started",
        cases=len(cases),
        experiments=len(plan),
        folds=n_splits,
        max_epochs=epochs,
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
        _log(progress_path, "experiment_started", experiment=key)
        train_cross_validation(
            cases,
            output_dir / "deep" / key,
            model_name=str(item["model"]),
            loss_name=str(item["loss"]),
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
            deep_supervision=False,
            small_polyp_crop_probability=0.0,
            consistency_weight=float(item["consistency_weight"]),
            ema_decay=0.99,
            consistency_rampup_epochs=5,
            write_predictions=False,
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
        _log(progress_path, "experiment_completed", experiment=key)

    state = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiments": statuses,
        "completed_experiments": sum(
            status == "completed" for status in statuses.values()
        ),
        "total_experiments": len(statuses),
    }
    _atomic_json(state_path, state)
    _log(
        progress_path,
        "frequency_boundary_benchmark_completed",
        completed_experiments=state["completed_experiments"],
        total_experiments=state["total_experiments"],
    )
    return state
