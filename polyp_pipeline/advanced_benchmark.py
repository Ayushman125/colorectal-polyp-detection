from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from .data import Case
from .deep import train_cross_validation


def advanced_benchmark_plan() -> list[dict[str, str | bool]]:
    return [
        {
            "model": model,
            "loss": "bce_dice",
            "preprocessing": "clahe",
            "pretrained_encoder": True,
            "test_time_augmentation": True,
        }
        for model in (
            "fpn_resnet34",
            "deeplabv3plus_resnet34",
            "segformer_mit_b0",
            "upernet_mit_b0",
        )
    ] + [
        {
            "model": "fpn_resnet34",
            "loss": loss,
            "preprocessing": "clahe",
            "pretrained_encoder": True,
            "test_time_augmentation": True,
        }
        for loss in ("focal_tversky", "dice_boundary")
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


def _experiment_id(item: dict[str, str | bool]) -> str:
    return (
        f"{item['model']}__{item['loss']}__{item['preprocessing']}"
        "__imagenet_pretrained__tta"
    )


def run_advanced_benchmark(
    cases: list[Case],
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    image_size: int = 352,
    batch_size: int = 4,
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
    plan = advanced_benchmark_plan()
    existing_plan = (
        json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    )
    existing_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
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
        "sessile_case_ids": sorted(sessile_case_ids or set()),
        "experiments": plan,
    }
    if existing_plan and existing_plan != run_config:
        raise ValueError(
            f"Cannot resume {output_dir}; advanced benchmark configuration differs. "
            "Resume with the original command or choose a new output directory."
        )
    _atomic_json(plan_path, run_config)

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
        "advanced_benchmark_started",
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
            preprocessing=str(item["preprocessing"]),
            n_splits=n_splits,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device_name=device_name,
            pretrained_encoder=bool(item["pretrained_encoder"]),
            num_workers=num_workers,
            overlays=overlays,
            progress_log_path=progress_path,
            sessile_case_ids=sessile_case_ids,
            test_time_augmentation=bool(item["test_time_augmentation"]),
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

    completed = sum(status == "completed" for status in statuses.values())
    state = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiments": statuses,
        "completed_experiments": completed,
        "total_experiments": len(statuses),
    }
    _atomic_json(state_path, state)
    _log(
        progress_path,
        "advanced_benchmark_completed",
        completed_experiments=completed,
        total_experiments=len(statuses),
    )
    return state
