from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from .classical import run_classical_cv
from .data import Case
from .deep import train_cross_validation
BASELINE_MODEL_NAMES = (
    "unet_resnet34",
    "unet_resnet50",
    "unet_efficientnet_b0",
    "unetpp_resnet34",
    "unetpp_resnet50",
    "unetpp_efficientnet_b0",
    "fpn_resnet34",
    "deeplabv3plus_resnet34",
)


def _atomic_json(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _log(path: Path, event: str, **details: object) -> None:
    with path.open("a", encoding="utf-8") as file:
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


def full_benchmark_plan() -> list[dict[str, str]]:
    plan = [
        {
            "model": model_name,
            "loss": "bce_dice",
            "preprocessing": "clahe",
            "kind": "deep",
        }
        for model_name in BASELINE_MODEL_NAMES
    ]
    plan.extend(
        [
            {
                "model": "unet_resnet34",
                "loss": "bce",
                "preprocessing": "raw",
                "kind": "deep",
            },
            {
                "model": "unet_resnet34",
                "loss": "bce_dice",
                "preprocessing": "raw",
                "kind": "deep",
            },
            {
                "model": "unet_resnet34",
                "loss": "bce",
                "preprocessing": "clahe",
                "kind": "deep",
            },
            {
                "model": "unet_resnet34",
                "loss": "weighted_bce_iou",
                "preprocessing": "clahe",
                "kind": "deep",
            },
        ]
    )
    return plan


def run_full_benchmark(
    cases: list[Case],
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    image_size: int = 352,
    batch_size: int = 4,
    epochs: int = 30,
    patience: int = 6,
    device_name: str = "cuda",
    pretrained_encoder: bool = False,
    num_workers: int = 0,
    sessile_case_ids: set[str] | None = None,
    overlays: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    state_path = output_dir / "benchmark_state.json"
    plan = full_benchmark_plan()
    existing_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    plan_path = output_dir / "benchmark_plan.json"
    existing_plan = (
        json.loads(plan_path.read_text(encoding="utf-8"))
        if plan_path.is_file()
        else {}
    )
    immutable_plan = {
        "dataset_cases": len(cases),
        "folds": n_splits,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": device_name,
        "pretrained_encoder": pretrained_encoder,
    }
    plan_changes = [
        key
        for key, value in immutable_plan.items()
        if key in existing_plan and existing_plan[key] != value
    ]
    if plan_changes:
        raise ValueError(
            "Cannot resume full benchmark after changing "
            f"{plan_changes}; choose a new --output-dir."
        )
    previous_max_epochs = int(existing_plan.get("max_epochs", epochs))
    if epochs < previous_max_epochs:
        raise ValueError(
            "Resuming with fewer epochs is not supported; keep or increase --epochs."
        )
    statuses = existing_state.get("experiments", {})
    statuses["classical"] = statuses.get("classical", "pending")
    for item in plan:
        key = f"{item['model']}__{item['loss']}__{item['preprocessing']}"
        statuses[key] = statuses.get(key, "pending")
    if epochs > previous_max_epochs:
        for item in plan:
            key = f"{item['model']}__{item['loss']}__{item['preprocessing']}"
            statuses[key] = "pending"
    _atomic_json(
        plan_path,
        {
            **immutable_plan,
            "max_epochs": epochs,
            "patience": patience,
            "num_workers": num_workers,
            "classical_experiment": "Random Forest + RBF-SVM, CLAHE",
            "deep_experiments": plan,
            "excluded_official_models": [
                "HarDNet-MSEG",
                "Polyp-PVT",
                "MedSAM 2",
            ],
        },
    )
    _atomic_json(
        state_path,
        {
            **existing_state,
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "experiments": statuses,
        },
    )
    _log(
        progress_path,
        "benchmark_started",
        cases=len(cases),
        deep_experiments=len(plan),
        total_folds=n_splits,
        max_epochs=epochs,
        completed_experiments=sum(status == "completed" for status in statuses.values()),
    )

    def set_status(key: str, status: str, **details: object) -> None:
        statuses[key] = status
        _atomic_json(
            state_path,
            {
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiments": statuses,
                **details,
            },
        )
        _log(progress_path, "experiment_status", experiment=key, status=status)

    classical_dir = output_dir / "classical__clahe"
    classical_key = "classical"
    if statuses[classical_key] != "completed":
        set_status(classical_key, "running")
        run_classical_cv(
            cases,
            classical_dir,
            n_splits=n_splits,
            seed=seed,
            size=image_size,
            preprocessing="clahe",
            sessile_case_ids=sessile_case_ids or set(),
        )
        set_status(classical_key, "completed")

    for index, item in enumerate(plan, start=1):
        key = f"{item['model']}__{item['loss']}__{item['preprocessing']}"
        experiment_dir = (
            output_dir
            / "deep"
            / f"{item['model']}__{item['loss']}__{item['preprocessing']}"
        )
        if statuses[key] == "completed" and not (
            experiment_dir / "summary.json"
        ).is_file():
            statuses[key] = "pending"
        if statuses[key] == "completed":
            continue
        set_status(
            key,
            "running",
            active_experiment=key,
            experiment_index=index,
            experiment_count=len(plan),
        )
        train_cross_validation(
            cases,
            experiment_dir,
            model_name=item["model"],
            loss_name=item["loss"],
            preprocessing=item["preprocessing"],
            n_splits=n_splits,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device_name=device_name,
            pretrained_encoder=pretrained_encoder,
            num_workers=num_workers,
            overlays=overlays,
            progress_log_path=progress_path,
            sessile_case_ids=sessile_case_ids,
        )
        set_status(key, "completed")

    completed = sum(status == "completed" for status in statuses.values())
    _log(
        progress_path,
        "benchmark_completed",
        completed_experiments=completed,
        total_experiments=len(statuses),
    )
    state = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiments": statuses,
        "completed_experiments": completed,
        "total_experiments": len(statuses),
    }
    _atomic_json(state_path, state)
    return state


def read_benchmark_status(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "benchmark_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"No benchmark state found at {state_path}; start with the benchmark command."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    experiment_statuses = state.get("experiments", {})
    latest_progress = None
    progress_path = output_dir / "progress.jsonl"
    if progress_path.is_file():
        with progress_path.open(encoding="utf-8") as file:
            for line in file:
                latest_progress = json.loads(line)
    return {
        **state,
        "completed_experiments": sum(
            status == "completed" for status in experiment_statuses.values()
        ),
        "total_experiments": len(experiment_statuses),
        "running_experiments": [
            name for name, status in experiment_statuses.items() if status == "running"
        ],
        "pending_experiments": [
            name for name, status in experiment_statuses.items() if status == "pending"
        ],
        "latest_progress_event": latest_progress,
    }
