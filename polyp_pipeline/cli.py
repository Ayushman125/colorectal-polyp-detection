from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import read_benchmark_status, run_full_benchmark
from .classical import run_classical_cv
from .data import discover_cases
from .deep import train_cross_validation
from .models import MODEL_SPECS
from .report import generate_report


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("kvasir-seg") / "Kvasir-SEG",
        help="Folder containing paired images/ and masks/ directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--preprocessing", choices=("raw", "clahe"), default="clahe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyp-pipeline",
        description="Leakage-aware Kvasir-SEG segmentation and feature benchmarks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    deep = commands.add_parser("train", help="Train/evaluate a deep model with outer K-fold CV.")
    _common_arguments(deep)
    deep.add_argument("--model", choices=tuple(MODEL_SPECS), default="unet_resnet34")
    deep.add_argument(
        "--loss",
        choices=("bce", "bce_dice", "weighted_bce_iou"),
        default="bce_dice",
    )
    deep.add_argument("--batch-size", type=int, default=4)
    deep.add_argument("--epochs", type=int, default=30)
    deep.add_argument("--patience", type=int, default=6)
    deep.add_argument("--learning-rate", type=float, default=1e-4)
    deep.add_argument("--validation-fraction", type=float, default=0.1)
    deep.add_argument("--threshold", type=float, default=0.5)
    deep.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    deep.add_argument("--num-workers", type=int, default=0)
    deep.add_argument("--pretrained-encoder", action="store_true")
    deep.add_argument("--no-overlays", action="store_true")

    classical = commands.add_parser(
        "classical", help="Run image-grouped Random Forest and SVM patch baselines."
    )
    _common_arguments(classical)
    classical.add_argument("--grid-size", type=int, default=4)

    report = commands.add_parser(
        "report", help="Combine completed experiments into a comparison table and chart."
    )
    report.add_argument("--input-dir", type=Path, default=Path("runs"))
    report.add_argument("--output-dir", type=Path, default=Path("reports"))

    benchmark = commands.add_parser(
        "benchmark",
        help="Run/resume the complete implemented model, ablation, and classical matrix.",
    )
    _common_arguments(benchmark)
    benchmark.add_argument("--batch-size", type=int, default=4)
    benchmark.add_argument("--epochs", type=int, default=30)
    benchmark.add_argument("--patience", type=int, default=6)
    benchmark.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    benchmark.add_argument("--num-workers", type=int, default=0)
    benchmark.add_argument("--pretrained-encoder", action="store_true")
    benchmark.add_argument(
        "--sessile-image-dir",
        type=Path,
        default=Path("kvasir-sessile") / "sessile-main-Kvasir-SEG" / "images",
    )
    benchmark.add_argument("--no-overlays", action="store_true")

    status = commands.add_parser(
        "status", help="Show completed, running, and pending benchmark experiments."
    )
    status.add_argument("--input-dir", type=Path, default=Path("runs") / "full_benchmark")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "report":
        print(generate_report(args.input_dir, args.output_dir))
        return
    if args.command == "status":
        print(json.dumps(read_benchmark_status(args.input_dir), indent=2))
        return
    cases = discover_cases(args.dataset_root)
    print(f"Found {len(cases)} paired Kvasir-SEG images in {args.dataset_root}")
    if args.command == "train":
        output_dir = args.output_dir / (
            f"{args.model}_{args.loss}_{args.preprocessing}"
        )
        train_cross_validation(
            cases,
            output_dir,
            model_name=args.model,
            loss_name=args.loss,
            preprocessing=args.preprocessing,
            n_splits=args.folds,
            seed=args.seed,
            image_size=args.size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            threshold=args.threshold,
            device_name=args.device,
            pretrained_encoder=args.pretrained_encoder,
            num_workers=args.num_workers,
            overlays=not args.no_overlays,
        )
    elif args.command == "classical":
        output_dir = args.output_dir / f"classical_{args.preprocessing}"
        run_classical_cv(
            cases,
            output_dir,
            n_splits=args.folds,
            seed=args.seed,
            size=args.size,
            grid_size=args.grid_size,
            preprocessing=args.preprocessing,
        )
    else:
        sessile_case_ids = (
            {path.stem for path in args.sessile_image_dir.glob("*") if path.is_file()}
            if args.sessile_image_dir.is_dir()
            else set()
        )
        state = run_full_benchmark(
            cases,
            args.output_dir / "full_benchmark",
            n_splits=args.folds,
            seed=args.seed,
            image_size=args.size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
            pretrained_encoder=args.pretrained_encoder,
            num_workers=args.num_workers,
            sessile_case_ids=sessile_case_ids,
            overlays=not args.no_overlays,
        )
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
