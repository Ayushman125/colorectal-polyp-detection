from __future__ import annotations

import argparse
import json
from pathlib import Path

from .advanced_benchmark import run_advanced_benchmark
from .benchmark import read_benchmark_status, run_full_benchmark
from .classical import run_classical_cv
from .data import discover_cases
from .deep import train_cross_validation
from .models import MODEL_SPECS
from .report import generate_report
from .specialized_benchmark import run_specialized_benchmark


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
        choices=(
            "bce",
            "bce_dice",
            "weighted_bce_iou",
            "focal_tversky",
            "dice_boundary",
        ),
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
    deep.add_argument("--test-time-augmentation", action="store_true")
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

    advanced = commands.add_parser(
        "advanced-benchmark",
        help="Run/resume pretrained CNN and transformer experiments with TTA.",
    )
    advanced.add_argument("--dataset-root", type=Path, default=Path("kvasir-seg") / "Kvasir-SEG")
    advanced.add_argument("--output-dir", type=Path, default=Path("runs") / "advanced_benchmark")
    advanced.add_argument("--folds", type=int, default=5)
    advanced.add_argument("--seed", type=int, default=42)
    advanced.add_argument("--size", type=int, default=352)
    advanced.add_argument("--batch-size", type=int, default=4)
    advanced.add_argument("--epochs", type=int, default=30)
    advanced.add_argument("--patience", type=int, default=6)
    advanced.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    advanced.add_argument("--num-workers", type=int, default=0)
    advanced.add_argument(
        "--sessile-image-dir",
        type=Path,
        default=Path("kvasir-sessile") / "sessile-main-Kvasir-SEG" / "images",
    )
    advanced.add_argument("--no-overlays", action="store_true")

    specialized = commands.add_parser(
        "specialized-benchmark",
        help="Run/resume PraNet-style and TransFuse-style architecture ablations.",
    )
    specialized.add_argument(
        "--dataset-root", type=Path, default=Path("kvasir-seg") / "Kvasir-SEG"
    )
    specialized.add_argument(
        "--aux-dataset-root",
        type=Path,
        action="append",
        default=[],
        help="Optional extra paired dataset with images/ and masks/; may be repeated.",
    )
    specialized.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "specialized_benchmark",
    )
    specialized.add_argument("--folds", type=int, default=5)
    specialized.add_argument("--seed", type=int, default=42)
    specialized.add_argument("--size", type=int, default=352)
    specialized.add_argument("--batch-size", type=int, default=2)
    specialized.add_argument("--epochs", type=int, default=30)
    specialized.add_argument("--patience", type=int, default=6)
    specialized.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    specialized.add_argument("--num-workers", type=int, default=0)
    specialized.add_argument(
        "--sessile-image-dir",
        type=Path,
        default=Path("kvasir-sessile") / "sessile-main-Kvasir-SEG" / "images",
    )
    specialized.add_argument(
        "--dense-crf",
        action="store_true",
        help="Evaluate DenseCRF post-processing; requires optional pydensecrf.",
    )
    specialized.add_argument("--no-overlays", action="store_true")

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
            f"{'_imagenet' if args.pretrained_encoder else ''}"
            f"{'_tta' if args.test_time_augmentation else ''}"
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
            test_time_augmentation=args.test_time_augmentation,
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
    elif args.command == "advanced-benchmark":
        sessile_case_ids = (
            {path.stem for path in args.sessile_image_dir.glob("*") if path.is_file()}
            if args.sessile_image_dir.is_dir()
            else set()
        )
        state = run_advanced_benchmark(
            cases,
            args.output_dir,
            n_splits=args.folds,
            seed=args.seed,
            image_size=args.size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
            num_workers=args.num_workers,
            sessile_case_ids=sessile_case_ids,
            overlays=not args.no_overlays,
        )
        print(json.dumps(state, indent=2))
    elif args.command == "specialized-benchmark":
        auxiliary_cases = []
        for index, dataset_root in enumerate(args.aux_dataset_root, start=1):
            auxiliary_cases.extend(
                discover_cases(
                    dataset_root,
                    case_id_prefix=f"aux{index}_{dataset_root.name}",
                )
            )
        sessile_case_ids = (
            {path.stem for path in args.sessile_image_dir.glob("*") if path.is_file()}
            if args.sessile_image_dir.is_dir()
            else set()
        )
        state = run_specialized_benchmark(
            cases,
            args.output_dir,
            auxiliary_cases=auxiliary_cases,
            n_splits=args.folds,
            seed=args.seed,
            image_size=args.size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
            num_workers=args.num_workers,
            sessile_case_ids=sessile_case_ids,
            dense_crf=args.dense_crf,
            overlays=not args.no_overlays,
        )
        print(json.dumps(state, indent=2))
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
