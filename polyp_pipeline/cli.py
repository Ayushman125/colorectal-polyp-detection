from __future__ import annotations

import argparse
import json
from pathlib import Path

from .advanced_benchmark import run_advanced_benchmark
from .benchmark import read_benchmark_status, run_full_benchmark
from .classical import run_classical_cv
from .data import discover_cases, discover_unlabeled_images
from .dataset_prep import prepare_zenodo_datasets
from .deep import evaluate_external_sets, train_cross_validation
from .hyperkvasir import download_hyperkvasir_unlabeled
from .models import MODEL_SPECS
from .report import generate_report
from .research_benchmark import run_frequency_boundary_benchmark
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

    download = commands.add_parser(
        "download-hyperkvasir",
        help="Download a reproducible HyperKvasir unlabeled-image sample from OSF.",
    )
    download.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "hyperkvasir" / "unlabeled",
    )
    download.add_argument("--max-images", type=int, default=1000)
    download.add_argument("--all-images", action="store_true")
    download.add_argument("--seed", type=int, default=42)
    download.add_argument("--workers", type=int, default=8)

    deep = commands.add_parser("train", help="Train/evaluate a deep model with outer K-fold CV.")
    _common_arguments(deep)
    deep.add_argument("--model", choices=tuple(MODEL_SPECS), default="unet_resnet34")
    deep.add_argument(
        "--mfps-haar-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learnable Haar fusion for MFPS-Net.",
    )
    deep.add_argument(
        "--mfps-global-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable global-local feature interaction for MFPS-Net.",
    )
    deep.add_argument(
        "--mfps-edge-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the auxiliary edge decoder for MFPS-Net.",
    )
    deep.add_argument(
        "--mfps-edge-loss-weight",
        type=float,
        default=0.1,
        help="Weight for MFPS-Net auxiliary edge BCE and Dice loss.",
    )
    deep.add_argument(
        "--loss",
        choices=(
            "bce",
            "bce_dice",
            "weighted_bce_iou",
            "focal_tversky",
            "dice_boundary",
            "bce_dice_boundary",
            "bce_dice_edge_boundary",
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
    deep.add_argument("--consistency-weight", type=float, default=0.0)
    deep.add_argument("--ema-decay", type=float, default=0.99)
    deep.add_argument("--consistency-rampup-epochs", type=int, default=5)
    deep.add_argument(
        "--unlabeled-image-dir",
        type=Path,
        help="Optional image-only corpus for uncertainty-filtered EMA pseudo-labeling.",
    )
    deep.add_argument(
        "--exclude-unlabeled-dataset-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Paired evaluation cohort to exclude from unlabeled training by exact "
            "decoded-image hash; repeat for each held-out cohort."
        ),
    )
    deep.add_argument("--no-overlays", action="store_true")
    deep.add_argument("--no-predictions", action="store_true")

    classical = commands.add_parser(
        "classical", help="Run image-grouped Random Forest and SVM patch baselines."
    )
    _common_arguments(classical)
    classical.add_argument("--grid-size", type=int, default=4)

    report = commands.add_parser(
        "report", help="Combine completed experiments into a comparison table and chart."
    )
    report.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="Experiment directory to include; repeat to combine separate run roots.",
    )
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

    research = commands.add_parser(
        "frequency-boundary-benchmark",
        help="Run/resume Swin frequency, boundary-loss, and EMA consistency ablations.",
    )
    research.add_argument(
        "--dataset-root", type=Path, default=Path("kvasir-seg") / "Kvasir-SEG"
    )
    research.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "frequency_boundary_benchmark",
    )
    research.add_argument("--folds", type=int, default=5)
    research.add_argument("--seed", type=int, default=42)
    research.add_argument("--size", type=int, default=352)
    research.add_argument("--batch-size", type=int, default=2)
    research.add_argument("--epochs", type=int, default=30)
    research.add_argument("--patience", type=int, default=6)
    research.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    research.add_argument("--num-workers", type=int, default=0)
    research.add_argument(
        "--sessile-image-dir",
        type=Path,
        default=Path("kvasir-sessile") / "sessile-main-Kvasir-SEG" / "images",
    )
    research.add_argument("--no-overlays", action="store_true")

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
        "--joint-only",
        action="store_true",
        help="Run only the auxiliary-data experiment cells; requires --aux-dataset-root.",
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

    external = commands.add_parser(
        "external-evaluate",
        help="Evaluate completed fold checkpoints on held-out paired datasets.",
    )
    external.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs") / "specialized_benchmark",
        help="Experiment directory or parent containing run_config.json files.",
    )
    external.add_argument(
        "--external-test-dataset-root",
        type=Path,
        action="append",
        required=True,
        help="Paired images/ and masks/ dataset to hold out for evaluation; repeatable.",
    )
    external.add_argument("--batch-size", type=int, default=2)
    external.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    external.add_argument("--num-workers", type=int, default=0)
    external.add_argument("--dense-crf", action="store_true")

    prepare = commands.add_parser(
        "prepare-zenodo-datasets",
        help="Verify and prepare the CVC/ETIS splits from Zenodo record 5579392.",
    )
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument(
        "--kvasir-root",
        type=Path,
        default=Path("kvasir-seg") / "Kvasir-SEG",
        help="Existing Kvasir-SEG root used to identify overlapping training cases.",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)

    status = commands.add_parser(
        "status", help="Show completed, running, and pending benchmark experiments."
    )
    status.add_argument("--input-dir", type=Path, default=Path("runs") / "full_benchmark")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download-hyperkvasir":
        print(
            json.dumps(
                download_hyperkvasir_unlabeled(
                    args.output_dir,
                    max_images=None if args.all_images else args.max_images,
                    seed=args.seed,
                    workers=args.workers,
                ),
                indent=2,
            )
        )
        return
    if args.command == "report":
        print(generate_report(args.input_dir or Path("runs"), args.output_dir))
        return
    if args.command == "status":
        print(json.dumps(read_benchmark_status(args.input_dir), indent=2))
        return
    if args.command == "external-evaluate":
        external_test_sets = {}
        for index, dataset_root in enumerate(
            args.external_test_dataset_root, start=1
        ):
            dataset_name = dataset_root.name.replace(" ", "_")
            external_test_sets[dataset_name] = discover_cases(
                dataset_root,
                case_id_prefix=f"external{index}_{dataset_name}",
            )
        experiment_dirs = sorted(
            {
                path.parent
                for path in args.input_dir.rglob("run_config.json")
                if (path.parent / "fold_metrics.csv").is_file()
            }
        )
        if args.input_dir.is_file() and args.input_dir.name == "run_config.json":
            experiment_dirs = [args.input_dir.parent]
        if not experiment_dirs:
            raise FileNotFoundError(
                f"No completed experiment configurations found under {args.input_dir}"
            )
        results = []
        for experiment_dir in experiment_dirs:
            results.extend(
                evaluate_external_sets(
                    experiment_dir,
                    external_test_sets,
                    batch_size=args.batch_size,
                    device_name=args.device,
                    num_workers=args.num_workers,
                    dense_crf=args.dense_crf,
                )
            )
        print(json.dumps({"evaluated_rows": len(results)}, indent=2))
        return
    if args.command == "prepare-zenodo-datasets":
        print(
            json.dumps(
                prepare_zenodo_datasets(
                    args.archive, args.kvasir_root, args.output_dir
                ),
                indent=2,
            )
        )
        return
    cases = discover_cases(args.dataset_root)
    print(f"Found {len(cases)} paired Kvasir-SEG images in {args.dataset_root}")
    if args.command == "train":
        unlabeled_image_paths = (
            discover_unlabeled_images(args.unlabeled_image_dir)
            if args.unlabeled_image_dir is not None
            else None
        )
        if args.exclude_unlabeled_dataset_root and unlabeled_image_paths is None:
            raise ValueError(
                "--exclude-unlabeled-dataset-root requires --unlabeled-image-dir."
            )
        unlabeled_exclusion_cases = [
            case
            for index, dataset_root in enumerate(
                args.exclude_unlabeled_dataset_root, start=1
            )
            for case in discover_cases(
                dataset_root,
                case_id_prefix=f"unlabeled_exclude{index}_{dataset_root.name}",
            )
        ]
        output_dir = args.output_dir / (
            f"{args.model}_{args.loss}_{args.preprocessing}"
            f"{'_imagenet' if args.pretrained_encoder else ''}"
            f"{'_tta' if args.test_time_augmentation else ''}"
            f"{'_semisup' if unlabeled_image_paths is not None else ''}"
            f"{'_ema' if unlabeled_image_paths is None and args.consistency_weight > 0 else ''}"
            f"{'_nohaar' if not args.mfps_haar_frequency else ''}"
            f"{'_nogl' if not args.mfps_global_local else ''}"
            f"{'_noedge' if not args.mfps_edge_head else ''}"
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
            consistency_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            consistency_rampup_epochs=args.consistency_rampup_epochs,
            write_predictions=not args.no_predictions,
            unlabeled_image_paths=unlabeled_image_paths,
            unlabeled_image_dir=args.unlabeled_image_dir,
            unlabeled_exclusion_cases=unlabeled_exclusion_cases,
            mfps_haar_frequency=args.mfps_haar_frequency,
            mfps_global_local=args.mfps_global_local,
            mfps_edge_head=args.mfps_edge_head,
            mfps_edge_loss_weight=args.mfps_edge_loss_weight,
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
    elif args.command == "frequency-boundary-benchmark":
        sessile_case_ids = (
            {path.stem for path in args.sessile_image_dir.glob("*") if path.is_file()}
            if args.sessile_image_dir.is_dir()
            else set()
        )
        state = run_frequency_boundary_benchmark(
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
            joint_only=args.joint_only,
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
