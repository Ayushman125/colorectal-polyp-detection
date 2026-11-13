from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from polyp_pipeline.data import (
    Case,
    NonSingletonBatchSampler,
    create_outer_folds,
    discover_cases,
)
from polyp_pipeline.features import extract_patch_features, label_patches
from polyp_pipeline.metrics import (
    average_precision_50,
    box_iou,
    scale_boxes,
    segmentation_metrics,
)
from polyp_pipeline.report import generate_report
from unittest.mock import patch


class DatasetTests(unittest.TestCase):
    def test_pair_discovery_fails_on_unmatched_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "images" / "a.jpg"), image))
            self.assertTrue(cv2.imwrite(str(root / "masks" / "b.png"), mask))
            with self.assertRaisesRegex(ValueError, "do not match"):
                discover_cases(root)

    def test_outer_folds_partition_cases_without_within_fold_overlap(self) -> None:
        cases = [
            Case(f"case_{index}", Path(f"{index}.jpg"), Path(f"{index}.png"))
            for index in range(10)
        ]
        folds = create_outer_folds(cases, n_splits=5, seed=7)
        test_ids = []
        for fold in folds:
            self.assertFalse(set(fold["train_ids"]) & set(fold["test_ids"]))
            test_ids.extend(fold["test_ids"])
        self.assertEqual(len(test_ids), len(set(test_ids)))
        self.assertEqual(set(test_ids), {case.case_id for case in cases})

    def test_dataset_bbox_annotations_are_read_and_scaled(self) -> None:
        box = (10, 5, 80, 45)
        scaled = scale_boxes([box], original_width=100, original_height=50, size=200)
        self.assertEqual(scaled, [(20, 20, 160, 180)])

    def test_training_batch_sampler_avoids_singleton_tail_without_dropping_data(self) -> None:
        batches = list(
            NonSingletonBatchSampler(dataset_size=7, batch_size=3, shuffle=False)
        )
        self.assertTrue(all(len(batch) > 1 for batch in batches))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(7)))


class FeatureAndMetricTests(unittest.TestCase):
    def test_handcrafted_descriptors_do_not_depend_on_mask(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (120, 70, 30)
        features = extract_patch_features(image, grid_size=2)
        self.assertEqual(features.shape, (4, 108))
        self.assertTrue(np.isfinite(features).all())

    def test_patch_labels_use_mask_coverage(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:2, :2] = 1
        labels = label_patches(mask, grid_size=2, positive_fraction=0.1)
        np.testing.assert_array_equal(labels, np.asarray([1, 0, 0, 0], dtype=np.uint8))

    def test_ap50_never_matches_boxes_across_images(self) -> None:
        box = (0, 0, 10, 10)
        self.assertEqual(average_precision_50([("image-a", box, 0.9)], [("image-b", box)]), 0.0)
        self.assertEqual(average_precision_50([("image-a", box, 0.9)], [("image-a", box)]), 1.0)
        self.assertEqual(box_iou(box, box), 1.0)

    def test_segmentation_metrics_handle_empty_prediction(self) -> None:
        target = np.ones((4, 4), dtype=np.uint8)
        prediction = np.zeros_like(target)
        result = segmentation_metrics(target, prediction)
        self.assertEqual(result["dice"], 0.0)
        self.assertEqual(result["miou"], 0.0)
        self.assertEqual(result["sensitivity"], 0.0)
        self.assertEqual(result["specificity"], 0.0)

    def test_advanced_losses_produce_finite_gradients(self) -> None:
        import torch

        from polyp_pipeline.models import segmentation_loss

        targets = torch.zeros(2, 1, 32, 32)
        targets[:, :, 8:24, 8:24] = 1.0
        for loss_name in ("focal_tversky", "dice_boundary"):
            logits = torch.zeros_like(targets, requires_grad=True)
            loss = segmentation_loss(loss_name, logits, targets)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_transformer_models_return_full_resolution_logits(self) -> None:
        import torch

        from polyp_pipeline.models import build_model

        for name in ("segformer_mit_b0", "upernet_mit_b0"):
            model = build_model(name, pretrained_encoder=False).eval()
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, 128, 128))
            self.assertEqual(tuple(logits.shape), (1, 1, 128, 128))

    def test_advanced_plan_is_pretrained_and_includes_transformers_and_tta(self) -> None:
        from polyp_pipeline.advanced_benchmark import advanced_benchmark_plan

        plan = advanced_benchmark_plan()
        self.assertEqual(len(plan), 6)
        self.assertTrue(all(item["pretrained_encoder"] for item in plan))
        self.assertTrue(all(item["test_time_augmentation"] for item in plan))
        self.assertIn("segformer_mit_b0", {item["model"] for item in plan})
        self.assertIn("upernet_mit_b0", {item["model"] for item in plan})
        self.assertIn("focal_tversky", {item["loss"] for item in plan})
        self.assertIn("dice_boundary", {item["loss"] for item in plan})

    def test_report_combines_classical_fold_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "runs" / "classical_clahe"
            run.mkdir(parents=True)
            (run / "classical_metrics.json").write_text(
                '[{"model":"svm","preprocessing":"clahe","fold":1,"dice":0.5},'
                '{"model":"svm","preprocessing":"clahe","fold":2,"dice":0.7}]',
                encoding="utf-8",
            )
            table = generate_report(root / "runs", root / "reports")
            self.assertTrue(table.is_file())
            text = table.read_text(encoding="utf-8")
            self.assertIn("svm,clahe,not_applicable,2", text)
            self.assertIn("dice_mean", text)

    def test_report_summarizes_sessile_and_small_polyp_subgroups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "runs" / "deep" / "unet"
            run.mkdir(parents=True)
            (run / "run_config.json").write_text(
                json.dumps(
                    {
                        "model": "unet_resnet34",
                        "loss": "bce_dice",
                        "preprocessing": "clahe",
                    }
                ),
                encoding="utf-8",
            )
            (run / "fold_metrics.csv").write_text(
                "model,loss,preprocessing,fold,dice,miou,sensitivity,specificity,"
                "ap50,box_precision,box_recall,inference_fps,tta_dice,tta_miou,"
                "tta_sensitivity,tta_specificity,tta_ap50,tta_box_precision,"
                "tta_box_recall,tta_inference_fps\n"
                "unet_resnet34,bce_dice,clahe,1,0.5,0.4,0.6,0.9,0.3,0.4,0.5,10,"
                "0.6,0.5,0.7,0.92,0.4,0.5,0.6,3\n",
                encoding="utf-8",
            )
            (run / "per_image_metrics.csv").write_text(
                "case_id,fold,dice,miou,sensitivity,specificity,"
                "target_box_count,target_box_area_fraction,is_sessile_subset\n"
                "case_a,1,0.5,0.4,0.6,0.9,1,0.01,True\n"
                "case_b,1,0.7,0.6,0.8,0.95,1,0.1,False\n",
                encoding="utf-8",
            )
            (run / "per_image_metrics_tta.csv").write_text(
                "case_id,fold,dice,miou,sensitivity,specificity,"
                "target_box_count,target_box_area_fraction,is_sessile_subset\n"
                "case_a,1,0.6,0.5,0.7,0.92,1,0.01,True\n",
                encoding="utf-8",
            )
            generate_report(root / "runs", root / "reports")
            subgroup_csv = (root / "reports" / "subgroup_comparison.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("sessile_subset", subgroup_csv)
            self.assertIn("small_bbox_area_le_2pct", subgroup_csv)
            comparison_csv = (root / "reports" / "benchmark_comparison.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("unet_resnet34_tta,clahe,bce_dice,1", comparison_csv)

    def test_training_resumes_from_last_completed_epoch(self) -> None:
        import torch
        from torch import nn

        from polyp_pipeline.data import Case
        from polyp_pipeline.deep import train_cross_validation

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, mask_dir = root / "images", root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            cases = []
            for index in range(4):
                image = np.full((32, 32, 3), 80 + index, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                image_path, mask_path = (
                    image_dir / f"case_{index}.png",
                    mask_dir / f"case_{index}.png",
                )
                self.assertTrue(cv2.imwrite(str(image_path), image))
                self.assertTrue(cv2.imwrite(str(mask_path), mask))
                cases.append(Case(f"case_{index}", image_path, mask_path))

            class TinySegmenter(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.layer = nn.Conv2d(3, 1, kernel_size=1)

                def forward(self, image: torch.Tensor) -> torch.Tensor:
                    return self.layer(image)

            validation_calls = 0

            def fail_after_one_saved_epoch(*args: object, **kwargs: object) -> float:
                nonlocal validation_calls
                validation_calls += 1
                if validation_calls == 2:
                    raise RuntimeError("simulated interruption")
                return 0.5

            output_dir = root / "resume_run"
            with (
                patch(
                    "polyp_pipeline.deep.build_model",
                    side_effect=lambda *args, **kwargs: TinySegmenter(),
                ),
                patch(
                    "polyp_pipeline.deep._validate_epoch",
                    side_effect=fail_after_one_saved_epoch,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_cross_validation(
                        cases,
                        output_dir,
                        model_name="unet_resnet34",
                        preprocessing="raw",
                        n_splits=2,
                        image_size=32,
                        batch_size=1,
                        epochs=3,
                        patience=3,
                        validation_fraction=0.5,
                        device_name="cpu",
                        overlays=False,
                        test_time_augmentation=True,
                    )

            self.assertTrue(
                (output_dir / "fold_1" / "latest_training.pt").is_file()
            )
            with patch(
                "polyp_pipeline.deep.build_model",
                side_effect=lambda *args, **kwargs: TinySegmenter(),
            ):
                results = train_cross_validation(
                    cases,
                    output_dir,
                    model_name="unet_resnet34",
                    preprocessing="raw",
                    n_splits=2,
                    image_size=32,
                    batch_size=1,
                    epochs=3,
                    patience=3,
                    validation_fraction=0.5,
                    device_name="cpu",
                    overlays=False,
                    test_time_augmentation=True,
                )
            self.assertEqual(len(results), 2)
            self.assertIn("tta_dice", results[0])
            progress = (output_dir / "progress.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "fold_resumed"', progress)
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "per_image_metrics_tta.csv").is_file())


if __name__ == "__main__":
    unittest.main()
