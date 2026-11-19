from __future__ import annotations

import csv
import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch

import cv2
import numpy as np
import torch

import polyp_pipeline.dataset_prep as dataset_prep
import polyp_pipeline.hyperkvasir as hyperkvasir
from polyp_pipeline.data import (
    Case,
    NonSingletonBatchSampler,
    UnlabeledImageDataset,
    create_outer_folds,
    discover_cases,
    discover_unlabeled_images,
)
from polyp_pipeline.deep import (
    _filter_unlabeled_images,
    _image_content_hash,
    _uncertainty_pseudo_label_loss,
    evaluate_external_sets,
    train_cross_validation,
)
from polyp_pipeline.features import extract_patch_features, label_patches
from polyp_pipeline.metrics import (
    average_precision_50,
    box_iou,
    scale_boxes,
    segmentation_metrics,
)
from polyp_pipeline.report import generate_report


class DatasetTests(unittest.TestCase):
    def test_preparation_records_within_split_and_rejects_cross_split_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "duplicate.zip"
            image_bytes = cv2.imencode(
                ".png", np.full((8, 8, 3), 30, dtype=np.uint8)
            )[1].tobytes()
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("train/image.png", image_bytes)
                archive.writestr("test/image.png", image_bytes)
            with ZipFile(archive_path) as archive:
                train_member = archive.getinfo("train/image.png")
                test_member = archive.getinfo("test/image.png")
                duplicates = dataset_prep._validate_unique_images(
                    archive,
                    {
                        "colon_test": {
                            "first": (train_member, train_member),
                            "second": (test_member, test_member),
                        }
                    },
                )
                self.assertEqual(len(duplicates), 1)
                with self.assertRaisesRegex(ValueError, "Exact decoded image"):
                    dataset_prep._validate_unique_images(
                        archive,
                        {
                            "clinic_train": {"train": (train_member, train_member)},
                            "clinic_test": {"test": (test_member, test_member)},
                        },
                    )

    def test_zenodo_preparation_excludes_kvasir_and_extracts_held_out_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "synthetic.zip"
            kvasir_root = root / "kvasir"
            output_root = root / "prepared"
            image = np.full((8, 8, 3), 20, dtype=np.uint8)
            auxiliary_image = np.full((8, 8, 3), 40, dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            image_bytes = cv2.imencode(".png", image)[1].tobytes()
            auxiliary_image_bytes = cv2.imencode(".png", auxiliary_image)[1].tobytes()
            mask_bytes = cv2.imencode(".png", mask)[1].tobytes()
            with ZipFile(archive_path, "w") as archive:
                for stem, encoded in (
                    ("kvasir", image_bytes),
                    ("auxiliary", auxiliary_image_bytes),
                ):
                    archive.writestr(f"NewTRimage/{stem}.png", encoded)
                    archive.writestr(f"NewTRmask/{stem}.png", mask_bytes)
                for index, dataset_name in enumerate(
                    ("CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB")
                ):
                    test_image = np.full((8, 8, 3), 100 + index, dtype=np.uint8)
                    test_image_bytes = cv2.imencode(".png", test_image)[1].tobytes()
                    archive.writestr(
                        f"TestDataset/TestDataset/{dataset_name}/images/test.png",
                        test_image_bytes,
                    )
                    archive.writestr(
                        f"TestDataset/TestDataset/{dataset_name}/masks/test.png",
                        mask_bytes,
                    )
            (kvasir_root / "images").mkdir(parents=True)
            (kvasir_root / "masks").mkdir()
            self.assertTrue(
                cv2.imwrite(str(kvasir_root / "images" / "kvasir.jpg"), image)
            )
            self.assertTrue(
                cv2.imwrite(str(kvasir_root / "masks" / "kvasir.png"), mask)
            )
            archive_md5 = hashlib.md5(
                archive_path.read_bytes(), usedforsecurity=False
            ).hexdigest()
            with (
                patch.object(dataset_prep, "ZENODO_ARCHIVE_MD5", archive_md5),
                patch.object(dataset_prep, "EXPECTED_TRAIN_PAIRS", 2),
                patch.object(dataset_prep, "EXPECTED_KVASIR_OVERLAP", 1),
                patch.object(
                    dataset_prep,
                    "EXPECTED_EXTERNAL_COUNTS",
                    {
                        "clinic_test": 1,
                        "colon_test": 1,
                        "etis_test": 1,
                    },
                ),
            ):
                manifest = dataset_prep.prepare_zenodo_datasets(
                    archive_path, kvasir_root, output_root
                )
                dataset_prep.prepare_zenodo_datasets(
                    archive_path, kvasir_root, output_root
                )

            self.assertEqual(manifest["excluded_kvasir_overlap"], 1)
            self.assertEqual(manifest["clinic_train_cases"], 1)
            self.assertEqual(
                [case.case_id for case in discover_cases(output_root / "clinic_train")],
                ["auxiliary"],
            )
            for name in ("clinic_test", "colon_test", "etis_test"):
                self.assertEqual(len(discover_cases(output_root / name)), 1)

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

    def test_unlabeled_images_are_deduplicated_against_protected_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_root = root / "train"
            eval_root = root / "evaluation"
            unlabeled_root = root / "unlabeled"
            for dataset_root in (train_root, eval_root):
                (dataset_root / "images").mkdir(parents=True)
                (dataset_root / "masks").mkdir()
            unlabeled_root.mkdir()

            training_image = np.full((16, 16, 3), 25, dtype=np.uint8)
            evaluation_image = np.full((16, 16, 3), 75, dtype=np.uint8)
            unique_image = np.full((16, 16, 3), 125, dtype=np.uint8)
            second_unique_image = np.full((16, 16, 3), 175, dtype=np.uint8)
            training_mask = np.zeros((16, 16), dtype=np.uint8)
            for dataset_root, image, case_name in (
                (train_root, training_image, "train"),
                (eval_root, evaluation_image, "eval"),
            ):
                cv2.imwrite(
                    str(dataset_root / "images" / f"{case_name}.png"), image
                )
                cv2.imwrite(
                    str(dataset_root / "masks" / f"{case_name}.png"),
                    training_mask,
                )
            cv2.imwrite(str(unlabeled_root / "train_copy.png"), training_image)
            cv2.imwrite(str(unlabeled_root / "eval_copy.png"), evaluation_image)
            cv2.imwrite(str(unlabeled_root / "unique.png"), unique_image)
            cv2.imwrite(str(unlabeled_root / "unique_copy.png"), unique_image)
            cv2.imwrite(str(unlabeled_root / "second_unique.png"), second_unique_image)
            paths = discover_unlabeled_images(unlabeled_root)
            protected_cases = [
                *discover_cases(train_root),
                *discover_cases(eval_root),
            ]

            eligible, manifest, config = _filter_unlabeled_images(
                paths,
                unlabeled_root,
                protected_cases,
                protected_cases[1:],
            )

            self.assertEqual(
                [path.name for path in eligible], ["second_unique.png", "unique.png"]
            )
            self.assertEqual(config["eligible_image_count"], 2)
            self.assertEqual(
                sum(entry["status"] == "excluded" for entry in manifest), 3
            )
            self.assertEqual(
                config["excluded_evaluation_roots"], [str(eval_root.resolve())]
            )

    def test_unlabeled_dataset_returns_normalized_image_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image = np.full((48, 64, 3), 100, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            tensor = UnlabeledImageDataset(
                [image_path],
                size=32,
                preprocessing="raw",
                augment=False,
            )[0]

            self.assertEqual(tuple(tensor.shape), (3, 32, 32))
            self.assertGreater(float(tensor.min()), 0.0)
            self.assertLess(float(tensor.max()), 1.0)


class SemiSupervisedTests(unittest.TestCase):
    def test_cross_validation_trains_and_checkpoints_with_unlabeled_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases = []
            for index in range(4):
                image_path = root / f"case_{index}.png"
                mask_path = root / f"mask_{index}.png"
                image = np.full((32, 32, 3), 20 + index * 20, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                cv2.imwrite(str(image_path), image)
                cv2.imwrite(str(mask_path), mask)
                cases.append(Case(f"case_{index}", image_path, mask_path))
            unlabeled_root = root / "unlabeled"
            unlabeled_root.mkdir()
            unlabeled_paths = []
            for index in range(4):
                image_path = unlabeled_root / f"unlabeled_{index}.png"
                image = np.full((32, 32, 3), 120 + index * 20, dtype=np.uint8)
                cv2.imwrite(str(image_path), image)
                unlabeled_paths.append(image_path)
            output_dir = root / "run"

            with (
                patch(
                    "polyp_pipeline.deep.build_model",
                    side_effect=lambda *args, **kwargs: torch.nn.Conv2d(3, 1, 1),
                ),
                patch("polyp_pipeline.deep._model_complexity", return_value=(0.0, 0.0)),
            ):
                results = train_cross_validation(
                    cases,
                    output_dir,
                    model_name="unet_resnet34",
                    preprocessing="raw",
                    n_splits=2,
                    image_size=32,
                    batch_size=2,
                    epochs=1,
                    patience=1,
                    device_name="cpu",
                    overlays=False,
                    write_predictions=False,
                    consistency_weight=0.1,
                    unlabeled_image_paths=unlabeled_paths,
                    unlabeled_image_dir=unlabeled_root,
                )

            run_config = json.loads(
                (output_dir / "run_config.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_dir / "unlabeled_data_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint_exists = (
                output_dir / "fold_1" / "latest_training.pt"
            ).is_file()

        self.assertEqual(len(results), 2)
        self.assertEqual(
            run_config["unlabeled_dataset"]["eligible_image_count"], 4
        )
        self.assertEqual(
            sum(entry["status"] == "included" for entry in manifest["images"]), 4
        )
        self.assertTrue(checkpoint_exists)

    def test_external_evaluation_rejects_unlabeled_training_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment_dir = root / "experiment"
            dataset_root = root / "external"
            (experiment_dir).mkdir()
            (dataset_root / "images").mkdir(parents=True)
            (dataset_root / "masks").mkdir()
            image_path = dataset_root / "images" / "same.png"
            mask_path = dataset_root / "masks" / "same.png"
            image = np.full((16, 16, 3), 90, dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(mask_path), mask)
            case = Case("external_same", image_path, mask_path)
            image_hash = _image_content_hash(case)
            (experiment_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model": "unet_resnet34",
                        "preprocessing": "raw",
                        "image_size": 32,
                        "folds": 2,
                        "unlabeled_dataset": {
                            "eligible_manifest_sha256": "manifest-hash"
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "unlabeled_data_manifest.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "decoded_image_sha256": image_hash,
                                "status": "included",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "overlap the unlabeled"):
                evaluate_external_sets(
                    experiment_dir,
                    {"external": [case]},
                    device_name="cpu",
                )

    def test_uncertainty_pseudo_label_loss_ignores_uncertain_teacher_pixels(self) -> None:
        student_logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
        teacher_logits = torch.zeros_like(student_logits)

        loss = _uncertainty_pseudo_label_loss(student_logits, teacher_logits)

        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())
        self.assertEqual(float(student_logits.grad.abs().sum()), 0.0)

    def test_uncertainty_pseudo_label_loss_trains_on_confident_teacher_pixels(self) -> None:
        student_logits = torch.full((1, 1, 4, 4), -3.0, requires_grad=True)
        teacher_logits = torch.full((1, 1, 4, 4), 3.0)

        loss = _uncertainty_pseudo_label_loss(student_logits, teacher_logits)

        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())
        self.assertLess(float(student_logits.grad.sum()), 0.0)

    def test_hyperkvasir_download_selects_and_verifies_a_seeded_sample(self) -> None:
        contents = {
            "first.jpg": b"first image bytes",
            "second.jpg": b"second image bytes",
            "third.jpg": b"third image bytes",
            "empty.jpg": b"",
        }

        def item(name: str, index: int) -> dict[str, object]:
            content = contents[name]
            return {
                "id": f"osf-{index}",
                "attributes": {
                    "name": name,
                    "kind": "file",
                    "size": len(content),
                    "extra": {"hashes": {"sha256": hashlib.sha256(content).hexdigest()}},
                },
                "links": {"download": f"https://example.test/{name}"},
            }

        first_page = {
            "data": [
                item("first.jpg", 1),
                item("second.jpg", 2),
                item("empty.jpg", 4),
            ],
            "links": {"next": "https://example.test/page/2"},
        }
        second_page = {
            "data": [item("third.jpg", 3)],
            "links": {"next": None},
        }

        def open_file(request: object, timeout: int) -> io.BytesIO:
            url = request.full_url
            return io.BytesIO(contents[Path(url).name])

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    hyperkvasir,
                    "_read_json",
                    side_effect=[first_page, second_page],
                ),
                patch.object(hyperkvasir.urllib.request, "urlopen", side_effect=open_file),
            ):
                result = hyperkvasir.download_hyperkvasir_unlabeled(
                    Path(temporary_directory),
                    max_images=2,
                    seed=12,
                    workers=2,
                )
            with (
                patch.object(
                    hyperkvasir,
                    "_read_json",
                    side_effect=[first_page, second_page],
                ),
                patch.object(
                    hyperkvasir.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("verified files should be reused"),
                ),
            ):
                resumed_result = hyperkvasir.download_hyperkvasir_unlabeled(
                    Path(temporary_directory),
                    max_images=2,
                    seed=12,
                    workers=2,
                )
            manifest = json.loads(
                (Path(temporary_directory) / "download_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["available_image_count"], 3)
        self.assertEqual(result["downloaded_image_count"], 2)
        self.assertEqual(resumed_result["downloaded_image_count"], 2)
        self.assertEqual(len(manifest["images"]), 2)
        self.assertTrue(
            all(
                record["status"] == "verified_existing"
                for record in manifest["images"]
            )
        )

    def test_external_dataset_cases_receive_a_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            self.assertTrue(
                cv2.imwrite(str(root / "images" / "frame.png"), np.zeros((8, 8, 3), np.uint8))
            )
            self.assertTrue(
                cv2.imwrite(str(root / "masks" / "frame.png"), np.zeros((8, 8), np.uint8))
            )
            cases = discover_cases(root, case_id_prefix="clinic")
            self.assertEqual([case.case_id for case in cases], ["clinic__frame"])

    def test_pair_discovery_supports_tiff_images_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            self.assertTrue(
                cv2.imwrite(
                    str(root / "images" / "frame.tiff"),
                    np.zeros((8, 8, 3), dtype=np.uint8),
                )
            )
            self.assertTrue(
                cv2.imwrite(
                    str(root / "masks" / "frame.tiff"),
                    np.zeros((8, 8), dtype=np.uint8),
                )
            )
            self.assertEqual(len(discover_cases(root)), 1)

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
        for loss_name in (
            "focal_tversky",
            "dice_boundary",
            "bce_dice_boundary",
            "bce_dice_edge_boundary",
        ):
            logits = torch.zeros_like(targets, requires_grad=True)
            loss = segmentation_loss(loss_name, logits, targets)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_edge_boundary_loss_is_safe_inside_mixed_precision_autocast(self) -> None:
        from polyp_pipeline.models import segmentation_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        logits = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 8:24, 8:24] = 1.0
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            loss = segmentation_loss("bce_dice_edge_boundary", logits, targets)

        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_transformer_models_return_full_resolution_logits(self) -> None:
        import torch

        from polyp_pipeline.models import build_model

        for name in ("segformer_mit_b0", "upernet_mit_b0"):
            model = build_model(name, pretrained_encoder=False).eval()
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, 128, 128))
            self.assertEqual(tuple(logits.shape), (1, 1, 128, 128))

    def test_specialized_models_and_auxiliary_heads_return_full_resolution(self) -> None:
        import torch

        from polyp_pipeline.models import build_model

        for name in (
            "pranet_resnet34",
            "transfuse_resnet34_swin_tiny",
            "transfuse_frequency_swin_tiny",
        ):
            model = build_model(name, pretrained_encoder=False, image_size=128).eval()
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 3, 128, 128))
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(tuple(output.shape) == (1, 1, 128, 128) for output in outputs))

    def test_deep_supervision_objective_backpropagates_through_auxiliary_heads(self) -> None:
        import torch

        from polyp_pipeline.models import segmentation_objective

        targets = torch.zeros(2, 1, 32, 32)
        targets[:, :, 8:24, 8:24] = 1.0
        outputs = tuple(
            torch.zeros_like(targets, requires_grad=True) for _ in range(3)
        )
        loss = segmentation_objective("bce_dice", outputs, targets, True)
        loss.backward()
        self.assertTrue(all(output.grad is not None for output in outputs))

    def test_small_polyp_crop_is_training_only_and_keeps_aligned_masks(self) -> None:
        import random
        import torch

        from polyp_pipeline.data import KvasirDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.full((64, 64, 3), 100, dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[28:34, 29:35] = 255
            image_path, mask_path = root / "image.png", root / "mask.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            case = Case("small", image_path, mask_path)
            training = KvasirDataset(
                [case],
                size=64,
                preprocessing="raw",
                augment=True,
                remove_specular=False,
                small_polyp_crop_probability=1.0,
            )
            evaluation = KvasirDataset(
                [case],
                size=64,
                preprocessing="raw",
                augment=False,
                remove_specular=False,
                small_polyp_crop_probability=1.0,
            )
            with patch("polyp_pipeline.data.random.random", return_value=0.0), patch(
                "polyp_pipeline.data.random.randrange", return_value=0
            ), patch("polyp_pipeline.data.random.randint", return_value=0), patch(
                "polyp_pipeline.data.random.uniform", return_value=1.0
            ):
                _, cropped_mask, _ = training[0]
            _, untouched_mask, _ = evaluation[0]
            self.assertEqual(tuple(cropped_mask.shape), (1, 64, 64))
            self.assertGreater(float(cropped_mask.mean()), float(untouched_mask.mean()))
            self.assertTrue(torch.all((cropped_mask == 0) | (cropped_mask == 1)))

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

    def test_frequency_boundary_plan_ablation_cells(self) -> None:
        from polyp_pipeline.research_benchmark import frequency_boundary_plan

        plan = frequency_boundary_plan()
        self.assertEqual(len(plan), 3)
        self.assertEqual(
            {item["model"] for item in plan},
            {"transfuse_resnet34_swin_tiny", "transfuse_frequency_swin_tiny"},
        )
        self.assertEqual(
            {float(item["consistency_weight"]) for item in plan}, {0.0, 0.1}
        )
        self.assertTrue(all(item["loss"] == "bce_dice_boundary" for item in plan))

    def test_frequency_boundary_runner_resumes_completed_experiments(self) -> None:
        from polyp_pipeline.research_benchmark import run_frequency_boundary_benchmark

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "benchmark"
            cases = [Case("case", Path("image.png"), Path("mask.png"))]
            with patch(
                "polyp_pipeline.research_benchmark.train_cross_validation",
                return_value=[],
            ) as train:
                state = run_frequency_boundary_benchmark(
                    cases,
                    output_dir,
                    n_splits=2,
                    device_name="cpu",
                )
                self.assertEqual(state["completed_experiments"], 3)
                self.assertEqual(train.call_count, 3)
                resumed_state = run_frequency_boundary_benchmark(
                    cases,
                    output_dir,
                    n_splits=2,
                    device_name="cpu",
                )
            self.assertEqual(resumed_state["completed_experiments"], 3)
            self.assertEqual(train.call_count, 3)

    def test_haar_frequency_fusion_preserves_shape_and_gradients(self) -> None:
        import torch

        from polyp_pipeline.specialized_models import HaarFrequencyFusion

        module = HaarFrequencyFusion(4)
        features = torch.randn(2, 4, 31, 34, requires_grad=True)
        fused = module(features)
        self.assertEqual(tuple(fused.shape), tuple(features.shape))
        fused.mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())

    def test_confidence_consistency_loss_accumulates_large_masks_in_float32(self) -> None:
        import torch

        from polyp_pipeline.deep import _confidence_consistency_loss

        student = torch.zeros(1, 1, 352, 352, dtype=torch.float16, requires_grad=True)
        teacher = torch.full_like(student, 2.0)
        loss = _confidence_consistency_loss(student, teacher)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all().item())

    def test_specialized_plan_has_separate_architecture_and_crop_ablations(self) -> None:
        from polyp_pipeline.specialized_benchmark import specialized_benchmark_plan

        plan = specialized_benchmark_plan(use_auxiliary_data=False)
        self.assertEqual(len(plan), 5)
        self.assertEqual(
            {item["model"] for item in plan},
            {"pranet_resnet34", "transfuse_resnet34_swin_tiny"},
        )
        self.assertTrue(any(item["deep_supervision"] for item in plan))
        self.assertTrue(
            any(float(item["small_polyp_crop_probability"]) == 0.5 for item in plan)
        )
        joint_plan = specialized_benchmark_plan(use_auxiliary_data=True)
        self.assertTrue(
            any(item["data_condition"] == "joint_auxiliary" for item in joint_plan)
        )
        joint_only_plan = specialized_benchmark_plan(
            use_auxiliary_data=True, joint_only=True
        )
        self.assertEqual(len(joint_only_plan), 2)
        self.assertTrue(
            all(item["data_condition"] == "joint_auxiliary" for item in joint_only_plan)
        )
        with self.assertRaisesRegex(ValueError, "requires at least one auxiliary dataset"):
            specialized_benchmark_plan(use_auxiliary_data=False, joint_only=True)

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

    def test_report_summarizes_external_test_fold_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "runs" / "deep" / "model"
            run.mkdir(parents=True)
            (run / "fold_metrics.csv").write_text(
                "model,loss,preprocessing,fold,dice,miou,sensitivity,specificity\n"
                "unet_resnet34,bce_dice,raw,1,0.7,0.6,0.8,0.95\n",
                encoding="utf-8",
            )
            (run / "external_metrics.csv").write_text(
                "experiment,model,data_condition,dataset,fold,deep_supervision,"
                "small_polyp_crop_probability,auxiliary_training_cases,"
                "exact_image_duplicate_pairs,dice,miou,sensitivity,specificity\n"
                "model,unet_resnet34,joint_auxiliary,ETIS-Larib,1,1,0.5,550,2,"
                "0.6,0.45,0.7,0.9\n",
                encoding="utf-8",
            )
            other_run = root / "joint_runs" / "model2"
            other_run.mkdir(parents=True)
            (other_run / "fold_metrics.csv").write_text(
                "model,loss,preprocessing,fold,dice,miou,sensitivity,specificity\n"
                "unet_resnet34,bce_dice,raw,2,0.7,0.6,0.8,0.95\n",
                encoding="utf-8",
            )
            (other_run / "external_metrics.csv").write_text(
                "experiment,model,data_condition,dataset,fold,deep_supervision,"
                "small_polyp_crop_probability,auxiliary_training_cases,"
                "exact_image_duplicate_pairs,dice,miou,sensitivity,specificity\n"
                "model2,unet_resnet34,joint_auxiliary,ETIS-Larib,2,1,0.5,550,2,"
                "0.8,0.67,0.85,0.92\n",
                encoding="utf-8",
            )
            generate_report(
                [root / "runs", root / "joint_runs"], root / "reports"
            )
            comparison = (
                root / "reports" / "external_test_comparison.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("ETIS-Larib", comparison)
            self.assertIn("exact_image_duplicate_pairs", comparison)
            self.assertIn("ETIS-Larib,2,2,1", comparison)
            self.assertIn("0.7,0.141421356", comparison)

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
                "tta_box_recall,tta_inference_fps,crf_dice,crf_miou,"
                "crf_sensitivity,crf_specificity,crf_ap50,crf_box_precision,"
                "crf_box_recall,crf_inference_fps\n"
                "unet_resnet34,bce_dice,clahe,1,0.5,0.4,0.6,0.9,0.3,0.4,0.5,10,"
                "0.6,0.5,0.7,0.92,0.4,0.5,0.6,3,0.65,0.55,0.75,0.93,0.42,"
                "0.55,0.65,2.5\n",
                encoding="utf-8",
            )
            (run / "per_image_metrics.csv").write_text(
                "case_id,fold,dice,miou,sensitivity,specificity,"
                "target_box_count,target_box_area_fraction,is_sessile_subset,"
                "crf_dice,crf_miou,crf_sensitivity,crf_specificity\n"
                "case_a,1,0.5,0.4,0.6,0.9,1,0.01,True,0.6,0.5,0.7,0.92\n"
                "case_b,1,0.7,0.6,0.8,0.95,1,0.1,False,0.7,0.6,0.8,0.95\n",
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
            self.assertIn("unet_resnet34_crf,clahe,bce_dice,1", comparison_csv)
            self.assertIn(
                "unet_resnet34_crf,bce_dice,clahe,0,0.0,0,sessile_subset",
                subgroup_csv,
            )

    def test_report_keeps_ema_consistency_ablation_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = root / "runs"
            for name, consistency_weight in (("supervised", 0.0), ("ema", 0.1)):
                experiment = runs / name
                experiment.mkdir(parents=True)
                (experiment / "run_config.json").write_text(
                    json.dumps(
                        {
                            "model": "transfuse_frequency_swin_tiny",
                            "loss": "bce_dice_boundary",
                            "preprocessing": "clahe",
                            "consistency_weight": consistency_weight,
                            "ema_decay": 0.99,
                        }
                    ),
                    encoding="utf-8",
                )
                (experiment / "fold_metrics.csv").write_text(
                    "model,loss,preprocessing,fold,dice,miou,sensitivity,specificity\n"
                    "transfuse_frequency_swin_tiny,bce_dice_boundary,clahe,1,"
                    "0.8,0.7,0.85,0.98\n",
                    encoding="utf-8",
                )
            generate_report(runs, root / "reports")
            with (root / "reports" / "benchmark_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["consistency_weight"] for row in rows}, {"0.0", "0.1"}
            )

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
                        consistency_weight=0.1,
                    )

            self.assertTrue(
                (output_dir / "fold_1" / "latest_training.pt").is_file()
            )
            latest = torch.load(
                    output_dir / "fold_1" / "latest_training.pt",
                    map_location="cpu",
                    weights_only=False,
            )
            self.assertIn("ema_model_state", latest)
            (output_dir / "fold_1" / "latest_training.pt").write_bytes(
                b"interrupted checkpoint write"
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
                    consistency_weight=0.1,
                )
            self.assertEqual(len(results), 2)
            self.assertIn("tta_dice", results[0])
            progress = (output_dir / "progress.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "fold_resumed"', progress)
            self.assertIn(
                '"event": "checkpoint_recovered_from_best_model"', progress
            )
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "per_image_metrics_tta.csv").is_file())
            best = torch.load(
                output_dir / "fold_1" / "best_model.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(best["selected_weights"], "ema_teacher")

    def test_external_evaluation_loads_fold_checkpoints_and_persists_metrics(self) -> None:
        import torch
        from torch import nn

        from polyp_pipeline.data import Case
        from polyp_pipeline.deep import evaluate_external_sets

        class TinySegmenter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return self.layer(image)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            config = {
                "model": "unet_resnet34",
                "preprocessing": "raw",
                "image_size": 32,
                "threshold": 0.5,
                "folds": 2,
                "pretrained_encoder": False,
                "test_time_augmentation": False,
                "auxiliary_case_ids": [],
            }
            (experiment_dir / "run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            state = TinySegmenter().state_dict()
            for fold_number in (1, 2):
                fold_dir = experiment_dir / f"fold_{fold_number}"
                fold_dir.mkdir()
                torch.save(
                    {"model_name": config["model"], "model_state": state},
                    fold_dir / "best_model.pt",
                )
            image_dir, mask_dir = root / "external_images", root / "external_masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            external_cases = []
            for index in range(2):
                image_path = image_dir / f"case_{index}.png"
                mask_path = mask_dir / f"case_{index}.png"
                self.assertTrue(
                    cv2.imwrite(
                        str(image_path),
                        np.full((32, 32, 3), 80, dtype=np.uint8),
                    )
                )
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8 + index : 24 + index, 8:24] = 255
                self.assertTrue(cv2.imwrite(str(mask_path), mask))
                external_cases.append(
                    Case(f"external__case_{index}", image_path, mask_path)
                )
            with patch(
                "polyp_pipeline.deep.build_model",
                side_effect=lambda *args, **kwargs: TinySegmenter(),
            ):
                results = evaluate_external_sets(
                    experiment_dir,
                    {"ETIS-Larib": external_cases},
                    batch_size=2,
                    device_name="cpu",
                )
            self.assertEqual(len(results), 2)
            self.assertTrue(all(row["dataset"] == "ETIS-Larib" for row in results))
            self.assertTrue(all("dice" in row for row in results))
            self.assertTrue((experiment_dir / "external_metrics.csv").is_file())
            self.assertTrue(
                (experiment_dir / "external_metrics_summary.json").is_file()
            )
            external_config = json.loads(
                (experiment_dir / "external_eval_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                len(
                    external_config["within_dataset_exact_image_duplicate_pairs"][
                        "ETIS-Larib"
                    ]
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "share an identical image"):
                evaluate_external_sets(
                    experiment_dir,
                    {
                        "dataset_a": [external_cases[0]],
                        "dataset_b": [external_cases[1]],
                    },
                    batch_size=2,
                    device_name="cpu",
                )

    def test_auxiliary_exact_test_duplicate_is_excluded_from_training(self) -> None:
        import torch
        from torch import nn

        from polyp_pipeline.data import Case, create_outer_folds
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
                mask[14:18, 14:18] = 255
                image_path, mask_path = (
                    image_dir / f"case_{index}.png",
                    mask_dir / f"case_{index}.png",
                )
                self.assertTrue(cv2.imwrite(str(image_path), image))
                self.assertTrue(cv2.imwrite(str(mask_path), mask))
                cases.append(Case(f"case_{index}", image_path, mask_path))
            auxiliary_image = root / "auxiliary.png"
            self.assertTrue(
                cv2.imwrite(
                    str(auxiliary_image),
                    cv2.imread(str(cases[0].image_path), cv2.IMREAD_COLOR),
                )
            )
            auxiliary = Case("clinic__duplicate", auxiliary_image, cases[0].mask_path)

            class TinyDeepSupervisionModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.layer = nn.Conv2d(3, 1, kernel_size=1)

                def forward(
                    self, image: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    logits = self.layer(image)
                    return logits, logits * 0.5, logits * 0.25

            with (
                patch(
                    "polyp_pipeline.deep.build_model",
                    side_effect=lambda *args, **kwargs: TinyDeepSupervisionModel(),
                ),
                patch("polyp_pipeline.deep._model_complexity", return_value=(1.0, 1.0)),
            ):
                train_cross_validation(
                    cases,
                    root / "auxiliary_run",
                    model_name="pranet_resnet34",
                    preprocessing="raw",
                    n_splits=2,
                    image_size=32,
                    batch_size=2,
                    epochs=1,
                    patience=1,
                    validation_fraction=0.5,
                    device_name="cpu",
                    overlays=False,
                    deep_supervision=True,
                    small_polyp_crop_probability=0.5,
                    auxiliary_cases=[auxiliary],
                )

            test_folds = create_outer_folds(cases, n_splits=2, seed=42)
            for index, fold in enumerate(test_folds, start=1):
                manifest = json.loads(
                    (root / "auxiliary_run" / f"fold_{index}" / "split_manifest.json")
                    .read_text(encoding="utf-8")
                )
                if "case_0" in fold["test_ids"]:
                    self.assertNotIn(
                        "clinic__duplicate", manifest["auxiliary_train_ids"]
                    )
                    self.assertEqual(
                        manifest["auxiliary_excluded_test_duplicates"],
                        ["clinic__duplicate"],
                    )
                else:
                    self.assertIn(
                        "clinic__duplicate", manifest["auxiliary_train_ids"]
                    )


if __name__ == "__main__":
    unittest.main()
