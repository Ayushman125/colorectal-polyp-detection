from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    mask_path: Path
    image_width: int | None = None
    image_height: int | None = None
    bbox_annotations: tuple[tuple[int, int, int, int], ...] = ()


class NonSingletonBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        indices = (
            torch.randperm(self.dataset_size).tolist()
            if self.shuffle
            else list(range(self.dataset_size))
        )
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, self.dataset_size, self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            if self.batch_size >= 3:
                batches[-2], batches[-1] = batches[-2][:-1], [
                    batches[-2][-1],
                    batches[-1][0],
                ]
            else:
                batches[-2].extend(batches.pop())
        yield from batches


def discover_cases(dataset_root: Path) -> list[Case]:
    """Find strictly paired image/mask files; fail rather than silently drop cases."""
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and masks/ under dataset root: {dataset_root}"
        )

    image_paths = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    mask_paths = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    missing_masks = sorted(image_paths.keys() - mask_paths.keys())
    missing_images = sorted(mask_paths.keys() - image_paths.keys())
    if missing_masks or missing_images:
        raise ValueError(
            "Dataset image/mask pairs do not match. "
            f"Images without masks: {missing_masks[:5]}; "
            f"masks without images: {missing_images[:5]}"
        )
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    bbox_file = dataset_root / "kavsir_bboxes.json"
    bbox_data: dict[str, Any] = {}
    if bbox_file.is_file():
        bbox_data = json.loads(bbox_file.read_text(encoding="utf-8"))
        missing_boxes = sorted(image_paths.keys() - bbox_data.keys())
        if missing_boxes:
            raise ValueError(
                f"Bounding-box annotations are missing for {missing_boxes[:5]}"
            )

    cases = []
    for case_id in sorted(image_paths):
        annotation = bbox_data.get(case_id, {})
        boxes = tuple(
            (
                int(box["xmin"]),
                int(box["ymin"]),
                int(box["xmax"]),
                int(box["ymax"]),
            )
            for box in annotation.get("bbox", [])
            if box.get("label", "polyp") == "polyp"
        )
        cases.append(
            Case(
                case_id,
                image_paths[case_id],
                mask_paths[case_id],
                int(annotation["width"]) if "width" in annotation else None,
                int(annotation["height"]) if "height" in annotation else None,
                boxes,
            )
        )
    return cases


def create_outer_folds(
    cases: list[Case], n_splits: int = 5, seed: int = 42
) -> list[dict[str, list[str]]]:
    """Create reproducible image-level folds; every case is test data once."""
    if n_splits < 2 or n_splits > len(cases):
        raise ValueError(f"n_splits must be between 2 and {len(cases)}")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ids = np.asarray([case.case_id for case in cases])
    return [
        {
            "train_ids": ids[train_idx].tolist(),
            "test_ids": ids[test_idx].tolist(),
        }
        for train_idx, test_idx in splitter.split(ids)
    ]


def inner_train_validation_split(
    train_ids: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    training_ids, validation_ids = train_test_split(
        train_ids,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    return list(training_ids), list(validation_ids)


def write_fold_manifest(
    path: Path, cases: list[Case], folds: list[dict[str, list[str]]], seed: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed": seed,
        "case_count": len(cases),
        "fold_count": len(folds),
        "folds": folds,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _preprocess_image(image: np.ndarray, mode: str, remove_specular: bool) -> np.ndarray:
    if remove_specular:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        highlights = cv2.inRange(
            hsv,
            np.array([0, 0, 242], dtype=np.uint8),
            np.array([180, 55, 255], dtype=np.uint8),
        )
        highlights = cv2.medianBlur(highlights, 3)
        if cv2.countNonZero(highlights):
            image = cv2.inpaint(image, highlights, 3, cv2.INPAINT_TELEA)

    if mode == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        luminance = clahe.apply(luminance)
        image = cv2.cvtColor(
            cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2RGB
        )
    elif mode != "raw":
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    return image


class KvasirDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        cases: list[Case],
        size: int = 352,
        preprocessing: str = "clahe",
        augment: bool = False,
        remove_specular: bool = True,
        imagenet_normalize: bool = False,
    ) -> None:
        self.cases = cases
        self.size = size
        self.preprocessing = preprocessing
        self.augment = augment
        self.remove_specular = remove_specular
        self.imagenet_normalize = imagenet_normalize
        if size < 32:
            raise ValueError("size must be at least 32 pixels")

    def __len__(self) -> int:
        return len(self.cases)

    def _augment_pair(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        turns = random.randrange(4)
        if turns:
            image = np.ascontiguousarray(np.rot90(image, turns))
            mask = np.ascontiguousarray(np.rot90(mask, turns))

        if random.random() < 0.7:
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-12.0, 12.0)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
            image = image.astype(np.uint8)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            hsv[..., 1] = np.clip(
                hsv[..., 1] * random.uniform(0.9, 1.1), 0, 255
            ).astype(np.int16)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case = self.cases[index]
        bgr = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(case.mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise OSError(f"Could not read image/mask for case {case.case_id}")
        if bgr.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask dimensions differ for {case.case_id}: "
                f"{bgr.shape[:2]} vs {mask.shape[:2]}"
            )

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (
            case.image_width is not None
            and case.image_height is not None
            and (bgr.shape[1] != case.image_width or bgr.shape[0] != case.image_height)
        ):
            raise ValueError(f"Image dimensions differ from bbox metadata for {case.case_id}")
        image = _preprocess_image(
            image,
            self.preprocessing,
            self.remove_specular and self.preprocessing == "clahe",
        )
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float() / 255.0
        if self.imagenet_normalize:
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=image_tensor.dtype)[:, None, None]
            standard_deviation = torch.tensor(
                (0.229, 0.224, 0.225), dtype=image_tensor.dtype
            )[:, None, None]
            image_tensor = (image_tensor - mean) / standard_deviation
        mask_tensor = torch.from_numpy(mask[None, ...].copy()).float()
        return image_tensor, mask_tensor, case.case_id
