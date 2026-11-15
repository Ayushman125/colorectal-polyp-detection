from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

import cv2
import numpy as np

from .data import discover_cases

ZENODO_ARCHIVE_MD5 = "0c45c453bc16424df7a2528966ccd939"
EXPECTED_TRAIN_PAIRS = 1450
EXPECTED_KVASIR_OVERLAP = 900
EXPECTED_EXTERNAL_COUNTS = {
    "clinic_test": 62,
    "colon_test": 380,
    "etis_test": 196,
}


def _archive_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _paired_members(
    archive: ZipFile, image_prefix: str, mask_prefix: str
) -> dict[str, tuple[ZipInfo, ZipInfo]]:
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_members: dict[str, ZipInfo] = {}
    mask_members: dict[str, ZipInfo] = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or len(path.parts) != 2:
            continue
        if path.suffix.lower() not in supported_extensions:
            continue
        if path.parts[0] == image_prefix:
            if path.stem in image_members:
                raise ValueError(f"Duplicate image stem in archive: {path.stem}")
            image_members[path.stem] = member
        elif path.parts[0] == mask_prefix:
            if path.stem in mask_members:
                raise ValueError(f"Duplicate mask stem in archive: {path.stem}")
            mask_members[path.stem] = member
    missing_masks = sorted(image_members.keys() - mask_members.keys())
    missing_images = sorted(mask_members.keys() - image_members.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Archive pairs do not match for {image_prefix}/{mask_prefix}. "
            f"Images without masks: {missing_masks[:5]}; "
            f"masks without images: {missing_images[:5]}"
        )
    return {
        stem: (image_members[stem], mask_members[stem])
        for stem in sorted(image_members)
    }


def _paired_test_members(
    archive: ZipFile, dataset_name: str
) -> dict[str, tuple[ZipInfo, ZipInfo]]:
    base = PurePosixPath("TestDataset", "TestDataset", dataset_name)
    image_prefix = base / "images"
    mask_prefix = base / "masks"
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_members: dict[str, ZipInfo] = {}
    mask_members: dict[str, ZipInfo] = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or path.suffix.lower() not in supported_extensions:
            continue
        if path.parent == image_prefix:
            if path.stem in image_members:
                raise ValueError(f"Duplicate image stem in archive: {path.stem}")
            image_members[path.stem] = member
        elif path.parent == mask_prefix:
            if path.stem in mask_members:
                raise ValueError(f"Duplicate mask stem in archive: {path.stem}")
            mask_members[path.stem] = member
    missing_masks = sorted(image_members.keys() - mask_members.keys())
    missing_images = sorted(mask_members.keys() - image_members.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Archive pairs do not match for {dataset_name}. "
            f"Images without masks: {missing_masks[:5]}; "
            f"masks without images: {missing_images[:5]}"
        )
    return {
        stem: (image_members[stem], mask_members[stem])
        for stem in sorted(image_members)
    }


def _copy_member(archive: ZipFile, member: ZipInfo, destination: Path) -> None:
    if destination.exists():
        checksum = 0
        size = 0
        with destination.open("rb") as existing:
            while block := existing.read(1024 * 1024):
                checksum = zlib.crc32(block, checksum)
                size += len(block)
        if size == member.file_size and checksum == member.CRC:
            return
        raise FileExistsError(
            f"Prepared file differs from the verified archive: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    checksum = 0
    size = 0
    with archive.open(member) as source, temporary.open("wb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
            checksum = zlib.crc32(block, checksum)
            size += len(block)
    if size != member.file_size or checksum != member.CRC:
        temporary.unlink(missing_ok=True)
        raise BadZipFile(f"CRC or size mismatch while extracting {member.filename}")
    os.replace(temporary, destination)


def _extract_pairs(
    archive: ZipFile,
    pairs: dict[str, tuple[ZipInfo, ZipInfo]],
    output_root: Path,
) -> None:
    for stem, (image_member, mask_member) in pairs.items():
        image_suffix = PurePosixPath(image_member.filename).suffix.lower()
        mask_suffix = PurePosixPath(mask_member.filename).suffix.lower()
        _copy_member(
            archive, image_member, output_root / "images" / f"{stem}{image_suffix}"
        )
        _copy_member(
            archive, mask_member, output_root / "masks" / f"{stem}{mask_suffix}"
        )


def _validate_unique_images(
    archive: ZipFile,
    datasets: dict[str, dict[str, tuple[ZipInfo, ZipInfo]]],
) -> list[dict[str, str]]:
    seen: dict[str, tuple[str, str]] = {}
    duplicate_pairs: list[dict[str, str]] = []
    for dataset_name, pairs in datasets.items():
        for case_id, (image_member, _) in pairs.items():
            encoded = archive.read(image_member)
            image = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(
                    f"Unreadable image in {dataset_name}: {image_member.filename}"
                )
            image_digest = hashlib.sha256()
            image_digest.update(str(image.shape).encode("ascii"))
            image_digest.update(np.ascontiguousarray(image).tobytes())
            digest = image_digest.hexdigest()
            if digest in seen:
                previous_dataset, previous_case = seen[digest]
                if previous_dataset != dataset_name:
                    raise ValueError(
                        "Exact decoded image appears across selected splits: "
                        f"{previous_dataset}/{previous_case} and "
                        f"{dataset_name}/{case_id}"
                    )
                duplicate_pairs.append(
                    {
                        "dataset": dataset_name,
                        "first_case_id": previous_case,
                        "duplicate_case_id": case_id,
                    }
                )
            else:
                seen[digest] = (dataset_name, case_id)
    return duplicate_pairs


def prepare_zenodo_datasets(
    archive_path: Path, kvasir_root: Path, output_root: Path
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Zenodo archive not found: {archive_path}")
    actual_md5 = _archive_md5(archive_path)
    if actual_md5 != ZENODO_ARCHIVE_MD5:
        raise ValueError(
            f"Unexpected Zenodo archive MD5 {actual_md5}; "
            f"expected {ZENODO_ARCHIVE_MD5}"
        )

    kvasir_case_ids = {case.case_id for case in discover_cases(kvasir_root)}
    with ZipFile(archive_path) as archive:
        training_pairs = _paired_members(archive, "NewTRimage", "NewTRmask")
        if len(training_pairs) != EXPECTED_TRAIN_PAIRS:
            raise ValueError(
                f"Expected {EXPECTED_TRAIN_PAIRS} training pairs, "
                f"found {len(training_pairs)}"
            )
        kvasir_overlap = set(training_pairs) & kvasir_case_ids
        if len(kvasir_overlap) != EXPECTED_KVASIR_OVERLAP:
            raise ValueError(
                f"Expected {EXPECTED_KVASIR_OVERLAP} Kvasir-ID overlaps in the "
                f"archive training split, found {len(kvasir_overlap)}"
            )
        clinic_train = {
            stem: pair
            for stem, pair in training_pairs.items()
            if stem not in kvasir_case_ids
        }
        if len(clinic_train) != EXPECTED_TRAIN_PAIRS - EXPECTED_KVASIR_OVERLAP:
            raise ValueError("Auxiliary ClinicDB training split has an unexpected size")

        external_pairs = {
            "clinic_test": _paired_test_members(archive, "CVC-ClinicDB"),
            "colon_test": _paired_test_members(archive, "CVC-ColonDB"),
            "etis_test": _paired_test_members(archive, "ETIS-LaribPolypDB"),
        }
        actual_external_counts = {
            name: len(pairs) for name, pairs in external_pairs.items()
        }
        if actual_external_counts != EXPECTED_EXTERNAL_COUNTS:
            raise ValueError(
                "Unexpected external held-out split counts: "
                f"{actual_external_counts}; expected {EXPECTED_EXTERNAL_COUNTS}"
            )
        within_split_duplicates = _validate_unique_images(
            archive, {"clinic_train": clinic_train, **external_pairs}
        )

        output_root.mkdir(parents=True, exist_ok=True)
        _extract_pairs(archive, clinic_train, output_root / "clinic_train")
        for name, pairs in external_pairs.items():
            _extract_pairs(archive, pairs, output_root / name)

    manifest: dict[str, Any] = {
        "source": "https://zenodo.org/records/5579392",
        "archive": str(archive_path.resolve()),
        "archive_md5": actual_md5,
        "kvasir_root": str(kvasir_root.resolve()),
        "kvasir_source_cases": len(kvasir_case_ids),
        "archive_training_pairs": len(training_pairs),
        "excluded_kvasir_overlap": len(kvasir_overlap),
        "clinic_train_cases": len(clinic_train),
        "external_test_cases": actual_external_counts,
        "within_split_exact_image_duplicates": within_split_duplicates,
        "outputs": {
            name: str((output_root / name).resolve())
            for name in ("clinic_train", *external_pairs)
        },
        "split_policy": (
            "Use the 550 non-Kvasir training pairs as auxiliary training data. "
            "Keep ClinicDB, ColonDB, and ETIS test splits held out. The archive's "
            "Kvasir test subset is not extracted because Kvasir is evaluated by "
            "the existing outer cross-validation."
        ),
    }
    manifest_path = output_root / "dataset_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest
