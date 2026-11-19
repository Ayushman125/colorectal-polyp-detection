from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import sys
import time
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OSF_UNLABELED_FILES_URL = (
    "https://api.osf.io/v2/nodes/mh9sj/files/osfstorage/"
    "5dfb5c8aa95d73000c3ac51b/?format=json&page%5Bsize%5D=100"
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PUBLISHED_IMAGE_COUNT = 99_417
PAGE_SIZE = 100


def _read_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "polyp-pipeline/1.0"}
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(min(2**attempt, 16))
    raise OSError(f"Could not read the public OSF file listing: {url}") from last_error


def _image_records(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in page["data"]
        if item.get("attributes", {}).get("kind") == "file"
        and int(item["attributes"].get("size") or 0) > 0
        and Path(item["attributes"]["name"]).suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _list_image_records(
    max_images: int | None, seed: int, workers: int
) -> tuple[list[dict[str, Any]], int | None]:
    first_page = _read_json(OSF_UNLABELED_FILES_URL)
    first_records = _image_records(first_page)
    next_url = first_page.get("links", {}).get("next")
    if max_images is None or len(first_page["data"]) < PAGE_SIZE:
        records = list(first_records)
        while next_url:
            page = _read_json(next_url)
            records.extend(_image_records(page))
            next_url = page.get("links", {}).get("next")
        return records, len(records)

    total_pages = math.ceil(PUBLISHED_IMAGE_COUNT / PAGE_SIZE)
    randomizer = random.Random(seed)
    page_count = min(
        total_pages, math.ceil(max(PAGE_SIZE, max_images * 4) / PAGE_SIZE)
    )
    remaining_pages = set(range(1, total_pages + 1))
    records: list[dict[str, Any]] = []
    while True:
        page_numbers = randomizer.sample(sorted(remaining_pages), page_count)
        remaining_pages.difference_update(page_numbers)
        page_urls = [
            f"{OSF_UNLABELED_FILES_URL}&page={page_number}"
            for page_number in page_numbers
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(page_urls))
        ) as executor:
            pages = list(executor.map(_read_json, page_urls))
        records.extend(record for page in pages for record in _image_records(page))
        if len(records) >= max_images or not remaining_pages:
            break
        page_count = min(len(remaining_pages), max(10, page_count))
    return records, None


def _fetch_file(item: dict[str, Any], output_dir: Path) -> dict[str, object]:
    attributes = item["attributes"]
    name = attributes["name"]
    if Path(name).name != name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unexpected HyperKvasir image filename: {name}")
    expected_size = int(attributes["size"])
    if expected_size < 1:
        raise ValueError(f"HyperKvasir image {name} has no file content")
    expected_hash = str(
        attributes.get("extra", {}).get("hashes", {}).get("sha256") or ""
    )
    output_path = output_dir / name
    if output_path.is_file() and output_path.stat().st_size == expected_size:
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if not expected_hash or digest == expected_hash:
            return {
                "name": name,
                "osf_file_id": item["id"],
                "size": expected_size,
                "sha256": digest,
                "status": "verified_existing",
            }

    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    download_url = item["links"]["download"]
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                download_url, headers={"User-Agent": "polyp-pipeline/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial_path.open("wb") as file:
                    while chunk := response.read(1024 * 1024):
                        file.write(chunk)
            actual_size = partial_path.stat().st_size
            digest = hashlib.sha256(partial_path.read_bytes()).hexdigest()
            if actual_size != expected_size:
                raise OSError(
                    f"Size mismatch for {name}: expected {expected_size}, got {actual_size}"
                )
            if expected_hash and digest != expected_hash:
                raise OSError(f"SHA-256 mismatch for HyperKvasir image {name}")
            partial_path.replace(output_path)
            return {
                "name": name,
                "osf_file_id": item["id"],
                "size": actual_size,
                "sha256": digest,
                "status": "downloaded",
            }
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if partial_path.exists():
                partial_path.unlink()
            if attempt < 4:
                time.sleep(min(2**attempt, 16))
    raise OSError(f"Could not download HyperKvasir image {name}") from last_error


def download_hyperkvasir_unlabeled(
    output_dir: Path,
    max_images: int | None = 1000,
    seed: int = 42,
    workers: int = 8,
) -> dict[str, object]:
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive or None for the full dataset")
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    files, available_count = _list_image_records(max_images, seed, workers)
    files.sort(key=lambda item: str(item["id"]))
    if not files:
        raise RuntimeError("The OSF HyperKvasir unlabeled-image folder is empty.")

    selected_count = min(max_images, len(files)) if max_images is not None else len(files)
    selected = random.Random(seed).sample(files, selected_count)
    selected.sort(key=lambda item: str(item["id"]))
    selected_names = {str(item["attributes"]["name"]) for item in selected}
    if len(selected_names) != selected_count:
        raise ValueError("The selected HyperKvasir files do not have unique names.")
    existing_images = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    unexpected_images = [
        path
        for path in existing_images
        if path.parent.resolve() != output_dir.resolve()
        or path.name not in selected_names
    ]
    if unexpected_images:
        raise ValueError(
            "The output directory contains images outside this seeded sample; "
            f"use a separate empty directory: {unexpected_images[:5]}"
        )
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_fetch_file, item, output_dir) for item in selected
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if index % 100 == 0 or index == selected_count:
                print(
                    f"Verified {index}/{selected_count} HyperKvasir images",
                    file=sys.stderr,
                    flush=True,
                )
    records.sort(key=lambda record: str(record["osf_file_id"]))

    manifest: dict[str, object] = {
        "dataset": "HyperKvasir unlabeled images",
        "source": "https://osf.io/mh9sj/",
        "download_source": OSF_UNLABELED_FILES_URL,
        "citation": "Borgli et al., Scientific Data 7, 283 (2020), doi:10.1038/s41597-020-00622-y",
        "selection": (
            "seeded sample from randomly selected pages across the full OSF listing"
            if max_images is not None
            else "complete OSF unlabeled-image folder"
        ),
        "seed": seed,
        "published_listing_entry_count": PUBLISHED_IMAGE_COUNT,
        "available_image_count": available_count,
        "sampled_metadata_count": len(files),
        "requested_image_count": max_images,
        "downloaded_image_count": selected_count,
        "workers": workers,
        "images": records,
    }
    manifest_path = output_dir / "download_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return {
        "output_dir": str(output_dir.resolve()),
        "published_listing_entry_count": PUBLISHED_IMAGE_COUNT,
        "available_image_count": available_count,
        "downloaded_image_count": selected_count,
        "manifest_path": str(manifest_path.resolve()),
    }
