from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_SAMPLE_COUNT = 803
EXPECTED_FEATURE_COUNT = 355
EXPECTED_CLASSES = set(range(1, 10))
EXPECTED_SPLIT_COUNTS = {"train": 562, "val": 120, "test": 121}
EXPECTED_SOURCE_SHA256 = {
    "subtrain.zip": "f22edaac223b5af79d30d811b26d596f70cdc7526f79d13c5e851a5830674b13",
    "subtrainLabels.csv": "9acc4758e0bed7ad05f7c435f47ceb4f8e7bcb75ca6e601d90e2ec6c13d4270b",
}
IMAGE_SIZE = (32, 32)
IMAGE_MODE = "L"
IMAGE_FORMAT = "PNG"
SPLITS = ("train", "val", "test")
QUALITY_COLUMNS = [
    "Id",
    "Class",
    "split",
    "unknown_ratio",
    "head_unknown_ratio",
    "quality",
    "reason",
]
DATASET_COLUMNS = [
    "Id",
    "Class",
    "split",
    "image",
    "head_image",
    "mask",
    "unknown_ratio",
    "quality",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_csv(path: Path, label: str, errors: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        errors.append(f"Missing required file: {label}")
        return None
    try:
        return pd.read_csv(path, dtype={"Id": "string"})
    except Exception as exc:
        errors.append(f"Cannot read {label}: {exc}")
        return None


def read_json(path: Path, label: str, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"Missing required file: {label}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
    errors: list[str],
) -> bool:
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{label} is missing columns: {missing}")
        return False
    return True


def add_count_error(errors: list[str], label: str, count: int) -> None:
    if count:
        errors.append(f"{label}: {count}")


def count_splits(frame: pd.DataFrame) -> dict[str, int]:
    return {
        split: int((frame["split"].astype(str) == split).sum())
        for split in SPLITS
    }


def source_check(path: Path, expected_hash: str, errors: list[str]) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.name,
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": None,
        "expected_sha256": expected_hash,
        "matches_expected": False,
    }
    if not path.is_file():
        errors.append(f"Missing source file: {path}")
        return result
    try:
        digest = file_sha256(path)
    except Exception as exc:
        errors.append(f"Cannot hash source file {path}: {exc}")
        return result
    result["sha256"] = digest
    result["matches_expected"] = digest == expected_hash
    if digest != expected_hash:
        errors.append(f"Source SHA-256 mismatch: {path.name}")
    return result


def actual_file_inventory(data: Path, root_name: str) -> tuple[set[str], int, bool]:
    root = data / root_name
    if not root.is_dir():
        return set(), 0, False
    files = [path for path in root.rglob("*") if path.is_file()]
    png_paths = {
        path.relative_to(data).as_posix()
        for path in files
        if path.suffix.lower() == ".png"
    }
    return png_paths, len(files) - len(png_paths), True


def normalize_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    return None


def metric_matches(expected: object, actual: int | bool) -> bool:
    if isinstance(actual, bool):
        normalized = normalize_bool(expected)
        return normalized is not None and normalized == actual
    try:
        numeric = float(expected)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(numeric) and numeric.is_integer() and int(numeric) == actual)


def collision_statistics(groups: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {
        "definition": "decoded pixel arrays grouped by identical SHA-256",
        "unique_pixel_hashes": int(len(groups)),
        "collision_groups": int(len(duplicates)),
        "samples_in_collision_groups": int(sum(len(group) for group in duplicates)),
        "collision_extra_images": int(sum(len(group) - 1 for group in duplicates)),
        "largest_collision_group": int(max((len(group) for group in duplicates), default=1)),
        "cross_class_collision_groups": int(
            sum(len({item["Class"] for item in group}) > 1 for group in duplicates)
        ),
        "cross_split_collision_groups": int(
            sum(len({item["split"] for item in group}) > 1 for group in duplicates)
        ),
    }


def inspect_image_collection(
    data: Path,
    frame: pd.DataFrame,
    root_name: str,
    path_column: str,
    file_hash_column: str,
    pixel_hash_column: str | None,
    metric_functions: dict[str, Callable[[np.ndarray], int | bool]],
    label: str,
    errors: list[str],
) -> dict[str, object]:
    required = {"Id", "Class", "split", path_column, file_hash_column}
    if pixel_hash_column is not None:
        required.add(pixel_hash_column)
    required.update(metric_functions)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        errors.append(f"{label} manifest is missing columns: {missing_columns}")
    specs: list[dict[str, object]] = []
    declared_path_mismatches = 0
    invalid_identity_rows = 0
    if {"Id", "Class", "split"}.issubset(frame.columns):
        for row in frame.to_dict(orient="records"):
            try:
                sample_id = str(row["Id"])
                class_id = int(row["Class"])
                split = str(row["split"])
            except (TypeError, ValueError, OverflowError):
                invalid_identity_rows += 1
                continue
            relative = (
                Path(root_name)
                / split
                / f"class_{class_id}"
                / f"{sample_id}.png"
            ).as_posix()
            declared = str(row.get(path_column, "")).replace("\\", "/")
            if declared != relative:
                declared_path_mismatches += 1
            specs.append(
                {
                    "Id": sample_id,
                    "Class": class_id,
                    "split": split,
                    "relative": relative,
                    "file_hash": row.get(file_hash_column),
                    "pixel_hash": row.get(pixel_hash_column) if pixel_hash_column else None,
                    "metrics": {name: row.get(name) for name in metric_functions},
                }
            )
    expected_paths = {str(spec["relative"]) for spec in specs}
    actual_paths, unexpected_non_png_files, root_exists = actual_file_inventory(data, root_name)
    missing_paths = expected_paths - actual_paths
    extra_paths = actual_paths - expected_paths
    unreadable = 0
    invalid_format = 0
    file_hash_mismatches = 0
    pixel_hash_mismatches = 0
    missing_file_hashes = 0
    missing_pixel_hashes = 0
    file_hashes_checked = 0
    pixel_hashes_checked = 0
    valid_format_count = 0
    readable_count = 0
    all_black_count = 0
    constant_count = 0
    total_pixel_count = 0
    total_pixel_sum = 0
    total_pixel_square_sum = 0
    total_zero_pixel_count = 0
    metadata_mismatches = {name: 0 for name in metric_functions}
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for spec in specs:
        relative = str(spec["relative"])
        path = data / Path(relative)
        if relative in missing_paths:
            continue
        try:
            with Image.open(path) as image:
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
                image.load()
                array = np.asarray(image).copy()
            readable_count += 1
            if image_format == IMAGE_FORMAT and image_mode == IMAGE_MODE and image_size == IMAGE_SIZE:
                valid_format_count += 1
            else:
                invalid_format += 1
            actual_file_hash = file_sha256(path)
            expected_file_hash = spec["file_hash"]
            if expected_file_hash is not None and not pd.isna(expected_file_hash):
                file_hashes_checked += 1
                if actual_file_hash != str(expected_file_hash).lower():
                    file_hash_mismatches += 1
            else:
                missing_file_hashes += 1
            actual_pixel_hash = pixel_sha256(array)
            expected_pixel_hash = spec["pixel_hash"]
            if pixel_hash_column is not None and expected_pixel_hash is not None and not pd.isna(expected_pixel_hash):
                pixel_hashes_checked += 1
                if actual_pixel_hash != str(expected_pixel_hash).lower():
                    pixel_hash_mismatches += 1
            elif pixel_hash_column is not None:
                missing_pixel_hashes += 1
            groups[actual_pixel_hash].append(
                {
                    "Id": spec["Id"],
                    "Class": spec["Class"],
                    "split": spec["split"],
                }
            )
            all_black_count += int(bool(np.all(array == 0)))
            constant_count += int(bool(array.size and np.ptp(array) == 0))
            integer_array = array.astype(np.int64)
            total_pixel_count += int(array.size)
            total_pixel_sum += int(integer_array.sum())
            total_pixel_square_sum += int(np.square(integer_array).sum())
            total_zero_pixel_count += int((array == 0).sum())
            for name, function in metric_functions.items():
                try:
                    actual_metric = function(array)
                except Exception:
                    metadata_mismatches[name] += 1
                    continue
                if not metric_matches(spec["metrics"][name], actual_metric):
                    metadata_mismatches[name] += 1
        except Exception:
            unreadable += 1
    add_count_error(errors, f"{label} invalid identity rows", invalid_identity_rows)
    add_count_error(errors, f"{label} declared path mismatches", declared_path_mismatches)
    if not root_exists:
        errors.append(f"Missing required directory: data/{root_name}")
    add_count_error(errors, f"{label} missing PNG files", len(missing_paths))
    add_count_error(errors, f"{label} extra PNG files", len(extra_paths))
    add_count_error(errors, f"{label} unexpected non-PNG files", unexpected_non_png_files)
    add_count_error(errors, f"{label} unreadable files", unreadable)
    add_count_error(errors, f"{label} invalid PNG/L/32x32 files", invalid_format)
    add_count_error(errors, f"{label} file SHA-256 mismatches", file_hash_mismatches)
    add_count_error(errors, f"{label} pixel SHA-256 mismatches", pixel_hash_mismatches)
    add_count_error(errors, f"{label} missing file SHA-256 values", missing_file_hashes)
    add_count_error(errors, f"{label} missing pixel SHA-256 values", missing_pixel_hashes)
    if len(specs) != EXPECTED_SAMPLE_COUNT:
        errors.append(f"{label} expected manifest rows {EXPECTED_SAMPLE_COUNT}, found {len(specs)}")
    if file_hashes_checked != EXPECTED_SAMPLE_COUNT:
        errors.append(
            f"{label} expected {EXPECTED_SAMPLE_COUNT} checked file SHA-256 values, found {file_hashes_checked}"
        )
    if pixel_hash_column is not None and pixel_hashes_checked != EXPECTED_SAMPLE_COUNT:
        errors.append(
            f"{label} expected {EXPECTED_SAMPLE_COUNT} checked pixel SHA-256 values, found {pixel_hashes_checked}"
        )
    for name, count in metadata_mismatches.items():
        add_count_error(errors, f"{label} metadata mismatches for {name}", count)
    statistics = collision_statistics(groups)
    statistics["all_black_images"] = int(all_black_count)
    statistics["constant_images"] = int(constant_count)
    statistics["pixel_count"] = int(total_pixel_count)
    statistics["pixel_sum"] = int(total_pixel_sum)
    statistics["pixel_square_sum"] = int(total_pixel_square_sum)
    statistics["zero_pixel_count"] = int(total_zero_pixel_count)
    if total_pixel_count:
        mean = total_pixel_sum / total_pixel_count
        variance = max(0.0, total_pixel_square_sum / total_pixel_count - mean * mean)
        statistics["pixel_mean"] = float(mean)
        statistics["pixel_std"] = float(np.sqrt(variance))
        statistics["zero_pixel_ratio"] = float(total_zero_pixel_count / total_pixel_count)
    else:
        statistics["pixel_mean"] = None
        statistics["pixel_std"] = None
        statistics["zero_pixel_ratio"] = None
    return {
        "directory": f"data/{root_name}",
        "expected": int(len(specs)),
        "found": int(len(actual_paths)),
        "missing": int(len(missing_paths)),
        "extra": int(len(extra_paths)),
        "unexpected_non_png_files": int(unexpected_non_png_files),
        "declared_path_mismatches": int(declared_path_mismatches),
        "readable": int(readable_count),
        "unreadable": int(unreadable),
        "format": {
            "expected_container": IMAGE_FORMAT,
            "expected_mode": IMAGE_MODE,
            "expected_size": list(IMAGE_SIZE),
            "valid": int(valid_format_count),
            "invalid": int(invalid_format),
        },
        "hashes": {
            "file_sha256_checked": int(file_hashes_checked),
            "file_sha256_mismatches": int(file_hash_mismatches),
            "file_sha256_missing": int(missing_file_hashes),
            "pixel_sha256_checked": int(pixel_hashes_checked),
            "pixel_sha256_mismatches": int(pixel_hash_mismatches),
            "pixel_sha256_missing": int(missing_pixel_hashes),
        },
        "metadata_mismatches": {
            name: int(count) for name, count in metadata_mismatches.items()
        },
        "pixel_statistics": statistics,
    }


def compare_manifest_metadata(
    manifest: pd.DataFrame,
    full_manifest: pd.DataFrame,
    errors: list[str],
) -> dict[str, object]:
    columns = [
        "Class",
        "split",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "unknown_byte_ratio",
        "bytes_zip_path",
        "raw_bytes_sha256",
        "normalized_bytes_sha256",
        "token_stream_sha256",
    ]
    result: dict[str, object] = {
        "id_sets_match": False,
        "mismatch_counts": {column: None for column in columns},
    }
    if "Id" not in manifest.columns or "Id" not in full_manifest.columns:
        return result
    clean_ids = set(manifest["Id"].dropna().astype(str))
    full_ids = set(full_manifest["Id"].dropna().astype(str))
    result["id_sets_match"] = clean_ids == full_ids
    if clean_ids != full_ids:
        errors.append(
            f"Clean/full manifest ID set mismatch: missing={len(clean_ids - full_ids)}, extra={len(full_ids - clean_ids)}"
        )
        return result
    if manifest["Id"].duplicated().any() or full_manifest["Id"].duplicated().any():
        return result
    clean_index = manifest.set_index("Id").sort_index()
    full_index = full_manifest.set_index("Id").sort_index()
    mismatch_counts: dict[str, int | None] = {}
    numeric_columns = {
        "Class",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "unknown_byte_ratio",
    }
    for column in columns:
        if column not in clean_index.columns or column not in full_index.columns:
            mismatch_counts[column] = None
            errors.append(f"Cannot compare clean/full manifest column: {column}")
            continue
        if column in numeric_columns:
            left = pd.to_numeric(clean_index[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(full_index[column], errors="coerce").to_numpy(dtype=float)
            tolerance = 1e-12 if column == "unknown_byte_ratio" else 0.0
            mismatch = int(
                (~np.isclose(left, right, rtol=0.0, atol=tolerance, equal_nan=False)).sum()
            )
        else:
            left = clean_index[column].fillna("").astype(str)
            right = full_index[column].fillna("").astype(str)
            mismatch = int((left != right).sum())
        mismatch_counts[column] = mismatch
        add_count_error(errors, f"Clean/full manifest mismatches for {column}", mismatch)
    result["mismatch_counts"] = mismatch_counts
    return result


def verify_dataset(
    dataset: pd.DataFrame | None,
    samples: pd.DataFrame | None,
    image_records: pd.DataFrame | None,
    quality: pd.DataFrame | None,
    errors: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "expected_rows": EXPECTED_SAMPLE_COUNT,
        "found_rows": int(len(dataset)) if dataset is not None else None,
        "expected_columns": DATASET_COLUMNS,
        "columns_match": False,
        "duplicate_ids": None,
        "id_sets_match": False,
        "rows_match": False,
        "mismatch_counts": {},
    }
    if dataset is None:
        return result
    result["columns_match"] = list(dataset.columns) == DATASET_COLUMNS
    if not result["columns_match"]:
        errors.append(
            f"data/dataset.csv columns must be exactly {DATASET_COLUMNS}, found {list(dataset.columns)}"
        )
    if len(dataset) != EXPECTED_SAMPLE_COUNT:
        errors.append(f"Expected {EXPECTED_SAMPLE_COUNT} dataset rows, found {len(dataset)}")
    if "Id" in dataset.columns:
        duplicate_count = int(dataset["Id"].duplicated().sum())
        result["duplicate_ids"] = duplicate_count
        add_count_error(errors, "dataset.csv duplicate Id values", duplicate_count)
        add_count_error(errors, "dataset.csv missing Id values", int(dataset["Id"].isna().sum()))
    required_sources = {
        "data/manifests/samples.csv": (samples, {"Id", "Class", "split", "image_path"}),
        "data/manifests/image_records.csv": (
            image_records,
            {"Id", "full_image_path", "full_unknown_mask_path"},
        ),
        "data/manifests/quality.csv": (quality, {"Id", "unknown_ratio", "quality"}),
    }
    sources_valid = True
    for label, pair in required_sources.items():
        frame, required = pair
        if frame is None or not require_columns(frame, required, label, errors):
            sources_valid = False
    if not sources_valid or not set(DATASET_COLUMNS).issubset(dataset.columns):
        return result
    frames = [samples, image_records, quality]
    if any(frame is None for frame in frames):
        return result
    source_frames = [frame for frame in frames if frame is not None]
    if any(frame["Id"].duplicated().any() for frame in source_frames):
        errors.append("Cannot compare data/dataset.csv with sources containing duplicate Id values")
        return result
    dataset_ids = set(dataset["Id"].dropna().astype(str))
    source_id_sets = [set(frame["Id"].dropna().astype(str)) for frame in source_frames]
    id_sets_match = all(dataset_ids == source_ids for source_ids in source_id_sets)
    result["id_sets_match"] = id_sets_match
    if not id_sets_match:
        errors.append("dataset.csv and samples/image_records/quality ID sets differ")
        return result
    if dataset["Id"].duplicated().any():
        return result
    expected = samples[["Id", "Class", "split", "image_path"]].rename(
        columns={"image_path": "head_image"}
    )
    expected = expected.merge(
        image_records[["Id", "full_image_path", "full_unknown_mask_path"]].rename(
            columns={"full_image_path": "image", "full_unknown_mask_path": "mask"}
        ),
        on="Id",
        how="inner",
        validate="one_to_one",
    )
    expected = expected.merge(
        quality[["Id", "unknown_ratio", "quality"]],
        on="Id",
        how="inner",
        validate="one_to_one",
    )[DATASET_COLUMNS]
    expected_index = expected.set_index("Id").sort_index()
    actual_index = dataset[DATASET_COLUMNS].set_index("Id").sort_index()
    mismatch_counts: dict[str, int] = {}
    for column in ("Class", "unknown_ratio"):
        left = pd.to_numeric(expected_index[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(actual_index[column], errors="coerce").to_numpy(dtype=float)
        tolerance = 1e-12 if column == "unknown_ratio" else 0.0
        mismatch_counts[column] = int(
            (~np.isclose(left, right, rtol=0.0, atol=tolerance, equal_nan=False)).sum()
        )
    for column in ("split", "image", "head_image", "mask", "quality"):
        left = expected_index[column].fillna("").astype(str)
        right = actual_index[column].fillna("").astype(str)
        mismatch_counts[column] = int((left != right).sum())
    result["mismatch_counts"] = mismatch_counts
    for column, count in mismatch_counts.items():
        add_count_error(errors, f"dataset.csv mismatches for {column}", count)
    result["rows_match"] = not any(mismatch_counts.values())
    return result


def verify_quality(
    quality: pd.DataFrame | None,
    samples: pd.DataFrame | None,
    features: pd.DataFrame | None,
    errors: list[str],
) -> dict[str, int]:
    if quality is None:
        return {}
    if list(quality.columns) != QUALITY_COLUMNS:
        errors.append(
            f"data/manifests/quality.csv columns must be exactly {QUALITY_COLUMNS}, found {list(quality.columns)}"
        )
    if not set(QUALITY_COLUMNS).issubset(quality.columns):
        return {}
    if len(quality) != EXPECTED_SAMPLE_COUNT:
        errors.append(f"Expected {EXPECTED_SAMPLE_COUNT} quality rows, found {len(quality)}")
    duplicate_count = int(quality["Id"].duplicated().sum())
    add_count_error(errors, "quality.csv duplicate Id values", duplicate_count)
    add_count_error(errors, "quality.csv missing Id values", int(quality["Id"].isna().sum()))
    samples_valid = samples is not None and require_columns(
        samples,
        {"Id", "Class", "split", "unknown_byte_ratio"},
        "data/manifests/samples.csv",
        errors,
    )
    features_valid = features is not None and require_columns(
        features,
        {"Id", "image_unknown_tokens"},
        "data/features/features.csv",
        errors,
    )
    if (
        samples_valid
        and features_valid
        and samples is not None
        and features is not None
        and not samples["Id"].duplicated().any()
        and not features["Id"].duplicated().any()
        and not quality["Id"].duplicated().any()
    ):
        quality_ids = set(quality["Id"].dropna().astype(str))
        sample_ids = set(samples["Id"].dropna().astype(str))
        feature_ids = set(features["Id"].dropna().astype(str))
        if not (quality_ids == sample_ids == feature_ids):
            errors.append("samples.csv, features.csv, and quality.csv ID sets differ")
        expected = samples[["Id", "Class", "split", "unknown_byte_ratio"]].rename(
            columns={"unknown_byte_ratio": "unknown_ratio"}
        ).merge(
            features[["Id", "image_unknown_tokens"]],
            on="Id",
            how="left",
            validate="one_to_one",
        )
        expected["unknown_ratio"] = pd.to_numeric(expected["unknown_ratio"], errors="coerce")
        expected["head_unknown_ratio"] = pd.to_numeric(
            expected["image_unknown_tokens"],
            errors="coerce",
        ) / 1024.0
        quality_values: list[str] = []
        reasons: list[str] = []
        for row in expected.itertuples(index=False):
            unknown_ratio = float(row.unknown_ratio)
            head_ratio = float(row.head_unknown_ratio)
            if unknown_ratio > 0.90 or head_ratio > 0.50:
                reason_parts: list[str] = []
                if unknown_ratio > 0.90:
                    reason_parts.append("unknown>90%")
                if head_ratio > 0.50:
                    reason_parts.append("head_unknown>50%")
                quality_values.append("review")
                reasons.append(";".join(reason_parts))
            elif unknown_ratio > 0.50:
                quality_values.append("caution")
                reasons.append("unknown>50%")
            else:
                quality_values.append("normal")
                reasons.append("")
        expected["quality"] = quality_values
        expected["reason"] = reasons
        expected["unknown_ratio"] = expected["unknown_ratio"].round(6)
        expected["head_unknown_ratio"] = expected["head_unknown_ratio"].round(6)
        expected = expected[QUALITY_COLUMNS]
        expected = expected.set_index("Id").sort_index()
        actual = quality.set_index("Id").sort_index()
        if set(expected.index.astype(str)) == set(actual.index.astype(str)):
            numeric_columns = ["Class", "unknown_ratio", "head_unknown_ratio"]
            for column in numeric_columns:
                left = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
                right = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
                mismatch = int(
                    (~np.isclose(left, right, rtol=0.0, atol=1e-12, equal_nan=False)).sum()
                )
                add_count_error(errors, f"quality.csv mismatches for {column}", mismatch)
            for column in ("split", "quality", "reason"):
                left = expected[column].fillna("").astype(str)
                right = actual[column].fillna("").astype(str)
                mismatch = int((left != right).sum())
                add_count_error(errors, f"quality.csv mismatches for {column}", mismatch)
    return {
        str(key): int(value)
        for key, value in quality["quality"].value_counts().sort_index().items()
    }


def verify_image_summary(
    summary: dict[str, object] | None,
    samples_path: Path,
    full_manifest: pd.DataFrame,
    full_images: dict[str, object],
    full_masks: dict[str, object],
    errors: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "exists_and_readable": summary is not None,
        "verification_passed": False,
        "mismatches": [],
    }
    if summary is None:
        return result
    verification = summary.get("verification", {})
    output = summary.get("output", {})
    input_data = summary.get("input", {})
    quality = summary.get("quality", {})
    if not isinstance(verification, dict):
        verification = {}
    if not isinstance(output, dict):
        output = {}
    if not isinstance(input_data, dict):
        input_data = {}
    if not isinstance(quality, dict):
        quality = {}
    class_counts: dict[str, int] = {}
    source_token_count: int | None = None
    known_token_count: int | None = None
    unknown_token_count: int | None = None
    bin_statistics: dict[str, int | float | None] = {
        "all_unknown_bin_count": None,
        "samples_with_all_unknown_bins": None,
        "empty_bin_count": None,
        "minimum_bin_tokens": None,
        "maximum_bin_tokens": None,
        "weighted_unknown_byte_ratio": None,
    }
    summary_columns = {
        "Class",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "full_image_all_unknown_bin_count",
        "full_image_empty_bin_count",
        "full_image_min_bin_tokens",
        "full_image_max_bin_tokens",
    }
    if summary_columns.issubset(full_manifest.columns) and len(full_manifest):
        class_counts = {
            str(int(key)): int(value)
            for key, value in full_manifest["Class"].value_counts().sort_index().items()
        }
        source_token_count = int(pd.to_numeric(full_manifest["byte_token_count"], errors="coerce").sum())
        known_token_count = int(pd.to_numeric(full_manifest["known_byte_count"], errors="coerce").sum())
        unknown_token_count = int(pd.to_numeric(full_manifest["unknown_byte_count"], errors="coerce").sum())
        all_unknown = pd.to_numeric(full_manifest["full_image_all_unknown_bin_count"], errors="coerce")
        empty_bins = pd.to_numeric(full_manifest["full_image_empty_bin_count"], errors="coerce")
        min_bins = pd.to_numeric(full_manifest["full_image_min_bin_tokens"], errors="coerce")
        max_bins = pd.to_numeric(full_manifest["full_image_max_bin_tokens"], errors="coerce")
        bin_statistics = {
            "all_unknown_bin_count": int(all_unknown.sum()),
            "samples_with_all_unknown_bins": int((all_unknown > 0).sum()),
            "empty_bin_count": int(empty_bins.sum()),
            "minimum_bin_tokens": int(min_bins.min()),
            "maximum_bin_tokens": int(max_bins.max()),
            "weighted_unknown_byte_ratio": float(unknown_token_count / source_token_count),
        }
    expected_pairs: dict[str, tuple[object, object]] = {
        "dataset": (summary.get("dataset"), "Microsoft BIG 2015 small sample (local derivative)"),
        "method": (summary.get("method"), "full-file equal-position-bin known-byte mean"),
        "method_version": (summary.get("method_version"), 1),
        "verification.passed": (verification.get("passed"), True),
        "verification.verified_samples": (verification.get("verified_samples"), EXPECTED_SAMPLE_COUNT),
        "verification.expected_samples": (verification.get("expected_samples"), EXPECTED_SAMPLE_COUNT),
        "verification.image_count": (verification.get("image_count"), full_images.get("found")),
        "verification.mask_count": (verification.get("mask_count"), full_masks.get("found")),
        "verification.errors": (verification.get("errors"), []),
        "output.samples": (output.get("samples"), EXPECTED_SAMPLE_COUNT),
        "output.split_counts": (output.get("split_counts"), EXPECTED_SPLIT_COUNTS),
        "output.image_directory": (output.get("image_directory"), "images"),
        "output.mask_directory": (output.get("mask_directory"), "masks"),
        "output.manifest": (output.get("manifest"), "manifests/image_records.csv"),
        "output.image_size": (output.get("image_size"), list(IMAGE_SIZE)),
        "output.image_mode": (output.get("image_mode"), IMAGE_MODE),
        "output.class_counts": (output.get("class_counts"), class_counts),
        "input.manifest_samples": (input_data.get("manifest_samples"), EXPECTED_SAMPLE_COUNT),
        "input.zip_byte_entries": (input_data.get("zip_byte_entries"), EXPECTED_SAMPLE_COUNT),
        "input.zip_sha256": (input_data.get("zip_sha256"), EXPECTED_SOURCE_SHA256["subtrain.zip"]),
        "input.manifest_sha256": (input_data.get("manifest_sha256"), file_sha256(samples_path)),
        "input.source_token_count": (input_data.get("source_token_count"), source_token_count),
        "input.known_token_count": (input_data.get("known_token_count"), known_token_count),
        "input.unknown_token_count": (input_data.get("unknown_token_count"), unknown_token_count),
    }
    pixels = full_images.get("pixel_statistics", {})
    mask_pixels = full_masks.get("pixel_statistics", {})
    if not isinstance(pixels, dict):
        pixels = {}
    if not isinstance(mask_pixels, dict):
        mask_pixels = {}
    expected_pairs.update(
        {
            "quality.constant_image_count": (quality.get("constant_image_count"), pixels.get("constant_images")),
            "quality.all_zero_image_count": (quality.get("all_zero_image_count"), pixels.get("all_black_images")),
            "quality.unique_image_pixel_hashes": (quality.get("unique_image_pixel_hashes"), pixels.get("unique_pixel_hashes")),
            "quality.duplicate_image_groups": (quality.get("duplicate_image_groups"), pixels.get("collision_groups")),
            "quality.samples_in_duplicate_image_groups": (quality.get("samples_in_duplicate_image_groups"), pixels.get("samples_in_collision_groups")),
            "quality.duplicate_extra_images": (quality.get("duplicate_extra_images"), pixels.get("collision_extra_images")),
            "quality.largest_duplicate_image_group": (quality.get("largest_duplicate_image_group"), pixels.get("largest_collision_group")),
            "quality.cross_class_duplicate_groups": (quality.get("cross_class_duplicate_groups"), pixels.get("cross_class_collision_groups")),
            "quality.cross_split_duplicate_groups": (quality.get("cross_split_duplicate_groups"), pixels.get("cross_split_collision_groups")),
            "quality.all_unknown_bin_count": (quality.get("all_unknown_bin_count"), bin_statistics["all_unknown_bin_count"]),
            "quality.samples_with_all_unknown_bins": (quality.get("samples_with_all_unknown_bins"), bin_statistics["samples_with_all_unknown_bins"]),
            "quality.empty_bin_count": (quality.get("empty_bin_count"), bin_statistics["empty_bin_count"]),
            "quality.minimum_bin_tokens": (quality.get("minimum_bin_tokens"), bin_statistics["minimum_bin_tokens"]),
            "quality.maximum_bin_tokens": (quality.get("maximum_bin_tokens"), bin_statistics["maximum_bin_tokens"]),
        }
    )
    mismatches = [name for name, pair in expected_pairs.items() if pair[0] != pair[1]]
    float_pairs = {
        "quality.image_pixel_mean": (quality.get("image_pixel_mean"), pixels.get("pixel_mean")),
        "quality.image_pixel_std": (quality.get("image_pixel_std"), pixels.get("pixel_std")),
        "quality.image_zero_pixel_ratio": (quality.get("image_zero_pixel_ratio"), pixels.get("zero_pixel_ratio")),
        "quality.unknown_mask_pixel_mean": (quality.get("unknown_mask_pixel_mean"), mask_pixels.get("pixel_mean")),
        "quality.unknown_mask_zero_pixel_ratio": (quality.get("unknown_mask_zero_pixel_ratio"), mask_pixels.get("zero_pixel_ratio")),
        "quality.weighted_unknown_byte_ratio": (quality.get("weighted_unknown_byte_ratio"), bin_statistics["weighted_unknown_byte_ratio"]),
    }
    for name, pair in float_pairs.items():
        try:
            matches = bool(np.isclose(float(pair[0]), float(pair[1]), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError, OverflowError):
            matches = False
        if not matches:
            mismatches.append(name)
    result["verification_passed"] = normalize_bool(verification.get("passed")) is True
    result["mismatches"] = mismatches
    if mismatches:
        errors.append(f"data/reports/image_summary.json mismatches: {mismatches}")
    return result


def build_submission_check(
    project: Path,
    source_zip: Path,
    source_labels: Path,
) -> dict[str, object]:
    project = Path(project).resolve()
    source_zip = Path(source_zip).resolve()
    source_labels = Path(source_labels).resolve()
    data = project / "data"
    manifests = data / "manifests"
    features_dir = data / "features"
    reports = data / "reports"
    errors: list[str] = []
    source_files = {
        "subtrain.zip": source_check(source_zip, EXPECTED_SOURCE_SHA256["subtrain.zip"], errors),
        "subtrainLabels.csv": source_check(source_labels, EXPECTED_SOURCE_SHA256["subtrainLabels.csv"], errors),
    }
    manifest_path = manifests / "samples.csv"
    full_manifest_path = manifests / "image_records.csv"
    manifest = read_csv(manifest_path, "data/manifests/samples.csv", errors)
    full_manifest = read_csv(full_manifest_path, "data/manifests/image_records.csv", errors)
    features = read_csv(features_dir / "features.csv", "data/features/features.csv", errors)
    quality = read_csv(manifests / "quality.csv", "data/manifests/quality.csv", errors)
    dataset = read_csv(data / "dataset.csv", "data/dataset.csv", errors)
    full_summary = read_json(reports / "image_summary.json", "data/reports/image_summary.json", errors)
    sample_count = int(len(manifest)) if manifest is not None else 0
    class_labels: list[int] = []
    clean_split_counts: dict[str, int] = {}
    cross_split_hash_overlaps: dict[str, int | None] = {}
    duplicate_ids: dict[str, int | None] = {
        "samples.csv": None,
        "features.csv": None,
        "image_records.csv": None,
    }
    manifest_required = {
        "Id",
        "Class",
        "split",
        "status",
        "image_path",
        "image_file_sha256",
        "bytes_zip_path",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "unknown_byte_ratio",
        "raw_bytes_sha256",
        "normalized_bytes_sha256",
        "token_stream_sha256",
        "raw_asm_sha256",
    }
    manifest_valid = manifest is not None and require_columns(
        manifest,
        manifest_required,
        "data/manifests/samples.csv",
        errors,
    )
    if manifest is not None and "Id" in manifest.columns:
        duplicate_ids["samples.csv"] = int(manifest["Id"].duplicated().sum())
        add_count_error(errors, "samples.csv duplicate Id values", int(duplicate_ids["samples.csv"] or 0))
        add_count_error(errors, "samples.csv missing Id values", int(manifest["Id"].isna().sum()))
    if manifest_valid:
        if sample_count != EXPECTED_SAMPLE_COUNT:
            errors.append(f"Expected {EXPECTED_SAMPLE_COUNT} clean samples, found {sample_count}")
        numeric_classes = pd.to_numeric(manifest["Class"], errors="coerce")
        add_count_error(errors, "samples.csv invalid Class values", int(numeric_classes.isna().sum()))
        class_labels = sorted({int(value) for value in numeric_classes.dropna().tolist()})
        if set(class_labels) != EXPECTED_CLASSES:
            errors.append(f"Expected class labels 1 through 9, found {class_labels}")
        clean_split_counts = count_splits(manifest)
        if clean_split_counts != EXPECTED_SPLIT_COUNTS:
            errors.append(f"Unexpected clean split counts: {clean_split_counts}")
        add_count_error(errors, "samples.csv invalid split values", int((~manifest["split"].astype(str).isin(SPLITS)).sum()))
        add_count_error(errors, "samples.csv rows not marked included", int((manifest["status"].astype(str) != "included").sum()))
        for column in ("raw_bytes_sha256", "normalized_bytes_sha256", "token_stream_sha256", "raw_asm_sha256"):
            add_count_error(errors, f"samples.csv missing {column}", int(manifest[column].isna().sum()))
            overlaps = int((manifest.dropna(subset=[column]).groupby(column)["split"].nunique() > 1).sum())
            cross_split_hash_overlaps[column] = overlaps
            add_count_error(errors, f"{column} overlaps across splits", overlaps)
    feature_count = 0
    missing_feature_cells = 0
    nonnumeric_feature_cells = 0
    infinite_feature_cells = 0
    remaining_constant_features: list[str] = []
    id_sets_match = False
    feature_class_matches_manifest = False
    if features is not None and require_columns(features, {"Id", "Class"}, "data/features/features.csv", errors):
        duplicate_ids["features.csv"] = int(features["Id"].duplicated().sum())
        add_count_error(errors, "features.csv duplicate Id values", int(duplicate_ids["features.csv"] or 0))
        add_count_error(errors, "features.csv missing Id values", int(features["Id"].isna().sum()))
        feature_columns = [column for column in features.columns if column not in {"Id", "Class"}]
        feature_count = len(feature_columns)
        if len(features) != EXPECTED_SAMPLE_COUNT:
            errors.append(f"Expected {EXPECTED_SAMPLE_COUNT} feature rows, found {len(features)}")
        if feature_count != EXPECTED_FEATURE_COUNT:
            errors.append(f"Expected {EXPECTED_FEATURE_COUNT} features, found {feature_count}")
        numeric_features = features[feature_columns].apply(pd.to_numeric, errors="coerce")
        missing_feature_cells = int(features[feature_columns].isna().sum().sum())
        coerced_missing = int(numeric_features.isna().sum().sum())
        nonnumeric_feature_cells = coerced_missing - missing_feature_cells
        add_count_error(errors, "Static feature missing cells", missing_feature_cells)
        add_count_error(errors, "Static feature nonnumeric cells", nonnumeric_feature_cells)
        infinite_feature_cells = int(np.isinf(numeric_features.to_numpy(dtype=float)).sum())
        add_count_error(errors, "Static feature infinite cells", infinite_feature_cells)
        remaining_constant_features = [
            column for column in feature_columns if numeric_features[column].nunique(dropna=False) <= 1
        ]
        if remaining_constant_features:
            errors.append(f"Static features still contain constants: {remaining_constant_features}")
        if manifest is not None and "Id" in manifest.columns:
            id_sets_match = set(manifest["Id"].dropna().astype(str)) == set(features["Id"].dropna().astype(str))
            if not id_sets_match:
                errors.append("samples.csv and features.csv ID sets differ")
            elif not manifest["Id"].duplicated().any() and not features["Id"].duplicated().any() and "Class" in manifest.columns:
                clean_class = pd.to_numeric(manifest.set_index("Id")["Class"], errors="coerce").sort_index()
                feature_class = pd.to_numeric(features.set_index("Id")["Class"], errors="coerce").sort_index()
                feature_class_matches_manifest = bool(clean_class.equals(feature_class))
                if not feature_class_matches_manifest:
                    errors.append("samples.csv and features.csv labels differ")
    quality_level_counts = verify_quality(quality, manifest, features, errors)
    full_manifest_check: dict[str, object] = {
        "sample_count": int(len(full_manifest)) if full_manifest is not None else 0,
        "split_counts": {},
        "class_labels": [],
        "metadata_match": {},
    }
    full_required = {
        "Id",
        "Class",
        "split",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "unknown_byte_ratio",
        "bytes_zip_path",
        "raw_bytes_sha256",
        "normalized_bytes_sha256",
        "token_stream_sha256",
        "unknown_position_sha256",
        "full_image_path",
        "full_unknown_mask_path",
        "full_image_file_sha256",
        "full_image_pixel_sha256",
        "full_unknown_mask_file_sha256",
        "full_unknown_mask_pixel_sha256",
        "full_image_all_unknown_bin_count",
        "full_image_empty_bin_count",
        "full_image_min_bin_tokens",
        "full_image_max_bin_tokens",
    }
    full_valid = full_manifest is not None and require_columns(
        full_manifest,
        full_required,
        "data/manifests/image_records.csv",
        errors,
    )
    if full_manifest is not None and "Id" in full_manifest.columns:
        duplicate_ids["image_records.csv"] = int(full_manifest["Id"].duplicated().sum())
        add_count_error(errors, "image_records.csv duplicate Id values", int(duplicate_ids["image_records.csv"] or 0))
        add_count_error(errors, "image_records.csv missing Id values", int(full_manifest["Id"].isna().sum()))
    if full_valid:
        if len(full_manifest) != EXPECTED_SAMPLE_COUNT:
            errors.append(f"Expected {EXPECTED_SAMPLE_COUNT} full-image rows, found {len(full_manifest)}")
        full_counts = count_splits(full_manifest)
        full_manifest_check["split_counts"] = full_counts
        if full_counts != EXPECTED_SPLIT_COUNTS:
            errors.append(f"Unexpected full-image split counts: {full_counts}")
        full_classes_numeric = pd.to_numeric(full_manifest["Class"], errors="coerce")
        full_classes = sorted({int(value) for value in full_classes_numeric.dropna().tolist()})
        full_manifest_check["class_labels"] = full_classes
        if set(full_classes) != EXPECTED_CLASSES:
            errors.append(f"Unexpected full-image class labels: {full_classes}")
        if manifest is not None:
            full_manifest_check["metadata_match"] = compare_manifest_metadata(manifest, full_manifest, errors)
    image_metrics: dict[str, Callable[[np.ndarray], int | bool]] = {}
    mask_metrics: dict[str, Callable[[np.ndarray], int | bool]] = {}
    head_images = inspect_image_collection(
        data,
        manifest if manifest is not None else pd.DataFrame(),
        "images_head",
        "image_path",
        "image_file_sha256",
        None,
        {},
        "head-1024 images",
        errors,
    )
    full_images = inspect_image_collection(
        data,
        full_manifest if full_manifest is not None else pd.DataFrame(),
        "images",
        "full_image_path",
        "full_image_file_sha256",
        "full_image_pixel_sha256",
        image_metrics,
        "full-file mean images",
        errors,
    )
    full_unknown_masks = inspect_image_collection(
        data,
        full_manifest if full_manifest is not None else pd.DataFrame(),
        "masks",
        "full_unknown_mask_path",
        "full_unknown_mask_file_sha256",
        "full_unknown_mask_pixel_sha256",
        mask_metrics,
        "full-file unknown masks",
        errors,
    )
    dataset_check = verify_dataset(dataset, manifest, full_manifest, quality, errors)
    if manifest_path.is_file():
        summary_check = verify_image_summary(
            full_summary,
            manifest_path,
            full_manifest if full_manifest is not None else pd.DataFrame(),
            full_images,
            full_unknown_masks,
            errors,
        )
    else:
        summary_check = {
            "exists_and_readable": full_summary is not None,
            "verification_passed": False,
            "mismatches": ["input.manifest_sha256"],
        }
    source_checksums = {name: details["sha256"] for name, details in source_files.items()}
    source_checksums_match_expected = all(bool(details["matches_expected"]) for details in source_files.values())
    unknown_byte_ratio_weighted: float | None = None
    samples_unknown_ratio_over_90_percent = 0
    if manifest is not None and {"byte_token_count", "unknown_byte_count", "unknown_byte_ratio"}.issubset(manifest.columns):
        token_count = pd.to_numeric(manifest["byte_token_count"], errors="coerce")
        unknown_count = pd.to_numeric(manifest["unknown_byte_count"], errors="coerce")
        unknown_ratio = pd.to_numeric(manifest["unknown_byte_ratio"], errors="coerce")
        if float(token_count.sum()) > 0:
            unknown_byte_ratio_weighted = float(unknown_count.sum() / token_count.sum())
        samples_unknown_ratio_over_90_percent = int((unknown_ratio > 0.90).sum())
    samples_head_unknown_ratio_over_50_percent = 0
    if features is not None and "image_unknown_tokens" in features.columns:
        head_unknown = pd.to_numeric(features["image_unknown_tokens"], errors="coerce") / 1024.0
        samples_head_unknown_ratio_over_50_percent = int((head_unknown > 0.50).sum())
    return {
        "schema_version": 2,
        "sample_count": sample_count,
        "feature_count_excluding_id_and_label": int(feature_count),
        "id_sets_match": bool(id_sets_match),
        "duplicate_ids": duplicate_ids,
        "feature_class_matches_manifest": bool(feature_class_matches_manifest),
        "class_labels": class_labels,
        "split_counts": clean_split_counts,
        "dataset": dataset_check,
        "quality_counts": quality_level_counts,
        "unknown_byte_ratio_weighted": unknown_byte_ratio_weighted,
        "samples_unknown_ratio_over_90_percent": samples_unknown_ratio_over_90_percent,
        "samples_head_unknown_ratio_over_50_percent": samples_head_unknown_ratio_over_50_percent,
        "missing_feature_cells": int(missing_feature_cells),
        "nonnumeric_feature_cells": int(nonnumeric_feature_cells),
        "infinite_feature_cells": int(infinite_feature_cells),
        "remaining_constant_features": remaining_constant_features,
        "cross_split_hash_overlaps": cross_split_hash_overlaps,
        "image_records": full_manifest_check,
        "head_images": head_images,
        "images": full_images,
        "masks": full_unknown_masks,
        "image_summary": summary_check,
        "source_checksums": source_checksums,
        "source_checksums_match_expected": bool(source_checksums_match_expected),
        "errors": errors,
        "passed": not errors,
    }


def write_submission_check(project: Path, report: dict[str, object]) -> Path:
    destination = Path(project).resolve() / "data" / "reports" / "check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify BIG 2015 preprocessing outputs")
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=project.parent / "kaggle2015-sample" / "subtrain.zip",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=project.parent / "kaggle2015-sample" / "subtrainLabels.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_submission_check(args.project, args.source_zip, args.source_labels)
    destination = write_submission_check(args.project, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {destination}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
