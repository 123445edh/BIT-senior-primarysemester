from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SIDE = 32
IMAGE_TOKENS = IMAGE_SIDE * IMAGE_SIDE
EXPECTED_ZIP_SHA256 = "f22edaac223b5af79d30d811b26d596f70cdc7526f79d13c5e851a5830674b13"
ADDRESS_PREFIX_RE = re.compile(rb"(?m)^[0-9a-fA-F]{8}[ \t]+")
WHITESPACE_RE = re.compile(rb"\s+")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate full-file 32x32 BIG 2015 images")
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=project_root.parent / "kaggle2015-sample" / "subtrain.zip",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "samples.csv",
    )
    parser.add_argument("--output-root", type=Path, default=project_root / "data")
    parser.add_argument("--compression-level", type=int, default=9, choices=range(10))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_manifest(path: Path) -> pd.DataFrame:
    required = {
        "Id",
        "Class",
        "split",
        "status",
        "bytes_zip_path",
        "byte_token_count",
        "known_byte_count",
        "unknown_byte_count",
        "unknown_byte_ratio",
        "raw_bytes_sha256",
        "normalized_bytes_sha256",
        "token_stream_sha256",
    }
    frame = pd.read_csv(path, dtype={"Id": "string", "split": "string", "status": "string"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError("Manifest is empty")
    if frame["Id"].isna().any() or frame["Id"].duplicated().any():
        raise ValueError("Manifest Id values must be non-null and unique")
    if not frame["Id"].map(lambda value: bool(SAFE_ID_RE.fullmatch(str(value)))).all():
        raise ValueError("Manifest contains an unsafe sample Id")
    if not (frame["status"] == "included").all():
        raise ValueError("Manifest contains a sample not marked included")
    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Manifest contains an invalid split")
    frame["Class"] = pd.to_numeric(frame["Class"], errors="raise").astype(int)
    if not frame["Class"].between(1, 9).all():
        raise ValueError("Manifest Class must be between 1 and 9")
    for column in ("byte_token_count", "known_byte_count", "unknown_byte_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    if (frame["byte_token_count"] <= 0).any():
        raise ValueError("Manifest contains a non-positive byte token count")
    if not (
        frame["known_byte_count"] + frame["unknown_byte_count"] == frame["byte_token_count"]
    ).all():
        raise ValueError("Manifest known and unknown counts do not match token counts")
    expected_ratio = frame["unknown_byte_count"] / frame["byte_token_count"]
    actual_ratio = pd.to_numeric(frame["unknown_byte_ratio"], errors="raise")
    if not np.allclose(expected_ratio, actual_ratio, rtol=0.0, atol=1e-12):
        raise ValueError("Manifest unknown byte ratios are inconsistent")
    return frame.sort_values("Id").reset_index(drop=True)


def collect_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    duplicates: list[str] = []
    for info in archive.infolist():
        if info.is_dir() or Path(info.filename).suffix.lower() != ".bytes":
            continue
        sample_id = Path(info.filename).stem
        if sample_id in entries:
            duplicates.append(sample_id)
        else:
            entries[sample_id] = info
    if duplicates:
        raise ValueError(f"Duplicate .bytes ZIP entries: {duplicates[:5]}")
    return entries


def parse_byte_entry(
    raw: bytes,
    expected_count: int,
    expected_unknown: int,
    expected_raw_hash: str,
    expected_normalized_hash: str,
    expected_token_hash: str,
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != expected_raw_hash:
        raise ValueError(f"Raw source hash mismatch for {sample_id}")
    payload = ADDRESS_PREFIX_RE.sub(b"", raw)
    token_stream = WHITESPACE_RE.sub(b"", payload).upper()
    if len(token_stream) % 2:
        raise ValueError(f"Odd token stream length for {sample_id}")
    token_pairs = np.frombuffer(token_stream, dtype=np.uint8).reshape(-1, 2)
    unknown = np.all(token_pairs == ord("?"), axis=1)
    if len(unknown) != expected_count:
        raise ValueError(f"Token count mismatch for {sample_id}: {len(unknown)} != {expected_count}")
    if int(unknown.sum()) != expected_unknown:
        raise ValueError(f"Unknown token count mismatch for {sample_id}")
    token_hash = hashlib.sha256(token_stream).hexdigest()
    if token_hash != expected_token_hash:
        raise ValueError(f"Token stream hash mismatch for {sample_id}")
    try:
        normalized = bytes.fromhex(token_stream.replace(b"??", b"00").decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid byte token for {sample_id}") from exc
    values = np.frombuffer(normalized, dtype=np.uint8).copy()
    if hashlib.sha256(values.tobytes()).hexdigest() != expected_normalized_hash:
        raise ValueError(f"Normalized source hash mismatch for {sample_id}")
    return values, unknown, array_sha256(unknown.astype(np.uint8))


def aggregate(values: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    count = len(values)
    edges = np.floor_divide(np.arange(IMAGE_TOKENS + 1, dtype=np.int64) * count, IMAGE_TOKENS)
    lengths = np.diff(edges)
    known = np.logical_not(unknown)
    cumulative_known = np.empty(count + 1, dtype=np.int64)
    cumulative_sum = np.empty(count + 1, dtype=np.int64)
    cumulative_known[0] = 0
    cumulative_sum[0] = 0
    np.cumsum(known, dtype=np.int64, out=cumulative_known[1:])
    np.cumsum(values.astype(np.int64) * known, dtype=np.int64, out=cumulative_sum[1:])
    known_counts = cumulative_known[edges[1:]] - cumulative_known[edges[:-1]]
    known_sums = cumulative_sum[edges[1:]] - cumulative_sum[edges[:-1]]
    unknown_counts = lengths - known_counts
    pixels = np.zeros(IMAGE_TOKENS, dtype=np.uint8)
    usable = known_counts > 0
    pixels[usable] = np.floor_divide(
        known_sums[usable] + np.floor_divide(known_counts[usable], 2),
        known_counts[usable],
    ).astype(np.uint8)
    mask = np.zeros(IMAGE_TOKENS, dtype=np.uint8)
    nonempty = lengths > 0
    mask[nonempty] = np.floor_divide(
        (255 * unknown_counts[nonempty]) + np.floor_divide(lengths[nonempty], 2),
        lengths[nonempty],
    ).astype(np.uint8)
    mask[np.logical_not(nonempty)] = 255
    if int(lengths.sum()) != count:
        raise AssertionError("Aggregation did not cover the full token sequence")
    if int(known_counts.sum()) != int(known.sum()):
        raise AssertionError("Aggregation known count mismatch")
    if int(unknown_counts.sum()) != int(unknown.sum()):
        raise AssertionError("Aggregation unknown count mismatch")
    stats = {
        "known_bin_count": int(usable.sum()),
        "all_unknown_bin_count": int(np.logical_and(nonempty, known_counts == 0).sum()),
        "empty_bin_count": int(np.logical_not(nonempty).sum()),
        "min_bin_tokens": int(lengths.min()),
        "max_bin_tokens": int(lengths.max()),
    }
    return pixels.reshape(IMAGE_SIDE, IMAGE_SIDE), mask.reshape(IMAGE_SIDE, IMAGE_SIDE), stats


def save_png(array: np.ndarray, path: Path, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array, mode="L").save(
        temporary,
        format="PNG",
        compress_level=compression_level,
        optimize=False,
    )
    os.replace(temporary, path)


def inspect_png(path: Path) -> tuple[np.ndarray, str, str]:
    with Image.open(path) as image_object:
        image_object.load()
        if image_object.size != (IMAGE_SIDE, IMAGE_SIDE) or image_object.mode != "L":
            raise ValueError(f"Invalid PNG format: {path}")
        array = np.asarray(image_object, dtype=np.uint8).copy()
    return array, file_sha256(path), array_sha256(array)


def verify_staging(frame: pd.DataFrame, staging_data: Path) -> dict[str, object]:
    errors: list[str] = []
    verified = 0
    for row in frame.itertuples(index=False):
        image_path = staging_data / row.full_image_path
        mask_path = staging_data / row.full_unknown_mask_path
        try:
            image, image_file_hash, image_pixel_hash = inspect_png(image_path)
            mask, mask_file_hash, mask_pixel_hash = inspect_png(mask_path)
            if image_file_hash != row.full_image_file_sha256:
                raise ValueError("image file hash mismatch")
            if image_pixel_hash != row.full_image_pixel_sha256:
                raise ValueError("image pixel hash mismatch")
            if mask_file_hash != row.full_unknown_mask_file_sha256:
                raise ValueError("mask file hash mismatch")
            if mask_pixel_hash != row.full_unknown_mask_pixel_sha256:
                raise ValueError("mask pixel hash mismatch")
            if int(image.min()) != row.full_image_min or int(image.max()) != row.full_image_max:
                raise ValueError("image range mismatch")
            if int(mask.min()) != row.full_unknown_mask_min or int(mask.max()) != row.full_unknown_mask_max:
                raise ValueError("mask range mismatch")
            verified += 1
        except Exception as exc:
            errors.append(f"{row.Id}: {exc}")
    image_root = staging_data / "images"
    mask_root = staging_data / "masks"
    actual_images = {
        path.relative_to(staging_data).as_posix() for path in image_root.rglob("*.png")
    } if image_root.is_dir() else set()
    actual_masks = {
        path.relative_to(staging_data).as_posix() for path in mask_root.rglob("*.png")
    } if mask_root.is_dir() else set()
    expected_images = set(frame["full_image_path"])
    expected_masks = set(frame["full_unknown_mask_path"])
    if actual_images != expected_images:
        errors.append(
            f"Image file set mismatch: missing={len(expected_images - actual_images)}, extra={len(actual_images - expected_images)}"
        )
    if actual_masks != expected_masks:
        errors.append(
            f"Mask file set mismatch: missing={len(expected_masks - actual_masks)}, extra={len(actual_masks - expected_masks)}"
        )
    return {
        "passed": not errors and verified == len(frame),
        "verified_samples": int(verified),
        "expected_samples": int(len(frame)),
        "image_count": int(len(actual_images)),
        "mask_count": int(len(actual_masks)),
        "errors": errors,
    }


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def promote(targets: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for source, destination in targets:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup_{token}")
                os.replace(destination, backup)
                backups.append((destination, backup))
            os.replace(source, destination)
            promoted.append(destination)
    except BaseException:
        for destination in reversed(promoted):
            remove_path(destination)
        for destination, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    for _, backup in backups:
        remove_path(backup)


def package_versions() -> dict[str, str]:
    import PIL

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
    }


def quality_statistics(frame: pd.DataFrame) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        groups[row.full_image_pixel_sha256].append(
            {"Id": row.Id, "Class": int(row.Class), "split": row.split}
        )
    duplicates = [group for group in groups.values() if len(group) > 1]
    details = [
        {
            "pixel_sha256": digest,
            "count": len(group),
            "samples": group,
        }
        for digest, group in sorted(groups.items())
        if len(group) > 1
    ]
    image_pixels = len(frame) * IMAGE_TOKENS
    image_sum = float(frame["full_image_pixel_sum"].sum())
    image_square_sum = float(frame["full_image_pixel_square_sum"].sum())
    image_mean = image_sum / image_pixels
    image_variance = max(0.0, (image_square_sum / image_pixels) - (image_mean * image_mean))
    mask_sum = float(frame["full_unknown_mask_pixel_sum"].sum())
    return {
        "constant_image_count": int(frame["full_image_is_constant"].sum()),
        "all_zero_image_count": int(frame["full_image_is_all_zero"].sum()),
        "unique_image_pixel_hashes": int(len(groups)),
        "duplicate_image_groups": int(len(duplicates)),
        "samples_in_duplicate_image_groups": int(sum(len(group) for group in duplicates)),
        "duplicate_extra_images": int(sum(len(group) - 1 for group in duplicates)),
        "largest_duplicate_image_group": int(max((len(group) for group in duplicates), default=1)),
        "cross_class_duplicate_groups": int(
            sum(len({sample["Class"] for sample in group}) > 1 for group in duplicates)
        ),
        "cross_split_duplicate_groups": int(
            sum(len({sample["split"] for sample in group}) > 1 for group in duplicates)
        ),
        "image_pixel_mean": image_mean,
        "image_pixel_std": float(np.sqrt(image_variance)),
        "image_zero_pixel_ratio": float(frame["full_image_zero_pixel_count"].sum() / image_pixels),
        "unknown_mask_pixel_mean": float(mask_sum / image_pixels),
        "unknown_mask_zero_pixel_ratio": float(
            frame["full_unknown_mask_zero_pixel_count"].sum() / image_pixels
        ),
        "all_unknown_bin_count": int(frame["full_image_all_unknown_bin_count"].sum()),
        "samples_with_all_unknown_bins": int((frame["full_image_all_unknown_bin_count"] > 0).sum()),
        "empty_bin_count": int(frame["full_image_empty_bin_count"].sum()),
        "minimum_bin_tokens": int(frame["full_image_min_bin_tokens"].min()),
        "maximum_bin_tokens": int(frame["full_image_max_bin_tokens"].max()),
        "weighted_unknown_byte_ratio": float(
            frame["unknown_byte_count"].sum() / frame["byte_token_count"].sum()
        ),
        "duplicate_image_details": details,
    }


def main() -> None:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if not input_zip.is_file() or not zipfile.is_zipfile(input_zip):
        raise FileNotFoundError(f"Valid ZIP not found: {input_zip}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if output_root == input_zip.parent or input_zip.parent in output_root.parents:
        raise ValueError("Output root must not be inside the raw-data directory")
    source_manifest_hash = file_sha256(manifest_path)
    print("Checking source ZIP SHA-256...", flush=True)
    source_zip_hash = file_sha256(input_zip)
    if source_zip_hash != EXPECTED_ZIP_SHA256:
        raise ValueError(f"Unexpected source ZIP SHA-256: {source_zip_hash}")
    source = load_manifest(manifest_path)
    staging_root = project_root / f".full_image_staging_{uuid.uuid4().hex}"
    staging_data = staging_root / "data"
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(input_zip, "r") as archive:
            entries = collect_entries(archive)
            missing = sorted(set(source["Id"].astype(str)) - set(entries))
            if missing:
                raise ValueError(f"Missing .bytes entries: {missing[:5]}")
            print(f"Generating {len(source)} full-file image and mask pairs...", flush=True)
            for position, row in enumerate(source.itertuples(index=False), start=1):
                info = entries[str(row.Id)]
                if str(row.bytes_zip_path) != info.filename:
                    raise ValueError(f"ZIP path mismatch for {row.Id}")
                with archive.open(info, "r") as stream:
                    raw = stream.read()
                values, unknown, unknown_position_hash = parse_byte_entry(
                    raw,
                    int(row.byte_token_count),
                    int(row.unknown_byte_count),
                    str(row.raw_bytes_sha256),
                    str(row.normalized_bytes_sha256),
                    str(row.token_stream_sha256),
                    str(row.Id),
                )
                image, mask, aggregation = aggregate(values, unknown)
                image_relative = (
                    Path("images")
                    / str(row.split)
                    / f"class_{int(row.Class)}"
                    / f"{row.Id}.png"
                )
                mask_relative = (
                    Path("masks")
                    / str(row.split)
                    / f"class_{int(row.Class)}"
                    / f"{row.Id}.png"
                )
                image_path = staging_data / image_relative
                mask_path = staging_data / mask_relative
                save_png(image, image_path, args.compression_level)
                save_png(mask, mask_path, args.compression_level)
                saved_image, image_file_hash, image_pixel_hash = inspect_png(image_path)
                saved_mask, mask_file_hash, mask_pixel_hash = inspect_png(mask_path)
                if not np.array_equal(image, saved_image) or not np.array_equal(mask, saved_mask):
                    raise ValueError(f"PNG pixel round-trip mismatch for {row.Id}")
                rows.append(
                    {
                        "Id": str(row.Id),
                        "Class": int(row.Class),
                        "split": str(row.split),
                        "byte_token_count": int(row.byte_token_count),
                        "known_byte_count": int(row.known_byte_count),
                        "unknown_byte_count": int(row.unknown_byte_count),
                        "unknown_byte_ratio": float(row.unknown_byte_ratio),
                        "bytes_zip_path": str(row.bytes_zip_path),
                        "raw_bytes_sha256": str(row.raw_bytes_sha256),
                        "normalized_bytes_sha256": str(row.normalized_bytes_sha256),
                        "token_stream_sha256": str(row.token_stream_sha256),
                        "unknown_position_sha256": unknown_position_hash,
                        "full_image_path": image_relative.as_posix(),
                        "full_unknown_mask_path": mask_relative.as_posix(),
                        "full_image_file_sha256": image_file_hash,
                        "full_image_pixel_sha256": image_pixel_hash,
                        "full_unknown_mask_file_sha256": mask_file_hash,
                        "full_unknown_mask_pixel_sha256": mask_pixel_hash,
                        "full_image_known_bin_count": aggregation["known_bin_count"],
                        "full_image_all_unknown_bin_count": aggregation["all_unknown_bin_count"],
                        "full_image_empty_bin_count": aggregation["empty_bin_count"],
                        "full_image_min_bin_tokens": aggregation["min_bin_tokens"],
                        "full_image_max_bin_tokens": aggregation["max_bin_tokens"],
                        "full_image_min": int(image.min()),
                        "full_image_max": int(image.max()),
                        "full_image_pixel_sum": int(image.astype(np.int64).sum()),
                        "full_image_pixel_square_sum": int(np.square(image.astype(np.int64)).sum()),
                        "full_image_zero_pixel_count": int((image == 0).sum()),
                        "full_image_is_constant": bool(np.ptp(image) == 0),
                        "full_image_is_all_zero": bool(np.all(image == 0)),
                        "full_unknown_mask_min": int(mask.min()),
                        "full_unknown_mask_max": int(mask.max()),
                        "full_unknown_mask_pixel_sum": int(mask.astype(np.int64).sum()),
                        "full_unknown_mask_zero_pixel_count": int((mask == 0).sum()),
                    }
                )
                if position % 25 == 0 or position == len(source):
                    print(f"  {position}/{len(source)}", flush=True)
        result = pd.DataFrame(rows)
        if list(result["Id"]) != list(source["Id"].astype(str)):
            raise ValueError("Generated sample order does not match the clean manifest")
        record_columns = [
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
        ]
        manifest_output = staging_data / "manifests" / "image_records.csv"
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        result[record_columns].to_csv(manifest_output, index=False, encoding="utf-8-sig")
        verification = verify_staging(result, staging_data)
        if not verification["passed"]:
            raise RuntimeError(f"Staging verification failed: {verification['errors'][:5]}")
        quality = quality_statistics(result)
        elapsed = time.perf_counter() - started
        summary = {
            "dataset": "Microsoft BIG 2015 small sample (local derivative)",
            "method": "full-file equal-position-bin known-byte mean",
            "method_version": 1,
            "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "zip_path": input_zip.name,
                "zip_sha256": source_zip_hash,
                "expected_zip_sha256": EXPECTED_ZIP_SHA256,
                "manifest_path": "data/manifests/samples.csv",
                "manifest_sha256": source_manifest_hash,
                "manifest_samples": int(len(source)),
                "zip_byte_entries": int(len(entries)),
                "source_token_count": int(result["byte_token_count"].sum()),
                "known_token_count": int(result["known_byte_count"].sum()),
                "unknown_token_count": int(result["unknown_byte_count"].sum()),
            },
            "output": {
                "image_directory": "images",
                "mask_directory": "masks",
                "manifest": "manifests/image_records.csv",
                "image_size": [IMAGE_SIDE, IMAGE_SIDE],
                "image_mode": "L",
                "samples": int(len(result)),
                "split_counts": {
                    key: int(value) for key, value in result["split"].value_counts().to_dict().items()
                },
                "class_counts": {
                    str(int(key)): int(value)
                    for key, value in result["Class"].value_counts().sort_index().items()
                },
            },
            "algorithm": {
                "partition": "bin_j=[floor(j*N/1024), floor((j+1)*N/1024))",
                "intensity": "round_half_up(mean(known byte values)); 0 when a bin has no known byte",
                "unknown_mask": "round_half_up(255*unknown_count/bin_length); 255 for an empty bin",
                "layout": "row-major 32x32",
                "coverage": "all observed byte and ?? positions exactly once",
                "scope": "full-file coverage with lossy aggregation; not a lossless representation",
            },
            "quality": quality,
            "verification": verification,
            "environment": package_versions(),
            "elapsed_seconds": elapsed,
        }
        report_output = staging_data / "reports" / "image_summary.json"
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        promote(
            [
                (
                    staging_data / "images",
                    output_root / "images",
                ),
                (
                    staging_data / "masks",
                    output_root / "masks",
                ),
                (
                    manifest_output,
                    output_root / "manifests" / "image_records.csv",
                ),
                (
                    report_output,
                    output_root / "reports" / "image_summary.json",
                ),
            ]
        )
        print(f"Completed and verified in {elapsed / 60:.1f} minutes.", flush=True)
        print(f"Output: {output_root}", flush=True)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
