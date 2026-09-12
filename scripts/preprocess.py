from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import zipfile
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SIDE = 32
IMAGE_TOKENS = IMAGE_SIDE * IMAGE_SIDE
DEFAULT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASS_NAMES = {
    1: "Ramnit",
    2: "Lollipop",
    3: "Kelihos_ver3",
    4: "Vundo",
    5: "Simda",
    6: "Tracur",
    7: "Kelihos_ver1",
    8: "Obfuscator.ACY",
    9: "Gatak",
}

OPCODES = (
    "mov", "push", "pop", "call", "ret", "jmp", "je", "jne", "jz", "jnz",
    "ja", "jb", "jg", "jl", "cmp", "test", "lea", "add", "sub", "inc",
    "dec", "xor", "and", "or", "not", "shl", "shr", "rol", "ror", "mul",
    "imul", "div", "idiv", "nop", "int", "db", "dw", "dd",
)
SECTIONS = ("header", ".text", ".data", ".rdata", ".bss", ".idata", ".edata", ".rsrc", ".tls")
ASM_OPCODE_RE = re.compile(
    rb"(?m)^[^\r\n:]+:[0-9a-f]{8}[ \t]+(?:(?:[0-9a-f?]{2})[ \t]+)+([a-z][a-z0-9]*)\b"
)
ADDRESS_PREFIX_RE = re.compile(rb"(?m)^[0-9a-fA-F]{8}[ \t]+")
WHITESPACE_RE = re.compile(rb"\s+")
BYTE_TOKEN_RE = re.compile(rb"(?i)(?:[0-9a-f]{2}|\?\?)")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser(description="Preprocess BIG 2015 small sample")
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=workspace_root / "kaggle2015-sample" / "subtrain.zip",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=workspace_root / "kaggle2015-sample" / "subtrainLabels.csv",
    )
    parser.add_argument("--output", type=Path, default=project_root / "data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-asm",
        action="store_true",
        help="Skip ASM content parsing; ZIP size metadata is still recorded.",
    )
    return parser.parse_args()


def check_inputs(input_zip: Path, labels_path: Path, output: Path) -> None:
    if not input_zip.is_file():
        raise FileNotFoundError(f"ZIP not found: {input_zip}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    if output.resolve() == input_zip.parent.resolve():
        raise ValueError("Output must not be the raw-data directory")
    if not zipfile.is_zipfile(input_zip):
        raise ValueError(f"Invalid ZIP archive: {input_zip}")


def clean_labels(labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    labels = pd.read_csv(labels_path, dtype={"Id": "string", "Class": "Int64"})
    required = {"Id", "Class"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels must contain columns {sorted(required)}")

    labels = labels[["Id", "Class"]].copy()
    labels["Id"] = labels["Id"].str.strip()
    if labels["Id"].isna().any() or (labels["Id"] == "").any() or labels["Class"].isna().any():
        raise ValueError("Labels contain missing/blank Id or Class")
    labels["Class"] = labels["Class"].astype(int)

    class_counts_per_id = labels.groupby("Id")["Class"].nunique()
    conflicts = class_counts_per_id[class_counts_per_id > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting labels found for {len(conflicts)} sample IDs")

    duplicate_mask = labels.duplicated(subset=["Id"], keep=False)
    duplicate_rows = labels.loc[duplicate_mask].sort_values(["Id", "Class"]).reset_index(drop=True)
    cleaned = labels.drop_duplicates(subset=["Id"], keep="first").sort_values("Id").reset_index(drop=True)
    stats = {
        "raw_label_rows": int(len(labels)),
        "clean_label_rows": int(len(cleaned)),
        "duplicate_extra_rows_removed": int(len(labels) - len(cleaned)),
        "duplicate_id_groups": int(labels.loc[duplicate_mask, "Id"].nunique()),
        "conflicting_label_ids": int(len(conflicts)),
    }
    return cleaned, duplicate_rows, stats


def collect_zip_entries(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, zipfile.ZipInfo], dict[str, zipfile.ZipInfo], list[str]]:
    bytes_entries: dict[str, zipfile.ZipInfo] = {}
    asm_entries: dict[str, zipfile.ZipInfo] = {}
    duplicate_entry_names: list[str] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        suffix = Path(info.filename).suffix.lower()
        if suffix not in {".bytes", ".asm"}:
            continue
        sample_id = Path(info.filename).stem
        target = bytes_entries if suffix == ".bytes" else asm_entries
        if sample_id in target:
            duplicate_entry_names.append(info.filename)
        else:
            target[sample_id] = info
    return bytes_entries, asm_entries, duplicate_entry_names


def _is_hex_byte(token: bytes) -> bool:
    if len(token) != 2:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True


def analyze_bytes(stream: BinaryIO, info: zipfile.ZipInfo) -> tuple[dict[str, object], np.ndarray]:
    raw = stream.read()
    raw_hash = hashlib.sha256(raw)
    payload = ADDRESS_PREFIX_RE.sub(b"", raw)
    unknown_tokens = payload.count(b"??")
    invalid_tokens = 0
    try:
        sequence = bytes.fromhex(payload.replace(b"??", b"00").decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        values = bytearray()
        unknown_tokens = 0
        invalid_tokens = 0
        for token in payload.split():
            if token == b"??":
                values.append(0)
                unknown_tokens += 1
            elif _is_hex_byte(token):
                values.append(int(token, 16))
            else:
                invalid_tokens += 1
        sequence = bytes(values)

    total_tokens = len(sequence)
    array = np.frombuffer(sequence, dtype=np.uint8)
    histogram = np.bincount(array, minlength=256).astype(np.int64)
    histogram[0] -= unknown_tokens
    if histogram[0] < 0:
        raise ValueError(f"Invalid unknown-byte accounting in {info.filename}")

    image_values = np.zeros(IMAGE_TOKENS, dtype=np.uint8)
    used = min(total_tokens, IMAGE_TOKENS)
    image_values[:used] = array[:used]
    image_unknown_tokens = sum(
        match.group(0) == b"??" for match in islice(BYTE_TOKEN_RE.finditer(payload), IMAGE_TOKENS)
    )
    token_stream = WHITESPACE_RE.sub(b"", payload).upper()
    content_hash = hashlib.sha256(sequence)
    token_stream_hash = hashlib.sha256(token_stream)

    known_tokens = int(histogram.sum())
    if known_tokens:
        frequencies = histogram.astype(np.float64) / known_tokens
        nonzero = frequencies[frequencies > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        values = np.arange(256, dtype=np.float64)
        mean = float((frequencies * values).sum())
        variance = float((frequencies * (values - mean) ** 2).sum())
        std = math.sqrt(max(variance, 0.0))
        printable_ratio = float(histogram[32:127].sum() / known_tokens)
        zero_ratio = float(histogram[0] / known_tokens)
        high_byte_ratio = float(histogram[128:].sum() / known_tokens)
    else:
        frequencies = np.zeros(256, dtype=np.float64)
        entropy = mean = std = printable_ratio = zero_ratio = high_byte_ratio = 0.0

    features: dict[str, object] = {
        "bytes_zip_path": info.filename,
        "bytes_text_size": int(info.file_size),
        "bytes_compressed_size": int(info.compress_size),
        "byte_token_count": int(total_tokens),
        "known_byte_count": int(known_tokens),
        "unknown_byte_count": int(unknown_tokens),
        "unknown_byte_ratio": float(unknown_tokens / total_tokens) if total_tokens else 0.0,
        "invalid_byte_token_count": int(invalid_tokens),
        "byte_entropy": entropy,
        "byte_mean": mean,
        "byte_std": std,
        "byte_zero_ratio": zero_ratio,
        "byte_printable_ratio": printable_ratio,
        "byte_high_ratio": high_byte_ratio,
        "image_source_tokens": int(min(total_tokens, IMAGE_TOKENS)),
        "image_padding_tokens": int(max(0, IMAGE_TOKENS - total_tokens)),
        "image_truncated_tokens": int(max(0, total_tokens - IMAGE_TOKENS)),
        "image_unknown_tokens": int(image_unknown_tokens),
        "raw_bytes_sha256": raw_hash.hexdigest(),
        "normalized_bytes_sha256": content_hash.hexdigest(),
        "token_stream_sha256": token_stream_hash.hexdigest(),
    }
    features.update({f"byte_freq_{i:02x}": float(frequencies[i]) for i in range(256)})
    return features, image_values.reshape(IMAGE_SIDE, IMAGE_SIDE)


def analyze_asm(stream: BinaryIO, info: zipfile.ZipInfo) -> dict[str, object]:
    raw = stream.read()
    lower = raw.lower()
    raw_hash = hashlib.sha256(raw)
    line_count = raw.count(b"\n") + int(bool(raw) and not raw.endswith(b"\n"))
    opcode_counts: Counter[str] = Counter(
        match.group(1).decode("ascii") for match in ASM_OPCODE_RE.finditer(lower)
    )
    instruction_count = int(sum(opcode_counts.values()))
    section_counts: Counter[str] = Counter()
    for section in SECTIONS:
        marker = section.encode("ascii") + b":"
        section_counts[section] = lower.count(b"\n" + marker) + int(lower.startswith(marker))
    proc_count = len(re.findall(rb"\bproc\b", lower))
    extrn_count = len(re.findall(rb"\bextrn\b", lower))

    features: dict[str, object] = {
        "asm_zip_path": info.filename,
        "asm_text_size": int(info.file_size),
        "asm_compressed_size": int(info.compress_size),
        "asm_line_count": int(line_count),
        "asm_instruction_count": int(instruction_count),
        "asm_proc_count": int(proc_count),
        "asm_extrn_count": int(extrn_count),
        "raw_asm_sha256": raw_hash.hexdigest(),
    }
    for opcode in OPCODES:
        count = int(opcode_counts[opcode])
        features[f"opcode_count_{opcode}"] = count
        features[f"opcode_freq_{opcode}"] = float(count / instruction_count) if instruction_count else 0.0
    for section in SECTIONS:
        safe_name = section.lstrip(".")
        features[f"section_lines_{safe_name}"] = int(section_counts[section])
    return features


def asm_metadata_only(info: zipfile.ZipInfo) -> dict[str, object]:
    features: dict[str, object] = {
        "asm_zip_path": info.filename,
        "asm_text_size": int(info.file_size),
        "asm_compressed_size": int(info.compress_size),
        "asm_line_count": pd.NA,
        "asm_instruction_count": pd.NA,
        "asm_proc_count": pd.NA,
        "asm_extrn_count": pd.NA,
        "raw_asm_sha256": pd.NA,
    }
    for opcode in OPCODES:
        features[f"opcode_count_{opcode}"] = pd.NA
        features[f"opcode_freq_{opcode}"] = pd.NA
    for section in SECTIONS:
        features[f"section_lines_{section.lstrip('.')}"] = pd.NA
    return features


def assign_content_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["status"] = "included"
    frame["duplicate_of"] = ""
    frame["content_duplicate_group_size"] = 1
    frame["quality_note"] = ""

    empty_mask = frame["known_byte_count"] == 0
    frame.loc[empty_mask, "status"] = "excluded_empty_bytes"
    frame.loc[empty_mask, "quality_note"] = "No known hexadecimal byte tokens"

    valid = frame.loc[~empty_mask]
    group_columns = ["token_stream_sha256", "byte_token_count"]
    for _, group in valid.groupby(group_columns, sort=False):
        if len(group) <= 1:
            continue
        indices = list(group.index)
        labels = set(group["Class"].astype(int))
        frame.loc[indices, "content_duplicate_group_size"] = len(indices)
        if len(labels) > 1:
            frame.loc[indices, "status"] = "excluded_conflicting_content_labels"
            frame.loc[indices, "quality_note"] = "Identical normalized bytes have conflicting labels"
            continue
        canonical_index = min(indices, key=lambda idx: str(frame.at[idx, "Id"]))
        canonical_id = str(frame.at[canonical_index, "Id"])
        for idx in indices:
            if idx == canonical_index:
                continue
            frame.at[idx, "status"] = "excluded_content_duplicate"
            frame.at[idx, "duplicate_of"] = canonical_id
            frame.at[idx, "quality_note"] = "Exact normalized byte duplicate"
    return frame


def stratified_split(frame: pd.DataFrame, seed: int) -> pd.Series:
    included = frame.index[frame["status"] == "included"].to_numpy()
    labels = frame.loc[included, "Class"].to_numpy()
    temp_ratio = SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]
    train_idx, temp_idx = train_test_split(
        included,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = frame.loc[temp_idx, "Class"].to_numpy()
    test_fraction_of_temp = SPLIT_RATIOS["test"] / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction_of_temp,
        stratify=temp_labels,
        random_state=seed,
    )
    split = pd.Series("excluded", index=frame.index, dtype="string")
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_images(frame: pd.DataFrame, images: dict[str, np.ndarray], output: Path) -> None:
    image_root = output / "images_head"
    for row in frame.loc[frame["status"] == "included"].itertuples():
        relative = Path("images_head") / str(row.split) / f"class_{row.Class}" / f"{row.Id}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[str(row.Id)], mode="L").save(destination, format="PNG", compress_level=9)
        frame.at[row.Index, "image_path"] = relative.as_posix()
        frame.at[row.Index, "image_file_sha256"] = file_sha256(destination)


def check_head_outputs(frame: pd.DataFrame, output: Path) -> dict[str, object]:
    included = frame.loc[frame["status"] == "included"].copy()
    errors: list[str] = []
    seen_paths: set[str] = set()
    for row in included.itertuples():
        path = output / str(row.image_path)
        if str(path) in seen_paths:
            errors.append(f"Duplicate image path: {path}")
        seen_paths.add(str(path))
        if not path.is_file():
            errors.append(f"Missing image: {path}")
            continue
        try:
            with Image.open(path) as image:
                if image.size != (IMAGE_SIDE, IMAGE_SIDE) or image.mode != "L":
                    errors.append(f"Invalid image format: {path} ({image.size}, {image.mode})")
                image.verify()
            if file_sha256(path) != row.image_file_sha256:
                errors.append(f"Image SHA-256 mismatch: {path}")
        except Exception as exc:
            errors.append(f"Unreadable image: {path}: {exc}")

    image_root = output / "images_head"
    actual_paths = {
        path.relative_to(output).as_posix() for path in image_root.rglob("*.png")
    } if image_root.is_dir() else set()
    expected_paths = set(included["image_path"].astype(str))
    if actual_paths != expected_paths:
        errors.append(
            f"Image file set mismatch: missing={len(expected_paths - actual_paths)}, "
            f"extra={len(actual_paths - expected_paths)}"
        )

    split_hashes = {
        split: set(included.loc[included["split"] == split, "raw_bytes_sha256"])
        for split in ("train", "val", "test")
    }
    overlap = (
        (split_hashes["train"] & split_hashes["val"])
        | (split_hashes["train"] & split_hashes["test"])
        | (split_hashes["val"] & split_hashes["test"])
    )
    if overlap:
        errors.append(f"Raw byte hashes overlap across splits: {len(overlap)}")

    return {
        "passed": not errors,
        "image_count": int(len(included)),
        "expected_image_count": int(len(included)),
        "cross_split_raw_hash_overlap": int(len(overlap)),
        "errors": errors,
    }


def package_versions() -> dict[str, str]:
    import PIL
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    labels_path = args.labels.resolve()
    output = args.output.resolve()
    check_inputs(input_zip, labels_path, output)
    official_output = (Path(__file__).resolve().parents[1] / "data").resolve()
    if args.skip_asm and output == official_output:
        raise ValueError("--skip-asm is only allowed with a separate --output directory")
    start = time.perf_counter()

    manifests_dir = output / "manifests"
    features_dir = output / "features"
    reports_dir = output / "reports"
    for directory in (manifests_dir, features_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"Class": class_id, "Family": family} for class_id, family in CLASS_NAMES.items()]
    ).to_csv(manifests_dir / "classes.csv", index=False, encoding="utf-8-sig")

    labels_clean, label_duplicates, label_stats = clean_labels(labels_path)
    label_duplicates.to_csv(manifests_dir / "duplicate_labels.csv", index=False, encoding="utf-8-sig")

    images: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(input_zip, "r") as archive:
        bytes_entries, asm_entries, duplicate_entry_names = collect_zip_entries(archive)

        label_ids = set(labels_clean["Id"].astype(str))
        paired_ids = sorted(label_ids & set(bytes_entries) & set(asm_entries))
        label_lookup = labels_clean.set_index("Id")["Class"].astype(int).to_dict()
        print(f"Processing {len(paired_ids)} paired samples from ZIP...", flush=True)
        for position, sample_id in enumerate(paired_ids, start=1):
            with archive.open(bytes_entries[sample_id], "r") as stream:
                byte_features, image_array = analyze_bytes(stream, bytes_entries[sample_id])
            if args.skip_asm:
                asm_features = asm_metadata_only(asm_entries[sample_id])
            else:
                with archive.open(asm_entries[sample_id], "r") as stream:
                    asm_features = analyze_asm(stream, asm_entries[sample_id])
            rows.append({"Id": sample_id, "Class": int(label_lookup[sample_id]), **byte_features, **asm_features})
            images[sample_id] = image_array
            if position % 25 == 0 or position == len(paired_ids):
                print(f"  {position}/{len(paired_ids)}", flush=True)

    frame = pd.DataFrame(rows).sort_values("Id").reset_index(drop=True)
    frame = assign_content_duplicates(frame)
    frame["split"] = stratified_split(frame, args.seed)
    frame["image_path"] = ""
    frame["image_file_sha256"] = ""
    save_images(frame, images, output)

    included = frame.loc[frame["status"] == "included"].copy()
    class_counts = included.groupby("Class").size().rename("total")
    split_counts = pd.crosstab(included["Class"], included["split"])
    for split in ("train", "val", "test"):
        if split not in split_counts:
            split_counts[split] = 0
    pd.concat(
        [class_counts, split_counts[["train", "val", "test"]]],
        axis=1,
    ).fillna(0).astype(int).reset_index().to_csv(
        reports_dir / "class_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_columns = [
        "Id", "Class", "split", "status", "image_path", "image_file_sha256", "bytes_zip_path", "asm_zip_path",
        "byte_token_count", "known_byte_count", "unknown_byte_count", "unknown_byte_ratio",
        "invalid_byte_token_count", "image_padding_tokens", "image_truncated_tokens",
        "raw_bytes_sha256", "normalized_bytes_sha256", "token_stream_sha256", "raw_asm_sha256",
        "content_duplicate_group_size", "duplicate_of", "quality_note",
    ]
    included[manifest_columns].to_csv(manifests_dir / "samples.csv", index=False, encoding="utf-8-sig")

    non_feature_columns = {
        "status", "split", "image_path", "image_file_sha256", "bytes_zip_path", "asm_zip_path",
        "raw_bytes_sha256", "normalized_bytes_sha256", "token_stream_sha256", "raw_asm_sha256",
        "content_duplicate_group_size", "duplicate_of", "quality_note",
    }
    feature_columns = [column for column in included.columns if column not in non_feature_columns]
    included[feature_columns].to_csv(features_dir / "features.csv", index=False, encoding="utf-8-sig")

    verification = check_head_outputs(frame, output)
    zip_stats = {
        "zip_path": input_zip.name,
        "zip_size_bytes": int(input_zip.stat().st_size),
        "bytes_entries": int(len(bytes_entries)),
        "asm_entries": int(len(asm_entries)),
        "paired_samples": int(len(paired_ids)),
        "labels_missing_bytes": int(len(set(labels_clean["Id"].astype(str)) - set(bytes_entries))),
        "labels_missing_asm": int(len(set(labels_clean["Id"].astype(str)) - set(asm_entries))),
        "bytes_without_labels": int(len(set(bytes_entries) - set(labels_clean["Id"].astype(str)))),
        "asm_without_labels": int(len(set(asm_entries) - set(labels_clean["Id"].astype(str)))),
        "duplicate_zip_entry_names": duplicate_entry_names,
    }
    elapsed = time.perf_counter() - start
    summary = {
        "dataset": "Microsoft BIG 2015 small sample (local derivative)",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": zip_stats,
        "labels": label_stats,
        "output": {
            "included_samples": int(len(included)),
            "excluded_samples": int((frame["status"] != "included").sum()),
            "split_counts": {key: int(value) for key, value in included["split"].value_counts().to_dict().items()},
            "feature_columns": int(len(feature_columns) - 2),
            "image_size": [IMAGE_SIDE, IMAGE_SIDE],
            "image_mode": "L",
        },
        "verification": verification,
        "environment": package_versions(),
        "elapsed_seconds": elapsed,
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not verification["passed"]:
        raise RuntimeError("Output verification failed; see reports/summary.json")
    print(f"Completed in {elapsed / 60:.1f} minutes. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
